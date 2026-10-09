"""
Curve comparison utilities.

This module provides the same behavior as the original script:
- Crop curves to a (possibly user-specified) overlapping q-range.
- Resample the denser curve onto the sparser one using scipy.interpolate.interp1d.
- Scale the simulated curve by the ratio of the last ~5 points (tail mean).
- Compare in log10-intensity space using Amplitude–Phase Distance (AP).
- Optionally save diagnostic figures including APDist phase-warp plots.
"""

import json
import os
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from apdist.distances import AmplitudePhaseDistance

DEFAULT_DP_COEFF = 0.5


class MetricFailed(ValueError):
    """The loss cannot be computed from a valid simulated S(q).

    Deterministic given the two curves, so rerunning the simulation does not
    help: the BO evaluation is failed and not given to the GP.
    """


def _sanitize_curve(arr: np.ndarray) -> np.ndarray:
    """Make a (N, 2) curve [q, y] safe for interp1d.

    Drops non-finite rows, sorts by q ascending, collapses duplicate q
    (keeps the first occurrence), and clips the second column to a tiny
    positive floor so log10 is safe downstream. Used by both the loss
    code (compare_saxs_curves) and the I(q) -> S(q) extractor in
    scattering.extract_exp_sq, where nearest-neighbor q snapping can
    introduce duplicate q values.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) array, got shape {arr.shape}")
    arr = arr[:, :2]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        raise ValueError("Curve is empty after dropping non-finite rows.")
    q, y = arr[:, 0], arr[:, 1]
    order = np.argsort(q, kind="stable")
    q, y = q[order], y[order]
    uq, idx = np.unique(q, return_index=True)
    q, y = uq, y[idx]
    y = np.clip(y, 1e-12, None)
    return np.column_stack([q, y])


def _warn_apdist_kwargs_ignored(metric, dp_coeff, plot_apdist):
    """Warn when apdist-only kwargs are passed but metric != apdist."""
    if metric == "apdist":
        return
    ignored = []
    if dp_coeff != DEFAULT_DP_COEFF:
        ignored.append(f"dp_coeff={dp_coeff}")
    if plot_apdist is not True:
        ignored.append(f"plot_apdist={plot_apdist}")
    if ignored:
        warnings.warn(
            f"metric={metric!r}: {', '.join(ignored)} have no effect "
            "(only used when metric='apdist').",
            stacklevel=3,
        )


def _save_apdist_plots(
    q_ref,
    I_exp,
    I_sim,
    save_dir,
    da,
    dp,
    dp_coeff=DEFAULT_DP_COEFF,
):
    """Save log-space curves after APDist phase warp (da/dp annotated)."""
    from apdist.geometry import SquareRootSlopeFramework
    from apdist.utils import plot_warping

    t_ap = np.linspace(0.0, 1.0, len(q_ref))
    eps = 1e-10
    log_I_exp = np.log10(np.clip(I_exp, eps, None))
    log_I_sim = np.log10(np.clip(I_sim, eps, None))

    srsf = SquareRootSlopeFramework(t_ap)
    gam = srsf.get_gamma(srsf.to_srsf(log_I_exp), srsf.to_srsf(log_I_sim))
    log_I_sim_warped = srsf.warp_f_gamma(log_I_sim, gam)
    weighted = dp_coeff * dp + (1.0 - dp_coeff) * da

    os.makedirs(save_dir, exist_ok=True)
    _rc = plt.rcParams.copy()
    try:
        plt.rcParams.update({"font.size": 14})

        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.scatter(q_ref, log_I_exp, linewidth=0.5, label="Exp (log10 I)", color="k")
        ax.plot(
            q_ref,
            log_I_sim_warped,
            linewidth=3,
            label="Sim warped (log10 I)",
            color="red",
        )
        ax.set_xscale("log")
        ax.set_ylabel("log10 Intensity", labelpad=10)
        ax.set_xlabel("q ($\\AA^{-1}$)", labelpad=10)
        ax.set_title(
            f"APDist after phase warp: da={da:.4f}, dp={dp:.4f}, "
            f"loss={weighted:.4f} (dp_coeff={dp_coeff})",
            pad=14,
        )
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout(pad=1.8)
        fig.savefig(
            os.path.join(save_dir, "compare_apdist_warped.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.35,
        )
        plt.close(fig)

        plot_warping(q_ref, log_I_exp, log_I_sim, log_I_sim_warped, gam)
        fig = plt.gcf()
        fig.set_size_inches(24, 6.5)
        fig.tight_layout(pad=2.0, h_pad=2.5, w_pad=2.5)
        fig.savefig(
            os.path.join(save_dir, "compare_apdist_warp_detail.png"),
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.4,
        )
        plt.close(fig)
    finally:
        plt.rcParams.update(_rc)


def compare_saxs_curves(
    exp_data,
    sim_data,
    q_range=None,
    scale_intensity=True,
    metric="mse",
    dp_coeff=DEFAULT_DP_COEFF,
):
    """
    Compare two SAXS curves in log space.

    Steps
    -----
    1) Determine the common overlap in q (optionally further restricted by q_range).
    2) Resample both curves onto a shared physical q-grid via linear interp1d.
    3) Scale the simulated intensity by the ratio of tail means ([-6:-1]).
    4) Compute Amplitude–Phase Distance (AP) on log10 intensities using a
       normalized parameter domain [0, 1], or MSE on log10 intensities.

    Parameters
    ----------
    exp_data : (N1, 2) ndarray
        Experimental data [q, I(q)].
    sim_data : (N2, 2) ndarray
        Simulated/model data [q, I(q)].
    q_range : tuple(float, float) or None
        Optional (q_min, q_max) window to restrict the comparison.
    scale_intensity : bool
        if True, scales simulated intensity to best match experimental.
        Kept for compatibility with the original signature; the current logic
        always performs tail-mean scaling as implemented originally.
    metric : str
        ``'mse'`` or ``'apdist'``.
    dp_coeff : float
        Weight on phase distance when ``metric='apdist'``:
        ``dist = dp_coeff*dp + (1-dp_coeff)*da``. Ignored for ``metric='mse'``.

    Returns
    -------
    distance : float
        Loss value (MSE or weighted APDist).
    q_ref : (K,) ndarray
        Physical q-grid used for interpolation, comparison output, and plotting.
    I_exp_resampled : (K,) ndarray
        Experimental intensity on q_ref (resampled if needed).
    I_sim_scaled : (K,) ndarray
        Simulated intensity on q_ref after tail-mean scaling.
    da : float or None
        Amplitude distance (``metric='apdist'`` only, else None).
    dp : float or None
        Phase distance (``metric='apdist'`` only, else None).
    """
    del scale_intensity  # kept for API compatibility; tail scaling always applied

    if not 0.0 <= dp_coeff <= 1.0:
        raise ValueError(f"dp_coeff must be in [0, 1]; got {dp_coeff}")

    exp_data = _sanitize_curve(exp_data)
    sim_data = _sanitize_curve(sim_data)

    q_exp, I_exp = exp_data[:, 0], exp_data[:, 1]
    q_sim, I_sim = sim_data[:, 0], sim_data[:, 1]

    q_min_common = max(q_exp.min(), q_sim.min())
    q_max_common = min(q_exp.max(), q_sim.max())

    if q_range is not None:
        q_min_user, q_max_user = q_range
        q_min_common = max(q_min_common, q_min_user)
        q_max_common = min(q_max_common, q_max_user)

    mask_exp = (q_exp >= q_min_common) & (q_exp <= q_max_common)
    mask_sim = (q_sim >= q_min_common) & (q_sim <= q_max_common)
    q_exp_crop, I_exp_crop = q_exp[mask_exp], I_exp[mask_exp]
    q_sim_crop, I_sim_crop = q_sim[mask_sim], I_sim[mask_sim]

    n_points = min(len(q_exp_crop), len(q_sim_crop))
    q_ref = np.logspace(np.log10(q_min_common), np.log10(q_max_common), n_points)

    I_exp_resampled = interp1d(
        q_exp_crop, I_exp_crop, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )(q_ref)

    I_sim_resampled = interp1d(
        q_sim_crop, I_sim_crop, kind="linear",
        bounds_error=False, fill_value="extrapolate",
    )(q_ref)

    eps = 1e-10
    I_exp_resampled = np.clip(I_exp_resampled, eps, None)
    I_sim_resampled = np.clip(I_sim_resampled, eps, None)

    scale_factor = np.mean(I_exp_resampled[-6:-1]) / np.mean(I_sim_resampled[-6:-1])
    I_sim_scaled = I_sim_resampled * scale_factor

    log_I_exp = np.log10(I_exp_resampled)
    log_I_sim = np.log10(I_sim_scaled)

    da = dp = None
    if metric == "apdist":
        t_ap = np.linspace(0.0, 1.0, len(q_ref))
        da, dp = AmplitudePhaseDistance(t_ap, log_I_exp, log_I_sim)
        distance = dp_coeff * dp + (1.0 - dp_coeff) * da
    elif metric == "mse":
        distance = np.mean((log_I_exp - log_I_sim) ** 2)
    else:
        raise ValueError(f"Unknown metric chosen: {metric}")

    return distance, q_ref, I_exp_resampled, I_sim_scaled, da, dp


# ------------------------- shift_rmse ------------------------- #
#
# L = M4 + lambda * |ln(q1_sim / q1_tgt)|, where M4 is the log10 RMSE on the
# reduced axis x = q / q1 (each curve divided by its own first peak) after
# scaling the simulated intensity by the high-q S(q) -> 1 plateau. The
# comparison window is the overlap of the two curves on x, minus
# ``overlap_trim`` points at each end (q_range=None).
#
# A curve whose smoothed S stays within ``dispersed_delta`` of 1 in the
# search window is dispersed: it has no first peak, so neither curve is
# aligned (both use x = q / q1_tgt) and the spacing term is replaced by
# ``dispersed_shift`` (0 when both curves are dispersed).
#
# Defaults are the task-0b choice (2026-10-08). On the 407 saved curves of
# the FCC, BCC and two HS-fluid inverse runs, s_max - 1 is <= 0.089 for the
# 17 flat curves and >= 0.746 for every curve with a peak, so
# ``dispersed_delta = 0.5`` sits in that gap. ``dispersed_shift = 0`` keeps
# the loss free of a step at the dispersed boundary, which matters for
# mapping ground truth that contains dispersed conditions; skipping the
# alignment alone already ranks flat curves outside the top 20 of both
# fluid runs.

SHIFT_RMSE_DEFAULTS = {
    "peak_search_range": (0.006, 0.12),
    "prominence_frac": 0.3,
    "dispersed_delta": 0.5,
    "dispersed_shift": 0.0,
    "asymptote_band": (0.085, 0.100),
    "n_points": 512,
    "lambda_shift": 1.0,
    "overlap_trim": 0,
    "no_peak": "fail",
    "asymptote_min_points": 3,
    "asymptote_fallback": "fail",
    "asymptote_tail_frac": 0.75,
    "asymptote_zero_den": "fail",
}

_SHIFT_RMSE_RANGES = ("peak_search_range", "asymptote_band")
_SHIFT_RMSE_CHOICES = {
    "no_peak": ("fallback", "fail"),
    "asymptote_fallback": ("tail", "fail"),
    "asymptote_zero_den": ("unity", "fail"),
}


def shift_rmse_params(metric_kwargs=None):
    """
    Return the full ``shift_rmse`` parameter dict: defaults overridden by
    ``metric_kwargs``. Lists (from JSON) become tuples. Unknown keys or
    invalid choices raise ``ValueError``.

    Parameters (keys of ``metric_kwargs``)
    --------------------------------------
    peak_search_range : (q_lo, q_hi)
        First-peak search window (A^-1). Default (0.006, 0.12).
    prominence_frac : float
        ``find_peaks`` prominence = prominence_frac * (max - min) of the
        smoothed curve in the search window. 0.3.
    dispersed_delta : float
        A curve whose smoothed maximum in the search window is below
        ``1 + dispersed_delta`` is dispersed (no first peak). 0.5.
    dispersed_shift : float
        Spacing term used when exactly one curve is dispersed. 0.
    asymptote_band : (q_lo, q_hi)
        High-q S(q) -> 1 band; the simulated curve is scaled by
        mean_band(S_tgt) / mean_band(S_sim). Default (0.085, 0.100).
    n_points : int
        Log-spaced grid points on the reduced axis. 512.
    lambda_shift : float
        Weight of ``|ln(q1_sim/q1_tgt)|``. 1.
    overlap_trim : int or None
        Used only when ``q_range is None``: compare over the overlap of the
        two curves on the reduced axis after dropping ``overlap_trim`` points
        at each end of each curve. 0. None makes ``q_range=None`` raise.
    no_peak : {"fallback", "fail"}
        A curve that is not dispersed but has no ``find_peaks`` peak in the
        search window: take the maximum of the smoothed curve, or fail.
        "fail".
    asymptote_min_points : int
        Fewer band points than this triggers ``asymptote_fallback``. 3.
    asymptote_fallback : {"tail", "fail"}
        "tail" uses ``q >= asymptote_tail_frac * q_max`` of that curve;
        "fail" fails. "fail".
    asymptote_tail_frac : float
        0.75.
    asymptote_zero_den : {"unity", "fail"}
        Simulated band mean non-finite or below 1e-12: scale 1.0 or fail.
        "fail".
    """
    unknown = set(metric_kwargs or {}) - set(SHIFT_RMSE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown shift_rmse metric_kwargs: {sorted(unknown)}")
    p = dict(SHIFT_RMSE_DEFAULTS)
    p.update(metric_kwargs or {})
    for key in _SHIFT_RMSE_RANGES:
        lo, hi = p[key]
        p[key] = (float(lo), float(hi))
    for key, choices in _SHIFT_RMSE_CHOICES.items():
        if p[key] not in choices:
            raise ValueError(f"shift_rmse {key} must be one of {choices}; got {p[key]!r}")
    p["n_points"] = int(p["n_points"])
    p["asymptote_min_points"] = int(p["asymptote_min_points"])
    if p["overlap_trim"] is not None:
        p["overlap_trim"] = int(p["overlap_trim"])
        if p["overlap_trim"] < 0:
            raise ValueError("shift_rmse overlap_trim must be >= 0")
    for key in ("prominence_frac", "dispersed_delta", "dispersed_shift",
                "lambda_shift", "asymptote_tail_frac"):
        p[key] = float(p[key])
    for key in ("dispersed_delta", "dispersed_shift"):
        if p[key] < 0.0:
            raise ValueError(f"shift_rmse {key} must be >= 0")
    return p


@dataclass
class PeakInfo:
    q1: float
    s_peak: float
    s_max: float
    baseline: float
    fwhm_points: float
    n_peaks_found: int
    dispersed: bool


def detect_first_peak(
    curve,
    search_range=SHIFT_RMSE_DEFAULTS["peak_search_range"],
    prominence_frac=SHIFT_RMSE_DEFAULTS["prominence_frac"],
    dispersed_delta=SHIFT_RMSE_DEFAULTS["dispersed_delta"],
):
    """
    Lowest-q peak of a sanitized (N, 2) curve with sub-grid refinement.

    Savitzky-Golay smoothing (window 5, order 2) inside ``search_range``.
    The smoothed maximum ``s_max`` and minimum (the baseline) are taken in
    that window. ``s_max - 1 < dispersed_delta`` marks the curve dispersed
    and no peak is searched (``q1 = nan``). Otherwise ``find_peaks`` runs with
    prominence ``prominence_frac * (s_max - baseline)``, the first peak is
    taken and refined by a parabola through the raw points. With no peak
    found, the smoothed maximum is used and ``n_peaks_found = 0``. Fewer than
    5 points in the window gives ``q1 = nan``.
    """
    q, s = curve[:, 0], curve[:, 1]
    mask = (q >= search_range[0]) & (q <= search_range[1])
    qw, sw = q[mask], s[mask]
    if qw.size < 5:
        return PeakInfo(np.nan, np.nan, np.nan, np.nan, np.nan, 0, False)

    sw_smooth = savgol_filter(sw, window_length=5, polyorder=2)
    s_max = float(sw_smooth.max())
    baseline = float(sw_smooth.min())
    if s_max - 1.0 < dispersed_delta:
        return PeakInfo(np.nan, np.nan, s_max, baseline, np.nan, 0, True)

    prominence = prominence_frac * max(s_max - baseline, 1e-12)
    idx_peaks, _ = find_peaks(sw_smooth, prominence=prominence)

    if idx_peaks.size == 0:
        idx = int(np.argmax(sw_smooth))
        n_found = 0
    else:
        idx = int(idx_peaks[0])
        n_found = int(idx_peaks.size)

    if 0 < idx < qw.size - 1:
        y0, y1, y2 = sw[idx - 1], sw[idx], sw[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-15 else 0.0
        delta = float(np.clip(delta, -1.0, 1.0))
        dq = qw[idx + 1] - qw[idx]
        q1 = float(qw[idx] + delta * dq)
    else:
        q1 = float(qw[idx])

    s_peak = float(sw[idx])
    half = 0.5 * (s_peak + baseline)
    fwhm_points = float(np.sum(sw >= half))
    return PeakInfo(q1, s_peak, s_max, baseline, fwhm_points, n_found, False)


def asymptote_scale(
    exp_curve,
    sim_curve,
    band=SHIFT_RMSE_DEFAULTS["asymptote_band"],
    min_points=SHIFT_RMSE_DEFAULTS["asymptote_min_points"],
    fallback=SHIFT_RMSE_DEFAULTS["asymptote_fallback"],
    tail_frac=SHIFT_RMSE_DEFAULTS["asymptote_tail_frac"],
    zero_den=SHIFT_RMSE_DEFAULTS["asymptote_zero_den"],
):
    """
    Intensity scale ``mean_band(S_exp) / mean_band(S_sim)`` in absolute q.

    Returns ``(scale, info)``; ``info`` records the band point counts and
    whether the tail fallback or the zero-denominator rule fired.
    """
    info = {}

    def band_mean(curve, tag):
        q, s = curve[:, 0], curve[:, 1]
        mask = (q >= band[0]) & (q <= band[1])
        info[f"{tag}_band_points"] = int(mask.sum())
        info[f"{tag}_tail_fallback"] = bool(mask.sum() < min_points)
        if mask.sum() < min_points:
            if fallback == "fail":
                raise MetricFailed(
                    f"shift_rmse: {tag} curve has {int(mask.sum())} points in "
                    f"asymptote band {band} (< {min_points})"
                )
            mask = q >= tail_frac * float(q.max())
        return float(np.mean(s[mask]))

    num = band_mean(exp_curve, "exp")
    den = band_mean(sim_curve, "sim")
    info["zero_den"] = bool(not np.isfinite(den) or abs(den) < 1e-12)
    if info["zero_den"]:
        if zero_den == "fail":
            raise MetricFailed(f"shift_rmse: simulated asymptote band mean is {den}")
        return 1.0, info
    return num / den, info


def _resample_reduced(exp_curve, sim_curve, window, exp_scale, sim_scale, n_points):
    """Resample both curves onto a log grid in reduced x = q / scale.

    Out-of-range grid points hold the first/last in-window intensity.
    """
    qe = exp_curve[:, 0] / exp_scale
    ie = exp_curve[:, 1]
    qs = sim_curve[:, 0] / sim_scale
    isim = sim_curve[:, 1]

    lo = max(qe.min(), qs.min(), window[0])
    hi = min(qe.max(), qs.max(), window[1])
    if not (hi > lo):
        raise MetricFailed(f"shift_rmse: empty comparison window [{lo}, {hi}]")

    me = (qe >= lo) & (qe <= hi)
    ms = (qs >= lo) & (qs <= hi)
    if int(me.sum()) < 4 or int(ms.sum()) < 4:
        raise MetricFailed(
            f"shift_rmse: too few points in window: exp={int(me.sum())}, sim={int(ms.sum())}"
        )

    grid = np.logspace(np.log10(lo), np.log10(hi), int(n_points))
    i_exp = interp1d(
        qe[me], ie[me], kind="linear", bounds_error=False,
        fill_value=(float(ie[me][0]), float(ie[me][-1])),
    )(grid)
    i_sim = interp1d(
        qs[ms], isim[ms], kind="linear", bounds_error=False,
        fill_value=(float(isim[ms][0]), float(isim[ms][-1])),
    )(grid)
    eps = 1e-10
    i_exp = np.clip(i_exp, eps, None)
    i_sim = np.clip(i_sim, eps, None)
    return np.log10(i_exp), np.log10(i_sim), grid


_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _peak_json(peak):
    return {k: (_finite_or_none(v) if isinstance(v, float) else v)
            for k, v in asdict(peak).items()}


def shift_rmse_loss(exp_data, sim_data, q_range, metric_kwargs=None):
    """
    ``L = M4 + lambda_shift * |ln(q1_sim / q1_tgt)|``.

    When either curve is dispersed (see :func:`detect_first_peak`), both
    curves use x = q / q1_tgt (x = q when the target is dispersed) and the
    spacing term is ``dispersed_shift``, or 0 when both are dispersed.

    Parameters
    ----------
    exp_data, sim_data : (N, 2) arrays [q, S(q)]
    q_range : (q_lo, q_hi) or None
        Absolute comparison window, converted to x with the target scale.
        None compares over the trimmed overlap (``overlap_trim``).
    metric_kwargs : dict, optional
        Overrides of :data:`SHIFT_RMSE_DEFAULTS`.

    Returns
    -------
    loss, diag, grid, log_exp, log_sim
        ``diag`` holds every field written to ``shift_rmse_diagnostics.json``.

    Raises
    ------
    MetricFailed
        No usable first peak on a curve that is not dispersed (per
        ``no_peak``), asymptote rules set to "fail" and triggered, or an
        empty window.
    ValueError
        ``q_range is None`` with ``overlap_trim=None`` (configuration error).
    """
    p = shift_rmse_params(metric_kwargs)
    if q_range is None and p["overlap_trim"] is None:
        raise ValueError(
            "shift_rmse requires an absolute q_range window, or "
            "metric_kwargs['overlap_trim'] for the curve overlap"
        )
    exp_curve = _sanitize_curve(exp_data)
    sim_curve = _sanitize_curve(sim_data)

    peak_kw = dict(
        search_range=p["peak_search_range"],
        prominence_frac=p["prominence_frac"],
        dispersed_delta=p["dispersed_delta"],
    )
    peaks = {"tgt": detect_first_peak(exp_curve, **peak_kw),
             "sim": detect_first_peak(sim_curve, **peak_kw)}
    for tag, label in (("tgt", "target"), ("sim", "simulated")):
        pk = peaks[tag]
        if pk.dispersed:
            continue
        if not np.isfinite(pk.q1):
            raise MetricFailed(f"shift_rmse: failed to detect {label} first Bragg peak")
        if p["no_peak"] == "fail" and pk.n_peaks_found == 0:
            raise MetricFailed(
                f"shift_rmse: no {label} peak found in {p['peak_search_range']} "
                "(no_peak='fail')"
            )

    disp_tgt = peaks["tgt"].dispersed
    disp_sim = peaks["sim"].dispersed
    aligned = not (disp_tgt or disp_sim)
    tgt_q1 = float(peaks["tgt"].q1)
    sim_q1 = float(peaks["sim"].q1)
    if aligned:
        tgt_scale, sim_scale = tgt_q1, sim_q1
    else:
        tgt_scale = sim_scale = 1.0 if disp_tgt else tgt_q1

    if q_range is not None:
        window_abs = (float(q_range[0]), float(q_range[1]))
        x_lo, x_hi = window_abs[0] / tgt_scale, window_abs[1] / tgt_scale
        window_source = "q_range"
    else:
        k = p["overlap_trim"]
        qe, qs = exp_curve[:, 0], sim_curve[:, 0]
        if qe.size <= 2 * k or qs.size <= 2 * k:
            raise MetricFailed(f"shift_rmse: overlap_trim={k} leaves no points")
        x_lo = max(qe[k] / tgt_scale, qs[k] / sim_scale)
        x_hi = min(qe[-1 - k] / tgt_scale, qs[-1 - k] / sim_scale)
        window_abs = (x_lo * tgt_scale, x_hi * tgt_scale)
        window_source = "overlap"

    anchor, asym_info = asymptote_scale(
        exp_curve, sim_curve,
        band=p["asymptote_band"],
        min_points=p["asymptote_min_points"],
        fallback=p["asymptote_fallback"],
        tail_frac=p["asymptote_tail_frac"],
        zero_den=p["asymptote_zero_den"],
    )
    sim_scaled = sim_curve.copy()
    sim_scaled[:, 1] = sim_scaled[:, 1] * anchor
    log_exp, log_sim, grid = _resample_reduced(
        exp_curve, sim_scaled, (x_lo, x_hi),
        exp_scale=tgt_scale, sim_scale=sim_scale, n_points=p["n_points"],
    )
    n = int(p["n_points"])
    t = np.linspace(0.0, 1.0, n)
    m4 = float(np.sqrt(_trapezoid((log_exp - log_sim) ** 2, t)))
    if aligned:
        q1_ratio = float(sim_q1 / tgt_q1)
        abs_log = float(np.abs(np.log(q1_ratio)))
        shift_term = float(p["lambda_shift"] * abs_log)
    else:
        q1_ratio = abs_log = None
        shift_term = 0.0 if (disp_tgt and disp_sim) else float(p["dispersed_shift"])
    loss = m4 + shift_term

    peak_fallback = {tag: (not peaks[tag].dispersed and peaks[tag].n_peaks_found == 0)
                     for tag in peaks}
    diag = {
        "loss": loss,
        "m4": m4,
        "shift_term": shift_term,
        "abs_log_q1_ratio": abs_log,
        "lambda_shift": p["lambda_shift"],
        "q1_ratio": q1_ratio,
        "q1_tgt": _finite_or_none(tgt_q1),
        "q1_sim": _finite_or_none(sim_q1),
        "aligned": aligned,
        "dispersed_tgt": bool(disp_tgt),
        "dispersed_sim": bool(disp_sim),
        "x_scale_tgt": float(tgt_scale),
        "x_scale_sim": float(sim_scale),
        "anchor_scale": float(anchor),
        "window_abs": [float(window_abs[0]), float(window_abs[1])],
        "window_source": window_source,
        "q_range": None if q_range is None else [float(q_range[0]), float(q_range[1])],
        "x_lo": float(x_lo),
        "x_hi": float(x_hi),
        "n_points": n,
        "grid_lo": float(grid[0]),
        "grid_hi": float(grid[-1]),
        "peak_tgt": _peak_json(peaks["tgt"]),
        "peak_sim": _peak_json(peaks["sim"]),
        "asymptote": asym_info,
        "fallbacks": {
            "peak_tgt_max_fallback": bool(peak_fallback["tgt"]),
            "peak_sim_max_fallback": bool(peak_fallback["sim"]),
            "asymptote_exp_tail_fallback": asym_info["exp_tail_fallback"],
            "asymptote_sim_tail_fallback": asym_info["sim_tail_fallback"],
            "asymptote_zero_den_unity": asym_info["zero_den"],
        },
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
    }
    return loss, diag, grid, log_exp, log_sim


def _compare_shift_rmse(experimental_data, simulated_data, save_dir, q_range, metric_kwargs):
    """Compute shift_rmse, write ``shift_rmse_diagnostics.json`` and the overlay plot."""
    os.makedirs(save_dir, exist_ok=True)
    try:
        loss, diag, grid, log_exp, log_sim = shift_rmse_loss(
            experimental_data, simulated_data, q_range, metric_kwargs=metric_kwargs,
        )
    except MetricFailed as exc:
        with open(os.path.join(save_dir, "shift_rmse_diagnostics.json"), "w") as fh:
            json.dump({"metric": "shift_rmse", "failed": True, "reason": str(exc),
                       "params": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in shift_rmse_params(metric_kwargs).items()}},
                      fh, indent=2)
        raise
    diag["metric"] = "shift_rmse"
    diag["failed"] = False
    with open(os.path.join(save_dir, "shift_rmse_diagnostics.json"), "w") as fh:
        json.dump(diag, fh, indent=2)

    if diag["aligned"]:
        peak_line = (f"q1_ratio={diag['q1_ratio']:.5g}  "
                     f"q1_tgt={diag['q1_tgt']:.5g}  q1_sim={diag['q1_sim']:.5g}")
        x_label = r"$x = q/q_1$ (aligned)"
    else:
        peak_line = (f"not aligned (dispersed: tgt={diag['dispersed_tgt']}, "
                     f"sim={diag['dispersed_sim']})")
        x_label = r"$x = q/q_{1,\mathrm{tgt}}$" if not diag["dispersed_tgt"] else r"$q$ ($\AA^{-1}$)"
    with plt.rc_context({"font.size": 18}):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(grid, 10.0 ** log_exp, linewidth=0.5, label="target", color="k")
        ax.plot(grid, 10.0 ** log_sim, linewidth=3, label="sim", color="red")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylabel("S(q) (arb. unit)")
        ax.set_xlabel(x_label)
        ax.set_title(
            f"L={loss:.6g}  M4={diag['m4']:.6g}  shift={diag['shift_term']:.6g}\n"
            + peak_line
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "compare_to_exp_saxsfft.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)
    return loss


def load_shift_rmse_components(save_dir):
    """Return ``{"rmse": M4, "shift": shift_term}`` from a diagnostics JSON, or {}."""
    path = os.path.join(save_dir, "shift_rmse_diagnostics.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if data.get("failed") or data.get("m4") is None:
        return {}
    return {"rmse": float(data["m4"]), "shift": float(data["shift_term"])}


def compare_to_exp(
    experimental_data,
    simulated_data,
    save_dir,
    metric="mse",
    dp_coeff=DEFAULT_DP_COEFF,
    plot_apdist=True,
):
    """
    Generate diagnostic plots and return the short-window score.

    Behavior
    --------
    - First compare in q in [0.003, 0.03]; save 'compare_to_exp.png'.
    - Then compare in q in [0.003, 0.07]; save 'compare_to_exp_full_curve.png'.
    - When ``metric='apdist'`` and ``plot_apdist=True``, also save phase-warp
      plots under ``save_dir/apdist_plots/`` for the primary window.
    - Return the first window's loss.
    """
    _warn_apdist_kwargs_ignored(metric, dp_coeff, plot_apdist)

    q = [0.003, 0.03]
    loss, q_ref, I_exp_resampled, I_sim_resampled, da, dp = compare_saxs_curves(
        experimental_data, simulated_data, q, metric=metric, dp_coeff=dp_coeff,
    )
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(q_ref, I_exp_resampled, linewidth=0.5, label="Exp_data", color="k")
    ax.plot(q_ref, I_sim_resampled, linewidth=3, label="Sim_data", color="red")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Intensity (arb. unit)")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    plt.title(str(loss))
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "compare_to_exp.png"), dpi=600, bbox_inches="tight")
    plt.close()

    if metric == "apdist" and plot_apdist:
        _save_apdist_plots(
            q_ref,
            I_exp_resampled,
            I_sim_resampled,
            os.path.join(save_dir, "apdist_plots"),
            da,
            dp,
            dp_coeff=dp_coeff,
        )

    q = [0.003, 0.07]
    loss2, q_ref2, I_exp2, I_sim2, _, _ = compare_saxs_curves(
        experimental_data, simulated_data, q, metric=metric, dp_coeff=dp_coeff,
    )
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(q_ref2, I_exp2, linewidth=0.5, label="Exp_data", color="k")
    ax.plot(q_ref2, I_sim2, linewidth=3, label="Sim_data", color="red")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Intensity (arb. unit)")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    plt.title(str(loss2))
    plt.legend()
    plt.savefig(
        os.path.join(save_dir, "compare_to_exp_full_curve.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()
    return loss


def compare_to_exp_saxsfft(
    experimental_data,
    simulated_data,
    save_dir,
    metric="mse",
    q_range=(0.003, 0.06),
    dp_coeff=DEFAULT_DP_COEFF,
    plot_apdist=True,
    metric_kwargs=None,
):
    """
    Compare experimental and simulated S(q) using the wider q-range available
    from saxs-fft.

    ``mse`` / ``apdist`` use the same :func:`compare_saxs_curves` engine as
    :func:`compare_to_exp`, with a wider default window ``[0.003, 0.06]``
    A^-1. ``shift_rmse`` uses :func:`shift_rmse_loss` and writes
    ``shift_rmse_diagnostics.json`` to ``save_dir``.

    Parameters
    ----------
    experimental_data : (N, 2) ndarray
        Experimental [q, S(q)].
    simulated_data : (N, 2) ndarray
        Simulated [q, S(q)] from saxs-fft.
    save_dir : str
        Directory for diagnostic plots.
    metric : str
        ``'mse'``, ``'apdist'`` or ``'shift_rmse'``.
    q_range : tuple(float, float) or None
        Optional (q_min, q_max) comparison window.
    dp_coeff : float
        Phase-distance weight for ``metric='apdist'`` (see :func:`compare_saxs_curves`).
    plot_apdist : bool
        When True and ``metric='apdist'``, save phase-warp plots under
        ``save_dir/apdist_plots/``.
    metric_kwargs : dict, optional
        ``shift_rmse`` parameters (see :func:`shift_rmse_params`). Must be
        empty for the other metrics.

    Returns
    -------
    float
        Loss over ``q_range``.

    Raises
    ------
    MetricFailed
        ``shift_rmse`` could not compute a loss from these curves.
    """
    _warn_apdist_kwargs_ignored(metric, dp_coeff, plot_apdist)

    if metric == "shift_rmse":
        return _compare_shift_rmse(
            experimental_data, simulated_data, save_dir, q_range, metric_kwargs,
        )
    if metric_kwargs:
        raise ValueError(f"metric_kwargs are only used by metric='shift_rmse' (got {metric!r})")

    loss, q_ref, I_exp_resampled, I_sim_resampled, da, dp = compare_saxs_curves(
        experimental_data,
        simulated_data,
        q_range,
        metric=metric,
        dp_coeff=dp_coeff,
    )
    plt.rcParams.update({"font.size": 18})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(q_ref, I_exp_resampled, linewidth=0.5, label="Exp_data", color="k")
    ax.plot(q_ref, I_sim_resampled, linewidth=3, label="Sim_data", color="red")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Intensity (arb. unit)")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    plt.title(str(loss))
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(
        os.path.join(save_dir, "compare_to_exp_saxsfft.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    if metric == "apdist" and plot_apdist:
        _save_apdist_plots(
            q_ref,
            I_exp_resampled,
            I_sim_resampled,
            os.path.join(save_dir, "apdist_plots"),
            da,
            dp,
            dp_coeff=dp_coeff,
        )

    return loss
