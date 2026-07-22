"""
Curve comparison utilities.

This module provides the same behavior as the original script:
- Crop curves to a (possibly user-specified) overlapping q-range.
- Resample the denser curve onto the sparser one using scipy.interpolate.interp1d.
- Scale the simulated curve by the ratio of the last ~5 points (tail mean).
- Compare in log10-intensity space using Amplitude–Phase Distance (AP).
- Optionally save diagnostic figures including APDist phase-warp plots.
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from apdist.distances import AmplitudePhaseDistance

DEFAULT_DP_COEFF = 0.5


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
    plt.rcParams.update({"font.size": 18})

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(q_ref, log_I_exp, linewidth=0.5, label="Exp (log10 I)", color="k")
    ax.plot(
        q_ref, log_I_sim_warped, linewidth=3, label="Sim warped (log10 I)", color="red"
    )
    ax.set_xscale("log")
    ax.set_ylabel("log10 Intensity")
    ax.set_xlabel("q ($\\AA^{-1}$)")
    ax.set_title(
        f"APDist after phase warp: da={da:.4f}, dp={dp:.4f}, "
        f"loss={weighted:.4f} (dp_coeff={dp_coeff})"
    )
    ax.legend()
    plt.savefig(
        os.path.join(save_dir, "compare_apdist_warped.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()

    plot_warping(q_ref, log_I_exp, log_I_sim, log_I_sim_warped, gam)
    plt.savefig(
        os.path.join(save_dir, "compare_apdist_warp_detail.png"),
        dpi=600,
        bbox_inches="tight",
    )
    plt.close()


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
):
    """
    Compare experimental and simulated S(q) using the wider q-range available
    from saxs-fft.

    Uses the same :func:`compare_saxs_curves` engine as :func:`compare_to_exp`,
    but defaults to a wider comparison window ``[0.003, 0.06]`` A^-1.

    Parameters
    ----------
    experimental_data : (N, 2) ndarray
        Experimental [q, S(q)].
    simulated_data : (N, 2) ndarray
        Simulated [q, S(q)] from saxs-fft.
    save_dir : str
        Directory for diagnostic plots.
    metric : str
        ``'mse'`` or ``'apdist'``.
    q_range : tuple(float, float) or None
        Optional (q_min, q_max) comparison window.
    dp_coeff : float
        Phase-distance weight for ``metric='apdist'`` (see :func:`compare_saxs_curves`).
    plot_apdist : bool
        When True and ``metric='apdist'``, save phase-warp plots under
        ``save_dir/apdist_plots/``.

    Returns
    -------
    float
        Loss over ``q_range``.
    """
    _warn_apdist_kwargs_ignored(metric, dp_coeff, plot_apdist)

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
