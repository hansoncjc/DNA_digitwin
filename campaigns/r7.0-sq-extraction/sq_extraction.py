"""
R7.0: audit of measured I(q) and of the form-factor file, and S(q) extraction
with a refitted form factor and background (2026-11-05).

The current extraction (``scattering.extract_exp_sq``) divides the measured
I(q) by the curve in ``formfactors/sasmodels_sphere_fit.txt``, scaled to the
data in q 0.02-0.03 A^-1. This module does not change that path; it checks the
inputs and writes a second S(q) for comparison:

    S(q) = (I(q) - B) / (c * P(q; R, pd)),   sigma_S = sigma_I / (c * P)

P is the sasmodels ``sphere`` shape (Gaussian radius polydispersity, 3 sigma,
35 points, the SasView defaults) without background. R and pd are shared by
all curves; the scale c and the flat background B >= 0 are per curve. They are
fitted to the measured I(q) in a window above the first form-factor minimum,
where S(q) is taken as 1 (an approximation: weak structure remains there).

Run (writes into --out):

    python campaigns/r7.0-sq-extraction/sq_extraction.py \
        --ff formfactors/sasmodels_sphere_fit.txt \
        --out campaigns/r7.0-sq-extraction/out/r4 --old-trim-tail 200 \
        campaigns/r4-measured/data/*.npy
"""
import argparse
import csv
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

CU_KALPHA = 1.5406       # A
EIGER2_PIXEL_MM = 0.075  # Eiger2 R 1M pixel pitch (mm)

DEFAULT_FIT_WINDOW = (0.033, 0.12)
DEFAULT_REL_FLOOR = 0.03
SCAN_WINDOWS = [(lo, hi) for lo in (0.033, 0.040, 0.045) for hi in (0.09, 0.12, 0.143)]


# ---------------------------------------------------------------- loading


def load_iq(path):
    """Return (q, I, sigma) from a .npy array or a text file.

    Text lines whose first two fields do not parse as numbers (headers,
    comments) are skipped. ``sigma`` is None when there is no third column.
    Rows with non-finite or non-positive q or I are dropped; q is sorted.
    """
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
    else:
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            try:
                vals = [float(x) for x in parts[:3]]
            except ValueError:
                continue
            if len(vals) >= 2:
                rows.append(vals if len(vals) == 3 else vals + [np.nan])
        arr = np.array(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns [q, I]")
    q, i = arr[:, 0], arr[:, 1]
    s = arr[:, 2] if arr.shape[1] >= 3 else np.full_like(q, np.nan)
    keep = np.isfinite(q) & np.isfinite(i) & (q > 0) & (i > 0)
    order = np.argsort(q[keep])
    q, i, s = q[keep][order], i[keep][order], s[keep][order]
    if not np.all(np.isfinite(s) & (s > 0)):
        s = None
    return q, i, s


# ---------------------------------------------------------- form factor


def sphere_pd(q, radius, pd, nsigma=3.0, npts=35):
    """Polydisperse sphere P(q), normalized to 1 at q -> 0.

    Gaussian number distribution of the radius with relative width ``pd``,
    sampled on ``npts`` points over +-``nsigma`` widths (sasmodels defaults),
    volume-squared weighted: <V^2 F^2> / <V^2>.
    """
    q = np.asarray(q, dtype=float)
    sig = pd * radius
    r = np.linspace(radius - nsigma * sig, radius + nsigma * sig, npts) if pd > 0 else np.array([radius])
    w = np.exp(-0.5 * ((r - radius) / sig) ** 2) if pd > 0 else np.ones(1)
    keep = r > 0
    r, w = r[keep], w[keep]
    x = np.outer(q, r)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(x > 1e-6, 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3, 1.0)
    wv2 = w * r ** 6
    return (f ** 2 * wv2).sum(axis=1) / wv2.sum()


def apply_correction(q, correction):
    """Multiplicative correction r(q) from ``empirical_correction`` (1 where absent)."""
    if correction is None:
        return np.ones_like(np.asarray(q, dtype=float))
    grid, r = correction
    return np.interp(q, grid, r, left=1.0, right=1.0)


def form_factor(q, radius, pd, correction=None):
    """``sphere_pd`` times an optional empirical correction (see ``empirical_correction``)."""
    return sphere_pd(q, radius, pd) * apply_correction(q, correction)


def empirical_correction(curves, radius, pd, scales, backgrounds, q_range=(0.015, 0.075),
                         taper=0.005, smooth=15, dq=3.4e-4):
    """Measured correction r(q) = P_true / sphere_pd from dispersed samples.

    Each curve gives (I - B_k) / (c_k sphere_pd); these are averaged on a
    common grid of step ``dq`` with weights 1/sigma^2, smoothed
    (Savitzky-Golay, ``smooth`` points, order 2) and set to 1 outside
    ``q_range`` with a linear ``taper`` (A^-1) inside each end. The curves must
    have S = 1 in ``q_range``; use only samples without structure.

    Returns a dict with the grid, the raw mean, its error and the smoothed,
    tapered correction ``r``.
    """
    q_lo = min(float(q.min()) for q, _, _ in curves)
    q_hi = max(float(q.max()) for q, _, _ in curves)
    grid = np.arange(q_lo, q_hi, dq)
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for (q, i, s), c, b in zip(curves, scales, backgrounds):
        if s is None:
            raise ValueError("empirical_correction needs sigma for every curve")
        p = sphere_pd(q, radius, pd)
        ok = (grid >= q.min()) & (grid <= q.max())
        y = np.interp(grid[ok], q, (i - b) / (c * p))
        sy = np.interp(grid[ok], q, s / (c * p))
        num[ok] += y / sy ** 2
        den[ok] += 1.0 / sy ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = num / den
        err = 1.0 / np.sqrt(den)
    sm = savgol_filter(np.nan_to_num(raw, nan=1.0), smooth, 2)
    lo, hi = q_range
    w = np.clip(np.minimum((grid - lo) / taper, (hi - grid) / taper), 0.0, 1.0)
    r = 1.0 + w * (sm - 1.0)
    return {"q": grid, "r": r, "raw": raw, "err": err, "q_range": [float(lo), float(hi)],
            "taper": float(taper), "smooth": int(smooth)}


def mean_s_artifact(sq_curves, band=(0.026, 0.045), step=2.5e-4):
    """RMS of (mean S - 1) over ``band`` on a common absolute-q grid.

    Structure peaks sit at different q in different samples and partly
    average out; a form-factor error sits at the same q in every sample and
    does not. Returns (rms, grid, mean S).
    """
    grid = np.arange(band[0], band[1], step)
    rows = []
    for sq in sq_curves:
        q, s = sq[:, 0], sq[:, 1]
        if q.min() <= band[0] and q.max() >= band[1]:
            rows.append(np.interp(grid, q, savgol_filter(s, 7, 2)))
    if not rows:
        return np.nan, grid, np.full_like(grid, np.nan)
    mean = np.mean(rows, axis=0)
    return float(np.sqrt(np.mean((mean - 1.0) ** 2))), grid, mean


def first_ff_minimum(radius, pd=0.0):
    """q of the first minimum of ``sphere_pd`` (qR = 4.4934 when pd = 0)."""
    q = np.linspace(3.5 / radius, 5.5 / radius, 2001)
    return float(q[np.argmin(sphere_pd(q, radius, pd))])


# Search windows of the I q^4 extrema in units of q * R_ref: first minimum,
# second maximum, second minimum, third maximum (monodisperse sphere:
# qR = 4.49, 5.76, 7.73, 9.10 for P; the q^4 factor moves the maxima up).
IQ4_EXTREMA = (("min1", 3.9, 5.36, "min"), ("max2", 5.36, 6.97, "max"),
               ("min2", 6.97, 8.84, "min"), ("max3", 8.30, 10.7, "max"))


def iq4_extrema(q, y, radius_ref, smooth=9):
    """Positions (A^-1) of the extrema of y q^4 listed in ``IQ4_EXTREMA``.

    ln(y q^4) is Savitzky-Golay smoothed (window ``smooth``, order 2) inside
    each window, the extreme point is refined by a parabola. Windows with
    fewer than ``smooth + 2`` points give NaN.
    """
    out = {}
    for name, lo, hi, kind in IQ4_EXTREMA:
        m = (q >= lo / radius_ref) & (q <= hi / radius_ref)
        if m.sum() < smooth + 2:
            out[name] = np.nan
            continue
        qq = q[m]
        ys = savgol_filter(np.log(y[m] * qq ** 4), smooth, 2)
        k = int(np.argmin(ys) if kind == "min" else np.argmax(ys))
        if 0 < k < qq.size - 1:
            y0, y1, y2 = ys[k - 1:k + 2]
            den = y0 - 2.0 * y1 + y2
            d = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
            out[name] = float(qq[k] + d * (qq[k + 1] - qq[k]))
        else:
            out[name] = float(qq[k])
    return out


def audit_formfactor(path):
    """Fit a form-factor file as ``scale * sphere_pd(q; R, pd) + background``.

    Returns the parameters, the RMS of the log residual, and the background
    share of the file's value at its largest q.
    """
    arr = np.loadtxt(path, skiprows=1)
    q, y = arr[:, 0], arr[:, 1]

    def resid(p):
        return np.log(np.exp(p[2]) * sphere_pd(q, p[0], p[1]) + p[3]) - np.log(y)

    p0 = [130.0, 0.1, np.log(y[0]), 1e-4]
    fit = least_squares(resid, p0, bounds=([20, 0.0, -50, 0.0], [1000, 0.5, 50, np.inf]),
                        x_scale=[10, 0.05, 1, 1e-3])
    radius, pd, lscale, bkg = (float(v) for v in fit.x)
    model_no_bkg = np.exp(lscale) * sphere_pd(q, radius, pd)
    return {
        "file": str(path),
        "n_points": int(q.size),
        "q_min": float(q.min()),
        "q_max": float(q.max()),
        "radius_A": radius,
        "pd": pd,
        "scale": float(np.exp(lscale)),
        "background": bkg,
        "rms_ln_resid": float(np.sqrt(np.mean(fit.fun ** 2))),
        "background_share_at_qmax": float(bkg / y[-1]),
        "background_share_at_0.075": float(bkg / np.interp(0.075, q, y)),
        "first_ff_min_q": first_ff_minimum(radius, pd),
        "_q": q, "_y": y, "_model_no_bkg": model_no_bkg,
    }


# ------------------------------------------------------- merged segments


def find_merge(q, ratio=1.5, run=5):
    """Find where the q spacing widens (two detector distances merged).

    The fine and coarse spacings are the medians of the first and last
    quarter of the steps. The coarse segment starts at the first point from
    which ``run`` consecutive steps are at least 0.9 of the coarse spacing; the
    single transition step between the two grids can be shorter than either.
    Returns None when the coarse spacing is below ``ratio`` times the fine one.
    """
    dq = np.diff(q)
    if dq.size < 4 * run:
        return None
    dq_low = float(np.median(dq[: dq.size // 4]))
    dq_high = float(np.median(dq[-(dq.size // 4):]))
    if dq_high < ratio * dq_low:
        return None
    coarse = dq >= 0.9 * dq_high
    for k in range(dq.size - run + 1):
        if coarse[k:k + run].all():
            return {"q_merge": float(q[k]), "index": k, "dq_low": dq_low, "dq_high": dq_high}
    return None


def implied_distance_mm(dq, wavelength=CU_KALPHA, pixel_mm=EIGER2_PIXEL_MM):
    """Sample-detector distance for which one pixel spans ``dq`` (small angles)."""
    return (2.0 * np.pi / wavelength) * pixel_mm / dq


def seam_ratio(q, i, sigma, q_merge, half_width=0.003):
    """I(high-q segment) / I(low-q segment) at the merge, with its 1-sigma error.

    Each side is fitted (weighted by I/sigma) with a line in (ln q, ln I) over
    ``half_width`` and evaluated at ``q_merge``. The error is the larger of the
    propagated and the scatter-based one; real structure near the merge also
    bends the lines, so the ratio is an estimate, not a calibration.
    """
    w_all = i / sigma if sigma is not None else np.ones_like(i)

    def side(mask):
        x = np.log(q[mask])
        a = np.column_stack([np.ones_like(x), x]) * w_all[mask][:, None]
        b = np.log(i[mask]) * w_all[mask]
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        cov = np.linalg.pinv(a.T @ a)
        dof = max(int(mask.sum()) - 2, 1)
        chi2_red = float(np.sum((a @ coef - b) ** 2) / dof)
        cov = cov * max(chi2_red, 1.0) if sigma is not None else cov * chi2_red
        xm = np.array([1.0, np.log(q_merge)])
        return float(xm @ coef), float(np.sqrt(xm @ cov @ xm))

    lo = (q < q_merge) & (q > q_merge - half_width)
    hi = (q >= q_merge) & (q < q_merge + half_width)
    if lo.sum() < 5 or hi.sum() < 5:
        return np.nan, np.nan
    a, ea = side(lo)
    b, eb = side(hi)
    r = float(np.exp(b - a))
    return r, float(r * np.hypot(ea, eb))


# ------------------------------------------------------------ joint fit


def fit_formfactor_joint(curves, window=DEFAULT_FIT_WINDOW, radius0=134.0, pd0=0.1,
                         rel_floor=DEFAULT_REL_FLOOR, fixed=False, correction=None):
    """Fit I_k = c_k * P(q; R, pd) + B_k (B_k >= 0) in ``window``, R and pd shared.

    With ``fixed=True``, R = ``radius0`` and pd = ``pd0`` are held and only the
    c_k and B_k are fitted. ``correction`` multiplies P (``form_factor``).

    ``curves`` is a list of (q, I, sigma). Residuals are divided by
    sqrt(sigma^2 + (rel_floor * I)^2); the floor stands for the structure left
    in the window, which the model sets to S = 1.
    """
    masks, sig = [], []
    for q, i, s in curves:
        m = (q >= window[0]) & (q <= window[1])
        if m.sum() < 10:
            raise ValueError(f"fewer than 10 points in fit window {window}")
        masks.append(m)
        s0 = s if s is not None else np.zeros_like(i)
        sig.append(np.sqrt(s0[m] ** 2 + (rel_floor * i[m]) ** 2))
    n = len(curves)

    def unpack(p):
        if fixed:
            return radius0, pd0, np.exp(p[:n]), p[n:2 * n]
        return p[0], p[1], np.exp(p[2:2 + n]), p[2 + n:2 + 2 * n]

    def resid(p):
        radius, pd, c, b = unpack(p)
        out = []
        for k, (q, i, _) in enumerate(curves):
            m = masks[k]
            out.append((i[m] - (c[k] * form_factor(q[m], radius, pd, correction) + b[k])) / sig[k])
        return np.concatenate(out)

    p0 = [] if fixed else [radius0, pd0]
    p0 += [float(np.log(np.median(i[m] / form_factor(q[m], radius0, pd0, correction))))
           for (q, i, _), m in zip(curves, masks)]
    p0 += [0.0] * n
    lb = ([] if fixed else [0.6 * radius0, 0.01]) + [-np.inf] * n + [0.0] * n
    ub = ([] if fixed else [1.4 * radius0, 0.4]) + [np.inf] * n + [np.inf] * n
    fit = least_squares(resid, p0, bounds=(lb, ub))
    radius, pd, c, b = unpack(fit.x)
    dof = max(fit.fun.size - fit.x.size, 1)
    return {
        "window": [float(window[0]), float(window[1])],
        "radius_A": float(radius),
        "pd": float(pd),
        "scale": [float(v) for v in c],
        "background": [float(v) for v in b],
        "chi2_red": float(np.sum(fit.fun ** 2) / dof),
        "rel_floor": float(rel_floor),
        "fixed_formfactor": bool(fixed),
    }


def extract_sq(q, i, sigma, radius, pd, scale, background, correction=None):
    """(N, 3) array [q, S, sigma_S] with S = (I - B) / (c P)."""
    denom = scale * form_factor(q, radius, pd, correction)
    s = (i - background) / denom
    ds = sigma / denom if sigma is not None else np.full_like(q, np.nan)
    return np.column_stack([q, s, ds])


# ---------------------------------------------------------- descriptors


def smoothed_max(q, s, window):
    m = (q >= window[0]) & (q <= window[1])
    if m.sum() < 7:
        return np.nan, np.nan
    sm = savgol_filter(s[m], 7, 2)
    k = int(np.argmax(sm))
    return float(sm[k]), float(q[m][k])


def window_mean(q, s, window):
    m = (q >= window[0]) & (q <= window[1])
    return float(np.mean(s[m])) if m.any() else np.nan


def peak_fwhm(q, s, window):
    """Gaussian + linear baseline fitted around the largest maximum in ``window``.

    Returns (q_peak, FWHM) in A^-1, or NaNs when the fit fails.
    """
    _, q0 = smoothed_max(q, s, window)
    if not np.isfinite(q0):
        return np.nan, np.nan
    m = (q >= 0.7 * q0) & (q <= 1.3 * q0)

    def model(p, x):
        return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3] + p[4] * (x - q0)

    p0 = [float(s[m].max() - s[m].min()), q0, 0.1 * q0, float(s[m].min()), 0.0]
    try:
        fit = least_squares(lambda p: model(p, q[m]) - s[m], p0,
                            bounds=([0, 0.8 * q0, 0.005 * q0, -np.inf, -np.inf],
                                    [np.inf, 1.2 * q0, 0.5 * q0, np.inf, np.inf]))
    except ValueError:
        return np.nan, np.nan
    return float(fit.x[1]), float(2.3548 * fit.x[2])


def resolution_fwhm(distance_mm, beam_fwhm_mm, q=0.0, wavelength=CU_KALPHA,
                    pixel_mm=EIGER2_PIXEL_MM, dlambda_rel=0.0025):
    """Approximate pinhole-SAXS resolution FWHM (A^-1), contributions in quadrature.

    Beam size on the detector (``beam_fwhm_mm``), one pixel (box of width p,
    Gaussian-equivalent FWHM 0.68 p) and the unresolved Cu K-alpha doublet
    (``dlambda_rel`` of q). The beam size on the detector is larger than at
    the sample, so a sample-side beam size gives a lower bound.
    """
    k = 2.0 * np.pi / wavelength
    beam = k * beam_fwhm_mm / distance_mm
    pixel = k * 0.68 * pixel_mm / distance_mm
    lam = q * dlambda_rel
    return float(np.sqrt(beam ** 2 + pixel ** 2 + lam ** 2))


# ------------------------------------------------------- old extraction


def _stub_if_missing(name, attrs=()):
    try:
        importlib.import_module(name)
    except ImportError:
        mod = types.ModuleType(name)
        for a in attrs:
            setattr(mod, a, None)
        sys.modules[name] = mod
        if "." in name:
            parent, child = name.rsplit(".", 1)
            _stub_if_missing(parent)
            setattr(sys.modules[parent], child, mod)


def old_extraction(q, i, ffpath, trim_tail):
    """S(q) as the BO objective builds it (``scattering.extract_exp_sq``).

    The GSD, MC-DFM and APDist imports of ``scattering`` / ``metrics`` are not
    used by this path; they are replaced by empty modules when missing, as in
    tests/conftest.py.
    """
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _stub_if_missing("gsd.hoomd")
    _stub_if_missing("Scattering_Simulator", ("pairwise_method",))
    _stub_if_missing("apdist.distances", ("AmplitudePhaseDistance",))
    from scattering import extract_exp_sq
    arr = np.column_stack([q, i])
    if trim_tail and arr.shape[0] > trim_tail:
        arr = arr[:-trim_tail]
    return extract_exp_sq(arr, str(ffpath), q_min=0.02, q_max=0.03)


# ------------------------------------------------------------------ CLI


def _plot_correction(out, ec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ok = np.isfinite(ec["raw"])
    ax.errorbar(ec["q"][ok], ec["raw"][ok], ec["err"][ok], fmt=".", ms=2, color="0.6", lw=0.5,
                label="weighted mean of (I - B)/(c P_sphere)")
    ax.plot(ec["q"], ec["r"], "r-", label=f"correction used ({ec['q_range'][0]}-{ec['q_range'][1]})")
    ax.axhline(1, color="k", ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.5, 1.8); ax.set_xlabel("q (A$^{-1}$)"); ax.legend(fontsize=8)
    ax.set_title("empirical form-factor correction r(q) = P / P_sphere")
    fig.tight_layout(); fig.savefig(out / "ff_correction.png", dpi=110); plt.close(fig)


def _plots(out, names, curves, merges, ff, joint, new_sq, old_sq, peak_window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_min_ff = first_ff_minimum(joint["radius_A"], joint["pd"])

    # form-factor file
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    q, y = ff["_q"], ff["_y"]
    ax[0].loglog(q, y, "k.", ms=3, label="form-factor file")
    ax[0].loglog(q, ff["_model_no_bkg"], "b-", label=f"sphere R={ff['radius_A']:.2f} A, pd={ff['pd']:.3f}")
    ax[0].axhline(ff["background"], color="r", ls="--", label=f"background {ff['background']:.3g}")
    ax[0].set_xlabel("q (A$^{-1}$)"); ax[0].legend(fontsize=8); ax[0].set_title("form-factor file = sphere + background")
    ax[1].semilogx(q, ff["background"] / y, "r-")
    ax[1].set_xlabel("q (A$^{-1}$)"); ax[1].set_ylabel("background / file value")
    ax[1].set_title("share of the background in the divisor")
    fig.tight_layout(); fig.savefig(out / "ff_audit.png", dpi=110); plt.close(fig)

    # I(q) audit
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, (q, i, s), mg in zip(names, curves, merges):
        ref = np.median((i * q ** 4)[(q > 0.03) & (q < 0.06)])
        line, = ax[0].loglog(q, i * q ** 4 / ref, lw=0.8, label=name)
        if s is not None:
            ax[1].loglog(q, s / i, lw=0.8, color=line.get_color())
        if mg:
            for a in ax:
                a.axvline(mg["q_merge"], color=line.get_color(), lw=0.6, ls=":")
    qq = np.geomspace(0.003, 0.14, 400)
    p = sphere_pd(qq, joint["radius_A"], joint["pd"])
    ax[0].loglog(qq, p * qq ** 4 / np.median((p * qq ** 4)[(qq > 0.03) & (qq < 0.06)]), "k--", lw=1.5,
                 label="refitted sphere")
    ax[0].set_title("I q$^4$ (dotted: merge of the two distances)"); ax[0].legend(fontsize=7)
    ax[1].set_title("sigma / I"); ax[0].set_xlabel("q (A$^{-1}$)"); ax[1].set_xlabel("q (A$^{-1}$)")
    fig.tight_layout(); fig.savefig(out / "iq_audit.png", dpi=110); plt.close(fig)

    # S(q) old vs new
    n = len(names)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.2 * nrow), squeeze=False)
    for ax, name, sq_new, sq_old, mg in zip(axes.flat, names, new_sq, old_sq, merges):
        if sq_old is not None:
            ax.plot(sq_old[:, 0], sq_old[:, 1], color="0.6", lw=1, label="current extraction")
        ax.plot(sq_new[:, 0], savgol_filter(sq_new[:, 1], 7, 2), "r-", lw=1.1, label="refit (smoothed)")
        ax.axhline(1, color="k", ls=":", lw=0.8)
        ax.axvline(q_min_ff, color="g", ls="--", lw=0.8, label="first FF minimum")
        if mg:
            ax.axvline(mg["q_merge"], color="orange", lw=1, label="merge")
        ax.axvspan(*peak_window, color="b", alpha=0.05)
        ax.set_xscale("log"); ax.set_xlim(sq_new[0, 0], min(0.12, sq_new[-1, 0])); ax.set_ylim(0, 2.6)
        ax.set_title(name, fontsize=10); ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"S = (I - B)/(c P), shared R={joint['radius_A']:.1f} A, pd={joint['pd']:.3f}")
    fig.tight_layout(); fig.savefig(out / "sq_compare.png", dpi=110); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="I(q) files (.npy or text: q, I[, sigma])")
    ap.add_argument("--ff", required=True, help="form-factor file used by the current extraction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-window", nargs=2, type=float, default=DEFAULT_FIT_WINDOW)
    ap.add_argument("--rel-floor", type=float, default=DEFAULT_REL_FLOOR)
    ap.add_argument("--old-trim-tail", type=int, default=0,
                    help="trim_tail for the comparison with the current extraction (R4: 200)")
    ap.add_argument("--peak-window", nargs=2, type=float, default=(0.006, 0.02),
                    help="q window of the first structure peak")
    ap.add_argument("--tail-window", nargs=2, type=float, default=(0.06, 0.075))
    ap.add_argument("--no-scan", action="store_true", help="skip the fit-window and per-file scans")
    ap.add_argument("--fix-ff", nargs=2, type=float, metavar=("R_A", "PD"),
                    help="hold R and pd (e.g. from a fit to dispersed samples); fit only scale and background")
    ap.add_argument("--ff-correction", help="ff_correction.npy written by --write-correction; multiplies P")
    ap.add_argument("--write-correction", action="store_true",
                    help="the files are dispersed samples: write ff_correction.npy from them")
    ap.add_argument("--correction-range", nargs=2, type=float, default=(0.015, 0.075))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in args.files]
    names = [f.stem for f in files]
    curves = [load_iq(f) for f in files]
    merges = [find_merge(q) for q, _, _ in curves]

    ff = audit_formfactor(args.ff)
    correction = None
    if args.ff_correction and not args.fix_ff:
        raise SystemExit("--ff-correction belongs to one R and pd: pass them with --fix-ff")
    if args.ff_correction:
        arr = np.load(args.ff_correction)
        correction = (arr[:, 0], arr[:, 1])
    if args.fix_ff:
        joint = fit_formfactor_joint(curves, tuple(args.fit_window), args.fix_ff[0], args.fix_ff[1],
                                     args.rel_floor, fixed=True, correction=correction)
    else:
        joint = fit_formfactor_joint(curves, tuple(args.fit_window), ff["radius_A"], ff["pd"], args.rel_floor)

    corr_info = None
    if args.write_correction:
        if correction is not None:
            raise SystemExit("--write-correction and --ff-correction are exclusive")
        ec = empirical_correction(curves, joint["radius_A"], joint["pd"], joint["scale"],
                                  joint["background"], q_range=tuple(args.correction_range))
        np.save(out / "ff_correction.npy", np.column_stack([ec["q"], ec["r"], ec["raw"], ec["err"]]))
        corr_info = {"file": "ff_correction.npy", "columns": ["q", "r", "raw_mean", "raw_err"],
                     "radius_A": joint["radius_A"], "pd": joint["pd"],
                     **{k: ec[k] for k in ("q_range", "taper", "smooth")}}
        _plot_correction(out, ec)
    elif correction is not None:
        corr_info = {"file": str(args.ff_correction)}

    scan = {}
    if not args.no_scan and not args.fix_ff:
        scan["windows"] = [
            {k: v for k, v in fit_formfactor_joint(curves, w, ff["radius_A"], ff["pd"], args.rel_floor).items()
             if k in ("window", "radius_A", "pd", "chi2_red")}
            for w in SCAN_WINDOWS
        ]
        scan["per_file"] = []
        for name, c in zip(names, curves):
            f1 = fit_formfactor_joint([c], tuple(args.fit_window), ff["radius_A"], ff["pd"], args.rel_floor)
            scan["per_file"].append({"file": name, "radius_A": f1["radius_A"], "pd": f1["pd"],
                                     "chi2_red": f1["chi2_red"]})

    q_ffmin = first_ff_minimum(joint["radius_A"], joint["pd"])
    r_ref = ff["radius_A"]
    q_dense = np.linspace(0.5 * 3.9 / r_ref, 1.5 * 10.7 / r_ref, 4000)
    model_extrema = {
        "formfactor_file_no_background": iq4_extrema(q_dense, sphere_pd(q_dense, r_ref, ff["pd"]), r_ref),
        "formfactor_file_as_is": iq4_extrema(
            q_dense, sphere_pd(q_dense, r_ref, ff["pd"]) + ff["background"] / ff["scale"], r_ref),
        "joint_refit": iq4_extrema(q_dense, sphere_pd(q_dense, joint["radius_A"], joint["pd"]), r_ref),
    }
    cluster_window = (args.peak_window[1], q_ffmin)
    rows, new_sq, old_sq = [], [], []
    for k, (name, (q, i, s), mg) in enumerate(zip(names, curves, merges)):
        sq = extract_sq(q, i, s, joint["radius_A"], joint["pd"], joint["scale"][k], joint["background"][k],
                        correction)
        np.save(out / f"{name}_sq.npy", sq)
        new_sq.append(sq)
        try:
            old = old_extraction(q, i, args.ff, args.old_trim_tail)
        except Exception as exc:  # comparison only
            print(f"[warn] current extraction unavailable for {name}: {exc}")
            old = None
        old_sq.append(old)

        row = {"file": name, "n": int(q.size), "q_min": float(q.min()), "q_max": float(q.max()),
               "has_sigma": s is not None}
        if mg:
            ratio, err = seam_ratio(q, i, s, mg["q_merge"])
            row.update(q_merge=mg["q_merge"], dq_low=mg["dq_low"], dq_high=mg["dq_high"],
                       distance_low_mm=implied_distance_mm(mg["dq_low"]),
                       distance_high_mm=implied_distance_mm(mg["dq_high"]),
                       seam_ratio=ratio, seam_ratio_err=err)
        b = joint["background"][k]
        row["background_share_at_0.075"] = float(b / np.interp(0.075, q, i))
        ext = iq4_extrema(q, i, r_ref)
        ref = model_extrema["formfactor_file_no_background"]
        for ek in ext:
            row[f"iq4_{ek}"] = ext[ek]
            row[f"iq4_{ek}_vs_file"] = ext[ek] / ref[ek]
        for tag, curve in (("old", old), ("new", sq)):
            if curve is None:
                continue
            qq, ss = curve[:, 0], curve[:, 1]
            row[f"{tag}_peak1"] = smoothed_max(qq, ss, args.peak_window)[0]
            row[f"{tag}_cluster2"] = smoothed_max(qq, ss, cluster_window)[0]
            row[f"{tag}_tail_mean"] = window_mean(qq, ss, args.tail_window)
        q1, fwhm = peak_fwhm(sq[:, 0], sq[:, 1], args.peak_window)
        row.update(q1=q1, fwhm1=fwhm, fwhm1_over_q1=fwhm / q1 if np.isfinite(q1) else np.nan)
        rows.append(row)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    report = {
        "formfactor_file": {k: v for k, v in ff.items() if not k.startswith("_")},
        "joint_fit": joint,
        "first_ff_min_q_refit": q_ffmin,
        "iq4_extrema_models": model_extrema,
        "ff_correction": corr_info,
        "mean_s_artifact_rms_0.026_0.045": {
            "new": mean_s_artifact(new_sq)[0],
            "current": mean_s_artifact([o for o in old_sq if o is not None])[0],
        },
        "cluster_window": list(cluster_window),
        "scan": scan,
        "files": [str(f) for f in files],
    }
    with open(out / "fit.json", "w") as fh:
        json.dump(report, fh, indent=2)

    _plots(out, names, curves, merges, ff, joint, new_sq, old_sq, args.peak_window)
    print(json.dumps({k: report[k] for k in ("formfactor_file", "joint_fit")}, indent=2))
    return report


if __name__ == "__main__":
    main()
