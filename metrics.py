"""
Curve comparison utilities.

This module provides the same behavior as the original script:
- Crop curves to a (possibly user-specified) overlapping q-range.
- Resample the denser curve onto the sparser one using scipy.interpolate.interp1d.
- Scale the simulated curve by the ratio of the last ~5 points (tail mean).
- Compare in log10-intensity space using Amplitude–Phase Distance (AP).
- Optionally save the two diagnostic figures with the same filenames.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from apdist.distances import AmplitudePhaseDistance


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


def compare_saxs_curves(exp_data, sim_data, q_range=None, scale_intensity=True, metric='mse'):
    """
    Compare two SAXS curves in log space.

    Steps
    -----
    1) Determine the common overlap in q (optionally further restricted by q_range).
    2) Resample both curves onto a shared physical q-grid via linear interp1d.
    3) Scale the simulated intensity by the ratio of tail means ([-6:-1]).
    4) Compute Amplitude–Phase Distance (AP) on log10 intensities using a
       normalized parameter domain [0, 1].

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

    Returns
    -------
    d_ap : float
        AP distance (da + dp) between log10(I_exp) and log10(I_sim_scaled).
    q_ref : (K,) ndarray
        Physical q-grid used for interpolation, comparison output, and plotting.
        It is distinct from APDist's normalized parameter domain.
    I_exp_resampled : (K,) ndarray
        Experimental intensity on q_ref (resampled if needed).
    I_sim_scaled : (K,) ndarray
        Simulated intensity on q_ref after tail-mean scaling.
    """
    # Sanitize: drop non-finite rows, sort, collapse duplicate q.
    # Necessary because extract_exp_sq's nearest-neighbor q snapping can
    # produce repeated q's, which makes interp1d return NaN/inf.
    exp_data = _sanitize_curve(exp_data)
    sim_data = _sanitize_curve(sim_data)

    # Extract q and I
    q_exp, I_exp = exp_data[:, 0], exp_data[:, 1]
    q_sim, I_sim = sim_data[:, 0], sim_data[:, 1]

    # Determine overlapping q-range
    q_min_common = max(q_exp.min(), q_sim.min())
    q_max_common = min(q_exp.max(), q_sim.max())

    # Apply user-specified q_range if provided
    if q_range is not None:
        q_min_user, q_max_user = q_range
        q_min_common = max(q_min_common, q_min_user)
        q_max_common = min(q_max_common, q_max_user)

    # Truncate both curves
    mask_exp = (q_exp >= q_min_common) & (q_exp <= q_max_common)
    mask_sim = (q_sim >= q_min_common) & (q_sim <= q_max_common)
    q_exp_crop, I_exp_crop = q_exp[mask_exp], I_exp[mask_exp]
    q_sim_crop, I_sim_crop = q_sim[mask_sim], I_sim[mask_sim]

    # Resample both curves onto a shared log-spaced q-grid
    n_points = min(len(q_exp_crop), len(q_sim_crop))
    q_ref = np.logspace(np.log10(q_min_common), np.log10(q_max_common), n_points)

    I_exp_resampled = interp1d(
        q_exp_crop, I_exp_crop, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )(q_ref)

    I_sim_resampled = interp1d(
        q_sim_crop, I_sim_crop, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )(q_ref)

    # Avoid log(0) by clipping
    eps = 1e-10
    I_exp_resampled = np.clip(I_exp_resampled, eps, None)
    I_sim_resampled = np.clip(I_sim_resampled, eps, None)

    # Tail-based scaling used in the original code (mean over last ~5 points)
    scale_factor = np.mean(I_exp_resampled[-6:-1]) / np.mean(I_sim_resampled[-6:-1])
    I_sim_scaled = I_sim_resampled * scale_factor

    # Distance evaluation
    log_I_exp = np.log10(I_exp_resampled)
    log_I_sim = np.log10(I_sim_scaled)
    
    if metric == 'apdist':
        # q_ref remains the physical q-grid. APDist requires its independent
        # function parameter domain to be mapped to [0, 1].
        t_ap = np.linspace(0.0, 1.0, len(q_ref))
        da, dp = AmplitudePhaseDistance(t_ap, log_I_exp, log_I_sim)
        distance = da + dp
    elif metric == 'mse':
        # Mean Squared Error on log10 intensities
        distance = np.mean((log_I_exp - log_I_sim)**2)
    else:
        raise ValueError(f"Unknown metric chosen: {metric}")

    return distance, q_ref, I_exp_resampled, I_sim_scaled


def compare_to_exp(experimental_data, simulated_data, save_dir, metric='mse'):
    """
    Generate the two diagnostic plots (as in the original) and return the short-window score.

    Behavior (unchanged)
    --------------------
    - First compare in q ∈ [0.003, 0.03]; save 'compare_to_exp.png'.
    - Then compare in q ∈ [0.003, 0.07]; save 'compare_to_exp_full_curve.png'.
    - Return the first window’s AP distance.
    """
    q = [0.003, 0.03]
    mse, q_ref, I_exp_resampled, I_sim_resampled = compare_saxs_curves(experimental_data, simulated_data, q, metric=metric)
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(q_ref, I_exp_resampled, linewidth=0.5, label='Exp_data', color='k')
    ax.plot(q_ref, I_sim_resampled, linewidth=3, label='Sim_data', color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Intensity (arb. unit)')
    ax.set_xlabel('q ($\\AA^{-1}$)')
    plt.title(str(mse))
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'compare_to_exp.png'), dpi=600, bbox_inches='tight')
    plt.close()

    q = [0.003, 0.07]
    mse2, q_ref, I_exp_resampled, I_sim_resampled = compare_saxs_curves(experimental_data, simulated_data, q, metric=metric)
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(q_ref, I_exp_resampled, linewidth=0.5, label='Exp_data', color='k')
    ax.plot(q_ref, I_sim_resampled, linewidth=3, label='Sim_data', color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Intensity (arb. unit)')
    ax.set_xlabel('q ($\\AA^{-1}$)')
    plt.title(str(mse2))
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'compare_to_exp_full_curve.png'), dpi=600, bbox_inches='tight')
    plt.close()
    return mse


def compare_to_exp_saxsfft(
    experimental_data,
    simulated_data,
    save_dir,
    metric='mse',
    q_range=(0.003, 0.06),
):
    """
    Compare experimental and simulated S(q) using the wider q-range available
    from saxs-fft.

    Uses the same :func:`compare_saxs_curves` engine as :func:`compare_to_exp`,
    but defaults to a wider comparison window ``[0.003, 0.06]`` Å⁻¹.

    Parameters
    ----------
    experimental_data : (N, 2) ndarray
        Experimental [q, S(q)].
    simulated_data : (N, 2) ndarray
        Simulated [q, S(q)] from saxs-fft.
    save_dir : str
        Directory for diagnostic plots.
    q_range : tuple(float, float) or None
        Optional (q_min, q_max) comparison window. Defaults to
        ``[0.003, 0.06]`` Å⁻¹.

    Returns
    -------
    float
        AP distance (loss) over ``q_range``.
    """
    mse, q_ref, I_exp_resampled, I_sim_resampled = compare_saxs_curves(
        experimental_data,
        simulated_data,
        q_range,
        metric=metric,
    )
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(q_ref, I_exp_resampled, linewidth=0.5, label='Exp_data', color='k')
    ax.plot(q_ref, I_sim_resampled, linewidth=3, label='Sim_data', color='red')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Intensity (arb. unit)')
    ax.set_xlabel('q ($\\AA^{-1}$)')
    plt.title(str(mse))
    plt.legend()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'compare_to_exp_saxsfft.png'), dpi=600, bbox_inches='tight')
    plt.close()
    return mse
