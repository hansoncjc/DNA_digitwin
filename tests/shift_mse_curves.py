"""Deterministic synthetic S(q) curves for the shift_mse tests.

All curves sit on the saxs-fft q grid of a cubic box (N_grid = 600,
L = 100 sigma, 24.6 nm particles, trim 3 bins at each end), which is the
grid of the inverse-design targets.
"""
import numpy as np

FCC_RATIOS = (1.0, 1.1547, 1.6330, 1.9149, 2.0, 2.3094)


def saxsfft_q(n_grid=600, box_length=100.0, diameter_nm=24.6, trim=slice(3, -3)):
    dq = 2.0 * np.pi / box_length
    q_corner = np.sqrt(3.0) * (n_grid // 2) * dq
    edges = np.arange(0.0, q_corner + dq, dq)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[trim] / (diameter_nm * 10.0)


def crystal_curve(q1=0.0129, amp=8.0, width=4e-4, plateau=1.0, upturn=200.0,
                  noise=0.0, seed=0, q=None):
    """FCC-like S(q): low-q upturn, Bragg peaks at q1 * FCC_RATIOS, plateau."""
    q = saxsfft_q() if q is None else q
    s = plateau + upturn * np.exp(-q / 0.0012)
    for i, ratio in enumerate(FCC_RATIOS):
        s = s + amp / (1.0 + 0.6 * i) * np.exp(-0.5 * ((q - q1 * ratio) / width) ** 2)
    if noise:
        s = s * (1.0 + noise * np.random.default_rng(seed).standard_normal(q.size))
    return np.column_stack([q, s])


def fluid_curve(q1=0.027, amp=1.6, decay=0.02, plateau=1.0, upturn=300.0,
                noise=0.0, seed=0, q=None):
    """Liquid-like S(q): damped oscillation with first maximum near q1."""
    q = saxsfft_q() if q is None else q
    x = q / q1
    osc = amp * np.exp(-(q - q1) / decay) * np.cos(2.0 * np.pi * (x - 1.0) * 0.82)
    s = plateau + upturn * np.exp(-q / 0.0010) + np.where(q > 0.4 * q1, osc, 0.0)
    s = np.clip(s, 0.05, None)
    if noise:
        s = s * (1.0 + noise * np.random.default_rng(seed).standard_normal(q.size))
    return np.column_stack([q, s])


def flat_curve(level=1.0, q=None):
    """No peak anywhere: monotonic decay onto a plateau (not dispersed)."""
    q = saxsfft_q() if q is None else q
    return np.column_stack([q, level + 50.0 * np.exp(-q / 0.002)])


def gas_curve(level=1.0, noise=0.03, seed=0, q=None):
    """Dispersed: S stays near 1 at every q."""
    q = saxsfft_q() if q is None else q
    rng = np.random.default_rng(seed)
    return np.column_stack([q, level * (1.0 + noise * rng.standard_normal(q.size))])
