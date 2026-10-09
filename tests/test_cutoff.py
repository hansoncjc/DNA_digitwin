"""Pair-table cutoff selection (dynamic t_tol_lj = c/U_0, fixed modes, L/2 check)."""
import itertools

import pytest

import simulation
from simulation import (
    DEFAULT_TAIL_ENERGY_CUT,
    _compute_table_bounds,
    resolve_table_bounds,
)

RHO = 0.005
N = 5000


def _driver_bounds(U_0, n, m, r0):
    """rmax of the 2026-08/09 inverse driver patch ``_run_simulation_dynamic_cutoff``.

    That patch used ``rmin = 0.7 * r0``. ``rmin`` here follows the current
    ``_table_rmin`` (``rmin_k * r0``).
    """
    t_tol_lj = 0.1 / float(U_0)
    return _compute_table_bounds("modified_lj", U_0, n, m, r0, None, t_tol_lj=t_tol_lj)


GRID = list(itertools.product(
    (0.5, 3.0, 8.0),          # U_0
    (5.5, 12.0, 15.0),        # n
    (0.0, 0.5, 1.0),          # m_rel
    (1.0, 2.5, 3.5),          # r0
))


def test_default_is_dynamic_point_one():
    assert DEFAULT_TAIL_ENERGY_CUT == 0.1


@pytest.mark.parametrize("U_0,n,m_rel,r0", GRID)
def test_default_matches_driver_patch_exactly(U_0, n, m_rel, r0):
    m = 4.0 + m_rel * (n - 5.0)
    rmin_ref, rmax_ref = _driver_bounds(U_0, n, m, r0)
    b = resolve_table_bounds("modified_lj", U_0, n, m, r0, None, N=N, density=RHO)
    assert b["rmin"] == rmin_ref
    assert b["rmin"] == pytest.approx(simulation.DEFAULT_RMIN_K * r0)
    assert b["rmax"] == rmax_ref
    assert b["t_tol_lj"] == 0.1 / float(U_0)
    assert b["tail_energy_cut"] == 0.1
    assert b["rmax_fixed"] is False
    assert b["L"] == pytest.approx(100.0)


def test_explicit_tail_energy_cut_equals_default():
    a = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO)
    b = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                             tail_energy_cut=0.1)
    assert a == b


def test_explicit_t_tol_lj_is_fixed_tolerance():
    b = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                             t_tol_lj=0.02)
    rmin, rmax = _compute_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, t_tol_lj=0.02)
    assert (b["rmin"], b["rmax"], b["t_tol_lj"]) == (rmin, rmax, 0.02)
    assert b["tail_energy_cut"] is None


def test_attractive_tail_at_rmax_equals_cut():
    U_0, n, m, r0 = 2.0, 12.0, 6.0, 2.5
    b = resolve_table_bounds("modified_lj", U_0, n, m, r0, None, N=N, density=RHO,
                             tail_energy_cut=0.25)
    tail = U_0 * n / (n - m) * (r0 / b["rmax"]) ** m
    assert tail == pytest.approx(0.25)


def test_fixed_rmax():
    b = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                             rmax=6.0)
    assert b["rmax"] == 6.0 and b["rmax_fixed"] is True
    assert b["rmin"] == pytest.approx(simulation.DEFAULT_RMIN_K * 2.5)
    assert b["t_tol_lj"] is None and b["tail_energy_cut"] is None
    with pytest.raises(ValueError, match="must exceed rmin"):
        resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                             rmax=1.0)


@pytest.mark.parametrize("kwargs", [
    {"tail_energy_cut": 0.1, "t_tol_lj": 0.02},
    {"tail_energy_cut": 0.1, "rmax": 6.0},
])
def test_conflicting_cutoff_arguments(kwargs):
    with pytest.raises(ValueError, match="not both"):
        resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                             **kwargs)


def test_dynamic_needs_positive_U0():
    with pytest.raises(ValueError, match="U_0 > 0"):
        resolve_table_bounds("modified_lj", 0.0, 12.0, 6.0, 2.5, None, N=N, density=RHO)


@pytest.mark.parametrize("kwargs", [{}, {"t_tol_lj": 0.02}, {"rmax": 6.0}])
def test_minimum_image_checked_for_every_mode(kwargs):
    # L = (500 / 0.5)^(1/3) = 10, L/2 = 5; dynamic rmax ~ 5.66, t_tol_lj=0.02 rmax ~ 10.1
    with pytest.raises(ValueError, match="minimum-image"):
        resolve_table_bounds("modified_lj", 0.5, 15.0, 4.0, 3.5, None, N=500, density=0.5,
                             **kwargs)


def test_rmin_k_scales_modified_lj_and_not_rmax():
    base = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO)
    moved = resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                                 rmin_k=0.65)
    assert base["rmin"] == pytest.approx(simulation.DEFAULT_RMIN_K * 2.5)
    assert moved["rmin"] == pytest.approx(0.65 * 2.5)
    assert moved["rmax"] == base["rmax"]
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError, match="rmin_k"):
            resolve_table_bounds("modified_lj", 3.0, 12.0, 6.0, 2.5, None, N=N, density=RHO,
                                 rmin_k=bad)


def test_shifted_mie_rmin_ignores_rmin_k():
    args = ("shifted_mie", 3.0, 12.0, 6.0, 1.0, 0.5)
    default = resolve_table_bounds(*args, N=N, density=RHO)
    other = resolve_table_bounds(*args, N=N, density=RHO, rmin_k=0.5)
    assert default["rmin"] == pytest.approx(1.0 + 0.7 * 0.5)
    assert other["rmin"] == default["rmin"]
    assert other["rmax"] == default["rmax"]


def test_shifted_mie_unchanged_and_rejects_tail_energy_cut():
    args = ("shifted_mie", 3.0, 12.0, 6.0, 1.0, 0.5)
    b = resolve_table_bounds(*args, N=N, density=RHO)
    assert (b["rmin"], b["rmax"]) == _compute_table_bounds(*args)
    assert b["t_tol_lj"] is None and b["tail_energy_cut"] is None
    with pytest.raises(ValueError, match="only defined"):
        resolve_table_bounds(*args, N=N, density=RHO, tail_energy_cut=0.1)


def test_run_simulation_rejects_conflicts_before_hoomd(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        simulation.run_simulation(
            RHO, 3.0, 2.5, 12.0, 6.0, str(tmp_path),
            tail_energy_cut=0.1, rmax=6.0,
        )
