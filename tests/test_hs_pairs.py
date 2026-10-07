"""r < rmin pair count, and the repulsive branch cut at the well minimum."""
import numpy as np
import pytest

from simulation import (
    _repulsive_branch,
    _well_separation,
    count_close_pairs,
    modified_LJ,
    shifted_mie,
)


def _brute_force(x, L, r_cut):
    d = x[:, None, :] - x[None, :, :]
    d -= L * np.round(d / L)
    r = np.sqrt((d ** 2).sum(-1))
    iu = np.triu_indices(len(x), k=1)
    return int((r[iu] < r_cut).sum()), float(r[iu].min())


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("r_cut", [1.0, 1.75, 2.45])
def test_count_matches_minimum_image_brute_force(seed, r_cut):
    rng = np.random.default_rng(seed)
    L = 12.0
    x = rng.uniform(-L / 2, L / 2, size=(300, 3))
    n, dmin = count_close_pairs(x, L, r_cut)
    n_ref, dmin_ref = _brute_force(x, L, r_cut)
    assert n == n_ref
    assert dmin == pytest.approx(dmin_ref)


def test_count_uses_periodic_images():
    L = 10.0
    x = np.array([[-4.9, 0.0, 0.0], [4.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    n, dmin = count_close_pairs(x, L, 0.5)
    assert n == 1
    assert dmin == pytest.approx(0.2)


def test_count_is_strictly_below_cut():
    L = 10.0
    x = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    assert count_close_pairs(x, L, 1.5)[0] == 0
    assert count_close_pairs(x, L, 1.5 + 1e-9)[0] == 1


def test_modified_lj_branch_matches_the_full_force_below_r0():
    r0, U0, n, m = 2.5, 3.0, 12.0, 6.0
    assert _well_separation("modified_lj", n, m, r0, None) == r0
    rmin = 0.65 * r0
    r = np.linspace(rmin, r0, 400, endpoint=False)
    U, F = modified_LJ(r, rmin, r0, U0, n, m, r0)
    Ur, Fr = _repulsive_branch(r, rmin, r0, U0, n, m, r0, potential="modified_lj")
    assert np.allclose(Ur, U + U0)
    assert np.allclose(Fr, F)
    # HOOMD 2.9.7 calls the table function with a Python float and appends
    # the returned scalars into a C++ vector.
    u_cut, f_cut = _repulsive_branch(float(r0), rmin, r0, U0, n, m, r0)
    assert type(u_cut) is float and type(f_cut) is float
    assert u_cut == 0.0 and f_cut == 0.0
    u_out, f_out = _repulsive_branch(r0 + 0.2, rmin, r0, U0, n, m, r0)
    assert u_out == 0.0 and f_out == 0.0


@pytest.mark.parametrize("n,m", [(12.0, 6.0), (15.0, 14.0), (5.5, 4.0)])
def test_shifted_mie_branch_cuts_at_the_analytic_minimum(n, m):
    r0, delta, U0 = 1.0, 0.5, 3.0
    r_well = _well_separation("shifted_mie", n, m, r0, delta)
    assert r_well == pytest.approx(r0 + delta * (n / m) ** (1.0 / (n - m)))
    U_well, F_well = shifted_mie(r_well, r0 + 0.7 * delta, r_well, U0, n, m, r0, delta)
    assert float(U_well) == pytest.approx(-U0)
    assert float(F_well) == pytest.approx(0.0, abs=1e-8)
    rmin = r0 + 0.7 * delta
    assert rmin < r_well
    r = np.linspace(rmin, r_well, 300, endpoint=False)
    U, F = shifted_mie(r, rmin, r_well, U0, n, m, r0, delta)
    Ur, Fr = _repulsive_branch(
        r, rmin, r_well, U0, n, m, r0, delta=delta, potential="shifted_mie",
    )
    assert np.allclose(Ur, U + U0)
    assert np.allclose(Fr, F)
    u_cut, f_cut = _repulsive_branch(
        float(r_well), rmin, r_well, U0, n, m, r0, delta=delta, potential="shifted_mie",
    )
    assert u_cut == 0.0 and f_cut == 0.0
