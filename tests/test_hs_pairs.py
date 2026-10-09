"""r < rmin pair count, and the repulsive branch cut at the well minimum."""
import numpy as np
import pytest

from simulation import (
    _repulsive_branch,
    count_close_pairs,
    mie_r0,
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


def test_repulsive_branch_matches_the_full_force_below_r0():
    r0, U0, n, m = 2.5, 3.0, 12.0, 6.0
    rmin = 0.5 * r0
    r = np.linspace(rmin, r0, 400, endpoint=False)
    U, F = mie_r0(r, rmin, r0, U0, n, m, r0)
    Ur, Fr = _repulsive_branch(r, rmin, r0, U0, n, m, r0)
    assert np.allclose(Ur, U + U0)
    assert np.allclose(Fr, F)
    # HOOMD 2.9.7 calls the table function with a Python float and appends
    # the returned scalars into a C++ vector.
    u_cut, f_cut = _repulsive_branch(float(r0), rmin, r0, U0, n, m, r0)
    assert type(u_cut) is float and type(f_cut) is float
    assert u_cut == 0.0 and f_cut == 0.0
    u_out, f_out = _repulsive_branch(r0 + 0.2, rmin, r0, U0, n, m, r0)
    assert u_out == 0.0 and f_out == 0.0

