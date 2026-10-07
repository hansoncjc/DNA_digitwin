"""Post-HS r < rmin pair count and the sigma-generalized Heyes-Melrose potential."""
import numpy as np
import pytest

from simulation import _hs_potential, count_close_pairs


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


def test_hs_potential_sigma_one_is_unchanged():
    r = np.linspace(0.0, 1.0, 1000)
    dt = 1e-4
    U, F = _hs_potential(r, 0.0, 1.0, dt)
    assert np.array_equal(U, 1.0 / (4.0 * dt) * (1.0 - r) ** 2)
    assert np.array_equal(F, 1.0 / (2.0 * dt) * (1.0 - r))


@pytest.mark.parametrize("sigma", [1.0, 1.75, 2.45])
def test_hs_potential_removes_overlap_in_one_step(sigma):
    dt = 1e-4
    r = np.linspace(0.0, sigma, 50)
    U, F = _hs_potential(r, 0.0, sigma, dt, sigma=sigma)
    assert U[-1] == 0.0 and F[-1] == 0.0
    # each particle moves F*dt (gamma = 1); the pair separates by 2*F*dt
    assert np.allclose(2.0 * F * dt, sigma - r)
    assert np.allclose(-np.gradient(U, r)[1:-1], F[1:-1])
