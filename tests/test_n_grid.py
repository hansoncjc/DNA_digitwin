"""N_grid inference from a saxs-fft q grid and the BO-side target check."""
import numpy as np
import pytest

import bo
from scattering import DEFAULT_N_GRID, estimate_saxsfft_n_grid


def _saxsfft_q(n_grid, box_length, diameter_nm=24.6, trim=slice(3, -3)):
    """q bin centres as built by ``saxsfft.structurefactor.compute_s_1d``."""
    dq = 2.0 * np.pi / box_length
    q_corner = np.sqrt(3.0) * (n_grid // 2) * dq
    edges = np.arange(0.0, q_corner + dq, dq)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres[trim] / (diameter_nm * 10.0)


def _curve(q):
    return np.column_stack([q, np.ones_like(q)])


class _FakeDataset:
    def __init__(self, curve, ds_id="d0", datatype="sq"):
        self.id = ds_id
        self.datatype = datatype
        self._curve = curve

    def load_exp_curve(self, trim_tail=0):
        return self._curve


def test_default_n_grid_is_600():
    assert DEFAULT_N_GRID == 600


@pytest.mark.parametrize("n_grid", [300, 600])
@pytest.mark.parametrize("box_length", [60.0, 100.0, 171.0])
def test_estimate_is_box_independent(n_grid, box_length):
    est = estimate_saxsfft_n_grid(_curve(_saxsfft_q(n_grid, box_length)))
    assert est == pytest.approx(n_grid, rel=0.01)


def test_estimate_matches_saxsfft_binning():
    saxsfft = pytest.importorskip("saxsfft.structurefactor")
    rng = np.random.default_rng(0)
    box = np.array([40.0, 40.0, 40.0])
    x = rng.uniform(-20.0, 20.0, size=(500, 3))
    for n_grid in (32, 64):
        q, _, _ = saxsfft.compute_s_1d(x, box, n_grid, particle_diameter=24.6)
        est = estimate_saxsfft_n_grid(_curve(q))
        assert est == pytest.approx(n_grid, rel=0.03)


def test_estimate_returns_none_for_log_grid():
    q = np.logspace(-3, -1, 300)
    assert estimate_saxsfft_n_grid(_curve(q)) is None


def test_check_raises_on_mismatch():
    ds = _FakeDataset(_curve(_saxsfft_q(300, 100.0)))
    with pytest.raises(ValueError, match="N_grid"):
        bo._check_target_n_grid([ds], {"N_grid": 600}, trim_tail=0)
    with pytest.raises(ValueError, match="N_grid"):
        bo._check_target_n_grid([ds], None, trim_tail=0)


def test_check_passes_on_match_and_skips_non_saxsfft():
    ok = _FakeDataset(_curve(_saxsfft_q(600, 100.0)))
    bo._check_target_n_grid([ok], {}, trim_tail=0)
    bo._check_target_n_grid([ok], {"N_grid": 600}, trim_tail=0)
    measured = _FakeDataset(_curve(np.logspace(-3, -1, 300)), ds_id="d1")
    bo._check_target_n_grid([measured], {"N_grid": 600}, trim_tail=0)
    iq = _FakeDataset(_curve(_saxsfft_q(300, 100.0)), ds_id="d2", datatype="iq")
    bo._check_target_n_grid([iq], {"N_grid": 600}, trim_tail=0)
