"""
shift_rmse resampling at the ends of the overlap (fixed 2026-10-09).

Before the fix each curve was interpolated only from its in-window samples,
so the curve whose first sample fell just below the overlap's lower end was
held flat over its first q bin. Found on the R3.2 truth evaluation: a q
perturbation of one ulp moved the loss of a curve against itself by up to
0.04.
"""
import numpy as np
import pytest

from metrics import _resample_reduced, shift_rmse_loss
from shift_rmse_curves import crystal_curve, fluid_curve


def _scaled(curve, s):
    out = curve.copy()
    out[:, 0] = curve[:, 0] * s
    return out


@pytest.mark.parametrize("curve", [crystal_curve(), fluid_curve()], ids=["crystal", "fluid"])
@pytest.mark.parametrize("s", [1 - 1e-15, 1 + 1e-15])
def test_ulp_perturbation_does_not_change_the_loss(curve, s):
    loss, diag, *_ = shift_rmse_loss(curve, _scaled(curve, s), None)
    assert loss == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("curve", [crystal_curve(), fluid_curve()], ids=["crystal", "fluid"])
def test_pure_q_scaling_costs_only_the_spacing_term(curve):
    """A curve rescaled in q has the same shape on x = q/q1: M4 stays ~0."""
    for s in (0.98, 0.995, 1.005, 1.02):
        loss, diag, *_ = shift_rmse_loss(curve, _scaled(curve, s), None)
        assert diag["aligned"]
        assert diag["m4"] < 2e-3, (s, diag["m4"])
        assert diag["shift_term"] == pytest.approx(abs(np.log(s)), rel=0.05)


def test_value_at_lower_end_is_interpolated():
    """The target's first sample lies below lo: its value at lo is the linear interpolant."""
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    exp = np.column_stack([q, np.array([10.0, 4.0, 3.0, 2.0, 1.5, 1.0])])
    sim = np.column_stack([q[1:] - 0.5, np.ones(5)])        # starts at 1.5
    log_exp, log_sim, grid = _resample_reduced(exp, sim, (0.0, 10.0), 1.0, 1.0, 64)
    assert grid[0] == pytest.approx(1.5)
    assert 10 ** log_exp[0] == pytest.approx(7.0)            # halfway between 10 and 4 (was held at 4)
