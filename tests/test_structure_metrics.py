"""campaigns/inverse/analysis/structure_metrics.py: frame selection, g(r) normalization, first peak
(the PTM and rendering parts need OVITO and are checked on real runs)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "campaigns" / "inverse" / "analysis"))
import structure_metrics as sm  # noqa: E402


def test_select_frames_matches_forward_model():
    assert sm.select_frames(450, *sm.FORWARD_FRAMES) == list(range(350, 450, 5))
    assert sm.select_frames(16, "all", 1) == list(range(16))
    assert sm.select_frames(1, "last:1", 1) == [0]
    assert sm.select_frames(6, "last:6", 2) == [0, 2, 4]       # saxs-fft docstring example
    with pytest.raises(ValueError):
        sm.select_frames(10, "first:3")


def test_rdf_ideal_gas_is_one():
    rng = np.random.default_rng(0)
    L = 20.0
    frames = [(rng.random((4000, 3)) * L - L / 2, np.array([L, L, L])) for _ in range(3)]
    r, g = sm.rdf(frames, r_max=5.0, dr=0.1)
    assert np.mean(g[r > 1.0]) == pytest.approx(1.0, abs=0.02)


def test_rdf_box_density_normalization_of_a_cluster():
    """A cluster in a padded box: g is the local over the box density at contact scale."""
    rng = np.random.default_rng(1)
    L, n = 60.0, 3000
    pts = rng.random((4 * n, 3)) * 2 - 1
    pts = pts[np.linalg.norm(pts, axis=1) < 1][:n] * 8.0          # ball of radius 8
    r, g = sm.rdf([(pts, np.array([L, L, L]))], r_max=2.0, dr=0.05)
    local = n / (4 / 3 * np.pi * 8.0 ** 3)
    box = n / L ** 3
    # at r << R the pair density is ~ local density (surface correction ~ 3r/4R)
    sel = (r > 0.5) & (r < 1.0)
    expected = local / box * (1 - 3 * r[sel] / (4 * 8.0))
    assert np.mean(g[sel]) == pytest.approx(np.mean(expected), rel=0.05)


def test_first_peak_refined():
    r = np.arange(0.01, 5.0, 0.02)
    g = 1 + 3 * np.exp(-((r - 1.234) / 0.08) ** 2) + 1.0 * np.exp(-((r - 2.2) / 0.2) ** 2)
    r1, g1 = sm.first_peak(r, g)
    assert r1 == pytest.approx(1.234, abs=0.005)
    assert g1 == pytest.approx(4.0, rel=0.02)


def test_metric_kwargs_from_run(tmp_path):
    import json
    run = tmp_path / "r6.8a"
    (run / "Optimization_Results").mkdir(parents=True)
    snap = {"locked": {"metric_kwargs": {"lambda_shift": 0.0}}}
    (run / "Optimization_Results" / "inverse_config.json").write_text(json.dumps(snap))
    assert sm.metric_kwargs_from_run(run) == {"lambda_shift": 0.0}
    snap["locked"].pop("metric_kwargs")            # runs before 2026-10-14
    (run / "Optimization_Results" / "inverse_config.json").write_text(json.dumps(snap))
    assert sm.metric_kwargs_from_run(run) == {}


@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_lambda_reaches_the_recomputed_loss(lam):
    sm._stub_mcdfm()
    from shift_rmse_curves import crystal_curve
    tgt, sim = crystal_curve(q1=0.0129), crystal_curve(q1=0.0131, amp=6.0)
    base, _ = sm.sq_metrics(tgt, sim)                         # defaults: lambda 1
    diag, _ = sm.sq_metrics(tgt, sim, {"lambda_shift": lam})
    assert base["shift_term"] > 0
    assert diag["m4"] == pytest.approx(base["m4"])
    assert diag["shift_term"] == pytest.approx(lam * base["shift_term"])
    assert diag["loss"] == pytest.approx(diag["m4"] + lam * base["shift_term"])
