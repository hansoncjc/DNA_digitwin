"""R6.10 inverse algebra and safeguards; no GPU or real GP load required."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t = load('r610_test_translate', ROOT / 'campaigns/r6.10-translation/translate.py')
gt = load('r610_test_gt', ROOT / 'campaigns/r3.1-gt/gt_config.py')


@pytest.mark.parametrize('r0', [2.1, 2.8, 3.5])
def test_equivalent_conditions_roundtrip(r0):
    c = t.coefficients()
    for salt in (0., 20., 40., 60.):
        x = t.condition(r0, 4., salt, c)
        p = t.physics_map(**x, **c)
        assert p['r0'] == pytest.approx(r0)
        assert p['U0'] == pytest.approx(4.)
        assert p['m'] / p['n'] == pytest.approx(c['a_m'] / (c['a_m'] + c['delta']))


@pytest.mark.parametrize('x', gt.TRAIN + gt.HELDOUT)
def test_ground_truth_matches_gt_config(x):
    c = gt.GROUND_TRUTH
    p = t.physics_map(x['L_bridge'], x['C_anchor'], x['C_NaCl'], **c)
    # Independent closed-form expectations using the construct in gt_config.
    r = 1 + c['k'] * (32.04 + .34 * x['L_bridge']) / 24.6
    expected = {'r0': r, 'U0': c['A'] * x['C_anchor'] / 100 * (1 + c['K_s'] * x['C_NaCl']),
                'n': (c['a_m'] + c['delta']) * r / (r - 1), 'm': c['a_m'] * r / (r - 1)}
    assert p == pytest.approx(expected)
    assert t.condition(r, p['U0'], x['C_NaCl'], c) == pytest.approx({k: x[k] for k in ('L_bridge','C_anchor','C_NaCl')})


def test_domain_and_no_training_clip():
    c = t.coefficients()
    lo, hi = t.feasible_r0(c)
    points = [[r, u] for r in np.linspace(lo, hi, 31) for u in (.5, 8)]
    unit = t.unit_points(points, c, ['m_rel', 'r0', 'U0', 'n'])
    assert np.all((unit >= 0) & (unit <= 1))
    assert t.condition(hi, 8., 60., c)['L_bridge'] > 80
    assert t.range_flags(t.condition(hi, 8., 60., c))['outside_training_range']
    with pytest.raises(ValueError, match='extrapolation'):
        t.unit_points([[hi, 8.1]], c, list(t.cfg.BOUNDS))


def test_gt_conditions_are_not_equivalent():
    c = t.coefficients()
    values = [t.physics_map(**t.condition(3., 4., s, c), **gt.GROUND_TRUTH)['U0'] for s in t.cfg.SALTS]
    assert np.ptp(values) > 0


def test_posterior_mismatch_is_fatal():
    record = {'best_post_mean_loss': .2, 'best_post_std_loss': .03}
    t.validate_posterior({'mean_loss': .2, 'std_loss': .03}, record)
    with pytest.raises(ValueError, match='posterior mismatch'):
        t.validate_posterior({'mean_loss': -.2, 'std_loss': .03}, record)


def test_optimizer_checks_boundary(monkeypatch):
    monkeypatch.setattr(t.cfg, 'GRID_SIZE', 15)
    monkeypatch.setattr(t.cfg, 'MULTISTART', 4)
    class SyntheticGP:
        def predict(self, points, c):
            p = np.asarray(points)
            return (p[:, 0] - 2.7)**2 - p[:, 1], np.zeros(len(p))
    best, _, _ = t.optimize(SyntheticGP(), t.coefficients())
    assert best == pytest.approx([2.7, 8.], abs=1e-5)


def test_negative_bridge_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        t.condition(1.01, 4., 20., t.coefficients())


def test_bundled_input_hashes():
    import json
    import hashlib
    manifest = json.loads((t.cfg.INPUTS / 'provenance.json').read_text())
    for name, entry in manifest.items():
        assert hashlib.sha256((t.cfg.INPUTS / name).read_bytes()).hexdigest() == entry['sha256']
