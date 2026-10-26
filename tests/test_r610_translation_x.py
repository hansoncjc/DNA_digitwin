"""R6.10 x-space rerun with a synthetic GP; no GPU or real GP load required."""
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


tx = load('r610x_test_translate_x', ROOT / 'campaigns/r6.10-translation/translate_x.py')
t, cfg = tx.t, tx.cfg
C = t.coefficients()
NAMES = ['m_rel', 'r0', 'U0', 'n']  # Saved-GP order differs from cfg.BOUNDS order on purpose.


class SyntheticGP:
    """Smooth mean over the R6 unit cube; depends on n so salt-dependent n breaks degeneracy."""
    names = NAMES

    def __init__(self, f):
        self.f = f
        self.trajectory = [{'iteration': '7', 'loss': '0.2', 'U0': '4', 'r0': '2.8', 'n': '10', 'm_rel': '.3'}]

    def predict_unit(self, unit):
        u = dict(zip(self.names, np.atleast_2d(unit).T))
        return self.f(u), np.full(len(u['r0']), .01)

    def predict(self, points, c):
        return self.predict_unit(t.unit_points(points, c, self.names))


def bowl(u):
    return (u['r0'] - .45)**2 + .5 * (u['U0'] - .6)**2 + .2 * (u['n'] - .3)**2


def edge(u):
    return (u['r0'] - .45)**2 - u['U0']


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(cfg, 'X_GRID', (41, 41, 6))  # Salts 0, 20, ..., 100.
    monkeypatch.setattr(cfg, 'GRID_SIZE', 41)
    monkeypatch.setattr(cfg, 'MULTISTART', 4)
    monkeypatch.setattr(cfg, 'SLICE_GRID', 21)


def first_translation(gp):
    """The first R6.10 route on the same GP: (r0, U0) optimum, inverted at CHECK_SALT."""
    best, _, diag = t.optimize(gp, C)
    x = t.condition(*best, cfg.CHECK_SALT, C)
    return [x[k] for k in cfg.X_NAMES], diag['selected_mean']


@pytest.mark.parametrize('f', [bowl, edge])
def test_x_space_matches_first_translation(small, f):
    """Same feasible set, same optimum: interior minimum and minimum on the U0 = 8 face."""
    gp = SyntheticGP(f)
    first, first_mean = first_translation(gp)
    row, profile, _, _ = tx.translate_run(gp, tx.trained_mapping(C), first, C)
    assert row['same_as_first']
    assert row['mean_loss'] == pytest.approx(first_mean, abs=1e-8)
    assert row['C_NaCl'] == cfg.CHECK_SALT
    assert row['salt_degenerate']
    assert row['slice_min_range'] <= cfg.FLAT_TOL
    # The numerical equivalence curve is the first translation's analytical family.
    assert row['max_abs_dL_vs_first_family'] < 1e-3
    assert row['max_rel_dC_vs_first_family'] < 1e-4
    assert [r['C_NaCl'] for r in profile] == pytest.approx(np.linspace(0, 100, 6))


def test_edge_minimum_is_on_box_face(small):
    gp = SyntheticGP(edge)
    row, _, _, _ = tx.translate_run(gp, tx.trained_mapping(C), first_translation(gp)[0], C)
    assert row['U0'] == pytest.approx(8., abs=1e-6)


def test_salt_dependent_mapping_is_not_degenerate(small):
    """n depends on salt beyond U0: salts reach different (r0, U0, n) sets and different minima."""
    def mapping(L, A, s):
        p = tx.physics_map(L, A, s, **C)
        return {**p, 'n': p['n'] * (1 + .004 * s)}
    gp = SyntheticGP(bowl)
    row, profile, _, _ = tx.translate_run(gp, mapping, first_translation(gp)[0], C)
    assert not row['salt_degenerate']
    assert row['slice_min_range'] > 1e-4
    assert row['mean_loss'] == pytest.approx(min(r['mean_loss'] for r in profile), abs=1e-6)


def test_translation_cost_is_gp_against_gp(small):
    gp = SyntheticGP(bowl)
    row, _, _, _ = tx.translate_run(gp, tx.trained_mapping(C), first_translation(gp)[0], C)
    u = [[(float(gp.trajectory[0][k]) - cfg.BOUNDS[k][0]) / np.ptp(cfg.BOUNDS[k]) for k in NAMES]]
    assert row['mean_at_bo_best'] == pytest.approx(gp.predict_unit(u)[0][0])
    assert row['translation_cost'] == pytest.approx(row['mean_loss'] - row['mean_at_bo_best'])


def test_outside_box_is_masked():
    land = tx.Landscape(SyntheticGP(bowl), tx.trained_mapping(C))
    mu, sd, inside = land.evaluate([[80., 160., 20.], [240., 160., 20.], [80., 5., 20.]])
    assert inside.tolist() == [True, False, False]
    assert np.isfinite(mu[0]) and np.isnan(mu[1:]).all() and np.isnan(sd[1:]).all()


def test_truncating_box_is_refused(small, monkeypatch):
    monkeypatch.setattr(cfg, 'X_BOX', {**cfg.X_BOX, 'L_bridge': (0., 120.)})
    gp = SyntheticGP(bowl)
    with pytest.raises(RuntimeError, match='enlarge X_BOX'):
        tx.translate_run(gp, tx.trained_mapping(C), [80., 160., 20.], C)


def test_default_box_encloses_preimage():
    """At the run settings the trained mapping's R6-box preimage stays off the upper L and C faces."""
    land = tx.Landscape(SyntheticGP(bowl), tx.trained_mapping(C))
    for k, name in ((0, 'L_bridge'), (1, 'C_anchor')):
        for salt in cfg.X_BOX['C_NaCl']:
            other = np.linspace(*cfg.X_BOX[cfg.X_NAMES[1 - k]], 401)
            pts = np.zeros((len(other), 3))
            pts[:, k], pts[:, 1 - k], pts[:, 2] = cfg.X_BOX[name][1], other, salt
            assert not land.evaluate(pts)[2].any()
