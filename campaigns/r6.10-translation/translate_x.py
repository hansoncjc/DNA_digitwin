"""R6.10 x-space rerun: minimize the saved R6 GP mean directly over conditions.

x = (L_bridge, C_anchor, C_NaCl) is mapped forward, f(x) = (r0, U0, n, m),
and scored by the latent posterior mean mu(f(x)) of each final R6 GP, only
where f(x) lies in the original R6 box (no GP extrapolation). The mapping is
never inverted: another mapping only replaces `trained_mapping`. Inputs and
their validation are those of translate.py; no simulations or GP refits.

Algorithm, per run:
1. mu, sd on an X_GRID grid over X_BOX; refuse if the feasible set reaches
   the upper L_bridge or C_anchor face (the box would then constrain x).
2. Global 3D minimum: grid + SLSQP from separated starts, with f(x) in the
   R6 box as inequality constraints.
3. Salt profile: the same minimization over (L_bridge, C_anchor) at every
   grid salt, each checked by a local 41x41 finer grid.
4. Representative point: among the 3D optimum and the slice optima within
   FLAT_TOL of the overall minimum, the one closest to CHECK_SALT. Salt is
   degenerate when every slice minimum is within FLAT_TOL of it.
5. Compare with the first translation (inputs/check_candidates.csv) and the
   GP mean at the observed BO best (translation cost, GP against GP).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('r610_translate', HERE / 'translate.py')
t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t)
cfg = t.cfg
from datasets import physics_map

INSIDE_TOL = 1e-9  # unit-cube tolerance for f(x) in the R6 box


def trained_mapping(c):
    def f(L_bridge, C_anchor, C_NaCl):
        return physics_map(L_bridge, C_anchor, C_NaCl, **c)
    return f


def theta(x, mapping):
    p = dict(mapping(*map(float, x)))
    p['m_rel'] = (p['m'] - 4.) / (p['n'] - 5.)
    return p


def unit_theta(points, mapping, names):
    """R6 unit-cube coordinates of f(x), unclipped, and whether f(x) is in the R6 box."""
    unit = np.array([[(theta(x, mapping)[k] - cfg.BOUNDS[k][0]) / np.ptp(cfg.BOUNDS[k]) for k in names]
                     for x in np.atleast_2d(points)])
    inside = np.all((unit >= -INSIDE_TOL) & (unit <= 1 + INSIDE_TOL), axis=1)
    return unit, inside


class Landscape:
    """mu(f(x)) and sd(f(x)) for one saved GP and one mapping."""

    def __init__(self, gp, mapping):
        self.gp, self.mapping = gp, mapping

    def evaluate(self, points):
        """Mean, SD and in-box flag at x; NaN where f(x) is outside the R6 box."""
        unit, inside = unit_theta(points, self.mapping, self.gp.names)
        mu, sd = np.full(len(unit), np.nan), np.full(len(unit), np.nan)
        if inside.any():
            mu[inside], sd[inside] = self.gp.predict_unit(unit[inside])
        return mu, sd, inside

    def mean(self, x):
        # Optimizer objective. Line searches may step just outside the box;
        # the constraints keep accepted solutions inside.
        return float(self.gp.predict_unit(unit_theta([x], self.mapping, self.gp.names)[0])[0][0])

    def margin(self, x):
        unit = unit_theta([x], self.mapping, self.gp.names)[0][0]
        return np.concatenate((unit, 1 - unit))


def box_arrays():
    lo = np.array([cfg.X_BOX[k][0] for k in cfg.X_NAMES])
    return lo, np.array([np.ptp(cfg.X_BOX[k]) for k in cfg.X_NAMES])


def grid3d(land):
    axes = [np.linspace(*cfg.X_BOX[k], n) for k, n in zip(cfg.X_NAMES, cfg.X_GRID)]
    if not np.any(np.isclose(axes[2], cfg.CHECK_SALT)):
        raise ValueError('The salt grid must contain CHECK_SALT')
    mesh = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack([m.ravel() for m in mesh])
    mu, sd, inside = land.evaluate(points)
    shape = tuple(cfg.X_GRID)
    return axes, points, mu.reshape(shape), sd.reshape(shape), inside.reshape(shape)


def check_enclosed(inside):
    """The upper L_bridge and C_anchor faces are grid limits, not conditions; nothing feasible may touch them."""
    for axis, name in ((0, 'L_bridge'), (1, 'C_anchor')):
        if np.take(inside, -1, axis=axis).any():
            raise RuntimeError(f'X_BOX upper {name} face reaches the R6 box preimage; enlarge X_BOX')


def separated(points, values, k, sep=.12):
    """Up to k lowest points, at least sep apart in units of the points' extent."""
    scale = np.ptp(points, axis=0)
    scale[scale == 0] = 1.
    starts = []
    for i in np.argsort(values):
        if all(np.linalg.norm((points[i] - s) / scale) > sep for s in starts):
            starts.append(points[i])
        if len(starts) == k:
            break
    return starts


def minimize_region(land, x_fixed, free, points, values):
    """Minimize mu(f(x)) over x[free], others held at x_fixed, with f(x) in the R6 box.

    points, values: feasible grid points (full x) and their means, for starts.
    """
    if not len(points):
        raise RuntimeError('No feasible grid point in this region')
    lo, width = box_arrays()
    x_fixed = np.asarray(x_fixed, dtype=float)

    def full(z):
        x = x_fixed.copy()
        x[free] = lo[free] + np.asarray(z) * width[free]
        return x

    i = int(np.argmin(values))
    candidates = [(float(values[i]), points[i], 'grid')]
    records = []
    for start in separated(points[:, free], values, cfg.MULTISTART):
        res = minimize(lambda z: land.mean(full(z)), (start - lo[free]) / width[free], method='SLSQP',
                       bounds=[(0., 1.)] * len(free),
                       constraints=[{'type': 'ineq', 'fun': lambda z: land.margin(full(z))}],
                       options={'ftol': 1e-12, 'maxiter': 500})
        x = full(res.x)
        ok = bool(res.success and land.evaluate([x])[2][0] and np.isfinite(res.fun))
        records.append({'success': bool(res.success), 'inside': ok, 'message': str(res.message),
                        'x': x.tolist(), 'loss': float(res.fun)})
        if ok:
            candidates.append((land.mean(x), x, 'slsqp'))
    if not any(r['inside'] for r in records):
        raise RuntimeError('All local refinements failed; refusing a candidate')
    value, best, method = min(candidates, key=lambda v: v[0])
    return np.asarray(best, dtype=float), float(value), {
        'refinements': records, 'selected_method': method, 'selected_mean': float(value)}


def fine_check(land, best, value, free, spacing):
    """41-point-per-axis grid of +-2 coarse spacings around best; stop if it is lower by >1e-6."""
    lo, width = box_arrays()
    axes = [np.linspace(max(lo[k], best[k] - 2 * spacing[k]), min(lo[k] + width[k], best[k] + 2 * spacing[k]), 41)
            for k in free]
    mesh = np.meshgrid(*axes, indexing='ij')
    points = np.repeat(best[None, :], mesh[0].size, axis=0)
    for k, m in zip(free, mesh):
        points[:, k] = m.ravel()
    mu = land.evaluate(points)[0]
    fine_min = float(np.nanmin(mu))
    if fine_min < value - 1e-6:
        raise RuntimeError('Local fine grid improves the optimum by >1e-6; review optimizer convergence')
    return fine_min


def slice_data(land, center, free, ranges, n):
    """Mean, SD and in-box flag on an n x n grid over x[free], other coordinates at center."""
    axes = [np.linspace(*r, n) for r in ranges]
    mesh = np.meshgrid(*axes, indexing='ij')
    points = np.repeat(np.asarray(center, dtype=float)[None, :], mesh[0].size, axis=0)
    for k, m in zip(free, mesh):
        points[:, k] = m.ravel()
    mu, sd, inside = land.evaluate(points)
    return {'axes': axes, 'mean_loss': mu.reshape(mesh[0].shape), 'std_loss': sd.reshape(mesh[0].shape),
            'inside': inside.reshape(mesh[0].shape)}


def feasible_range(axis_values, inside_any, spacing, k):
    """Extent of feasible grid values along one axis, padded by one spacing, within X_BOX."""
    v = axis_values[inside_any]
    lo, hi = cfg.X_BOX[cfg.X_NAMES[k]]
    return max(lo, v.min() - spacing), min(hi, v.max() + spacing)


def translate_run(gp, mapping, first, c):
    """Everything for one run.

    first: the first translation's CHECK_SALT condition (L_bridge, C_anchor, C_NaCl).
    c: eval-56 coefficients, used only to draw the first translation's analytical family.
    """
    land = Landscape(gp, mapping)
    axes, points, mu, sd, inside = grid3d(land)
    check_enclosed(inside)
    spacing = np.array([a[1] - a[0] for a in axes])
    flat = inside.ravel()
    best3, value3, diag3 = minimize_region(land, np.zeros(3), [0, 1, 2], points[flat], mu.ravel()[flat])
    profile, slices = [], {}
    for j, salt in enumerate(axes[2]):
        sel = inside[:, :, j].ravel()
        pts = points.reshape(*cfg.X_GRID, 3)[:, :, j].reshape(-1, 3)
        best, value, diag = minimize_region(land, [0., 0., salt], [0, 1], pts[sel], mu[:, :, j].ravel()[sel])
        diag['fine_grid_minimum'] = fine_check(land, best, value, [0, 1], spacing)
        slices[f'{salt:g}'] = diag
        profile.append({'C_NaCl': float(salt), 'x': best, 'mean': value})
    slice_means = [r['mean'] for r in profile]
    overall = min([value3] + slice_means)
    eligible = [e for e in [(best3, value3, '3d')] + [(r['x'], r['mean'], 'slice') for r in profile]
                if e[1] <= overall + cfg.FLAT_TOL]
    rep, _, rep_source = min(eligible, key=lambda e: (abs(e[0][2] - cfg.CHECK_SALT), e[1]))
    # Degenerate: every salt reaches the same minimum and the 3D search finds nothing lower between grid salts.
    slice_range = max(slice_means) - min(slice_means)
    degenerate = bool(slice_range <= cfg.FLAT_TOL and value3 >= min(slice_means) - cfg.FLAT_TOL)

    first = np.asarray(first, dtype=float)
    lo, width = box_arrays()
    mu_rep, sd_rep, _ = land.evaluate([rep])
    mu_first, sd_first, in_first = land.evaluate([first])
    if not in_first[0]:
        raise ValueError('First-translation condition maps outside the R6 box')
    dx = float(np.linalg.norm((rep - first) / width))
    dmu = float(mu_rep[0] - mu_first[0])
    p_rep, p_first = theta(rep, mapping), theta(first, mapping)
    bo_best = min(gp.trajectory, key=lambda r: float(r['loss']))
    unit_best = [[(float(bo_best[k]) - cfg.BOUNDS[k][0]) / np.ptp(cfg.BOUNDS[k]) for k in gp.names]]
    mu_best, sd_best = gp.predict_unit(unit_best)
    x_rep = dict(zip(cfg.X_NAMES, map(float, rep)))
    row = {**x_rep, **{k: p_rep[k] for k in ('r0', 'U0', 'n', 'm', 'm_rel')},
           'mean_loss': float(mu_rep[0]), 'std_loss': float(sd_rep[0]), 'representative_from': rep_source,
           'min3d_mean_loss': value3, **{'min3d_' + k: float(v) for k, v in zip(cfg.X_NAMES, best3)},
           'slice_min_range': float(slice_range), 'salt_degenerate': degenerate,
           **{'first_' + k: float(v) for k, v in zip(cfg.X_NAMES, first)},
           'first_mean_loss': float(mu_first[0]), 'delta_mean_vs_first': dmu,
           'x_distance_to_first': dx,
           'same_as_first': bool(abs(dmu) <= cfg.SAME_MU_TOL and dx <= cfg.SAME_X_TOL),
           'best_eval': int(bo_best['iteration']), 'best_observed_loss': float(bo_best['loss']),
           'mean_at_bo_best': float(mu_best[0]), 'std_at_bo_best': float(sd_best[0]),
           'translation_cost': float(mu_rep[0] - mu_best[0]), **t.range_flags(x_rep)}
    # The first translation's analytical family, for comparison only.
    profile_rows = []
    for r in profile:
        p = theta(r['x'], mapping)
        fam = t.condition(p_first['r0'], p_first['U0'], r['C_NaCl'], c)
        profile_rows.append({'C_NaCl': r['C_NaCl'], 'L_bridge': float(r['x'][0]), 'C_anchor': float(r['x'][1]),
                             'mean_loss': r['mean'], 'std_loss': float(land.evaluate([r['x']])[1][0]),
                             **{k: p[k] for k in ('r0', 'U0', 'n', 'm', 'm_rel')},
                             'first_family_L_bridge': fam['L_bridge'], 'first_family_C_anchor': fam['C_anchor']})
    row['max_abs_dL_vs_first_family'] = max(abs(r['L_bridge'] - r['first_family_L_bridge']) for r in profile_rows)
    row['max_rel_dC_vs_first_family'] = max(abs(r['C_anchor'] / r['first_family_C_anchor'] - 1) for r in profile_rows)

    # Figure data: the CHECK_SALT slice over (L_bridge, C_anchor), and the
    # fixed-L_bridge slice through the representative point over (C_anchor, salt).
    j = int(np.argmin(abs(axes[2] - cfg.CHECK_SALT)))
    salt_slice = slice_data(land, [0., 0., cfg.CHECK_SALT], [0, 1],
                            [feasible_range(axes[0], inside[:, :, j].any(1), spacing[0], 0),
                             feasible_range(axes[1], inside[:, :, j].any(0), spacing[1], 1)], cfg.SLICE_GRID)
    i = int(np.argmin(abs(axes[0] - rep[0])))
    bridge_slice = slice_data(land, rep, [1, 2],
                              [feasible_range(axes[1], inside[i].any(1), spacing[1], 1), cfg.X_BOX['C_NaCl']],
                              cfg.SLICE_GRID)
    figure_data = {'grid3d': {'axes': axes, 'mean_loss': mu, 'std_loss': sd, 'inside': inside},
                   'slice_salt': salt_slice, 'slice_bridge': bridge_slice}
    diagnostics = {'min3d': diag3, 'slices': slices}
    return row, profile_rows, figure_data, diagnostics


def save_npz(path, data, extra):
    arrays = {f'axis_{k}': v for k, v in zip(data.get('names', ()), data['axes'])}
    np.savez_compressed(path, **arrays, mean_loss=data['mean_loss'], std_loss=data['std_loss'],
                        inside=data['inside'], **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--out', type=Path, default=cfg.OUT_X)
    a = ap.parse_args()
    provenance, c, gps = t.validated_inputs()  # Validate BOTH before any optimization.
    first_path = cfg.INPUTS / 'check_candidates.csv'
    first = {r['run']: [float(r[k]) for k in cfg.X_NAMES] for r in t.read_csv(first_path) if r['mapping'] == 'trained'}
    report = {'coefficients': c, 'validation': [gp.validation for _, gp in gps],
              'input_provenance': provenance, 'versions': t.versions(),
              'first_translation': {'path': 'inputs/check_candidates.csv', 'conditions': first,
                                    'sha256': hashlib.sha256(first_path.read_bytes()).hexdigest()},
              'objective': 'latent posterior mean loss of f(x), f(x) inside the R6 box',
              'settings': {k: getattr(cfg, k) for k in ('X_NAMES', 'X_BOX', 'X_GRID', 'SLICE_GRID', 'FLAT_TOL',
                                                         'SAME_MU_TOL', 'SAME_X_TOL', 'MULTISTART', 'CHECK_SALT')},
              'optimizations': {}}
    if a.validate_only:
        print(json.dumps(report, indent=2))
        return
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}; choose a fresh --out')
    a.out.mkdir(parents=True)
    (a.out / 'figure_data').mkdir()
    rows, proposed = [], []
    for run, gp in gps:
        row, profile, figs, diagnostics = translate_run(gp, trained_mapping(c), first[run], c)
        rows.append({'run': run, **row})
        t.write_csv(a.out / f'salt_profile_{run}.csv', profile)
        rep = {k: row[k] for k in cfg.X_NAMES}
        for kind in ('trained', 'ground_truth'):
            proposed.append({'run': run, 'mapping': kind, **rep, 'seed': cfg.SEED})
        meta = {'names': list(cfg.X_NAMES)}
        mark = {'x_representative': [rep[k] for k in cfg.X_NAMES], 'x_first': first[run]}
        save_npz(a.out / 'figure_data' / f'{run}_grid3d.npz', {**figs['grid3d'], **meta}, mark)
        save_npz(a.out / 'figure_data' / f'{run}_slice_salt{cfg.CHECK_SALT:g}mM.npz',
                 {**figs['slice_salt'], 'names': ['L_bridge', 'C_anchor']}, {**mark, 'C_NaCl': cfg.CHECK_SALT})
        save_npz(a.out / 'figure_data' / f'{run}_slice_bridge.npz',
                 {**figs['slice_bridge'], 'names': ['C_anchor', 'C_NaCl']}, {**mark, 'L_bridge': rep['L_bridge']})
        report['optimizations'][run] = diagnostics
    t.write_csv(a.out / 'translation_x_results.csv', rows)
    t.write_csv(a.out / 'check_candidates.proposed.csv', proposed)
    report['same_as_first'] = {r['run']: r['same_as_first'] for r in rows}
    (a.out / 'translation_x_report.json').write_text(json.dumps(report, indent=2, allow_nan=False, default=float) + '\n')
    print(f'x-space translation complete: {a.out}')
    for r in rows:
        print(f"{r['run']}: same_as_first={r['same_as_first']} salt_degenerate={r['salt_degenerate']} "
              f"dmu={r['delta_mean_vs_first']:.3g} dx={r['x_distance_to_first']:.3g}")


if __name__ == '__main__':
    main()
