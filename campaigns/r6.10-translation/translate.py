"""CPU translation on the trained mapping manifold. No simulations or GP refits.

Run --validate-only first. A failed load or posterior check is fatal; there is
no fallback kernel, new GP fit, or relaxed-domain optimization.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
_spec = importlib.util.spec_from_file_location('r610_config', HERE / 'config.py')
cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfg)
from datasets import (physics_map, check_mapped_params, PHYSICS_COEFFS,
                      FIXED_LINKER_NM, DSDNA_RISE_NM, C_ANCHOR_REF)


def read_csv(path):
    with Path(path).open() as f:
        return [r for r in csv.DictReader(line for line in f if not line.startswith('#'))
                if r.get('iteration') != 'iteration']


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def coefficients():
    rows = read_csv(cfg.INPUTS / 'r3.2_eval_056.csv')
    if len(rows) != 24 or {r['iteration'] for r in rows} != {'56'}:
        raise ValueError('Expected all 24 dataset rows of R3.2 eval 56')
    c = {k: float(rows[0][k]) for k in PHYSICS_COEFFS}
    if any(float(r[k]) != c[k] for r in rows for k in c):
        raise ValueError('Inconsistent eval-56 coefficients')
    return c


def condition(r0, U0, salt, c):
    """Continuous inverse; no R3.2 range clipping or bridge rounding."""
    L = (cfg.D_SI * (r0 - 1) / c['k'] - FIXED_LINKER_NM) / DSDNA_RISE_NM
    anchor = C_ANCHOR_REF * U0 / (c['A'] * (1 + c['K_s'] * salt))
    if min(L, anchor, salt) < -1e-10 or not np.isfinite([L, anchor, salt]).all():
        raise ValueError('No finite nonnegative condition at this point')
    return {'L_bridge': max(0., L), 'C_anchor': max(0., anchor), 'C_NaCl': salt}


def mapped(r0, U0, c):
    p = physics_map(**condition(r0, U0, 0., c), **c)
    p['m_rel'] = (p['m'] - 4.) / (p['n'] - 5.)
    return p


def feasible_r0(c):
    """Intersect all R6 constraints analytically in g=r0/(r0-1).

n bounds constrain g; 0 <= m_rel <= 1 is equivalent to m >= 4
and n-m >= 1 because n >= 5.5. Nonnegative L also bounds r0.
    """
    lo, hi = cfg.BOUNDS['r0']
    lo = max(lo, 1 + c['k'] * FIXED_LINKER_NM / cfg.D_SI)
    g_lo = max(hi / (hi - 1), cfg.BOUNDS['n'][0] / (c['a_m'] + c['delta']),
               4 / c['a_m'], 1 / c['delta'])
    g_hi = min(lo / (lo - 1), cfg.BOUNDS['n'][1] / (c['a_m'] + c['delta']))
    if g_lo > g_hi or g_lo <= 1:
        raise ValueError('Empty supported mapping domain')
    return g_hi / (g_hi - 1), g_lo / (g_lo - 1)


def unit_points(points, c, names):
    out = []
    for r0, U0 in np.atleast_2d(points):
        p = mapped(r0, U0, c)
        check_mapped_params(p['U0'], p['n'], p['m'])
        row = [(p[k] - cfg.BOUNDS[k][0]) / (cfg.BOUNDS[k][1] - cfg.BOUNDS[k][0]) for k in names]
        if min(row) < -1e-8 or max(row) > 1 + 1e-8:
            raise ValueError('Refusing GP extrapolation outside the R6 box')
        out.append(np.clip(row, 0, 1))
    return np.asarray(out)


def range_flags(x):
    return {'outside_training_range': not (20 <= x['L_bridge'] <= 80 and
            60 <= x['C_anchor'] <= 140 and 0 <= x['C_NaCl'] <= 40),
            'outside_experimental_range': not (20 <= x['L_bridge'] <= 80 and
            20 <= x['C_anchor'] <= 140 and 0 <= x['C_NaCl'] <= 40)}


def validate_posterior(actual, record):
    for key in ('mean_loss', 'std_loss'):
        expected = record['best_post_' + key]
        if not np.isclose(actual[key], expected, atol=cfg.POSTERIOR_ATOL, rtol=cfg.POSTERIOR_RTOL):
            raise ValueError(f'GP posterior mismatch: {key}: {actual[key]} vs {expected}')


class SavedGP:
    def __init__(self, run):
        import torch
        import bo
        self.torch = torch
        root = cfg.INPUTS / run
        self.config = json.loads((root / 'inverse_config.json').read_text())
        for name, bounds in cfg.BOUNDS.items():
            if tuple(self.config['locked']['search'][name]['bounds']) != bounds:
                raise ValueError(f'{run}: unexpected original search bounds')
        if hashlib.md5((root / 'target_sq.npy').read_bytes()).hexdigest() != self.config['target']['md5']:
            raise ValueError(f'{run}: target hash mismatch')
        final = [r for r in bo.read_gp_log(root / 'gp_log') if r['stage'] == 'final']
        if len(final) != 1 or final[0]['n_train'] != 100:
            raise ValueError(f'{run}: expected one final GP with 100 observations')
        row = final[0]
        self.model, bundle = bo.load_gp_model(root / 'gp_log' / row['state_file'])
        self.names = list(bundle['param_names'])
        if self.names != row['param_names'] or set(self.names) != set(cfg.BOUNDS):
            raise ValueError('GP coordinate order mismatch')
        if bundle['stage'] != 'final' or len(bundle['train_x']) != 100:
            raise ValueError('Unexpected GP snapshot')
        self.trajectory = read_csv(root / 'bo_trajectory.csv')
        if len(self.trajectory) != 100 or len({r['iteration'] for r in self.trajectory}) != 100:
            raise ValueError('Expected 100 unique R6 evaluations')
        ordered = sorted(self.trajectory, key=lambda r: int(r['iteration']))
        train = np.array([[(float(r[n]) - cfg.BOUNDS[n][0]) / np.ptp(cfg.BOUNDS[n])
                           for n in self.names] for r in ordered])
        np.testing.assert_allclose(bundle['train_x'].cpu().numpy(), train, atol=1e-10)
        np.testing.assert_allclose(bundle['train_y'].cpu().numpy().reshape(-1),
                                   [-float(r['loss']) for r in ordered], atol=1e-10)
        post = bo.gp_posterior(self.model, torch.tensor([row['best_unit']], dtype=torch.float64))
        actual = {k: float(post[k].detach().reshape(-1)[0]) for k in ('mean_loss', 'std_loss')}
        validate_posterior(actual, row)
        self.validation = {'run': run, 'n_train': 100, 'posterior': actual,
                           'saved_botorch': bundle.get('botorch_version'),
                           'saved_gpytorch': bundle.get('gpytorch_version')}

    def predict(self, points, c):
        # Marginal variances only: no huge grid-by-grid covariance allocation.
        unit = unit_points(points, c, self.names)
        means, stds = [], []
        with self.torch.no_grad():
            for block in np.array_split(unit, max(1, (len(unit) + 255) // 256)):
                p = self.model.posterior(self.torch.tensor(block, dtype=self.torch.float64),
                                         observation_noise=False)
                means.extend((-p.mean).reshape(-1).cpu().numpy())
                stds.extend(p.variance.clamp_min(0).sqrt().reshape(-1).cpu().numpy())
        return np.asarray(means), np.asarray(stds)


def optimize(gp, c):
    """Grid + multistart bounded refinement + explicit edge minimizations."""
    rb = feasible_r0(c)
    ub = cfg.BOUNDS['U0']
    rr, uu = np.meshgrid(np.linspace(*rb, cfg.GRID_SIZE), np.linspace(*ub, cfg.GRID_SIZE))
    points = np.column_stack((rr.ravel(), uu.ravel()))
    mu, sd = gp.predict(points, c)
    objective = lambda v: float(gp.predict([v], c)[0][0])
    # Use spatially separated grid minima, not twelve adjacent pixels.
    order = np.argsort(mu)
    starts = []
    scale = np.array([np.ptp(rb), np.ptp(ub)])
    for i in order:
        if all(np.linalg.norm((points[i] - x) / scale) > .12 for x in starts):
            starts.append(points[i])
        if len(starts) == cfg.MULTISTART:
            break
    candidates = [(float(mu[order[0]]), points[order[0]], 'grid')]
    records = []
    for start in starts:
        res = minimize(objective, start, method='L-BFGS-B', bounds=[rb, ub],
                       options={'ftol': 1e-12, 'maxiter': 300})
        records.append({'success': bool(res.success), 'message': str(res.message),
                        'point': res.x.tolist(), 'loss': float(res.fun)})
        if res.success and np.isfinite(res.fun):
            candidates.append((float(res.fun), res.x, 'multistart'))
    for axis, bounds in enumerate((rb, ub)):
        for fixed in bounds:
            other = (ub, rb)[axis]
            def edge(v):
                return [fixed, v] if axis == 0 else [v, fixed]
            res = minimize_scalar(lambda v: objective(edge(v)), bounds=other, method='bounded')
            if res.success:
                candidates.append((float(res.fun), np.array(edge(res.x)), 'edge'))
            for v in other:
                candidates.append((objective(edge(v)), np.array(edge(v)), 'corner'))
    if not any(r['success'] for r in records):
        raise RuntimeError('All local refinements failed; refusing a final candidate')
    value, best, method = min(candidates, key=lambda v: v[0])
    # Independent finer grid near the selected minimum. If it reveals a
    # materially lower value, stop rather than silently claiming convergence.
    radius = scale / (cfg.GRID_SIZE - 1) * 2
    fine_r = np.linspace(max(rb[0], best[0] - radius[0]), min(rb[1], best[0] + radius[0]), 41)
    fine_u = np.linspace(max(ub[0], best[1] - radius[1]), min(ub[1], best[1] + radius[1]), 41)
    fr, fu = np.meshgrid(fine_r, fine_u)
    fine_mean, _ = gp.predict(np.column_stack((fr.ravel(), fu.ravel())), c)
    fine_min = float(np.min(fine_mean))
    if fine_min < value - 1e-6:
        raise RuntimeError('Local fine grid improves the optimum by >1e-6; review optimizer convergence')
    return best, (rr, uu, mu.reshape(rr.shape), sd.reshape(rr.shape)), {
        'refinements': records, 'selected_method': method,
        'fine_grid_minimum': fine_min, 'selected_mean': float(value)}


def plot_all(results, c, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    out.mkdir()
    def save(fig, name):
        for ext in ('pdf', 'png'):
            fig.savefig(out / (name + '.' + ext), dpi=300, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for col, (run, gp, row, grid) in enumerate(results):
        rr, uu, mu, sd = grid
        for ax, values, title in zip(axes[:, col], (mu, sd), ('Posterior mean loss', 'Latent posterior SD')):
            im = ax.pcolormesh(rr, uu, values, shading='auto', cmap='viridis')
            ax.plot(row['r0'], row['U0'], '*', color='#E66A00', ms=12)
            ax.set(xlabel=r'$r_0/\sigma$', ylabel=r'$U_0/k_BT$', title=f'{run}: {title}')
            fig.colorbar(im, ax=ax)
    save(fig, 'gp_mapping_surface')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, (run, gp, row, grid) in zip(axes, results):
        salt = np.linspace(0, cfg.DISPLAY_SALT_MAX, 401)
        anchor = [condition(row['r0'], row['U0'], s, c)['C_anchor'] for s in salt]
        ax.add_patch(Rectangle((0, 60), 40, 80, facecolor='0.9', label='R3.2 range (reference only)'))
        ax.plot(salt, anchor, color='#7B2CBF')
        for s in cfg.SALTS:
            y = condition(row['r0'], row['U0'], s, c)['C_anchor']
            ax.plot(s, y, 'o', color='#E66A00' if s == cfg.CHECK_SALT else '#7B2CBF')
        ax.set(xlabel='NaCl (mM)', ylabel='Anchor (molecules/particle)',
               title=f"{run}: L = {row['L_bridge']:.3f} bp", xlim=(0, cfg.DISPLAY_SALT_MAX), ylim=(0, None))
        ax.legend(fontsize=7)
    save(fig, 'equivalent_conditions')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    for ax, (run, gp, row, grid) in zip(axes, results):
        t = gp.trajectory
        im = ax.scatter([float(v['r0']) for v in t], [float(v['n']) for v in t],
                        c=[float(v['loss']) for v in t], cmap='viridis', s=18)
        rr = np.linspace(*feasible_r0(c), 300)
        ax.plot(rr, [mapped(r, 1., c)['n'] for r in rr], color='#E66A00', label='Trained mapping')
        best = min(t, key=lambda v: float(v['loss']))
        ax.plot(float(best['r0']), float(best['n']), 'x', color='black', label='Observed BO best')
        ax.plot(row['r0'], row['n'], '*', color='#7B2CBF', ms=12, label='Translation')
        ax.set(xlabel=r'$r_0/\sigma$', ylabel='n', title=run + ': projection of evaluated points')
        ax.legend(fontsize=7)
        fig.colorbar(im, ax=ax, label='Observed loss')
    save(fig, 'mapping_vs_r6')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true')
    ap.add_argument('--out', type=Path, default=cfg.OUT_ROOT / 'translation')
    a = ap.parse_args()
    provenance = json.loads((cfg.INPUTS / 'provenance.json').read_text())
    for name, info in provenance.items():
        if hashlib.sha256((cfg.INPUTS / name).read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Input changed: {name}')
    c = coefficients()
    gps = [(run, SavedGP(run)) for run in cfg.RUNS]  # Validate BOTH before any optimization.
    report = {'coefficients': c, 'validation': [gp.validation for _, gp in gps],
              'input_provenance': provenance, 'feasible_r0': feasible_r0(c),
              'objective': 'latent posterior mean loss', 'grid_size': cfg.GRID_SIZE,
              'multistart': cfg.MULTISTART, 'optimizations': {}, 'families': {}}
    import importlib.metadata
    report['versions'] = {p: importlib.metadata.version(p) for p in ('torch','botorch','gpytorch','numpy','scipy')}
    if a.validate_only:
        print(json.dumps(report, indent=2))
        return
    if a.out.exists():
        raise FileExistsError(f'Refusing to overwrite {a.out}; choose a fresh --out')
    a.out.mkdir(parents=True)
    rows, conditions, proposed, plots = [], [], [], []
    for run, gp in gps:
        best, grid, diagnostics = optimize(gp, c)
        r0, U0 = map(float, best)
        p = mapped(r0, U0, c)
        x = condition(r0, U0, cfg.CHECK_SALT, c)
        mu, sd = gp.predict([best], c)
        bo_best = min(gp.trajectory, key=lambda r: float(r['loss']))
        row = {'run': run, **p, 'L_bridge': x['L_bridge'], 'mean_loss': float(mu[0]),
               'std_loss': float(sd[0]), 'q1_target_times_r0': gp.config['target']['q1'] * r0,
               'best_eval': int(bo_best['iteration']), 'best_observed_loss': float(bo_best['loss']),
               'outside_R6_box': False, **range_flags(x)}
        diffs = {k: p[k] - float(bo_best[k]) for k in cfg.BOUNDS}
        row.update({'delta_' + k: v for k, v in diffs.items()})
        row['normalized_distance_to_best'] = float(np.linalg.norm([diffs[k] / np.ptp(cfg.BOUNDS[k]) for k in cfg.BOUNDS]))
        row['on_search_boundary'] = any(min(abs(p[k]-v) for v in cfg.BOUNDS[k]) < 1e-6 for k in cfg.BOUNDS)
        rows.append(row)
        for s in cfg.SALTS:
            xx = condition(r0, U0, s, c)
            conditions.append({'run': run, **xx, **range_flags(xx), 'selected_for_check': s == cfg.CHECK_SALT})
        for kind in ('trained', 'ground_truth'):
            proposed.append({'run': run, 'mapping': kind, **x, 'seed': cfg.SEED})
        report['optimizations'][run] = diagnostics
        report['families'][run] = {'L_bridge': x['L_bridge'], 'anchor_numerator': C_ANCHOR_REF * U0 / c['A'],
                                   'salt_coefficient': c['K_s'], 'formula': 'anchor_numerator / (1 + salt_coefficient * salt_mM)',
                                   'salt_domain': '[0, infinity)', 'display_only_salt_max': cfg.DISPLAY_SALT_MAX}
        plots.append((run, gp, row, grid))
    write_csv(a.out / 'translation_results.csv', rows)
    write_csv(a.out / 'equivalent_conditions.csv', conditions)
    write_csv(a.out / 'check_candidates.proposed.csv', proposed)
    plot_all(plots, c, a.out / 'figures')
    (a.out / 'translation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Translation complete: {a.out}; review and freeze check_candidates.proposed.csv in inputs/')


if __name__ == '__main__':
    main()
