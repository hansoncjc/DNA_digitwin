"""Four independent GPU checks from the reviewed, Git-tracked candidate CSV."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('r610_translate', HERE / 'translate.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
cfg = t.cfg
from datasets import physics_map, check_mapped_params


def gt_coefficients():
    spec = importlib.util.spec_from_file_location('r610_gt', cfg.REPO / 'campaigns/r3.1-gt/gt_config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GROUND_TRUTH


def prepare():
    path = cfg.INPUTS / 'check_candidates.csv'
    if not path.exists():
        raise FileNotFoundError('Review translation/check_candidates.proposed.csv, then copy it to inputs/check_candidates.csv and commit it before checks.')
    rows = t.read_csv(path)
    expected = {(r, k) for r in cfg.RUNS for k in ('trained', 'ground_truth')}
    if len(rows) != 4 or {(r['run'], r['mapping']) for r in rows} != expected:
        raise ValueError('Expected exactly one trained and one ground_truth check per run')
    from simulation import resolve_table_bounds
    jobs = []
    for r in rows:
        x = {k: float(r[k]) for k in ('L_bridge', 'C_anchor', 'C_NaCl')}
        if min(x.values()) < 0 or not np.isfinite(list(x.values())).all():
            raise ValueError('Conditions must be finite and nonnegative')
        if x['C_NaCl'] != cfg.CHECK_SALT or int(r['seed']) != cfg.SEED:
            raise ValueError('Expected salt 20 mM and the configured single seed')
        c = t.coefficients() if r['mapping'] == 'trained' else gt_coefficients()
        p = physics_map(**x, **c)
        check_mapped_params(p['U0'], p['n'], p['m'])
        if r['mapping'] == 'trained':
            t.unit_points([[p['r0'], p['U0']]], c, list(cfg.BOUNDS))
        run = json.loads((cfg.INPUTS / r['run'] / 'inverse_config.json').read_text())
        target = cfg.INPUTS / r['run'] / 'target_sq.npy'
        if hashlib.md5(target.read_bytes()).hexdigest() != run['target']['md5']:
            raise ValueError('Target hash mismatch')
        settings = run['locked']
        bounds = resolve_table_bounds(p['U0'], p['n'], p['m'], p['r0'],
                                      N=settings['N'], density=settings['density'])
        if not bounds['rmin'] < bounds['rmax'] < bounds['L'] / 2:
            raise ValueError('Invalid pair-table or box bounds')
        jobs.append({'run': r['run'], 'mapping': r['mapping'], 'condition': x,
                     'coefficients': c, 'mapped': p, 'seed': int(r['seed']),
                     'N': settings['N'], 'density': settings['density'],
                     'metric_kwargs': settings.get('metric_kwargs', {}),
                     'target': str(target), 'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for run in cfg.RUNS:
        pair = [j for j in jobs if j['run'] == run]
        if pair[0]['condition'] != pair[1]['condition']:
            raise ValueError('Trained and GT checks must use exactly the same experimental condition')
    return jobs


def execute(job, out, steps):
    from simulation import run_simulation
    from scattering import convert_to_SAXS_fft
    from metrics import shift_rmse_loss
    dest = out / (job['run'] + '_' + job['mapping'])
    dest.mkdir(parents=True, exist_ok=False)
    job['smoke_only'] = steps is not None
    job['steps_override'] = steps
    (dest / 'check_config.json').write_text(json.dumps(job, indent=2))
    p = job['mapped']
    kwargs = {'N': job['N'], 'seed': job['seed'], 'plot': False}
    if steps is not None:
        kwargs['steps'] = steps
    run_simulation(density=job['density'], U_0=p['U0'], r0=p['r0'], n=p['n'], m=p['m'],
                   outdir=str(dest), **kwargs)
    convert_to_SAXS_fft(str(dest))  # CPU S(q), as in the original R6 campaigns.
    sq = np.load(dest / 'S(q)_data/average_structure_factor.npy')
    target = np.load(job['target'])
    loss, diag, *_ = shift_rmse_loss(target, sq, None, metric_kwargs=job['metric_kwargs'])
    row = {'run': job['run'], 'mapping': job['mapping'], 'seed': job['seed'],
           'smoke_only': job['smoke_only'], **job['condition'], **p, 'loss': float(loss),
           'rmse': diag['m4'], 'shift': diag['shift_term'], 'q1_ratio': diag['q1_ratio'],
           'q1_sim_times_r0': None if diag['q1_sim'] is None else diag['q1_sim'] * p['r0']}
    (dest / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False))
    (dest / 'shift_rmse_diagnostics.json').write_text(json.dumps(diag, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.plot(target[:, 0], target[:, 1], color='#E66A00', alpha=.8, label='Target')
    ax.plot(sq[:, 0], sq[:, 1], color='#7B2CBF', alpha=.8, label='Check simulation')
    ax.set(xlabel=r'$q$ ($\mathring{A}^{-1}$)', ylabel='S(q)', title=dest.name)
    ax.legend()
    for ext in ('pdf', 'png'):
        fig.savefig(dest / ('sq_overlay.' + ext), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true')
    group.add_argument('--index', type=int, choices=range(4))
    group.add_argument('--collect', action='store_true')
    ap.add_argument('--out', type=Path, default=cfg.OUT_ROOT / 'checks')
    ap.add_argument('--steps', type=int, help='Smoke only; automatically uses checks_smoke by default')
    a = ap.parse_args()
    if a.steps is not None:
        if a.steps <= 0:
            ap.error('--steps must be positive')
        if a.out == cfg.OUT_ROOT / 'checks':
            a.out = cfg.OUT_ROOT / 'checks_smoke'
    jobs = prepare()  # Check all four before any simulation starts.
    if a.list:
        print(json.dumps(jobs, indent=2))
    elif a.collect:
        rows = [json.loads((a.out / (j['run'] + '_' + j['mapping']) / 'result.json').read_text()) for j in jobs]
        if any(r['smoke_only'] for r in rows) and a.out == cfg.OUT_ROOT / 'checks':
            raise ValueError('Smoke outputs cannot be collected as production')
        t.write_csv(a.out / 'check_results.csv', rows)
    else:
        execute(jobs[a.index], a.out, a.steps)


if __name__ == '__main__':
    main()
