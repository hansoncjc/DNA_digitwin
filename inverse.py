"""
Inverse design: BO over simulation parameters (mode="sim") against one
target S(q). Everything a campaign sets is in a Python module that defines
``CONFIG`` (see campaigns/inverse/inverse-example/config.py and the README)
and runs as sbatch inverse_sequential.sbatch (parallel=False) or
inverse_parallel.sbatch (parallel=True).

    python inverse.py CONFIG.py              # start, or resume the run in out_root
    python inverse.py CONFIG.py --check      # startup checks and resume state; no simulation
    python inverse.py CONFIG.py --summary    # rewrite the summary from bo_trajectory.csv

Smoke tests only (recorded in out_root/inverse_config.json):

    --out DIR  --target FILE  --parallel {true,false}  --steps N  --device {gpu,cpu}

The same command starts and resumes a run (bo.run_bo_resumable): after a
walltime end, maintenance or preemption, resubmit it unchanged.

Startup checks, all before the first simulation:
  1. target file exists and is a finite (N, 2+) array; its md5 is recorded
     (inverse_config.json, summary) and identifies the target on a resume;
  2. the target's saxs-fft N_grid equals the simulation side's (bo, C6);
  3. first-peak detection with the metric_kwargs actually used ({} =
     defaults): a dispersed target is reported and the run continues (the
     metric's dispersed rule applies); a target that is not dispersed must
     have a find_peaks peak. The target compared with itself gives loss 0;
  4. compare_q_range is None;
  5. rmax < L/2 on a grid over the search box (6 points per free axis plus
     the initial point), with m = 4 + m_rel (n - 5) and n > m;
  6. the ParamSpace passes bo's mode="sim" rules; with m_rel, the lower
     bound of n is above 5 and m_rel lies in [0, 1];
  7. on a resume, the settings that define the run equal the ones in
     out_root/inverse_config.json (only max_evals and the parallel settings
     may change; batch_size is locked);
  8. batch_size > 1 needs parallel=True (BATCH_NEEDS_PARALLEL).
"""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FORMFACTOR = REPO / "formfactors" / "sasmodels_sphere_fit.txt"
DATASET_ID = "d0"
SNAPSHOT = "inverse_config.json"
SUMMARY = "inverse_summary"

# Selection by name. Each currently has one value; a new potential, metric
# or scattering method is added here (and in the module that implements it).
POTENTIALS = {"mie_r0": {"potential": "mie_r0"}}
METRICS = {"shift_rmse": {"metric": "shift_rmse", "metric_kwargs": {}}}
SCATTERING = {"saxsfft": {"scattering_method": "saxsfft", "scattering_kwargs": {}}}

REQUIRED = ("target", "N", "density", "search", "max_evals", "out_root", "parallel")
DEFAULTS = {
    "description": "",
    "gp_log_dir": None,          # None: <out_root>/gp_log
    "bo_seed": 0,
    "batch_size": 1,
    "potential": "mie_r0",
    "metric": "shift_rmse",
    "scattering": "saxsfft",
    "compare_q_range": None,     # must stay None (shift_rmse compares over the overlap)
    "mc_dfm_root": None,         # None: $MC_DFM_ROOT
    "parallel_cfg": None,
}
PARALLEL_REQUIRED = ("partition", "account", "mem", "time", "module_loads", "venv_activate")
PARALLEL_OPTIONAL = ("gpus", "poll_interval", "max_wait", "max_job_retries")
SIM_NAMES = ("U0", "r0", "n")    # must be in "search", plus one of m / m_rel

# Fields that define the run. A resume must keep them; max_evals, parallel,
# parallel_cfg, description, mc_dfm_root and --device may change. target_md5
# is the md5 recorded at the first start, so the target file may move but
# its content may not change. gp_log_dir is compared relative to out_root
# when it lies inside it, so the whole run folder may move.
LOCKED = ("target_md5", "N", "density", "search", "potential", "metric", "scattering",
          "compare_q_range", "bo_seed", "batch_size", "gp_log_dir", "steps")

RMAX_GRID_POINTS = 6

BATCH_NEEDS_PARALLEL = (
    "batch_size > 1 needs parallel=True. With parallel=False this process "
    "evaluates the candidates of a batch one after another on its one GPU: a "
    "batch of q takes q times as long as one evaluation, so the run takes as "
    "long as with batch_size=1, while the GP is refitted only once per q "
    "evaluations and each candidate is chosen with less data. Batches only "
    "pay off when their q evaluations run at the same time, one GPU job each."
)


class ConfigError(ValueError):
    """The configuration or a startup check failed; nothing was simulated."""


# ------------------------------------------------------------------ config


def load_config(path):
    """Import ``path`` and return a copy of its ``CONFIG`` dict."""
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    spec = importlib.util.spec_from_file_location("inverse_campaign_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(getattr(module, "CONFIG", None), dict):
        raise ConfigError(f"{path} must define a dict CONFIG")
    return copy.deepcopy(module.CONFIG)


def normalize_config(raw, overrides=None):
    """
    Check keys, apply defaults and command-line overrides, resolve paths.

    ``overrides`` (smoke tests): out_root, target, parallel, steps, device.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    unknown = set(raw) - set(REQUIRED) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"unknown CONFIG keys: {sorted(unknown)}")
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise ConfigError(f"CONFIG is missing: {missing}")
    cfg = {**copy.deepcopy(DEFAULTS), **raw}
    for key in ("out_root", "target", "parallel"):
        if key in overrides:
            cfg[key] = overrides[key]
    cfg["steps"] = overrides.get("steps")
    cfg["device"] = overrides.get("device")

    for name, table in (("potential", POTENTIALS), ("metric", METRICS),
                        ("scattering", SCATTERING)):
        if cfg[name] not in table:
            raise ConfigError(f"{name}={cfg[name]!r}; available: {sorted(table)}")
    if cfg["compare_q_range"] is not None:
        raise ConfigError(
            f"compare_q_range must be None (shift_rmse compares over the curve overlap); "
            f"got {cfg['compare_q_range']!r}"
        )
    if not isinstance(cfg["parallel"], bool):
        raise ConfigError(f"parallel must be True or False; got {cfg['parallel']!r}")
    cfg["batch_size"] = int(cfg["batch_size"])
    if cfg["batch_size"] < 1:
        raise ConfigError(f"batch_size must be >= 1; got {cfg['batch_size']}")
    if cfg["batch_size"] > 1 and not cfg["parallel"]:
        raise ConfigError(BATCH_NEEDS_PARALLEL)
    cfg["N"] = int(cfg["N"])
    cfg["density"] = float(cfg["density"])
    cfg["max_evals"] = int(cfg["max_evals"])
    if cfg["max_evals"] < 1:
        raise ConfigError("max_evals must be >= 1 (successful evaluations, first batch included)")
    if cfg["steps"] is not None:
        cfg["steps"] = int(cfg["steps"])

    search = cfg["search"]
    if not isinstance(search, dict):
        raise ConfigError("search must be a dict {name: spec}")
    if "density" in search:
        raise ConfigError("density is the CONFIG key 'density' (fixed), not a search entry")
    absent = [k for k in SIM_NAMES if k not in search]
    if absent or ("m" in search) == ("m_rel" in search):
        raise ConfigError(
            "search must hold U0, r0, n and exactly one of m, m_rel (each free or "
            f"\"fixed\"); missing {absent}, m/m_rel: "
            f"{[k for k in ('m', 'm_rel') if k in search]}"
        )
    cfg["search"] = {name: _plain(spec) for name, spec in search.items()}

    cfg["target"] = str(Path(cfg["target"]).expanduser().resolve())
    cfg["out_root"] = str(Path(cfg["out_root"]).expanduser().resolve())
    if cfg["gp_log_dir"] is None:
        cfg["gp_log_dir"] = str(Path(cfg["out_root"]) / "gp_log")
    else:
        cfg["gp_log_dir"] = str(Path(cfg["gp_log_dir"]).expanduser().resolve())
    if cfg["mc_dfm_root"] is None:
        cfg["mc_dfm_root"] = os.environ.get("MC_DFM_ROOT", "")

    if cfg["parallel"]:
        pcfg = cfg["parallel_cfg"]
        if not isinstance(pcfg, dict):
            raise ConfigError("parallel=True needs parallel_cfg (dict)")
        bad = set(pcfg) - set(PARALLEL_REQUIRED) - set(PARALLEL_OPTIONAL)
        if bad:
            raise ConfigError(
                f"unknown parallel_cfg keys: {sorted(bad)} (code_root and mc_dfm_root "
                "are set by inverse.py)"
            )
        miss = [k for k in PARALLEL_REQUIRED if k not in pcfg]
        if miss:
            raise ConfigError(f"parallel_cfg is missing: {miss}")
    return cfg


def _plain(obj):
    """JSON-comparable copy: tuples become lists; ints stay ints, other
    numbers become floats (1 == 1.0, so either compares equal)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return float(obj)


def _gp_log_key(gp_log_dir, out_root):
    """gp_log_dir as locked: relative to out_root when it lies inside it, so a
    run folder that is moved as a whole still resumes; absolute otherwise."""
    gp, root = Path(gp_log_dir), Path(out_root)
    try:
        return gp.relative_to(root).as_posix()
    except ValueError:
        return str(gp)


def locked_fields(cfg):
    out = {key: _plain(cfg[key]) for key in LOCKED}
    out["gp_log_dir"] = _gp_log_key(cfg["gp_log_dir"], cfg["out_root"])
    return out


def _snapshot_locked(snapshot):
    """Locked fields of a snapshot; an absolute gp_log_dir inside the
    snapshot's own out_root (inverse_config.json before 2026-10-11) is read
    as relative, like locked_fields writes it now."""
    old = dict(snapshot["locked"])
    gp = old.get("gp_log_dir")
    if gp and Path(gp).is_absolute() and snapshot.get("out_root"):
        old["gp_log_dir"] = _gp_log_key(gp, snapshot["out_root"])
    return old


# ------------------------------------------------------------------ objects


def _import_pipeline(cfg):
    """Import bo (and through it MC-DFM, HOOMD, saxs-fft)."""
    if cfg["mc_dfm_root"] and cfg["mc_dfm_root"] not in sys.path:
        sys.path.insert(0, cfg["mc_dfm_root"])
    import bo
    return bo


def build_param_space(bo, cfg):
    ps = bo.ParamSpace({**cfg["search"], "density": {"fixed": cfg["density"]}})
    bo._validate_param_mode(ps, "sim")
    return ps


def build_dataset(cfg):
    from datasets import Dataset, ExperimentalParams, SimulationParams

    return Dataset(
        id=DATASET_ID,
        exp_path=cfg["target"],
        exp=ExperimentalParams(),
        sim=SimulationParams(density=cfg["density"]),
        datatype="sq",
    )


def sim_defaults(cfg):
    out = {"N": cfg["N"], "plot": False, **POTENTIALS[cfg["potential"]]}
    if cfg["steps"] is not None:
        out["steps"] = cfg["steps"]
    if cfg["device"] is not None:
        out["device"] = cfg["device"]
    return out


def parallel_cfg(cfg):
    if not cfg["parallel"]:
        return None
    return {**cfg["parallel_cfg"], "code_root": str(REPO), "mc_dfm_root": cfg["mc_dfm_root"]}


def build_objective(bo, cfg, ps, dataset):
    metric = METRICS[cfg["metric"]]
    scattering = SCATTERING[cfg["scattering"]]
    return bo.make_global_objective(
        [dataset], ps, ffpath=str(FORMFACTOR), out_root=cfg["out_root"], trim_tail=0,
        sim_defaults=sim_defaults(cfg), mode="sim",
        scattering_method=scattering["scattering_method"],
        scattering_kwargs=dict(scattering["scattering_kwargs"]),
        metric=metric["metric"], compare_q_range=cfg["compare_q_range"],
        metric_kwargs=dict(metric["metric_kwargs"]),
        parallel=cfg["parallel"], parallel_cfg=parallel_cfg(cfg),
    )


# ------------------------------------------------------------------ checks


def file_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_target(cfg):
    """Checks 1 and 3. Returns a dict for the log and the snapshot."""
    import metrics

    path = Path(cfg["target"])
    if not path.is_file():
        raise ConfigError(f"target not found: {path}")
    md5 = file_md5(path)
    try:
        arr = np.load(path)
    except Exception as exc:
        raise ConfigError(f"target {path} is not a .npy array: {exc}") from exc
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr[:, :2]).all():
        raise ConfigError(f"target must be a finite (N, 2+) array [q, S]; got shape {arr.shape}")
    curve = metrics._sanitize_curve(arr)

    kwargs = METRICS[cfg["metric"]]["metric_kwargs"]
    p = metrics.shift_rmse_params(kwargs)
    peak = metrics.detect_first_peak(
        curve, search_range=p["peak_search_range"], prominence_frac=p["prominence_frac"],
        dispersed_delta=p["dispersed_delta"],
    )
    if peak.dispersed:
        print(
            f"[inverse] WARNING: target is dispersed (smoothed S max - 1 = "
            f"{peak.s_max - 1:.4f} < dispersed_delta={p['dispersed_delta']} in "
            f"{p['peak_search_range']}): no alignment, spacing term dispersed_shift="
            f"{p['dispersed_shift']}; continuing"
        )
    elif not np.isfinite(peak.q1) or peak.n_peaks_found == 0:
        raise ConfigError(
            f"no first peak on the target in {p['peak_search_range']} "
            f"(q1={peak.q1}, peaks found={peak.n_peaks_found})"
        )
    try:
        self_loss, _, _, _, _ = metrics.shift_rmse_loss(curve, curve, None, kwargs)
    except metrics.MetricFailed as exc:
        raise ConfigError(f"shift_rmse fails on the target itself: {exc}") from exc
    if not abs(self_loss) < 1e-9:
        raise ConfigError(f"shift_rmse(target, target) = {self_loss}, expected 0")
    return {
        "path": str(path), "md5": md5, "points": int(curve.shape[0]),
        "q_range": [float(curve[0, 0]), float(curve[-1, 0])],
        "dispersed": bool(peak.dispersed),
        "q1": _finite_or_none(peak.q1), "s_peak": _finite_or_none(peak.s_peak),
        "s_max": float(peak.s_max),
        "n_peaks_found": int(peak.n_peaks_found), "metric_kwargs": dict(kwargs),
    }


def _finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def _axis(spec, n_points):
    if "fixed" in spec:
        return [float(spec["fixed"])]
    lo, hi = (float(v) for v in spec["bounds"])
    values = list(np.linspace(lo, hi, n_points))
    if "init" in spec:
        values.append(float(spec["init"]))
    return sorted(set(values))


def check_rmax_grid(cfg, n_points=RMAX_GRID_POINTS):
    """Check 5: rmax < L/2 (and n > m) on a grid over the search box."""
    import bo
    from simulation import resolve_table_bounds

    search = cfg["search"]
    m_name = "m_rel" if "m_rel" in search else "m"
    names = ("U0", "r0", "n", m_name)
    axes = [_axis(search[name], n_points) for name in names]
    worst = None
    for U0, r0, n, mv in itertools.product(*axes):
        m = bo.m_from_m_rel(n, mv) if m_name == "m_rel" else mv
        point = {"U0": float(U0), "r0": float(r0), "n": float(n), m_name: float(mv)}
        if not n > m:
            raise ConfigError(f"search box has n <= m at {point} (m={m})")
        try:
            b = resolve_table_bounds(U0, n, m, r0, N=cfg["N"], density=cfg["density"])
        except ValueError as exc:
            raise ConfigError(f"search box corner {point}: {exc}") from exc
        if worst is None or b["rmax"] > worst["rmax"]:
            worst = {"rmax": float(b["rmax"]), "L_half": float(b["L"] / 2), "at": point}
    worst["grid_points"] = int(np.prod([len(a) for a in axes]))
    return worst


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                              text=True, timeout=30)
    try:
        head = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=no")
    except Exception:
        return {"commit": "unknown", "dirty": None}
    if head.returncode != 0:
        return {"commit": "unknown", "dirty": None}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def check_resume_settings(cfg, snapshot):
    """Check 7: a resume keeps the locked fields."""
    if snapshot is None:
        return
    old = _snapshot_locked(snapshot)
    new = locked_fields(cfg)
    changed = [k for k in LOCKED if old.get(k) != new.get(k)]
    if changed:
        lines = "\n".join(f"  {k}: run started with {old.get(k)!r}, now {new.get(k)!r}"
                          for k in changed)
        raise ConfigError(
            f"{cfg['out_root']} was started with other settings; only max_evals and the "
            f"parallel settings may change on a resume. Use a new out_root.\n{lines}"
        )


def read_snapshot(out_root):
    path = Path(out_root) / SNAPSHOT
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def preflight(cfg, bo):
    """Run checks 1-7. Returns ``(info, ps, dataset, objective, snapshot)``."""
    target = check_target(cfg)
    cfg["target_md5"] = target["md5"]
    try:
        ps = build_param_space(bo, cfg)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    dataset = build_dataset(cfg)
    rmax = check_rmax_grid(cfg)
    try:
        objective = build_objective(bo, cfg, ps, dataset)   # N_grid check (C6)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    snapshot = read_snapshot(cfg["out_root"])
    check_resume_settings(cfg, snapshot)
    info = {"target": target, "rmax": rmax, "git": git_state()}
    peak = ("dispersed (no first peak)" if target["dispersed"] else
            f"first peak q1={target['q1']:.5g} (S={target['s_peak']:.3g}, "
            f"{target['n_peaks_found']} peak(s))")
    print(
        f"[inverse] target {target['path']}\n"
        f"[inverse]   md5 {target['md5']}; {target['points']} points, q "
        f"{target['q_range'][0]:.5g}-{target['q_range'][1]:.5g}; {peak}; self-loss 0\n"
        f"[inverse] rmax < L/2 on {rmax['grid_points']} grid points; worst rmax="
        f"{rmax['rmax']:.3f} (L/2={rmax['L_half']:.3f}) at {rmax['at']}\n"
        f"[inverse] commit {info['git']['commit']}"
        + (" (tracked files modified)" if info["git"]["dirty"] else "")
    )
    if info["git"]["dirty"]:
        print("[inverse] WARNING: the repo has uncommitted changes; the commit does not "
              "describe the code that runs")
    return info, ps, dataset, objective, snapshot


def write_snapshot(cfg, info, snapshot, config_path):
    """Create or extend out_root/inverse_config.json (one entry per start)."""
    start = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(Path(config_path).resolve()),
        "out_root": cfg["out_root"],
        "commit": info["git"]["commit"],
        "dirty": info["git"]["dirty"],
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "max_evals": cfg["max_evals"],
        "parallel": cfg["parallel"],
        "parallel_cfg": _plain(parallel_cfg(cfg)),
        "device": cfg["device"],
    }
    if snapshot is None:
        snapshot = {
            "description": cfg["description"],
            "out_root": cfg["out_root"],
            "target": info["target"],
            "rmax_check": info["rmax"],
            "locked": locked_fields(cfg),
            "starts": [],
        }
    snapshot["starts"].append(start)
    path = Path(cfg["out_root"]) / SNAPSHOT
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2) + "\n")
    os.replace(tmp, path)
    return snapshot


# ------------------------------------------------------------------ summary


def _block_statuses(path):
    """``{eval_id: 'ok' | 'FAILED' | 'ABANDONED'}`` from the block headers."""
    out = {}
    if not Path(path).is_file():
        return out
    import csv
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("# Iteration"):
                try:
                    eval_id = int(row[0].split()[-1])
                except ValueError:
                    continue
                out[eval_id] = "ok" if row[1] == "total_loss" else row[2]
    return out


def write_summary(cfg, bo, snapshot=None):
    """Best evaluation and loss components, from bo_trajectory.csv."""
    out_root = Path(cfg["out_root"])
    traj = out_root / "bo_trajectory.csv"
    snapshot = snapshot if snapshot is not None else read_snapshot(out_root)
    successful, _ = bo._read_trajectory(traj, 1e8) if traj.is_file() else ([], set())
    statuses = _block_statuses(traj)
    starts = (snapshot or {}).get("starts", [])
    commits = sorted({s["commit"] for s in starts})

    def num(rec, key):
        value = rec.get(key, "")
        return float(value) if value not in ("", None) else None

    best = None
    if successful:
        eval_id, total, rec = min(successful, key=lambda item: item[1])
        best = {
            "eval_id": eval_id,
            "eval_dir": str(out_root / f"eval_{eval_id:03d}" / DATASET_ID),
            "loss": total,
            "rmse": num(rec, "rmse"),
            "shift": num(rec, "shift"),
            **{k: num(rec, k) for k in ("U0", "r0", "n", "m", "m_rel", "density")},
        }
    summary = {
        "description": cfg["description"],
        "written": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "target": {"path": cfg["target"], "md5": cfg.get("target_md5")},
        "commits": commits,
        "dirty_starts": sum(1 for s in starts if s.get("dirty")),
        "snapshot": str(out_root / SNAPSHOT),
        "loss": "shift_rmse (defaults): loss = rmse + lambda_shift * |ln(q1_sim / q1_tgt)|",
        "max_evals": cfg["max_evals"],
        "batch_size": cfg["batch_size"],
        "n_successful": len(successful),
        "n_failed": sum(1 for s in statuses.values() if s == "FAILED"),
        "n_abandoned": sum(1 for s in statuses.values() if s == "ABANDONED"),
        "complete": len(successful) >= cfg["max_evals"],
        "best": best,
    }
    (out_root / f"{SUMMARY}.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"Inverse design summary ({summary['written']})",
        f"description : {summary['description']}",
        f"out_root    : {summary['out_root']}",
        f"target      : {summary['target']['path']}",
        f"target md5  : {summary['target']['md5']}",
        f"commit(s)   : {', '.join(commits) or 'unknown'}"
        + (f"  ({summary['dirty_starts']} start(s) with modified files)"
           if summary["dirty_starts"] else ""),
        f"loss        : {summary['loss']}",
        f"evaluations : {summary['n_successful']} successful of max_evals="
        f"{summary['max_evals']} (batch_size {summary['batch_size']}); "
        f"{summary['n_failed']} failed; "
        f"{summary['n_abandoned']} abandoned"
        + ("" if summary["complete"] else "  (not complete)"),
    ]
    if best:
        lines += [
            f"best        : eval {best['eval_id']}  ({best['eval_dir']})",
            f"  loss  = {best['loss']:.6f}   rmse = {best['rmse']}   shift = {best['shift']}",
            "  " + "  ".join(f"{k} = {best[k]:.6g}" for k in ("U0", "r0", "n", "m", "density")
                             if best[k] is not None)
            + (f"  m_rel = {best['m_rel']:.6g}" if best["m_rel"] is not None else ""),
        ]
    else:
        lines.append("best        : no successful evaluation yet")
    (out_root / f"{SUMMARY}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


# ------------------------------------------------------------------ main


def run(config_path, overrides=None, check=False, summary_only=False):
    cfg = normalize_config(load_config(config_path), overrides)
    bo = _import_pipeline(cfg)
    print(f"[inverse] config {Path(config_path).resolve()}; out_root {cfg['out_root']}")

    if summary_only:
        snap = read_snapshot(cfg["out_root"])
        if snap is not None:
            cfg["target_md5"] = snap["locked"]["target_md5"]
        return write_summary(cfg, bo, snap)

    info, ps, _dataset, objective, snapshot = preflight(cfg, bo)
    mode = "parallel (one GPU job per evaluation)" if cfg["parallel"] else "sequential (this process)"
    print(f"[inverse] {mode}; max_evals={cfg['max_evals']} successful evaluations "
          f"(first batch included, resumes counted); batch_size={cfg['batch_size']}; "
          f"bo_seed={cfg['bo_seed']}"
          + (f"; TEST steps={cfg['steps']}" if cfg["steps"] is not None else ""))
    if check:
        state = bo.inspect_resume(cfg["out_root"])
        print(f"[inverse] --check passed; resume state: {state}")
        return {"info": info, "resume": state}

    os.makedirs(cfg["out_root"], exist_ok=True)
    snapshot = write_snapshot(cfg, info, snapshot, config_path)
    best, history = bo.run_bo_resumable(
        objective, ps, str(FORMFACTOR), cfg["out_root"], max_evals=cfg["max_evals"],
        seed=cfg["bo_seed"], batch_size=cfg["batch_size"], gp_log_dir=cfg["gp_log_dir"],
    )
    return write_summary(cfg, bo, snapshot)


def _bool(text):
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="Python module defining CONFIG")
    ap.add_argument("--check", action="store_true", help="startup checks only; no simulation")
    ap.add_argument("--summary", action="store_true", help="rewrite the summary and exit")
    ap.add_argument("--out", help="override out_root (smoke tests)")
    ap.add_argument("--target", help="override target (smoke tests)")
    ap.add_argument("--parallel", type=_bool, help="override parallel (smoke tests)")
    ap.add_argument("--steps", type=int, help="production steps (smoke tests only)")
    ap.add_argument("--device", choices=("gpu", "cpu"), help="HOOMD device (local tests)")
    a = ap.parse_args(argv)
    overrides = {"out_root": a.out, "target": a.target, "parallel": a.parallel,
                 "steps": a.steps, "device": a.device}
    try:
        run(a.config, overrides, check=a.check, summary_only=a.summary)
    except ConfigError as exc:
        print(f"[inverse] STARTUP CHECK FAILED, nothing was simulated:\n{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
