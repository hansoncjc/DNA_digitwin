"""
Bayesian Optimization (BO) utilities for global + per-dataset fitting.

What this gives you
-------------------
1) A simple way to declare which parameters are optimized and whether they are:
   - GLOBAL (one value shared by all datasets), or
   - LOCAL (a separate value per dataset).

2) A generic BO loop using BoTorch (SingleTaskGP + qLogEI) that minimizes an
   objective you define via dataset simulation → SAXS → compare_to_exp → sum loss.

3) A pack/unpack system that maps an optimizer vector x ∈ [0,1]^D to a dict of
   named parameters (globals + locals) with your bounds.

Minimal usage (Stage 1: globals only)
-------------------------------------
param_cfg = {
    "global": {
        "alpha":   {"bounds": (0.2, 5.0),   "init": 1.0},    # density coeff
        "n":       {"bounds": (6.0, 20.0),  "init": 12.0},
        "m":       {"bounds": (4.0, 12.0),  "init": 6.0},
        # Optional global mapping coeffs you want to learn in Stage 1 as well:
        # "k":       {"bounds": (0.5, 1.2),   "init": 0.76},  # r0 mapping coeff
        # "A":       {"bounds": (1.0, 20.0),  "init": 2.0},
        # "mu_c":    {"bounds": (40.0, 200.0),"init": 100.0},
        # "sigma_c": {"bounds": (5.0, 25.0),  "init": 15.0},
        # "sigma_b": {"bounds": (0.05, 0.5),  "init": 0.1},
    },
    "local": {
        # leave empty in Stage 1 (or put "U0"/"r0" here for Stage 2 refinements)
    }
}

from bo import ParamSpace, make_global_objective, run_bo
ps = ParamSpace(param_cfg, dataset_ids=[d.id for d in datasets])

obj = make_global_objective(datasets, ps, out_root="Optimization_Results",
                            trim_tail=0, sim_defaults={"steps": 1_500_000})

best, history = run_bo(obj, ps, n_iters=20, seed=0)
print("Best params (physical):", ps.decode(best))
"""

import os
import json
import random
import traceback
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import csv

from simulation import run_simulation
from scattering import (
    DEFAULT_N_GRID,
    convert_to_SAXS,
    convert_to_SAXS_fft,
    estimate_saxsfft_n_grid,
    extract_exp_sq,
)
from metrics import (
    MetricFailed,
    compare_to_exp,
    compare_to_exp_saxsfft,
    load_shift_rmse_components,
    shift_rmse_params,
)

# ------------------------- Evaluation failures ------------------------- #

class EvaluationFailed(Exception):
    """Raised when a BO objective evaluation fails after GPU job retry."""


DEFAULT_MAX_JOB_RETRIES = 1
DEFAULT_MAX_ACQ_ATTEMPTS = 5

# ------------------------- Modes & param types ------------------------- #

# Parameters that correspond to direct simulation inputs
_SIM_PARAMS = {"density", "r0", "U0"}

# Parameters that don't correpond to modes
_ALWAYS_OK_SIM = {"n", "m"}

# Parameters that correspond to mapping coefficients
_MAP_PARAMS = {"alpha", "k", "A", "mu_c", "sigma_c", "K_s"}


def _validate_param_mode(ps, mode: str) -> None:
    """
    Ensure that the ParamSpace configuration is consistent with the chosen mode.

    mode = "map": only mapping parameters (alpha, k, A, ...) are allowed.
    mode = "sim": only direct simulation parameters (density, r0, U0, ...) are allowed.

    n and m are always treated as *simulation* parameters and are allowed in both modes.
    """
    if mode not in ("map", "sim"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'map' or 'sim'.")

    g_names = set(ps.cfg.get("global", {}).keys())
    l_names = set(ps.cfg.get("local", {}).keys())

    if mode == "map":
        illegal = (g_names | l_names) & _SIM_PARAMS
        if illegal:
            raise ValueError(
                "ParamSpace configuration is inconsistent with mode='map'. "
                f"These direct simulation parameters are not allowed here: {sorted(illegal)}"
            )
    else:  # mode == "sim"
        illegal = (g_names | l_names) & _MAP_PARAMS
        if illegal:
            raise ValueError(
                "ParamSpace configuration is inconsistent with mode='sim'. "
                f"These mapping parameters are not allowed here: {sorted(illegal)}"
            )
def describe_training_config(ps, mode: str) -> str:
    """
    Return a human-readable description of what the BO objective will train,
    given the ParamSpace and the chosen mode.

    You can simply print(describe_training_config(ps, mode)) from your script.
    """
    g_names = set(ps.cfg.get("global", {}).keys())
    l_names = set(ps.cfg.get("local", {}).keys())

    map_params = (g_names | l_names) & _MAP_PARAMS
    sim_params     = (g_names | l_names) & _SIM_PARAMS

    lines = []
    lines.append(f"Training mode: {mode}")
    lines.append(f"  Global params: {sorted(g_names)}")
    lines.append(f"  Local  params: {sorted(l_names)}")
    lines.append(f"  Recognized map params: {sorted(map_params)}")
    lines.append(f"  Recognized sim params:     {sorted(sim_params)}")
    if mode == "map" and sim_params:
        lines.append("  [WARNING] sim params present but will cause an error if used with mode='map'.")
    if mode == "sim" and map_params:
        lines.append("  [WARNING] map params present but will cause an error if used with mode='sim'.")
    return "\n".join(lines)

# ------------------------- Parameter packing ------------------------- #
class ParamSpace:
    """
    Pack/unpack parameters for BO with global + per-dataset roles.

    param_cfg schema:
        {
          "global": {
             "alpha":   {"bounds": (0.2, 5.0),  "init": 1.0},
             "n":       {"bounds": (6.0, 20.0), "init": 12.0},
             "m":       {"bounds": (4.0, 12.0), "init": 6.0},
             # You can also "freeze" any param:
             # "k": {"bounds": (0.5,1.2), "init": 0.76, "fixed": 0.76}
          },
          "local": {
             # example for Stage 2:
             # "U0": {"bounds": (0.1, 150.0), "init": 50.0},
             # "r0": {"bounds": (2.0, 2.5),   "init": 2.2},
          }
        }

    Notes
    -----
    - GLOBAL params appear once in the vector.
    - LOCAL params appear once **per dataset**, ordered by dataset_ids.
      e.g., for local "U0" and dataset_ids ["d1","d2"], the vector holds
      ["U0:d1", "U0:d2"] (after the globals).
    - "fixed" bypasses optimization (not placed in the vector) but the fixed
      value is exposed in decode()/unpack() so your objective can use it.
    """

    def __init__(self, param_cfg: Dict[str, Dict[str, Dict[str, float]]],
                 dataset_ids: List[str]):
        self.cfg = {"global": dict(param_cfg.get("global", {})),
                    "local":  dict(param_cfg.get("local",  {}))}
        self.dataset_ids = list(dataset_ids)

        # Build ordered vector schema
        self._names: List[str] = []          # vector labels (for debug)
        self._lo: List[float] = []
        self._hi: List[float] = []
        self._init: List[float] = []
        self._fixed_globals: Dict[str, float] = {}
        self._fixed_locals: Dict[Tuple[str, str], float] = {}  # (name, dsid) -> val

        # Globals first
        for name, spec in self.cfg["global"].items():
            if "fixed" in spec:
                self._fixed_globals[name] = float(spec["fixed"])
            else:
                lo, hi = spec["bounds"]
                self._names.append(name)
                self._lo.append(float(lo)); self._hi.append(float(hi))
                self._init.append(float(spec.get("init", (lo + hi) / 2)))

        # Then locals, expanded per dataset
        for lname, spec in self.cfg["local"].items():
            if "fixed" in spec:
                # one fixed value applies to all datasets
                for dsid in self.dataset_ids:
                    self._fixed_locals[(lname, dsid)] = float(spec["fixed"])
            else:
                lo, hi = spec["bounds"]
                for dsid in self.dataset_ids:
                    label = f"{lname}:{dsid}"
                    self._names.append(label)
                    self._lo.append(float(lo)); self._hi.append(float(hi))
                    self._init.append(float(spec.get("init", (lo + hi) / 2)))

        self.d = len(self._names)
        self._lo_t = torch.tensor(self._lo, dtype=torch.float64)
        self._hi_t = torch.tensor(self._hi, dtype=torch.float64)
        self._init_t = torch.tensor(self._init, dtype=torch.float64)

    # ---- scaling helpers ---- #

    def unit_to_phys(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Map x in [0,1]^d to physical bounds."""
        return self._lo_t + x_unit * (self._hi_t - self._lo_t)

    def phys_to_unit(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Map x in physical bounds to [0,1]^d."""
        return (x_phys - self._lo_t) / (self._hi_t - self._lo_t)

    def init_unit(self) -> torch.Tensor:
        """Return initial x in [0,1]^d from provided 'init' values."""
        return self.phys_to_unit(self._init_t)

    def bounds_unit(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lb, ub) tensors in unit cube."""
        lb = torch.zeros(self.d, dtype=torch.float64)
        ub = torch.ones(self.d, dtype=torch.float64)
        return lb, ub

    # ---- decoding ---- #

    def decode(self, x_phys: torch.Tensor) -> Dict[str, Any]:
        """
        Convert a physical vector into structured dicts:
        {
          "global": {name: val, ...},
          "local":  {dsid: {lname: val, ...}, ...}
        }
        Includes fixed params.
        """
        x = x_phys.detach().cpu().numpy().tolist()
        out_g: Dict[str, float] = dict(self._fixed_globals)
        out_l: Dict[str, Dict[str, float]] = {dsid: {} for dsid in self.dataset_ids}

        for label, val in zip(self._names, x):
            if ":" in label:
                pname, dsid = label.split(":")
                out_l[dsid][pname] = float(val)
            else:
                out_g[label] = float(val)

        # include fixed locals
        for (lname, dsid), v in self._fixed_locals.items():
            out_l[dsid][lname] = float(v)
        return {"global": out_g, "local": out_l}


# ------------------------- Objective factory ------------------------- #

def _write_iteration_block(filepath: str, iteration: int, total_loss: float, records: List[Dict[str, Any]]):
    """
    Append one iteration block to the global trajectory CSV.

    Parameters
    ----------
    filepath : str
        Path to the global trajectory CSV file (e.g., "Optimization_Results/bo_trajectory.csv")
    iteration : int
        Current iteration number
    total_loss : float
        Total loss summed across all datasets for this iteration
    records : List[Dict[str, Any]]
        List of parameter/loss records for each dataset in this iteration.
        Each record should have keys: iteration, dataset_id, loss, k, alpha, A, etc.

    Format
    ------
    # Iteration N,total_loss,<value>
    iteration,dataset_id,loss,k,alpha,A,mu_c,mu_b,sigma_c,sigma_b,K_s,density,n,m,r0,U0,
        rmax,rmax_over_r0,t_tol_lj,n_pairs_below_rmin,rmse,shift
    N,d0,loss_val,k_val,...
    N,d1,loss_val,k_val,...
    <blank line>

    The audit columns (``AUDIT_COLUMNS``) are per (eval, dataset): in map
    mode each dataset has its own U0, hence its own cutoff. ``rmse`` and
    ``shift`` are filled only for ``metric='shift_rmse'``. Each block carries
    its own header row, so readers that look columns up by name also read
    older blocks without these columns.
    """
    mode = 'a' if os.path.exists(filepath) else 'w'

    with open(filepath, mode, newline='') as f:
        writer = csv.writer(f)

        # Header line with total loss
        writer.writerow([f"# Iteration {iteration}", "total_loss", total_loss])

        # Column headers
        if len(records) > 0:
            writer.writerow(list(records[0].keys()))

        # Data rows
        for record in records:
            writer.writerow(list(record.values()))

        # Blank separator line
        writer.writerow([])


AUDIT_COLUMNS = ("rmax", "rmax_over_r0", "t_tol_lj", "n_pairs_below_rmin", "rmse", "shift")


def _audit_value(value, cast=float):
    """``cast(value)``, or "" for missing / None (sim results may arrive as strings)."""
    if value is None or value == "" or str(value) == "None":
        return ""
    try:
        return cast(value)
    except (TypeError, ValueError):
        return ""


def _audit_fields(sim_result: Optional[Dict[str, Any]], r0: float, save_dir: str) -> Dict[str, Any]:
    """
    Per-(eval, dataset) audit columns for ``bo_trajectory.csv``.

    ``sim_result`` is the ``run_simulation`` dict (sequential) or the
    stringified ``result`` from the worker's DONE flag (parallel). ``rmse``
    (M4) and ``shift`` come from ``save_dir/shift_rmse_diagnostics.json`` and
    are blank for other metrics.
    """
    sim_result = sim_result or {}
    rmax = _audit_value(sim_result.get("rmax"))
    comps = load_shift_rmse_components(save_dir)
    return {
        "rmax": rmax,
        "rmax_over_r0": rmax / float(r0) if rmax != "" else "",
        "t_tol_lj": _audit_value(sim_result.get("t_tol_lj")),
        "n_pairs_below_rmin": _audit_value(sim_result.get("n_pairs_below_rmin"), int),
        "rmse": comps.get("rmse", ""),
        "shift": comps.get("shift", ""),
    }


def _append_loss_components(
    out_root: str, iteration: int, total_loss: float, records: List[Dict[str, Any]],
) -> None:
    """
    Append one row per dataset to ``out_root/loss_components.txt`` (CSV):
    iteration, dataset_id, rmse (M4), shift (lambda*|ln q1 ratio|), loss
    (= rmse + shift for shift_rmse), and the iteration's weighted total_loss.
    Components are blank for metrics other than shift_rmse.
    """
    path = os.path.join(out_root, "loss_components.txt")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["iteration", "dataset_id", "rmse", "shift", "loss", "total_loss"])
        for rec in records:
            writer.writerow([
                iteration, rec.get("dataset_id", ""), rec.get("rmse", ""),
                rec.get("shift", ""), rec.get("loss", ""), total_loss,
            ])


def _write_failed_iteration_block(
    filepath: str,
    iteration: int,
    records: List[Dict[str, Any]],
    reason: str = "",
):
    """Append a trajectory block for a failed evaluation (not used by the GP)."""
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "iteration", "dataset_id", "loss", "k", "alpha", "A", "mu_c",
                "mu_b", "sigma_c", "sigma_b", "K_s", "density", "n", "m", "r0", "U0",
            ])
        writer.writerow([f"# Iteration {iteration}", "EVALUATION", "FAILED", reason])
        if records:
            writer.writerows([
                [r.get(c, "") for c in (
                    "iteration", "dataset_id", "loss", "k", "alpha", "A", "mu_c",
                    "mu_b", "sigma_c", "sigma_b", "K_s", "density", "n", "m", "r0", "U0",
                )]
                for r in records
            ])


def _failed_trajectory_record(eval_id, ds_id, G, plan=None, reason="FAILED"):
    rec = {
        "iteration": eval_id,
        "dataset_id": ds_id,
        "loss": reason,
        "k": G.get("k", ""),
        "alpha": G.get("alpha", ""),
        "A": G.get("A", ""),
        "mu_c": G.get("mu_c", ""),
        "mu_b": G.get("mu_b", ""),
        "sigma_c": G.get("sigma_c", ""),
        "sigma_b": G.get("sigma_b", ""),
        "K_s": G.get("K_s", ""),
        "density": "ERROR",
        "n": "ERROR",
        "m": "ERROR",
        "r0": "ERROR",
        "U0": "ERROR",
    }
    if plan is not None:
        rec.update({
            "density": float(plan["density"]),
            "n": float(plan["n"]),
            "m": float(plan["m"]),
            "r0": float(plan["r0"]),
            "U0": float(plan["U0"]),
        })
    return rec


def _launcher_config_from_pcfg(pcfg: Dict[str, Any], run_dir: Path):
    """Build ``LauncherConfig`` from a parallel-cfg dict."""
    from parallel import LauncherConfig

    pcfg = pcfg or {}
    return LauncherConfig(
        run_dir=run_dir,
        partition=pcfg.get("partition", "gpu-a40"),
        account=pcfg.get("account", "zeelab"),
        gpus=int(pcfg.get("gpus", 1)),
        mem=pcfg.get("mem", "40G"),
        time=pcfg.get("time", "02:00:00"),
        module_loads=list(pcfg.get("module_loads", [])),
        venv_activate=pcfg.get("venv_activate", ""),
        code_root=pcfg.get("code_root", ""),
        mc_dfm_root=pcfg.get("mc_dfm_root", ""),
        poll_interval=float(pcfg.get("poll_interval", 15.0)),
        max_wait=float(pcfg.get("max_wait", 4 * 3600)),
    )


def _parallel_prepare_eval_jobs(
    *,
    datasets: List[Any],
    eval_id: int,
    G: Dict[str, Any],
    L: Dict[str, Any],
    out_root: str,
    trim_tail: int,
    sim_defaults: Dict[str, Any],
    mode: str,
    scattering_method: str,
    scattering_kwargs: Dict[str, Any],
    metric: str,
    compare_q_range: Optional[Tuple[float, float]],
    dp_coeff: float,
    plot_apdist: bool,
    ffpath: str,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[dict], List[dict], List[Dict[str, Any]], Optional[str]]:
    """
    Resolve per-dataset sim params and build Slurm job specs for one BO eval.

    Returns ``(job_specs, plans, fail_records, fail_reason)``. On success
    ``fail_records`` is empty and ``fail_reason`` is ``None``.
    """
    plans: List[Dict[str, Any]] = []
    phase_a_failed: List[str] = []
    fail_records: List[Dict[str, Any]] = []

    for ds in datasets:
        try:
            n = float(G["n"]) if "n" in G else float(ds.sim.n)
            m = float(G["m"]) if "m" in G else float(ds.sim.m)

            if mode == "map":
                alpha = float(G.get("alpha", 1.0))
                density = ds.rho_N(alpha=alpha)
            else:
                if "density" in L[ds.id]:
                    density = float(L[ds.id]["density"])
                elif "density" in G:
                    density = float(G["density"])
                elif ds.sim.density is not None:
                    density = float(ds.sim.density)
                else:
                    raise ValueError(
                        f"Dataset {ds.id}: density not provided in mode='sim' "
                        "and dataset.sim.density is None."
                    )

            if mode == "map":
                if "k" in G:
                    r0 = float(ds.r0_sigma(k=float(G["k"])))
                elif ds.sim.r0 is not None:
                    r0 = float(ds.sim.r0)
                else:
                    raise ValueError(
                        f"Dataset {ds.id}: r0 not provided and no mapping coeff 'k' "
                        "found in mode='map'."
                    )
            else:
                if "r0" in L[ds.id]:
                    r0 = float(L[ds.id]["r0"])
                elif "r0" in G:
                    r0 = float(G["r0"])
                elif ds.sim.r0 is not None:
                    r0 = float(ds.sim.r0)
                else:
                    raise ValueError(
                        f"Dataset {ds.id}: r0 not provided in mode='sim' "
                        "and dataset.sim.r0 is None."
                    )

            if mode == "map":
                if all(k in G for k in ("A", "mu_c", "sigma_c")):
                    U0 = float(ds.U0_from_gaussian(
                        A=G["A"],
                        mu_c=G["mu_c"],
                        sigma_c=G["sigma_c"],
                        K_s=G.get("K_s", 0.05),
                    ))
                elif ds.sim.U0 is not None:
                    U0 = float(ds.sim.U0)
                else:
                    raise ValueError(
                        f"Dataset {ds.id}: U0 not provided and no global Gaussian coeffs "
                        "found in mode='map'."
                    )
            else:
                if "U0" in L[ds.id]:
                    U0 = float(L[ds.id]["U0"])
                elif "U0" in G:
                    U0 = float(G["U0"])
                elif ds.sim.U0 is not None:
                    U0 = float(ds.sim.U0)
                else:
                    raise ValueError(
                        f"Dataset {ds.id}: U0 not provided in mode='sim' "
                        "and dataset.sim.U0 is None."
                    )

            save_dir = os.path.join(out_root, f"eval_{eval_id:03d}", ds.id)
            os.makedirs(save_dir, exist_ok=True)

            sim_params_record = {
                "dataset_id": ds.id,
                "eval_id": eval_id,
                "k": G.get("k", ""),
                "alpha": G.get("alpha", ""),
                "A": G.get("A", ""),
                "mu_c": G.get("mu_c", ""),
                "mu_b": G.get("mu_b", ""),
                "sigma_c": G.get("sigma_c", ""),
                "sigma_b": G.get("sigma_b", ""),
                "K_s": G.get("K_s", ""),
                "density": float(density),
                "n": float(n),
                "m": float(m),
                "r0": float(r0),
                "U0": float(U0),
            }
            param_path = os.path.join(save_dir, f"sim_params_{ds.id}.csv")
            with open(param_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sim_params_record.keys()))
                writer.writeheader()
                writer.writerow(sim_params_record)

            plans.append({
                "ok":       True,
                "ds":       ds,
                "save_dir": save_dir,
                "n":        n,
                "m":        m,
                "density":  density,
                "r0":       r0,
                "U0":       U0,
            })
        except Exception as e:
            phase_a_failed.append(ds.id)
            try:
                os.makedirs(out_root, exist_ok=True)
                with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                    fh.write(str(e) + "\n")
            except Exception:
                pass
            fail_records.append(_failed_trajectory_record(eval_id, ds.id, G))

    if phase_a_failed:
        return [], [], fail_records, f"param resolution failed for {phase_a_failed}"

    job_specs = []
    for plan in plans:
        ds = plan["ds"]
        run_kwargs = {
            "density": float(plan["density"]),
            "U_0":     float(plan["U0"]),
            "r0":      float(plan["r0"]),
            "n":       float(plan["n"]),
            "m":       float(plan["m"]),
        }
        for k, v in (sim_defaults or {}).items():
            run_kwargs.setdefault(k, v)

        job_specs.append({
            "name":  f"i{eval_id:02d}_d{int(ds.id[1:]):02d}",
            "ds_id": ds.id,
            "worker_config": {
                "outdir":     plan["save_dir"],
                "run_kwargs": run_kwargs,
                "scattering": {"method": scattering_method,
                               "kwargs": dict(scattering_kwargs or {})},
                "loss": {
                    "exp_path":          str(ds.exp_path),
                    "trim_tail":         int(trim_tail),
                    "datatype":          getattr(ds, "datatype", "sq"),
                    "ffpath":            ffpath,
                    "metric":            metric,
                    "scattering_method": scattering_method,
                    "compare_q_range": (
                        list(compare_q_range) if compare_q_range is not None else None
                    ),
                    "q_min":             0.02,
                    "q_max":             0.03,
                    "dp_coeff":          dp_coeff,
                    "plot_apdist":       plot_apdist,
                    "metric_kwargs":     dict(metric_kwargs or {}),
                },
            },
        })

    return job_specs, plans, [], None


def _collect_parallel_eval_loss(
    *,
    eval_id: int,
    G: Dict[str, Any],
    plans: List[Dict[str, Any]],
    finished_jobs: List[Any],
    out_root: str,
) -> Tuple[float, bool, List[Dict[str, Any]]]:
    """
    Collect per-dataset loss from finished Slurm jobs for one eval.

    Returns ``(total_loss, success, iteration_records)``.
    """
    finished_by_ds = {j.ds_id: j for j in finished_jobs}
    total_loss = 0.0
    eval_failed = False
    iteration_records: List[Dict[str, Any]] = []

    for plan in plans:
        ds = plan["ds"]
        job = finished_by_ds.get(ds.id)
        if job is not None and job.done_status == "DONE":
            loss = None
            try:
                done_data = json.loads((Path(job.sim_dir) / "DONE").read_text())
                loss = float(done_data["loss"])
            except Exception as e:
                eval_failed = True
                try:
                    with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                        fh.write(f"DONE file unreadable: {e}\n")
                except Exception:
                    pass
                iteration_records.append(
                    _failed_trajectory_record(eval_id, ds.id, G, plan=plan)
                )
                continue

            total_loss += ds.weight * loss
            iteration_records.append({
                "iteration": eval_id,
                "dataset_id": ds.id,
                "loss": loss,
                "k": G.get("k", ""),
                "alpha": G.get("alpha", ""),
                "A": G.get("A", ""),
                "mu_c": G.get("mu_c", ""),
                "mu_b": G.get("mu_b", ""),
                "sigma_c": G.get("sigma_c", ""),
                "sigma_b": G.get("sigma_b", ""),
                "K_s": G.get("K_s", ""),
                "density": float(plan["density"]),
                "n":       float(plan["n"]),
                "m":       float(plan["m"]),
                "r0":      float(plan["r0"]),
                "U0":      float(plan["U0"]),
                **_audit_fields(done_data.get("result"), plan["r0"], str(job.sim_dir)),
            })
        elif job is not None and job.done_status == "METRIC_FAILED":
            eval_failed = True
            try:
                detail = json.loads((Path(job.sim_dir) / "METRIC_FAILED").read_text())
                msg = str(detail.get("reason", ""))
            except Exception as e:
                msg = f"METRIC_FAILED file unreadable: {e}"
            try:
                with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                    fh.write(f"eval {eval_id} METRIC_FAILED: {msg}\n")
            except Exception:
                pass
            iteration_records.append(
                _failed_trajectory_record(eval_id, ds.id, G, plan=plan, reason="METRIC_FAILED")
            )
        else:
            eval_failed = True
            reason = f"parallel job status={getattr(job, 'done_status', None)}"
            try:
                with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                    fh.write(reason + "\n")
            except Exception:
                pass
            iteration_records.append(
                _failed_trajectory_record(eval_id, ds.id, G, plan=plan)
            )

    return total_loss, not eval_failed, iteration_records


def _run_objective_parallel(
    datasets: List[Any],
    eval_id: int,
    G: Dict[str, Any],
    L: Dict[str, Any],
    out_root: str,
    ffpath: str,
    trim_tail: int,
    sim_defaults: Dict[str, Any],
    mode: str,
    scattering_method: str,
    scattering_kwargs: Dict[str, Any],
    metric: str,
    compare_q_range: Optional[Tuple[float, float]],
    dp_coeff: float,
    plot_apdist: bool,
    parallel_cfg: Dict[str, Any],
    iteration_data: List[Dict[str, Any]],
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Parallel analogue of the per-dataset sequential loop in `objective`.

    Submits one Slurm GPU job per dataset via `parallel.submit_jobs`, waits
    for all of them to reach a terminal state, and collects each job's
    loss from its `DONE` flag file. Mutates `iteration_data` in place so
    the caller can write the trajectory CSV block exactly as in the
    sequential path.
    """
    from parallel import submit_jobs_with_retry  # lazy import

    job_specs, plans, fail_records, fail_reason = _parallel_prepare_eval_jobs(
        datasets=datasets,
        eval_id=eval_id,
        G=G,
        L=L,
        out_root=out_root,
        trim_tail=trim_tail,
        sim_defaults=sim_defaults,
        mode=mode,
        scattering_method=scattering_method,
        scattering_kwargs=scattering_kwargs,
        metric=metric,
        compare_q_range=compare_q_range,
        dp_coeff=dp_coeff,
        plot_apdist=plot_apdist,
        ffpath=ffpath,
        metric_kwargs=metric_kwargs,
    )

    if fail_reason is not None:
        iteration_data.extend(fail_records)
        _write_failed_iteration_block(
            os.path.join(out_root, "bo_trajectory.csv"),
            eval_id,
            iteration_data,
            reason=fail_reason,
        )
        iteration_data.clear()
        raise EvaluationFailed(fail_reason)

    finished_jobs: List[Any] = []
    if job_specs:
        pcfg = parallel_cfg or {}
        cfg = _launcher_config_from_pcfg(
            pcfg, Path(out_root) / f"eval_{eval_id:03d}" / "_jobs",
        )
        finished_jobs = submit_jobs_with_retry(
            job_specs,
            cfg,
            max_job_retries=int(pcfg.get("max_job_retries", DEFAULT_MAX_JOB_RETRIES)),
            clean=True,
            poll_interval=pcfg.get("poll_interval"),
            max_wait=pcfg.get("max_wait"),
        )

    total_loss, success, records = _collect_parallel_eval_loss(
        eval_id=eval_id,
        G=G,
        plans=plans,
        finished_jobs=finished_jobs,
        out_root=out_root,
    )
    iteration_data.extend(records)

    if not success:
        reason = f"one or more parallel jobs failed for eval_id={eval_id}"
        _write_failed_iteration_block(
            os.path.join(out_root, "bo_trajectory.csv"),
            eval_id,
            iteration_data,
            reason=reason,
        )
        iteration_data.clear()
        raise EvaluationFailed(reason)

    return total_loss


N_GRID_RTOL = 0.05


def _check_target_n_grid(
    datasets: List[Any],
    scattering_kwargs: Optional[Dict[str, Any]],
    trim_tail: int,
    rtol: float = N_GRID_RTOL,
) -> None:
    """
    Raise if a saxs-fft S(q) target was made with a different N_grid than the
    simulation side will use. Targets without a uniform q grid are skipped.
    """
    n_grid = int((scattering_kwargs or {}).get("N_grid", DEFAULT_N_GRID))
    for ds in datasets:
        if getattr(ds, "datatype", "sq") != "sq":
            continue
        n_est = estimate_saxsfft_n_grid(ds.load_exp_curve(trim_tail=trim_tail))
        if n_est is None:
            print(
                f"[bo] N_grid check skipped for {ds.id}: target q grid is not "
                "uniform (not a saxs-fft curve)"
            )
            continue
        if abs(n_est - n_grid) > rtol * n_grid:
            raise ValueError(
                f"Dataset {ds.id}: target S(q) looks like saxs-fft N_grid~{n_est:.0f} "
                f"but the simulation side uses N_grid={n_grid}. "
                "Set scattering_kwargs['N_grid'] to the target's value."
            )
        print(f"[bo] N_grid check {ds.id}: target ~{n_est:.0f}, simulation {n_grid}")


def make_global_objective(
    datasets: List[Any],
    ps: ParamSpace,
    ffpath: str,
    out_root: str = "Optimization_Results",
    trim_tail: int = 200,
    sim_defaults: Dict[str, Any] = None,
    mode: str = "map",
    scattering_method: str = "saxsfft",
    scattering_kwargs: Dict[str, Any] = None,
    metric: str = "mse",
    compare_q_range: Optional[Tuple[float, float]] = (0.003, 0.06),
    dp_coeff: float = 0.5,
    plot_apdist: bool = True,
    parallel: bool = False,
    parallel_cfg: Dict[str, Any] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Create an objective(x_unit) that:
      - unpacks GLOBAL and LOCAL parameters from x_unit,
      - runs sim → SAXS → compare_to_exp on each dataset,
      - returns the weighted sum of losses.

    Policy / defaults
    -----------------
    - density = alpha * dataset.exp.theoretical_base (computed via dataset.rho_N(alpha))
    - n, m: taken from GLOBALs if present; otherwise fall back to dataset.sim.n / m.
    - r0:
        * if LOCAL "r0" present for a dataset, use it,
        * else if GLOBAL "k" present, use dataset.r0_sigma(k),
        * else if dataset.sim.r0 is set, use it,
        * else raise.
    - U0:
        * if LOCAL "U0" present for a dataset, use it,
        * else if GLOBAL ("A","mu_c","sigma_c") present, use dataset.U0_from_gaussian(...),
        * else if dataset.sim.U0 is set, use it,
        * else raise.
    mode: default to be "map"
    ----
    "map" (default):
        density, r0, U0 are computed via dataset mappings
        (alpha → density, k → r0, A/mu_c/sigma_c/K_s → U0).
        Direct sim params (density/r0/U0) are not allowed in ParamSpace.
    "sim":
        density, r0, U0 are taken directly from ParamSpace (global/local),
        or fall back to dataset.sim.* if not optimized.
        Mapping params (alpha/k/A/...) are not allowed in ParamSpace.

    "trim_tail":
        number of points to drop from the end of the experimental intensity 
            curve returned by Dataset.load_exp_curve.
        If exp_path already points to a processed S(q), set trim_tail=0.
    "compare_q_range":
        q-range used for the final saxsfft loss comparison. This is distinct
        from q_min/q_max used when extracting experimental S(q) from intensity.
        For ``metric='shift_rmse'`` pass None to compare over the curve
        overlap (the default (0.003, 0.06) is a fixed window).
    "dp_coeff":
        Phase-distance weight for ``metric='apdist'`` (see ``metrics.compare_saxs_curves``).
        Default 0.5. Ignored when ``metric='mse'``.
    "plot_apdist":
        When True and ``metric='apdist'``, save phase-warp diagnostic plots under
        ``eval_XXX/<dataset_id>/apdist_plots/``. Default True.
    "metric" / "metric_kwargs":
        ``metric`` is ``'mse'``, ``'apdist'`` or ``'shift_rmse'`` (saxsfft only).
        ``metric_kwargs`` holds the ``shift_rmse`` parameters (peak search
        range, prominence_frac, dispersed_delta, dispersed_shift, asymptote
        band, n_points, lambda_shift, overlap_trim, no_peak, asymptote
        fallback rules; see ``metrics.shift_rmse_params``); they are
        validated here and passed unchanged to both execution paths. Each
        eval writes ``shift_rmse_diagnostics.json``. A ``metrics.MetricFailed``
        fails the evaluation without rerunning the simulation.
    "scattering_kwargs":
        Passed to ``convert_to_SAXS_fft`` / ``convert_to_SAXS``. In
        ``mode='sim'`` with saxs-fft, each ``datatype='sq'`` target is checked
        once here: the N_grid inferred from its q grid must match
        ``scattering_kwargs['N_grid']`` (default 600) within 5 %, otherwise
        ``ValueError`` is raised before any simulation runs.
    Failed evaluations (after GPU job retry) raise ``EvaluationFailed``; they are
    logged to ``bo_trajectory.csv`` but not fed to the GP. ``run_bo`` re-acquires
    a new candidate instead.
    """
    sim_defaults = {} if sim_defaults is None else dict(sim_defaults)

    # Ensure the ParamSpace is consistent with the chosen mode
    _validate_param_mode(ps, mode)
    if mode == "sim" and scattering_method == "saxsfft":
        _check_target_n_grid(datasets, scattering_kwargs, trim_tail)
    metric_kwargs = dict(metric_kwargs or {})
    if metric == "shift_rmse":
        if scattering_method != "saxsfft":
            raise ValueError("metric='shift_rmse' requires scattering_method='saxsfft'")
        params = shift_rmse_params(metric_kwargs)
        if compare_q_range is None and params["overlap_trim"] is None:
            raise ValueError(
                "metric='shift_rmse' with compare_q_range=None needs "
                "metric_kwargs['overlap_trim']"
            )
    elif metric_kwargs:
        raise ValueError(f"metric_kwargs are only used by metric='shift_rmse' (got {metric!r})")

    def objective(x_unit: torch.Tensor, ffpath: str) -> torch.Tensor:
        objective._eval_failed = False
        # assign a unique id to this BO evaluation
        # ffpath: path to polydispersed sphere formfactor
        if not hasattr(objective, "_eval_id"):
            objective._eval_id = 0
        if not hasattr(objective, "_iteration_data"):
            objective._iteration_data = []  # Track data for global CSV
        eval_id = objective._eval_id
        objective._eval_id += 1

        # x_unit: (1,d) or (d,)
        x_unit = x_unit.reshape(-1)
        # 1) map [0,1] → physical
        x_phys = ps.unit_to_phys(x_unit)
        # 2) decode into globals/locals
        decoded = ps.decode(x_phys)
        G = decoded["global"]
        L = decoded["local"]

        total_loss = 0.0

        # --- Parallel fast path: submit one sbatch GPU job per dataset ---
        # All N simulations (+ SAXS conversion + loss computation) run
        # concurrently on separate GPUs via DNA_digitwin/parallel/. The
        # trajectory CSV block below is shared with the sequential path.
        if parallel:
            try:
                total_loss = _run_objective_parallel(
                    datasets=datasets,
                    eval_id=eval_id,
                    G=G,
                    L=L,
                    out_root=out_root,
                    ffpath=ffpath,
                    trim_tail=trim_tail,
                    sim_defaults=sim_defaults,
                    mode=mode,
                    scattering_method=scattering_method,
                    scattering_kwargs=scattering_kwargs,
                    metric=metric,
                    compare_q_range=compare_q_range,
                    dp_coeff=dp_coeff,
                    plot_apdist=plot_apdist,
                    parallel_cfg=parallel_cfg or {},
                    iteration_data=objective._iteration_data,
                    metric_kwargs=metric_kwargs,
                )
            except EvaluationFailed as exc:
                objective._eval_failed = True
                print(f"[bo] evaluation {eval_id} failed: {exc}")
                raise

            if len(objective._iteration_data) > 0:
                trajectory_path = os.path.join(out_root, "bo_trajectory.csv")
                _write_iteration_block(
                    filepath=trajectory_path,
                    iteration=eval_id,
                    total_loss=float(total_loss),
                    records=objective._iteration_data,
                )
                _append_loss_components(
                    out_root, eval_id, float(total_loss), objective._iteration_data,
                )
                objective._iteration_data = []

            return torch.tensor([[total_loss]], dtype=torch.float64)

        # --- Sequential path (original behavior, unchanged) ---
        eval_failed = False
        for ds in datasets:
            try:
                # ---- Shared n, m (always "sim" style) ----
                n = float(G["n"]) if "n" in G else float(ds.sim.n)
                m = float(G["m"]) if "m" in G else float(ds.sim.m)

                # ---- density ----
                if mode == "map":
                    alpha = float(G.get("alpha", 1.0))
                    density = ds.rho_N(alpha=alpha)
                else:  # mode == "sim"
                    # Prefer local, then global, then dataset.sim
                    if "density" in L[ds.id]:
                        density = float(L[ds.id]["density"])
                    elif "density" in G:
                        density = float(G["density"])
                    elif ds.sim.density is not None:
                        density = float(ds.sim.density)
                    else:
                        raise ValueError(
                            f"Dataset {ds.id}: density not provided in mode='sim' "
                            "and dataset.sim.density is None."
                        )

                # ---- r0 ----
                if mode == "map":
                    if "k" in G:
                        r0 = float(ds.r0_sigma(k=float(G["k"])))
                    elif ds.sim.r0 is not None:
                        r0 = float(ds.sim.r0)
                    else:
                        raise ValueError(
                            f"Dataset {ds.id}: r0 not provided and no mapping coeff 'k' found "
                            "in mode='map'."
                        )
                else:  # mode == "sim"
                    if "r0" in L[ds.id]:
                        r0 = float(L[ds.id]["r0"])
                    elif "r0" in G:
                        r0 = float(G["r0"])
                    elif ds.sim.r0 is not None:
                        r0 = float(ds.sim.r0)
                    else:
                        raise ValueError(
                            f"Dataset {ds.id}: r0 not provided in mode='sim' "
                            "and dataset.sim.r0 is None."
                        )

                # ---- U0 ----
                if mode == "map":
                    if all(k in G for k in ("A", "mu_c", "sigma_c")):
                        U0 = float(
                            ds.U0_from_gaussian(
                                A=G["A"],
                                mu_c=G["mu_c"],
                                sigma_c=G["sigma_c"],
                                K_s=G.get("K_s", 0.05),
                            )
                        )
                    elif ds.sim.U0 is not None:
                        U0 = float(ds.sim.U0)
                    else:
                        raise ValueError(
                            f"Dataset {ds.id}: U0 not provided and no global Gaussian coeffs "
                            "found in mode='map'."
                        )
                else:  # mode == "sim"
                    if "U0" in L[ds.id]:
                        U0 = float(L[ds.id]["U0"])
                    elif "U0" in G:
                        U0 = float(G["U0"])
                    elif ds.sim.U0 is not None:
                        U0 = float(ds.sim.U0)
                    else:
                        raise ValueError(
                            f"Dataset {ds.id}: U0 not provided in mode='sim' "
                            "and dataset.sim.U0 is None."
                        )

                # ---- Output directory ----
                # New structure: eval_XXX/d0/, eval_XXX/d1/, etc.
                # (Groups all datasets from same iteration together)
                save_dir = os.path.join(out_root, f"eval_{eval_id:03d}", ds.id)
                os.makedirs(save_dir, exist_ok=True)
                # ---- save sim params for this eval + dataset ----
                sim_params_record = {
                    "dataset_id": ds.id,
                    "eval_id": eval_id,
                    # Mapping coefficients (saved regardless of mode)
                    "k": G.get("k", ""),
                    "alpha": G.get("alpha", ""),
                    "A": G.get("A", ""),
                    "mu_c": G.get("mu_c", ""),
                    "mu_b": G.get("mu_b", ""),
                    "sigma_c": G.get("sigma_c", ""),
                    "sigma_b": G.get("sigma_b", ""),
                    "K_s": G.get("K_s", ""),
                    # Simulation parameters
                    "density": float(density),
                    "n": float(n),
                    "m": float(m),
                    "r0": float(r0),
                    "U0": float(U0),
                }
                param_path = os.path.join(save_dir, f"sim_params_{ds.id}.csv")
                with open(param_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(sim_params_record.keys()))
                    writer.writeheader()
                    writer.writerow(sim_params_record)
                # ---- 1) Simulation ----
                sim_result = run_simulation(
                    density=density, U_0=U0, r0=r0, n=n, m=m, outdir=save_dir, **sim_defaults
                )

                # ---- 2) Sim → S(q) ----
                _sc_kw = dict(scattering_kwargs) if scattering_kwargs else {}
                if scattering_method == "saxsfft":
                    convert_to_SAXS_fft(save_dir, **_sc_kw)
                else:
                    convert_to_SAXS(save_dir, **_sc_kw)

                # ---- 3) Compare to experiment ----
                cand_paths = [
                    os.path.join(save_dir, "S(q)_data", "average_structure_factor.npy"),
                    os.path.join(save_dir, "scattering_data", "average_structure_factor.npy"),
                    os.path.join(save_dir, "S(q)_", "average_structure_factor.npy"),
                    os.path.join(save_dir, "S(q)", "average_structure_factor.npy"),
                ]
                sim_sq_path = next((p for p in cand_paths if os.path.exists(p)), None)
                if sim_sq_path is None:
                    raise FileNotFoundError(f"Missing S(q): tried {cand_paths}")
                sim_sq = np.load(sim_sq_path)

                exp_data = ds.load_exp_curve(trim_tail=trim_tail)
                if getattr(ds, "datatype", "sq") == "sq":
                    exp_sq = exp_data
                else:
                    exp_sq = extract_exp_sq(
                        exp_scattering=exp_data,
                        ffpath=ffpath,
                        q_min=0.02,
                        q_max=0.03,
                        normalize=False)
                if scattering_method == "saxsfft":
                    loss = float(compare_to_exp_saxsfft(
                        exp_sq,
                        sim_sq,
                        save_dir,
                        metric=metric,
                        q_range=compare_q_range,
                        dp_coeff=dp_coeff,
                        plot_apdist=plot_apdist,
                        metric_kwargs=metric_kwargs,
                    ))
                else:
                    loss = float(compare_to_exp(
                        exp_sq,
                        sim_sq,
                        save_dir,
                        metric=metric,
                        dp_coeff=dp_coeff,
                        plot_apdist=plot_apdist,
                    ))
                total_loss += ds.weight * loss

                # Store data for global trajectory CSV
                trajectory_record = {
                    "iteration": eval_id,
                    "dataset_id": ds.id,
                    "loss": loss,
                    "k": G.get("k", ""),
                    "alpha": G.get("alpha", ""),
                    "A": G.get("A", ""),
                    "mu_c": G.get("mu_c", ""),
                    "mu_b": G.get("mu_b", ""),
                    "sigma_c": G.get("sigma_c", ""),
                    "sigma_b": G.get("sigma_b", ""),
                    "K_s": G.get("K_s", ""),
                    "density": float(density),
                    "n": float(n),
                    "m": float(m),
                    "r0": float(r0),
                    "U0": float(U0),
                    **_audit_fields(sim_result, r0, save_dir),
                }
                objective._iteration_data.append(trajectory_record)

            except Exception as e:
                eval_failed = True
                reason = "METRIC_FAILED" if isinstance(e, MetricFailed) else "FAILED"
                try:
                    with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                        fh.write(f"eval {eval_id} {reason}: {e}\n")
                except Exception:
                    pass

                trajectory_record = _failed_trajectory_record(eval_id, ds.id, G, reason=reason)
                objective._iteration_data.append(trajectory_record)

        if eval_failed:
            reason = f"sequential evaluation failed for eval_id={eval_id}"
            if len(objective._iteration_data) > 0:
                _write_failed_iteration_block(
                    os.path.join(out_root, "bo_trajectory.csv"),
                    eval_id,
                    objective._iteration_data,
                    reason=reason,
                )
                objective._iteration_data = []
            objective._eval_failed = True
            raise EvaluationFailed(reason)

        # Write iteration block to global trajectory CSV
        if len(objective._iteration_data) > 0:
            trajectory_path = os.path.join(out_root, "bo_trajectory.csv")
            _write_iteration_block(
                filepath=trajectory_path,
                iteration=eval_id,
                total_loss=float(total_loss),
                records=objective._iteration_data
            )
            _append_loss_components(
                out_root, eval_id, float(total_loss), objective._iteration_data,
            )
            # Reset for next iteration
            objective._iteration_data = []

        # Return as a 1-element tensor (BoTorch expects a tensor)
        return torch.tensor([[total_loss]], dtype=torch.float64)

    objective._eval_failed = False
    return objective


# ------------------------- Warm start from trajectory ------------------------- #

def remaining_bo_iters(n_iters: int, n_successful: int) -> int:
    """
    Acquisition steps left when resuming BO.

    Cold start runs one initial eval plus ``n_iters`` acquisitions (``n_iters + 1``
    evaluations total). ``n_successful`` is the number of successful evaluations
    already recorded (including the initial point).
    """
    if n_successful <= 0:
        return n_iters
    return max(0, n_iters - (n_successful - 1))


def load_warm_start_from_trajectory(
    trajectory_path: os.PathLike,
    ps: ParamSpace,
    *,
    loss_penalty_threshold: float = 1e8,
    m_rel: bool = False,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
    """
    Build ``run_bo`` warm-start tensors from ``bo_trajectory.csv``.

    Returns ``(train_x, train_y, next_eval_id)`` or ``None`` if no successful
    iterations are found. Failed blocks (penalty loss, ERROR rows, EVALUATION
    FAILED headers) are skipped for the GP but still advance ``next_eval_id`` so
    eval folder indices do not collide with prior attempts.

    Parameters
    ----------
    m_rel
        If True, reconstruct ``m_rel`` from trajectory ``m`` and ``n`` via
        ``m_rel = (m - 3) / (n - 4)`` (``ParamSpaceConstrainedNM`` convention).
    """
    path = Path(trajectory_path)
    if not path.is_file():
        return None

    successful: List[Tuple[int, float, Dict[str, str]]] = []
    max_eval_id = -1
    pending_iter: Optional[int] = None
    pending_loss: Optional[float] = None
    header: Optional[Dict[str, int]] = None

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                pending_iter = None
                pending_loss = None
                header = None
                continue

            if row[0].startswith("# Iteration"):
                try:
                    iter_num = int(row[0].split()[-1])
                except ValueError:
                    pending_iter = None
                    pending_loss = None
                    continue

                max_eval_id = max(max_eval_id, iter_num)
                pending_iter = None
                pending_loss = None

                if len(row) >= 3 and row[1] == "total_loss":
                    try:
                        total_loss = float(row[2])
                    except ValueError:
                        continue
                    if total_loss >= loss_penalty_threshold:
                        continue
                    pending_iter = iter_num
                    pending_loss = total_loss
                continue

            if row[0] == "iteration" and len(row) > 1 and row[1] == "dataset_id":
                header = {name: idx for idx, name in enumerate(row)}
                continue

            if pending_iter is None or pending_loss is None or header is None:
                continue

            if str(row[0]) != str(pending_iter):
                continue

            rec = {name: row[idx] for name, idx in header.items() if idx < len(row)}
            if rec.get("loss") in ("ERROR", "FAILED") or rec.get("n") == "ERROR":
                pending_iter = None
                pending_loss = None
                continue

            successful.append((pending_iter, pending_loss, rec))
            pending_iter = None
            pending_loss = None

    if not successful:
        return None

    successful.sort(key=lambda item: item[0])
    x_rows: List[torch.Tensor] = []
    y_rows: List[float] = []

    for _iter_num, total_loss, rec in successful:
        n_val = float(rec["n"])
        m_val = float(rec["m"])
        phys_vals: List[float] = []
        for name in ps._names:
            if name == "m_rel":
                if not m_rel:
                    raise ValueError(
                        "ParamSpace uses m_rel but load_warm_start_from_trajectory "
                        "was called with m_rel=False"
                    )
                span = n_val - 4.0
                if span <= 0.0:
                    raise ValueError(
                        f"Cannot reconstruct m_rel from trajectory n={n_val}"
                    )
                phys_vals.append((m_val - 3.0) / span)
            else:
                phys_vals.append(float(rec[name]))

        x_phys = torch.tensor(phys_vals, dtype=torch.float64)
        x_rows.append(ps.phys_to_unit(x_phys))
        y_rows.append(-float(total_loss))

    train_x = torch.stack(x_rows)
    train_y = torch.tensor(y_rows, dtype=torch.float64).reshape(-1, 1)
    next_eval_id = max_eval_id + 1
    return train_x, train_y, next_eval_id


def run_bo_resumable(
    objective_fn,
    ps: ParamSpace,
    ffpath: str,
    out_root: str,
    n_iters: int = 20,
    seed: int = 0,
    *,
    m_rel: bool = False,
    max_acq_attempts: int = DEFAULT_MAX_ACQ_ATTEMPTS,
    gp_log_dir: Optional[str] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Run ``run_bo``, resuming from ``<out_root>/bo_trajectory.csv`` when present.

    Sets ``objective_fn._eval_id`` so the next eval folder index does not collide
    with prior attempts (including failed iterations).

    Parameters
    ----------
    gp_log_dir
        Forwarded to ``run_bo``. When the trajectory already contains every
        requested iteration, ``run_bo`` is not called and nothing new is written.
    """
    trajectory_path = os.path.join(out_root, "bo_trajectory.csv")
    warm = load_warm_start_from_trajectory(trajectory_path, ps, m_rel=m_rel)

    warm_start = None
    remaining = n_iters
    if warm is not None:
        train_x, train_y, next_eval_id = warm
        warm_start = (train_x, train_y)
        objective_fn._eval_id = next_eval_id
        remaining = remaining_bo_iters(n_iters, train_x.shape[0])
        print(
            f"[warm start] loaded {train_x.shape[0]} successful eval(s) from "
            f"{trajectory_path}"
        )
        print(
            f"[warm start] next eval_id={next_eval_id}, "
            f"remaining BO iterations={remaining}"
        )

        if remaining == 0:
            y = train_y.squeeze(-1).detach().cpu().numpy()
            history = [-float(v) for v in y]
            best_idx = int(np.argmin(history))
            best_x_phys = ps.unit_to_phys(train_x[best_idx])
            print("[warm start] target iteration count already reached; skipping run_bo")
            return best_x_phys, history

    return run_bo(
        objective_fn=objective_fn,
        ps=ps,
        ffpath=ffpath,
        n_iters=remaining,
        seed=seed,
        warm_start=warm_start,
        max_acq_attempts=max_acq_attempts,
        gp_log_dir=gp_log_dir,
    )


# ------------------------- GP state log ------------------------- #
#
# Optional record of every GP that ``run_bo`` actually fits. Nothing here is
# fed back into acquisition. ``_PreserveRng`` puts the torch / numpy / random
# generators back, so a posterior evaluation or the extra final refit cannot
# change the candidate sequence.

GP_LOG_VERSION = 1

GP_LOG_UNITS = (
    "lengthscale: unit cube, columns follow param_names "
    "(ParamSpace order: globals, then locals as name:dataset_id). "
    "SAAS lengthscale, outputscale, noise, and mean_constant list every "
    "retained MCMC sample; they are not a summary. "
    "outputscale, noise, mean_constant: Standardize z-scored train_y space. "
    "standardize_mean and standardize_std: Standardize buffers of train_y=-loss. "
    "post_*_loss: latent posterior (observation_noise=False); "
    "mean_loss=-mean(train_y), std_loss=std(train_y)."
)


class _PreserveRng:
    """Restore the global RNGs on the way out, including when the block raises."""

    def __enter__(self):
        self._torch = torch.random.get_rng_state()
        self._numpy = np.random.get_state()
        self._py = random.getstate()
        # Do not call into CUDA unless this process already did. Initializing
        # it here would be a side effect the unlogged run does not have.
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            self._cuda = torch.cuda.get_rng_state_all()
        else:
            self._cuda = None
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.random.set_rng_state(self._torch)
        np.random.set_state(self._numpy)
        random.setstate(self._py)
        if self._cuda is not None:
            torch.cuda.set_rng_state_all(self._cuda)
        return False


def _align_lengthscale(lengthscale: torch.Tensor, n_dims: int) -> torch.Tensor:
    """Drop gpytorch's singleton output axis.

    One GP is ``(d,)`` or ``(1, d)``. A SAAS MCMC batch is ``(S, 1, d)``.
    The last axis is the unit-cube dimension, in ``param_names`` order.
    """
    ls = lengthscale.detach()
    if ls.ndim == 0 or ls.shape[-1] != n_dims:
        raise RuntimeError(
            f"ARD lengthscale shape {tuple(ls.shape)} does not end with d={n_dims}"
        )
    while ls.ndim > 2 and ls.shape[-2] == 1:
        ls = ls.squeeze(-2)
    return ls


def _json_hyper(values: torch.Tensor, n_mcmc: Optional[int], name: str):
    """Scalar for STGP, one JSON number per MCMC sample for SAAS."""
    flat = values.detach().reshape(-1).cpu()
    if n_mcmc is None:
        if flat.numel() != 1:
            raise RuntimeError(
                f"{name} shape {tuple(values.shape)} is not a scalar STGP hyperparameter"
            )
        return float(flat[0].item())
    if flat.numel() != n_mcmc:
        raise RuntimeError(
            f"{name} has {flat.numel()} values, expected {n_mcmc} MCMC samples "
            f"(shape {tuple(values.shape)})"
        )
    return [float(v) for v in flat.tolist()]


def _standardize_stats(model) -> Tuple[float, float]:
    ot = getattr(model, "outcome_transform", None)
    if ot is None:
        raise RuntimeError("GP has no outcome_transform; expected Standardize(m=1)")
    means = getattr(ot, "means", None)
    stdvs = getattr(ot, "stdvs", None)
    if means is None or stdvs is None:
        raise RuntimeError("Standardize buffers 'means' and 'stdvs' are missing")
    if means.numel() != 1 or stdvs.numel() != 1:
        raise RuntimeError(
            f"expected scalar Standardize stats, got means {tuple(means.shape)} "
            f"stdvs {tuple(stdvs.shape)}"
        )
    return float(means.reshape(-1)[0].item()), float(stdvs.reshape(-1)[0].item())


def _as_vector(values: torch.Tensor, n: int) -> torch.Tensor:
    flat = values.reshape(-1)
    if flat.numel() != n:
        raise RuntimeError(
            f"expected {n} posterior values, got shape {tuple(values.shape)}"
        )
    return flat


def _as_cov(cov: torch.Tensor, n: int) -> torch.Tensor:
    c = cov
    while c.ndim > 2 and c.shape[0] == 1:
        c = c[0]
    if tuple(c.shape) != (n, n):
        raise RuntimeError(
            f"posterior covariance shape {tuple(cov.shape)} is not ({n}, {n})"
        )
    return c


def _mixture_covariance(posterior) -> torch.Tensor:
    """Posterior covariance of a fully Bayesian mixture.

    Uses ``mixture_covariance_matrix`` when the installed botorch has it
    (newer than 0.9). Otherwise applies the same law of total covariance to
    ``posterior.distribution`` (or ``.mvn``).
    """
    cov = getattr(posterior, "mixture_covariance_matrix", None)
    if cov is not None:
        return cov
    dist = getattr(posterior, "distribution", None)
    if dist is None:
        dist = posterior.mvn
    cov_s = dist.covariance_matrix
    if hasattr(cov_s, "to_dense"):
        cov_s = cov_s.to_dense()
    mean_s = dist.mean
    if (
        mean_s.ndim >= 1
        and mean_s.shape[-1] == 1
        and cov_s.shape[-1] != 1
    ):
        mean_s = mean_s.squeeze(-1)
    q = cov_s.shape[-1]
    if mean_s.shape[-1] != q:
        raise RuntimeError(
            f"cannot align mixture mean {tuple(mean_s.shape)} "
            f"with covariance {tuple(cov_s.shape)}"
        )
    mean_flat = mean_s.reshape(-1, q)
    cov_flat = cov_s.reshape(-1, q, q)
    if mean_flat.shape[0] != cov_flat.shape[0]:
        raise RuntimeError(
            f"MCMC batch mismatch: mean {tuple(mean_s.shape)} "
            f"cov {tuple(cov_s.shape)}"
        )
    centered = mean_flat - mean_flat.mean(dim=0, keepdim=True)
    within = cov_flat.mean(dim=0)
    between = torch.matmul(centered.unsqueeze(-1), centered.unsqueeze(-2)).mean(dim=0)
    return within + between


def _posterior_moments(posterior, n: int):
    """Latent moments in train_y units. SAAS uses the MCMC mixture."""
    if hasattr(posterior, "mixture_mean"):
        mean = posterior.mixture_mean
        var = posterior.mixture_variance
        cov = _mixture_covariance(posterior)
    else:
        mean = posterior.mean
        var = posterior.variance
        dist = getattr(posterior, "distribution", None)
        if dist is None:
            dist = posterior.mvn
        cov = dist.covariance_matrix
    if hasattr(cov, "to_dense"):
        cov = cov.to_dense()
    return _as_vector(mean, n), _as_vector(var, n), _as_cov(cov, n)


def gp_posterior(model, X_unit: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Latent posterior at unit-cube points, in loss units.

    ``train_y`` is ``-loss``. The mean's sign is flipped back; standard
    deviation and covariance are unchanged by that flip. Observation noise
    is not included (``observation_noise=False``), matching the posterior
    ``qLogExpectedImprovement`` reads.

    Parameters
    ----------
    model
        A GP returned by :func:`load_gp_model` (or any model with
        ``posterior``). Call this outside ``torch.no_grad`` when the mean
        must stay differentiable with respect to ``X_unit``.
    X_unit
        Shape ``(n, d)`` or ``(d,)``. Columns follow ``param_names``.

    Returns
    -------
    dict
        ``mean_loss`` ``(n,)``, ``std_loss`` ``(n,)``, ``cov_loss`` ``(n, n)``.
    """
    if X_unit.ndim == 1:
        X = X_unit.unsqueeze(0)
    elif X_unit.ndim == 2:
        X = X_unit
    else:
        raise ValueError(
            f"X_unit must have shape (d,) or (n, d), got {tuple(X_unit.shape)}"
        )
    train_inputs = getattr(model, "train_inputs", None)
    if train_inputs:
        ref = train_inputs[0]
        if X.dtype != ref.dtype or X.device != ref.device:
            X = X.to(dtype=ref.dtype, device=ref.device)
    model.eval()
    posterior = model.posterior(X, observation_noise=False)
    mean, var, cov = _posterior_moments(posterior, X.shape[-2])
    return {
        "mean_loss": -mean,
        "std_loss": var.clamp_min(0).sqrt(),
        "cov_loss": cov,
    }


def _read_gp_hypers(model, n_dims: int, surrogate: str) -> Dict[str, Any]:
    ls = _align_lengthscale(model.covar_module.base_kernel.lengthscale, n_dims)
    if surrogate == "stgp":
        if ls.ndim == 2 and ls.shape[0] == 1:
            ls = ls[0]
        if ls.ndim != 1 or ls.shape[0] != n_dims:
            raise RuntimeError(
                f"STGP lengthscale did not reduce to ({n_dims},); "
                f"raw shape {tuple(model.covar_module.base_kernel.lengthscale.shape)}"
            )
        lengthscale_json = [float(v) for v in ls.cpu().tolist()]
        n_mcmc = None
        mcmc_samples = None
    elif surrogate == "saas":
        if ls.ndim == 1:
            ls = ls.unsqueeze(0)
        if ls.ndim != 2 or ls.shape[1] != n_dims:
            raise RuntimeError(
                f"SAAS lengthscale did not reduce to (n_mcmc, {n_dims}); "
                f"raw shape {tuple(model.covar_module.base_kernel.lengthscale.shape)}"
            )
        lengthscale_json = [[float(v) for v in row] for row in ls.cpu().tolist()]
        n_mcmc = int(ls.shape[0])
        mcmc_samples = {
            "lengthscale": ls.detach().cpu().contiguous().clone(),
        }
    else:
        raise ValueError(f"surrogate must be 'stgp' or 'saas', got {surrogate!r}")

    outputscale = _json_hyper(model.covar_module.outputscale, n_mcmc, "outputscale")
    noise = _json_hyper(model.likelihood.noise, n_mcmc, "noise")
    mean_constant = _json_hyper(model.mean_module.constant, n_mcmc, "mean_constant")
    std_mean, std_std = _standardize_stats(model)
    if mcmc_samples is not None:
        mcmc_samples["outputscale"] = (
            model.covar_module.outputscale.detach().reshape(-1).cpu().clone()
        )
        mcmc_samples["noise"] = model.likelihood.noise.detach().reshape(-1).cpu().clone()
        mcmc_samples["mean"] = model.mean_module.constant.detach().reshape(-1).cpu().clone()
    return {
        "lengthscale": lengthscale_json,
        "outputscale": outputscale,
        "noise": noise,
        "mean_constant": mean_constant,
        "n_mcmc": n_mcmc,
        "standardize_mean": std_mean,
        "standardize_std": std_std,
        "mcmc_samples": mcmc_samples,
    }


def _cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if not torch.is_tensor(value):
            raise RuntimeError(f"state_dict[{key!r}] is not a tensor")
        out[key] = value.detach().cpu().clone()
    return out


def _cpu_row(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(dtype=torch.float64, device="cpu").reshape(-1)


def _allocate_state_path(log_dir: Path, n_train: int, stage: str) -> Tuple[Path, int]:
    """Pick a new file. Existing records are never replaced."""
    states = log_dir / "states"
    states.mkdir(parents=True, exist_ok=True)
    if stage == "final":
        path = states / f"ntrain_{n_train:04d}_final.pt"
        extra = 0
        while path.exists():
            extra += 1
            path = states / f"ntrain_{n_train:04d}_final_{extra}.pt"
        return path, extra
    if stage != "acquisition":
        raise ValueError(f"stage must be 'acquisition' or 'final', got {stage!r}")
    attempt = 0
    while True:
        path = states / f"ntrain_{n_train:04d}_attempt_{attempt:02d}.pt"
        if not path.exists():
            return path, attempt
        attempt += 1


def _report_gp_log_error(gp_log_dir) -> None:
    """A log failure must not stop BO or change the candidate that was chosen."""
    text = traceback.format_exc()
    print(f"[bo] GP log failed; the BO loop is continuing\n{text}")
    try:
        path = Path(gp_log_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "errors.txt", "a") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
    except Exception:
        traceback.print_exc()


def _package_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def _acq_to_float(acq_value) -> Optional[float]:
    if acq_value is None:
        return None
    if torch.is_tensor(acq_value):
        flat = acq_value.detach().reshape(-1)
        if flat.numel() != 1:
            raise RuntimeError(
                f"acquisition value shape {tuple(acq_value.shape)} is not scalar"
            )
        return float(flat[0].item())
    return float(acq_value)


def _write_gp_iteration(
    gp_log_dir,
    model,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    ps: ParamSpace,
    *,
    surrogate: str,
    stage: str,
    acq_value,
    candidate: Optional[torch.Tensor],
    next_eval_id: Optional[int],
) -> None:
    """Append one GP fit to ``gp_log.jsonl`` and write a new ``states/*.pt``.

    ``iteration`` is ``n_train``, the number of successful observations this
    GP was trained on. A later resume continues from the loaded count.
    Acquisition retries and a second final snapshot at the same ``n_train``
    get a new file (``attempt`` / ``final_k``) instead of overwriting.
    """
    log_dir = Path(gp_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    names = list(ps._names)
    n_train = int(train_x.shape[0])
    n_dims = int(train_x.shape[-1])
    if len(names) != n_dims:
        raise RuntimeError(
            f"ParamSpace has {len(names)} dimensions but train_x has {n_dims}"
        )

    with _PreserveRng():
        state_dict = _cpu_state_dict(model)
        hypers = _read_gp_hypers(model, n_dims, surrogate)
        best_index = int(torch.argmax(train_y))
        best_unit = _cpu_row(train_x[best_index])
        best_phys = ps.unit_to_phys(best_unit)
        points = []
        if candidate is not None:
            points.append(_cpu_row(candidate))
        points.append(best_unit)
        with torch.no_grad():
            post = gp_posterior(model, torch.stack(points, dim=0))

        if candidate is None:
            cand_unit = cand_phys = None
            cand_mean = cand_std = None
            best_slot = 0
        else:
            cand_unit = points[0].tolist()
            cand_phys = ps.unit_to_phys(points[0]).tolist()
            cand_mean = float(post["mean_loss"][0].item())
            cand_std = float(post["std_loss"][0].item())
            best_slot = 1

        state_path, attempt = _allocate_state_path(log_dir, n_train, stage)
        record = {
            "gp_log_version": GP_LOG_VERSION,
            "iteration": n_train,
            "n_train": n_train,
            "stage": stage,
            "attempt": attempt,
            "surrogate": surrogate,
            "param_names": names,
            "lengthscale": hypers["lengthscale"],
            "outputscale": hypers["outputscale"],
            "noise": hypers["noise"],
            "mean_constant": hypers["mean_constant"],
            "n_mcmc": hypers["n_mcmc"],
            "standardize_mean": hypers["standardize_mean"],
            "standardize_std": hypers["standardize_std"],
            "observation_noise": False,
            "acq_value": _acq_to_float(acq_value),
            "candidate_unit": cand_unit,
            "candidate_phys": cand_phys,
            "candidate_post_mean_loss": cand_mean,
            "candidate_post_std_loss": cand_std,
            "best_index": best_index,
            "best_unit": best_unit.tolist(),
            "best_phys": best_phys.tolist(),
            "best_observed_loss": -float(train_y.reshape(-1)[best_index].item()),
            "best_post_mean_loss": float(post["mean_loss"][best_slot].item()),
            "best_post_std_loss": float(post["std_loss"][best_slot].item()),
            "next_eval_id": None if next_eval_id is None else int(next_eval_id),
            "state_file": state_path.relative_to(log_dir).as_posix(),
            "units": GP_LOG_UNITS,
        }
        mcmc_samples = hypers["mcmc_samples"]
        bundle = {
            "gp_log_version": GP_LOG_VERSION,
            "state_dict": state_dict,
            "train_x": train_x.detach().to(dtype=torch.float64, device="cpu").clone(),
            "train_y": train_y.detach().to(dtype=torch.float64, device="cpu").clone(),
            "surrogate": surrogate,
            "outcome_transform": "Standardize",
            "outcome_transform_kwargs": {"m": 1},
            "param_names": names,
            "mcmc_samples": mcmc_samples,
            "botorch_version": _package_version("botorch"),
            "gpytorch_version": _package_version("gpytorch"),
        }
        torch.save(bundle, state_path)
        with open(log_dir / "gp_log.jsonl", "a") as fh:
            fh.write(json.dumps(record, allow_nan=False) + "\n")
            fh.flush()


def read_gp_log(gp_log_dir) -> List[dict]:
    """Read ``<gp_log_dir>/gp_log.jsonl`` in write order. Missing file → ``[]``."""
    path = Path(gp_log_dir) / "gp_log.jsonl"
    if not path.is_file():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_outcome_transform(model, state_dict: Dict[str, torch.Tensor]) -> None:
    prefix = "outcome_transform."
    sub = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not sub:
        return
    # Constructor already ran Standardize on this same train_y. If the saved
    # buffer names differ across botorch versions, keep that result.
    try:
        model.outcome_transform.load_state_dict(sub)
    except (RuntimeError, ValueError, KeyError):
        return


def _require_botorch(symbol: str):
    try:
        import botorch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{symbol} requires botorch, which is installed in the cluster venv "
            "and not in the local digitwin environment"
        ) from exc


def load_gp_model(state_path):
    """Rebuild a GP from a ``states/*.pt`` file written by ``run_bo``.

    Returns ``(model, bundle)``. ``model`` is in eval mode. For SAAS, the
    state dict is loaded when this botorch version can do that; otherwise
    the retained MCMC samples are passed to ``load_mcmc_samples``.

    The posterior at new points is :func:`gp_posterior` or
    :func:`load_gp_posterior`.
    """
    _require_botorch("load_gp_model")
    bundle = _torch_load(state_path)
    if bundle.get("outcome_transform") != "Standardize":
        raise ValueError(
            f"unsupported outcome_transform {bundle.get('outcome_transform')!r}"
        )
    if int(bundle.get("outcome_transform_kwargs", {}).get("m", 1)) != 1:
        raise ValueError("only Standardize(m=1) can be reloaded")
    train_x = bundle["train_x"].to(dtype=torch.float64, device="cpu")
    train_y = bundle["train_y"].to(dtype=torch.float64, device="cpu")
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)
    surrogate = bundle["surrogate"]
    if surrogate == "stgp":
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        model.load_state_dict(bundle["state_dict"])
    elif surrogate == "saas":
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
        from botorch.models.transforms.outcome import Standardize

        def _new():
            return SaasFullyBayesianSingleTaskGP(
                train_x, train_y, outcome_transform=Standardize(m=1)
            )

        model = _new()
        try:
            model.load_state_dict(bundle["state_dict"])
        except (AttributeError, KeyError, RuntimeError, ValueError) as exc:
            samples = bundle.get("mcmc_samples")
            if not samples:
                raise RuntimeError(
                    "SAAS state_dict did not load and the file has no mcmc_samples"
                ) from exc
            model = _new()
            model.load_mcmc_samples(samples)
            _load_outcome_transform(model, bundle["state_dict"])
        if getattr(model, "covar_module", None) is None:
            raise RuntimeError("SAAS reload left covar_module unset")
    else:
        raise ValueError(f"surrogate must be 'stgp' or 'saas', got {surrogate!r}")
    model.eval()
    return model, bundle


def load_gp_posterior(state_path, X_unit: torch.Tensor) -> Dict[str, Any]:
    """Load a saved GP and evaluate its latent posterior at ``X_unit``.

    Parameters
    ----------
    state_path
        A ``states/*.pt`` file from ``run_bo(..., gp_log_dir=...)``.
    X_unit
        Unit-cube points, shape ``(n, d)`` or ``(d,)``, column order
        ``param_names``.

    Returns
    -------
    dict
        ``mean_loss`` ``(n,)``, ``std_loss`` ``(n,)``, ``cov_loss`` ``(n, n)``
        in loss units. ``model`` is the rebuilt GP. ``param_names`` is the
        column order. Differentiate ``mean_loss`` w.r.t. ``X_unit`` for the
        posterior-mean gradient (do not wrap this call in ``torch.no_grad``).
    """
    model, bundle = load_gp_model(state_path)
    out: Dict[str, Any] = gp_posterior(model, X_unit)
    out["model"] = model
    out["param_names"] = list(bundle["param_names"])
    return out


# ------------------------- BO runner ------------------------- #

def run_bo(
    objective_fn,
    ps: ParamSpace,
    ffpath: str,
    n_iters: int = 20,
    seed: int = 0,
    warm_start: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    max_acq_attempts: int = DEFAULT_MAX_ACQ_ATTEMPTS,
    surrogate: str = "stgp",
    gp_log_dir: Optional[str] = None,
):
    """
    Run a BoTorch loop over the parameter space defined by ParamSpace.

    Minimizes the provided objective (sum of losses). Returns:
        best_x_phys (1D torch tensor), history (list of floats)

    Parameters
    ----------
    warm_start
        If given, skip the initial ``init_unit`` evaluation and start the GP
        from these tensors instead. Use this to resume after a crash without
        re-evaluating completed BO steps. Expected shapes: ``train_x`` is
        ``(n, d)`` in the **unit** cube (same convention as ``ParamSpace``),
        ``train_y`` is ``(n, 1)`` with values ``-total_loss`` (same sign as
        internal ``train_y`` in the cold-start path). The caller must align
        ``objective_fn``'s evaluation counter (e.g. ``objective._eval_id``) with
        ``n`` so the next Slurm/eval folder index does not collide.

    max_acq_attempts
        When a candidate evaluation fails after GPU job retry, re-run acquisition
        and try a new candidate up to this many times per completed iteration.
        Failed evaluations are not added to the GP.

    surrogate
        GP surrogate to use. ``"stgp"`` (default) is the original
        ``SingleTaskGP`` (ARD Matern) + ``qLogExpectedImprovement``. ``"saas"``
        swaps in ``SaasFullyBayesianSingleTaskGP`` (sparse axis-aligned subspace
        priors, fit by NUTS), which is designed for high-dimensional,
        low-sample BO -- it automatically shrinks unimportant input dimensions,
        effectively searching a low-dimensional subspace. The acquisition and
        everything else are unchanged. ``"saas"`` costs a NUTS refit per
        iteration (tens of seconds) but that is negligible next to a HOOMD eval.

    gp_log_dir
        If given, append one JSON object per fitted GP to
        ``<gp_log_dir>/gp_log.jsonl`` and write a new ``states/*.pt``
        (existing files are not overwritten). ``None`` (default) does not
        write anything and does not change the candidate sequence. The log
        is taken after ``optimize_acqf``; a final refit of the full sample
        is stored with ``stage="final"`` and is not used to pick a point.
        Reload with :func:`load_gp_posterior`. A failure while writing is
        appended to ``errors.txt`` and does not stop the loop.

    Notes
    -----
    - Uses SingleTaskGP + qLogExpectedImprovement (maximize EI on -loss).
    - Works in UNIT cube; ParamSpace handles scaling to physical.
    - Only successful evaluations count toward ``n_iters``.
    """
    torch.manual_seed(seed)
    dtype = torch.float64
    if surrogate not in ("stgp", "saas"):
        raise ValueError(f"surrogate must be 'stgp' or 'saas', got {surrogate!r}")
    print(f"[bo] surrogate={surrogate}")
    if gp_log_dir is not None:
        gp_log_dir = os.fspath(gp_log_dir)
        print(f"[bo] gp log dir: {gp_log_dir}")

    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

    lb, ub = ps.bounds_unit()

    def _evaluate_unit(x_unit: torch.Tensor) -> Optional[torch.Tensor]:
        """Return -loss tensor on success, or None if evaluation failed."""
        if x_unit.ndim == 1:
            x_unit = x_unit.unsqueeze(0)
        try:
            return -objective_fn(x_unit, ffpath=ffpath)
        except EvaluationFailed:
            return None

    def _fit_gp(train_x, train_y):
        # Raw train_y (e.g. -total_loss) is on an arbitrary scale (mean ~ -14,
        # std ~ 1 in this run), which trips BoTorch's InputDataWarning on every
        # refit and can hurt GP hyperparameter fitting. Standardize(m=1)
        # z-scores the targets internally (and un-standardizes the posterior
        # automatically), silencing the warning and improving numerical
        # conditioning.
        if surrogate == "saas":
            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
            from botorch.fit import fit_fully_bayesian_model_nuts
            gp = SaasFullyBayesianSingleTaskGP(
                train_x, train_y, outcome_transform=Standardize(m=1)
            )
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=256, num_samples=128, thinning=16,
                disable_progbar=True,
            )
        else:
            gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
        return gp

    def _optimize_candidate(train_x, train_y):
        gp = _fit_gp(train_x, train_y)
        acq = qLogExpectedImprovement(model=gp, best_f=train_y.max())
        cand, acq_value = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([lb, ub]).to(dtype),
            q=1,
            num_restarts=5,
            raw_samples=64,
        )
        if gp_log_dir is not None:
            try:
                _write_gp_iteration(
                    gp_log_dir,
                    gp,
                    train_x,
                    train_y,
                    ps,
                    surrogate=surrogate,
                    stage="acquisition",
                    acq_value=acq_value,
                    candidate=cand,
                    next_eval_id=getattr(objective_fn, "_eval_id", None),
                )
            except Exception:
                _report_gp_log_error(gp_log_dir)
        return cand

    if warm_start is not None:
        train_x, train_y = warm_start
        train_x = train_x.to(dtype=dtype)
        train_y = train_y.to(dtype=dtype)
        if train_x.ndim == 1:
            train_x = train_x.unsqueeze(0)
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("warm_start: train_x and train_y must have the same leading size")
        if train_y.shape[-1] != 1:
            train_y = train_y.reshape(train_x.shape[0], 1)
        history = [-float(train_y[i, 0].item()) for i in range(train_x.shape[0])]
    else:
        x_unit = ps.init_unit().to(dtype).unsqueeze(0)
        y = None
        for attempt in range(max_acq_attempts):
            if attempt > 0:
                print(
                    f"[bo] initial evaluation failed; retry {attempt + 1}/{max_acq_attempts}"
                )
            y = _evaluate_unit(x_unit)
            if y is not None:
                break
        if y is None:
            raise RuntimeError(
                f"Initial evaluation failed after {max_acq_attempts} attempt(s)"
            )
        train_x = x_unit.clone()
        train_y = y.clone()
        history = [-float(y.item())]

    completed = 0
    while completed < n_iters:
        cand = None
        y_new = None
        for attempt in range(max_acq_attempts):
            if attempt > 0:
                print(
                    f"[bo] iteration {completed + 1}/{n_iters}: "
                    f"re-acquiring after failure ({attempt + 1}/{max_acq_attempts})"
                )
            cand = _optimize_candidate(train_x, train_y)
            y_new = _evaluate_unit(cand)
            if y_new is not None:
                break

        if y_new is None or cand is None:
            raise RuntimeError(
                f"BO stopped at {completed}/{n_iters} successful iterations: "
                f"no successful evaluation after {max_acq_attempts} acquisition attempt(s)"
            )

        train_x = torch.cat([train_x, cand], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)
        history.append(-float(y_new.item()))
        completed += 1

    if gp_log_dir is not None:
        # Full-sample GP for the sensitivity analysis. Candidates are already
        # chosen. The fit sees a clone, and the RNG snapshot is restored
        # before run_bo returns.
        try:
            with _PreserveRng():
                x_final = train_x.detach().clone()
                y_final = train_y.detach().clone()
                gp_final = _fit_gp(x_final, y_final)
                _write_gp_iteration(
                    gp_log_dir,
                    gp_final,
                    x_final,
                    y_final,
                    ps,
                    surrogate=surrogate,
                    stage="final",
                    acq_value=None,
                    candidate=None,
                    next_eval_id=getattr(objective_fn, "_eval_id", None),
                )
        except Exception:
            _report_gp_log_error(gp_log_dir)

    # Best observed (lowest loss)
    best_idx = int(torch.argmax(train_y))  # since train_y = -loss
    best_x_unit = train_x[best_idx]
    best_x_phys = ps.unit_to_phys(best_x_unit)

    return best_x_phys, history

# # ------------------------- 2-Stage BO runner ------------------------- #
# def run_two_stage_bo(
#     datasets,
#     global_param_cfg: Dict[str, Dict[str, float]],
#     local_param_cfg: Dict[str, Dict[str, float]],
#     ffpath: str,
#     out_root: str = "Optimization_Results",
#     trim_tail: int = 200,
#     sim_defaults: Dict[str, Any] = None,
#     mode: str = "map",
#     n_global_iters: int = 20,
#     n_local_iters: int = 20,
#     seed: int = 0,
# ):
#     """
#     Convenience helper that implements the 2-stage schedule:

#     1) Stage 1 (GLOBAL):
#        - Optimize only the GLOBAL parameters (shared across all datasets).
#        - Loss = sum of dataset losses.

#     2) Stage 2 (LOCAL):
#        - For each dataset individually, fit LOCAL parameters while keeping
#          the GLOBAL ones fixed at the Stage-1 optimum.

#     Returns
#     -------
#     results : dict with keys
#         "global"         : dict of best global params
#         "global_history" : list of losses from global BO
#         "local"          : {ds.id: {local_param_name: value, ...}, ...}
#         "local_history"  : {ds.id: [loss_0, loss_1, ...], ...}
#     """
#     sim_defaults = {} if sim_defaults is None else dict(sim_defaults)
#     dataset_ids = [ds.id for ds in datasets]

#     # -------------------- Stage 1: GLOBAL only -------------------- #
#     ps_global = ParamSpace(
#         {"global": dict(global_param_cfg), "local": {}},
#         dataset_ids=dataset_ids,
#     )
#     _validate_param_mode(ps_global, mode)

#     obj_global = make_global_objective(
#         datasets=datasets,
#         ps=ps_global,
#         ffpath = ffpath,
#         out_root=out_root,
#         trim_tail=trim_tail,
#         sim_defaults=sim_defaults,
#         mode=mode,
#     )

#     best_global_vec, global_history = run_bo(
#         obj_global, ps_global, ffpath = ffpath, n_iters=n_global_iters, seed=seed
#     )
#     decoded_global = ps_global.decode(best_global_vec)["global"]

#     results = {
#         "global": decoded_global,
#         "global_history": global_history,
#         "local": {},
#         "local_history": {},
#     }

#     # -------------------- Stage 2: LOCAL per-dataset -------------------- #
#     for ds in datasets:
#         # Freeze globals at the Stage-1 optimum by marking them as "fixed"
#         global_cfg_stage2: Dict[str, Dict[str, float]] = {}
#         for name, val in decoded_global.items():
#             v = float(val)
#             global_cfg_stage2[name] = {
#                 "bounds": (v, v),
#                 "init": v,
#                 "fixed": v,
#             }

#         # Local config is the same for every dataset; ParamSpace will expand it
#         param_cfg_stage2 = {
#             "global": global_cfg_stage2,
#             "local": dict(local_param_cfg),
#         }

#         ps_local = ParamSpace(param_cfg_stage2, dataset_ids=[ds.id])
#         _validate_param_mode(ps_local, mode)

#         obj_local = make_global_objective(
#             datasets=[ds],
#             ps=ps_local,
#             ffpath = ffpath,
#             out_root=out_root,
#             trim_tail=trim_tail,
#             sim_defaults=sim_defaults,
#             mode=mode,
#         )

#         best_local_vec, local_history = run_bo(
#             obj_local, ps_local, ffpath = ffpath, n_iters=n_local_iters, seed=seed
#         )
#         decoded_local = ps_local.decode(best_local_vec)

#         # Store only the locals for this dataset id
#         results["local"][ds.id] = decoded_local["local"][ds.id]
#         results["local_history"][ds.id] = local_history

#     return results
