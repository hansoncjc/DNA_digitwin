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
import numpy as np
import torch
from pathlib import Path
from torch import tensor
from typing import Dict, List, Tuple, Any, Optional
import csv

from simulation import run_simulation
from scattering import convert_to_SAXS, convert_to_SAXS_fft, extract_exp_sq
from metrics import compare_to_exp, compare_to_exp_saxsfft

# ------------------------- Modes & param types ------------------------- #

# Parameters that correspond to direct simulation inputs
_SIM_PARAMS = {"density", "r0", "U0"}

# Parameters that don't correpond to modes
_ALWAYS_OK_SIM = {"n", "m"}

# Parameters that correspond to mapping coefficients
_MAP_PARAMS = {"alpha", "k", "A", "mu_c", "sigma_c", "sigma_b", "mu_b", "K_s"}


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

# ------------------------- Nan Handling ------------------------- #
def _sanitize_curve(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr).all(axis=1)]
    q, I = arr[:, 0], arr[:, 1]
    order = np.argsort(q)
    q, I = q[order], I[order]
    uq, idx = np.unique(q, return_index=True)  # remove duplicate q (interp1d hates these)
    q, I = uq, I[idx]
    I = np.clip(I, 1e-12, None)               # avoid log(0) downstream

    return np.column_stack([q, I])

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
    iteration,dataset_id,loss,k,alpha,A,mu_c,mu_b,sigma_c,sigma_b,density,n,m,r0,U0
    N,d0,loss_val,k_val,...
    N,d1,loss_val,k_val,...
    <blank line>
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
    parallel_cfg: Dict[str, Any],
    iteration_data: List[Dict[str, Any]],
) -> float:
    """
    Parallel analogue of the per-dataset sequential loop in `objective`.

    Submits one Slurm GPU job per dataset via `parallel.submit_jobs`, waits
    for all of them to reach a terminal state, and collects each job's
    loss from its `DONE` flag file. Mutates `iteration_data` in place so
    the caller can write the trajectory CSV block exactly as in the
    sequential path.
    """
    from parallel import LauncherConfig, submit_jobs  # lazy import

    total_loss = 0.0
    plans: List[Dict[str, Any]] = []

    # Phase A: resolve per-dataset sim params with the same precedence as
    # the sequential path; write per-dataset `sim_params_<id>.csv`.
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
                if all(k in G for k in ("A", "mu_c", "sigma_c", "sigma_b")):
                    U0 = float(ds.U0_from_gaussian(
                        A=G["A"],
                        mu_c=G["mu_c"],
                        sigma_c=G["sigma_c"],
                        sigma_b=G["sigma_b"],
                        K_s=G.get("K_s", 0.5),
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
            total_loss += 1e9
            try:
                os.makedirs(out_root, exist_ok=True)
                with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                    fh.write(str(e) + "\n")
            except Exception:
                pass
            iteration_data.append({
                "iteration": eval_id,
                "dataset_id": ds.id,
                "loss": 1e9,
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
            })

    # Phase B: build job specs for OK plans and submit them all.
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
                },
            },
        })

    finished_by_ds: Dict[str, Any] = {}
    if job_specs:
        pcfg = parallel_cfg or {}
        cfg = LauncherConfig(
            run_dir=Path(out_root) / f"eval_{eval_id:03d}" / "_jobs",
            partition    =pcfg.get("partition", "gpu-a40"),
            account      =pcfg.get("account",   "zeelab"),
            gpus         =int(pcfg.get("gpus", 1)),
            mem          =pcfg.get("mem",  "40G"),
            time         =pcfg.get("time", "02:00:00"),
            module_loads =list(pcfg.get("module_loads", [])),
            venv_activate=pcfg.get("venv_activate", ""),
            code_root    =pcfg.get("code_root", ""),
            mc_dfm_root  =pcfg.get("mc_dfm_root", ""),
            poll_interval=float(pcfg.get("poll_interval", 15.0)),
            max_wait     =float(pcfg.get("max_wait", 4 * 3600)),
        )
        finished = submit_jobs(job_specs, cfg)
        finished_by_ds = {j.ds_id: j for j in finished}

    # Phase C: collect per-dataset loss; emit trajectory records.
    for plan in plans:
        ds  = plan["ds"]
        job = finished_by_ds.get(ds.id)
        if job is not None and job.done_status == "DONE":
            loss = 1e9
            try:
                done_data = json.loads((Path(job.sim_dir) / "DONE").read_text())
                loss = float(done_data["loss"])
            except Exception as e:
                try:
                    with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                        fh.write(f"DONE file unreadable: {e}\n")
                except Exception:
                    pass
                loss = 1e9

            total_loss += ds.weight * loss
            iteration_data.append({
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
            })
        else:
            total_loss += 1e9
            reason = f"parallel job status={getattr(job, 'done_status', None)}"
            try:
                with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                    fh.write(reason + "\n")
            except Exception:
                pass
            iteration_data.append({
                "iteration": eval_id,
                "dataset_id": ds.id,
                "loss": 1e9,
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
            })

    return total_loss


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
    parallel: bool = False,
    parallel_cfg: Dict[str, Any] = None,
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
        * else if GLOBAL ("A","mu_c","sigma_c","sigma_b") present, use dataset.U0_from_gaussian(...),
        * else if dataset.sim.U0 is set, use it,
        * else raise.
    mode: default to be "map"
    ----
    "map" (default):
        density, r0, U0 are computed via dataset mappings
        (alpha → density, k → r0, A/mu_c/sigma_* → U0).
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
    Exceptions during any dataset evaluation yield a large penalty to keep BO stable.
    """
    sim_defaults = {} if sim_defaults is None else dict(sim_defaults)

    # Ensure the ParamSpace is consistent with the chosen mode
    _validate_param_mode(ps, mode)

    def objective(x_unit: torch.Tensor, ffpath: str) -> torch.Tensor:
        # assign a unique id to this BO evaluation
        # ffpath: path to polydispersed sphere formfactor
        if not hasattr(objective, "_eval_id"):
            objective._eval_id = 0
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
                parallel_cfg=parallel_cfg or {},
                iteration_data=objective._iteration_data,
            )

            if len(objective._iteration_data) > 0:
                trajectory_path = os.path.join(out_root, "bo_trajectory.csv")
                _write_iteration_block(
                    filepath=trajectory_path,
                    iteration=eval_id,
                    total_loss=float(total_loss),
                    records=objective._iteration_data,
                )
                objective._iteration_data = []

            return torch.tensor([[total_loss]], dtype=torch.float64)

        # --- Sequential path (original behavior, unchanged) ---
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
                    if all(k in G for k in ("A", "mu_c", "sigma_c", "sigma_b")):
                        U0 = float(
                            ds.U0_from_gaussian(
                                A=G["A"],
                                mu_c=G["mu_c"],
                                sigma_c=G["sigma_c"],
                                sigma_b=G["sigma_b"],
                                K_s=G.get("K_s", 0.5),
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
                _ = run_simulation(
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
                    ))
                else:
                    loss = float(compare_to_exp(exp_sq, sim_sq, save_dir, metric=metric))
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
                }
                objective._iteration_data.append(trajectory_record)

            except Exception as e:
                total_loss += 1e9
                try:
                    with open(os.path.join(out_root, f"error_{ds.id}.txt"), "a") as fh:
                        fh.write(str(e) + "\n")
                except Exception:
                    pass

                # Still track failed dataset in trajectory with error marker
                trajectory_record = {
                    "iteration": eval_id,
                    "dataset_id": ds.id,
                    "loss": 1e9,
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
                objective._iteration_data.append(trajectory_record)

        # Write iteration block to global trajectory CSV
        if len(objective._iteration_data) > 0:
            trajectory_path = os.path.join(out_root, "bo_trajectory.csv")
            _write_iteration_block(
                filepath=trajectory_path,
                iteration=eval_id,
                total_loss=float(total_loss),
                records=objective._iteration_data
            )
            # Reset for next iteration
            objective._iteration_data = []

        # Return as a 1-element tensor (BoTorch expects a tensor)
        return torch.tensor([[total_loss]], dtype=torch.float64)

    return objective


# ------------------------- BO runner ------------------------- #

def run_bo(
    objective_fn,
    ps: ParamSpace,
    ffpath: str,
    n_iters: int = 20,
    seed: int = 0,
):
    """
    Run a BoTorch loop over the parameter space defined by ParamSpace.

    Minimizes the provided objective (sum of losses). Returns:
        best_x_phys (1D torch tensor), history (list of floats)

    Notes
    -----
    - Uses SingleTaskGP + qLogExpectedImprovement (maximize EI on -loss).
    - Works in UNIT cube; ParamSpace handles scaling to physical.
    """
    torch.manual_seed(seed)
    dtype = torch.float64

    # Initial design (1 point at provided init)
    x_unit = ps.init_unit().to(dtype).unsqueeze(0)  # (1,d)
    y = -objective_fn(x_unit, ffpath = ffpath)  # maximize EI on negative loss
    train_x = x_unit.clone()
    train_y = y.clone()

    history = [-float(y.item())]  # store the actual loss

    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

    lb, ub = ps.bounds_unit()

    for _ in range(n_iters):
        # Fit GP
        gp = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Acquisition
        acq = qLogExpectedImprovement(model=gp, best_f=train_y.max())

        # Optimize acquisition over unit cube
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([lb, ub]).to(dtype),
            q=1,
            num_restarts=5,
            raw_samples=64,
        )

        # Evaluate objective at candidate
        y_new = -objective_fn(cand, ffpath = ffpath)  # negative loss for maximization
        train_x = torch.cat([train_x, cand], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

        history.append(-float(y_new.item()))

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
