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
        # "sigma_c": {"bounds": (0.05, 0.5),  "init": 0.1},
        # "sigma_b": {"bounds": (5.0, 25.0),  "init": 15.0},
    },
    "local": {
        # leave empty in Stage 1 (or put "U0"/"r0" here for Stage 2 refinements)
    }
}

from bo import ParamSpace, make_global_objective, run_bo
ps = ParamSpace(param_cfg, dataset_ids=[d.id for d in datasets])

obj = make_global_objective(datasets, ps, out_root="Optimization_Results",
                            trim_tail=200, sim_defaults={"steps": 1_500_000})

best, history = run_bo(obj, ps, n_iters=20, seed=0)
print("Best params (physical):", ps.decode(best))
"""

import os
import numpy as np
import torch
from torch import tensor
from typing import Dict, List, Tuple, Any

from simulation import run_simulation
from scattering import convert_to_SAXS
from metrics import compare_to_exp


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

def make_global_objective(
    datasets: List[Any],
    ps: ParamSpace,
    out_root: str = "Optimization_Results",
    trim_tail: int = 200,
    sim_defaults: Dict[str, Any] = None,
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

    Exceptions during any dataset evaluation yield a large penalty to keep BO stable.
    """
    sim_defaults = {} if sim_defaults is None else dict(sim_defaults)

    def objective(x_unit: torch.Tensor) -> torch.Tensor:
        # x_unit: (1,d) or (d,)
        x_unit = x_unit.reshape(-1)
        # 1) map [0,1] → physical
        x_phys = ps.unit_to_phys(x_unit)
        # 2) decode into globals/locals
        decoded = ps.decode(x_phys)
        G = decoded["global"]
        L = decoded["local"]

        total_loss = 0.0

        for ds in datasets:
            try:
                # ---- Shared params ----
                alpha = float(G.get("alpha", 1.0))
                density = ds.rho_N(alpha=alpha)

                n = float(G["n"]) if "n" in G else float(ds.sim.n)
                m = float(G["m"]) if "m" in G else float(ds.sim.m)

                # ---- Per-dataset r0 ----
                if "r0" in L[ds.id]:
                    r0 = float(L[ds.id]["r0"])
                elif "k" in G:
                    r0 = float(ds.r0_sigma(k=float(G["k"])))
                elif ds.sim.r0 is not None:
                    r0 = float(ds.sim.r0)
                else:
                    raise ValueError(f"Dataset {ds.id}: r0 not provided and no mapping coeff 'k' found.")

                # ---- Per-dataset U0 ----
                if "U0" in L[ds.id]:
                    U0 = float(L[ds.id]["U0"])
                elif all(k in G for k in ("A", "mu_c", "sigma_c", "sigma_b")):
                    U0 = float(ds.U0_from_gaussian(A=G["A"], mu_c=G["mu_c"],
                                                   sigma_c=G["sigma_c"], sigma_b=G["sigma_b"]))
                elif ds.sim.U0 is not None:
                    U0 = float(ds.sim.U0)
                else:
                    raise ValueError(f"Dataset {ds.id}: U0 not provided and no global Gaussian coeffs found.")

                # ---- Output directory ----
                save_dir = ds.out_dir or os.path.join(out_root, f"dataset_{ds.id}")
                os.makedirs(save_dir, exist_ok=True)

                # ---- 1) Simulation ----
                _ = run_simulation(
                    density=density, U_0=U0, r0=r0, n=n, m=m, outdir=save_dir, **sim_defaults
                )

                # ---- 2) Sim → S(q) ----
                convert_to_SAXS(save_dir)

                # ---- 3) Compare to experiment ----
                sim_sq_path = os.path.join(save_dir, "scattering_data", "average_structure_factor.npy")
                if not os.path.exists(sim_sq_path):
                    raise FileNotFoundError(f"Missing S(q): {sim_sq_path}")
                sim_sq = np.load(sim_sq_path)

                exp_curve = ds.load_exp_curve(trim_tail=trim_tail)
                loss = float(compare_to_exp(exp_curve, sim_sq, save_dir))

                total_loss += ds.weight * loss

            except Exception as e:
                # Large penalty for any failure in this dataset
                total_loss += 1e9
                # Optional: write the error to a file for debugging
                try:
                    with open(os.path.join(ds.out_dir or out_root, f"error_{ds.id}.txt"), "a") as fh:
                        fh.write(str(e) + "\n")
                except Exception:
                    pass

        # Return as a 1-element tensor (BoTorch expects a tensor)
        return torch.tensor([total_loss], dtype=torch.float64)

    return objective


# ------------------------- BO runner ------------------------- #

def run_bo(
    objective_fn,
    ps: ParamSpace,
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
    y = -objective_fn(x_unit)  # maximize EI on negative loss
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
        y_new = -objective_fn(cand)  # negative loss for maximization
        train_x = torch.cat([train_x, cand], dim=0)
        train_y = torch.cat([train_y, y_new], dim=0)

        history.append(-float(y_new.item()))

    # Best observed (lowest loss)
    best_idx = int(torch.argmax(train_y))  # since train_y = -loss
    best_x_unit = train_x[best_idx]
    best_x_phys = ps.unit_to_phys(best_x_unit)

    return best_x_phys, history
