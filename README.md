# DNA_digitwin

Digital-twin training framework for DNA-mediated silica nanoparticle (siNP) assemblies.

The package wraps a coarse-grained HOOMD-blue simulator, a SAXS / S(q)
extraction pipeline, and a BoTorch Bayesian-optimization (BO) loop, and
ties them together with a small `Dataset` abstraction that carries
**experimental conditions** + **mapping equations** from experimental
inputs to simulation parameters. The BO loop can be told to either:

- **`mode="map"`** – optimize *mapping coefficients* (`alpha`, `k`, `A`,
  `mu_c`, `mu_b`, `sigma_c`, `sigma_b`, `K_s`) that translate
  experimental inputs (`C_NaCl`, `C_chol`, `b_bridge`, `L_bridge`, …)
  into simulation inputs (`density`, `r0`, `U0`), shared across many
  experimental conditions, or
- **`mode="sim"`** – optimize the simulation inputs themselves
  (`density`, `r0`, `U0`, `n`, `m`) directly, either globally or
  per-dataset.

Per-dataset evaluations can be run **sequentially** in one Python
process or **in parallel** by submitting one Slurm GPU job per dataset
per BO iteration.

---

## Repository layout

```
DNA_digitwin/
├── bo.py                 # ParamSpace, make_global_objective, run_bo
├── datasets.py           # ExperimentalParams, SimulationParams, Dataset
├── simulation.py         # run_simulation (HOOMD), modified_LJ + shifted_mie
├── scattering.py         # GSD → I(q)/S(q): convert_to_SAXS, convert_to_SAXS_fft, extract_exp_sq
├── metrics.py            # compare_to_exp, compare_to_exp_saxsfft (log-space MSE / AP-distance)
├── parallel/             # Slurm launcher: submits 1 GPU job per dataset per BO iteration
│   ├── __init__.py
│   ├── submit_parallel.py
│   └── worker.py
├── formfactors/
│   └── sasmodels_sphere_fit.txt   # polydisperse sphere form factor for I(q)→S(q)
├── LICENSE
└── README.md
```

---

## External dependencies

The framework is a thin layer on top of several libraries. Install /
make importable on `PYTHONPATH` before training:

- `hoomd` (v2.x API – uses `hoomd.context`, `hoomd.md.pair.table`,
  `hoomd.dump.gsd`, `hoomd.analyze.log`)
- `torch`, `botorch`, `gpytorch` – Gaussian-process surrogate +
  `qLogExpectedImprovement` acquisition
- `numpy`, `scipy`, `pandas`, `matplotlib`, `gsd`
- `apdist` – optional, only needed if you choose `metric="apdist"`
- [**MC-DFM**](../MC-DFM) – `Scattering_Simulator.pairwise_method` for
  the GSD → I(q) path used by `convert_to_SAXS` (`scattering_method="mcdfm"`)
- [**saxsfft**](https://github.com/your/saxsfft) – `StructureFactor`
  used by `convert_to_SAXS_fft` (`scattering_method="saxsfft"`),
  recommended when an experimental S(q) is already available

Both `MC-DFM` and the `DNA_digitwin` repo root must be on
`sys.path` / `PYTHONPATH` before importing `bo`.

---

## Core concepts

### `ExperimentalParams` (`datasets.py`)
Flat container for everything that comes from the experiment: stock
chemistry (`C_stock`, `V_stock`, `V_total`, `rho_si`), solution
composition (`C_NaCl`, `C_chol`, `b_bridge`), DNA sequence lengths
(`L_poly`, `L_bridge`, `L_HBP`), and geometry (`d_si`, `t_b`).
`C_bridge = b_bridge * C_chol` is exposed as a property.

### `SimulationParams` (`datasets.py`)
What the simulator actually consumes: `density`, `n`, `m`, `U0`, `r0`,
`N`, `steps`, `dt`, `kT`. Any of these can be left `None` and filled in
by the mapping equations.

### `Dataset` (`datasets.py`)
Bundles one `(experimental, simulation, exp_path, weight, out_dir, datatype)`
record and exposes the **mapping equations** that connect them:

- `rho_N(alpha)` → number density, with theoretical base
  `(6/π) · (C_stock / (ρ_Si·1000)) · (V_stock / V_total)` scaled by
  the global `alpha`.
- `r0_sigma(k, LC_ss=0.63, LC_ds=0.34)` → `r0` in units of σ:
  `r0/σ = 1 + k · ( 2(t_b + LC_ss·L_poly) + LC_ds·(2·L_HBP + L_bridge) ) / d_si`
- `U0_from_gaussian(A, mu_c, mu_b, sigma_c, sigma_b, K_s)` → separable
  Gaussian in `(C_chol_eff, b_bridge)` where the salt-modulated
  effective cholesterol uses a saturating term plus unity (`K_s` couples
  salt and cholesterol in the denominator; default `0.5`). Let
  `ε = 10⁻⁶` (same units as concentrations, avoids `0/0` when
  `C_NaCl = C_chol = 0`):
  - `sat = C_NaCl / (C_NaCl + K_s · C_chol + ε)`
  - `C_chol_eff = (1 + sat) · C_chol`  
    (`C_NaCl = 0` gives `C_chol_eff = C_chol`; large salt gives `C_chol_eff → 2·C_chol`.)
  - `U0 = A · exp( -(C_chol_eff - mu_c)²/(2σ_c²) - (b_bridge - mu_b)²/(2σ_b²) )`

### `datatype` — S(q) vs I(q) input switch
The `datatype` argument on `Dataset` (and the matching key in
`Dataset.from_dict`) tells the objective how to interpret the file at
`exp_path`:

- `datatype="sq"` (default) — the file is already a structure factor
  `[q, S(q)]`. It is loaded as-is by `Dataset.load_exp_curve` and fed
  straight into the comparison metric. `ffpath` is unused on the
  experimental side.
- `datatype="iq"` — the file is an experimental scattering intensity
  `[q, I(q)]`. The objective converts it to an effective S(q) on the
  fly via `scattering.extract_exp_sq(...)`, which divides out the
  polydisperse sphere form factor stored at `ffpath` (default:
  `formfactors/sasmodels_sphere_fit.txt`) over the q-window
  `[q_min=0.02, q_max=0.03]` Å⁻¹.

Pick `"sq"` when you have already pre-processed your experimental
SAXS into a structure factor (e.g. via `saxs-fft` or an external
fitting tool); pick `"iq"` when you want the framework to handle the
form-factor division itself. The simulated side always produces an
S(q), so the choice only affects how the experimental curve is read.

Use `trim_tail=N` on `make_global_objective` to drop the last `N`
points of the loaded experimental curve before comparison (useful for
noisy I(q) tails; set `0` for clean S(q) files).

### `ParamSpace` (`bo.py`)
Declarative description of the BO search vector. Each parameter is
either **global** (one value shared by all datasets) or **local**
(one value per dataset). Each entry takes `bounds` and an `init`; add
`"fixed": value` to bypass optimization while still feeding the value
into the mappings/sim. The vector is optimized in `[0, 1]^D` and
mapped back to physical bounds internally.

```python
param_cfg = {
    "global": {
        "k":     {"bounds": (0.3, 1.2), "init": 0.76},
        "alpha": {"bounds": (1.0, 5.0), "init": 3.0, "fixed": 3.0},
    },
    "local": {
        # "U0": {"bounds": (0.5, 50.0), "init": 5.0},   # one U0 per dataset
    },
}
```

### `make_global_objective(...)` (`bo.py`)
Builds a callable `objective(x_unit, ffpath)` that, for one BO query:
1. Decodes the unit vector into globals + locals.
2. For every `Dataset`, resolves `(density, r0, U0, n, m)` according
   to `mode`:
   - `"map"`: applies the mapping equations using the optimized
     coefficients above (`alpha→density`, `k→r0`,
     `A,mu_c,mu_b,sigma_c,sigma_b,K_s → U0`).
   - `"sim"`: takes `density, r0, U0` directly from the param space
     (local > global > `dataset.sim.*`).
3. Runs `simulation.run_simulation(...)` (HOOMD).
4. Converts GSD → S(q) via either `convert_to_SAXS` (MC-DFM) or
   `convert_to_SAXS_fft` (FFT-based).
5. Compares against the experimental curve in log-space with `mse`
   (default) or `apdist`.
6. Sums `weight · loss` across datasets and appends a row block to
   `out_root/bo_trajectory.csv`.

Pass `parallel=True` plus `parallel_cfg=...` to submit one Slurm GPU
job per dataset per iteration via `parallel.submit_jobs` (see the
multithread example below); the master process is then CPU-only.

### `run_bo(...)` (`bo.py`)
A small BoTorch loop: `SingleTaskGP` + `qLogExpectedImprovement`,
optimized over the unit cube. Returns `(best_x_phys, history)`.

---

## End-to-end workflow

1. **Build datasets.** For each experimental condition you have an
   S(q) (or I(q)) file for, instantiate `ExperimentalParams`,
   `SimulationParams`, and bundle them in a `Dataset`.
2. **Declare the parameter space.** Decide which mapping or sim
   parameters are global vs. per-dataset; freeze the rest with
   `"fixed"`.
3. **Make the objective.** Choose `mode` (`"map"` or `"sim"`),
   `scattering_method` (`"saxsfft"` or `"mcdfm"`), `metric`
   (`"mse"` or `"apdist"`), `trim_tail`, and `sim_defaults`
   (`steps`, `N`, `device`, `plot`, …).
4. **Run BO.** `bo.run_bo(objective, ps, ffpath=..., n_iters=...)`.
5. **Inspect outputs.** Each evaluation writes to
   `out_root/eval_XXX/<dataset_id>/`:
   - `DNA_assembly_<timestamp>.gsd` – HOOMD trajectory
   - `potential_energy.csv`, `potential_plot.png` (if `plot=True`)
   - `S(q)_data/average_structure_factor.{npy,png}`
   - `compare_to_exp[_saxsfft].png` – diagnostic overlay
   - `sim_params_<id>.csv` – the resolved sim inputs for that eval

   Plus, at the run root:
   - `bo_trajectory.csv` – per-iteration block of every dataset's
     parameters and loss, with the iteration's total loss in the
     header line.

---

## Example: train a digital twin with the multithread feature

The script below mirrors the validated 9-sample test
(`testing/digitwin_test/two_mapping_coeff/9_sample_test/multithread/`)
and recovers two mapping coefficients (`k`, `A`) by globally fitting
9 experimental conditions. All other mapping coefficients are frozen
at their ground-truth defaults. Each BO iteration submits 9 GPU jobs
in parallel through `DNA_digitwin/parallel/`; the master process runs
on a CPU node.

> **Note on candidates.** The `CANDIDATES` list below is just an
> illustrative choice — `(L_bridge, C_chol)` pairs that span the two
> experimental axes the chosen mapping coefficients (`k` from
> `r0_sigma`, `A` from `U0_from_gaussian`) are sensitive to. For a
> different training session you should pick candidates that
> **(a)** vary the experimental inputs the coefficients you are
> optimizing actually depend on (e.g. add a temperature/`C_NaCl`
> sweep if those parameters enter the mapping you are training), and
> **(b)** have matching experimental S(q) / I(q) curves available.
> The number of candidates is also free to choose — fewer candidates
> mean faster iterations but a less constrained fit; more candidates
> mean stronger global constraints at higher per-iteration cost.

```python
import os
import sys

CODE_ROOT      = "/path/to/DNA_digitwin"
MC_DFM_ROOT    = "/path/to/MC-DFM"
FORMFACTOR     = os.path.join(CODE_ROOT, "formfactors", "sasmodels_sphere_fit.txt")

sys.path.append(CODE_ROOT)
sys.path.append(MC_DFM_ROOT)

from datasets import ExperimentalParams, SimulationParams, Dataset
import bo

OUT_ROOT = "./Optimization_Results"

CANDIDATES = [
    # (L_bridge [nt], C_chol [molecules/siNP], experimental S(q) file)
    (20.0,  95.0, "/path/to/exp/d0_sq.npy"),
    (40.0,  95.0, "/path/to/exp/d1_sq.npy"),
    (80.0,  95.0, "/path/to/exp/d2_sq.npy"),
    (20.0, 110.0, "/path/to/exp/d3_sq.npy"),
    (40.0, 110.0, "/path/to/exp/d4_sq.npy"),
    (80.0, 110.0, "/path/to/exp/d5_sq.npy"),
    (20.0, 125.0, "/path/to/exp/d6_sq.npy"),
    (40.0, 125.0, "/path/to/exp/d7_sq.npy"),
    (80.0, 125.0, "/path/to/exp/d8_sq.npy"),
]

datasets = []
for idx, (L_bridge, C_chol, exp_path) in enumerate(CANDIDATES):
    ds = Dataset(
        id       = f"d{idx}",
        exp_path = exp_path,
        exp      = ExperimentalParams(L_bridge=L_bridge, C_chol=C_chol),
        sim      = SimulationParams(),       # filled by mappings during BO
        out_dir  = os.path.join(OUT_ROOT, f"d{idx}"),
        datatype = "sq",                     # set "iq" if files are I(q)
    )
    datasets.append(ds)

FIXED = {
    "alpha":   3.0,  "n":  12.0, "m": 6.0,
    "mu_c":  100.0,  "mu_b": 0.5,
    "sigma_c": 10.0, "sigma_b": 0.2,
    "K_s":     0.5,
}

param_cfg = {
    "global": {
        "k": {"bounds": (0.3, 1.2),  "init": 0.76},
        "A": {"bounds": (1.0, 15.0), "init": 2.0},

        **{name: {"bounds": (v, v), "init": v, "fixed": v}
           for name, v in FIXED.items()},
    },
    "local": {},
}
ps = bo.ParamSpace(param_cfg, dataset_ids=[d.id for d in datasets])

print(bo.describe_training_config(ps, mode="map"))

PARALLEL_CFG = {
    "partition":     "gpu-a40",
    "account":       "zeelab",
    "gpus":          1,
    "mem":           "10G",
    "time":          "02:00:00",
    "module_loads":  ["ssmc/miniconda/3.9", "cuda/11.8.0"],
    "venv_activate": "/path/to/hoomd-venv/bin/activate",
    "code_root":     CODE_ROOT,
    "mc_dfm_root":   MC_DFM_ROOT,
    "poll_interval": 15.0,
    "max_wait":      4 * 3600,
}

os.makedirs(OUT_ROOT, exist_ok=True)
objective = bo.make_global_objective(
    datasets          = datasets,
    ps                = ps,
    ffpath            = FORMFACTOR,
    out_root          = OUT_ROOT,
    trim_tail         = 0,
    sim_defaults      = {"steps": 15_000_000, "N": 5000,
                         "device": "gpu", "plot": False},
    mode              = "map",
    scattering_method = "saxsfft",
    scattering_kwargs = {"N_grid": 300},
    metric            = "mse",
    parallel          = True,
    parallel_cfg      = PARALLEL_CFG,
)

best_x_phys, history = bo.run_bo(
    objective_fn = objective,
    ps           = ps,
    ffpath       = FORMFACTOR,
    n_iters      = 20,
    seed         = 42,
)

best = ps.decode(best_x_phys)["global"]
print(f"Optimized k = {best['k']:.4f}")
print(f"Optimized A = {best['A']:.4f}")
print(f"Best loss   = {history[-1]:.6f}")
```

### Switching off multithreading

Set `parallel=False` (and drop `parallel_cfg`) in
`make_global_objective` to run all per-dataset evaluations
sequentially in the same process. The objective signature, outputs,
and `bo_trajectory.csv` format are identical; only the wall-clock
behaviour changes.

### Switching mode

Replace the mapping-coefficient block in `param_cfg["global"]` with
sim parameters (e.g. `"density"`, `"r0"`, `"U0"`, `"n"`, `"m"`) – or
move `"r0"` / `"U0"` into `"local"` – and pass `mode="sim"` to
`make_global_objective`. `bo._validate_param_mode` will reject mixed
configurations (mapping coeffs in `"sim"` mode, or vice versa).
