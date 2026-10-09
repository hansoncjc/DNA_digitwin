# DNA_digitwin

Digital-twin training framework for DNA-mediated silica nanoparticle (siNP) assemblies.

The package wraps a coarse-grained HOOMD-blue simulator, a SAXS / S(q)
extraction pipeline, and a BoTorch Bayesian-optimization (BO) loop, and
ties them together with a small `Dataset` abstraction that carries
**experimental conditions** + **mapping equations** from experimental
inputs to simulation parameters. The BO loop can be told to either:

- **`mode="map"`** – optimize the coefficients of the physics-based
  mapping (`k`, `A`, `K_s`, `a_m`, `delta`) that translate experimental
  inputs (`L_bridge`, `C_chol`, `C_NaCl`) into simulation inputs
  (`r0`, `U0`, `n`, `m`), shared across many experimental conditions;
  `density` is fixed per dataset, or
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
├── simulation.py         # run_simulation (HOOMD), mie_r0
├── scattering.py         # GSD → I(q)/S(q): convert_to_SAXS, convert_to_SAXS_fft, extract_exp_sq
├── metrics.py            # compare_saxs_curves, compare_to_exp[_saxsfft] (MSE / weighted APDist + plots)
├── campaigns/
│   ├── r3.1-gt/          # R3.1: virtual ground truth (conditions, coefficients, generation, QC)
│   └── r3.2-physics-validate/  # R3.2: BO training of the physics-based mapping on it
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
record and exposes the **mapping equations** that connect them.

`physics_params(k, A, K_s, a_m, delta)` (module function `physics_map`,
task 0c-1, 2026-10-09) is the mapping `mode="map"` uses:

- `r0/σ = 1 + k · (l + 0.34 · L_bridge) / d_si`, with `l = 32.04` nm the
  length of every linker part except the bridge (two bilayers, two ssDNA
  spacers, two hybridized ends of the Chiang et al. 2025 construct) and
  `0.34` nm/bp the dsDNA contour length; `L_bridge` in bp. Same `r0` as
  `r0_sigma` at its defaults.
- `g = r0 / (r0 − 1)` (particle size over shell thickness);
  `m = a_m · g`, `n = (a_m + delta) · g`.
- `U0 = A · (C_chol / 100) · (1 + K_s · C_NaCl)`.
- `check_mapped_params` rejects `U0 < 0.5`, `n < 5.5`, `m < 4`,
  `n − m < 1` (the range checked in task 0a); such an evaluation fails.

The older mapping functions below are kept but not used by `bo.py`:

- `rho_N(alpha)` → number density, with theoretical base
  `(6/π) · (C_stock / (ρ_Si·1000)) · (V_stock / V_total)` scaled by
  the global `alpha`.
- `r0_sigma(k, LC_ss=0.63, LC_ds=0.34)` → `r0` in units of σ:
  `r0/σ = 1 + k · ( 2(t_b + LC_ss·L_poly) + LC_ds·(2·L_HBP + L_bridge) ) / d_si`
- `U0_from_gaussian(A, mu_c, sigma_c, K_s)` → **1D** Gaussian in
  `C_chol` multiplied by a **monotonic linear salt prefactor**.
  `K_s` is the salt-response slope `k_s` (units 1/mM, default `0.05`);
  salt and cholesterol are decoupled. The former `b_bridge` Gaussian
  term has been removed (always ≡ 1 under fixed `b_bridge = 0.5`):
  - `S = 1 + K_s · C_NaCl`  
    (`C_NaCl = 0` gives `S = 1`, i.e. full cholesterol-driven well;
    more salt only raises `U0`, never collapses it → no re-entrant
    high-salt dispersion.)
  - `U0 = A · S · exp( -(C_chol - mu_c)²/(2σ_c²) )`

  *(Superseded: an earlier form fed a salt-modulated `C_chol_eff =
  (1 + sat)·C_chol` into a 2D Gaussian in `(C_chol_eff, b_bridge)`,
  which caused unphysical high-salt re-dispersion. See
  `Salt_Concentration.md` §3 for history.)*

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

### Curve metrics (`metrics.py`)

Three loss modes are supported via `metric=` on `make_global_objective`:

- **`mse`** – mean squared error on log10 intensities (default).
- **`apdist`** – weighted Amplitude–Phase Distance on log10 intensities
  over a normalized parameter domain `[0, 1]`:

  ```
  loss = dp_coeff * dp + (1 - dp_coeff) * da
  ```

  Default `dp_coeff=0.5`. Set `dp_coeff=0.0` for amplitude-only loss
  (`da`); set `dp_coeff=1.0` for phase-only loss (`dp`).

  **Note:** this weighted form differs numerically from the earlier
  unweighted `da + dp` sum; do not compare loss values across versions.

When `metric="apdist"`, phase-warp diagnostic plots are saved by default
(`plot_apdist=True`) under `eval_XXX/<dataset_id>/apdist_plots/`:

- `compare_apdist_warped.png` – log10 overlay after phase warp
- `compare_apdist_warp_detail.png` – warp detail panel

The standard overlay `compare_to_exp_saxsfft.png` is still written in
the eval directory. Pass `plot_apdist=False` to skip the warp plots.
When `metric="mse"`, `dp_coeff` and `plot_apdist` are ignored (a warning
is emitted if non-default values are supplied).

- **`shift_rmse`** (saxs-fft only) – peak-aligned log10 RMSE plus a
  first-peak spacing term:

  ```
  loss = M4 + lambda_shift * |ln(q1_sim / q1_tgt)|
  ```

  Each curve's q axis is divided by its own first peak `q1`; the
  simulated intensity is scaled by the high-q `S(q) → 1` band; M4 is the
  RMS log10 difference on a 512-point log grid. With
  `compare_q_range=None` the window is the overlap of the two curves on
  `x = q/q1`, minus `overlap_trim` points per end (default 0; the saxs-fft
  trim already drops 6 bins per end). All parameters go in
  `metric_kwargs` (see `metrics.shift_rmse_params`); the defaults are the
  task-0b choice (2026-10-08): peak search (0.006, 0.12) Å⁻¹, prominence
  `0.3 × (max − min)` of the smoothed curve in that window, asymptote band
  (0.085, 0.100) Å⁻¹, `no_peak="fail"`, both asymptote rules `"fail"`.

  A curve whose smoothed S stays below `1 + dispersed_delta` in the search
  window is dispersed: neither curve is aligned (both use
  `x = q/q1_tgt`) and the spacing term is `dispersed_shift` (0 when both
  curves are dispersed). `dispersed_delta=0.5` sits in the gap between
  flat curves (`s_max − 1 ≤ 0.089`) and curves with a peak (`≥ 0.746`) on
  the saved inverse-run curves; `dispersed_shift=0` avoids a step in the
  loss at the dispersed boundary (mapping ground truth contains
  dispersed conditions).

  The diagnostics key `loss` is the total `M4 + shift_term` (it was `mse`
  before 2026-10-08, when the metric was named `shift_mse`).

  Every eval writes `shift_rmse_diagnostics.json` with the parameters,
  both peaks, and which fallbacks fired. A `metrics.MetricFailed` fails
  the evaluation (logged as `METRIC_FAILED`, not given to the GP); in the
  parallel path the worker writes a `METRIC_FAILED` flag and the job is
  not resubmitted, since the simulation itself succeeded.

### `ParamSpace` (`bo.py`)
Declarative description of the BO search vector. Each parameter is
either **global** (one value shared by all datasets) or **local**
(one value per dataset). Each entry takes `bounds` and an `init`; add
`"fixed": value` to bypass optimization while still feeding the value
into the mappings/sim. The vector is optimized in `[0, 1]^D` and
mapped back to physical bounds internally.

```python
param_cfg = {   # mode="map": exactly the five global coefficients
    "global": {
        "k":     {"bounds": (0.4, 0.9),  "init": 0.76},
        "A":     {"bounds": (0.85, 2.5), "init": 2.0},
        "K_s":   {"bounds": (0.0, 0.10), "init": 0.05},
        "a_m":   {"bounds": (2.91, 4.0), "init": 3.27},
        "delta": {"bounds": (1.09, 4.0), "init": 3.27, "fixed": 3.2},  # "fixed" freezes one
    },
    "local": {},
}
```

### `make_global_objective(...)` (`bo.py`)
Builds a callable `objective(x_unit, ffpath)` that, for one BO query:
1. Decodes the unit vector into globals + locals.
2. For every `Dataset`, resolves `(density, r0, U0, n, m)` according
   to `mode`:
   - `"map"`: `dataset.physics_params(k, A, K_s, a_m, delta)` gives
     `r0, U0, n, m`; `density` is `dataset.sim.density` (the former
     `alpha` coefficient was removed). The ParamSpace must hold exactly
     these five global coefficients; `bo_trajectory.csv` and
     `sim_params_*.csv` carry one column per coefficient, which
     `load_warm_start_from_trajectory` reads back on resume.
   - `"sim"`: takes `density, r0, U0` directly from the param space
     (local > global > `dataset.sim.*`).
3. Runs `simulation.run_simulation(...)` (HOOMD). The pair-table
   cutoff is dynamic by default: `t_tol = tail_energy_cut / U_0`
   with `tail_energy_cut = 0.1`, i.e. the attractive tail at `rmax` is
   `0.1 kT` for every `U_0`. For a fixed cutoff put either `t_tol`
   or `rmax` in `sim_defaults` (each conflicts with `tail_energy_cut`).
   Every mode must give `rmax < L/2`, checked before HOOMD starts.
   The result dict carries `rmax`, `t_tol`, `tail_energy_cut`, `L`.
   Initialization is a Langevin segment on the repulsive branch of the
   same potential, cut at the well minimum (`t_init = 10`, the production
   `dt`). `mie_r0` uses `rmin = rmin_k * r0` (`rmin_k = 0.5`).
   The number of pairs with `r < rmin` in the configuration that starts
   production (GSD frame 0) is logged and returned as `n_pairs_below_rmin`,
   with `min_pair_distance`.
4. Converts GSD → S(q) via either `convert_to_SAXS` (MC-DFM) or
   `convert_to_SAXS_fft` (FFT-based, default `N_grid=600`). The
   simulation and a saxs-fft target must use the same `N_grid`. In
   `mode="sim"` the objective infers each `datatype="sq"` target's
   `N_grid` from its q grid (`scattering.estimate_saxsfft_n_grid`) and
   raises before the first simulation if it differs from
   `scattering_kwargs["N_grid"]` by more than 5 %.
5. Compares against the experimental curve in log-space with `mse`
   (default) or weighted `apdist` (see **Curve metrics** above).
6. Sums `weight · loss` across datasets and appends a row block to
   `out_root/bo_trajectory.csv`.

Pass `parallel=True` plus `parallel_cfg=...` to submit one Slurm GPU
job per dataset per iteration via `parallel.submit_jobs` (see the
multithread example below); the master process is then CPU-only.

### `run_bo(...)` (`bo.py`)
A small BoTorch loop: `SingleTaskGP` + `qLogExpectedImprovement`,
optimized over the unit cube. Returns `(best_x_phys, history)`.

---

### Forward model

The defaults of `simulation.run_simulation` and
`scattering.convert_to_SAXS_fft` are the forward model shared by inverse
design and mapping training (ground truth and BO). Leave them at their
defaults and pass only `N` (each run states its own `N` next to its target
path), `density` and the potential parameters:

| Stage | Defaults |
|---|---|
| Pair potential | `mie_r0` table; dynamic cutoff `t_tol = 0.1/U_0`, no fixed `rmax`, `rmax < L/2`, `rmin = rmin_k * r0` with `rmin_k = 0.5` |
| Initialization | Repulsive branch of that potential, cut at the well, Langevin `t_init = 10` at the production `dt` |
| Langevin production | `kT = 1`, γ = 1, `dt = 1e-3`, `steps = 22_500_000` (450 GSD frames), `seed = 42` |
| saxs-fft | `N_grid = 600`, `frames = 'last:100'`, `step = 5`, `trim = slice(3, -3)`, `particle_diameter = 24.6` nm |

`tests/test_forward_model.py` pins these values.

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
   (`"mse"` or `"apdist"`), `dp_coeff`, `plot_apdist`, `trim_tail`,
   and `sim_defaults` (`steps`, `N`, `device`, `plot`, …).
4. **Run BO.** `bo.run_bo(objective, ps, ffpath=..., n_iters=...)`.
5. **Inspect outputs.** Each evaluation writes to
   `out_root/eval_XXX/<dataset_id>/`:
   - `DNA_assembly_<timestamp>.gsd` – HOOMD trajectory
   - `potential_energy.csv`, `potential_plot.png` (if `plot=True`)
   - `S(q)_data/average_structure_factor.{npy,png}`
   - `compare_to_exp[_saxsfft].png` – diagnostic overlay
   - `apdist_plots/` – phase-warp diagnostics (when `metric="apdist"`)
   - `sim_params_<id>.csv` – the resolved sim inputs for that eval

   - `shift_rmse_diagnostics.json` – loss components, peaks, parameters
     and fallbacks (when `metric="shift_rmse"`)

   Plus, at the run root:
   - `bo_trajectory.csv` – per-iteration block of every dataset's
     parameters and loss, with the iteration's total loss in the
     header line. Each dataset row also carries the audit columns
     `rmax`, `rmax_over_r0`, `t_tol`, `n_pairs_below_rmin` and, for
     `shift_rmse`, `rmse` (M4) and `shift`. Older trajectories without
     these columns still load in `load_warm_start_from_trajectory`.
   - `loss_components.txt` – CSV, one row per successful (iteration,
     dataset): `rmse`, `shift`, `loss` and the iteration's `total_loss`.

---

## Example: train a digital twin with the multithread feature

The script below has the layout of the 9-sample test
(`testing/digitwin_test/two_mapping_coeff/9_sample_test/multithread/`),
rewritten for the physics-based mapping (2026-10-09): it fits two
mapping coefficients (`k`, `A`) over 9 experimental conditions with the
other three frozen. The full R3.2 campaign (24 conditions, all five
coefficients) is `campaigns/r3.1-gt/` (ground truth) and
`campaigns/r3.2-physics-validate/` (training). Each BO iteration submits 9 GPU jobs
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
        exp      = ExperimentalParams(L_bridge=L_bridge, C_chol=C_chol, C_NaCl=10.0),
        sim      = SimulationParams(density=0.005),   # density is fixed; r0, U0, n, m mapped
        out_dir  = os.path.join(OUT_ROOT, f"d{idx}"),
        datatype = "sq",                     # set "iq" if files are I(q)
    )
    datasets.append(ds)

FIXED = {"K_s": 0.07, "a_m": 3.2, "delta": 3.2}

param_cfg = {
    "global": {
        "k": {"bounds": (0.4, 0.9),  "init": 0.76},
        "A": {"bounds": (0.85, 2.5), "init": 2.0},

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
    sim_defaults      = {"N": 5000, "device": "gpu", "plot": False},
    mode              = "map",
    scattering_method = "saxsfft",
    scattering_kwargs = {},                  # forward-model defaults (N_grid=600)
    metric            = "shift_rmse",
    compare_q_range   = None,                # shift_rmse compares over the curve overlap
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
