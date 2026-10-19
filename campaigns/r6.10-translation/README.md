# R6.10: translation of R6.4 and R6.5 through the trained physics mapping

Find the minimum **latent posterior mean loss** of each saved final R6 GP
on the eval-56 mapping manifold, inside the original four-dimensional R6
box. Invert the selected `(r0, U0)` to continuous bridge length and the
entire nonnegative anchor/salt family. R3.2 training ranges are annotations,
not constraints. No GP refit, forward-model change, or local execution is
required. A candidate is a surrogate prediction, not a demonstrated recovery.

## Decisions and scientific limits

- Both R6.4 and R6.5; R3.2 eval 56 coefficients extracted from its 24 raw
  dataset rows. The R3.2 GP is not needed.
- Original R6 bounds: U0 [0.5,8], r0 [2,3.5], n [5.5,15], m_rel [0,1].
  The last two coordinates follow the trained mapping. Negative bridge or
  concentrations are not experimental solutions. Bridge is not rounded.
- Minimize mean loss only. Report latent GP SD separately; it is neither
  repeat-simulation SD nor mapping-coefficient uncertainty.
- Plot salt 0–100 mM; report conditions at 0,20,40,60 mM. No concentration
  maximum is imposed on the analytical family.
- One candidate per run. Check its **20 mM** condition with trained and GT
  coefficients, one seed (42, the existing default) each: four simulations.
  GT must receive the SAME condition, not a new inverse solved with GT.
- Fixed N and density come from each original run; other forward defaults
  remain unchanged. S(q) extraction is on CPU as in R6.
- q1_target*r0 is a reference, NOT a prediction of q1_sim*r0. Actual q1,
  loss and shift come from the check simulations. No success criterion or
  PTM-based structural conclusion is applied here.
- Parameter distances to the observed BO best use the R6 box widths;
  they are not distances between structures. The (r0,n) plot is a projection
  of observations with different U0 and m_rel, not a 2D loss function.

## Files and provenance

`config.py`, `translate.py`, `check_translation.py` and the two sbatch files
are the only campaign programs. Tests: `tests/test_r610_translation.py`.

`inputs/provenance.json` records original sources and SHA-256 hashes.
All files below must be committed with the campaign (no LFS pointer or
external Desktop input at runtime):

- `inputs/r3.2_eval_056.csv`: the raw 24 eval-56 rows, including coefficients.
- For each `inputs/r6.4/` and `inputs/r6.5/`:
  `gp_log/states/ntrain_0100_final.pt`, full `gp_log/gp_log.jsonl`,
  `bo_trajectory.csv`, `inverse_config.json`, `target_sq.npy`.
  Earlier state paths mentioned in the complete log are historical only;
  the program loads and validates the single final snapshot only.
- `inputs/check_candidates.csv` is intentionally absent initially. It is
  created only after reviewing the translation, then committed before GPU
  checks. Never fabricate this table before optimization.

Historical absolute paths inside the provenance and original run configs
are metadata only; scripts use bundled inputs. GT coefficients come from
`../r3.1-gt/gt_config.py`. Existing repository simulation, scattering,
metrics and GP APIs are imported without modification.

Software is not vendored: use the cluster environment that produced R6,
including torch/BoTorch/GPyTorch, NumPy/SciPy/matplotlib and the repository's
normal dependencies. GPU checks additionally need HOOMD 2.x, saxs-fft and
MC-DFM. sbatch files use the existing `hoomd-venv` and MC-DFM installation.
They require only `R610_REPO` (the pinned CODE location), not additional
scientific input paths. Do not upgrade BoTorch to work around a load error.
The default SingleTaskGP kernel changed between versions; failure to load
or reproduce the recorded posterior must stop the workflow and be reported.

## Algorithm and checks

1. SHA-256-check bundled inputs, target MD5, original bounds, final stage,
   100-point training set, coordinate order and negative-loss convention.
2. Load with `bo.load_gp_model`; reproduce final logged best-point posterior
   mean and SD (atol 1e-7, rtol 1e-5). Validate both runs before optimizing
   either. No fallback, tolerance relaxation or GP refitting.
3. Intersect R6 inequalities analytically using g=r0/(r0-1), including
   m>=4 and n-m>=1. Optimize over the resulting r0 interval and U0 [0.5,8].
4. Evaluate a 121x121 grid. Refine from up to 12 spatially separated starts
   using bounded L-BFGS-B, check all four edges and corners. Preserve local
   optimizer statuses; check a 41x41 finer grid around the selected minimum.
   This is a numerical search, not a global proof.
5. Invert L=(d*(r0-1)/k-l)/b; anchor=100*U0/[A*(1+K_s*salt)].
   Generate the four proposed check rows (two mappings per run at 20 mM).

Figure contract: inspect the reachable surrogate landscape and uncertainty,
show experimental nonuniqueness, and locate candidates relative to observed
R6 points. Python/matplotlib, vector PDF and 300 dpi PNG; observed and
predicted quantities are labelled separately. No images have been rendered
or visually approved locally; inspect cluster outputs before using in a paper.

## Run order (commands for the author)

All Git mutations are performed by the author. Commit/push the new campaign
and test first; do not accidentally include unrelated R3.2 working changes.
On the cluster, update the main clone, then make a fixed copy. The following
assumes the clone has already been updated and has a clean tracked state.

```bash
R610_MAIN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin
R610_REPO=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin_r610
R610_RUN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r6/r6.10-translate
# Refuse to overwrite an existing fixed copy. If one exists, choose a new name.
test ! -e "$R610_REPO" && cp -a "$R610_MAIN" "$R610_REPO"
export R610_REPO
mkdir -p "$R610_RUN"
cd "$R610_RUN"
module load cuda/11.8
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM
export PYTHONPATH="$R610_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
# Run tests on an allocated CPU node, from the repo so test imports resolve.
(cd "$R610_REPO" && python3 -m pytest tests)
# Read-only validation; prints versions and posterior checks.
python3 "$R610_REPO/campaigns/r6.10-translation/translate.py" --validate-only
# CPU-only job: repeats validation, then translates both runs.
sbatch --export=ALL,R610_REPO="$R610_REPO" "$R610_REPO/campaigns/r6.10-translation/translate.sbatch"
```

Do not proceed if tests or validation fail. The CPU sbatch requests 4 CPUs,
16 GB and 2 h; these are initial resource limits, not measured runtime.
Translation refuses an existing output directory. For a deliberate rerun,
use a new `--out` with the Python entry point; retain the old evidence.

### Review and freeze candidates

Review `translation/translation_results.csv`, figures and optimizer status.
Copy `translation/check_candidates.proposed.csv` into the MAIN clone's
`campaigns/r6.10-translation/inputs/check_candidates.csv`, commit and push it.
Update the cluster clone and create a NEW fixed copy, e.g.
`DNA_digitwin_r610_checks`; set/export `R610_REPO` to that copy. Do not edit
or overwrite the translation fixed copy. This extra commit freezes actual
computed conditions; no manually supplied data path is needed by checks.

```bash
python3 "$R610_REPO/campaigns/r6.10-translation/check_translation.py" --list
# Smoke all four inputs; separate checks_smoke directory, NOT scientific data.
sbatch --array=0-3 --export=ALL,R610_REPO="$R610_REPO",R610_SMOKE_STEPS=200000 \
  "$R610_REPO/campaigns/r6.10-translation/check_translation.sbatch"
# After smoke jobs succeed: production, default forward-model duration.
unset R610_SMOKE_STEPS
sbatch --array=0-3 --export=ALL,R610_REPO="$R610_REPO" \
  "$R610_REPO/campaigns/r6.10-translation/check_translation.sbatch"
# After all four production jobs succeed:
python3 "$R610_REPO/campaigns/r6.10-translation/check_translation.py" --collect
```

One GPU per array task, 4 h requested per task (16 allocated GPU-hours if
all hit their limit). Historical R6 configurations report up to about 2.3 h
per evaluation on A40; four checks might therefore be of order 4–10 GPU-hours,
not a benchmark or guarantee. Queue time, parameters and GPU type matter.
Four concurrent jobs reduce elapsed time, not total GPU-hours. Failed jobs
refuse to overwrite their directories: preserve them and use a fresh `--out`
for diagnostic reruns. No automatic resume or resubmission is implemented.

## Outputs and download

Default root is the author-specified `.../r6/r6.10-translate`:

- `translation/translation_results.csv`: one candidate per run, mapped
  parameters, bridge, latent mean/SD, bounds flags, BO-best differences.
  Range flags here refer to the selected 20 mM condition.
- `translation/equivalent_conditions.csv`: 0,20,40,60 mM conditions, flags.
- `translation/translation_report.json`: validation, hashes, versions,
  optimizer diagnostics and complete analytical nonnegative salt families.
- `translation/check_candidates.proposed.csv`: four proposed simulations.
- `translation/figures/{gp_mapping_surface,equivalent_conditions,mapping_vs_r6}.{pdf,png}`.
- `checks/<run>_<mapping>/`: trajectory, S(q), settings, diagnostics,
  `result.json`, `sq_overlay.pdf/png`; `checks/check_results.csv` after collect.
- `checks_smoke/`: short-run diagnostics only. Do not combine with production.

Suggested download destination (author creates/downloads):
`~/Desktop/12_R610_analysis/`, preserving `translation/`, `checks/` and job logs.
No Desktop data are moved/deleted by this campaign. Later vault note name and
image-copy destination require the author's confirmation. Paper TIMELINE,
EXIST_PROBLEMS and vault discrepancies are not edited by these scripts.

## Status and handoff

Implementation prepared; tests, GP validation, translation, visual review and
cluster checks are NOT yet run. R6.10 is not complete. The overview chat can
record implementation ready / awaiting cluster validation. The two-step
translation problem cannot yet be checked off: no computed candidate or
check-simulation evidence exists. Retain nonuniqueness and unreachable-best
handling as the intended scope; do not interpret a low GP mean as success.
