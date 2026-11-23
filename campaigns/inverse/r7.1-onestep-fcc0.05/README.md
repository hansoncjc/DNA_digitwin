# R7.1 one-step inverse design (physics-based endpoint)

TIMELINE task 4.1. BO over the experimental conditions
x = (L_bridge, C_anchor, C_NaCl); each candidate goes through the trained
mapping (R3.2 eval 56, `campaigns/r3.2-physics-validate/mapping/r3.2_eval056.json`)
to (r0, U0, n, m), is simulated with the R6 forward model and scored with
`shift_rmse` (defaults, λ = 1) against the R6.5 target (FCC ρ* = 0.05, CPU
S(q), md5 `780fd41a9f25b73dba663037a2415a25`). Settings and their sources:
`config.py` docstring.

| Setting | Value |
|---|---|
| Box | L_bridge 16.72–183.15 bp (r0 2–3.5), C_anchor 41.57–170.52, C_NaCl 0–40 mM (computed in `config.py`) |
| Initial point | (80 bp, 100, 20 mM) → r0 2.570, U0 2.947, n 10.80, m 5.25 |
| BO | batch 10, `max_evals` 100 (first batch = initial point + 9 Sobol), `bo_seed` 0; simulation seed 42 |
| GPU jobs | `gpu-a40`, 4 h, 10G, `max_job_retries` 2, `max_wait` 6 h |
| Master | `inverse_parallel.sbatch`: `cpu-g2`, 80 h, `--signal=B:TERM@120` |
| out_root | `/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r7/r7.1-onestep-fcc0.05/Optimization_Results` |

## Submission (commands for the author)

Rules (TIMELINE, from §9): each campaign runs from its own pinned repository
copy (C33); configuration changes are committed before the copy is made or
the run is submitted, never edited inside the copy (C37).

1. Locally: commit and push the change (the new `mapping.py`, mapping files,
   this campaign folder, tests). Then on klone update the main clone
   (`git pull`) and check it is clean (`git status`, `git log -1`).

2. Make the pinned copy (refuses to overwrite an existing one):

```bash
R7_MAIN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin
R7_REPO=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin_r7
R7_RUN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r7/r7.1-onestep-fcc0.05
test ! -e "$R7_REPO" && cp -a "$R7_MAIN" "$R7_REPO"
git -C "$R7_REPO" status --short --untracked-files=no   # must print nothing
git -C "$R7_REPO" log -1 --oneline
mkdir -p "$R7_RUN"
```

3. Tests and startup check, on a CPU node (`salloc -A zeelab -p cpu-g2 -c 2 --mem 8G -t 1:00:00`):

```bash
module load cuda/11.8
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM
export PYTHONPATH="$R7_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
(cd "$R7_REPO" && python3 -m pytest tests -q)
python3 "$R7_REPO/inverse.py" "$R7_REPO/campaigns/inverse/r7.1-onestep-fcc0.05/config.py" --check
```

   `--check` writes nothing. Expected lines (numbers from the local check):

```
[inverse]   md5 780fd41a9f25b73dba663037a2415a25; 508 points, ...; first peak q1=0.010224 ...; self-loss 0
[inverse] rmax < L/2 on 343 grid points; worst rmax=10.767 (L/2=50.057) at {'L_bridge': 183.15..., 'C_anchor': 170.51..., 'C_NaCl': 40.0}
[inverse] mapping .../campaigns/r3.2-physics-validate/mapping/r3.2_eval056.json
[inverse]   sha256 14ba39b6d4cea0f0c13fc6c2854a8120caaf5f4daca091bd38df170bb62b2d95; physics; coefficients {...}; BO over conditions ['L_bridge', 'C_anchor', 'C_NaCl']
[inverse] commit <hash>          # no "(tracked files modified)"
[inverse] --check passed; resume state: {... 'next_eval_id': 0}
```

   The mapping sha256 is that of the committed file; if the line ends with
   "(tracked files modified)", stop (C37).

4. Smoke test (short production, separate folder, not scientific data). The
   first batch is 10 GPU jobs of 2 × 10^6 steps; cancel the master once the
   first batch has its 10 blocks:

```bash
sbatch -J inv_r7.1_smoke --time=02:00:00 -o "$R7_RUN/smoke_master_%j.out" \
  --export=ALL,INV_REPO=$R7_REPO,INV_CONFIG=$R7_REPO/campaigns/inverse/r7.1-onestep-fcc0.05/config.py,INV_ARGS="--out $R7_RUN/smoke --steps 2000000" \
  "$R7_REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch"
# after "[bo] ... 10/10 successful" (or a failure) in the smoke log:
scancel --batch --signal=TERM <smoke master id>
```

   Check in `$R7_RUN/smoke/`:
   - `bo_trajectory.csv`: each block's header ends with `m_rel,L_bridge,C_anchor,C_NaCl,rmax,...`;
     eval 0 has L_bridge 80, C_anchor 100, C_NaCl 20 and r0 2.5703, U0 2.9473, n 10.7999, m 5.2521;
     the coefficient columns hold the eval-56 values;
   - `inverse_config.json`: `locked.mapping.sha256` as above, `starts[0].dirty` false;
   - after the TERM: `[launcher] SIGTERM: cancelling ...` in the smoke log and no job left in `squeue -u $USER`.

5. The run:

```bash
sbatch -J inv_r7.1 -o "$R7_RUN/master_%j.out" \
  --export=ALL,INV_REPO=$R7_REPO,INV_CONFIG=$R7_REPO/campaigns/inverse/r7.1-onestep-fcc0.05/config.py \
  "$R7_REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch"
```

   Resubmitting the same command resumes (README "Resuming a run"); check
   `squeue` for GPU jobs of the old master first. To stop by hand:
   `scancel --batch --signal=TERM <master id>`.

## What to watch after the start

- Master log, start: the `--check` lines above, then
  `[inverse] parallel (one GPU job per evaluation); max_evals=100 ...; batch_size=10; bo_seed=0`
  and `[bo] max_evals=100, batch_size=10`.
- `[launcher] submitted i00_d00 -> SLURM <id>` … ten per batch (`i00_d00` to `i09_d00` for the first).
- After each batch: `[bo] <batch label>: k/10 successful; N/100 in total`.
  Failures: `[bo] evaluation <id> failed: ...` and `out_root/error_d0.txt`;
  a mapped point outside `check_mapped_params` cannot occur inside this box
  (tested), so a failure here is a simulation or metric failure.
- `out_root/loss_components.txt` grows by 10 rows per batch.
- At the end: `inverse_summary.txt` with the best evaluation and its
  `conditions: L_bridge = ... C_anchor = ... C_NaCl = ...`.

## Expected time

About 12–16 h for 100 evaluations (estimate): R6.5, same forward model and
batch size, took 14.3 h on `ckpt-g2`; R6.4 on `gpu-a40` took 15.8 h. Queue
time on `gpu-a40` adds to this. The 80 h master walltime leaves room for
one resume.

## After the run (another session)

R7.2: whether the route reaches the target and which conditions it
returns; the returned conditions checked through the ground-truth mapping
(`campaigns/r3.1-gt/mapping/ground_truth.json`) as in R6.10. Salt and
C_anchor are degenerate in U0 (C_anchor (1 + K_s C_NaCl)), so the returned
point is one point of a curve of equivalent conditions.
