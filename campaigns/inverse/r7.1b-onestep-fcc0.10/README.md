# R7.1b one-step inverse design (physics-based endpoint), FCC rho* = 0.10 (R6.4 target)

TIMELINE task 4.1 extended to the other R6 targets (author, 2026-11-23). As R7.1
(`../r7.1-onestep-fcc0.05/`) except the target: BO over x = (L_bridge, C_anchor, C_NaCl)
through the R3.2 eval-56 mapping file, R6 forward model, `shift_rmse` (defaults, λ = 1).
Settings and sources: `config.py` docstring.

| Setting | Value |
|---|---|
| Target | `inverse-targets/fcc_rho0.10_D45.7_pad2.2/cpu/target_sq.npy`, md5 `9131767b29aa1004d8ffb092891ea359`, N 5017 (as R6.4) |
| Box | `mapping.physics_condition_box` on the R6.4 ranges r0 [2, 3.5], U0 [0.5, 8], salt 0–40 mM, L_bridge ≥ 5 bp: L_bridge 16.72–183.15 bp, C_anchor 41.57–170.52, C_NaCl 0–40 mM (the R7.1 box; the floor does not bind) |
| Initial point | (80 bp, 100, 20 mM), as R7.1 |
| BO | batch 10, `max_evals` 100, `bo_seed` 0; simulation seed 42 |
| GPU jobs | `gpu-a40`, 4 h, 10G, `max_job_retries` 2, `max_wait` 6 h |
| Master | `inverse_parallel.sbatch`: `cpu-g2`, 80 h, `--signal=B:TERM@120` |
| out_root | `/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r7/r7.1b-onestep-fcc0.10/Optimization_Results` |
| Pinned copy | `DNA_digitwin_r7b` (C33: one copy per campaign) |

## Submission (commands for the author)

The steps are those of `../r7.1-onestep-fcc0.05/README.md`, with these names.

1. Locally: commit and push (C37: before the copy is made). On klone: `git pull`
   in the main clone, `git status` clean, `git log -1`.

2. Pinned copy and run folder:

```bash
R7_MAIN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin
R7_REPO=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin_r7b
R7_RUN=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r7/r7.1b-onestep-fcc0.10
R7_CFG=$R7_REPO/campaigns/inverse/r7.1b-onestep-fcc0.10/config.py
test ! -e "$R7_REPO" && cp -a "$R7_MAIN" "$R7_REPO"
git -C "$R7_REPO" status --short --untracked-files=no   # must print nothing
git -C "$R7_REPO" log -1 --oneline
mkdir -p "$R7_RUN"
```

3. Tests and startup check on a CPU node
   (`salloc -A zeelab -p cpu-g2 -c 2 --mem 8G -t 1:00:00`; in the new shell set
   `R7_REPO`, `R7_CFG` again):

```bash
module load cuda/11.8
source /gscratch/zeelab/hanson/codes/HOOMD-start/hoomd-venv/bin/activate
export MC_DFM_ROOT=/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM
export PYTHONPATH="$R7_REPO:$MC_DFM_ROOT:${PYTHONPATH:-}"
(cd "$R7_REPO" && python3 -m pytest tests -q)
python3 "$R7_REPO/inverse.py" "$R7_CFG" --check
```

   Expected lines (local check, hoomd stubbed):

```
[inverse]   md5 9131767b29aa1004d8ffb092891ea359; 508 points, ...; first peak q1=0.012891 ...; self-loss 0
[inverse] rmax < L/2 on 343 grid points; worst rmax=10.767 (L/2=50.057) at {'L_bridge': 183.15..., 'C_anchor': 170.51..., 'C_NaCl': 40.0}
[inverse]   sha256 14ba39b6d4cea0f0c13fc6c2854a8120caaf5f4daca091bd38df170bb62b2d95; physics; ...
[inverse] commit <hash>          # no "(tracked files modified)"
[inverse] --check passed; resume state: {... 'next_eval_id': 0}
```

4. Smoke test: optional. The run path (bo mode "map" with conditions, worker,
   launcher, TERM forwarding) is the one the R7.1 smoke test passed at
   `b4885bd`; this commit adds only `mapping.physics_condition_box`, which runs
   when the config is loaded and is covered by `--check`. If wanted, as R7.1
   step 4 with `-J inv_r7.1b_smoke` and `--out $R7_RUN/smoke --steps 2000000`.

5. The run:

```bash
sbatch -J inv_r7.1b -o "$R7_RUN/master_%j.out" \
  --export=ALL,INV_REPO=$R7_REPO,INV_CONFIG=$R7_CFG \
  "$R7_REPO/campaigns/inverse/inverse-example/inverse_parallel.sbatch"
```

   Stop by hand: `scancel --batch --signal=TERM <master id>`. Resubmitting the
   same command resumes.

## What to watch

As R7.1 (README there): the `--check` lines at the top of the master log,
`[bo] <batch>: k/10 successful; N/100 in total` after each batch,
`loss_components.txt` growing by 10 rows per batch, and `inverse_summary.txt`
with `conditions: ...` at the end. R7.1 took 10.0 h for 100 evaluations on
`gpu-a40` (2026-11-23); R6 runs took 12–16 h.
