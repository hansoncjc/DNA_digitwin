# Known issues

Latent problems found during the 2026-09-16 review of `bo.py`. None of them
is currently reachable from the campaigns we run, which is why they have never
shown up in a real run. Recorded here so they are not rediscovered from
scratch.

Issues already closed are listed at the bottom for context.

---

## D. `remaining_bo_iters` assumes every recorded success is an acquisition step

**Where** `bo.py:1103`.

**What** Not a bug in the current code, but a constraint worth remembering:

```python
return max(0, n_iters - (n_successful - 1))
```

The arithmetic treats the first successful evaluation in `bo_trajectory.csv` as
the cold-start `init` point and every later one as an acquisition step. Any
evaluation written to the trajectory that is *not* an acquisition step silently
eats one of the remaining iterations on resume.

**History** The removed Sobol init-batch feature violated exactly this: its 8
Sobol points were logged as ordinary successful evaluations, so resuming such a
campaign lost 8 acquisition steps. Deleting the feature restored the assumption;
there is no leftover code to change.

**When it would come back** Any future feature that evaluates extra design
points outside the acquisition loop (a replacement init strategy, a
space-filling pre-pass, injected candidates from another model) must either
count them separately or tag them in the CSV, and update this function to match.

**Note on old data** The `testing/inverse/02_fcc/05..08` campaigns did run with
the Sobol batch, so their `bo_trajectory.csv` files contain `eval_001..008` as
successful non-acquisition evaluations. Resuming those specific runs would
mis-count remaining iterations.

---

## E. `ParamSpace.phys_to_unit` does not clamp to the unit cube

**Where** `bo.py:212-214`.

**What** The mapping is a plain affine rescale with no bounds check:

```python
return (x_phys - self._lo_t) / (self._hi_t - self._lo_t)
```

A physical value outside the current `bounds` maps outside `[0, 1]`.

**When it triggers** Resuming from a `bo_trajectory.csv` that was produced with
different (wider) bounds than the current `ParamSpace`. The out-of-range points
are fed to the GP as training data, while `optimize_acqf` still searches only
`[0, 1]`, so the surrogate is conditioned on regions the optimizer cannot
propose. Nothing crashes; the search just behaves oddly.

**Why we have never hit it** Each campaign uses its own `OUT_ROOT`, so a
trajectory is never reused across a bounds change. The narrow/full variants of
the same campaign are separate directories.

**Possible fix** Clamp to `[0, 1]` and warn when a value is out of range, so a
bounds mismatch is visible instead of silent.

---

## Fixed in this review

- **C.** `load_warm_start_from_trajectory` could not restore non-fixed
  `local` parameters: `ps._names` labelled them `U0:d0`, while
  `bo_trajectory.csv` only has bare columns, so the lookup raised
  `KeyError`. Closed on 2026-10-09: `ParamSpace` takes a flat
  `{name: spec}` config with no global/local split, so every vector label
  is a bare column name.

- **A.** `_collect_parallel_eval_loss` matched finished Slurm jobs by comparing
  `plan["save_dir"]` against `str(job.sim_dir)`. `pathlib` normalizes away the
  leading `./` that every driver's `OUT_ROOT` carries, so the lookup never
  matched and every `parallel=True` evaluation reported all datasets as failed
  even though the jobs had succeeded. Reverted to keying on `ds_id`, which the
  launcher echoes back verbatim. Fixed in `c195645`.
- **B.** A failed parallel evaluation did not clear
  `objective._iteration_data`, so its records leaked into every later
  trajectory block: consecutive failures grew `bo_trajectory.csv`
  quadratically, and a successful block could carry stale `loss=FAILED` rows
  from an earlier evaluation. Both failure paths now clear the list before
  raising, matching the sequential path. Fixed in `c195645`.
