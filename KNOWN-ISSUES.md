# Known issues

Latent problems found during the 2026-09-16 review of `bo.py`. None of them
is currently reachable from the campaigns we run, which is why they have never
shown up in a real run. Recorded here so they are not rediscovered from
scratch.

Issues already closed are listed at the bottom for context.

---

No open issue (D closed on 2026-10-11).

---

## Fixed in this review

- **D.** `remaining_bo_iters` (`max(0, n_iters - (n_successful - 1))`)
  treated the first successful evaluation in `bo_trajectory.csv` as the
  initial point and every later one as an acquisition step, so an
  evaluation made outside the acquisition loop silently ate one
  acquisition on resume; the Sobol init batch removed before 2026-09-16 did
  exactly that. Closed on 2026-10-11 (C24, batch BO): `run_bo` and
  `run_bo_resumable` count `max_evals` successful evaluations of any kind
  (initial point, the Sobol points of the first batch, acquisitions), and
  `remaining_evals(max_evals, n_successful) = max_evals - n_successful`
  replaces `remaining_bo_iters`. How a point was produced no longer enters
  the count, also across a batch cut off mid-way. Old data: the
  `testing/inverse/02_fcc/05..08` trajectories (old Sobol batch) would now
  count every success against `max_evals`, which is the new rule.

- **E.** `ParamSpace.phys_to_unit` does not clamp, so a warm start from a
  trajectory run with a wider box fed the GP points outside the unit cube
  that `optimize_acqf` cannot propose. Closed on 2026-10-09:
  `load_warm_start_from_trajectory` raises when a point is outside the
  current bounds by more than `WARM_START_UNIT_TOL` (1e-6 in the unit cube;
  smaller round-off is clamped) and when a fixed value differs from the one
  recorded in the trajectory. `phys_to_unit` itself is unchanged.

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
