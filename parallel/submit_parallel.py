"""
Parallel Slurm launcher library for BO evaluations.

This is a generalized, importable version of the launcher validated in
`Multithread_test/submit_parallel.py`. It drops the built-in smoke-test
`param_grid` path in favor of accepting arbitrary per-job specs from the
caller (one per dataset), making it reusable from `bo.make_global_objective`.

Typical usage from Python::

    from parallel import LauncherConfig, submit_jobs

    cfg = LauncherConfig(
        run_dir=Path(out_root) / f"eval_{eval_id:03d}" / "_jobs",
        partition="gpu-a40", account="zeelab", gpus=1,
        mem="40G", time="02:00:00",
        module_loads=["ssmc/miniconda/3.9", "cuda/11.8.0"],
        venv_activate=".../hoomd-venv/bin/activate",
        code_root="/.../DNA_digitwin",
        mc_dfm_root="/.../MC-DFM",
    )
    finished = submit_jobs(job_specs, cfg)

Per-job `spec` dict format::

    {
        "name":   "d0_eval000",                        # unique within run_dir
        "ds_id":  "d0",                                # opaque tag, echoed back
        "worker_config": {                             # passed verbatim to worker.py
            "outdir":     "/abs/path/sim_dir",
            "run_kwargs": {density, U_0, r0, n, m, ...},
            "scattering": {"method": "saxsfft", "kwargs": {...}},
            "loss":       {"exp_path", "trim_tail", "datatype",
                           "ffpath", "metric", "scattering_method",
                           "q_min", "q_max"}
        },
    }

The launcher writes `<run_dir>/summary.json` and `<run_dir>/launcher.log`,
and returns a list of `Job` objects with `done_status`, `slurm_state`,
`host`, `run_time_seconds`, etc. populated. Callers read per-job losses
by reading `<sim_dir>/DONE` (JSON) written by the worker.

Python: written for the HOOMD venv (3.9+).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

THIS_DIR = Path(__file__).resolve().parent
WORKER_PATH = THIS_DIR / "worker.py"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Job:
    idx:          int
    name:         str
    ds_id:        str
    sim_dir:      Path
    config_path:  Path
    sbatch_path:  Path
    out_path:     Path
    slurm_id:     Optional[str] = None
    state:        str = "PENDING_SUBMIT"
    slurm_state:  str = ""               # last observed squeue state
    done_status:      Optional[str] = None  # "DONE" / "FAILED" / None
    run_time_seconds: Optional[float] = None
    host:             str = ""


@dataclass
class LauncherConfig:
    run_dir:       Path
    partition:     str = "gpu-a40"
    account:       str = "zeelab"
    gpus:          int = 1
    mem:           str = "40G"
    time:          str = "02:00:00"
    module_loads:  list = field(default_factory=list)
    venv_activate: str = ""
    code_root:     str = ""              # added to PYTHONPATH inside sbatch
    mc_dfm_root:   str = ""              # added to PYTHONPATH inside sbatch
    poll_interval: float = 15.0
    max_wait:      float = 4 * 3600


# ---------------------------------------------------------------------------
# Script + config generation
# ---------------------------------------------------------------------------

def build_sbatch_script(job: Job, cfg: LauncherConfig) -> str:
    module_lines = "\n".join(f"module load {m}" for m in cfg.module_loads)
    venv_line    = f"source {cfg.venv_activate}" if cfg.venv_activate else ""

    pp_parts = [p for p in (cfg.code_root, cfg.mc_dfm_root) if p]
    pythonpath_line = (
        f'export PYTHONPATH="{":".join(pp_parts)}:${{PYTHONPATH:-}}"'
        if pp_parts else ""
    )

    return f"""#!/bin/bash
#SBATCH -J {job.name}
#SBATCH -A {cfg.account}
#SBATCH -p {cfg.partition}
#SBATCH -G {cfg.gpus}
#SBATCH --time={cfg.time}
#SBATCH --mem={cfg.mem}
#SBATCH -o {job.out_path}
#SBATCH -e {job.out_path}

set -euo pipefail

echo "=========================================="
echo "Job:       {job.name}"
echo "JobID:     ${{SLURM_JOB_ID}}"
echo "Node:      ${{SLURM_NODELIST}}"
echo "Start:     $(date)"
echo "Config:    {job.config_path}"
echo "=========================================="

{module_lines}
{venv_line}
{pythonpath_line}

echo "python: $(which python3)"
echo "PYTHONPATH: ${{PYTHONPATH:-<unset>}}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python3 {WORKER_PATH} {job.config_path}
EXIT_CODE=$?

echo "=========================================="
echo "End:       $(date)"
echo "ExitCode:  $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
"""


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

def sbatch_submit(sbatch_path: Path) -> str:
    out = subprocess.check_output(["sbatch", str(sbatch_path)], text=True).strip()
    for tok in out.split():
        if tok.isdigit():
            return tok
    raise RuntimeError(f"Could not parse sbatch output: {out!r}")


def squeue_states(job_ids: List[str]) -> dict:
    if not job_ids:
        return {}
    cmd = ["squeue", "-h", "-o", "%i %T", "-j", ",".join(job_ids)]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if "Invalid job id specified" in (e.output or ""):
            return {}
        print(f"[launcher] squeue error: {e.output}", file=sys.stderr)
        return {}

    states: dict = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0]] = parts[1]
    return states


def sacct_final_state(job_id: str) -> str:
    try:
        out = subprocess.check_output(
            ["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except Exception:
        return ""
    line = out.strip().splitlines()[0] if out.strip() else ""
    return line.strip()


# ---------------------------------------------------------------------------
# Flag-file inspection
# ---------------------------------------------------------------------------

def inspect_flags(job: Job) -> None:
    done   = job.sim_dir / "DONE"
    failed = job.sim_dir / "FAILED"

    if done.exists():
        job.done_status = "DONE"
        try:
            data = json.loads(done.read_text())
            job.run_time_seconds = float(data.get("run_time_seconds", 0.0))
            job.host = str(data.get("host", ""))
        except Exception:
            pass
    elif failed.exists():
        job.done_status = "FAILED"
        try:
            for line in failed.read_text().splitlines():
                if line.startswith("run_time_seconds="):
                    job.run_time_seconds = float(line.split("=", 1)[1])
                elif line.startswith("host="):
                    job.host = line.split("=", 1)[1]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main launcher primitives
# ---------------------------------------------------------------------------

def make_run_dir(path: Path, clean: bool) -> None:
    if path.exists() and clean:
        shutil.rmtree(path)
    (path / "configs").mkdir(parents=True, exist_ok=True)
    (path / "jobs").mkdir(parents=True, exist_ok=True)


def prepare_jobs(cfg: LauncherConfig, job_specs: Iterable[dict]) -> List[Job]:
    """
    Materialize one sbatch file + one JSON config per spec.

    `job_specs[i]["worker_config"]["outdir"]` is used as the sim directory
    (the worker writes DONE/FAILED there). This lets callers keep their
    existing output-tree convention (e.g. out_root/eval_XXX/ds.id).
    """
    jobs: List[Job] = []
    for i, spec in enumerate(job_specs):
        name = str(spec["name"])
        ds_id = str(spec.get("ds_id", name))
        worker_cfg = dict(spec["worker_config"])
        sim_dir = Path(worker_cfg["outdir"])
        sim_dir.mkdir(parents=True, exist_ok=True)

        job = Job(
            idx=i,
            name=name,
            ds_id=ds_id,
            sim_dir=sim_dir,
            config_path=cfg.run_dir / "configs" / f"{name}.json",
            sbatch_path=cfg.run_dir / "jobs"    / f"{name}.sbatch",
            out_path=cfg.run_dir / "jobs"    / f"{name}.out",
        )
        job.config_path.write_text(json.dumps(worker_cfg, indent=2))
        job.sbatch_path.write_text(build_sbatch_script(job, cfg))
        jobs.append(job)
    return jobs


def submit_all(jobs: List[Job]) -> None:
    for job in jobs:
        job.slurm_id = sbatch_submit(job.sbatch_path)
        job.state = "SUBMITTED"
        print(f"[launcher] submitted {job.name} -> SLURM {job.slurm_id}")


def poll_until_done(
    jobs: List[Job],
    cfg: LauncherConfig,
    poll_interval: Optional[float] = None,
    max_wait: Optional[float] = None,
) -> None:
    poll_interval = cfg.poll_interval if poll_interval is None else poll_interval
    max_wait      = cfg.max_wait      if max_wait is None      else max_wait

    log_path = cfg.run_dir / "launcher.log"
    t0 = time.time()

    with log_path.open("w") as log:
        while True:
            elapsed = time.time() - t0

            # 1) Flag files are authoritative for completion. Overwrite
            #    slurm_state immediately so a stale RUNNING snapshot from
            #    the previous poll cycle does not leak into summary.json.
            for job in jobs:
                if job.done_status is None:
                    inspect_flags(job)
                    if job.done_status == "DONE":
                        job.slurm_state = "COMPLETED"
                    elif job.done_status == "FAILED":
                        job.slurm_state = "FAILED"

            # 2) Check squeue for still-active ids.
            active_ids = [j.slurm_id for j in jobs
                          if j.slurm_id and j.done_status is None]
            states = squeue_states(active_ids) if active_ids else {}
            for job in jobs:
                if job.slurm_id and job.done_status is None:
                    job.slurm_state = states.get(job.slurm_id, "")
                    # Gone from squeue AND no flag file -> the job crashed
                    # before worker.py could write one (e.g. OOM, GPU init,
                    # SIGTERM after TIMEOUT). Write a FAILED flag on its
                    # behalf so the caller sees a terminal state.
                    if (not job.slurm_state
                            and not (job.sim_dir / "DONE").exists()
                            and not (job.sim_dir / "FAILED").exists()):
                        fs = sacct_final_state(job.slurm_id)
                        job.done_status = "FAILED"
                        (job.sim_dir / "FAILED").write_text(
                            f"No flag file written. sacct state: {fs}\n"
                        )

            # 3) Status line.
            n_done    = sum(1 for j in jobs if j.done_status == "DONE")
            n_failed  = sum(1 for j in jobs if j.done_status == "FAILED")
            n_running = sum(1 for j in jobs
                            if j.done_status is None and j.slurm_state == "RUNNING")
            n_pending = len(jobs) - n_done - n_failed - n_running

            status_line = (
                f"[{elapsed:7.1f}s] done={n_done} failed={n_failed} "
                f"running={n_running} pending={n_pending}"
            )
            print(status_line)
            log.write(status_line + "\n")
            log.flush()

            # 4) Exit conditions.
            if all(j.done_status is not None for j in jobs):
                return
            if elapsed > max_wait:
                print(f"[launcher] max_wait={max_wait}s exceeded; giving up")
                return

            time.sleep(poll_interval)


def write_summary(jobs: List[Job], cfg: LauncherConfig) -> None:
    summary = {
        "run_dir":          str(cfg.run_dir),
        "n_jobs":           len(jobs),
        "slurm_time_limit": cfg.time,
        "slurm_mem":        cfg.mem,
        "jobs": [
            {
                "name":              j.name,
                "ds_id":             j.ds_id,
                "slurm_id":          j.slurm_id,
                "status":            j.done_status,
                "slurm_state":       j.slurm_state,
                "host":              j.host,
                "run_time_seconds":  j.run_time_seconds,
                "sim_dir":           str(j.sim_dir),
                "out_log":           str(j.out_path),
            }
            for j in jobs
        ],
    }
    (cfg.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n================ FINAL SUMMARY ================")
    print(
        f"  Time Limit: {cfg.time}    Mem: {cfg.mem}    "
        f"Partition: {cfg.partition}    Account: {cfg.account}"
    )
    for j in jobs:
        t = (
            f"{j.run_time_seconds:.1f}s"
            if j.run_time_seconds is not None
            else "n/a"
        )
        print(
            f"  {j.name:<20}  slurm={str(j.slurm_id or '-'):>8}  "
            f"status={str(j.done_status):<6}  host={j.host or '-':<12}  "
            f"run time={t}"
        )
    print("===============================================\n")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def submit_jobs(
    job_specs: Iterable[dict],
    cfg: LauncherConfig,
    *,
    clean: bool = True,
    poll_interval: Optional[float] = None,
    max_wait: Optional[float] = None,
) -> List[Job]:
    """
    One-call convenience: prepare sbatch files + json configs, submit them
    all, poll until every job has a terminal state, write summary.json,
    and return the list of finished `Job` objects.

    Callers typically iterate the returned jobs and read each
    `job.sim_dir / "DONE"` (JSON) for per-job `loss`, or check
    `job.done_status == "FAILED"` for the penalty branch.
    """
    specs = list(job_specs)
    make_run_dir(cfg.run_dir, clean=clean)
    jobs = prepare_jobs(cfg, specs)
    print(f"[launcher] run_dir = {cfg.run_dir}")
    print(f"[launcher] prepared {len(jobs)} jobs")

    submit_all(jobs)
    poll_until_done(jobs, cfg, poll_interval=poll_interval, max_wait=max_wait)
    write_summary(jobs, cfg)
    return jobs
