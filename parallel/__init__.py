"""
Parallel Slurm launcher for per-dataset BO evaluations.

Public API:
    from parallel import Job, LauncherConfig, submit_jobs
"""
from .submit_parallel import (  # noqa: F401
    Job,
    LauncherConfig,
    submit_jobs,
    prepare_jobs,
    submit_all,
    poll_until_done,
    write_summary,
    inspect_flags,
    make_run_dir,
)
