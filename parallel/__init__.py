"""
Parallel Slurm launcher for per-dataset BO evaluations.

Public API:
    from parallel import Job, LauncherConfig, submit_jobs

The master calls ``install_signal_handlers()`` once (main thread) so that
SIGTERM cancels its GPU jobs; ``atexit`` does the same on a normal exit or an
uncaught exception (``cancel_live_jobs``).
"""
from .submit_parallel import (  # noqa: F401
    Job,
    LauncherConfig,
    submit_jobs,
    submit_jobs_with_retry,
    clear_job_flags,
    prepare_jobs,
    submit_all,
    poll_until_done,
    write_summary,
    inspect_flags,
    make_run_dir,
    cancel_unfinished,
    cancel_live_jobs,
    install_signal_handlers,
    live_jobs,
    scancel,
    LauncherShutdown,
    OWNER_FILE,
)
