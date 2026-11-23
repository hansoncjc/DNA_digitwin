# Sourced by the master sbatch scripts (run_r4.sbatch, run_r32.sbatch,
# inverse_parallel.sbatch); not run on its own.
#
#   run_forward_term CMD [ARG...]
#
# Runs CMD in the background and waits for it. A TERM sent to this shell is
# passed on to CMD, so the python master can scancel its GPU jobs and exit
# before Slurm kills the job. Returns CMD's exit status.
#
# Slurm sends that TERM 120 s before the walltime because the sbatch script
# has `#SBATCH --signal=B:TERM@120` (B: the batch shell only). Without it the
# master got no SIGTERM it could act on at the walltime: smoke test B
# (2026-11-22) left its GPU jobs running. A shell acts on a trapped signal only
# between commands or during `wait`, hence the background job.
run_forward_term() {
    "$@" &
    _FORWARD_TERM_PID=$!
    trap 'kill -TERM "$_FORWARD_TERM_PID" 2>/dev/null' TERM
    local status=0
    while :; do
        # wait returns early (status > 128) when the trap runs; wait again
        # until CMD itself has exited, so its cleanup is not cut short.
        wait "$_FORWARD_TERM_PID" && status=0 || status=$?
        kill -0 "$_FORWARD_TERM_PID" 2>/dev/null || break
    done
    trap - TERM
    return "$status"
}
