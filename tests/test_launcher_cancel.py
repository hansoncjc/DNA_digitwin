"""The launcher cancels its GPU jobs when it gives up waiting, when it raises,
and when the master exits (TIMELINE 1.1, C32). Without a give-up nothing
changes: no scancel, same Job fields, same summary.json, same Slurm calls."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from parallel import submit_parallel as sp

REPO = Path(__file__).resolve().parents[1]


class FakeSlurm:
    """``subprocess.check_output`` stand-in for sbatch / squeue / scancel / sacct.

    ``queue`` maps job id -> state. scancel removes the ids from the queue,
    unless ``linger`` (id -> number of further squeue answers that still list
    it, as COMPLETING) says otherwise. ``sbatch_plan`` / ``scancel_plan`` hold
    "fail" steps taken in order before the calls succeed. ``on_submit(name,
    job_id)`` runs after each successful sbatch.
    """

    def __init__(self, sbatch_plan=(), scancel_plan=(), on_submit=None, linger=None):
        self.sbatch_plan = list(sbatch_plan)
        self.scancel_plan = list(scancel_plan)
        self.on_submit = on_submit
        self.linger = dict(linger or {})
        self.queue = {}
        self.calls = []
        self.next_id = 101

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        prog = cmd[0]
        if prog == "sbatch":
            if self.sbatch_plan and self.sbatch_plan.pop(0) == "fail":
                raise subprocess.CalledProcessError(1, cmd, output="Socket timed out")
            job_id = str(self.next_id)
            self.next_id += 1
            self.queue[job_id] = "PENDING"
            if self.on_submit:
                self.on_submit(Path(cmd[1]).stem, job_id)
            return f"Submitted batch job {job_id}\n"
        if prog == "squeue":
            if "-n" in cmd:                       # sbatch_submit's "already queued?" check
                return ""
            ids = cmd[cmd.index("-j") + 1].split(",")
            lines = []
            for i in ids:
                if i in self.queue:
                    lines.append(f"{i} {self.queue[i]}")
                elif self.linger.get(i, 0) > 0:
                    self.linger[i] -= 1
                    lines.append(f"{i} COMPLETING")
            return "\n".join(lines) + ("\n" if lines else "")
        if prog == "scancel":
            if self.scancel_plan and self.scancel_plan.pop(0) == "fail":
                raise subprocess.CalledProcessError(1, cmd, output="Socket timed out")
            for i in cmd[1:]:
                self.queue.pop(i, None)
            return ""
        if prog == "sacct":
            return ""
        raise AssertionError(f"unexpected command {cmd}")

    def names(self):
        return [c[0] for c in self.calls]

    def scancels(self):
        return [c[1:] for c in self.calls if c[0] == "scancel"]


@pytest.fixture
def slurm(monkeypatch):
    sleeps = []
    hooks = []

    def fake_sleep(s):
        sleeps.append(s)
        for hook in hooks:
            hook()

    monkeypatch.setattr(sp.time, "sleep", fake_sleep)

    def install(fake):
        monkeypatch.setattr(sp.subprocess, "check_output", fake)
        return fake

    return install, sleeps, hooks


def _specs(tmp_path, n=2, eval_id=7):
    return [{
        "name": f"i{eval_id}_d{k:02d}",
        "ds_id": f"d{k}",
        "worker_config": {"outdir": str(tmp_path / f"eval_{eval_id:03d}" / f"d{k}")},
    } for k in range(n)]


def _cfg(tmp_path, eval_id=7):
    return sp.LauncherConfig(run_dir=tmp_path / f"eval_{eval_id:03d}" / "_jobs", poll_interval=0)


def _done(sim_dir, loss=0.1):
    Path(sim_dir, "DONE").write_text(json.dumps(
        {"run_time_seconds": 5.0, "host": "g3001", "loss": loss, "result": {}}))


# ------------------------------------------------------------ default case


def test_no_give_up_means_no_scancel_and_the_same_outputs(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    specs = _specs(tmp_path)

    def finish_all():
        for k, job_id in enumerate(list(fake.queue)):
            _done(tmp_path / "eval_007" / f"d{k}", loss=0.1 * (k + 1))
            fake.queue.pop(job_id)
    hooks.append(finish_all)

    jobs = sp.submit_jobs(specs, _cfg(tmp_path), poll_interval=0, max_wait=3600)

    # The Slurm calls of the code before this change: two sbatch, one squeue.
    assert fake.names() == ["sbatch", "sbatch", "squeue"]
    assert sleeps == [0]
    assert sp.live_jobs() == {}
    assert [(j.name, j.slurm_id, j.done_status, j.slurm_state, j.state, j.host,
             j.run_time_seconds) for j in jobs] == [
        ("i7_d00", "101", "DONE", "COMPLETED", "SUBMITTED", "g3001", 5.0),
        ("i7_d01", "102", "DONE", "COMPLETED", "SUBMITTED", "g3001", 5.0),
    ]
    run_dir = tmp_path / "eval_007" / "_jobs"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary == {
        "run_dir": str(run_dir), "n_jobs": 2,
        "slurm_time_limit": "02:00:00", "slurm_mem": "40G",
        "jobs": [{
            "name": f"i7_d{k:02d}", "ds_id": f"d{k}", "slurm_id": str(101 + k),
            "status": "DONE", "slurm_state": "COMPLETED", "host": "g3001",
            "run_time_seconds": 5.0, "sim_dir": str(tmp_path / "eval_007" / f"d{k}"),
            "out_log": str(run_dir / "jobs" / f"i7_d{k:02d}.out"),
        } for k in range(2)],
    }
    assert "cancel" not in (run_dir / "launcher.log").read_text()


def test_no_give_up_with_retry_wrapper_is_unchanged(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm(on_submit=lambda name, job_id: _done(
        tmp_path / "eval_007" / f"d{int(name[-2:])}")))
    jobs = sp.submit_jobs_with_retry(_specs(tmp_path), _cfg(tmp_path), max_job_retries=2,
                                     poll_interval=0, max_wait=3600)
    assert fake.names() == ["sbatch", "sbatch"]
    assert [j.done_status for j in jobs] == ["DONE", "DONE"]
    assert not (tmp_path / "eval_007" / "_jobs" / "retry_1").exists()


# ------------------------------------------------------------ give up / error


def test_give_up_cancels_only_the_unfinished_job(tmp_path, slurm):
    install, sleeps, hooks = slurm
    # d0 finishes at once; d1 stays PENDING past max_wait.
    fake = install(FakeSlurm(on_submit=lambda name, job_id: (
        _done(tmp_path / "eval_007" / "d0") if name == "i7_d00" else None)))

    jobs = sp.submit_jobs(_specs(tmp_path), _cfg(tmp_path), poll_interval=0, max_wait=-1)

    assert fake.scancels() == [["102"]]
    # scancel, then squeue until the job is gone, then return.
    assert fake.calls[-2:] == [["scancel", "102"], ["squeue", "-h", "-o", "%i %T", "-j", "102"]]
    assert [(j.done_status, j.slurm_state) for j in jobs] == [
        ("DONE", "COMPLETED"), (None, "CANCELLED")]
    assert sp.live_jobs() == {}
    log = (tmp_path / "eval_007" / "_jobs" / "launcher.log").read_text()
    assert "max_wait exceeded: cancelling 1 unfinished job(s): i7_d01 (102)" in log
    assert "cancelled and gone from squeue: 102" in log


def test_retry_is_submitted_only_after_the_old_job_left_the_queue(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm(linger={"101": 2}))   # COMPLETING for two more checks
    jobs = sp.submit_jobs_with_retry(_specs(tmp_path, n=1), _cfg(tmp_path), max_job_retries=1,
                                     poll_interval=0, max_wait=-1)

    first_retry = next(i for i, c in enumerate(fake.calls)
                       if c[0] == "sbatch" and c[1].endswith("i7_d00_r1.sbatch"))
    last_check = max(i for i, c in enumerate(fake.calls[:first_retry])
                     if c[0] == "squeue" and c[-1] == "101")
    cancel = fake.calls.index(["scancel", "101"])
    assert cancel < last_check < first_retry
    # Three squeue checks after scancel: COMPLETING, COMPLETING, gone.
    checks = [c for c in fake.calls[cancel:first_retry] if c[0] == "squeue"]
    assert len(checks) == 3
    assert sleeps.count(sp.CANCEL_POLL_WAIT) == 2
    # The retry gave up as well and was cancelled too.
    assert fake.scancels() == [["101"], ["102"]]
    assert jobs[0].name == "i7_d00_r1" and jobs[0].slurm_state == "CANCELLED"
    assert sp.live_jobs() == {}


def test_launcher_error_cancels_the_jobs_already_submitted(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm(sbatch_plan=["ok"] + ["fail"] * sp.SBATCH_ATTEMPTS))
    with pytest.raises(subprocess.CalledProcessError):
        sp.submit_jobs(_specs(tmp_path), _cfg(tmp_path), poll_interval=0, max_wait=3600)
    assert fake.scancels() == [["101"]]
    assert sp.live_jobs() == {}
    log = (tmp_path / "eval_007" / "_jobs" / "launcher.log").read_text()
    assert "launcher error: cancelling 1 unfinished job(s): i7_d00 (101)" in log


def test_failed_scancel_is_bounded_and_keeps_the_job_for_the_exit_hook(tmp_path, slurm, capsys):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm(scancel_plan=["fail"] * sp.SCANCEL_ATTEMPTS))
    jobs = sp.submit_jobs(_specs(tmp_path, n=1), _cfg(tmp_path), poll_interval=0, max_wait=-1)
    assert len(fake.scancels()) == sp.SCANCEL_ATTEMPTS
    assert sleeps.count(sp.SCANCEL_RETRY_WAIT) == sp.SCANCEL_ATTEMPTS - 1
    assert jobs[0].done_status is None and jobs[0].slurm_state != "CANCELLED"
    assert sp.live_jobs() == {"101": "i7_d00"}       # atexit tries again
    assert "scancel 101 failed; continuing" in capsys.readouterr().err


def test_job_that_never_leaves_the_queue_is_bounded(tmp_path, slurm, capsys):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm(linger={"101": 10 ** 6}))
    sp.submit_jobs(_specs(tmp_path, n=1), _cfg(tmp_path), poll_interval=0, max_wait=-1)
    after = fake.calls[fake.calls.index(["scancel", "101"]) + 1:]
    assert len(after) == sp.CANCEL_POLLS and all(c[0] == "squeue" for c in after)
    assert "still listed by squeue" in capsys.readouterr().err
    assert sp.live_jobs() == {}


# ------------------------------------------------------------ master exit


def test_cancel_live_jobs_cancels_everything_once(slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    for job_id, name in (("201", "i3_d00"), ("202", "i3_d01")):
        sp._register(sp.Job(idx=0, name=name, ds_id="d0", sim_dir=Path("."), config_path=Path("."),
                            sbatch_path=Path("."), out_path=Path("."), slurm_id=job_id))
    assert sp.cancel_live_jobs("test") == ["201", "202"]
    assert fake.scancels() == [["201", "202"]]
    assert sp.cancel_live_jobs("test again") == []
    assert len(fake.scancels()) == 1


def test_no_submission_after_shutdown(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    cfg = _cfg(tmp_path)
    sp.make_run_dir(cfg.run_dir, clean=True)
    jobs = sp.prepare_jobs(cfg, _specs(tmp_path, n=1))
    sp.cancel_live_jobs("test")
    with pytest.raises(sp.LauncherShutdown):
        sp.submit_all(jobs)
    assert fake.calls == []


def test_no_retry_is_prepared_after_shutdown(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    hooks.append(lambda: sp.cancel_live_jobs("SIGTERM"))   # arrives during the first poll
    with pytest.raises(sp.LauncherShutdown):
        sp.submit_jobs_with_retry(_specs(tmp_path, n=1), _cfg(tmp_path), max_job_retries=1,
                                  poll_interval=0, max_wait=3600)
    assert fake.scancels() == [["101"]]
    assert not (tmp_path / "eval_007" / "_jobs" / "retry_1").exists()
    assert (tmp_path / "eval_007" / "d0" / "OWNER").read_text().startswith("i7_d00:")


def test_id_that_comes_back_after_shutdown_is_cancelled(tmp_path, slurm, monkeypatch):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    cfg = _cfg(tmp_path)
    sp.make_run_dir(cfg.run_dir, clean=True)
    jobs = sp.prepare_jobs(cfg, _specs(tmp_path, n=1))

    def sbatch_during_sigterm(path):
        sp.cancel_live_jobs("SIGTERM")         # arrives while sbatch is answering
        return "777"
    monkeypatch.setattr(sp, "sbatch_submit", sbatch_during_sigterm)
    with pytest.raises(sp.LauncherShutdown):
        sp.submit_all(jobs)
    assert fake.scancels() == [["777"]]
    assert sp.live_jobs() == {}


def test_poll_stops_without_writing_failed_flags_after_shutdown(tmp_path, slurm):
    install, sleeps, hooks = slurm
    fake = install(FakeSlurm())
    cfg = _cfg(tmp_path)
    sp.make_run_dir(cfg.run_dir, clean=True)
    jobs = sp.prepare_jobs(cfg, _specs(tmp_path, n=1))
    jobs[0].slurm_id = "301"
    sp.cancel_live_jobs("test")
    sp.poll_until_done(jobs, cfg, poll_interval=0, max_wait=3600)
    assert fake.calls == [] and not (jobs[0].sim_dir / "FAILED").exists()


CHILD = textwrap.dedent("""
    import os, sys, threading, time
    sys.path.insert(0, sys.argv[2])
    from pathlib import Path
    from parallel import submit_parallel as sp

    job = sp.Job(idx=0, name="i3_d00", ds_id="d0", sim_dir=Path("."), config_path=Path("."),
                 sbatch_path=Path("."), out_path=Path("."), slurm_id="4242")
    sp._register(job)
    mode = sys.argv[1]
    if mode == "exit":
        sys.exit(0)
    if mode == "raise":
        raise RuntimeError("master crashed")
    # sigterm: like the master, the main thread waits on a launcher thread.
    sp.install_signal_handlers()
    t = threading.Thread(target=time.sleep, args=(60,), daemon=True)
    t.start()
    print("ready", flush=True)
    t.join()
""")


@pytest.mark.parametrize("mode, code", [("exit", 0), ("raise", 1), ("sigterm", 143)])
def test_master_exit_cancels_live_jobs(tmp_path, mode, code):
    """A real interpreter exit: atexit (normal exit, uncaught exception) and the
    SIGTERM handler call the scancel found on PATH (a fake that logs its args)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "scancel.log"
    fake = bindir / "scancel"
    fake.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n')
    fake.chmod(0o755)
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
    proc = subprocess.Popen([sys.executable, str(script), mode, str(REPO)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if mode == "sigterm":
        assert proc.stdout.readline().strip() == "ready"
        proc.terminate()                                  # SIGTERM
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == code, err
    assert log.read_text().split("\n") == ["4242", ""]   # cancelled exactly once
    reason = {"exit": "master exit", "raise": "master exit", "sigterm": "SIGTERM"}[mode]
    assert f"[launcher] {reason}: cancelled 1 job(s)" in out
