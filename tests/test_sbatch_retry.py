"""sbatch_submit retries a failed sbatch call and never submits a job twice."""
import subprocess

import pytest

from parallel import submit_parallel as sp

TIMEOUT = "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation"


class FakeSlurm:
    """``subprocess.check_output`` stand-in. ``sbatch`` follows ``sbatch_plan``
    (an id, or "fail"); ``squeue`` follows ``squeue_plan`` (a list of
    "<id>|<script>" lines, or "fail")."""

    def __init__(self, sbatch_plan, squeue_plan=()):
        self.sbatch_plan = list(sbatch_plan)
        self.squeue_plan = list(squeue_plan)
        self.calls = []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd[0])
        plan = self.sbatch_plan if cmd[0] == "sbatch" else self.squeue_plan
        step = plan.pop(0)
        if step == "fail":
            raise subprocess.CalledProcessError(1, cmd, output=TIMEOUT)
        if cmd[0] == "sbatch":
            return f"Submitted batch job {step}\n"
        return "\n".join(step) + ("\n" if step else "")


@pytest.fixture
def slurm(monkeypatch, tmp_path):
    waits = []
    monkeypatch.setattr(sp.time, "sleep", waits.append)
    script = tmp_path / "jobs" / "i44_d00.sbatch"
    script.parent.mkdir()
    script.write_text("#!/bin/bash\n")

    def install(fake):
        monkeypatch.setattr(sp.subprocess, "check_output", fake)
        return fake

    return install, script, waits


def test_first_call_succeeds_without_waiting(slurm):
    install, script, waits = slurm
    fake = install(FakeSlurm(["123"]))
    assert sp.sbatch_submit(script) == "123"
    assert fake.calls == ["sbatch"] and waits == []


def test_timeout_then_success(slurm):
    install, script, waits = slurm
    fake = install(FakeSlurm(["fail", "456"], squeue_plan=[[]]))
    assert sp.sbatch_submit(script) == "456"
    assert fake.calls == ["sbatch", "squeue", "sbatch"]
    assert waits == [sp.SBATCH_RETRY_WAIT]


def test_job_queued_despite_the_error_is_not_submitted_again(slurm):
    install, script, waits = slurm
    other = script.parent.parent / "other_campaign" / "i44_d00.sbatch"   # same name, other run
    fake = install(FakeSlurm(["fail"], squeue_plan=[[f"777|{other}", f"789|{script}"]]))
    assert sp.sbatch_submit(script) == "789"
    assert fake.calls == ["sbatch", "squeue"]


def test_same_job_name_in_another_campaign_is_not_taken(slurm):
    install, script, waits = slurm
    other = script.parent.parent / "other_campaign" / "i44_d00.sbatch"
    fake = install(FakeSlurm(["fail", "456"], squeue_plan=[[f"777|{other}"]]))
    assert sp.sbatch_submit(script) == "456"


def test_failed_squeue_check_does_not_resubmit_blindly(slurm):
    install, script, waits = slurm
    fake = install(FakeSlurm(["fail", "456"], squeue_plan=["fail", []]))
    assert sp.sbatch_submit(script) == "456"
    assert fake.calls == ["sbatch", "squeue", "squeue", "sbatch"]
    assert len(waits) == 2


def test_gives_up_after_the_last_attempt(slurm):
    install, script, waits = slurm
    n = sp.SBATCH_ATTEMPTS
    install(FakeSlurm(["fail"] * n, squeue_plan=[[]] * (n - 1)))
    with pytest.raises(subprocess.CalledProcessError):
        sp.sbatch_submit(script)
    assert len(waits) == n - 1


def test_submit_all_uses_the_retry(slurm, tmp_path):
    install, script, waits = slurm
    install(FakeSlurm(["fail", "456"], squeue_plan=[[]]))
    job = sp.Job(idx=0, name="i44_d00", ds_id="d0", sim_dir=tmp_path, config_path=tmp_path / "c.json",
                 sbatch_path=script, out_path=tmp_path / "o.out")
    sp.submit_all([job])
    assert job.slurm_id == "456" and job.state == "SUBMITTED"
