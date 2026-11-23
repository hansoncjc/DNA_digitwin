"""
parallel/forward_term.sh and the master sbatch scripts that source it.

Smoke test B (2026-11-22): at the walltime the python master got no SIGTERM it
could act on before SIGKILL, and its GPU jobs ran on. The masters now ask Slurm
for a TERM to the batch shell 120 s before the walltime (#SBATCH
--signal=B:TERM@120); run_forward_term passes it on to python. These tests run
the real shell function with bash; only Slurm's delivery of that TERM needs
the cluster.
"""
import os
import re
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FORWARD = REPO / "parallel" / "forward_term.sh"

# Every master sbatch whose python process runs the parallel launcher.
LAUNCHER_MASTERS = [
    "campaigns/r4-measured/run_r4.sbatch",
    "campaigns/r3.2-physics-validate/run_r32.sbatch",
    "campaigns/inverse/inverse-example/inverse_parallel.sbatch",
]


def _bash(body: str, **kw) -> subprocess.Popen:
    """bash with the same options as the sbatch scripts, forward_term.sh sourced."""
    script = f'set -eo pipefail\n. "{FORWARD}"\n{body}\n'
    return subprocess.Popen(["bash", "-c", script], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, **kw)


@pytest.mark.parametrize("code", [0, 3])
def test_without_a_signal_output_and_exit_status_are_unchanged(code):
    proc = _bash(f'run_forward_term sh -c "echo out; echo err >&2; exit {code}"\necho after')
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == code
    assert err == "err\n"
    # set -e still stops the script when the command fails, as before.
    assert out == ("out\nafter\n" if code == 0 else "out\n")


def test_arguments_reach_the_command_unchanged():
    proc = _bash('run_forward_term printf "[%s]" "a b" "" \'$HOME\'')
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 0 and out == "[a b][][$HOME]"


CLEANUP_CHILD = textwrap.dedent("""
    import signal, sys, time
    from pathlib import Path
    def on_term(signum, frame):
        time.sleep(1.0)                        # cleanup that takes a while
        Path(sys.argv[1]).write_text("cleaned up")
        sys.exit(7)
    signal.signal(signal.SIGTERM, on_term)
    print("ready", flush=True)
    time.sleep(60)
""")


def test_term_to_the_shell_reaches_the_command_and_the_shell_waits_for_it(tmp_path):
    child = tmp_path / "child.py"
    child.write_text(CLEANUP_CHILD)
    marker = tmp_path / "marker"
    proc = _bash(f'run_forward_term "{sys.executable}" -u "{child}" "{marker}"\necho after')
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)           # to bash only, like --signal=B:
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 7, err           # the command's status
    assert marker.read_text() == "cleaned up"  # bash did not exit before it
    assert "after" not in out


MASTER_CHILD = textwrap.dedent("""
    import sys, threading, time
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from parallel import submit_parallel as sp

    job = sp.Job(idx=0, name="i3_d00", ds_id="d0", sim_dir=Path("."), config_path=Path("."),
                 sbatch_path=Path("."), out_path=Path("."), slurm_id="4242")
    sp._register(job)
    sp.install_signal_handlers()
    # Like the master: the main thread waits on a launcher thread.
    t = threading.Thread(target=time.sleep, args=(60,), daemon=True)
    t.start()
    print("ready", flush=True)
    t.join()
""")


def test_term_to_the_shell_makes_the_master_cancel_its_jobs(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "scancel.log"
    fake = bindir / "scancel"
    fake.write_text(f'#!/bin/sh\necho "$@" >> "{log}"\n')
    fake.chmod(0o755)
    child = tmp_path / "master.py"
    child.write_text(MASTER_CHILD)
    env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}")
    # A command after it, as in the sbatch scripts: otherwise bash may exec the
    # last command of `bash -c` in place of itself and the test would pass
    # even without forwarding.
    proc = _bash(f'run_forward_term "{sys.executable}" -u "{child}" "{REPO}"\necho after',
                 env=env)
    assert proc.stdout.readline().strip() == "ready"
    proc.send_signal(signal.SIGTERM)           # to bash only, like --signal=B:
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 143, err
    assert log.read_text().split("\n") == ["4242", ""]
    assert "[launcher] SIGTERM: cancelled 1 job(s)" in out


@pytest.mark.parametrize("rel", LAUNCHER_MASTERS)
def test_every_launcher_master_asks_for_the_signal_and_forwards_it(rel):
    path = REPO / rel
    text = path.read_text()
    assert "#SBATCH --signal=B:TERM@120\n" in text
    assert re.search(r'^\. "\$\{?\w+\}?/parallel/forward_term\.sh"$', text, re.M)
    runs = [line for line in text.splitlines()
            if re.match(r"\s*(run_forward_term\s+)?python3?\s", line)]
    assert runs and all(line.startswith("run_forward_term python3 -u ") for line in runs)
    subprocess.run(["bash", "-n", str(path)], check=True)
