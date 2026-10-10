"""Resume after the master was killed (run_bo_resumable)."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bo
from shift_rmse_curves import crystal_curve
from simulation import resolve_table_bounds


class _Killed(BaseException):
    """Stands in for SIGTERM/SIGKILL: not caught by the objective."""


class _DS:
    id = "d0"
    weight = 1.0
    datatype = "sq"
    exp_path = "/dev/null/d0.npy"
    sim = SimpleNamespace(n=12.0, m=6.0, density=0.005, r0=None, U0=None)

    def load_exp_curve(self, trim_tail=0):
        return crystal_curve(q1=0.0129)


CFG = {
    "U0": {"bounds": (0.5, 8.0), "init": 3.0},
    "r0": {"bounds": (2.0, 3.5), "init": 2.5},
    "n": {"bounds": (5.5, 15.0), "init": 12.0},
    "m_rel": {"bounds": (0.0, 1.0), "init": 2.0 / 7.0},
    "density": {"fixed": 0.005},
}


@pytest.fixture
def pipeline(monkeypatch):
    """Stub sim + S(q). ``state['kill_at']`` = sim call index that raises _Killed;
    ``state['fail_at']`` = set of call indices that raise RuntimeError."""
    state = {"calls": 0, "kill_at": None, "fail_at": set()}

    def fake_sim(density, U_0, r0, n, m, outdir, **kw):
        i = state["calls"]
        state["calls"] += 1
        if i == state["kill_at"]:
            (Path(outdir) / "DNA_assembly_partial.gsd").write_text("partial")
            raise _Killed()
        if i in state["fail_at"]:
            raise RuntimeError("simulated HOOMD failure")
        b = resolve_table_bounds(U_0, n, m, r0, N=5000, density=density)
        return {"rmax": b["rmax"], "t_tol": b["t_tol"], "n_pairs_below_rmin": 0}

    def fake_saxs(save_dir, **kw):
        d = Path(save_dir) / "S(q)_data"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "average_structure_factor.npy", crystal_curve(q1=0.0131, amp=6.0))

    monkeypatch.setattr(bo, "run_simulation", fake_sim)
    monkeypatch.setattr(bo, "convert_to_SAXS_fft", fake_saxs)
    return state


def _fake_run_bo(log):
    """``run_bo`` without botorch: random candidates, same counting rules."""

    def run_bo(objective_fn, ps, ffpath, n_iters, seed, warm_start, max_acq_attempts,
               gp_log_dir):
        log.append({"n_iters": n_iters, "warm_start": warm_start})
        gen = torch.Generator().manual_seed(1000 + len(log))

        def ev(x):
            try:
                return -objective_fn(x.reshape(1, -1), ffpath=ffpath)
            except bo.EvaluationFailed:
                return None

        if warm_start is None:
            X = ps.init_unit().reshape(1, -1)
            y = None
            while y is None:
                y = ev(X[0])
            Y = y
        else:
            X, Y = warm_start
        done = 0
        while done < n_iters:
            x = torch.rand(ps.d, generator=gen, dtype=torch.float64)
            y = ev(x)
            if y is None:
                continue
            X = torch.cat([X, x.reshape(1, -1)])
            Y = torch.cat([Y, y])
            done += 1
        log[-1]["X"] = X
        best = int(torch.argmax(Y))
        return ps.unit_to_phys(X[best]), [-float(v) for v in Y.reshape(-1)]

    return run_bo


def _objective(out):
    ps = bo.ParamSpace(CFG)
    obj = bo.make_global_objective(
        [_DS()], ps, ffpath="", out_root=str(out), trim_tail=0, mode="sim",
        metric="shift_rmse", compare_q_range=None,
    )
    return obj, ps


def _blocks(path):
    out = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if row and row[0].startswith("# Iteration"):
                out.append((int(row[0].split()[-1]), row[1] if row[1] == "total_loss" else row[2]))
    return out


def test_kill_mid_evaluation_then_resume(tmp_path, pipeline, monkeypatch):
    log = []
    monkeypatch.setattr(bo, "run_bo", _fake_run_bo(log))
    n_iters = 4                                  # 5 successful evaluations in total
    pipeline["fail_at"] = {1}                    # eval_001 fails (not counted)
    pipeline["kill_at"] = 3                      # killed inside eval_003

    obj, ps = _objective(tmp_path)
    with pytest.raises(_Killed):
        bo.run_bo_resumable(obj, ps, "", str(tmp_path), n_iters=n_iters,
                            gp_log_dir=str(tmp_path / "gp_log"))
    assert _blocks(tmp_path / "bo_trajectory.csv") == [
        (0, "total_loss"), (1, "FAILED"), (2, "total_loss")]
    assert (tmp_path / "eval_003" / "d0" / "sim_params_d0.csv").exists()

    # New master: fresh objective, same out_root and settings.
    pipeline["kill_at"] = None
    obj2, ps2 = _objective(tmp_path)
    best, history = bo.run_bo_resumable(obj2, ps2, "", str(tmp_path), n_iters=n_iters,
                                        gp_log_dir=str(tmp_path / "gp_log"))

    resumed = log[1]
    train_x, train_y = resumed["warm_start"]
    assert train_x.shape == (2, 4)
    assert resumed["n_iters"] == n_iters - 1     # 2 loaded = initial + 1 acquisition
    # m_rel restored from the trajectory equals the value that was evaluated.
    rows = {}
    with open(tmp_path / "bo_trajectory.csv", newline="") as fh:
        header = None
        for row in csv.reader(fh):
            if row and row[0] == "iteration":
                header = row
            elif row and header and not row[0].startswith("#"):
                rows.setdefault(int(row[0]), dict(zip(header, row)))
    for x, eval_id in zip(train_x, (0, 2)):
        G = ps2.decode(ps2.unit_to_phys(x))
        assert G["m_rel"] == pytest.approx(float(rows[eval_id]["m_rel"]), abs=1e-12)

    blocks = _blocks(tmp_path / "bo_trajectory.csv")
    assert [b[0] for b in blocks] == list(range(len(blocks)))          # every id once
    assert blocks[3] == (3, "ABANDONED")
    assert sum(1 for _, s in blocks if s == "total_loss") == n_iters + 1
    assert len(history) == n_iters + 1
    folders = sorted(int(p.name[5:]) for p in tmp_path.glob("eval_*"))
    assert folders == list(range(len(blocks)))
    flag = json.loads((tmp_path / "eval_003" / "ABANDONED").read_text())
    assert flag["eval_id"] == 3 and flag["resumed_at_eval_id"] == 4
    assert (tmp_path / "eval_003" / "d0" / "DNA_assembly_partial.gsd").exists()

    # A third start finds the cap reached and evaluates nothing.
    calls = pipeline["calls"]
    obj3, ps3 = _objective(tmp_path)
    _, history3 = bo.run_bo_resumable(obj3, ps3, "", str(tmp_path), n_iters=n_iters)
    assert len(log) == 2 and pipeline["calls"] == calls
    assert history3 == pytest.approx(history)
    assert _blocks(tmp_path / "bo_trajectory.csv") == blocks           # no second ABANDONED


def test_cold_start_after_failed_initial_evaluations(tmp_path, pipeline, monkeypatch):
    log = []
    monkeypatch.setattr(bo, "run_bo", _fake_run_bo(log))
    pipeline["fail_at"] = {0, 1}
    pipeline["kill_at"] = 2
    obj, ps = _objective(tmp_path)
    with pytest.raises(_Killed):
        bo.run_bo_resumable(obj, ps, "", str(tmp_path), n_iters=1)

    pipeline["kill_at"] = None
    obj2, ps2 = _objective(tmp_path)
    bo.run_bo_resumable(obj2, ps2, "", str(tmp_path), n_iters=1)
    assert log[1]["warm_start"] is None and log[1]["n_iters"] == 1
    assert _blocks(tmp_path / "bo_trajectory.csv") == [
        (0, "FAILED"), (1, "FAILED"), (2, "ABANDONED"), (3, "total_loss"), (4, "total_loss")]


def test_inspect_resume_does_not_write(tmp_path, pipeline):
    obj, ps = _objective(tmp_path)
    obj(ps.init_unit(), ffpath="")
    (tmp_path / "eval_001" / "d0").mkdir(parents=True)
    (tmp_path / "eval_007_killed").mkdir()         # renamed by hand: ignored
    before = (tmp_path / "bo_trajectory.csv").read_bytes()
    state = bo.inspect_resume(tmp_path)
    assert state == {"n_successful": 1, "failed_ids": [], "folder_ids": [0, 1],
                     "orphan_ids": [1], "next_eval_id": 2}
    assert (tmp_path / "bo_trajectory.csv").read_bytes() == before
    assert not (tmp_path / "eval_001" / "ABANDONED").exists()


def test_inspect_resume_empty(tmp_path):
    assert bo.inspect_resume(tmp_path / "nothing")["next_eval_id"] == 0


def _block(path, iteration, **vals):
    rec = {"iteration": iteration, "dataset_id": "d0", "loss": 0.3, "density": 0.005,
           "n": 12.0, "m": 6.0, "r0": 2.5, "U0": 3.0, "m_rel": 2.0 / 7.0}
    rec.update(vals)
    bo._write_iteration_block(str(path), iteration, 0.3, [rec])


def test_point_outside_the_box_raises(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _block(traj, 0, r0=1.5)                        # r0 box is (2.0, 3.5)
    with pytest.raises(ValueError, match=r"r0=1.5 not in \[2, 3.5\]"):
        bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(CFG))


def test_round_off_at_the_bound_is_clamped(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _block(traj, 0, n=15.000000000000002, m=bo.m_from_m_rel(15.000000000000002, 2.0 / 7.0))
    train_x, _, _ = bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(CFG))
    assert float(train_x.max()) <= 1.0 and float(train_x.min()) >= 0.0


def test_changed_fixed_value_raises(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _block(traj, 0)
    cfg = {**CFG, "density": {"fixed": 0.006}}
    with pytest.raises(ValueError, match="fixes density=0.006"):
        bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(cfg))


def test_cut_off_last_block(tmp_path):
    traj = tmp_path / "bo_trajectory.csv"
    _block(traj, 0)
    with open(traj, "a", newline="") as fh:       # block 1 cut by a kill
        fh.write("# Iteration 1,total_loss,0.2\r\niteration,dataset_id,loss,density,n\r\n1,d0,0.2,0.0")
    bo._ensure_trailing_newline(traj)
    assert traj.read_bytes().endswith(b"\n")
    train_x, _, next_id = bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(CFG))
    assert train_x.shape[0] == 1 and next_id == 2
    _block(traj, 2)                                # the next append starts on its own line
    train_x, _, next_id = bo.load_warm_start_from_trajectory(traj, bo.ParamSpace(CFG))
    assert train_x.shape[0] == 2 and next_id == 3


def test_gp_log_cut_line_gets_a_newline(tmp_path, monkeypatch):
    log = tmp_path / "gp_log"
    log.mkdir()
    (log / "gp_log.jsonl").write_text('{"iteration": 1}\n{"iteration": 2, "noi')
    monkeypatch.setattr(bo, "run_bo", lambda **kw: (torch.zeros(4), [0.0]))
    obj = lambda *a, **k: None  # noqa: E731
    bo.run_bo_resumable(obj, bo.ParamSpace(CFG), "", str(tmp_path), gp_log_dir=str(log))
    text = (log / "gp_log.jsonl").read_text()
    assert text.endswith("\n") and text.count("\n") == 2


def test_m_rel_keyword_removed():
    import inspect
    assert "m_rel" not in inspect.signature(bo.run_bo_resumable).parameters
    assert "m_rel" not in inspect.signature(bo.load_warm_start_from_trajectory).parameters
