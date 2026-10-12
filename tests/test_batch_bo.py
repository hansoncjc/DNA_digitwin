"""Batch BO (C24): first batch, counting, failures, GP log v2, parallel
batches and resume.

Most tests replace botorch with a stub (``fake_botorch``), so they run in the
local digitwin env and on the cluster alike. The last two need botorch and
are skipped without it.
"""
import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import bo
import inverse
from parallel import submit_parallel as sp
from test_resume import CFG, _DS, _Killed, _blocks, _objective, pipeline  # noqa: F401

D3 = {
    "a": {"bounds": (0.0, 1.0), "init": 0.2},
    "b": {"bounds": (0.0, 2.0), "init": 1.0},
    "c": {"bounds": (-1.0, 1.0), "init": 0.0},
}


# ---------------------------------------------------------------- stubs


class _FakeGP(torch.nn.Module):
    """Attribute layout of a fitted SingleTaskGP, enough for the GP log."""

    def __init__(self, train_x, train_y, outcome_transform=None):
        super().__init__()
        d = train_x.shape[-1]
        self.register_buffer("tx", train_x.detach().clone())
        self.train_inputs = (self.tx,)
        self.covar_module = SimpleNamespace(
            base_kernel=SimpleNamespace(lengthscale=torch.full((1, d), 0.5, dtype=torch.float64)),
            outputscale=torch.tensor(1.0, dtype=torch.float64),
        )
        self.likelihood = SimpleNamespace(noise=torch.tensor([1e-4], dtype=torch.float64))
        self.mean_module = SimpleNamespace(constant=torch.tensor(0.0, dtype=torch.float64))
        self.outcome_transform = SimpleNamespace(
            means=train_y.mean().reshape(1, 1), stdvs=torch.ones(1, 1, dtype=torch.float64))

    def posterior(self, X, observation_noise=False):
        n = X.shape[-2]
        mean = -(X - 0.25).pow(2).sum(dim=-1, keepdim=True)        # train_y = -loss units
        return SimpleNamespace(
            mean=mean,
            variance=torch.full((n, 1), 0.04, dtype=torch.float64),
            distribution=SimpleNamespace(
                covariance_matrix=torch.eye(n, dtype=torch.float64) * 0.04),
        )


@pytest.fixture
def fake_botorch(monkeypatch):
    """botorch modules that run_bo imports, recording every call. Candidates
    come from their own generator; the q values mimic botorch: one value per
    sequential step for q > 1, a scalar for q = 1."""
    calls = {"fit": [], "acqf": []}
    gen = torch.Generator().manual_seed(123)

    def optimize_acqf(acq_function, bounds, q, num_restarts, raw_samples, sequential=False):
        calls["acqf"].append({"q": q, "sequential": sequential, "num_restarts": num_restarts,
                              "raw_samples": raw_samples, "n_train": acq_function.n_train})
        cand = torch.rand(q, bounds.shape[-1], generator=gen, dtype=torch.float64)
        vals = (torch.linspace(-2.0, -1.0, q, dtype=torch.float64) if sequential and q > 1
                else torch.tensor(-1.0, dtype=torch.float64))
        return cand, vals

    class _LogEI:
        def __init__(self, model, best_f):
            self.model, self.best_f = model, best_f
            self.n_train = int(model.tx.shape[0])

    def fit(mll):
        calls["fit"].append(mll)

    stubs = {
        "botorch": {},
        "botorch.models": {"SingleTaskGP": _FakeGP},
        "botorch.models.transforms": {},
        "botorch.models.transforms.outcome": {"Standardize": lambda m: None},
        "botorch.fit": {"fit_gpytorch_mll": fit},
        "botorch.acquisition": {},
        "botorch.acquisition.logei": {"qLogExpectedImprovement": _LogEI},
        "botorch.optim": {"optimize_acqf": optimize_acqf},
        "gpytorch.mlls.exact_marginal_log_likelihood": {
            "ExactMarginalLogLikelihood": lambda likelihood, model: model},
    }
    try:   # the GP log asks for gpytorch's version: import the real one first
        import gpytorch  # noqa: F401
    except ImportError:
        pass
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _analytic(record, fail=()):
    """Plain objective: loss = |x - 0.25|^2. Call indices in ``fail`` raise
    EvaluationFailed (``fail`` may be a callable of the call index)."""
    def objective(x_unit, ffpath=""):
        i = len(record)
        x = x_unit.detach().reshape(-1).clone()
        record.append(x)
        bad = fail(i) if callable(fail) else i in fail
        if bad:
            raise bo.EvaluationFailed(f"call {i}")
        return (x - 0.25).pow(2).sum().reshape(1, 1)
    return objective


def _sobol(d, n, seed, skip=0):
    from torch.quasirandom import SobolEngine
    eng = SobolEngine(d, scramble=True, seed=seed)
    if skip:
        eng.draw(skip, dtype=torch.float64)
    return eng.draw(n, dtype=torch.float64)


# ---------------------------------------------------------------- run_bo


def test_remaining_evals():
    assert bo.remaining_evals(100, 0) == 100
    assert bo.remaining_evals(100, 37) == 63
    assert bo.remaining_evals(10, 12) == 0
    assert not hasattr(bo, "remaining_bo_iters")


def test_first_batch_is_the_initial_point_plus_sobol_points(fake_botorch):
    ps = bo.ParamSpace(D3)
    seen = []
    bo.run_bo(_analytic(seen), ps, "", max_evals=4, seed=5, batch_size=4)
    X = torch.stack(seen)
    assert torch.equal(X[0], ps.init_unit())
    assert torch.equal(X[1:], _sobol(3, 3, seed=5))
    assert fake_botorch["acqf"] == []                     # no GP for the first batch
    # The Sobol engine has its own generator: the global stream is untouched.
    after = torch.rand(2)
    torch.manual_seed(5)
    assert torch.equal(after, torch.rand(2))


def test_batches_count_successful_evaluations_to_max_evals(fake_botorch):
    seen = []
    _, history = bo.run_bo(_analytic(seen), bo.ParamSpace(D3), "", max_evals=10, seed=0,
                           batch_size=4)
    assert len(seen) == len(history) == 10
    assert [c["q"] for c in fake_botorch["acqf"]] == [4, 2]   # 4 + 4 + 2: the last takes the rest
    assert [c["n_train"] for c in fake_botorch["acqf"]] == [4, 8]
    assert all(c["sequential"] is True for c in fake_botorch["acqf"])
    assert all((c["num_restarts"], c["raw_samples"]) == (5, 64) for c in fake_botorch["acqf"])


def test_batch_size_1_follows_the_same_rule(fake_botorch, monkeypatch):
    import torch.quasirandom as tq

    def no_sobol(*a, **k):
        raise AssertionError("batch_size=1 draws no Sobol point")

    monkeypatch.setattr(tq, "SobolEngine", no_sobol)
    ps = bo.ParamSpace(D3)
    seen = []
    bo.run_bo(_analytic(seen, fail={2}), ps, "", max_evals=3, seed=0, batch_size=1)
    assert torch.equal(seen[0], ps.init_unit())
    assert len(seen) == 4                                  # call 2 failed and was replaced
    assert [c["q"] for c in fake_botorch["acqf"]] == [1, 1, 1]
    assert [c["n_train"] for c in fake_botorch["acqf"]] == [1, 2, 2]
    assert all(c["sequential"] is True for c in fake_botorch["acqf"])


def test_failed_points_are_filled_in_by_later_batches(fake_botorch):
    seen = []
    _, history = bo.run_bo(_analytic(seen, fail={1, 2, 5}), bo.ParamSpace(D3), "",
                           max_evals=8, seed=0, batch_size=4)
    # first batch: 2 of 4; then 3 of 4 (5 in total); then the 3 left.
    assert [c["q"] for c in fake_botorch["acqf"]] == [4, 3]
    assert [c["n_train"] for c in fake_botorch["acqf"]] == [2, 5]
    assert len(seen) == 11 and len(history) == 8
    ok = [x for i, x in enumerate(seen) if i not in {1, 2, 5}]
    assert history == pytest.approx([float((x - 0.25).pow(2).sum()) for x in ok])


def test_batches_without_any_success_stop_the_run(fake_botorch):
    seen = []
    with pytest.raises(RuntimeError, match="3 batch\\(es\\) in a row"):
        bo.run_bo(_analytic(seen, fail=lambda i: i >= 4), bo.ParamSpace(D3), "",
                  max_evals=10, seed=0, batch_size=4, max_acq_attempts=3)
    assert len(seen) == 4 + 3 * 4


def test_a_partial_batch_resets_the_failure_count(fake_botorch):
    # After the first batch: batch 2 fails, batch 3 has 1 success, batches
    # 4 and 5 fail, batch 6 succeeds. With max_acq_attempts=3 the run goes on.
    bad = set(range(4, 8)) | {9, 10, 11} | set(range(12, 20))
    seen = []
    _, history = bo.run_bo(_analytic(seen, fail=bad), bo.ParamSpace(D3), "",
                           max_evals=9, seed=0, batch_size=4, max_acq_attempts=3)
    assert len(history) == 9
    assert [c["n_train"] for c in fake_botorch["acqf"]] == [4, 4, 5, 5, 5]


def test_a_failed_initial_batch_draws_new_sobol_points(fake_botorch):
    ps = bo.ParamSpace(D3)
    seen = []
    bo.run_bo(_analytic(seen, fail={0, 1, 2}), ps, "", max_evals=3, seed=2, batch_size=3)
    X = torch.stack(seen)
    assert len(seen) == 6 and fake_botorch["acqf"] == []
    assert torch.equal(X[0], ps.init_unit()) and torch.equal(X[3], ps.init_unit())
    assert torch.equal(X[1:3], _sobol(3, 2, seed=2))
    assert torch.equal(X[4:6], _sobol(3, 2, seed=2, skip=2))


def test_run_bo_requires_max_evals():
    import inspect
    params = inspect.signature(bo.run_bo).parameters
    assert "n_iters" not in params and params["max_evals"].default is inspect.Parameter.empty
    assert params["batch_size"].default == 1
    rparams = inspect.signature(bo.run_bo_resumable).parameters
    assert "n_iters" not in rparams and rparams["batch_size"].default == 1


# ------------------------------------------- sequential objective, kill, resume


def test_sequential_batch_killed_mid_batch_then_resumed(  # noqa: F811
        tmp_path, pipeline, fake_botorch):
    log = tmp_path / "gp_log"
    pipeline["kill_at"] = 4                       # batch 2 is evals 3, 4, 5: killed in eval 4
    obj, ps = _objective(tmp_path)
    with pytest.raises(_Killed):
        bo.run_bo_resumable(obj, ps, "", str(tmp_path), max_evals=7, batch_size=3,
                            gp_log_dir=str(log))
    assert _blocks(tmp_path / "bo_trajectory.csv") == [
        (0, "total_loss"), (1, "total_loss"), (2, "total_loss"), (3, "total_loss")]
    assert (tmp_path / "eval_005").is_dir()        # created with the batch, never started
    [row] = bo.read_gp_log(log)
    assert row["batch"] == 1 and row["n_train"] == 3 and row["eval_ids"] == [3, 4, 5]

    pipeline["kill_at"] = None
    obj2, ps2 = _objective(tmp_path)
    _, history = bo.run_bo_resumable(obj2, ps2, "", str(tmp_path), max_evals=7, batch_size=3,
                                     gp_log_dir=str(log))
    blocks = _blocks(tmp_path / "bo_trajectory.csv")
    assert [b[0] for b in sorted(blocks)] == list(range(9))          # ids never reused
    status = dict(blocks)
    assert status[4] == status[5] == "ABANDONED"
    assert sum(1 for s in status.values() if s == "total_loss") == 7 == len(history)
    for eval_id in (4, 5):
        assert (tmp_path / f"eval_{eval_id:03d}" / "ABANDONED").exists()
    rows = bo.read_gp_log(log)
    assert [r["batch"] for r in rows] == [1, 2, None]
    assert [r["n_train"] for r in rows] == [3, 4, 7]                 # refit on all 4 done
    assert rows[1]["eval_ids"] == [6, 7, 8] and rows[1]["q"] == 3


def test_gp_log_v2_rows_from_a_batch_run(tmp_path, pipeline, fake_botorch):  # noqa: F811
    log = tmp_path / "gp_log"
    obj, ps = _objective(tmp_path)
    bo.run_bo_resumable(obj, ps, "", str(tmp_path), max_evals=5, batch_size=2,
                        gp_log_dir=str(log))
    rows = bo.read_gp_log(log)
    assert [r["stage"] for r in rows] == ["acquisition", "acquisition", "final"]
    assert [r["q"] for r in rows] == [2, 1, 0]
    assert [r["eval_ids"] for r in rows] == [[2, 3], [4], []]
    assert rows[0]["acq_value_steps"] == pytest.approx([-2.0, -1.0])
    assert rows[0]["acq_value"] == pytest.approx(-1.0)
    assert rows[1]["acq_value_steps"] == pytest.approx([-1.0])
    for row in rows:
        assert row["gp_log_version"] == 2
        assert len(row["candidates_unit"]) == len(row["candidates_post_mean_loss"]) == row["q"]
        bundle = bo._torch_load(log / row["state_file"])
        assert bundle["gp_log_version"] == 2 and bundle["batch"] == row["batch"]
    # The logged candidates are the points that were evaluated.
    import csv
    with open(tmp_path / "bo_trajectory.csv", newline="") as fh:
        params = {}
        header = None
        for r in csv.reader(fh):
            if r and r[0] == "iteration":
                header = r
            elif r and header and not r[0].startswith("#"):
                params[int(r[0])] = dict(zip(header, r))
    for row in rows[:2]:
        for eval_id, phys in zip(row["eval_ids"], row["candidates_phys"]):
            G = dict(zip(row["param_names"], phys))
            assert float(params[eval_id]["U0"]) == pytest.approx(G["U0"], abs=1e-9)


# ---------------------------------------------------------- parallel batches


def _fake_launcher(behaviour, log, barrier=None):
    """``submit_jobs_with_retry`` stand-in: ``behaviour[eval_id] = (status, delay)``,
    status DONE, FAILED or 'kill'. Waits on ``barrier`` first when given."""
    def submit(job_specs, cfg, **kw):
        specs = list(job_specs)
        log.append({"names": [s["name"] for s in specs], "run_dir": str(cfg.run_dir)})
        if barrier is not None:
            barrier.wait(timeout=10)
        jobs = []
        for s in specs:
            sim_dir = Path(s["worker_config"]["outdir"])
            eval_id = int(sim_dir.parent.name[5:])
            status, delay = behaviour[eval_id]
            time.sleep(delay)
            if status == "kill":
                raise _Killed()
            if status == "DONE":
                (sim_dir / "DONE").write_text(json.dumps({
                    "loss": 0.1 * (eval_id + 1),
                    "result": {"rmax": 5.0, "t_tol": 0.03, "n_pairs_below_rmin": 0}}))
            else:
                (sim_dir / "FAILED").write_text("simulated\n")
            jobs.append(sp.Job(idx=0, name=s["name"], ds_id=s["ds_id"], sim_dir=sim_dir,
                               config_path=sim_dir / "c.json", sbatch_path=sim_dir / "s.sbatch",
                               out_path=sim_dir / "o.out", done_status=status))
        with open(Path(cfg.run_dir).parent / "finished_at", "w") as fh:
            fh.write(str(time.monotonic()))
        return jobs
    return submit


def _parallel_objective(out):
    ps = bo.ParamSpace(CFG)
    obj = bo.make_global_objective(
        [_DS()], ps, ffpath="", out_root=str(out), trim_tail=0, mode="sim",
        metric="shift_rmse", compare_q_range=None, parallel=True, parallel_cfg={},
    )
    return obj, ps


def _points(ps, q):
    return torch.stack([ps.init_unit()] * q) * torch.linspace(0.5, 1.0, q,
                                                               dtype=torch.float64)[:, None]


def test_parallel_batch_submits_at_once_and_writes_in_eval_order(tmp_path, monkeypatch):
    import parallel
    log = []
    # eval 0 finishes last, eval 2 first; eval 1 fails.
    behaviour = {0: ("DONE", 0.6), 1: ("FAILED", 0.3), 2: ("DONE", 0.0)}
    monkeypatch.setattr(parallel, "submit_jobs_with_retry",
                        _fake_launcher(behaviour, log, threading.Barrier(3)))
    obj, ps = _parallel_objective(tmp_path)
    out = obj.evaluate_batch(_points(ps, 3), ffpath="")

    assert out[1] is None
    assert [float(out[i]) for i in (0, 2)] == pytest.approx([0.1, 0.3])
    finished = {i: float((tmp_path / f"eval_{i:03d}" / "finished_at").read_text())
                for i in range(3)}
    assert finished[2] < finished[1] < finished[0]                 # out of order ...
    assert _blocks(tmp_path / "bo_trajectory.csv") == [            # ... written in order
        (0, "total_loss"), (1, "FAILED"), (2, "total_loss")]
    names = sorted(n for entry in log for n in entry["names"])
    assert names == ["i00_d00", "i01_d00", "i02_d00"]
    assert len({entry["run_dir"] for entry in log}) == 3            # eval_XXX/_jobs each
    assert obj._eval_id == 3
    with open(tmp_path / "loss_components.txt") as fh:
        assert [line.split(",")[0] for line in fh.read().splitlines()[1:]] == ["0", "2"]


def test_parallel_batch_killed_mid_batch_then_resumed(tmp_path, monkeypatch):
    import parallel
    log = []
    behaviour = {0: ("DONE", 0.0), 1: ("kill", 0.3), 2: ("DONE", 0.0)}
    monkeypatch.setattr(parallel, "submit_jobs_with_retry", _fake_launcher(behaviour, log))
    obj, ps = _parallel_objective(tmp_path)
    with pytest.raises(_Killed):
        obj.evaluate_batch(_points(ps, 3), ffpath="")
    assert _blocks(tmp_path / "bo_trajectory.csv") == [(0, "total_loss")]
    assert (tmp_path / "eval_002" / "d0" / "DONE").exists()          # finished, no block
    assert bo.inspect_resume(tmp_path)["orphan_ids"] == [1, 2]

    captured = {}

    def fake_run_bo(**kw):
        captured.update(kw)
        return torch.zeros(ps.d), [0.0]

    monkeypatch.setattr(bo, "run_bo", fake_run_bo)
    obj2, ps2 = _parallel_objective(tmp_path)
    bo.run_bo_resumable(obj2, ps2, "", str(tmp_path), max_evals=6, batch_size=3)
    assert captured["warm_start"][0].shape[0] == 1                    # eval 2's DONE is not read
    assert captured["batch_size"] == 3 and captured["max_evals"] == 6
    assert obj2._eval_id == 3
    assert dict(_blocks(tmp_path / "bo_trajectory.csv")) == {
        0: "total_loss", 1: "ABANDONED", 2: "ABANDONED"}


def test_single_evaluation_call_is_unchanged(tmp_path, monkeypatch):
    import parallel
    log = []
    monkeypatch.setattr(parallel, "submit_jobs_with_retry",
                        _fake_launcher({0: ("DONE", 0.0), 1: ("FAILED", 0.0)}, log))
    obj, ps = _parallel_objective(tmp_path)
    assert float(obj(ps.init_unit(), ffpath="")) == pytest.approx(0.1)
    with pytest.raises(bo.EvaluationFailed):
        obj(ps.init_unit(), ffpath="")
    assert _blocks(tmp_path / "bo_trajectory.csv") == [(0, "total_loss"), (1, "FAILED")]
    assert [entry["names"] for entry in log] == [["i00_d00"], ["i01_d00"]]


# ------------------------------------------------------------------ inverse.py


def test_batch_example_config_needs_parallel():
    path = inverse.REPO / "campaigns" / "inverse" / "inverse-example" / "config_batch.py"
    raw = inverse.load_config(path)
    cfg = inverse.normalize_config(raw)
    assert cfg["batch_size"] == 4 and cfg["max_evals"] == 12 and cfg["parallel"] is True
    assert cfg["parallel_cfg"]["partition"] == "gpu-a40"
    with pytest.raises(inverse.ConfigError, match="needs parallel=True") as exc:
        inverse.normalize_config(raw, {"parallel": False})
    assert "q times as long" in str(exc.value)


def test_inverse_passes_batch_size_and_locks_it(tmp_path, monkeypatch):
    import numpy as np
    from shift_rmse_curves import crystal_curve
    from test_inverse import _write_config

    target = tmp_path / "target_sq.npy"
    np.save(target, crystal_curve(q1=0.0129))
    captured = []
    monkeypatch.setattr(bo, "run_bo_resumable",
                        lambda *a, **kw: captured.append(kw) or (torch.zeros(4), [0.0]))
    pcfg = {"partition": "gpu-a40", "account": "zeelab", "mem": "10G", "time": "01:00:00",
            "module_loads": [], "venv_activate": ""}
    path = _write_config(tmp_path, target, batch_size=4, max_evals=12, parallel=True,
                         parallel_cfg=pcfg)
    summary = inverse.run(path)
    assert captured[0]["batch_size"] == 4 and captured[0]["max_evals"] == 12
    assert summary["batch_size"] == 4
    snap = json.loads((tmp_path / "out" / "inverse_config.json").read_text())
    assert snap["locked"]["batch_size"] == 4

    path = _write_config(tmp_path, target, batch_size=2, max_evals=12, parallel=True,
                         parallel_cfg=pcfg)
    with pytest.raises(inverse.ConfigError, match="batch_size: run started with 4, now 2"):
        inverse.run(path)
    assert len(captured) == 1


# ------------------------------------------------- real botorch (cluster only)


def test_real_batch_run_and_reload(tmp_path):
    pytest.importorskip("botorch")
    ps = bo.ParamSpace(D3)
    seen = []
    log = tmp_path / "gp_log"
    _, history = bo.run_bo(_analytic(seen), ps, "", max_evals=12, seed=0, batch_size=4,
                           gp_log_dir=log)
    X = torch.stack(seen)
    assert len(seen) == len(history) == 12
    assert torch.equal(X[0], ps.init_unit()) and torch.equal(X[1:4], _sobol(3, 3, seed=0))
    rows = bo.read_gp_log(log)
    assert [(r["stage"], r["batch"], r["q"], r["n_train"]) for r in rows] == [
        ("acquisition", 1, 4, 4), ("acquisition", 2, 4, 8), ("final", None, 0, 12)]
    for row, chunk in zip(rows[:2], (X[4:8], X[8:12])):
        assert len(row["acq_value_steps"]) == 4
        assert row["acq_value"] == row["acq_value_steps"][-1]
        assert torch.allclose(torch.tensor(row["candidates_unit"], dtype=torch.float64), chunk)
        post = bo.load_gp_posterior(log / row["state_file"],
                                    torch.tensor(row["candidates_unit"], dtype=torch.float64))
        assert post["mean_loss"].tolist() == pytest.approx(
            row["candidates_post_mean_loss"], abs=1e-6)
        assert post["std_loss"].tolist() == pytest.approx(
            row["candidates_post_std_loss"], abs=1e-6)


def test_batch_size_1_matches_the_previous_q1_loop():
    """run_bo(batch_size=1) against the loop of 145ca00 (optimize_acqf q=1
    without sequential): same seed, same candidates, bit for bit."""
    pytest.importorskip("botorch")
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

    ps = bo.ParamSpace(D3)
    new = []
    bo.run_bo(_analytic(new), ps, "", max_evals=5, seed=3, batch_size=1)

    old = []
    objective = _analytic(old)
    torch.manual_seed(3)
    lb, ub = ps.bounds_unit()
    train_x = ps.init_unit().to(torch.float64).unsqueeze(0)
    train_y = -objective(train_x)
    for _ in range(4):
        gp = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        acq = qLogExpectedImprovement(model=gp, best_f=train_y.max())
        cand, _ = optimize_acqf(acq_function=acq, bounds=torch.stack([lb, ub]).to(torch.float64),
                                q=1, num_restarts=5, raw_samples=64)
        train_x = torch.cat([train_x, cand])
        train_y = torch.cat([train_y, -objective(cand)])
    assert len(new) == len(old) == 5
    for a, b in zip(new, old):
        assert torch.equal(a, b), (a, b)
