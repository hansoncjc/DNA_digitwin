"""GP log written by run_bo: schema, append, loss-unit posterior, RNG isolation.

End-to-end SingleTaskGP checks need botorch and are skipped in the
local digitwin env. On the cluster they also check that logging does not
change the candidate sequence.
"""
import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch

import bo


def _space():
    """Declaration order is ParamSpace vector order."""
    return bo.ParamSpace(
        {
            "n": {"bounds": (0.0, 1.0), "init": 0.2},
            "m": {"bounds": (0.0, 1.0), "init": 0.4},
            "U0": {"bounds": (0.0, 1.0), "init": 0.5},
            "r0": {"bounds": (0.0, 1.0), "init": 0.6},
        }
    )


_NAMES = ["n", "m", "U0", "r0"]


class _ToyGP(torch.nn.Module):
    """Attribute layout matches a fitted gpytorch ExactGP.

    Shapes were checked against gpytorch 1.14: STGP lengthscale ``(1, d)``,
    noise ``(1,)``.
    """

    def __init__(self, *, burn_rng=False):
        super().__init__()
        self.burn_rng = burn_rng
        d = len(_NAMES)
        ls = torch.tensor([[0.7, 1.1, 0.3, 0.9]], dtype=torch.float64)
        outputscale = torch.tensor(0.8, dtype=torch.float64)
        noise = torch.tensor([1e-3], dtype=torch.float64)
        mean = torch.tensor(0.2, dtype=torch.float64)
        self.covar_module = SimpleNamespace(
            base_kernel=SimpleNamespace(lengthscale=ls),
            outputscale=outputscale,
        )
        self.likelihood = SimpleNamespace(noise=noise)
        self.mean_module = SimpleNamespace(constant=mean)
        self.outcome_transform = SimpleNamespace(
            means=torch.tensor([[-3.0]], dtype=torch.float64),
            stdvs=torch.tensor([[2.0]], dtype=torch.float64),
        )
        self.register_buffer("train_x_buf", torch.zeros(1, d, dtype=torch.float64))
        self.train_inputs = (self.train_x_buf,)

    def posterior(self, X, observation_noise=False):
        assert observation_noise is False
        if self.burn_rng:
            torch.rand(8)
        n = X.shape[-2]
        # Latent mean is in train_y = -loss units.
        mean = torch.full((n, 1), -1.25, dtype=torch.float64)
        var = torch.full((n, 1), 0.25, dtype=torch.float64)
        cov = torch.eye(n, dtype=torch.float64) * 0.25
        return SimpleNamespace(
            mean=mean,
            variance=var,
            distribution=SimpleNamespace(covariance_matrix=cov),
        )


def _train():
    train_x = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.2, 0.2, 0.2, 0.2],
        ],
        dtype=torch.float64,
    )
    # argmax is row 1: -0.4 > -1 and -2, so the observed loss there is 0.4.
    train_y = torch.tensor([[-1.0], [-0.4], [-2.0]], dtype=torch.float64)
    candidate = torch.tensor([[0.15, 0.25, 0.35, 0.45]], dtype=torch.float64)
    return train_x, train_y, candidate


def _write(log_dir, *, burn_rng=False, stage="acquisition", candidate=None, acq=0.123):
    train_x, train_y, default_cand = _train()
    if candidate is None and stage == "acquisition":
        candidate = default_cand
    bo._write_gp_iteration(
        log_dir,
        _ToyGP(burn_rng=burn_rng),
        train_x,
        train_y,
        _space(),
        stage=stage,
        acq_value=None if stage == "final" else torch.tensor([acq]),
        candidate=None if stage == "final" else candidate,
        next_eval_id=7,
    )
    return train_x, train_y


def test_preserve_rng_restores_after_an_exception():
    torch.manual_seed(1)
    expected_t = torch.rand(2)
    torch.manual_seed(1)
    with pytest.raises(RuntimeError, match="boom"):
        with bo._PreserveRng():
            torch.rand(5)
            raise RuntimeError("boom")
    assert torch.equal(torch.rand(2), expected_t)


def test_run_bo_logging_is_optional():
    assert inspect.signature(bo.run_bo).parameters["gp_log_dir"].default is None
    assert inspect.signature(bo.run_bo_resumable).parameters["gp_log_dir"].default is None


def test_resumable_forwards_gp_log_dir(monkeypatch, tmp_path):
    captured = {}

    def fake_run_bo(*args, **kwargs):
        captured["kwargs"] = kwargs
        return torch.zeros(1, dtype=torch.float64), [0.0]

    monkeypatch.setattr(bo, "run_bo", fake_run_bo)
    ps = bo.ParamSpace({"n": {"bounds": (1.0, 2.0), "init": 1.5}})
    log = tmp_path / "gp_log"
    bo.run_bo_resumable(
        lambda *a, **k: None,
        ps,
        ffpath="",
        out_root=str(tmp_path),
        gp_log_dir=str(log),
    )
    assert captured["kwargs"]["gp_log_dir"] == str(log)
    assert captured["kwargs"]["warm_start"] is None


def test_posterior_mean_is_loss_and_stays_differentiable():
    class _Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            buf = torch.zeros(1, 2, dtype=torch.float64)
            self.register_buffer("xbuf", buf)
            self.train_inputs = (self.xbuf,)

        def posterior(self, X, observation_noise=False):
            mean = -X.sum(dim=-1, keepdim=True)
            n = X.shape[-2]
            var = torch.ones(n, 1, dtype=X.dtype)
            cov = torch.eye(n, dtype=X.dtype)
            return SimpleNamespace(
                mean=mean,
                variance=var,
                distribution=SimpleNamespace(covariance_matrix=cov),
            )

    x = torch.tensor([0.2, 0.3], dtype=torch.float64, requires_grad=True)
    post = bo.gp_posterior(_Linear(), x)
    assert post["mean_loss"].shape == (1,)
    assert post["std_loss"].shape == (1,)
    assert post["cov_loss"].shape == (1, 1)
    assert torch.allclose(post["mean_loss"], torch.tensor([0.5], dtype=torch.float64))
    post["mean_loss"].sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_gp_log_schema_append_and_rng(tmp_path):
    log = tmp_path / "gp_log"
    _write(log)
    rows = bo.read_gp_log(log)
    assert len(rows) == 1
    row = rows[0]
    assert row["gp_log_version"] == bo.GP_LOG_VERSION
    assert row["units"] == bo.GP_LOG_UNITS
    assert row["param_names"] == _NAMES
    assert row["iteration"] == row["n_train"] == 3
    assert row["stage"] == "acquisition"
    assert row["attempt"] == 0
    assert row["surrogate"] == "stgp"
    assert row["n_mcmc"] is None
    assert row["lengthscale"] == pytest.approx([0.7, 1.1, 0.3, 0.9])
    assert isinstance(row["lengthscale"][0], float)
    assert row["outputscale"] == pytest.approx(0.8)
    assert row["noise"] == pytest.approx(1e-3)
    assert row["mean_constant"] == pytest.approx(0.2)
    assert row["standardize_mean"] == pytest.approx(-3.0)
    assert row["standardize_std"] == pytest.approx(2.0)
    assert row["observation_noise"] is False
    assert row["acq_value"] == pytest.approx(0.123)
    assert row["candidate_unit"] == pytest.approx([0.15, 0.25, 0.35, 0.45])
    assert row["candidate_phys"] == pytest.approx(row["candidate_unit"])
    assert row["candidate_post_mean_loss"] == pytest.approx(1.25)
    assert row["candidate_post_std_loss"] == pytest.approx(0.5)
    assert row["best_index"] == 1
    assert row["best_unit"] == pytest.approx([0.5, 0.6, 0.7, 0.8])
    assert row["best_phys"] == pytest.approx(row["best_unit"])
    assert row["best_observed_loss"] == pytest.approx(0.4)
    assert row["best_post_mean_loss"] == pytest.approx(1.25)
    assert row["best_post_std_loss"] == pytest.approx(0.5)
    assert row["next_eval_id"] == 7
    assert row["state_file"] == "states/ntrain_0003_attempt_00.pt"

    state_path = log / row["state_file"]
    first_bytes = state_path.read_bytes()
    first_line = (log / "gp_log.jsonl").read_text().splitlines()[0]
    bundle = bo._torch_load(state_path)
    assert bundle["surrogate"] == "stgp"
    assert bundle["outcome_transform"] == "Standardize"
    assert bundle["outcome_transform_kwargs"] == {"m": 1}
    assert bundle["param_names"] == _NAMES
    assert bundle["mcmc_samples"] is None
    assert torch.equal(bundle["train_x"], _train()[0])
    assert torch.equal(bundle["train_y"], _train()[1])
    assert "state_dict" in bundle

    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    _write(log, burn_rng=True)
    assert torch.equal(torch.rand(3), expected)

    text = (log / "gp_log.jsonl").read_text().splitlines()
    assert text[0] == first_line
    assert state_path.read_bytes() == first_bytes
    rows = bo.read_gp_log(log)
    assert len(rows) == 2
    assert rows[1]["attempt"] == 1
    assert rows[1]["iteration"] == 3
    assert rows[1]["state_file"] == "states/ntrain_0003_attempt_01.pt"
    assert (log / rows[1]["state_file"]).is_file()

    _write(log, stage="final")
    final = bo.read_gp_log(log)[-1]
    assert final["stage"] == "final"
    assert final["iteration"] == 3
    assert final["acq_value"] is None
    assert final["candidate_unit"] is None
    assert final["candidate_post_mean_loss"] is None
    assert final["state_file"] == "states/ntrain_0003_final.pt"
    assert final["best_post_mean_loss"] == pytest.approx(1.25)

    # A later resume trains on more points; the iteration number moves forward
    # and the earlier lines stay put.
    train_x, train_y, candidate = _train()
    train_x = torch.cat([train_x, candidate.reshape(1, -1)], dim=0)
    train_y = torch.cat([train_y, torch.tensor([[-0.2]], dtype=torch.float64)], dim=0)
    bo._write_gp_iteration(
        log,
        _ToyGP(),
        train_x,
        train_y,
        _space(),
        stage="acquisition",
        acq_value=torch.tensor(0.5),
        candidate=torch.tensor([[0.9, 0.1, 0.2, 0.3]], dtype=torch.float64),
        next_eval_id=8,
    )
    resumed = bo.read_gp_log(log)
    assert (log / "gp_log.jsonl").read_text().splitlines()[0] == first_line
    assert resumed[-1]["iteration"] == 4
    assert resumed[-1]["stage"] == "acquisition"
    assert [r["iteration"] for r in resumed] == [3, 3, 3, 4]


def test_gp_log_error_is_recorded_and_does_not_raise(tmp_path):
    try:
        raise RuntimeError("disk full")
    except RuntimeError:
        bo._report_gp_log_error(tmp_path)
    assert "disk full" in (tmp_path / "errors.txt").read_text()


def test_load_gp_model_needs_botorch():
    if importlib.util.find_spec("botorch") is not None:
        pytest.skip("botorch is installed; reload is covered below")
    with pytest.raises(ImportError, match="botorch"):
        bo.load_gp_model("missing.pt")


def _analytic(store):
    def objective(x_unit, ffpath=""):
        x = x_unit.detach().reshape(-1).to(dtype=torch.float64).clone()
        store.append(x)
        loss = (x - 0.25).pow(2).sum()
        return loss.to(dtype=torch.float64).reshape(1, 1)
    return objective


def _bo_space():
    return bo.ParamSpace(
        {
            "alpha": {"bounds": (0.2, 5.0), "init": 1.0},
            "n": {"bounds": (6.0, 20.0), "init": 12.0},
        }
    )


def _equal_points(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert torch.equal(a, b), (a, b)


def _check_reloaded_posterior(log_dir, row):
    path = log_dir / row["state_file"]
    cols = []
    if row["candidate_unit"] is not None:
        cols.append(row["candidate_unit"])
    cols.append(row["best_unit"])
    X = torch.tensor(cols, dtype=torch.float64)
    post = bo.load_gp_posterior(path, X)
    assert post["param_names"] == row["param_names"]
    assert post["mean_loss"].shape == (len(cols),)
    assert post["cov_loss"].shape == (len(cols), len(cols))
    assert torch.allclose(post["cov_loss"], post["cov_loss"].T, atol=1e-8)
    assert torch.allclose(
        post["cov_loss"].diag().clamp_min(0).sqrt(),
        post["std_loss"],
        atol=1e-6,
    )
    offset = 0
    if row["candidate_unit"] is not None:
        assert post["mean_loss"][0].item() == pytest.approx(
            row["candidate_post_mean_loss"], abs=1e-6
        )
        assert post["std_loss"][0].item() == pytest.approx(
            row["candidate_post_std_loss"], abs=1e-6
        )
        offset = 1
    assert post["mean_loss"][offset].item() == pytest.approx(
        row["best_post_mean_loss"], abs=1e-6
    )
    assert post["std_loss"][offset].item() == pytest.approx(
        row["best_post_std_loss"], abs=1e-6
    )
    x = torch.tensor(row["best_unit"], dtype=torch.float64, requires_grad=True)
    grad_post = bo.gp_posterior(post["model"], x)
    assert grad_post["mean_loss"].requires_grad, "posterior mean did not keep grad w.r.t. X"
    g = torch.autograd.grad(grad_post["mean_loss"].sum(), x)[0]
    assert g.shape == x.shape
    assert torch.isfinite(g).all()


def test_stgp_log_matches_unlogged_run_and_reloads(tmp_path):
    pytest.importorskip("botorch")
    ps_kwargs = dict(n_iters=2, seed=0)
    plain, logged = [], []
    best_plain, hist_plain = bo.run_bo(
        _analytic(plain), _bo_space(), ffpath="", gp_log_dir=None, **ps_kwargs
    )
    log = tmp_path / "gp_log"
    best_logged, hist_logged = bo.run_bo(
        _analytic(logged), _bo_space(), ffpath="", gp_log_dir=log, **ps_kwargs
    )
    _equal_points(plain, logged)
    assert hist_plain == hist_logged
    assert torch.equal(best_plain, best_logged)

    rows = bo.read_gp_log(log)
    assert [r["stage"] for r in rows] == ["acquisition", "acquisition", "final"]
    assert [r["iteration"] for r in rows] == [1, 2, 3]
    assert rows[0]["param_names"] == ["alpha", "n"]
    for row in rows:
        assert row["surrogate"] == "stgp"
        assert row["n_mcmc"] is None
        assert len(row["lengthscale"]) == 2
        assert isinstance(row["outputscale"], float)
        assert isinstance(row["noise"], float)
        assert isinstance(row["mean_constant"], float)
        assert isinstance(row["standardize_mean"], float)
        assert isinstance(row["standardize_std"], float)
        assert (log / row["state_file"]).is_file()
        _check_reloaded_posterior(log, row)
    assert rows[0]["acq_value"] is not None
    assert rows[-1]["acq_value"] is None
    assert rows[-1]["candidate_unit"] is None

    final = bo._torch_load(log / rows[-1]["state_file"])
    warm = (final["train_x"].clone(), final["train_y"].clone())
    assert warm[0].shape[0] == 3
    first_line = (log / "gp_log.jsonl").read_text().splitlines()[0]
    first_state = (log / rows[0]["state_file"]).read_bytes()
    cont, cont_plain = [], []
    bo.run_bo(
        _analytic(cont),
        _bo_space(),
        ffpath="",
        n_iters=1,
        seed=7,
        warm_start=(warm[0].clone(), warm[1].clone()),
        gp_log_dir=log,
    )
    bo.run_bo(
        _analytic(cont_plain),
        _bo_space(),
        ffpath="",
        n_iters=1,
        seed=7,
        warm_start=(warm[0].clone(), warm[1].clone()),
        gp_log_dir=None,
    )
    _equal_points(cont, cont_plain)
    assert (log / "gp_log.jsonl").read_text().splitlines()[0] == first_line
    assert (log / rows[0]["state_file"]).read_bytes() == first_state
    more = bo.read_gp_log(log)
    assert [r["iteration"] for r in more] == [1, 2, 3, 3, 4]
    assert [r["stage"] for r in more] == [
        "acquisition", "acquisition", "final", "acquisition", "final",
    ]
    _check_reloaded_posterior(log, more[-1])

