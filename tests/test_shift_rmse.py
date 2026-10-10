"""
shift_rmse (metrics.shift_rmse_loss / compare_to_exp_saxsfft) on synthetic curves.

OLD_FCC and FLUID are the peak-search, asymptote and fallback settings of the
2026-08/09 inverse-design drivers, used here only as non-default parameters.
The bit-for-bit regressions against the driver module ``shift_mse_metric.py``
(FCC_REFS, FLUID_REFS, 2026-10-06) were removed on 2026-10-09: that module
interpolated only in-window samples and held the low-q end flat, which the
repo no longer does (tests/test_resample_overlap.py).
"""
import json

import numpy as np
import pytest
from scipy.signal import savgol_filter

import bo
import metrics
from metrics import MetricFailed, asymptote_scale, shift_rmse_loss, shift_rmse_params
from shift_rmse_curves import crystal_curve, flat_curve, fluid_curve, gas_curve, saxsfft_q

W_NARROW = (0.004, 0.040)
OLD_FCC = dict(
    peak_search_range=(0.006, 0.021),
    asymptote_band=(0.050, 0.0654),
    no_peak="fallback",
    asymptote_fallback="tail",
    asymptote_zero_den="unity",
)
FLUID = dict(OLD_FCC, peak_search_range=(0.015, 0.040), asymptote_band=(0.095, 0.120))
FLUID_WINDOWS = {"narrow": (0.01, 0.09), "full": (0.001, 0.10)}

def _qa():
    return saxsfft_q()


FCC_SIMS = {
    "q1_0125": lambda: crystal_curve(q1=0.0125, amp=6.0, plateau=0.8, noise=0.03, seed=1),
    "q1_0145": lambda: crystal_curve(q1=0.0145, amp=6.0, plateau=0.8, noise=0.03, seed=2),
    "flat_sim": flat_curve,
    "truncated_sim": lambda: crystal_curve(q1=0.0131, q=_qa()[_qa() < 0.0505]),
}
FLUID_SIMS = {
    "q1_025": lambda: fluid_curve(q1=0.025, noise=0.02, seed=3),
    "q1_029": lambda: fluid_curve(q1=0.029, amp=1.3, plateau=1.1, noise=0.02, seed=4),
}


def test_identical_curves_give_zero():
    loss, diag, *_ = shift_rmse_loss(crystal_curve(), crystal_curve(), None)
    assert loss == 0.0 and diag["q1_ratio"] == 1.0 and diag["aligned"] is True


# ---------------- lambda ---------------- #

@pytest.mark.parametrize("lam", [0.0, 1.0, 2.5])
def test_lambda_weights_only_the_shift_term(lam):
    sim = FCC_SIMS["q1_0145"]()
    ref = shift_rmse_loss(crystal_curve(), sim, W_NARROW,
                          metric_kwargs=dict(OLD_FCC, lambda_shift=0.0))[1]
    loss, diag, *_ = shift_rmse_loss(crystal_curve(), sim, W_NARROW,
                                    metric_kwargs=dict(OLD_FCC, lambda_shift=lam))
    assert diag["m4"] == pytest.approx(ref["m4"], rel=1e-12)
    assert diag["abs_log_q1_ratio"] == pytest.approx(abs(np.log(diag["q1_sim"] / diag["q1_tgt"])))
    assert diag["shift_term"] == pytest.approx(lam * diag["abs_log_q1_ratio"], rel=1e-12)
    assert loss == pytest.approx(ref["m4"] + lam * diag["abs_log_q1_ratio"], rel=1e-12)


# ---------------- q_range=None ---------------- #

def test_q_range_none_with_overlap_trim_none_raises():
    with pytest.raises(ValueError, match="q_range"):
        shift_rmse_loss(crystal_curve(), FCC_SIMS["q1_0125"](), None,
                       metric_kwargs={"overlap_trim": None})


@pytest.mark.parametrize("k", [0, 3, 10])
def test_q_range_none_uses_trimmed_overlap(k):
    exp, sim = crystal_curve(), FCC_SIMS["q1_0125"]()
    loss, diag, *_ = shift_rmse_loss(exp, sim, None, metric_kwargs={"overlap_trim": k})
    q1t, q1s = diag["q1_tgt"], diag["q1_sim"]
    x_lo = max(exp[k, 0] / q1t, sim[k, 0] / q1s)
    x_hi = min(exp[-1 - k, 0] / q1t, sim[-1 - k, 0] / q1s)
    assert diag["window_source"] == "overlap" and diag["q_range"] is None
    assert diag["x_lo"] == pytest.approx(x_lo) and diag["x_hi"] == pytest.approx(x_hi)
    assert diag["grid_lo"] == pytest.approx(x_lo) and diag["grid_hi"] == pytest.approx(x_hi)
    same, *_ = shift_rmse_loss(exp, sim, tuple(diag["window_abs"]))
    assert loss == pytest.approx(same, rel=1e-12)


def test_larger_overlap_trim_narrows_window():
    exp, sim = crystal_curve(), FCC_SIMS["q1_0125"]()
    d3 = shift_rmse_loss(exp, sim, None, metric_kwargs={"overlap_trim": 3})[1]
    d20 = shift_rmse_loss(exp, sim, None, metric_kwargs={"overlap_trim": 20})[1]
    assert d20["x_lo"] > d3["x_lo"] and d20["x_hi"] < d3["x_hi"]


# ---------------- no peak ---------------- #

def test_no_peak_fallback_is_flagged():
    loss, diag, *_ = shift_rmse_loss(crystal_curve(), flat_curve(), W_NARROW,
                                    metric_kwargs=OLD_FCC)
    assert np.isfinite(loss) and diag["shift_term"] > 0
    assert diag["dispersed_sim"] is False
    assert diag["peak_sim"]["n_peaks_found"] == 0
    assert diag["fallbacks"]["peak_sim_max_fallback"] is True
    assert diag["fallbacks"]["peak_tgt_max_fallback"] is False


def test_no_peak_fail_is_default():
    with pytest.raises(MetricFailed, match="no simulated peak"):
        shift_rmse_loss(crystal_curve(), flat_curve(), W_NARROW)
    loss, *_ = shift_rmse_loss(crystal_curve(), FCC_SIMS["q1_0125"](), W_NARROW,
                              metric_kwargs=dict(OLD_FCC, no_peak="fail"))
    same, *_ = shift_rmse_loss(crystal_curve(), FCC_SIMS["q1_0125"](), W_NARROW,
                              metric_kwargs=OLD_FCC)
    assert loss == pytest.approx(same, rel=1e-12)   # a curve with a peak: no_peak irrelevant


# ---------------- baseline and dispersed curves ---------------- #

def test_baseline_is_smoothed_minimum_in_search_window():
    curve = metrics._sanitize_curve(crystal_curve())
    pk = metrics.detect_first_peak(curve)
    q, s = curve[:, 0], curve[:, 1]
    lo, hi = metrics.SHIFT_RMSE_DEFAULTS["peak_search_range"]
    smooth = savgol_filter(s[(q >= lo) & (q <= hi)], window_length=5, polyorder=2)
    assert pk.baseline == pytest.approx(smooth.min())
    assert pk.s_max == pytest.approx(smooth.max())
    assert pk.dispersed is False and pk.n_peaks_found > 0


def test_dispersed_is_s_max_minus_one_below_delta():
    gas = metrics._sanitize_curve(gas_curve())
    pk = metrics.detect_first_peak(gas)
    assert pk.dispersed is True and np.isnan(pk.q1)
    assert pk.s_max - 1.0 < metrics.SHIFT_RMSE_DEFAULTS["dispersed_delta"]
    assert metrics.detect_first_peak(gas, dispersed_delta=0.0).dispersed is False


def test_dispersed_sim_is_not_aligned():
    exp, gas = crystal_curve(), gas_curve()
    loss, diag, *_ = shift_rmse_loss(exp, gas, None)
    assert diag["aligned"] is False and diag["dispersed_sim"] is True
    assert diag["q1_sim"] is None and diag["q1_ratio"] is None
    assert diag["x_scale_sim"] == diag["x_scale_tgt"] == pytest.approx(diag["q1_tgt"])
    assert diag["shift_term"] == 0.0 and loss == pytest.approx(diag["m4"])
    loss2, diag2, *_ = shift_rmse_loss(exp, gas, None, metric_kwargs={"dispersed_shift": 0.7})
    assert diag2["m4"] == pytest.approx(diag["m4"], rel=1e-12)
    assert loss2 == pytest.approx(diag["m4"] + 0.7, rel=1e-12)


def test_dispersed_target_uses_absolute_q():
    loss, diag, *_ = shift_rmse_loss(gas_curve(), crystal_curve(), None,
                                    metric_kwargs={"dispersed_shift": 0.7})
    assert diag["dispersed_tgt"] is True and diag["dispersed_sim"] is False
    assert diag["x_scale_tgt"] == diag["x_scale_sim"] == 1.0
    assert diag["shift_term"] == 0.7


def test_both_dispersed_have_no_shift_term():
    loss, diag, *_ = shift_rmse_loss(gas_curve(seed=1), gas_curve(seed=2), None,
                                    metric_kwargs={"dispersed_shift": 0.7})
    assert diag["dispersed_tgt"] and diag["dispersed_sim"]
    assert diag["shift_term"] == 0.0 and loss == pytest.approx(diag["m4"])


def test_too_few_search_points_always_fails():
    q = _qa()
    short = crystal_curve(q=q[q < 0.0062])
    with pytest.raises(MetricFailed, match="failed to detect simulated"):
        shift_rmse_loss(crystal_curve(), short, W_NARROW)


# ---------------- asymptote band ---------------- #

def test_asymptote_tail_fallback_is_flagged():
    loss, diag, *_ = shift_rmse_loss(crystal_curve(), FCC_SIMS["truncated_sim"](), W_NARROW,
                                    metric_kwargs=OLD_FCC)
    assert np.isfinite(loss)
    assert diag["fallbacks"]["asymptote_sim_tail_fallback"] is True
    assert diag["fallbacks"]["asymptote_exp_tail_fallback"] is False
    assert diag["asymptote"]["sim_band_points"] < 3


def test_asymptote_tail_fallback_fail_and_tail_frac():
    sim = FCC_SIMS["truncated_sim"]()
    with pytest.raises(MetricFailed, match="asymptote band"):
        shift_rmse_loss(crystal_curve(), sim, W_NARROW)
    exp = metrics._sanitize_curve(crystal_curve())
    s = metrics._sanitize_curve(sim)
    scale, info = asymptote_scale(exp, s, band=(0.050, 0.0654), fallback="tail", tail_frac=0.9)
    band = (exp[:, 0] >= 0.050) & (exp[:, 0] <= 0.0654)
    tail = s[:, 0] >= 0.9 * s[:, 0].max()
    assert scale == pytest.approx(exp[band, 1].mean() / s[tail, 1].mean())
    assert info["sim_tail_fallback"] and not info["exp_tail_fallback"]


def test_asymptote_zero_denominator_rule():
    exp = metrics._sanitize_curve(crystal_curve())
    sim = exp.copy()
    sim[:, 1] = 0.0
    scale, info = asymptote_scale(exp, sim, zero_den="unity")
    assert scale == 1.0 and info["zero_den"] is True
    with pytest.raises(MetricFailed, match="asymptote band mean"):
        asymptote_scale(exp, sim)


# ---------------- parameters and JSON ---------------- #

def test_params_validation_and_json_lists():
    p = shift_rmse_params(json.loads(json.dumps(FLUID)))
    assert p["peak_search_range"] == (0.015, 0.040)
    assert p["asymptote_band"] == (0.095, 0.120)
    with pytest.raises(ValueError, match="Unknown"):
        shift_rmse_params({"peak_range": (0.01, 0.02)})
    with pytest.raises(ValueError, match="no_peak"):
        shift_rmse_params({"no_peak": "skip"})
    for removed in ("peak_baseline_range", "min_prom_ratio", "require_reliable"):
        with pytest.raises(ValueError, match="Unknown"):
            shift_rmse_params({removed: None})
    with pytest.raises(ValueError, match="dispersed_delta"):
        shift_rmse_params({"dispersed_delta": -0.1})


def test_compare_writes_diagnostics_json(tmp_path):
    loss = metrics.compare_to_exp_saxsfft(
        crystal_curve(), FCC_SIMS["q1_0125"](), str(tmp_path),
        metric="shift_rmse", q_range=W_NARROW, metric_kwargs=OLD_FCC,
    )
    data = json.loads((tmp_path / "shift_rmse_diagnostics.json").read_text())
    direct, *_ = shift_rmse_loss(crystal_curve(), FCC_SIMS["q1_0125"](), W_NARROW,
                                 metric_kwargs=OLD_FCC)
    assert loss == pytest.approx(direct, rel=1e-12)
    old_fields = {"loss", "m4", "shift_term", "q1_ratio", "q1_tgt", "q1_sim",
                  "anchor_scale", "window_abs",
                  "x_lo", "x_hi", "n_points", "grid_lo", "grid_hi", "metric"}
    assert old_fields <= set(data)
    assert {"params", "fallbacks", "lambda_shift", "abs_log_q1_ratio",
            "peak_tgt", "peak_sim", "asymptote", "aligned",
            "dispersed_tgt", "dispersed_sim"} <= set(data)
    assert data["failed"] is False and data["loss"] == pytest.approx(loss)
    assert metrics.load_shift_rmse_components(str(tmp_path)) == {
        "rmse": pytest.approx(data["m4"]), "shift": pytest.approx(data["shift_term"]),
    }
    assert (tmp_path / "compare_to_exp_saxsfft.png").exists()


def test_compare_failure_writes_json_and_raises(tmp_path):
    with pytest.raises(MetricFailed):
        metrics.compare_to_exp_saxsfft(
            crystal_curve(), flat_curve(), str(tmp_path), metric="shift_rmse",
            q_range=W_NARROW,
        )
    data = json.loads((tmp_path / "shift_rmse_diagnostics.json").read_text())
    assert data["failed"] is True and "no simulated peak" in data["reason"]
    assert data["params"]["no_peak"] == "fail"
    assert metrics.load_shift_rmse_components(str(tmp_path)) == {}


def test_metric_kwargs_rejected_for_other_metrics(tmp_path):
    with pytest.raises(ValueError, match="only used by"):
        metrics.compare_to_exp_saxsfft(
            crystal_curve(), crystal_curve(), str(tmp_path), metric="mse",
            metric_kwargs={"lambda_shift": 2.0},
        )


# ---------------- make_global_objective validation ---------------- #

def _ps():
    fixed = {name: {"fixed": 1.0} for name in bo.PHYSICS_COEFFS}
    return bo.ParamSpace(fixed)


def test_objective_rejects_shift_rmse_without_window():
    with pytest.raises(ValueError, match="overlap_trim"):
        bo.make_global_objective([], _ps(), ffpath="", mode="map", metric="shift_rmse",
                                 compare_q_range=None, metric_kwargs={"overlap_trim": None})
    bo.make_global_objective([], _ps(), ffpath="", mode="map", metric="shift_rmse",
                             compare_q_range=None)


def test_objective_rejects_bad_metric_kwargs():
    with pytest.raises(ValueError, match="Unknown"):
        bo.make_global_objective([], _ps(), ffpath="", mode="map", metric="shift_rmse",
                                 metric_kwargs={"lamda_shift": 1.0})
    with pytest.raises(ValueError, match="only used by"):
        bo.make_global_objective([], _ps(), ffpath="", mode="map", metric="mse",
                                 metric_kwargs={"lambda_shift": 1.0})
    with pytest.raises(ValueError, match="saxsfft"):
        bo.make_global_objective([], _ps(), ffpath="", mode="map", metric="shift_rmse",
                                 scattering_method="mcdfm")
