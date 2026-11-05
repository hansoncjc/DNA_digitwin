"""R7.0: I(q) / form-factor audit and the refitted S(q) extraction."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "campaigns" / "r7.0-sq-extraction"))
import sq_extraction as sx  # noqa: E402

FF_FILE = REPO / "formfactors" / "sasmodels_sphere_fit.txt"
R4_DATA = REPO / "campaigns" / "r4-measured" / "data"


def test_formfactor_file_is_sphere_plus_default_background():
    a = sx.audit_formfactor(FF_FILE)
    assert a["radius_A"] == pytest.approx(134.21, abs=0.05)
    assert a["pd"] == pytest.approx(0.100, abs=0.002)
    assert a["background"] == pytest.approx(1e-3, rel=0.02)
    assert a["rms_ln_resid"] < 1e-3
    assert 0.2 < a["background_share_at_qmax"] < 0.3


def test_sphere_pd_limits():
    q = np.linspace(0.002, 0.08, 200)
    r = 128.0
    x = q * r
    mono = (3 * (np.sin(x) - x * np.cos(x)) / x ** 3) ** 2
    np.testing.assert_allclose(sx.sphere_pd(q, r, 0.0), mono, rtol=1e-12)
    np.testing.assert_allclose(sx.sphere_pd(q, r, 1e-4), mono, rtol=1e-3, atol=1e-8)
    assert sx.sphere_pd(np.array([1e-6]), r, 0.1)[0] == pytest.approx(1.0, abs=1e-6)
    assert sx.first_ff_minimum(r) == pytest.approx(4.4934 / r, rel=1e-3)
    assert sx.first_ff_minimum(r, 0.1) < sx.first_ff_minimum(r)


@pytest.mark.parametrize("transition", [0.85e-4, 2.54e-4, 3.4e-4])
def test_find_merge(transition):
    # The step between the two grids seen in the R4 files is 0.85e-4 or
    # 2.54e-4; when it equals the coarse spacing, coarse stepping starts at
    # the last fine point.
    fine = 0.0026 + 1.7e-4 * np.arange(150)
    start = fine[-1] + transition
    coarse = start + 3.4e-4 * np.arange(300)
    m = sx.find_merge(np.concatenate([fine, coarse]))
    assert m["q_merge"] == pytest.approx(fine[-1] if transition >= 0.9 * 3.4e-4 else start)
    assert m["dq_low"] == pytest.approx(1.7e-4)
    assert m["dq_high"] == pytest.approx(3.4e-4)
    assert sx.implied_distance_mm(m["dq_low"]) == pytest.approx(1799, abs=2)
    assert sx.find_merge(0.0026 + 3.4e-4 * np.arange(400)) is None


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    q = np.linspace(0.003, 0.14, 500)
    r, pd = 128.0, 0.11
    curves, truth = [], []
    for k, (c, b, q1) in enumerate([(150.0, 0.02, 0.0135), (120.0, 0.01, 0.0105), (90.0, 0.03, 0.0139)]):
        s_true = 1.0 + 0.8 * np.exp(-0.5 * ((q - q1) / 0.0015) ** 2) - 0.5 * np.exp(-q / 0.006)
        i = c * sx.sphere_pd(q, r, pd) * s_true + b
        sig = 0.01 * i
        curves.append((q, i * (1 + 0.01 * rng.standard_normal(q.size)), sig))
        truth.append((c, b, s_true))
    return curves, truth, r, pd


def test_joint_fit_and_extraction_recover_truth():
    curves, truth, r, pd = _synthetic()
    fit = sx.fit_formfactor_joint(curves, (0.033, 0.12), radius0=134.0, pd0=0.1)
    assert fit["radius_A"] == pytest.approx(r, abs=0.5)
    assert fit["pd"] == pytest.approx(pd, abs=0.005)
    for k, (c, b, s_true) in enumerate(truth):
        assert fit["scale"][k] == pytest.approx(c, rel=0.02)
        assert fit["background"][k] == pytest.approx(b, rel=0.1)
        q, i, s = curves[k]
        sq = sx.extract_sq(q, i, s, fit["radius_A"], fit["pd"], fit["scale"][k], fit["background"][k])
        m = q < 0.03
        assert np.median(np.abs(sq[m, 1] - s_true[m])) < 0.03
        q1, fwhm = sx.peak_fwhm(sq[:, 0], sq[:, 1], (0.006, 0.02))
        assert fwhm == pytest.approx(2.3548 * 0.0015, rel=0.1)


def test_load_iq_text(tmp_path):
    p3 = tmp_path / "a.dat"
    p3.write_text("# header\nq I err\n0.01 5.0 0.1\n0.02, 4.0, 0.1\n0.005 6.0 0.2\nnan 1 1\n")
    q, i, s = sx.load_iq(p3)
    np.testing.assert_allclose(q, [0.005, 0.01, 0.02])
    np.testing.assert_allclose(i, [6.0, 5.0, 4.0])
    np.testing.assert_allclose(s, [0.2, 0.1, 0.1])
    p2 = tmp_path / "b.txt"
    p2.write_text("0.01 5.0\n0.02 4.0\n")
    assert sx.load_iq(p2)[2] is None


def test_cli_on_two_r4_curves(tmp_path):
    files = [str(R4_DATA / "chol_100_nacl_10_bridge_20.npy"), str(R4_DATA / "chol_100_nacl_10_bridge_80.npy")]
    rep = sx.main(["--ff", str(FF_FILE), "--out", str(tmp_path), "--old-trim-tail", "200", "--no-scan"] + files)
    assert 120 < rep["joint_fit"]["radius_A"] < 135
    for name in ("summary.csv", "fit.json", "sq_compare.png", "iq_audit.png", "ff_audit.png",
                 "chol_100_nacl_10_bridge_20_sq.npy"):
        assert (tmp_path / name).exists()
    sq = np.load(tmp_path / "chol_100_nacl_10_bridge_80_sq.npy")
    assert sq.shape[1] == 3 and np.all(np.isfinite(sq[:, 2]))


def test_iq4_extrema_monodisperse():
    r = 130.0
    q = np.linspace(0.02, 0.1, 6000)
    ext = sx.iq4_extrema(q, sx.sphere_pd(q, r, 0.0), r)
    # zeros of the sphere amplitude: tan(x) = x
    assert ext["min1"] * r == pytest.approx(4.4934, abs=0.01)
    assert ext["min2"] * r == pytest.approx(7.7253, abs=0.01)
    assert ext["min1"] < ext["max2"] < ext["min2"] < ext["max3"]
    # a smaller radius moves every extremum up by R_ref / R
    ext2 = sx.iq4_extrema(q, sx.sphere_pd(q, 0.96 * r, 0.0), r)
    for k in ext:
        assert ext2[k] / ext[k] == pytest.approx(1 / 0.96, rel=2e-3)


def _dispersed_with_bump(seed, bump):
    """I = c * P_sphere * (1 + bump(q)) + B, sigma 2 % of I, S = 1."""
    rng = np.random.default_rng(seed)
    q = np.arange(0.004, 0.16, 3.4e-4)
    c, b = 50.0 + 10 * seed, 0.002
    i = c * sx.sphere_pd(q, 123.5, 0.127) * (1 + bump(q)) + b
    sig = 0.02 * i
    return q, i * (1 + 0.02 * rng.standard_normal(q.size)), sig, c, b


def test_empirical_correction_recovers_a_bump():
    bump = lambda q: 0.3 * np.exp(-0.5 * ((q - 0.045) / 0.004) ** 2)
    data = [_dispersed_with_bump(k, bump) for k in range(6)]
    ec = sx.empirical_correction([(q, i, s) for q, i, s, _, _ in data], 123.5, 0.127,
                                 [c for *_, c, _ in data], [b for *_, b in data])
    m = (ec["q"] > 0.03) & (ec["q"] < 0.07)
    np.testing.assert_allclose(ec["r"][m], 1 + bump(ec["q"][m]), atol=0.03)
    assert np.all(ec["r"][(ec["q"] < 0.015) | (ec["q"] > 0.075)] == 1.0)
    p = sx.form_factor(np.array([0.045]), 123.5, 0.127, (ec["q"], ec["r"]))
    assert p[0] / sx.sphere_pd(np.array([0.045]), 123.5, 0.127)[0] == pytest.approx(1.3, abs=0.03)


def test_mean_s_artifact():
    q = np.linspace(0.005, 0.1, 400)
    flat = [np.column_stack([q, np.ones_like(q)]) for _ in range(3)]
    assert sx.mean_s_artifact(flat)[0] == pytest.approx(0.0, abs=1e-12)
    bumped = [np.column_stack([q, 1 + 0.2 * np.exp(-0.5 * ((q - 0.035) / 0.002) ** 2)]) for _ in range(3)]
    assert sx.mean_s_artifact(bumped)[0] > 0.05


def test_cli_dispersed_reference_then_bridged(tmp_path):
    d = REPO / "Experimental_Data" / "Huat" / "SAXS_data_rt_Collection"
    if not d.exists():
        pytest.skip("Experimental_Data not in this checkout")
    disp = [str(d / f"chol_{c}_nacl_10_bridge_00.npy") for c in ("060", "100", "140")]
    ref = sx.main(["--ff", str(FF_FILE), "--out", str(tmp_path / "ref"), "--fit-window", "0.015", "0.16",
                   "--no-scan", "--write-correction"] + disp)
    j = ref["joint_fit"]
    assert 118 < j["radius_A"] < 130 and 0.1 < j["pd"] < 0.16
    corr = tmp_path / "ref" / "ff_correction.npy"
    assert corr.exists()
    rep = sx.main(["--ff", str(FF_FILE), "--out", str(tmp_path / "br"), "--fix-ff", str(j["radius_A"]),
                   str(j["pd"]), "--ff-correction", str(corr), str(d / "chol_100_nacl_10_bridge_20.npy")])
    assert rep["joint_fit"]["fixed_formfactor"] and rep["joint_fit"]["radius_A"] == j["radius_A"]
    assert np.isfinite(rep["mean_s_artifact_rms_0.026_0.045"]["new"])
    with pytest.raises(SystemExit):
        sx.main(["--ff", str(FF_FILE), "--out", str(tmp_path / "x"), "--ff-correction", str(corr),
                 str(d / "chol_100_nacl_10_bridge_20.npy")])
