"""
GP sensitivity for the R3.2 note (Results 6; advisor meeting 2026-10-07, point 5).

Usage: python gp_r32.py RUN_DIR OUT_DIR

Figures go to OUT_DIR (the note's image folder); r32_gp_summary.json goes next
to this script.

RUN_DIR is the downloaded Optimization_Results copy (gp_log/gp_log.jsonl,
gp_log/states/*.pt, bo_trajectory.csv; read only). Writes into OUT_DIR:
  r32_gp_lengthscales.png  ARD lengthscales and the GP noise level against the
                           number of training points
  r32_gp_profiles.png      posterior mean +- 2 sd along each coefficient
                           through the best evaluation
  r32_gp_maps.png          posterior mean on the (A, K_s) and (a_m, delta)
                           planes through the best evaluation
  r32_gp_summary.json      posterior at the best evaluation, at the ground
                           truth and at the posterior-mean minimum; Hessian of
                           the posterior mean at the minimum, its eigenvectors,
                           and the coefficient widths it implies

The GP is rebuilt in numpy from the logged hyperparameters (SingleTaskGP of
botorch 0.10 on the cluster: ScaleKernel(Matern 5/2, ARD), constant mean,
Gaussian noise, Standardize on train_y = -loss). bo.load_gp_model cannot be
used locally: botorch >= 0.12 builds a different default kernel, so the saved
state_dict does not load. The rebuild is checked against the posterior mean
and sd that run_bo logged for every fitted GP (best point and candidates;
rows of GP log format 1 and 2). The analysed GP is the last row of the log:
stage="final" when the run ended normally (R3.2: n_train = 100), otherwise the
last acquisition GP.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

NAMES = ["k", "A", "K_s", "a_m", "delta"]
LABEL = {"k": r"$k$", "A": r"$A$ ($k_BT$)", "K_s": r"$K_s$ (1/mM)", "a_m": r"$a_m$", "delta": r"$\delta$"}
LO = np.array([0.40, 0.85, 0.00, 2.91, 1.09])     # train_config.PARAM_CFG
HI = np.array([0.90, 2.50, 0.10, 4.00, 4.00])
GT = np.array([0.65, 1.2, 0.07, 3.2, 3.2])
WINDOW = {"k": (0.55, 0.75), "A": (0.95, 1.50), "K_s": (0.04, 0.10), "a_m": (2.91, 4.00), "delta": (1.09, 4.00)}

BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d23c3c"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e6e5e1", "#fcfcfb"
COL = {"k": BLUE, "A": ORANGE, "K_s": AQUA, "a_m": YELLOW, "delta": VIOLET}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 160,
})


def to_unit(p):
    return (np.asarray(p) - LO) / (HI - LO)


def to_phys(u):
    return LO + np.asarray(u) * (HI - LO)


class GP:
    """Latent posterior of one logged GP, in loss units (loss = -train_y)."""

    def __init__(self, run, row):
        b = torch.load(Path(run) / "gp_log" / row["state_file"], map_location="cpu", weights_only=False)
        assert b["surrogate"] == "stgp" and b["outcome_transform"] == "Standardize"
        self.X = b["train_x"].double().numpy()
        y = b["train_y"].double().numpy().ravel()
        self.ls = np.array(row["lengthscale"])
        self.os, self.nz, self.mc = row["outputscale"], row["noise"], row["mean_constant"]
        self.mu, self.sd = row["standardize_mean"], row["standardize_std"]
        K = self.os * self._k(self.X, self.X) + self.nz * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, (y - self.mu) / self.sd - self.mc))
        self.noise_sd = self.sd * np.sqrt(self.nz)

    def _k(self, A, B):
        d = np.sqrt((((A[:, None, :] - B[None, :, :]) / self.ls) ** 2).sum(-1))
        s = np.sqrt(5.0) * d
        return (1.0 + s + 5.0 / 3.0 * d ** 2) * np.exp(-s)

    def __call__(self, U):
        U = np.atleast_2d(U)
        ks = self.os * self._k(U, self.X)
        m = self.mc + ks @ self.alpha
        v = np.linalg.solve(self.L, ks.T)
        var = np.clip(self.os - (v ** 2).sum(0), 0.0, None)
        return -(self.mu + self.sd * m), self.sd * np.sqrt(var)


def check_rebuild(run, rows):
    worst = 0.0
    for r in rows:
        gp = GP(run, r)
        if r["gp_log_version"] == 1:
            cu, cm, cs = [r["candidate_unit"]], [r["candidate_post_mean_loss"]], [r["candidate_post_std_loss"]]
        else:
            cu, cm, cs = r["candidates_unit"], r["candidates_post_mean_loss"], r["candidates_post_std_loss"]
        m, s = gp(np.array([r["best_unit"], *cu]))
        ref = [r["best_post_mean_loss"], *cm, r["best_post_std_loss"], *cs]
        worst = max(worst, np.abs(np.r_[m, s] - np.array(ref)).max())
    return worst


def hessian_phys(gp, p0, rel=1e-3):
    h = rel * (HI - LO)
    f = lambda p: gp(to_unit(p))[0][0]
    H = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            ei = np.zeros(5); ej = np.zeros(5); ei[i] = h[i]; ej[j] = h[j]
            H[i, j] = (f(p0 + ei + ej) - f(p0 + ei - ej) - f(p0 - ei + ej) + f(p0 - ei - ej)) / (4 * h[i] * h[j])
    return 0.5 * (H + H.T)


def main(run, out):
    run, out = Path(run), Path(out)
    rows = [json.loads(l) for l in (run / "gp_log" / "gp_log.jsonl").read_text().splitlines() if l.strip()]
    worst = check_rebuild(run, rows)
    print(f"{len(rows)} GP rows, stages {sorted(set(r['stage'] for r in rows))}; "
          f"rebuild vs logged posterior: max abs difference {worst:.2e}")
    last = rows[-1]
    gp = GP(run, last)
    lines = (run / "bo_trajectory.csv").read_text().splitlines()
    complete = {int(l.split(",")[0].split()[-1]) for l in lines if l.startswith("# Iteration") and ",total_loss," in l}
    ev = {}
    for l in lines:
        if l and not l.startswith(("#", "iteration")):
            t = l.split(",")
            if int(t[0]) in complete:
                ev.setdefault(int(t[0]), [float(x) for x in t[3:8]])
    print(f"GP analysed: stage {last['stage']}, n_train {last['n_train']}, {last['state_file']}; "
          f"{len(ev)} complete evaluations in the trajectory")
    ub = np.array(last["best_unit"])
    pb = to_phys(ub)
    best_eval = next(e for e, v in ev.items() if np.allclose(v, pb))   # best_index counts training rows, not eval ids

    # posterior-mean minimum
    starts = [ub] + [gp.X[i] for i in np.argsort(gp.X @ np.zeros(5))[:0]] + list(gp.X[np.argsort(gp(gp.X)[0])[:10]])
    res = min((minimize(lambda u: gp(u)[0][0], s, bounds=[(0, 1)] * 5, method="L-BFGS-B") for s in starts),
              key=lambda r: r.fun)
    pmin = to_phys(res.x)
    H = hessian_phys(gp, pmin)
    sn = gp.noise_sd
    # widths where the posterior mean rises by one noise sd: along an axis (others fixed),
    # and along the profile (others re-optimized, quadratic approximation)
    w_axis = np.sqrt(2 * sn / np.diag(H))
    Hinv = np.linalg.inv(H)
    w_prof = np.sqrt(2 * sn * np.diag(Hinv))
    corr = Hinv / np.sqrt(np.outer(np.diag(Hinv), np.diag(Hinv)))
    # (a_m, a_n) coordinates: delta = a_n - a_m
    J = np.eye(5); J[4, 3] = -1.0          # d(theta)/d(theta') with theta' = (k, A, K_s, a_m, a_n)
    H2 = J.T @ H @ J
    H2inv = np.linalg.inv(H2)
    w2_axis = np.sqrt(2 * sn / np.diag(H2)); w2_prof = np.sqrt(2 * sn * np.diag(H2inv))
    corr2 = H2inv / np.sqrt(np.outer(np.diag(H2inv), np.diag(H2inv)))
    # eigenvectors in relative units (fraction of the ground-truth value)
    S = np.diag(GT)
    w, V = np.linalg.eigh(S @ H @ S)
    eig = [{"eigenvalue": float(wi), "rel_width_1noise": float(np.sqrt(2 * sn / wi)),
            "vector_rel": dict(zip(NAMES, np.round(vi * np.sign(vi[np.abs(vi).argmax()]), 3).tolist()))}
           for wi, vi in zip(w, V.T)]

    def post(p):
        m, s = gp(to_unit(p))
        return {"mean_loss": float(m[0]), "sd_loss": float(s[0])}

    summary = {
        "gp": {"n_train": last["n_train"], "stage": last["stage"], "state_file": last["state_file"],
               "lengthscale_unit": dict(zip(NAMES, last["lengthscale"])),
               "lengthscale_phys": dict(zip(NAMES, (np.array(last["lengthscale"]) * (HI - LO)).tolist())),
               "noise_sd_loss": float(sn), "rebuild_max_abs_diff": float(worst)},
        "best_eval": best_eval, "best_phys": dict(zip(NAMES, pb.tolist())), "posterior_at_best": post(pb),
        "posterior_at_gt": post(GT),
        "post_mean_min_phys": dict(zip(NAMES, pmin.tolist())), "posterior_at_min": post(pmin),
        "post_mean_min_rel_err_pct": dict(zip(NAMES, (100 * (pmin - GT) / GT).tolist())),
        "post_mean_min_a_n": float(pmin[3] + pmin[4]),
        "width_1noise_axis": dict(zip(NAMES, w_axis.tolist())),
        "width_1noise_profile": dict(zip(NAMES, w_prof.tolist())),
        "corr_from_inverse_hessian": {a: dict(zip(NAMES, np.round(corr[i], 3).tolist())) for i, a in enumerate(NAMES)},
        "am_an_coords": {"names": ["k", "A", "K_s", "a_m", "a_n"],
                         "width_1noise_axis": w2_axis.tolist(), "width_1noise_profile": w2_prof.tolist(),
                         "corr_a_m_a_n": float(corr2[3, 4])},
        "eigen_rel": eig,
    }
    (Path(__file__).resolve().parent / "r32_gp_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("posterior_at_best", "posterior_at_gt", "post_mean_min_phys",
                                              "posterior_at_min", "post_mean_min_rel_err_pct", "post_mean_min_a_n",
                                              "width_1noise_axis", "width_1noise_profile")}, indent=1))
    print("lengthscale phys", summary["gp"]["lengthscale_phys"], "noise sd", sn)
    print("(a_m, a_n): axis widths", np.round(w2_axis, 4), "profile widths", np.round(w2_prof, 4), "corr", round(corr2[3, 4], 3))
    print("corr (theta):\n", np.round(corr, 2))
    for e in eig:
        print(e)

    # ---------------------------------------------------------------- lengthscales
    n = np.array([r["n_train"] for r in rows])
    ls = np.array([r["lengthscale"] for r in rows])
    nsd = np.array([r["standardize_std"] * np.sqrt(r["noise"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for i, c in enumerate(NAMES):
        ax.plot(n, ls[:, i], color=COL[c], label=f"{c}: {ls[-1, i]:.2f} (= {ls[-1, i] * (HI - LO)[i]:.3g} in {c} units)")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.text(n[-1], 1.05, "box width", ha="right", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_xlabel("training points of the GP (= evaluations so far)")
    ax.set_ylabel("ARD lengthscale (unit cube)")
    ax.set_title("(a) lengthscales; short = the loss changes fast along that coefficient", loc="left", fontsize=9.5)
    ax.set_ylim(0.08, 3.0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
              title=f"at n = {n[-1]}", title_fontsize=8)
    ax = axes[1]
    ax.plot(n, nsd, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("training points of the GP")
    ax.set_ylabel("GP noise sd (loss units)")
    ax.set_title(f"(b) fitted noise level: {nsd[-1]:.3f} at n = {n[-1]}", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / "r32_gp_lengthscales.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- profiles
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, NAMES)):
        x = np.linspace(*WINDOW[c], 300)
        P = np.tile(pb, (len(x), 1)); P[:, i] = x
        m, s = gp(to_unit(P))
        ax.fill_between(x, m - 2 * s, m + 2 * s, color=COL[c], alpha=0.18, lw=0)
        ax.plot(x, m, color=COL[c])
        ax.axvline(GT[i], color=INK, lw=1.0, ls="--")
        ax.axvline(pb[i], color=RED, lw=1.0)
        ax.axvline(pmin[i], color=MUTED, lw=1.0, ls=":")
        mb = gp(ub)[0][0]
        ax.axhspan(mb, mb + sn, color=GRID, alpha=0.8, lw=0)
        ax.set_xlabel(LABEL[c])
        ax.set_xlim(*WINDOW[c])
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3.0)
        ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0], ["0.6", "0.8", "1.0", "1.5", "2.0", "3.0"])
        ax.minorticks_off()
        ax.set_title(f"{c}: 1-noise half-width {w_axis[i]:.3g}\n(profile, others re-fitted: {w_prof[i]:.3g})",
                     loc="left", fontsize=8.5, pad=30 if c == "delta" else 6)
        if c == "delta":
            sec = ax.secondary_xaxis("top", functions=(lambda d: d + pb[3], lambda a: a - pb[3]))
            sec.set_xlabel(r"$a_n = a_m + \delta$ at the best $a_m$", fontsize=8, color=INK2)
    axes[0].set_ylabel("GP posterior mean loss (+- 2 sd)")
    fig.legend([plt.Line2D([], [], color=INK, ls="--", lw=1), plt.Line2D([], [], color=RED, lw=1),
                plt.Line2D([], [], color=MUTED, ls=":", lw=1), plt.Rectangle((0, 0), 1, 1, color=GRID)],
               ["ground truth", f"best evaluation (eval {best_eval})", "posterior-mean minimum",
                "best + one noise sd"], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "r32_gp_profiles.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- 2-D maps
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    pts = np.array(list(ev.values()))
    pairs = [("A", "K_s", (0.95, 1.5), (0.04, 0.10)), ("a_m", "delta", (2.91, 4.0), (1.09, 4.0))]
    for ax, (a, b, xr, yr) in zip(axes, pairs):
        i, j = NAMES.index(a), NAMES.index(b)
        xs, ys = np.linspace(*xr, 120), np.linspace(*yr, 120)
        XX, YY = np.meshgrid(xs, ys)
        P = np.tile(pb, (XX.size, 1)); P[:, i] = XX.ravel(); P[:, j] = YY.ravel()
        m = gp(to_unit(P))[0].reshape(XX.shape)
        mb = gp(ub)[0][0]
        cs = ax.contourf(XX, YY, np.log10(m), levels=20, cmap="Blues_r")
        ax.contour(XX, YY, m, levels=[mb + sn, mb + 2 * sn, mb + 4 * sn], colors=[INK, INK2, MUTED],
                   linewidths=[1.2, 0.9, 0.7])
        inbox = (pts[:, i] >= xr[0]) & (pts[:, i] <= xr[1]) & (pts[:, j] >= yr[0]) & (pts[:, j] <= yr[1])
        ax.scatter(pts[inbox, i], pts[inbox, j], s=8, color="white", edgecolor=INK2, lw=0.4, zorder=3, label="evaluations (projected)")
        ax.scatter(GT[i], GT[j], marker="*", s=140, color=YELLOW, edgecolor=INK, zorder=5, label="ground truth")
        ax.scatter(pb[i], pb[j], marker="o", s=45, color=RED, edgecolor=INK, zorder=5, label=f"best (eval {best_eval})")
        ax.scatter(pmin[i], pmin[j], marker="X", s=55, color=AQUA, edgecolor=INK, zorder=5, label="posterior-mean minimum")
        if (a, b) == ("A", "K_s"):
            for salt, ls_ in ((10.0, "-"), (20.0, "--")):
                u0 = GT[1] * (1 + GT[2] * salt)
                ax.plot(xs, (u0 / xs - 1) / salt, color=ORANGE, lw=1.0, ls=ls_)
                xt = 1.30 if salt == 10.0 else 1.42
                ax.text(xt, (u0 / xt - 1) / salt, f"U0({salt:g} mM) = GT ", color=ORANGE,
                        fontsize=7.5, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        else:
            for an in (6.0, 6.4, 7.0):
                ax.plot(xs, an - xs, color=ORANGE, lw=0.8, ls="--")
                ax.text(3.98, an - 3.98 + 0.03, f"a_n = {an:g}", color=ORANGE, fontsize=7.5, ha="right", va="bottom",
                        bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        ax.set_xlabel(LABEL[a]); ax.set_ylabel(LABEL[b])
        ax.grid(False)
        others = ", ".join(f"{c}={pb[q]:.4g}" for q, c in enumerate(NAMES) if c not in (a, b))
        ax.set_title(f"posterior mean on ({a}, {b}); {others}", loc="left", fontsize=8.5)
        fig.colorbar(cs, ax=ax, fraction=0.045, pad=0.02).set_label("log10 posterior mean loss")
    axes[0].legend(frameon=True, fontsize=7.5, loc="lower left", framealpha=0.9)
    fig.text(0.01, -0.03, "Contours: best-evaluation posterior mean + 1, 2, 4 noise sd (black to grey). "
             "Evaluation markers are projections of 5-D points onto the plane.", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out / "r32_gp_maps.png", bbox_inches="tight")
    plt.close(fig)
    print("figures written to", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])
