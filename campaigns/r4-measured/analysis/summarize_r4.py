"""
Tables and figures for the R4 note (training on measured curves).

Usage: python summarize_r4.py RUN_DIR FIG_DIR [--eval N]

RUN_DIR is the downloaded Optimization_Results copy (bo_trajectory.csv and
whichever eval_XXX folders were downloaded), read only; the measured I(q)
files are the repo copies in r4_config.DATA_DIR. ``--eval`` picks the
evaluation drawn against the measured curves (default: the best). Writes
next to this script:
  r4_evals.csv       one row per evaluation: status (ok / FAILED / ABANDONED),
                     coefficients, total loss, best so far
  r4_conditions.csv  one row per condition at the drawn evaluation: mapped
                     (U0, r0, n, m), loss, and, when the eval folder is there,
                     m4, shift term, q1 of both curves, aligned / dispersed
and to FIG_DIR:
  r4_trajectory.png  total loss (log) and coefficients against evaluation
  r4_sq_eval_XXX.png measured S(q) (as the objective computes it) and the
                     simulated S(q) of the drawn evaluation, per condition
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
import r4_config as config  # noqa: E402
from datasets import physics_map  # noqa: E402

COEFFS = list(config.PARAM_CFG)


def read_trajectory(path):
    """One dict per (evaluation, condition) row, with the block status."""
    rows, status, header = [], None, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("# Iteration"):
            parts = line[2:].split(",")
            status = {"total_loss": "ok"}.get(parts[1], parts[-1] if parts[1] != "EVALUATION"
                                               else parts[2])
            continue
        if line.startswith("iteration,"):
            header = line.split(",")
            continue
        if not line or line.startswith("#") or header is None:
            continue
        rec = dict(zip(header, line.split(",")))
        rec["status"] = status
        rows.append(rec)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c not in ("dataset_id", "status"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def eval_table(df):
    g = df.groupby("iteration")
    ev = g[COEFFS].first()
    ev["status"] = g["status"].first()
    ev["total_loss"] = g["loss"].sum(min_count=1).where(ev["status"] == "ok")
    ev["best_so_far"] = ev["total_loss"].cummin()
    return ev.reset_index().rename(columns={"iteration": "eval"})


def measured_sq(cond):
    from scattering import extract_exp_sq

    arr = np.load(config.DATA_DIR / cond["file"])[:, :2]
    lo, hi = config.EXTRACT_Q_WINDOW
    return extract_exp_sq(arr[:-config.TRIM_TAIL], ffpath=str(REPO / config.FORMFACTOR),
                          q_min=lo, q_max=hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("fig_dir")
    ap.add_argument("--eval", type=int, default=None)
    a = ap.parse_args()
    run, fig_dir = Path(a.run_dir), Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = read_trajectory(run / "bo_trajectory.csv")
    ev = eval_table(df)
    ev.to_csv(HERE / "r4_evals.csv", index=False)
    ok = ev[ev["status"] == "ok"]
    pick = int(ok.loc[ok["total_loss"].idxmin(), "eval"]) if a.eval is None else a.eval
    print(f"evaluations: {len(ev)} ({len(ok)} ok); best eval "
          f"{int(ok.loc[ok['total_loss'].idxmin(), 'eval'])} total {ok['total_loss'].min():.4f}; "
          f"drawn: eval {pick}")

    coeffs = ev.set_index("eval").loc[pick, COEFFS].to_dict()
    rows = []
    for c in config.CONDITIONS:
        m = physics_map(c["L_bridge"], c["C_anchor"], c["C_NaCl"], **coeffs)
        row = {"id": c["id"], "L_bridge": c["L_bridge"], "C_anchor": c["C_anchor"],
               "C_NaCl": c["C_NaCl"], **m}
        hit = df[(df["iteration"] == pick) & (df["dataset_id"] == c["id"])]
        row["loss"] = float(hit["loss"].iloc[0]) if len(hit) else np.nan
        diag_path = run / f"eval_{pick:03d}" / c["id"] / "shift_rmse_diagnostics.json"
        if diag_path.exists():
            d = json.loads(diag_path.read_text())
            row.update({k: d.get(k) for k in ("m4", "shift_term", "q1_tgt", "q1_sim", "aligned",
                                              "dispersed_tgt", "dispersed_sim")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "r4_conditions.csv", index=False)

    # trajectory
    fig, ax = plt.subplots(1 + len(COEFFS), 1, figsize=(8, 11), sharex=True)
    ax[0].semilogy(ok["eval"], ok["total_loss"], "o", ms=3, color="0.4", label="total loss")
    ax[0].semilogy(ok["eval"], ok["best_so_far"], "-", color="C3", label="best so far")
    bad = ev[ev["status"] != "ok"]
    for e in bad["eval"]:
        ax[0].axvline(e, color="C1", lw=0.5, alpha=0.5)
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"R4: {len(ok)} ok, {len(bad)} failed / abandoned (orange)")
    for axi, name in zip(ax[1:], COEFFS):
        axi.plot(ok["eval"], ok[name], "o", ms=3)
        axi.set_ylim(*config.PARAM_CFG[name]["bounds"])
        axi.set_ylabel(name)
    ax[-1].set_xlabel("evaluation")
    fig.tight_layout()
    fig.savefig(fig_dir / "r4_trajectory.png", dpi=150)
    plt.close(fig)

    # S(q) at the drawn evaluation
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axi, c, row in zip(axes.ravel(), config.CONDITIONS, rows):
        tgt = measured_sq(c)
        axi.loglog(tgt[:, 0], tgt[:, 1], ".", ms=3, color="C3", label="measured")
        sim_path = run / f"eval_{pick:03d}" / c["id"] / "S(q)_data" / "average_structure_factor.npy"
        if sim_path.exists():
            sim = np.load(sim_path)
            axi.loglog(sim[:, 0], sim[:, 1], "-", color="C2", lw=1, label=f"eval {pick}")
        axi.axhline(1, color="k", lw=0.6)
        axi.set_title(f"{c['id']}: L {c['L_bridge']:.0f}, C_anchor {c['C_anchor']:.0f}, "
                      f"NaCl {c['C_NaCl']:.0f}, loss {row['loss']:.3f}", fontsize=10)
        axi.set_xlabel("q (Å⁻¹)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"r4_sq_eval_{pick:03d}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
