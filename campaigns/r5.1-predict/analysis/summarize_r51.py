"""
R5.1 predictions: collect per-condition facts into one CSV and check the run.

Sources (read only):
  predictions/<id>/pred_params.json        condition, coefficients, mapped (U0, r0, n, m),
                                           table bounds, run kwargs, seed, git head, sim_result
  predictions/<id>.npy                     S(q) (must equal <id>/S(q)_data/average_structure_factor.npy)
  predictions/<id>/potential_energy.csv    energy / temperature in the S(q) window
  predictions/<id>/DNA_assembly_*.gsd      frame count; clusters in the last frame
  pred_<i>.out                             GPU name, production TPS, neighbor-list stats, errors

First peak as in the loss and the R3.1 QC: metrics.detect_first_peak with the
shift_rmse defaults (smoothed s_max - 1 in (0.006, 0.12) 1/A, dispersed below
0.5, NEAR within 0.25 of it). Clusters as in the R3.1/R3.2 notes: neighbours
closer than 1.3 r0 (minimum image), clusters of >= 50 particles, last frame.

Checks (printed; exit code 1 if any fails): the configured coefficients and
mapped parameters match predict_config, seed and run kwargs are the defaults,
production ran the full 22,500,000 steps, no pair below rmin, no dangerous
neighbor-list update, the copied S(q) equals the run's S(q).

Usage: python summarize_r51.py RUN_DIR [OUT_CSV]
RUN_DIR holds predictions/ and pred_<i>.out. OUT_CSV defaults to
r51_summary.csv next to this script.
"""
import csv
import json
import re
import sys
from pathlib import Path

import gsd.hoomd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))

import predict_config as config  # noqa: E402
from metrics import SHIFT_RMSE_DEFAULTS, _sanitize_curve, detect_first_peak  # noqa: E402

PROD_STEPS = 22_500_000
SQ_WINDOW_STEPS = 100 * 50_000     # S(q) uses the last 100 GSD frames, one per 50 tau
NEAR = 0.25
CLUSTER_CUT = 1.3                  # x r0
CLUSTER_MIN = 50


def parse_log(path):
    text = path.read_text(errors="replace")
    gpu = re.search(r"^(NVIDIA [^\n]+)$", text, re.M)
    tps = re.findall(r"Average TPS: ([0-9.]+)", text)
    neigh = re.findall(r"n_neigh_min: (\d+) / n_neigh_max: (\d+) / n_neigh_avg: ([0-9.]+)", text)
    dang = re.findall(r"(\d+) dangerous updates", text)
    out = {"gpu": gpu.group(1).strip() if gpu else "",
           "n_tracebacks": text.count("Traceback"),
           "done_line": "[r5.1]" in text}
    if tps:
        out["tps_prod"] = float(tps[-1])
    if neigh:
        out["n_neigh_min"], out["n_neigh_max"], out["n_neigh_avg"] = (
            int(neigh[-1][0]), int(neigh[-1][1]), float(neigh[-1][2]))
    out["dangerous_updates"] = sum(int(d) for d in dang)
    return out


def energy_window(path, n):
    """Mean PE/N, PE drift between the halves and mean T over the S(q) window."""
    data = np.genfromtxt(path, names=True, delimiter="\t")
    step = data["timestep"]
    w = step >= step[-1] - SQ_WINDOW_STEPS
    pe = data["potential_energy"][w] / n
    t = data["temperature"][w]
    half = len(pe) // 2
    return {
        "pe_per_particle_window": float(pe.mean()),
        "pe_drift_halves": float(pe[half:].mean() - pe[:half].mean()),
        "pe_window_std": float(pe.std()),
        "T_window_mean": float(t.mean()),
        "prod_steps": int(step[-1] - step[0]),
    }


def clusters(gsd_path, r0):
    with gsd.hoomd.open(str(gsd_path), "r") as traj:
        n_frames = len(traj)
        frame = traj[n_frames - 1]
        box = np.asarray(frame.configuration.box[:3], dtype=float)
        pos = np.asarray(frame.particles.position, dtype=float)
    pos = np.mod(pos + box / 2, box)
    pairs = cKDTree(pos, boxsize=box).query_pairs(CLUSTER_CUT * r0, output_type="ndarray")
    n = len(pos)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    sizes = np.sort(np.bincount(labels))[::-1]
    big = sizes[sizes >= CLUSTER_MIN]
    return {
        "n_frames": n_frames,
        "n_clusters_ge50": int(big.size),
        "frac_in_clusters_ge50": float(big.sum() / n),
        "largest_clusters": " ".join(str(int(s)) for s in big[:5]),
    }


def summarize(run_dir):
    run = Path(run_dir)
    pred = run / "predictions"
    rows, problems = [], []
    for i, cond in enumerate(config.CONDITIONS):
        cid = cond["id"]
        p = json.loads((pred / cid / "pred_params.json").read_text())
        sr = p["sim_result"]
        sq = np.load(pred / f"{cid}.npy")
        sq_run = np.load(pred / cid / "S(q)_data" / "average_structure_factor.npy")
        peak = detect_first_peak(_sanitize_curve(sq))
        smax1 = peak.s_max - 1.0
        row = {
            "id": cid, "array_index": i, "part": cond["part"], "range": cond["range"],
            "L_bridge": cond["L_bridge"], "C_anchor": cond["C_anchor"], "C_NaCl": cond["C_NaCl"],
            "U0": p["mapped"]["U0"], "r0": p["mapped"]["r0"], "n": p["mapped"]["n"], "m": p["mapped"]["m"],
            "rmin": float(sr["rmin"]), "rmax": float(sr["rmax"]), "t_tol": float(sr["t_tol"]),
            "n_pairs_below_rmin": int(sr["n_pairs_below_rmin"]),
            "min_pair_distance": float(sr["min_pair_distance"]),
            "git_head": p["git_head"], "seed": p["seed"],
            "s_max_minus_1": smax1, "dispersed": bool(peak.dispersed),
            "near_threshold": abs(smax1 - SHIFT_RMSE_DEFAULTS["dispersed_delta"]) < NEAR,
            "q1": peak.q1, "q1_times_r0": peak.q1 * p["mapped"]["r0"],
            "sq_low_q": float(sq[0, 1]),
            "n_q": int(sq.shape[0]), "q_min": float(sq[0, 0]), "q_max": float(sq[-1, 0]),
        }
        row.update(parse_log(run / f"pred_{i}.out"))
        row.update(energy_window(pred / cid / "potential_energy.csv", config.N))
        gsd_path = next((pred / cid).glob("DNA_assembly_*.gsd"))
        row.update(clusters(gsd_path, row["r0"]))
        rows.append(row)

        # ---- checks
        def bad(msg):
            problems.append(f"{cid}: {msg}")
        if p["coefficients"] != config.COEFFS:
            bad("coefficients differ from predict_config")
        if p["condition"] != cond:
            bad("condition differs from predict_config")
        if p["seed"] != 42 or p["run_kwargs"] != {"N": config.N, "plot": False}:
            bad(f"seed/run_kwargs {p['seed']} {p['run_kwargs']}")
        if row["prod_steps"] != PROD_STEPS - 5000:   # energy log every 5000 steps from production step 0
            bad(f"production steps {row['prod_steps']}")
        if row["n_pairs_below_rmin"] or row["dangerous_updates"] or row["n_tracebacks"] or not row["done_line"]:
            bad("pair below rmin, dangerous update, traceback or no done line")
        if not np.array_equal(sq, sq_run):
            bad("copied S(q) differs from the run's S(q)")
    return rows, problems


def main(run_dir, out_csv=None):
    out_csv = Path(out_csv) if out_csv else HERE / "r51_summary.csv"
    rows, problems = summarize(run_dir)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    heads = sorted({r["git_head"] for r in rows})
    print(f"wrote {out_csv} ({len(rows)} rows); git heads: {heads}; GPUs: {sorted({r['gpu'] for r in rows})}")
    for msg in problems:
        print("CHECK FAILED", msg)
    if not problems:
        print("all checks passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
