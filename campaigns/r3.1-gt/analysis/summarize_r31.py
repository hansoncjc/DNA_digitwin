"""
R3.1 ground truth (Sweep8_R3.1_GT): collect per-condition facts into one CSV.

Sources (read only):
  ground_truth/<id>/gt_params.json   condition, mapped (U0, r0, n, m), table bounds, pair check
  ground_truth/qc.csv                first-peak QC (s_max - 1, dispersed, q1)
  gt_<i>.out                         GPU name, production average TPS, neighbor-list stats
  ground_truth/<id>/potential_energy.csv   energy / temperature in the S(q) window
  sacct (pasted by the user, 2026-10-09)   elapsed time per array task (job 41681503)

Usage: python summarize_r31.py RUN_DIR OUT_CSV
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

# sacct -j 41681503 -X, Elapsed (user paste 2026-10-09)
ELAPSED = {
    0: "00:27:47", 1: "00:28:27", 2: "00:33:31", 3: "00:34:35", 4: "00:28:36",
    5: "00:34:23", 6: "00:34:04", 7: "00:36:58", 8: "00:30:29", 9: "00:34:19",
    10: "00:36:35", 11: "00:36:10", 12: "00:28:21", 13: "00:29:05", 14: "00:34:37",
    15: "00:38:44", 16: "00:26:41", 17: "00:34:14", 18: "00:37:33", 19: "00:43:14",
    20: "00:34:21", 21: "00:36:38", 22: "00:40:29", 23: "00:43:20", 24: "00:34:01",
    25: "00:35:10", 26: "00:37:15", 27: "00:40:36", 28: "00:34:08",
}
ORDER = [f"d{i}" for i in range(24)] + [f"h{i}" for i in range(5)]

# S(q) uses the last 100 GSD frames (one per 50 tau); frame 0 is production step 0.
SQ_WINDOW_STEPS = 100 * 50_000


def minutes(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 60 + m + s / 60


def parse_log(path):
    text = path.read_text(errors="replace")
    gpu = re.search(r"^(NVIDIA [^\n]+)$", text, re.M)
    tps = re.findall(r"Average TPS: ([0-9.]+)", text)
    neigh = re.findall(r"n_neigh_min: (\d+) / n_neigh_max: (\d+) / n_neigh_avg: ([0-9.]+)", text)
    dang = re.findall(r"(\d+) dangerous updates", text)
    out = {"gpu": gpu.group(1).strip() if gpu else ""}
    if tps:
        out["tps_prod"] = float(tps[-1])          # last block = production
    if neigh:
        out["n_neigh_min"], out["n_neigh_max"], out["n_neigh_avg"] = (
            int(neigh[-1][0]), int(neigh[-1][1]), float(neigh[-1][2]))
    out["dangerous_updates"] = sum(int(d) for d in dang)
    return out


def energy_window(path, n):
    """Mean PE/N, relative PE drift and mean T over the S(q) window."""
    data = np.genfromtxt(path, names=True, delimiter="\t")
    step = data["timestep"]
    prod0 = step[0]
    w = step >= step[-1] - SQ_WINDOW_STEPS
    pe = data["potential_energy"][w] / n
    t = data["temperature"][w]
    half = len(pe) // 2
    return {
        "pe_per_particle_window": float(pe.mean()),
        "pe_drift_halves": float(pe[half:].mean() - pe[:half].mean()),
        "pe_window_std": float(pe.std()),
        "T_window_mean": float(t.mean()),
        "prod_steps": int(step[-1] - prod0),
    }


def main(run_dir, out_csv):
    run = Path(run_dir)
    gt = run / "ground_truth"
    qc = {r["id"]: r for r in csv.DictReader(open(gt / "qc.csv"))}
    rows = []
    for i, cid in enumerate(ORDER):
        p = json.loads((gt / cid / "gt_params.json").read_text())
        c, mp, sr = p["condition"], p["mapped"], p["sim_result"]
        row = {
            "id": cid, "array_index": i, "role": c["role"], "note": c.get("note", ""),
            # gt_params.json was written at 49aac10 with the key "C_chol"
            "L_bridge": c["L_bridge"], "C_anchor": c["C_chol"], "C_NaCl": c["C_NaCl"],
            "U0": mp["U0"], "r0": mp["r0"], "n": mp["n"], "m": mp["m"],
            "rmin": sr["rmin"], "rmax": sr["rmax"], "t_tol": sr["t_tol"],
            "n_pairs_below_rmin": sr["n_pairs_below_rmin"],
            "min_pair_distance": sr["min_pair_distance"],
            "git_head": p["git_head"], "seed": p["run_kwargs"]["seed"],
            "s_max_minus_1": float(qc[cid]["s_max_minus_1"]),
            "dispersed": qc[cid]["dispersed"] == "True",
            "q1": float(qc[cid]["q1"]) if qc[cid]["q1"] != "nan" else float("nan"),
            "qc_status": qc[cid]["status"],
            "elapsed_min": round(minutes(ELAPSED[i]), 2),
        }
        row["q1_times_r0"] = row["q1"] * row["r0"]
        row.update(parse_log(run / f"gt_{i}.out"))
        row.update(energy_window(gt / cid / "potential_energy.csv", 5000))
        rows.append(row)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
