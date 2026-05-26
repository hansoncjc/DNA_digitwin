"""
Analysis-only subprocess for parallel BO workers.

Consumes a worker config plus SIM_DONE.json, converts the exact GSD trajectory
produced by the simulation stage, computes the dataset loss, and writes
ANALYSIS_DONE.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np


_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
for p in (_PARENT, _THIS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scattering import convert_to_SAXS, convert_to_SAXS_fft, extract_exp_sq  # noqa: E402
from metrics import compare_to_exp, compare_to_exp_saxsfft  # noqa: E402


_SQ_CANDIDATES = (
    "S(q)_data/average_structure_factor.npy",
    "scattering_data/average_structure_factor.npy",
    "S(q)_/average_structure_factor.npy",
    "S(q)/average_structure_factor.npy",
)


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _locate_sim_sq(save_dir: Path) -> Path:
    for rel in _SQ_CANDIDATES:
        p = save_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing simulated S(q); tried {[str(save_dir / r) for r in _SQ_CANDIDATES]}"
    )


def _load_exp_curve(exp_path: str, trim_tail: int) -> np.ndarray:
    if trim_tail < 0:
        raise ValueError("trim_tail must be >= 0")
    arr = np.load(exp_path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected (N, 2+) array in {exp_path}, got {arr.shape}")
    arr = arr[:, :2].astype(np.float64, copy=False)
    if trim_tail and arr.shape[0] > trim_tail:
        arr = arr[:-trim_tail]
    return arr


def _run_analysis(cfg: dict, sim_done: dict, save_dir: Path) -> dict:
    sim_result = sim_done["result"]
    gsd_path = sim_result.get("gsd_path")
    if not gsd_path:
        raise ValueError("SIM_DONE.json result is missing 'gsd_path'")
    if not Path(gsd_path).exists():
        raise FileNotFoundError(f"Simulation GSD does not exist: {gsd_path}")

    scat = cfg.get("scattering", {}) or {}
    method = scat.get("method", "saxsfft")
    sc_kw = dict(scat.get("kwargs", {}) or {})
    if method == "saxsfft":
        convert_to_SAXS_fft(str(save_dir), path=gsd_path, **sc_kw)
    else:
        convert_to_SAXS(str(save_dir), path=gsd_path, **sc_kw)

    loss_cfg = cfg["loss"]
    sim_sq_path = _locate_sim_sq(save_dir)
    sim_sq = np.load(sim_sq_path)

    exp_data = _load_exp_curve(
        loss_cfg["exp_path"],
        int(loss_cfg.get("trim_tail", 0)),
    )
    datatype = loss_cfg.get("datatype", "sq")
    if datatype == "sq":
        exp_sq = exp_data
    else:
        exp_sq = extract_exp_sq(
            exp_scattering=exp_data,
            ffpath=loss_cfg["ffpath"],
            q_min=float(loss_cfg.get("q_min", 0.02)),
            q_max=float(loss_cfg.get("q_max", 0.03)),
            normalize=False,
        )

    compare_method = loss_cfg.get("scattering_method", method)
    metric = loss_cfg.get("metric", "mse")
    compare_q_range = loss_cfg.get("compare_q_range", (0.003, 0.06))
    if compare_method == "saxsfft":
        loss = float(compare_to_exp_saxsfft(
            exp_sq,
            sim_sq,
            str(save_dir),
            metric=metric,
            q_range=compare_q_range,
        ))
    else:
        loss = float(compare_to_exp(exp_sq, sim_sq, str(save_dir), metric=metric))

    return {
        "loss": loss,
        "sim_sq_path": str(sim_sq_path),
        "gsd_path": str(gsd_path),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <config.json> <SIM_DONE.json>", file=sys.stderr)
        return 2

    cfg = json.loads(Path(argv[1]).read_text())
    sim_done = json.loads(Path(argv[2]).read_text())
    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        result = _run_analysis(cfg, sim_done, outdir)
        payload = {
            "run_time_seconds": time.time() - t0,
            "host": os.uname().nodename,
            **result,
        }
        _write_json_atomic(outdir / "ANALYSIS_DONE.json", payload)
        print(f"[analysis_stage] SUCCESS loss={payload['loss']:.6g}", flush=True)
        return 0
    except Exception:
        payload = {
            "run_time_seconds": time.time() - t0,
            "host": os.uname().nodename,
            "traceback": traceback.format_exc(),
        }
        _write_json_atomic(outdir / "ANALYSIS_FAILED.json", payload)
        print(payload["traceback"], file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
