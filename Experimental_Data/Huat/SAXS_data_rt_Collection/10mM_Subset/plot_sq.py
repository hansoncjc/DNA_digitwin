#!/usr/bin/env python3
"""
Plot each S(q) curve in 10mM_Subset/S(q)/ as its own log-log figure.

Usage:
    python3 plot_sq.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SQ_DIR = SCRIPT_DIR / "S(q)"
OUT_DIR = SCRIPT_DIR / "S(q)_plots"


def load_sq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected (N, 2+) array, got {data.shape}")
    order = np.argsort(data[:, 0])
    q = data[order, 0]
    sq = data[order, 1]
    return q, sq


def plot_sq_file(path: Path, out_dir: Path) -> Path:
    q, sq = load_sq(path)

    mask = (q > 0) & (sq > 0) & np.isfinite(q) & np.isfinite(sq)
    q = q[mask]
    sq = sq[mask]
    if q.size == 0:
        raise ValueError(f"{path}: no positive finite points for log-log plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(q, sq, color="blue", linewidth=1.5)
    ax.set_xlim(left=q.min(), right=2e-2)
    ax.set_xlabel(r"$q\,(\AA^{-1})$")
    ax.set_ylabel(r"$S(q)$")
    ax.set_title(path.name)
    ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    npy_files = sorted(SQ_DIR.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {SQ_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in npy_files:
        out_path = plot_sq_file(path, OUT_DIR)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
