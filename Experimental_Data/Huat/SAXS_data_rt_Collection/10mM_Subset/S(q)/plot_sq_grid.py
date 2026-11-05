#!/usr/bin/env python3
"""Plot all S(q) curves in 10mM_Subset/S(q)/ as a 2x3 log-log grid."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "sq_grid.png"


def format_title(stem: str) -> str:
    title = stem.replace("chol", "Anchor")
    title = title.replace("_nacl_10", "")
    title = title.replace("bridge", "Bridge")
    return title


def load_sq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path}: expected (N, 2+) array, got {data.shape}")
    order = np.argsort(data[:, 0])
    q = data[order, 0]
    sq = data[order, 1]
    return q, sq


def plot_grid() -> None:
    npy_files = sorted(SCRIPT_DIR.glob("*.npy"))
    if len(npy_files) != 6:
        raise ValueError(f"Expected 6 .npy files in {SCRIPT_DIR}, found {len(npy_files)}")

    n_rows = 2
    n_cols = 3
    figsize = (n_cols * 5.5, n_rows * 4.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    loaded_count = 0
    missing_count = 0

    for idx, (ax, path) in enumerate(zip(axes.ravel(), npy_files)):
        title = format_title(path.stem)

        try:
            q, sq = load_sq(path)
            finite = np.isfinite(q) & np.isfinite(sq) & (q > 0) & (sq > 0)
            if not np.any(finite):
                ax.text(
                    0.5,
                    0.5,
                    "No Positive\nFinite Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
                missing_count += 1
            else:
                ax.plot(q[finite], sq[finite], linewidth=1.5, color="black")
                loaded_count += 1
        except (OSError, ValueError) as exc:
            ax.text(
                0.5,
                0.5,
                str(exc),
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
                color="red",
            )
            missing_count += 1

        ax.set_title(title, fontsize=18)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1.0e-1, 5e1)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_major_locator(plt.LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10, subs="auto", numticks=10))
        ax.tick_params(labelsize=8)

    fig.suptitle("Experiment Data", fontsize=18, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.993])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PNG}")
    print(f"Data loaded: {loaded_count}")
    print(f"Data missing: {missing_count}")


if __name__ == "__main__":
    plot_grid()
