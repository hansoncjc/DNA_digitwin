#!/usr/bin/env python3
"""
Trim Huat 10 mM subset I(q) and S(q) curves at user-specified q values.

Folder layout (under SAXS_data_rt_Collection/):

    10mM_Subset/
        I(q)/   original experimental I(q) .npy files
        S(q)/   S(q) written by this script (I(q) -> S(q) via repo form factor)

    10mM_Subset_trimmed/
        (sq)/   trimmed S(q) as ``<stem>_sq_trimmed.npy``
        (Iq)/   trimmed I(q) as ``<stem>_trimmed.npy``
        trim_preview_2x3.png   preview figure

Workflow each run:
    1. Read I(q) from ../10mM_Subset/I(q)/
    2. Convert to S(q) with DNA_digitwin/scattering.extract_exp_sq, save to ../10mM_Subset/S(q)/
    3. Trim both I(q) and S(q) at the same q cutoff; save to (Iq)/ and (sq)/
    4. Plot full S(q) with trim cutoff lines

Edit TRIM_Q below (one q value per file), then run:

    python3 trim_subset_by_q.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUBSET_DIR = SCRIPT_DIR.parent / "10mM_Subset"
IQ_DIR = SUBSET_DIR / "I(q)"
SQ_DIR = SUBSET_DIR / "S(q)"
OUT_DIR = SCRIPT_DIR
TRIM_SQ_DIR = OUT_DIR / "Sq"
TRIM_IQ_DIR = OUT_DIR / "Iq"
PLOT_PATH = SCRIPT_DIR / "trim_preview_2x3.png"
CODE_ROOT = Path("/gscratch/zeelab/hanson/codes/DNA_lipid_silica/DNA_digitwin")
MC_DFM_ROOT = Path("/gscratch/zeelab/hanson/codes/DNA_lipid_silica/MC-DFM")
FORMFACTOR_PATH = CODE_ROOT / "formfactors" / "sasmodels_sphere_fit.txt"

# q-window for form-factor normalization when converting I(q) -> S(q)
SQ_NORM_Q_MIN = 0.003
SQ_NORM_Q_MAX = 0.06

# ---------------------------------------------------------------------------
# Edit these q cutoffs (Å^-1). Rows with q <= q_trim are kept (inclusive).
# ---------------------------------------------------------------------------
TRIM_Q: dict[str, float] = {
    "chol_020_nacl_10_bridge_20.npy": 0.018,
    "chol_060_nacl_10_bridge_20.npy": 0.019,
    "chol_100_nacl_10_bridge_20.npy": 0.0185,
    "chol_100_nacl_10_bridge_40.npy": 0.017,
    "chol_100_nacl_10_bridge_80.npy": 0.0145,
    "chol_140_nacl_10_bridge_20.npy": 0.019,
}


def trimmed_sq_name(src_name: str) -> str:
    stem = Path(src_name).stem
    return f"{stem}_sq_trimmed.npy"


def trimmed_iq_name(src_name: str) -> str:
    stem = Path(src_name).stem
    return f"{stem}_trimmed.npy"


def load_curve(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: expected (N, 2+) array, got {arr.shape}")
    order = np.argsort(arr[:, 0])
    return arr[order]


def trim_at_q(arr: np.ndarray, q_trim: float) -> np.ndarray:
    q = arr[:, 0]
    if q_trim < q.min():
        raise ValueError(f"q_trim={q_trim} is below minimum q={q.min()}")
    if q_trim > q.max():
        raise ValueError(f"q_trim={q_trim} is above maximum q={q.max()}")
    kept = arr[q <= q_trim]
    if kept.size == 0:
        raise ValueError(f"No rows kept at q_trim={q_trim}")
    return kept


def iq_to_sq(iq_arr: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(CODE_ROOT))
    sys.path.insert(0, str(MC_DFM_ROOT))
    from scattering import extract_exp_sq  # noqa: E402

    iq = iq_arr[:, :2]
    return extract_exp_sq(
        iq,
        ffpath=str(FORMFACTOR_PATH),
        q_min=SQ_NORM_Q_MIN,
        q_max=SQ_NORM_Q_MAX,
        normalize=False,
    )


def build_sq_from_iq(src_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load I(q), convert to S(q), save full S(q) under 10mM_Subset/S(q)/."""
    iq_path = IQ_DIR / src_name
    if not iq_path.is_file():
        raise FileNotFoundError(f"Missing I(q) file: {iq_path}")

    iq_arr = load_curve(iq_path)
    sq_arr = iq_to_sq(iq_arr)

    SQ_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SQ_DIR / src_name, sq_arr)
    return iq_arr, sq_arr


def plot_trim_preview_panels(
    panels: list[tuple[str, np.ndarray, float]],
    out_png: Path,
) -> None:
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.4, n_rows * 3.5))
    axes = axes.ravel()

    for ax, (title, sq, q_trim) in zip(axes, panels):
        q = sq[:, 0]
        s = sq[:, 1]
        finite = np.isfinite(q) & np.isfinite(s) & (q > 0) & (s > 0)
        if not np.any(finite):
            ax.text(0.5, 0.5, "No positive finite data", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(q[finite], s[finite], color="black", linewidth=1.5, label="S(q)")
            ax.axvline(
                q_trim,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"q trim = {q_trim:.5g}",
            )
            ax.legend(loc="best", frameon=False, fontsize=8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=8)

    for idx, ax in enumerate(axes):
        row, col = divmod(idx, n_cols)
        if col == 0:
            ax.set_ylabel(r"$S(q)$", fontsize=9)
        if row == n_rows - 1:
            ax.set_xlabel(r"$q$ ($\AA^{-1}$)", fontsize=9)

    fig.suptitle("Huat 10 mM subset — trim preview (S(q))", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not IQ_DIR.is_dir():
        raise FileNotFoundError(f"I(q) directory not found: {IQ_DIR}")
    if not FORMFACTOR_PATH.is_file():
        raise FileNotFoundError(f"Form factor file not found: {FORMFACTOR_PATH}")

    missing = [name for name in TRIM_Q if not (IQ_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing I(q) files in {IQ_DIR}: {missing}")

    extra = sorted(p.name for p in IQ_DIR.glob("*.npy") if p.name not in TRIM_Q)
    if extra:
        print(f"Warning: I(q) files without TRIM_Q entry (skipped): {extra}")

    print(f"I(q) in:       {IQ_DIR}")
    print(f"S(q) out:      {SQ_DIR}")
    print(f"Trimmed S(q):  {TRIM_SQ_DIR}")
    print(f"Trimmed I(q):  {TRIM_IQ_DIR}")
    print(f"Plot:          {PLOT_PATH}\n")

    TRIM_SQ_DIR.mkdir(parents=True, exist_ok=True)
    TRIM_IQ_DIR.mkdir(parents=True, exist_ok=True)

    plot_panels: list[tuple[str, np.ndarray, float]] = []

    for src_name, q_trim in TRIM_Q.items():
        iq_arr, sq = build_sq_from_iq(src_name)
        trimmed_iq = trim_at_q(iq_arr, q_trim)
        trimmed_sq = trim_at_q(sq, q_trim)

        sq_out_path = TRIM_SQ_DIR / trimmed_sq_name(src_name)
        iq_out_path = TRIM_IQ_DIR / trimmed_iq_name(src_name)
        np.save(sq_out_path, trimmed_sq)
        np.save(iq_out_path, trimmed_iq)

        plot_panels.append((Path(src_name).stem, sq, q_trim))

        q_kept = trimmed_sq[:, 0]
        print(
            f"{src_name}\n"
            f"  full S(q):    {SQ_DIR / src_name}\n"
            f"  q_trim = {q_trim:.6g}\n"
            f"  S(q) rows:    {sq.shape[0]} -> {trimmed_sq.shape[0]}\n"
            f"  I(q) rows:    {iq_arr.shape[0]} -> {trimmed_iq.shape[0]}\n"
            f"  q kept:       [{q_kept.min():.6g}, {q_kept.max():.6g}]\n"
            f"  trimmed S(q): {sq_out_path.relative_to(OUT_DIR)}\n"
            f"  trimmed I(q): {iq_out_path.relative_to(OUT_DIR)}\n"
        )

    plot_trim_preview_panels(plot_panels, PLOT_PATH)
    print(f"plot:   {PLOT_PATH}\n")
    print("Done.")


if __name__ == "__main__":
    main()
