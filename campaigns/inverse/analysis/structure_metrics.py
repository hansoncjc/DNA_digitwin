"""
Structure metrics of simulated configurations against an inverse-design
target: S(q) loss, q1 ratio, PTM fractions and g(r). Runs locally (OVITO and
saxs-fft; MC-DFM is not needed and is stubbed). The same computation for
every target: nothing branches on FCC or fluid.

    python structure_metrics.py --target-dir TARGET_DIR --run RUN_DIR [--top 3] --out OUT
    python structure_metrics.py --target-dir TARGET_DIR --gsd A.gsd B.gsd [--labels a b] --out OUT

TARGET_DIR is one folder of campaigns/inverse/inverse-targets (meta.json,
the configuration GSD, cpu/target_sq.npy). RUN_DIR is an unpacked
pack_run.py tar (<name>/pack_summary, <name>/Optimization_Results): the GSDs
are those of the top evaluations in pack_summary/summary.json, and the
convergence plot is drawn from pack_summary/evals.csv.

For each GSD, on the frames of the forward model (convert_to_SAXS_fft
defaults: the last 100 frames, every 5th, i.e. 20 frames); for the target,
on the frames named in meta.json (s_q_common.frames, step):

1. S(q) loss. S(q) recomputed with scattering.convert_to_SAXS_fft defaults
   (CPU, as in R6) and scored with metrics.shift_rmse_loss (defaults)
   against the target S(q) (--target-sq, default cpu/target_sq.npy): total
   loss, M4, spacing term. When the GSD sits in an eval folder, the run-time
   S(q) (S(q)_data/average_structure_factor.npy) and loss
   (shift_rmse_diagnostics.json) are read for the check: the run-time S(q)
   rescored here must give the run-time loss, and the recomputed loss may
   differ from it by the CPU-model effect (C26, ~1e-3); a difference above
   --loss-tol is reported as CHECK FAILED and the exit code is 2.
2. q1 ratio q1_sim / q1_tgt and both q1 (metrics.detect_first_peak, as in
   the loss; None for a dispersed curve).
3. PTM: OVITO PolyhedralTemplateMatchingModifier with the FCC, HCP, BCC, ICO
   and SC templates (the set of the 2026-08 and 2026-09 inverse notes; the
   diamond and graphene templates mean nothing for an isotropic pair
   potential). Run once with rmsd_cutoff = 0 and the per-particle RMSD
   output; a particle counts as type T at cutoff c when its best template is
   T and its RMSD <= c, which is what OVITO's own rmsd_cutoff = c gives.
   Fractions are over all N particles, averaged over the frames (std over
   frames as *_std); --ptm-cutoffs, headline 0.10 (OVITO default, as before).
   fcc_hcp = f_FCC + f_HCP; fcc_share = f_FCC / (f_FCC + f_HCP).
   ptm_rmsd_median: median RMSD of every particle's best template (no
   cutoff), mean over frames; it still moves when every fraction is 0.
4. g(r) on 0 < r <= --r-max with bin --dr (sigma), minimum image, averaged
   over the frames. Normalized by the box number density on both sides:

       g(r) = <n_pairs(r, r + dr)> / (N/2 * rho_box * 4/3 pi ((r + dr)^3 - r^3)),
       rho_box = N / V_box,

   so a cluster or droplet in a padded box has g >> 1 at contact and g -> its
   share of the box at large r. Scalars: first peak position (first
   find_peaks maximum with prominence 0.3 * (max - min), parabola refined)
   and g(r) at it; RMSD of g_sim - g_tgt over --rdf-range, and that RMSD over
   the RMS of g_tgt on the same range.

Outputs in OUT: metrics.csv and metrics.json (one row per configuration,
target first), gr.npz (r and every g(r)), <label>/sq_recomputed.npy, and the
figures sq_<label>.png (x = q/q1 as compared by the loss, and absolute q),
gr.png, ptm.png, convergence.png (with --run), snapshot_<label>.png (OVITO
Tachyon render of the last analysed frame, particles colored by PTM type at
the headline cutoff; the cell is not drawn).
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PTM_TYPES = ("FCC", "HCP", "BCC", "ICO", "SC")
FORWARD_FRAMES = ("last:100", 5)      # convert_to_SAXS_fft defaults
# Categorical slots of the reference palette (dataviz skill); target in ink.
INK = "#0b0b0b"
MUTED = "#8a8985"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
PTM_COLORS = dict(zip(PTM_TYPES, SERIES), OTHER="#c9c8c3")
# S(q) overlay: target violet, simulation green (reference palette), alpha 0.8.
SQ_TARGET, SQ_SIM, SQ_ALPHA = "#4a3aa7", "#008300", 0.8


def _stub_mcdfm():
    """scattering imports MC-DFM's Scattering_Simulator at module level; it is
    not used by convert_to_SAXS_fft, so an empty stub stands in when absent."""
    try:
        import Scattering_Simulator  # noqa: F401
    except ImportError:
        root = types.ModuleType("Scattering_Simulator")
        root.__path__ = []
        sub = types.ModuleType("Scattering_Simulator.pairwise_method")
        root.pairwise_method = sub
        sys.modules["Scattering_Simulator"] = root
        sys.modules["Scattering_Simulator.pairwise_method"] = sub


def select_frames(n_frames, frames, step=1):
    """Frame indices as saxs-fft's read_configuration picks them: 'all' (step
    ignored), or 'last:N' then every step-th frame from the oldest one."""
    if frames == "all":
        return list(range(n_frames))
    if isinstance(frames, str) and frames.startswith("last:"):
        n = int(frames.split(":", 1)[1])
        return list(range(max(0, n_frames - n), n_frames))[::int(step)]
    raise ValueError(f"unsupported frames specifier {frames!r}")


def read_frames(gsd_path, indices):
    """``[(positions (N, 3) float64, box lengths (3,))]`` of the given frames."""
    import gsd.hoomd
    out = []
    with gsd.hoomd.open(str(gsd_path), "r") as traj:
        for i in indices:
            f = traj[i]
            out.append((np.asarray(f.particles.position, dtype=np.float64),
                        np.asarray(f.configuration.box[:3], dtype=np.float64)))
    return out


def gsd_n_frames(gsd_path):
    import gsd.hoomd
    with gsd.hoomd.open(str(gsd_path), "r") as traj:
        return len(traj)


# ------------------------------------------------------------------ S(q)


def recompute_sq(gsd_path, workdir):
    """convert_to_SAXS_fft with its defaults on a symlink in ``workdir``, so
    that saxs-fft's side files (position text dump, average S(q)) are written
    there and not next to the GSD."""
    import scattering
    workdir.mkdir(parents=True, exist_ok=True)
    link = workdir / Path(gsd_path).name
    if not link.exists():
        os.symlink(Path(gsd_path).resolve(), link)
    scattering.convert_to_SAXS_fft(str(workdir), path=str(link))
    return np.load(workdir / "S(q)_data" / "average_structure_factor.npy")


def sq_metrics(target_sq, sim_sq):
    import metrics
    loss, diag, grid, log_exp, log_sim = metrics.shift_rmse_loss(target_sq, sim_sq, None)
    return diag, (grid, log_exp, log_sim)


# ------------------------------------------------------------------ PTM


def ptm_fractions(gsd_path, indices, cutoffs, keep_frame=None):
    """
    PTM fractions per cutoff, mean and std over ``indices``. With
    ``keep_frame``, also returns that frame's per-particle type at the first
    cutoff (for the snapshot color).
    """
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    from ovito.io import import_file
    from ovito.modifiers import PolyhedralTemplateMatchingModifier as PTM

    pipeline = import_file(str(gsd_path))
    mod = PTM(rmsd_cutoff=0.0, output_rmsd=True)
    for name in dir(PTM.Type):
        if name.isupper() and name != "OTHER":
            mod.structures[getattr(PTM.Type, name)].enabled = name in PTM_TYPES
    pipeline.modifiers.append(mod)
    codes = {name: int(getattr(PTM.Type, name)) for name in (*PTM_TYPES, "OTHER")}

    per_frame = {c: {t: [] for t in (*PTM_TYPES, "OTHER")} for c in cutoffs}
    rmsd_median = []
    kept = None
    for i in indices:
        data = pipeline.compute(i)
        st = np.asarray(data.particles["Structure Type"])
        rmsd = np.asarray(data.particles["RMSD"])
        n = st.size
        matched = st != codes["OTHER"]
        rmsd_median.append(float(np.median(rmsd[matched])) if matched.any() else np.nan)
        for c in cutoffs:
            assigned = np.zeros(n, dtype=bool)
            for t in PTM_TYPES:
                mask = (st == codes[t]) & (rmsd <= c)
                assigned |= mask
                per_frame[c][t].append(mask.sum() / n)
            per_frame[c]["OTHER"].append((~assigned).sum() / n)
        if keep_frame is not None and i == keep_frame:
            c0 = cutoffs[0]
            kept = np.where(rmsd <= c0, st, codes["OTHER"])
    out = {}
    for c in cutoffs:
        row = {}
        for t, vals in per_frame[c].items():
            row[t] = float(np.mean(vals))
            row[f"{t}_std"] = float(np.std(vals))
        fh = np.array(per_frame[c]["FCC"]) + np.array(per_frame[c]["HCP"])
        row["fcc_hcp"] = float(fh.mean())
        row["fcc_hcp_std"] = float(fh.std())
        row["fcc_share"] = (row["FCC"] / (row["FCC"] + row["HCP"])
                            if row["FCC"] + row["HCP"] > 0 else None)
        out[c] = row
    out["all"] = {"rmsd_median": float(np.nanmean(rmsd_median)),
                  "rmsd_median_std": float(np.nanstd(rmsd_median))}
    return out, kept, codes


# ------------------------------------------------------------------ g(r)


def rdf(frames, r_max, dr):
    """Box-density-normalized g(r) (module docstring), averaged over frames."""
    from scipy.spatial import cKDTree
    edges = np.arange(0.0, r_max + 0.5 * dr, dr)
    hist = np.zeros(edges.size - 1)
    norm = 0.0
    for x, box in frames:
        n = x.shape[0]
        wrapped = np.mod(x, box)
        wrapped[wrapped >= box] = 0.0
        tree = cKDTree(wrapped, boxsize=box)
        pairs = tree.query_pairs(r_max, output_type="ndarray")
        d = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
        d -= box * np.round(d / box)
        h, _ = np.histogram(np.linalg.norm(d, axis=1), bins=edges)
        hist += h
        rho = n / float(np.prod(box))
        norm += 0.5 * n * rho
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (norm * shell)
    return 0.5 * (edges[1:] + edges[:-1]), g


def first_peak(r, g, prominence_frac=0.3):
    from scipy.signal import find_peaks
    if not np.any(g > 0):
        return None, None
    prom = prominence_frac * (g.max() - g.min())
    idx, _ = find_peaks(g, prominence=prom)
    i = int(idx[0]) if idx.size else int(np.argmax(g))
    if 0 < i < g.size - 1:
        y0, y1, y2 = g[i - 1], g[i], g[i + 1]
        den = y0 - 2 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / den, -1, 1)) if abs(den) > 1e-15 else 0.0
        return float(r[i] + delta * (r[1] - r[0])), float(y1)
    return float(r[i]), float(g[i])


# ------------------------------------------------------------------ inputs


def configurations_from_run(run_dir, top):
    """``[(label, gsd, info)]`` of the top evaluations of an unpacked pack."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "pack_summary" / "summary.json").read_text())
    out = []
    for rec in summary["top"][:top]:
        ds = run_dir / "Optimization_Results" / rec["dataset_dir"]
        gsds = sorted(ds.glob("DNA_assembly_*.gsd"))
        if not gsds:
            raise FileNotFoundError(f"no GSD in {ds}")
        out.append((f"eval_{rec['eval_id']:03d}", gsds[-1], rec))
    return out, summary


def runtime_files(gsd_path):
    ds = Path(gsd_path).parent
    sq = ds / "S(q)_data" / "average_structure_factor.npy"
    diag = ds / "shift_rmse_diagnostics.json"
    return (np.load(sq) if sq.is_file() else None,
            json.loads(diag.read_text()) if diag.is_file() else None)


# ------------------------------------------------------------------ figures


# scattering.convert_to_SAXS_fft sets the global font size to 18; every
# figure here is drawn at its own size.
PLOT_RC = {"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
           "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}


def _with_rc(fn):
    def wrapped(*args, **kwargs):
        import matplotlib.pyplot as plt
        with plt.rc_context(PLOT_RC):
            return fn(*args, **kwargs)
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors="#52514e")
    ax.grid(True, color="#ecebe8", linewidth=0.6)
    ax.set_axisbelow(True)


@_with_rc
def plot_sq(path, label, target_sq, sim_sq, diag, reduced):
    import matplotlib.pyplot as plt
    grid, log_exp, log_sim = reduced
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))  # two 8:4.8 panels
    ax = axes[0]
    ax.plot(grid, 10 ** log_exp, color=SQ_TARGET, lw=2, alpha=SQ_ALPHA, label="target")
    ax.plot(grid, 10 ** log_sim, color=SQ_SIM, lw=2, alpha=SQ_ALPHA, label=f"{label} (scaled)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$x = q/q_1$ (each curve its own $q_1$)" if diag["aligned"]
                  else r"$x = q/q_{1,\mathrm{tgt}}$ (not aligned)")
    ax.set_ylabel("S(q)")
    ax.set_title(f"as compared by the loss: L={diag['loss']:.4f}, "
                 f"M4={diag['m4']:.4f}, shift={diag['shift_term']:.4f}", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    ax.plot(target_sq[:, 0], target_sq[:, 1], color=SQ_TARGET, lw=2, alpha=SQ_ALPHA, label="target")
    ax.plot(sim_sq[:, 0], sim_sq[:, 1], color=SQ_SIM, lw=2, alpha=SQ_ALPHA, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$q$ ($\AA^{-1}$)")
    q1r = diag.get("q1_ratio")
    ax.set_title("absolute q, unscaled" + (f"; q1 ratio {q1r:.4f}" if q1r else ""), fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    for a in axes:
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@_with_rc
def plot_gr(path, r, curves, rdf_range):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, xlim in zip(axes, ((0, r[-1]), (0, min(4.0, r[-1])))):
        for i, (label, g) in enumerate(curves):
            color = INK if i == 0 else SERIES[(i - 1) % len(SERIES)]
            ax.plot(r, g, color=color, lw=2 if i == 0 else 1.6, label=label)
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$r$ ($\sigma$)")
        ax.set_ylabel(r"$g(r)$, normalized by $N/V_\mathrm{box}$")
        _style(ax)
    axes[0].axvspan(*rdf_range, color="#f4f3f0", zorder=-1)
    axes[0].set_title(f"0-{r[-1]:.0f} sigma (shaded: RMSD range)", fontsize=10)
    axes[1].set_title("zoom 0-4 sigma", fontsize=10)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@_with_rc
def plot_ptm(path, rows, cutoff):
    import matplotlib.pyplot as plt
    labels = [r["label"] for r in rows]
    cats = (*PTM_TYPES, "OTHER")
    fig, ax = plt.subplots(figsize=(2.6 + 1.1 * len(rows), 4.0))
    bottom = np.zeros(len(rows))
    for t in cats:
        vals = np.array([r[f"ptm{cutoff:g}_{t}"] for r in rows])
        ax.bar(labels, vals, bottom=bottom, color=PTM_COLORS[t], width=0.6, label=t,
               edgecolor="white", linewidth=1.0)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of all particles")
    ax.set_title(f"PTM, RMSD <= {cutoff:g}; mean over the analysed frames")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


@_with_rc
def plot_convergence(path, evals_csv):
    import matplotlib.pyplot as plt
    with open(evals_csv, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["loss"]]
    eid = np.array([int(r["eval_id"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    batch = np.array([int(r["batch"]) for r in rows])
    disp = np.array([r["dispersed_sim"] == "True" for r in rows])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for b in np.unique(batch)[1:]:
        ax.axvline(eid[batch == b].min() - 0.5, color="#ecebe8", lw=1)
    ax.scatter(eid[~disp], loss[~disp], s=22, color=SERIES[0], label="evaluation")
    ax.scatter(eid[disp], loss[disp], s=26, facecolor="white", edgecolor=SERIES[1],
               linewidth=1.4, label="dispersed simulated S(q)")
    order = np.argsort(eid)
    ax.step(eid[order], np.minimum.accumulate(loss[order]), where="post", color=INK, lw=2,
            label="best so far")
    ax.set_xlabel("evaluation id (vertical lines: batches)")
    ax.set_ylabel("loss = M4 + |ln(q1_sim/q1_tgt)|")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_snapshot(gsd_path, frame, types_, codes, path, size=(900, 900)):
    """OVITO Tachyon render of one frame, colored by PTM type; no cell."""
    warnings.filterwarnings("ignore", message=".*OVITO.*PyPI")
    from ovito.io import import_file
    from ovito.vis import TachyonRenderer, Viewport
    from ovito.data import DataCollection  # noqa: F401
    from ovito.pipeline import ModifierInterface  # noqa: F401

    pipeline = import_file(str(gsd_path))
    color_of = {codes[t]: PTM_COLORS[t] for t in codes}

    def color(frame_, data):
        rgb = np.array([[int(color_of[c][k:k + 2], 16) / 255 for k in (1, 3, 5)]
                        for c in types_])
        data.particles_.create_property("Color", data=rgb)
        data.particles_.vis.radius = 0.5
        data.cell_.vis.enabled = False

    pipeline.modifiers.append(color)
    pipeline.add_to_scene()
    vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(-1, -0.6, -0.5))
    pipeline.compute(frame)
    vp.zoom_all(size=size)
    vp.render_image(size=size, filename=str(path), frame=frame,
                    background=(1, 1, 1), renderer=TachyonRenderer(ambient_occlusion=True))
    pipeline.remove_from_scene()


# ------------------------------------------------------------------ main


def analyse(label, gsd, indices, cutoffs, r_max, dr, render_dir=None):
    ptm, kept, codes = ptm_fractions(gsd, indices, cutoffs,
                                     keep_frame=indices[-1] if render_dir else None)
    r, g = rdf(read_frames(gsd, indices), r_max, dr)
    r1, g1 = first_peak(r, g)
    row = {"label": label, "gsd": str(gsd), "frames": f"{indices[0]}..{indices[-1]} "
           f"({len(indices)} frames)", "gr_r1": r1, "gr_g1": g1}
    for c, vals in ptm.items():
        for k, v in vals.items():
            row[f"ptm_{k}" if c == "all" else f"ptm{c:g}_{k}"] = v
    if render_dir is not None:
        render_snapshot(gsd, indices[-1], kept, codes, render_dir / f"snapshot_{label}.png")
    return row, r, g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--target-sq", default=None, help="default <target-dir>/cpu/target_sq.npy")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="unpacked pack_run.py folder (pack_summary, Optimization_Results)")
    src.add_argument("--gsd", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptm-cutoffs", type=float, nargs="+", default=[0.10, 0.05, 0.15],
                    help="first one is the headline (figure, snapshot)")
    ap.add_argument("--r-max", type=float, default=10.0)
    ap.add_argument("--dr", type=float, default=0.02)
    ap.add_argument("--rdf-range", type=float, nargs=2, default=[0.0, 10.0])
    ap.add_argument("--loss-tol", type=float, default=5e-3)
    ap.add_argument("--reuse-sq", action="store_true",
                    help="use <out>/<label>/sq_recomputed.npy when present")
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    _stub_mcdfm()

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tdir = Path(a.target_dir)
    meta = json.loads((tdir / "meta.json").read_text())
    tgt_gsd = tdir / meta["configuration"]["gsd"]
    target_sq = np.load(a.target_sq or tdir / "cpu" / "target_sq.npy")
    summary = None
    if a.run:
        configs, summary = configurations_from_run(Path(a.run).expanduser(), a.top)
    else:
        labels = a.labels or [Path(g).stem for g in a.gsd]
        configs = [(lab, Path(g), {}) for lab, g in zip(labels, a.gsd)]
    cutoffs = list(dict.fromkeys(a.ptm_cutoffs))
    render_dir = None if a.no_render else out

    tgt_frames = select_frames(gsd_n_frames(tgt_gsd), meta["s_q_common"]["frames"],
                               meta["s_q_common"].get("step", 1))
    import metrics
    tgt_peak = metrics.detect_first_peak(metrics._sanitize_curve(target_sq))
    row, r, g_tgt = analyse("target", tgt_gsd, tgt_frames, cutoffs, a.r_max, a.dr, render_dir)
    row.update({"q1": None if tgt_peak.dispersed else tgt_peak.q1, "loss": 0.0})
    rows, curves = [row], [("target", g_tgt)]
    lo, hi = a.rdf_range
    cmp_mask = (r >= lo) & (r <= hi)
    rms_tgt = float(np.sqrt(np.mean(g_tgt[cmp_mask] ** 2)))

    failed = []
    for label, gsd, info in configs:
        print(f"[structure_metrics] {label}: {gsd}")
        cache = out / label / "sq_recomputed.npy"
        if a.reuse_sq and cache.is_file():
            sim_sq = np.load(cache)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                sim_sq = recompute_sq(gsd, Path(tmp) / label)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, sim_sq)
        diag, reduced = sq_metrics(target_sq, sim_sq)
        n_frames = gsd_n_frames(gsd)
        indices = select_frames(n_frames, *FORWARD_FRAMES)
        row, _, g = analyse(label, gsd, indices, cutoffs, a.r_max, a.dr, render_dir)
        rmsd = float(np.sqrt(np.mean((g[cmp_mask] - g_tgt[cmp_mask]) ** 2)))
        row.update({
            "eval_id": info.get("eval_id"),
            **{k: info.get(k) for k in ("U0", "r0", "n", "m", "m_rel")},
            "loss": diag["loss"], "m4": diag["m4"], "shift": diag["shift_term"],
            "q1": diag["q1_sim"], "q1_tgt": diag["q1_tgt"], "q1_ratio": diag["q1_ratio"],
            "aligned": diag["aligned"], "dispersed_sim": diag["dispersed_sim"],
            "anchor_scale": diag["anchor_scale"],
            "gr_rmsd": rmsd, "gr_rmsd_rel": rmsd / rms_tgt if rms_tgt > 0 else None,
            "gr_r1_ratio": (row["gr_r1"] / rows[0]["gr_r1"]
                            if row["gr_r1"] and rows[0]["gr_r1"] else None),
        })
        rt_sq, rt_diag = runtime_files(gsd)
        if rt_diag is not None and not rt_diag.get("failed"):
            row["loss_runtime"] = rt_diag["loss"]
            row["loss_diff"] = diag["loss"] - rt_diag["loss"]
            if abs(row["loss_diff"]) > a.loss_tol:
                failed.append(label)
        if rt_sq is not None:
            rescored, _ = sq_metrics(target_sq, rt_sq)
            row["loss_runtime_sq_rescored"] = rescored["loss"]
            same_grid = rt_sq.shape == sim_sq.shape and np.allclose(rt_sq[:, 0], sim_sq[:, 0])
            row["sq_max_abs_diff"] = (float(np.max(np.abs(rt_sq[:, 1] - sim_sq[:, 1])))
                                      if same_grid else None)
            row["sq_shift_rmse_vs_runtime"] = sq_metrics(rt_sq, sim_sq)[0]["loss"]
        plot_sq(out / f"sq_{label}.png", label, target_sq, sim_sq, diag, reduced)
        rows.append(row)
        curves.append((label, g))

    rows[0]["gr_rmsd"] = 0.0
    rows[0]["gr_rmsd_rel"] = 0.0
    plot_gr(out / "gr.png", r, curves, a.rdf_range)
    plot_ptm(out / "ptm.png", rows, cutoffs[0])
    if summary is not None:
        plot_convergence(out / "convergence.png",
                         Path(a.run).expanduser() / "pack_summary" / "evals.csv")
    np.savez(out / "gr.npz", r=r, **{lab: g for lab, g in curves})

    settings = {
        "target_dir": str(tdir), "target_sq": str(a.target_sq or tdir / "cpu" / "target_sq.npy"),
        "target_frames": tgt_frames, "forward_frames": list(FORWARD_FRAMES),
        "ptm_types": PTM_TYPES, "ptm_cutoffs": cutoffs, "ptm_fractions": "of all N particles",
        "gr_normalization": "box number density: g = <pairs in shell> / "
                            "(N/2 * (N/V_box) * shell volume); minimum image",
        "r_max": a.r_max, "dr": a.dr, "rdf_range": a.rdf_range,
        "gr_rmsd_rel": "gr_rmsd / RMS(g_tgt) on rdf_range",
        "loss_tol": a.loss_tol, "check_failed": failed,
    }
    (out / "metrics.json").write_text(json.dumps({"settings": settings, "rows": rows},
                                                 indent=2, default=str) + "\n")
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})
    print(json.dumps(settings, indent=2, default=str))
    if failed:
        print(f"CHECK FAILED: |recomputed - run-time loss| > {a.loss_tol} for {failed}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
