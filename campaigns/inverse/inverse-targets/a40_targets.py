#!/usr/bin/env python
"""Compute the R6 target S(q) on an A40 GPU from the existing configurations.

Each subdirectory of <in_root> holds one target's GSD (fcc_crystallite.gsd or
fluid_droplets.gsd). S(q): saxsfft.StructureFactor, N_grid 600, trim slice(6, -6),
particle_diameter 24.6 nm, step 1; frames "all" for the fluid, "last:1" for FCC.
The GSD is copied to a temporary directory first (StructureFactor writes files next
to its input). Writes <out_root>/<target>/{target_sq.npy, target_sq.txt, a40_info.json}.
Refuses to overwrite an existing <out_root>.

Usage: python -u a40_targets.py <in_root> <out_root> [--device cuda] [--require-gpu A40]
"""
import argparse
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import saxsfft
from saxsfft import StructureFactor

N_GRID, TRIM, DIAMETER_NM = 600, slice(6, -6), 24.6

ap = argparse.ArgumentParser()
ap.add_argument("in_root")
ap.add_argument("out_root")
ap.add_argument("--device", default="cuda")
ap.add_argument("--require-gpu", default="A40", help="substring of the GPU name; '' to skip")
args = ap.parse_args()
in_root, out_root = Path(args.in_root), Path(args.out_root)

if args.device == "cuda":
    if not torch.cuda.is_available():
        sys.exit("cuda not available")
    gpu = torch.cuda.get_device_name(0)
    if args.require_gpu and args.require_gpu not in gpu:
        sys.exit(f"GPU is {gpu!r}, required {args.require_gpu!r}")
else:
    gpu = None
if out_root.exists():
    sys.exit(f"{out_root} exists; not overwriting")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


repo = Path(saxsfft.__file__).resolve().parent.parent
try:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    commit = None
env = {"host": socket.gethostname(), "device": args.device, "gpu": gpu,
       "python": platform.python_version(), "torch": torch.__version__,
       "numpy": np.__version__, "saxsfft_version": saxsfft.__version__,
       "saxsfft_path": str(repo), "saxsfft_commit": commit}
print(json.dumps(env))

targets = sorted(d for d in in_root.iterdir() if d.is_dir() and list(d.glob("*.gsd")))
if not targets:
    sys.exit(f"no target directories with a .gsd under {in_root}")
for d in targets:
    gsds = sorted(d.glob("*.gsd"))
    if len(gsds) != 1:
        sys.exit(f"{d}: expected one .gsd, found {[g.name for g in gsds]}")
    gsd = gsds[0]
    frames = "all" if gsd.name.startswith("fluid") else "last:1"
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / gsd.name
        shutil.copy2(gsd, g)
        q, s = StructureFactor(str(g), N_GRID, frames=frames, step=1,
                               particle_diameter=DIAMETER_NM, trim=TRIM,
                               device=args.device).compute_s_1d()
    data = np.column_stack([q, s])
    od = out_root / d.name
    od.mkdir(parents=True)
    np.save(od / "target_sq.npy", data)
    np.savetxt(od / "target_sq.txt", data, header="q S(q)", fmt="%.8e")
    info = dict(env, target=d.name, input_gsd=str(gsd), input_gsd_md5=md5(gsd),
                frames=frames, step=1, N_grid=N_GRID, trim=[6, -6],
                particle_diameter_nm=DIAMETER_NM, n_points=int(data.shape[0]),
                q_range=[float(q[0]), float(q[-1])],
                target_sq_npy_md5=md5(od / "target_sq.npy"),
                created=datetime.now().isoformat(timespec="seconds"))
    (od / "a40_info.json").write_text(json.dumps(info, indent=2))
    print(f"{d.name}: {data.shape[0]} points, npy md5 {info['target_sq_npy_md5']}")
print("done")
