#!/usr/bin/env python
"""Compute the R6 target S(q) on CPU from the existing configurations (CPU twin of
a40_targets.py, same settings).

Each subdirectory of <in_root> holds one target's GSD (fcc_crystallite.gsd or
fluid_droplets.gsd). S(q): saxsfft.StructureFactor, N_grid 600, trim slice(6, -6),
particle_diameter 24.6 nm, step 1, device cpu; frames "all" for the fluid, "last:1"
for FCC. The GSD is copied to a temporary directory first (StructureFactor writes
files next to its input). Writes <out_root>/<target>/{target_sq.npy, target_sq.txt,
cpu_info.json}. Refuses to overwrite an existing <out_root>.

Usage: python -u cpu_targets.py <in_root> <out_root>
"""
import hashlib
import json
import os
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

if len(sys.argv) != 3:
    sys.exit(__doc__)
in_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
if out_root.exists():
    sys.exit(f"{out_root} exists; not overwriting")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


repo = Path(saxsfft.__file__).resolve().parent.parent
try:
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    commit = None
env = {"host": socket.gethostname(), "device": "cpu", "cpu_model": cpu_model(),
       "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
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
                               device="cpu").compute_s_1d()
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
    (od / "cpu_info.json").write_text(json.dumps(info, indent=2))
    print(f"{d.name}: {data.shape[0]} points, npy md5 {info['target_sq_npy_md5']}")
print("done")
