from __future__ import annotations
import os, time
import numpy as np
import hoomd
import hoomd.md

def modified_LJ(r, rmin, rmax, U_0, n, m, r0):
    """
    n–m Lennard-Jones-like table potential used in the project.
    Returns (U(r), F(r)). Pure function (no side effects).
    """
    U = U_0/(n - m) * (m*(r0/r)**n - n*(r0/r)**m)
    F = U_0*m*n*((r0/r)**n - (r0/r)**m)/((n-m)*r)
    return U, F

def run_simulation(
    density: float,
    U_0: float,
    r0: float,
    n: float,
    m: float,
    outdir: str,
    *,
    N: int = 5000,
    dt: float = 1e-3,
    steps: int = 15_000_000,
    kT: float = 1.0,
    rmax: float = 5.0,
    device: str = "gpu",   # "cpu" also works on HOOMD 2.x
    seed: int = 42
) -> dict:
    """
    Run a HOOMD simulation of N spheres with an n–m LJ table potential.

    Parameters
    ----------
    density : float
        Number density (N / box_volume).
    U_0, r0, n, m : float
        Potential parameters for modified_LJ.
    outdir : str
        Directory to write artifacts (gsd, csv).
    N, dt, steps, kT : see defaults
    rmax : float
        Table potential r-maximum; rmin is set to 0.75*r0.
    device : {"gpu","cpu"}
        HOOMD context device mode.
    seed : int
        Langevin thermostat seed.

    Returns
    -------
    dict with:
        {
          "gsd_path": ".../DNA_assembly_<timestamp>.gsd",
          "energy_csv": ".../potential_energy.csv",
          "rmin": float,
          "rmax": float,
          "table_width": int
        }

    Notes
    -----
    - This function is *purely simulation*: no plotting, no pandas.
    - The potential and energy logging mirror the original script behavior.
    """
    os.makedirs(outdir, exist_ok=True)

    # --- HOOMD context ---
    mode_flag = "--mode=gpu" if device == "gpu" else "--mode=cpu"
    hoomd.context.initialize(mode_flag)

    # --- Derived params & box ---
    rmin = 0.75 * r0
    volume = N / density
    L = volume ** (1.0 / 3.0) # cubic box from density.

    # --- Generate non-overlapping initial positions (same strategy) ---
    def generate_positions(N, L, min_dist=1.1):
        positions, attempts, max_attempts = [], 0, N*1000
        while len(positions) < N and attempts < max_attempts:
            pos = np.random.uniform(-L/2, L/2, 3)
            if all(np.linalg.norm(pos - np.array(p)) >= min_dist for p in positions):
                positions.append(pos)
            attempts += 1
        if len(positions) < N:
            raise RuntimeError("Failed to generate non-overlapping configuration.")
        return positions
    positions = generate_positions(N, L)

    # --- Snapshot and system init ---
    snapshot = hoomd.data.make_snapshot(
        N=N, box=hoomd.data.boxdim(L=L), particle_types=['A']
    )
    for i, pos in enumerate(positions):
        snapshot.particles.position[i] = pos
        snapshot.particles.diameter[i] = 1.0
    hoomd.init.read_snapshot(snapshot) 

    # --- Pair potential via table (same math, no plotting) ---
    width = 1000
    nl = hoomd.md.nlist.cell()
    table = hoomd.md.pair.table(width=width, nlist=nl)
    table.pair_coeff.set(
        'A', 'A',
        rmin=rmin, rmax=rmax,
        func=modified_LJ,
        coeff=dict(U_0=U_0, n=n, m=m, r0=r0)
    ) 

    # --- Integrator ---
    group_all = hoomd.group.all()
    hoomd.md.integrate.mode_standard(dt=dt)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=seed)
    langevin.set_gamma('A', gamma=1.0)

    # --- Outputs: GSD + energy CSV (no figure here) ---
    ts = time.localtime()
    timestamp = f"{ts.tm_year:02d}{ts.tm_mon:02d}{ts.tm_mday:02d}{ts.tm_hour:02d}{ts.tm_min:02d}{ts.tm_sec:02d}"
    gsd_path = os.path.join(outdir, f"DNA_assembly_{timestamp}.gsd")
    hoomd.dump.gsd(filename=gsd_path, period=50000, group=group_all, overwrite=True)

    energy_csv = os.path.join(outdir, "potential_energy.csv")
    hoomd.analyze.log(
        filename=energy_csv,
        quantities=['potential_energy'],
        period=5000,
        overwrite=True
    ) 

    # --- Run ---
    print(f"Running {steps} steps with {N} spheres at density {density:.3f}")
    hoomd.run(steps)
    print("Simulation complete.")

    return {
        "gsd_path": gsd_path,
        "energy_csv": energy_csv,
        "rmin": rmin,
        "rmax": rmax,
        "table_width": width,
    }

# ---------------------------------------
# Plotting Block from original codebase; Could be used here or dumped to another file
# ---------------------------------------

# plotting.py (example)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_pair_potential(rmin, rmax, width, U_0, n, m, r0, out_png, potential_fn):
    r_vals = np.linspace(rmin, rmax, width)
    U, _ = potential_fn(r_vals, rmin, rmax, U_0, n, m, r0)
    plt.figure(figsize=(6,4))
    plt.plot(r_vals, U, label=f"n={n}, m={m}, r0={r0}, U0={U_0}")
    plt.xlabel("r"); plt.ylabel("U(r)"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

def plot_energy(csv_path, out_png):
    df = pd.read_csv(csv_path, delimiter='\t').values
    plt.figure(figsize=(6,4))
    plt.plot(df[6:,0], df[6:,1])
    plt.xlabel("Time"); plt.ylabel("Potential Energy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()
