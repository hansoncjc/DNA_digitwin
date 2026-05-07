"""
Simulation of DNA-mediated SiNP using HOOMD-blue.
Supports two pair potentials selected via the `potential` argument of
run_simulation():
  - "vdw_es"  : vdw and screened electrostatics summed together.

Table bounds (rmin, rmax) are derived analytically from the potential
parameters rather than being hard-coded, so they remain valid across
parameter sweeps.  See _compute_table_bounds() for the derivation.
"""
from __future__ import annotations
import os, time
import numpy as np
import hoomd
import hoomd.md
from datetime import datetime
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Pair potential definitions
# ---------------------------------------------------------------------------
def hs_potential(r, rmin, rmax, dt):
    '''
    Hard-sphere-like potential using a quadratic form that goes to zero at r=2.
    '''
    U = 1./(4.*dt)*(2. - r)**2.
    F = 1./(2.*dt)*(2. - r)
    return(U, F)


def screen_potential(r, deb_length, z, radius):
    '''
    Screened electrostatic repulsion for monodisperse spheres.
    '''
    charge = z * 4 * np.pi * 80 * 8.854e-12 * radius * (1 + radius/deb_length)
    gamma = charge**2 / (4 * np.pi * 80 * 8.854e-12 * radius * (1 + radius/deb_length)**2 * 298 * 1.38e-23)

    U = gamma*np.exp(-(radius/deb_length)*(r - 2))/r
    F = gamma*np.exp(-(radius/deb_length)*(r - 2))*((radius/deb_length)/r + 1/(r**2))
    return(U, F)


def vdw_potential(r, A):
    '''
    Van der Waals potential for monodisperse spheres.
    Input A is hamaker constant.
    Returns (U(r), F(r)). Pure function (no side effects).
    '''
    U = -(A/3)*(1/(r**2 - 4) + 1/(r**2) + 0.5*np.log((r**2 - 4)/(r**2))) 
    F = (A/3)*((-2*r)/(r**2 - 4)**2 - 2/(r**3) + 4/(r**3 - 4*r)) 
    return(U, F)


def vdw_es(r, rmin, rmax, A, deb_length, z, radius):
    """
    Combined van der Waals attraction and screened electrostatic
    for monodisperse spheres.

    Returns (U(r), F(r)).  Pure function (no side effects).
    """
    vdw_U, vdw_F = vdw_potential(r, A)
    es_U, es_F = screen_potential(r, deb_length, z, radius)
    return vdw_U + es_U, vdw_F + es_F


# Registry: selector string -> (function, requires_delta)
_POTENTIALS = {
    "vdw_es" : (vdw_es, False)
}

# ---------------------------------------------------------------------------
# Table bounds helper functions
# ---------------------------------------------------------------------------

def U_vdw_es(r, A, deb_length, z, radius):
    '''Helper to compute just the potential energy U(r) for root-finding.'''
    U, _ = vdw_es(r, None, None, A, deb_length, z, radius)
    return U

def root_func(r, U_target, A, deb_length, z, radius):
    '''Root function for solving r at which U_vdw_es(r) = U_target.'''
    return U_vdw_es(r, A, deb_length, z, radius) - U_target

def solve_r(U_target, r_low, r_high, A, deb_length, z, radius):
    '''Solve for r such that U_vdw_es(r) = U_target using Brent's method.'''
    return brentq(
        root_func,
        r_low,
        r_high,
        args=(U_target, A, deb_length, z, radius)
    )

def find_rmax_decay(
    A,
    deb_length,
    z,
    radius,
    r_start=4.5,
    r_end=10.0,
    npts=2000,
    eps_U=0.02,
    eps_F=0.02):
    r_grid = np.linspace(r_start, r_end, npts)

    for r in r_grid:
        U, F = vdw_es(r, None, None, A, deb_length, z, radius)

        if (abs(U) < eps_U) and (abs(F) < eps_F):
            return r

    raise ValueError("No decay region found within search window")

# ---------------------------------------------------------------------------
# Physics-derived table bounds
# ---------------------------------------------------------------------------

def _compute_table_bounds(
    potential: str,
    A: float,
    deb_length: float,
    z: float,
    radius: float,
    delta: float | None,
    *,
    dt: float = 1e-4,
    min_tol_vdw_es: float = -150., # Must be negative
    max_tol_vdw_es: float = 0.02, # Must be positive
) -> tuple[float, float]:
    """
    Compute (rmin, rmax) analytically from the potential parameters.

    The strategy for both potentials is:
      - rmin: place it far enough from any singularity to keep forces
              finite and table values well-defined.
      - rmax: solve for the separation at which the attractive tail
              decays to the tolerance t_tol. This guarantees the
              cutoff never truncates the well prematurely, regardless of
              how (A, deb_length, z, radius) vary across a sweep.

    Parameters
    ----------
    potential : {"vdw_es"}
    A : float   -- Hamaker constant for VdW attraction (vdw_es)
    deb_length : float  -- Debye length for screened electrostatics (vdw_es)
    z : float  -- particle charge number for screened electrostatics (vdw_es)
    radius : float  -- particle radius for screened electrostatics (vdw_es)
    dt : float  -- time step for hard-sphere repulsion strength (vdw_es)
    delta : float -- length scale for shifted_mie; ignored for modified_lj
    max_tol_vdw_es : float
        Tail-energy tolerance used as rmax cutoff for vdw_es.
        rmax is where U = max_tol_vdw_es.  Default 0.02.
    min_tol_vdw_es : float
        Tail-energy tolerance used as rmin cutoff for vdw_es.
        rmin is where U = min_tol_vdw_es.  Default -300.

    Returns
    -------
    rmin, rmax : float

    Raises
    ------
    ValueError  -- if parameters would produce rmin >= rmax.
    """
    if potential == "vdw_es":

        # pick safe search brackets (important!)
        rmin = solve_r(min_tol_vdw_es, r_low=2.001, r_high=3.0, A=A,
                    deb_length=deb_length, z=z, radius=radius)

        # --- rmax now decay-based (NO root solving) ---
        rmax = find_rmax_decay(
            A=A,
            deb_length=deb_length,
            z=z,
            radius=radius,
            r_start=rmin + 1.0,
            r_end=15.0,
            eps_U=max_tol_vdw_es,
            eps_F=max_tol_vdw_es
        )
    else:
        raise ValueError(f"Unknown potential: {potential!r}")

    if rmin >= rmax:
        raise ValueError(
            f"Computed rmin ({rmin:.4f}) >= rmax ({rmax:.4f}) for "
            f"potential={potential!r}, A={A}, deb_length={deb_length}, z={z}, radius={radius}.  "
            f"Check that tolerance and bracket parameters are reasonable."
        )

    return rmin, rmax


# ---------------------------------------------------------------------------
# Main simulation entry-point
# ---------------------------------------------------------------------------

def run_simulation(
    phi: float,
    A: float,
    deb_length: float,
    z: float,
    outdir: str,
    *,
    potential: str = "vdw_es",
    delta: float | None = None,
    radius: float = 1E-8,
    N: int = 8000,
    dt: float = 1e-4,
    steps: int = 20_000_000,
    kT: float = 1.0,
    max_tol_vdw_es: float = 0.02, # Must be positive
    min_tol_vdw_es: float = -150., # Must be negative
    init_offset: float = 0.1,
    device: str = "gpu",   # "cpu" also works on HOOMD 2.x
    seed: int = datetime.now().microsecond,
    plot: bool = True
) -> dict:
    """
    Run a HOOMD simulation of N spheres with a selectable pair potential.

    Parameters
    ----------
    phi : float
        Volume fraction (particle_volume / box_volume).
    A : float
        Hamaker constant (kT's).
    deb_length : float
        Debye length.
    z : float
        Particle zeta potential (V).
    radius : float
        Particle radius (m).
    outdir : str
        Directory to write artifacts (gsd, csv).
    potential : {"vdw_es"}
        Selects the pair potential.  Default is ``"vdw_es"``.
    delta : float, optional
        Length-scale parameter required by ``"shifted_mie"``.
    N, dt, steps, kT : see defaults
    max_tol_vdw_es : float
        Tail-energy tolerance for rmax (vdw_es).  rmax is where the
        attractive tail falls to max_tol_vdw_es.  Default 0.02.
    min_tol_vdw_es : float
        Tail-energy tolerance for rmin (vdw_es).  rmin is where the
        attractive tail falls to min_tol_vdw_es.  Default -300.
    init_offset : float
        Added to rmin to set the minimum distance between particles during
        random initialization. Default 0.1.
    device : {"gpu","cpu"}
        HOOMD context device mode.
    seed : int
        Integrator seed
    plot : bool
        Controls whether potential and energy plots are generated.

    Returns
    -------
    dict with keys:
        gsd_path, energy_csv, rmin, rmax, table_width, potential

    Raises
    ------
    ValueError
        If an unknown potential name is given, if ``"shifted_mie"`` is
        selected without providing ``delta``, or if the derived rmin >= rmax.
    """
    if potential not in _POTENTIALS:
        raise ValueError(
            f"Unknown potential {potential!r}. "
            f"Choose from: {list(_POTENTIALS)}"
        )
    pot_fn, needs_delta = _POTENTIALS[potential]
    if needs_delta and delta is None:
        raise ValueError("`delta` must be provided when potential='shifted_mie'.")

    os.makedirs(outdir, exist_ok=True)

    # --- Analytically derived table bounds ---
    rmin, rmax = _compute_table_bounds(
        potential, A, deb_length, z, radius, delta,
        max_tol_vdw_es=max_tol_vdw_es, min_tol_vdw_es=min_tol_vdw_es
    )
    print(f"Table bounds: rmin={rmin:.4f}, rmax={rmax:.4f} "
          f"(min_tol_vdw_es={min_tol_vdw_es}, max_tol_vdw_es={max_tol_vdw_es})")

    # --- HOOMD context ---
    mode_flag = "--mode=gpu" if device == "gpu" else "--mode=cpu"
    hoomd.context.initialize(mode_flag)

    # --- Derived params & box ---
    volume = 4 * N * np.pi / (3 * phi)  # from N, phi, and particle volume
    L = volume ** (1.0 / 3.0)  # cubic box from density.

    # --- Generate non-overlapping initial positions (same strategy) ---
    def generate_positions(N, L, rmin, offset=0.1):
        # min_dist = rmin + offset
        # positions, attempts, max_attempts = [], 0, N * 1000
        # while len(positions) < N and attempts < max_attempts:
        #     pos = np.random.uniform(-L / 2, L / 2, 3)
        #     if all(np.linalg.norm(pos - np.array(p)) >= min_dist for p in positions):
        #         positions.append(pos)
        #     attempts += 1
        # if len(positions) < N:
        #     raise RuntimeError("Failed to generate non-overlapping configuration.")
        # return positions

        # cubic lattice initialization
        min_dist = rmin + offset
        n_side = int(np.ceil(N ** (1/3)))
        spacing = L / n_side
        if spacing < min_dist:
            raise ValueError(
                f"Box too small for non-overlapping init: "
                f"grid spacing {spacing:.3f} < min_dist {min_dist:.3f}. "
                f"Increase box size (lower density) or reduce rmin."
            )
        # Build a simple cubic lattice
        coords = np.linspace(-L/2 + spacing/2, L/2 - spacing/2, n_side)
        grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
        # Shuffle and take N points
        rng = np.random.default_rng()
        rng.shuffle(grid)
        return grid[:N].tolist()

    positions = generate_positions(N, L, rmin=rmin, offset=init_offset)

    # --- Snapshot and system init ---
    snapshot = hoomd.data.make_snapshot(
        N=N, box=hoomd.data.boxdim(L=L), particle_types=['A']
    )
    for i, pos in enumerate(positions):
        snapshot.particles.position[i] = pos
        snapshot.particles.diameter[i] = 2.0
    hoomd.init.read_snapshot(snapshot)

    # --- Pair potential via table ---
    width = 1000
    nl = hoomd.md.nlist.cell()
    table_combined = hoomd.md.pair.table(width=width, nlist=nl)
    table_hs = hoomd.md.pair.table(width=width, nlist=nl)

    # Build coefficient dict; add delta only for shifted_mie.
    coeff = dict(A=A, deb_length=deb_length, z=z, radius=radius)
    extra_coeff = {}
    if potential == "shifted_mie":
        coeff["delta"] = delta
        extra_coeff["delta"] = delta

    table_combined.pair_coeff.set(
        'A', 'A',
        rmin=rmin, rmax=rmax,
        func=pot_fn,
        coeff=coeff
    )

    table_hs.pair_coeff.set(
        'A', 'A',
        rmin=0.0, rmax=2.0,
        func=hs_potential,
        coeff=dict(dt=dt)
    )

    # --- Generate Potential Plot ---
    if plot:
        out_png = os.path.join(outdir, "potential_plot.png")
        plot_pair_potential(rmin, rmax, width, A, deb_length,
                            z, radius, out_png, pot_fn,
                            extra_coeff=extra_coeff)

    # --- Integrator ---
    group_all = hoomd.group.all()
    hoomd.md.integrate.mode_standard(dt=dt)
    bd = hoomd.md.integrate.brownian(group=group_all, kT=kT, seed=seed)
    bd.set_gamma('A', gamma=1.0)

    # --- Outputs: GSD + energy CSV ---
    ts = time.localtime()
    timestamp = f"{ts.tm_year:02d}{ts.tm_mon:02d}{ts.tm_mday:02d}{ts.tm_hour:02d}{ts.tm_min:02d}{ts.tm_sec:02d}"
    gsd_path = os.path.join(outdir, f"AuNP_assembly_{timestamp}.gsd")
    hoomd.dump.gsd(filename=gsd_path, period=50000, group=group_all, overwrite=True)

    energy_csv = os.path.join(outdir, "potential_energy.csv")
    hoomd.analyze.log(
        filename=energy_csv,
        quantities=['potential_energy'],
        period=5000,
        overwrite=True
    )

    # --- Run ---
    print(f"Running {steps} steps with {N} spheres at phi {phi:.3f} "
          f"using potential='{potential}'")
    hoomd.run(steps)

    # --- Generate Potential Energy Plot ---
    if plot:
        plot_energy(energy_csv, os.path.join(outdir, "potential_energy_plot.png"))
        print(f"Simulation complete. Plots Generated. rmin={rmin:.4f}, rmax={rmax:.4f}")
    else:
        print("Simulation complete.")

    return {
        "gsd_path"    : gsd_path,
        "energy_csv"  : energy_csv,
        "rmin"        : rmin,
        "rmax"        : rmax,
        "table_width" : width,
        "potential"   : potential,
    }


# ---------------------------------------------------------------------------
# Plotting Block from original codebase; could be moved to a separate file
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd


def plot_pair_potential(rmin, rmax, width, A, deb_length,
                        z, radius, out_png, potential_fn,
                        extra_coeff=None):
    """Plot U(r) for any potential that follows the HOOMD table-function API.

    extra_coeff : dict, optional
        Additional keyword arguments forwarded to potential_fn beyond the
        standard (r, rmin, rmax, A, deb_length, z, radius) signature (e.g. ``delta``).
    """
    extra_coeff = extra_coeff or {}
    r_vals = np.linspace(rmin, rmax, width)
    U, _ = potential_fn(r_vals, rmin, rmax, A, deb_length, z, radius, **extra_coeff)
    label = f"A={A}, deb_length={deb_length}, z={z}, radius={radius}"
    if extra_coeff:
        label += ", " + ", ".join(f"{k}={v}" for k, v in extra_coeff.items())
    plt.figure(figsize=(6, 4))
    plt.plot(r_vals, U, label=label)
    plt.xlabel("r"); plt.ylabel("U(r)"); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=600); plt.close()


def plot_energy(csv_path, out_png):
    df = pd.read_csv(csv_path, delimiter='\t').values
    plt.figure(figsize=(6, 4))
    plt.plot(df[6:, 0], df[6:, 1])
    plt.xlabel("Time"); plt.ylabel("Potential Energy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=600); plt.close()
