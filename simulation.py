"""
Simulation of DNA-mediated SiNP using HOOMD-blue.
Supports two pair potentials selected via the `potential` argument of
run_simulation():
  - "modified_lj"  : n-m Lennard-Jones-like potential (original)
  - "shifted_mie"  : shifted Mie potential with hard-core offset r0 and
                     length-scale delta

Table bounds (rmin, rmax) are derived analytically from the potential
parameters rather than being hard-coded, so they remain valid across
parameter sweeps.  See _compute_table_bounds() for the derivation.
"""
from __future__ import annotations
import os, time
import numpy as np
import hoomd
import hoomd.md


# ---------------------------------------------------------------------------
# Pair potential definitions
# ---------------------------------------------------------------------------

def modified_LJ(r, rmin, rmax, U_0, n, m, r0):
    """
    n-m Lennard-Jones-like table potential (original).

    .. math::
        U(r) = \\frac{U_0}{n-m}\\left[m\\left(\\frac{r_0}{r}\\right)^n
                - n\\left(\\frac{r_0}{r}\\right)^m\\right]

    Returns (U(r), F(r)).  Pure function (no side effects).
    """
    U = U_0 / (n - m) * (m * (r0 / r) ** n - n * (r0 / r) ** m)
    F = U_0 * m * n * ((r0 / r) ** n - (r0 / r) ** m) / ((n - m) * r)
    return U, F


def shifted_mie(r, rmin, rmax, U_0, n, m, r0, delta):
    """
    Shifted Mie pair potential.

    .. math::
        U(r) = U_0 \\, C \\left[
            \\left(\\frac{\\delta}{r - r_0}\\right)^n -
            \\left(\\frac{\\delta}{r - r_0}\\right)^m
        \\right]

    with the standard Mie prefactor

    .. math::
        C = \\frac{n}{n-m}\\left(\\frac{n}{m}\\right)^{\\frac{m}{n-m}}

    Parameters
    ----------
    r : array-like
        Pair separation distances.
    rmin, rmax : float
        Table bounds (passed by HOOMD; not used in the math directly).
    U_0 : float
        Energy scale / well depth.
    n, m : float
        Repulsive and attractive exponents (n > m).
    r0 : float
        Hard-core shift origin; effective variable is xi = r - r0.
        Must satisfy rmin > r0 to avoid the xi = 0 singularity.
    delta : float
        Length scale.  The potential minimum sits at
        r_well = r0 + delta * (n/m)^(1/(n-m)).

    Returns
    -------
    U : ndarray -- potential energy
    F : ndarray -- force magnitude (-dU/dr, positive = repulsive)
    """
    C  = (n / (n - m)) * (n / m) ** (m / (n - m))
    xi = r - r0
    dn = (delta / xi) ** n
    dm = (delta / xi) ** m
    U  = U_0 * C * (dn - dm)
    F  = U_0 * C * (n * dn - m * dm) / xi   # F = -dU/dr
    return U, F


# Registry: selector string -> (function, requires_delta)
_POTENTIALS = {
    "modified_lj" : (modified_LJ, False),
    "shifted_mie" : (shifted_mie, True),
}


# ---------------------------------------------------------------------------
# Physics-derived table bounds
# ---------------------------------------------------------------------------

def _compute_table_bounds(
    potential: str,
    U_0: float,
    n: float,
    m: float,
    r0: float,
    delta: float | None,
    *,
    tol_frac: float = 1e-4,
    f_inner: float = 0.75,
) -> tuple[float, float]:
    """
    Compute (rmin, rmax) analytically from the potential parameters.

    The strategy for both potentials is:
      - rmin: place it on the repulsive side of the well minimum, far
              enough from any singularity to keep forces finite and
              table values well-defined.
      - rmax: solve for the separation at which the attractive tail
              decays to U_tol = tol_frac * |U_0|.  This guarantees the
              cutoff never truncates the well prematurely, regardless of
              how (n, m, r0, delta) vary across a sweep.

    Parameters
    ----------
    potential : {"modified_lj", "shifted_mie"}
    U_0 : float   -- energy scale / well depth
    n, m : float  -- repulsive and attractive exponents (n > m > 0)
    r0 : float    -- reference length (modified_lj) or hard-core shift (shifted_mie)
    delta : float -- length scale for shifted_mie; ignored for modified_lj
    tol_frac : float
        Fraction of |U_0| used as the tail-energy tolerance.
        Default 1e-4 gives ~4 decimal places accuracy at the cutoff.
    f_inner : float
        Fraction of the inner potential minimum distance used to establish rmin.
        For modified_lj: rmin = f_inner * r0.
        For shifted_mie: rmin is placed at f_inner * xi_well above r0.
        Must satisfy 0 < f_inner < 1. Default 0.75.

    Returns
    -------
    rmin, rmax : float

    Raises
    ------
    ValueError  -- if parameters would produce rmin >= rmax or rmin <= r0
                   (shifted_mie only).

    Notes
    -----
    modified_lj
    -----------
    The well minimum is exactly at r = r0.  The attractive tail behaves
    asymptotically as:

        U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m

    Setting U_attr(rmax) = U_tol = tol_frac * |U_0| and solving:

        rmax = r0 * (n / ((n-m) * tol_frac))^(1/m)

    For rmin, a fraction f_inner below the minimum is used (repulsive side):

        rmin = f_inner * r0

    shifted_mie
    -----------
    Let xi = r - r0.  The singularity is at xi = 0 (r = r0).
    The well minimum is at xi_well = delta * (n/m)^(1/(n-m)).
    The Mie prefactor is C = n/(n-m) * (n/m)^(m/(n-m)).

    rmin = r0 + f_inner * xi_well     (repulsive side, away from xi=0)
           with hard clamp rmin > r0 + 1e-3

    The attractive tail: U_attr ~ U_0 * C * (delta/xi)^m
    Setting U_attr(xi_max) = U_tol:

        xi_max = delta * (|U_0| * C / U_tol)^(1/m)
               = delta * (C / tol_frac)^(1/m)
        rmax   = r0 + xi_max
    """
    U_tol = tol_frac * abs(U_0)

    if potential == "modified_lj":
        # rmin: f_inner fraction of r0 (repulsive side of minimum at r0)
        rmin = f_inner * r0

        # rmax: tail decay to U_tol
        # U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m  => rmax = r0*(n/((n-m)*tol_frac))^(1/m)
        prefactor_m = n / (n - m)        # coefficient of the attractive (r0/r)^m term
        rmax = r0 * (prefactor_m / tol_frac) ** (1.0 / m)

    elif potential == "shifted_mie":
        if delta is None:
            raise ValueError("delta is required for shifted_mie bounds.")

        C       = (n / (n - m)) * (n / m) ** (m / (n - m))
        xi_well = delta * (n / m) ** (1.0 / (n - m))   # distance from r0 to well min

        # rmin: f_inner fraction of the way to the well minimum
        rmin = r0 + f_inner * xi_well
        rmin = max(rmin, r0 + 1e-3)     # hard clamp away from singularity

        # rmax: attractive tail decay to U_tol
        # U_attr(xi) ~ U_0*C*(delta/xi)^m  => xi_max = delta*(|U_0|*C/U_tol)^(1/m)
        xi_max = delta * (abs(U_0) * C / U_tol) ** (1.0 / m)
        rmax   = r0 + xi_max

    else:
        raise ValueError(f"Unknown potential: {potential!r}")

    if rmin >= rmax:
        raise ValueError(
            f"Computed rmin ({rmin:.4f}) >= rmax ({rmax:.4f}) for "
            f"potential={potential!r}, n={n}, m={m}, r0={r0}, delta={delta}, "
            f"tol_frac={tol_frac}.  Consider increasing tol_frac or checking "
            f"that n > m > 0."
        )

    return rmin, rmax


# ---------------------------------------------------------------------------
# Main simulation entry-point
# ---------------------------------------------------------------------------

def run_simulation(
    density: float,
    U_0: float,
    r0: float,
    n: float,
    m: float,
    outdir: str,
    *,
    potential: str = "modified_lj",
    delta: float | None = None,
    N: int = 5000,
    dt: float = 1e-3,
    steps: int = 15_000_000,
    kT: float = 1.0,
    tol_frac: float = 1e-4,
    f_inner: float = 0.75,
    device: str = "gpu",   # "cpu" also works on HOOMD 2.x
    seed: int = 42,
    plot: bool = True
) -> dict:
    """
    Run a HOOMD simulation of N spheres with a selectable pair potential.

    Parameters
    ----------
    density : float
        Number density (N / box_volume).
    U_0 : float
        Energy scale / well depth.
    r0 : float
        Reference length.
        - modified_lj : equilibrium distance scale; well minimum is at r = r0.
        - shifted_mie : hard-core shift origin; effective variable is xi = r - r0.
    n, m : float
        Repulsive and attractive exponents (n > m > 0).
    outdir : str
        Directory to write artifacts (gsd, csv).
    potential : {"modified_lj", "shifted_mie"}
        Selects the pair potential.  Default is ``"modified_lj"``.
    delta : float, optional
        Length-scale parameter required by ``"shifted_mie"``.
        Ignored when ``potential="modified_lj"``.
    N, dt, steps, kT : see defaults
    tol_frac : float
        Fraction of |U_0| used as the energy tolerance for rmax.
        rmax is the separation at which the attractive tail falls below
        tol_frac * |U_0|.  Default 1e-4.  Increase (e.g. 1e-3) for a
        shorter cutoff; decrease (e.g. 1e-6) for a longer one.
    f_inner : float
        Fraction of inner coordinate boundary to establish rmin.
        rmin = f_inner * r0 (modified_lj) or r0 + f_inner * xi_well (shifted_mie). Default 0.75.
    device : {"gpu","cpu"}
        HOOMD context device mode.
    seed : int
        Langevin thermostat seed.
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
        potential, U_0, n, m, r0, delta,
        tol_frac=tol_frac, f_inner=f_inner
    )
    print(f"Table bounds: rmin={rmin:.4f}, rmax={rmax:.4f} "
          f"(tol_frac={tol_frac}, f_inner={f_inner})")

    # --- HOOMD context ---
    mode_flag = "--mode=gpu" if device == "gpu" else "--mode=cpu"
    hoomd.context.initialize(mode_flag)

    # --- Derived params & box ---
    volume = N / density
    L = volume ** (1.0 / 3.0)  # cubic box from density.

    # --- Generate non-overlapping initial positions (same strategy) ---
    def generate_positions(N, L, min_dist=1.1):
        positions, attempts, max_attempts = [], 0, N * 1000
        while len(positions) < N and attempts < max_attempts:
            pos = np.random.uniform(-L / 2, L / 2, 3)
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

    # --- Pair potential via table ---
    width = 1000
    nl = hoomd.md.nlist.cell()
    table = hoomd.md.pair.table(width=width, nlist=nl)

    # Build coefficient dict; add delta only for shifted_mie.
    coeff = dict(U_0=U_0, n=n, m=m, r0=r0)
    extra_coeff = {}
    if potential == "shifted_mie":
        coeff["delta"] = delta
        extra_coeff["delta"] = delta

    table.pair_coeff.set(
        'A', 'A',
        rmin=rmin, rmax=rmax,
        func=pot_fn,
        coeff=coeff
    )

    # --- Generate Potential Plot ---
    if plot:
        out_png = os.path.join(outdir, "potential_plot.png")
        plot_pair_potential(rmin, rmax, width, U_0, n, m, r0, out_png,
                            pot_fn, extra_coeff=extra_coeff)

    # --- Integrator ---
    group_all = hoomd.group.all()
    hoomd.md.integrate.mode_standard(dt=dt)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=seed)
    langevin.set_gamma('A', gamma=1.0)

    # --- Outputs: GSD + energy CSV ---
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
    print(f"Running {steps} steps with {N} spheres at density {density:.3f} "
          f"using potential='{potential}'")
    hoomd.run(steps)

    # --- Generate Potential Energy Plot ---
    if plot:
        plot_energy(energy_csv, os.path.join(outdir, "potential_energy_plot.png"))
        print("Simulation complete. Plots Generated.")
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


def plot_pair_potential(rmin, rmax, width, U_0, n, m, r0, out_png,
                        potential_fn, extra_coeff=None):
    """Plot U(r) for any potential that follows the HOOMD table-function API.

    extra_coeff : dict, optional
        Additional keyword arguments forwarded to potential_fn beyond the
        standard (r, rmin, rmax, U_0, n, m, r0) signature (e.g. ``delta``).
    """
    extra_coeff = extra_coeff or {}
    r_vals = np.linspace(rmin, rmax, width)
    U, _ = potential_fn(r_vals, rmin, rmax, U_0, n, m, r0, **extra_coeff)
    label = f"n={n}, m={m}, r0={r0}, U0={U_0}"
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
