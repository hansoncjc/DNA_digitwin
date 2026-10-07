"""
Simulation of DNA-mediated SiNP using HOOMD-blue.
Supports two pair potentials selected via the `potential` argument of
run_simulation():
  - "modified_lj"  : n-m Lennard-Jones-like potential (original)
  - "shifted_mie"  : shifted Mie potential with hard-core offset r0 and
                     length-scale delta

The simple-cubic lattice is shuffled with ``numpy.random.default_rng(seed)``
before it is written. Before 2026-10 that shuffle was unseeded, so a fixed
``seed`` did not fix the configuration that enters the hard-sphere step.
After writing the lattice GSD, run_simulation() randomizes positions
in-memory with a Heyes-Melrose hard-sphere step (from
HS_fluid_core/HS_fluid.py; contact distance max(1, rmin) by default, or
the particle diameter 1 with hs_sigma_follows_rmin=False) before applying
the selected potential. The HS segment is not dumped;
DNA_assembly_*.gsd starts at the post-HS configuration, and the number of
pairs closer than the table's rmin in that configuration is logged.

Table bounds (rmin, rmax) are derived analytically from the potential
parameters rather than being hard-coded, so they remain valid across
parameter sweeps.  See _compute_table_bounds() for the derivation and
resolve_table_bounds() for how run_simulation() picks the cutoff
(default: dynamic, t_tol_lj = tail_energy_cut / U_0 with
tail_energy_cut = 0.1).
"""
from __future__ import annotations
import os, time
import numpy as np
import gsd.hoomd
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
    t_tol_lj: float = 0.02,
    t_tol_mie: float = 4.7,
) -> tuple[float, float]:
    """
    Compute (rmin, rmax) analytically from the potential parameters.

    The strategy for both potentials is:
      - rmin: place it on the repulsive side of the well minimum, far
              enough from any singularity to keep forces finite and
              table values well-defined.
      - rmax: solve for the separation at which the attractive tail
              decays to the tolerance t_tol.  This guarantees the
              cutoff never truncates the well prematurely, regardless of
              how (n, m, r0, delta) vary across a sweep.

    Parameters
    ----------
    potential : {"modified_lj", "shifted_mie"}
    U_0 : float   -- energy scale / well depth
    n, m : float  -- repulsive and attractive exponents (n > m > 0)
    r0 : float    -- reference length (modified_lj) or hard-core shift (shifted_mie)
    delta : float -- length scale for shifted_mie; ignored for modified_lj
    t_tol_lj : float
        Tail-energy tolerance used as rmax cutoff for modified_lj.
        rmax is where U_attr = t_tol_lj * (n/(n-m)).  Default 0.02.
    t_tol_mie : float
        Tail-energy tolerance used as rmax cutoff for shifted_mie.
        rmax is where U_attr tail equals t_tol_mie.  Default 4.7.

    Returns
    -------
    rmin, rmax : float

    Raises
    ------
    ValueError  -- if parameters would produce rmin >= rmax.

    Notes
    -----
    modified_lj
    -----------
    The well minimum is exactly at r = r0.  rmin is fixed at 0.7 * r0
    (repulsive side).  The attractive tail behaves asymptotically as:

        U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m

    Setting U_attr(rmax) = t_tol_lj * |U_0| and solving:

        rmax = r0 * (n / ((n-m) * t_tol_lj))^(1/m)

    shifted_mie
    -----------
    Let xi = r - r0.  The singularity is at xi = 0 (r = r0).
    The Mie prefactor is C = n/(n-m) * (n/m)^(m/(n-m)).

    rmin is fixed at r0 + 0.7 * delta (symmetric with modified_lj).

    The attractive tail: U_attr ~ U_0 * C * (delta/xi)^m
    Setting U_attr(xi_max) = t_tol_mie and solving directly:

        xi_max = delta * (|U_0| * C / t_tol_mie)^(1/m)
        rmax   = r0 + xi_max
    """
    rmin = _table_rmin(potential, r0, delta)

    if potential == "modified_lj":
        # rmax: tail decay to t_tol_lj
        # U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m  => rmax = r0*(n/((n-m)*t_tol_lj))^(1/m)
        prefactor_m = n / (n - m)        # coefficient of the attractive (r0/r)^m term
        rmax = r0 * (prefactor_m / t_tol_lj) ** (1.0 / m)

    elif potential == "shifted_mie":
        if delta is None:
            raise ValueError("delta is required for shifted_mie bounds.")

        C = (n / (n - m)) * (n / m) ** (m / (n - m))

        # rmax: attractive tail decay to t_tol_mie directly
        # U_attr(xi) ~ U_0*C*(delta/xi)^m  => xi_max = delta*(|U_0|*C/t_tol_mie)^(1/m)
        xi_max = delta * (abs(U_0) * C / t_tol_mie) ** (1.0 / m)
        rmax_raw = r0 + xi_max
        # Sanity cap: rmax should not exceed half the box length,
        # and realistically the tail is negligible beyond ~5*delta from r0
        rmax_physical = r0 + 5.0 * delta
        if rmax_raw > rmax_physical:
            import warnings
            warnings.warn(
                f"Computed rmax ({rmax_raw:.3f}) exceeds physical cap "
                f"({rmax_physical:.3f}). Clamping. Consider increasing t_tol_mie.",
                RuntimeWarning
            )
            rmax = rmax_physical
        else:
            rmax = rmax_raw

    else:
        raise ValueError(f"Unknown potential: {potential!r}")

    if rmin >= rmax:
        raise ValueError(
            f"Computed rmin ({rmin:.4f}) >= rmax ({rmax:.4f}) for "
            f"potential={potential!r}, n={n}, m={m}, r0={r0}, delta={delta}.  "
            f"Check that n > m > 0 and tolerance parameters are reasonable."
        )

    return rmin, rmax


def _table_rmin(potential: str, r0: float, delta: float | None) -> float:
    """Repulsive-side table start: 0.7*r0 (modified_lj), r0 + 0.7*delta (shifted_mie)."""
    if potential == "modified_lj":
        return 0.7 * r0
    if potential == "shifted_mie":
        if delta is None:
            raise ValueError("delta is required for shifted_mie bounds.")
        return r0 + 0.7 * delta
    raise ValueError(f"Unknown potential: {potential!r}")


DEFAULT_TAIL_ENERGY_CUT = 0.1


def resolve_table_bounds(
    potential: str,
    U_0: float,
    n: float,
    m: float,
    r0: float,
    delta: float | None,
    *,
    N: int,
    density: float,
    t_tol_lj: float | None = None,
    tail_energy_cut: float | None = None,
    t_tol_mie: float = 4.7,
    rmax: float | None = None,
) -> dict:
    """
    Choose the pair-table cutoff used by ``run_simulation``.

    Cutoff modes (modified_lj):
      - dynamic (default): ``t_tol_lj = tail_energy_cut / U_0`` with
        ``tail_energy_cut = 0.1`` unless given. The attractive term at rmax
        then equals ``tail_energy_cut`` in absolute energy units (kT = 1),
        whatever U_0 is.
      - fixed tolerance: an explicit ``t_tol_lj`` (relative to U_0).
      - fixed cutoff: an explicit ``rmax``.
    ``tail_energy_cut`` conflicts with both ``t_tol_lj`` and ``rmax``.
    shifted_mie always uses ``t_tol_mie`` (or a fixed ``rmax``) and rejects
    ``tail_energy_cut``.

    For every mode, rmax must satisfy the minimum-image condition
    ``rmax < L/2`` with ``L = (N/density)^(1/3)``.

    Returns
    -------
    dict with keys ``rmin``, ``rmax``, ``rmax_fixed``, ``t_tol_lj``
    (tolerance actually used, None when rmax is fixed or for shifted_mie),
    ``tail_energy_cut`` (None unless dynamic), and ``L``.
    """
    if tail_energy_cut is not None and t_tol_lj is not None:
        raise ValueError("Give either tail_energy_cut or t_tol_lj, not both.")
    if tail_energy_cut is not None and rmax is not None:
        raise ValueError("Give either tail_energy_cut or a fixed rmax, not both.")
    if tail_energy_cut is not None and potential != "modified_lj":
        raise ValueError("tail_energy_cut is only defined for potential='modified_lj'.")

    L = (N / density) ** (1.0 / 3.0)
    t_tol_used = None
    cut_used = None
    rmax_fixed = rmax is not None

    if rmax_fixed:
        rmin = _table_rmin(potential, r0, delta)
        rmax = float(rmax)
        if rmax <= rmin:
            raise ValueError(
                f"Fixed rmax ({rmax:.4f}) must exceed rmin ({rmin:.4f})."
            )
    else:
        bounds_kw = {"t_tol_mie": t_tol_mie}
        if potential == "modified_lj":
            if t_tol_lj is not None:
                t_tol_used = float(t_tol_lj)
            else:
                cut_used = float(
                    DEFAULT_TAIL_ENERGY_CUT if tail_energy_cut is None else tail_energy_cut
                )
                if U_0 <= 0:
                    raise ValueError(
                        f"Dynamic cutoff needs U_0 > 0 (got {U_0}); "
                        "pass t_tol_lj or rmax instead."
                    )
                t_tol_used = cut_used / float(U_0)
            bounds_kw["t_tol_lj"] = t_tol_used
        rmin, rmax = _compute_table_bounds(
            potential, U_0, n, m, r0, delta, **bounds_kw
        )

    if rmax >= L / 2:
        raise ValueError(
            f"rmax={rmax:.4f} violates the minimum-image condition "
            f"(L/2={L / 2:.4f}, N={N}, density={density})."
        )

    return {
        "rmin": rmin,
        "rmax": rmax,
        "rmax_fixed": rmax_fixed,
        "t_tol_lj": t_tol_used,
        "tail_energy_cut": cut_used,
        "L": L,
    }


# ---------------------------------------------------------------------------
# Heyes-Melrose HS randomization (diameter units, contact distance sigma_HS)
# ---------------------------------------------------------------------------

def _hs_potential(r, rmin, rmax, dt, sigma=1.0):
    """Heyes-Melrose harmonic repulsion, contact distance ``sigma`` (default 1).

    With overlap ``delta = sigma - r`` and gamma = 1, ``F = delta / (2 dt)``
    moves each particle by ``delta/2`` per step, so the pair separates by
    ``delta`` in one step for any ``sigma``. The table runs over [0, sigma].
    """
    U = 1.0 / (4.0 * dt) * (sigma - r) ** 2
    F = 1.0 / (2.0 * dt) * (sigma - r)
    return U, F


def count_close_pairs(positions, L, r_cut):
    """
    Count pairs with separation ``< r_cut`` in a cubic periodic box.

    Parameters
    ----------
    positions : (N, 3) array-like
        Coordinates in a box of side ``L`` (any origin; wrapped here).
    L : float
        Box side length.
    r_cut : float
        Separation threshold.

    Returns
    -------
    n_pairs : int
        Number of unordered pairs i < j with minimum-image distance < r_cut.
    min_distance : float
        Smallest minimum-image pair distance.
    """
    from scipy.spatial import cKDTree

    x = np.mod(np.asarray(positions, dtype=np.float64), L)
    x[x >= L] -= L
    tree = cKDTree(x, boxsize=L)
    n = x.shape[0]
    n_ordered = tree.count_neighbors(tree, np.nextafter(float(r_cut), 0.0))
    d, _ = tree.query(x, k=2)
    return int((n_ordered - n) // 2), float(d[:, 1].min())


def _run_hs_randomization(
    nlist,
    group,
    integrator,
    *,
    dt_hs: float = 1e-4,
    t_rand: float = 10.0,
    kT: float = 1.0,
    seed: int = 42,
    gamma: float = 1.0,
    sigma_hs: float = 1.0,
) -> None:
    """Randomize the lattice with Heyes-Melrose HS. Does not dump a GSD.

    Call after ``hoomd.init.read_gsd`` and before creating the production
    pair table or Langevin integrator. Disables its own pair table and
    Brownian integrator before returning so the caller can switch potentials.

    Overdamped Brownian is required: the per-step displacement
    ``F*dt/gamma = delta/2`` removes an overlap of depth delta in one step.
    ``dt_hs`` defaults to 1e-4 because ``d_eff = sigma_HS - sqrt(pi * dt)``.
    The shortfall ``sqrt(pi * dt)`` (~0.018 at dt = 1e-4) depends only on
    dt and D = kT/gamma = 1, not on ``sigma_hs``, so a larger contact
    distance keeps the same dt with a smaller relative error.
    """
    n_steps = int(np.round(t_rand / dt_hs))
    if n_steps <= 0:
        print(f"Skipping HS randomization (t_rand={t_rand}, dt_hs={dt_hs})")
        return

    integrator.set_params(dt=dt_hs)
    hs = hoomd.md.pair.table(width=1000, nlist=nlist, name="hs")
    hs.pair_coeff.set(
        "A", "A",
        func=_hs_potential,
        rmin=0.0,
        rmax=float(sigma_hs),
        coeff=dict(dt=dt_hs, sigma=float(sigma_hs)),
    )
    bd = hoomd.md.integrate.brownian(group=group, kT=kT, seed=seed)
    bd.set_gamma("A", gamma=gamma)

    print(
        f"HS randomization: t_rand={t_rand:g} ({n_steps} steps), "
        f"dt_hs={dt_hs}, sigma_HS={sigma_hs:g}, brownian"
    )
    hoomd.run(n_steps)

    hs.disable()
    bd.disable()


_SPHERE_TYPE_SHAPE = [{'type': 'Sphere', 'diameter': 1.0}]


def _stamp_sphere_type_shapes(gsd_path, diameter=1.0):
    """Write GSD ``particles/type_shapes`` so OVITO locks Radius = diameter/2.

    ``particles/diameter = 1`` is the GSD default and is omitted from the
    file, so OVITO falls back to Standard radius. ``type_shapes`` is not
    the default empty dict and is actually stored.
    """
    shape = [{'type': 'Sphere', 'diameter': float(diameter)}]
    tmp_path = gsd_path + '.tmp'
    with gsd.hoomd.open(gsd_path, 'r') as src, gsd.hoomd.open(tmp_path, 'w') as dst:
        for src_frame in src:
            src_frame.particles.type_shapes = shape
            dst.append(src_frame)
    os.replace(tmp_path, gsd_path)


# ---------------------------------------------------------------------------
# Initial lattice
# ---------------------------------------------------------------------------

def generate_lattice_positions(N, L, rmin, offset=0.1, *, seed):
    """Shuffled simple-cubic lattice, in the order particles are created.

    Sites are the centers of ``n_side**3`` cells, with
    ``n_side = ceil(N**(1/3))``. The lattice is permuted with
    ``numpy.random.default_rng(seed)`` and the first ``N`` sites are kept.

    ``seed`` is the same integer passed to HOOMD's Brownian and Langevin
    integrators. Those integrators do not read this NumPy generator, so
    drawing the permutation does not advance their streams. A spawned
    stream is not used: this is the only NumPy generator in the run, and
    the permutation a given ``seed`` produces is then just
    ``default_rng(seed)``.

    Before 2026-10 the permutation used an unseeded ``default_rng()``.
    The same ``seed`` now yields the same order, and a different ``seed``
    yields a different order. Runs from before this change are not
    reproduced by repeating ``seed``, because that permutation was not
    saved.
    """
    min_dist = rmin + offset
    n_side = int(np.ceil(N ** (1 / 3)))
    spacing = L / n_side
    if spacing < min_dist:
        raise ValueError(
            f"Box too small for non-overlapping init: "
            f"grid spacing {spacing:.3f} < min_dist {min_dist:.3f}. "
            f"Increase box size (lower density) or reduce rmin."
        )
    # Build a simple cubic lattice
    coords = np.linspace(-L / 2 + spacing / 2, L / 2 - spacing / 2, n_side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    rng.shuffle(grid)
    return grid[:N]


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
    steps: int = 22_500_000,
    kT: float = 1.0,
    t_tol_lj: float | None = None,
    t_tol_mie: float = 4.7,
    init_offset: float = 0.1,
    device: str = "gpu",   # "cpu" also works on HOOMD 2.x
    seed: int = 42,
    plot: bool = True,
    rmax: float | None = None,
    tail_energy_cut: float | None = None,
    t_rand: float = 10.0,
    dt_hs: float = 1e-4,
    hs_sigma_follows_rmin: bool = True,
    transient_log_steps: int = 0,
    transient_log_period: int = 1,
    log_max_force: bool = False,
) -> dict:
    """
    Run a HOOMD simulation of N spheres with a selectable pair potential.

    Workflow: shuffled cubic lattice GSD -> in-memory Heyes-Melrose HS
    randomization (no dump) -> selected pair potential production run.
    ``DNA_assembly_*.gsd`` frame 0 is the post-HS configuration.

    ``potential_energy.csv`` is written every 5000 steps during production.
    After the HOOMD timestep the columns are ``potential_energy``,
    ``kinetic_energy``, ``temperature``, and ``pressure`` (names from the
    HOOMD 2.9.7 ``hoomd.analyze.log`` docstring). ``potential_energy`` stays
    in column 1, which ``plot_energy`` plots. ``transient_log_steps > 0``
    also writes ``transient_energy.csv`` over the start of production and
    splits ``hoomd.run`` only so that logger can be disabled. The Langevin
    update itself is unchanged: HOOMD 2.9.7 draws the noise from the seed,
    the particle, and the timestep, and the default (``transient_log_steps``
    0) is still a single ``hoomd.run``.

    Forward model
    -------------
    The defaults of this function, together with those of
    ``scattering.convert_to_SAXS_fft``, are the forward model shared by
    inverse design and mapping training (ground truth and BO): call both
    with defaults and pass only ``N``, ``density`` and the potential
    parameters. Changing a default changes the forward model.

      - modified_lj table potential; dynamic cutoff
        ``t_tol_lj = 0.1 / U_0`` (``tail_energy_cut``), no fixed rmax,
        ``rmax < L/2`` enforced
      - Heyes-Melrose HS randomization: ``t_rand = 10``, ``dt_hs = 1e-4``,
        contact distance ``max(1, rmin)``
      - initial lattice shuffled by ``numpy.random.default_rng(seed)``
        (``seed = 42``). Before 2026-10 this shuffle was unseeded, so the
        same ``seed`` did not repeat the configuration that enters the HS
        step. This NumPy generator is not HOOMD's, so the draw does not
        consume the Brownian or Langevin streams.
      - Langevin production: ``kT = 1``, gamma = 1, ``dt = 1e-3``,
        ``steps = 22_500_000`` (22,500 tau; one GSD frame every 50,000
        steps -> 450 frames). ``seed = 42`` also seeds the HS Brownian
        integrator and the Langevin thermostat
      - ``N`` comes from the caller (default 5000); each run states its N
        next to its target path

    Parameters
    ----------
    density : float
        Number density N/V in units of particles/σ³ (particle diameter σ=1).
        This is **not** volume fraction φ; φ = (π/6) × density for unit spheres.
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
    N, dt, steps, kT : see defaults (``steps`` was 15_000_000 before 2026-10)
    t_tol_lj : float, optional
        Fixed tail-energy tolerance for rmax (modified_lj): rmax is where the
        attractive tail falls to t_tol_lj * |U_0|.  Only used when given;
        conflicts with ``tail_energy_cut``.  Before 2026-10 the default was
        a fixed 0.02.
    tail_energy_cut : float, optional
        Dynamic cutoff (modified_lj), used when neither ``t_tol_lj`` nor
        ``rmax`` is given: ``t_tol_lj = tail_energy_cut / U_0``, so the
        attractive tail at rmax equals ``tail_energy_cut`` (kT units).
        Default 0.1.  Conflicts with ``t_tol_lj`` and ``rmax``.
    t_tol_mie : float
        Tail-energy tolerance for rmax (shifted_mie).  rmax is where the
        attractive tail falls to t_tol_mie.  Default 4.7.
    init_offset : float
        Added to rmin to set the minimum distance between particles during
        random initialization. Default 0.1.
    device : {"gpu","cpu"}
        HOOMD context device mode.
    seed : int
        Seed for the lattice shuffle (``numpy.random.default_rng``), the
        HS Brownian integrator, and the Langevin thermostat. The shuffle
        used to be unseeded, so a fixed ``seed`` did not fix the initial
        configuration. It does now. HOOMD's generators are separate and
        are not advanced by the NumPy draw.
    plot : bool
        Controls whether potential and energy plots are generated.
    rmax : float, optional
        If set, use this value as the pair-potential table cutoff instead of
        the analytically derived rmax from ``_compute_table_bounds``.
        Every cutoff mode must satisfy ``rmax < L/2`` (minimum image),
        checked before HOOMD starts; see ``resolve_table_bounds``.
    t_rand : float
        HS randomization time in reduced units (D = kT/γ = 1). Default 10.
    dt_hs : float
        HS timestep. Default 1e-4 (locked by d_eff = σ_HS - sqrt(π dt)).
    hs_sigma_follows_rmin : bool
        If True (default), the HS contact distance is ``σ_HS = max(1, rmin)``
        so no pair starts production inside the table's ``r < rmin`` region
        (where HOOMD's table gives zero energy and force). The HS step only
        serves to randomize positions, so its contact distance is free.
        False gives ``σ_HS = 1`` (particle diameter), the behaviour before
        2026-10.
    transient_log_steps : int
        If positive, also write ``transient_energy.csv`` for this many
        steps at the start of production, then disable that logger.
        Default 0 leaves production as one ``hoomd.run`` and does not
        write the file. The HOOMD timestep column includes the HS steps:
        production step 0 is ``round(t_rand / dt_hs)`` (100000 at the
        defaults). Rows are the timesteps in
        ``[t0, t0 + transient_log_steps)`` on which the period lands.
        The endpoint itself is not in this file.
    transient_log_period : int
        Period of ``transient_energy.csv``, in steps. Default 1 records
        every step in the window. A larger period subsamples it. Used
        only when ``transient_log_steps > 0``.
    log_max_force : bool
        If True, add a ``max_force`` column to both logs: the maximum
        per-particle net-force magnitude, from a Python callback on the
        particle proxy. Default False. Each logged row then loops over
        all particles, so leave this off for a full-length run.

    Returns
    -------
    dict with keys:
        gsd_path, energy_csv, transient_energy_csv, rmin, rmax,
        rmax_fixed, t_tol_lj, tail_energy_cut, L, table_width, potential,
        t_rand, dt_hs, sigma_hs, n_pairs_below_rmin, min_pair_distance
        (``transient_energy_csv`` is None when ``transient_log_steps`` is
        0. ``t_tol_lj`` is the tolerance actually used, None when rmax is
        fixed or for shifted_mie; ``tail_energy_cut`` is None unless the
        dynamic cutoff was used; ``L`` is the cubic box length;
        ``n_pairs_below_rmin`` and ``min_pair_distance`` are measured on
        the configuration that starts production, i.e. GSD frame 0,
        under periodic boundaries.)

    Raises
    ------
    ValueError
        If an unknown potential name is given, if ``"shifted_mie"`` is
        selected without providing ``delta``, if the derived rmin >= rmax,
        if cutoff arguments conflict, if rmax >= L/2, or if the transient
        log arguments are out of range.
    """
    if potential not in _POTENTIALS:
        raise ValueError(
            f"Unknown potential {potential!r}. "
            f"Choose from: {list(_POTENTIALS)}"
        )
    pot_fn, needs_delta = _POTENTIALS[potential]
    if needs_delta and delta is None:
        raise ValueError("`delta` must be provided when potential='shifted_mie'.")
    if transient_log_steps < 0 or transient_log_period < 1:
        raise ValueError(
            "transient_log_steps must be >= 0 and transient_log_period "
            f"must be >= 1 (got steps={transient_log_steps}, "
            f"period={transient_log_period})."
        )
    if transient_log_steps > steps:
        raise ValueError(
            f"transient_log_steps ({transient_log_steps}) is longer than "
            f"the production run ({steps})."
        )

    os.makedirs(outdir, exist_ok=True)

    # --- Analytically derived table bounds ---
    bounds = resolve_table_bounds(
        potential, U_0, n, m, r0, delta,
        N=N, density=density, t_tol_lj=t_tol_lj,
        tail_energy_cut=tail_energy_cut, t_tol_mie=t_tol_mie, rmax=rmax,
    )
    rmin, rmax = bounds["rmin"], bounds["rmax"]
    if bounds["rmax_fixed"]:
        cut_desc = "rmax fixed"
    elif potential == "modified_lj" and bounds["tail_energy_cut"] is not None:
        cut_desc = (f"dynamic: t_tol_lj={bounds['t_tol_lj']:.6g} "
                    f"= {bounds['tail_energy_cut']:g}/U_0")
    elif potential == "modified_lj":
        cut_desc = f"t_tol_lj={bounds['t_tol_lj']}"
    else:
        cut_desc = f"t_tol_mie={t_tol_mie}"
    print(f"Table bounds: rmin={rmin:.4f}, rmax={rmax:.4f} "
          f"({cut_desc}; L/2={bounds['L'] / 2:.4f})")

    # --- HOOMD context ---
    mode_flag = "--mode=gpu" if device == "gpu" else "--mode=cpu"
    hoomd.context.initialize(mode_flag)

    # --- Derived params & box ---
    # density is number density ρ = N/V in particles/σ³ (σ = particle diameter).
    L = bounds["L"]  # cubic box side length (N / density)^(1/3)

    # Lattice order is fixed by seed. See generate_lattice_positions.
    positions = np.ascontiguousarray(
        generate_lattice_positions(N, L, rmin, offset=init_offset, seed=seed),
        dtype=np.float32,
    )

    # HOOMD 2.9.7: make_snapshot + read_snapshot does not load lattice coords
    # correctly for large N (GSD frame 0 ends with 5016/5017 at origin).
    # Write init via gsd.hoomd.Frame, then hoomd.init.read_gsd().
    init_gsd_path = os.path.join(outdir, "_init_lattice.gsd")
    init_frame = gsd.hoomd.Frame()
    init_frame.particles.N = N
    init_frame.particles.types = ["A"]
    init_frame.particles.typeid = np.zeros(N, dtype=np.uint32)
    init_frame.particles.position = positions
    init_frame.particles.diameter = np.ones(N, dtype=np.float32)
    init_frame.particles.type_shapes = _SPHERE_TYPE_SHAPE
    init_frame.configuration.box = [L, L, L, 0, 0, 0]
    with gsd.hoomd.open(init_gsd_path, "w") as traj:
        traj.append(init_frame)

    system = hoomd.init.read_gsd(init_gsd_path)

    width = 1000
    nl = hoomd.md.nlist.cell()
    group_all = hoomd.group.all()
    integrator = hoomd.md.integrate.mode_standard(dt=dt_hs)

    # --- In-memory HS randomization (no dump) ---
    sigma_hs = max(1.0, rmin) if hs_sigma_follows_rmin else 1.0
    _run_hs_randomization(
        nl, group_all, integrator,
        dt_hs=dt_hs, t_rand=t_rand, kT=kT, seed=seed, sigma_hs=sigma_hs,
    )

    # --- Production frame 0: pairs inside the table's r < rmin region ---
    # Read through the particle proxy: under numpy 2 the HOOMD 2.9.7
    # snapshot position buffer comes back with every row equal to x.
    positions_post_hs = np.array([p.position for p in system.particles])
    n_pairs_below_rmin, min_pair_distance = count_close_pairs(
        positions_post_hs, L, rmin
    )
    print(f"Post-HS pairs with r < rmin={rmin:.4f}: {n_pairs_below_rmin} "
          f"(min pair distance {min_pair_distance:.4f}, sigma_HS={sigma_hs:g})")

    # --- Production pair potential via table ---
    table = hoomd.md.pair.table(width=width, nlist=nl, name="prod")

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

    # --- Production integrator ---
    integrator.set_params(dt=dt)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=seed)
    langevin.set_gamma('A', gamma=1.0)

    # --- Outputs: GSD + energy CSV (after HS, so frame 0 is post-HS) ---
    ts = time.localtime()
    timestamp = f"{ts.tm_year:02d}{ts.tm_mon:02d}{ts.tm_mday:02d}{ts.tm_hour:02d}{ts.tm_min:02d}{ts.tm_sec:02d}"
    gsd_path = os.path.join(outdir, f"DNA_assembly_{timestamp}.gsd")
    gsd_dump = hoomd.dump.gsd(
        filename=gsd_path, period=50000, group=group_all, overwrite=True
    )

    energy_csv = os.path.join(outdir, "potential_energy.csv")
    _attach_thermo_log(energy_csv, period=5000, log_max_force=log_max_force,
                       system=system)

    # --- Run ---
    # The dense logger is removed by ending its segment. HOOMD 2.9.7
    # Langevin noise depends on (seed, particle, timestep), not on how
    # many times hoomd.run was called, so the split does not change the
    # trajectory. transient_log_steps == 0 keeps the single hoomd.run.
    print(f"Running {steps} steps with {N} spheres at number density "
          f"{density:.6g} particles/σ³ using potential='{potential}' "
          f"(dt={dt}, after HS t_rand={t_rand:g})")
    transient_csv = None
    if transient_log_steps:
        transient_csv = os.path.join(outdir, "transient_energy.csv")
        dense_log = _attach_thermo_log(
            transient_csv, period=transient_log_period,
            log_max_force=log_max_force, system=system,
        )
        print(f"Transient log: first {transient_log_steps} production steps "
              f"every {transient_log_period} -> {transient_csv}")
        hoomd.run(transient_log_steps)
        dense_log.disable()
        remaining = steps - transient_log_steps
        if remaining:
            hoomd.run(remaining)
    else:
        hoomd.run(steps)
    gsd_dump.disable()
    _stamp_sphere_type_shapes(gsd_path)

    # --- Generate Potential Energy Plot ---
    if plot:
        plot_energy(energy_csv, os.path.join(outdir, "potential_energy_plot.png"))
        print("Simulation complete. Plots Generated.")
    else:
        print("Simulation complete.")

    return {
        "gsd_path"    : gsd_path,
        "energy_csv"  : energy_csv,
        "transient_energy_csv": transient_csv,
        "rmin"        : rmin,
        "rmax"        : rmax,
        "rmax_fixed"  : bounds["rmax_fixed"],
        "t_tol_lj"    : bounds["t_tol_lj"],
        "tail_energy_cut": bounds["tail_energy_cut"],
        "L"           : L,
        "table_width" : width,
        "potential"   : potential,
        "t_rand"      : t_rand,
        "dt_hs"       : dt_hs,
        "sigma_hs"    : sigma_hs,
        "n_pairs_below_rmin": n_pairs_below_rmin,
        "min_pair_distance" : min_pair_distance,
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


# Quantities hoomd.analyze.log accepts on HOOMD 2.9.7 (analyze.py docstring).
# potential_energy stays first: plot_energy reads column 1 of the TSV
# (column 0 is the timestep HOOMD writes itself).
_THERMO_LOG_QUANTITIES = (
    "potential_energy",
    "kinetic_energy",
    "temperature",
    "pressure",
)


def _energy_log_quantities(log_max_force=False):
    """Column names after ``timestep`` in the production energy logs."""
    quantities = list(_THERMO_LOG_QUANTITIES)
    if log_max_force:
        quantities.append("max_force")
    return quantities


def _max_net_force(system):
    """Largest per-particle net-force magnitude.

    Read through the particle proxy. Under numpy 2, HOOMD 2.9.7
    ``take_snapshot()`` returns a wrong position buffer; this path does
    not use that buffer.
    """
    f2_max = 0.0
    for p in system.particles:
        fx, fy, fz = p.net_force
        f2 = fx * fx + fy * fy + fz * fz
        if f2 > f2_max:
            f2_max = f2
    return f2_max ** 0.5


def _attach_thermo_log(filename, period, log_max_force, system):
    """Tab-separated HOOMD log of the thermo columns, optionally max force.

    ``max_force`` is a Python callback, not a built-in quantity. It runs
    on every row this logger writes. Leave ``log_max_force`` false on a
    long run: each row walks all particles from Python.
    """
    logger = hoomd.analyze.log(
        filename=filename,
        quantities=_energy_log_quantities(log_max_force),
        period=period,
        overwrite=True,
    )
    if log_max_force:
        def max_force(_timestep, system=system):
            return _max_net_force(system)
        logger.register_callback("max_force", max_force)
    return logger


def _energy_log_xy(csv_path):
    """Timestep and potential-energy columns plotted by ``plot_energy``.

    The first 6 data rows are skipped, as before. HOOMD writes the
    timestep in column 0 and the logged quantities after it.
    ``potential_energy`` is the first logged quantity, so it stays in
    column 1 when kinetic energy, temperature, and pressure are added
    to the right. A two-column log from before that change still plots.
    """
    data = pd.read_csv(csv_path, delimiter='\t').values
    return data[6:, 0], data[6:, 1]


def plot_energy(csv_path, out_png):
    time, energy = _energy_log_xy(csv_path)
    plt.figure(figsize=(6, 4))
    plt.plot(time, energy)
    plt.xlabel("Time"); plt.ylabel("Potential Energy")
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=600); plt.close()
