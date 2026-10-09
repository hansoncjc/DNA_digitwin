"""
Simulation of DNA-mediated SiNP using HOOMD-blue.
The pair potential is the n-m Lennard-Jones-like table potential
"modified_lj", the only value the `potential` argument of
run_simulation() accepts.

The simple-cubic lattice is shuffled with ``numpy.random.default_rng(seed)``
before it is written. Before 2026-10 that shuffle was unseeded, so a fixed
``seed`` did not fix the configuration that enters initialization.
After the lattice GSD is read, run_simulation() integrates for ``t_init``
(default 10) with the production Langevin integrator — same dt, gamma,
kT and seed — on the repulsive branch of the potential, cut at
the well minimum. That segment is not dumped. DNA_assembly_*.gsd starts
at the configuration that enters production, and the number of pairs
closer than the table's rmin in that configuration is logged.

Table bounds (rmin, rmax) are derived analytically from the potential
parameters rather than being hard-coded, so they remain valid across
parameter sweeps. modified_lj uses rmin = rmin_k * r0 with rmin_k = 0.5
by default (0.65 on 2026-10-07, 0.7 before). At 0.65 and 0.6, pairs at the
soft corner of the search box (U_0 = 0.5, n = 5.5, m = 4) entered r < rmin
during production; at 0.5 none did (task 0a, 2026-10-08).
See _compute_table_bounds() for the derivation and resolve_table_bounds()
for how run_simulation() picks the cutoff (default: dynamic,
t_tol_lj = tail_energy_cut / U_0 with tail_energy_cut = 0.1).
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


# ---------------------------------------------------------------------------
# Physics-derived table bounds
# ---------------------------------------------------------------------------

DEFAULT_RMIN_K = 0.5


def _compute_table_bounds(
    U_0: float,
    n: float,
    m: float,
    r0: float,
    *,
    t_tol_lj: float = 0.02,
    rmin_k: float = DEFAULT_RMIN_K,
) -> tuple[float, float]:
    """
    Compute (rmin, rmax) analytically from the potential parameters.

    The strategy is:
      - rmin: place it on the repulsive side of the well minimum, far
              enough from any singularity to keep forces finite and
              table values well-defined.
      - rmax: solve for the separation at which the attractive tail
              decays to the tolerance t_tol.  This guarantees the
              cutoff never truncates the well prematurely, regardless of
              how (n, m, r0) vary across a sweep.

    Parameters
    ----------
    U_0 : float   -- energy scale / well depth
    n, m : float  -- repulsive and attractive exponents (n > m > 0)
    r0 : float    -- reference length; the well minimum is at r0
    t_tol_lj : float
        Tail-energy tolerance used as rmax cutoff for modified_lj.
        rmax is where U_attr = t_tol_lj * (n/(n-m)).  Default 0.02.

    Returns
    -------
    rmin, rmax : float

    Raises
    ------
    ValueError  -- if parameters would produce rmin >= rmax.

    Notes
    -----
    The well minimum is exactly at r = r0.  rmin is ``rmin_k * r0``
    (default 0.5; 0.65 on 2026-10-07, 0.7 before), on the repulsive side.  The
    attractive tail behaves asymptotically as:

        U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m

    Setting U_attr(rmax) = t_tol_lj * |U_0| and solving:

        rmax = r0 * (n / ((n-m) * t_tol_lj))^(1/m)
    """
    rmin = _table_rmin(r0, rmin_k)

    # rmax: tail decay to t_tol_lj
    # U_attr(r) ~ U_0 * n/(n-m) * (r0/r)^m  => rmax = r0*(n/((n-m)*t_tol_lj))^(1/m)
    prefactor_m = n / (n - m)        # coefficient of the attractive (r0/r)^m term
    rmax = r0 * (prefactor_m / t_tol_lj) ** (1.0 / m)

    if rmin >= rmax:
        raise ValueError(
            f"Computed rmin ({rmin:.4f}) >= rmax ({rmax:.4f}) for "
            f"n={n}, m={m}, r0={r0}.  "
            f"Check that n > m > 0 and tolerance parameters are reasonable."
        )

    return rmin, rmax


def _table_rmin(r0: float, rmin_k: float = DEFAULT_RMIN_K) -> float:
    """Repulsive-side table start.

    ``rmin_k * r0`` (default 0.5). ``rmin_k`` must lie in (0, 1) so the
    table starts before the well at r0.
    """
    if not 0.0 < float(rmin_k) < 1.0:
        raise ValueError(
            f"rmin_k must lie in (0, 1) for modified_lj (got {rmin_k})."
        )
    return float(rmin_k) * float(r0)


DEFAULT_TAIL_ENERGY_CUT = 0.1


def resolve_table_bounds(
    U_0: float,
    n: float,
    m: float,
    r0: float,
    *,
    N: int,
    density: float,
    t_tol_lj: float | None = None,
    tail_energy_cut: float | None = None,
    rmax: float | None = None,
    rmin_k: float = DEFAULT_RMIN_K,
) -> dict:
    """
    Choose the pair-table cutoff used by ``run_simulation``.

    Cutoff modes:
      - dynamic (default): ``t_tol_lj = tail_energy_cut / U_0`` with
        ``tail_energy_cut = 0.1`` unless given. The attractive term at rmax
        then equals ``tail_energy_cut`` in absolute energy units (kT = 1),
        whatever U_0 is.
      - fixed tolerance: an explicit ``t_tol_lj`` (relative to U_0).
      - fixed cutoff: an explicit ``rmax``.
    ``tail_energy_cut`` conflicts with both ``t_tol_lj`` and ``rmax``.

    For every mode, rmax must satisfy the minimum-image condition
    ``rmax < L/2`` with ``L = (N/density)^(1/3)``.

    Returns
    -------
    dict with keys ``rmin``, ``rmax``, ``rmax_fixed``, ``t_tol_lj``
    (tolerance actually used, None when rmax is fixed),
    ``tail_energy_cut`` (None unless dynamic), and ``L``.
    """
    if tail_energy_cut is not None and t_tol_lj is not None:
        raise ValueError("Give either tail_energy_cut or t_tol_lj, not both.")
    if tail_energy_cut is not None and rmax is not None:
        raise ValueError("Give either tail_energy_cut or a fixed rmax, not both.")

    L = (N / density) ** (1.0 / 3.0)
    t_tol_used = None
    cut_used = None
    rmax_fixed = rmax is not None

    if rmax_fixed:
        rmin = _table_rmin(r0, rmin_k)
        rmax = float(rmax)
        if rmax <= rmin:
            raise ValueError(
                f"Fixed rmax ({rmax:.4f}) must exceed rmin ({rmin:.4f})."
            )
    else:
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
        rmin, rmax = _compute_table_bounds(
            U_0, n, m, r0, t_tol_lj=t_tol_used, rmin_k=rmin_k
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
# Repulsive branch used to leave the lattice (WCA-style cut at the well)
# ---------------------------------------------------------------------------

def _repulsive_branch(r, rmin, rmax, U_0, n, m, r0):
    """Repulsive branch of ``modified_LJ``, cut at the well minimum r0.

    For r < r0, U_rep = U + U_0 and the force is the full-potential
    force. For r >= r0, both are 0. The initialization table runs over
    [rmin, r0], so HOOMD's own r >= rmax rule agrees with this cut.
    HOOMD 2.9.7 evaluates the table function once per grid point (a
    scalar); arrays are accepted for tests.
    """
    r_well = float(r0)
    U, F = modified_LJ(r, rmin, rmax, U_0, n, m, r0)
    r_arr = np.asarray(r, dtype=float)
    U = np.asarray(U, dtype=float) + float(U_0)
    F = np.asarray(F, dtype=float)
    outside = r_arr >= r_well
    U = np.where(outside, 0.0, U)
    F = np.where(outside, 0.0, F)
    if np.ndim(r) == 0:
        return float(U), float(F)
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


# HOOMD 2.9.7 phase=-1 starts on the timestep where the analyzer is
# attached and then repeats every period. The Python docstring's
# "(step + phase) % period == 0" does not describe that binary: a
# positive phase equal to the current step modulo the period is
# deferred to the next period. Attach loggers with phase=-1 at the
# timestep that should be their first row.
_LOG_PHASE_FROM_NOW = -1


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

    ``seed`` is the same integer passed to HOOMD's Langevin integrator.
    That integrator does not read this NumPy generator, so drawing the
    permutation does not advance its stream. A spawned stream is not
    used: this is the only NumPy generator in the run, and the
    permutation a given ``seed`` produces is then just
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
    N: int = 5000,
    dt: float = 1e-3,
    steps: int = 22_500_000,
    kT: float = 1.0,
    t_tol_lj: float | None = None,
    init_offset: float = 0.1,
    device: str = "gpu",   # "cpu" also works on HOOMD 2.x
    seed: int = 42,
    plot: bool = True,
    rmax: float | None = None,
    tail_energy_cut: float | None = None,
    rmin_k: float = DEFAULT_RMIN_K,
    t_init: float = 10.0,
    transient_log_steps: int = 0,
    transient_log_period: int = 1,
    transient_log_init_steps: int = 0,
    log_max_force: bool = False,
) -> dict:
    """
    Run a HOOMD simulation of N spheres with the modified_lj pair potential.

    Workflow: shuffled cubic lattice GSD -> Langevin on the repulsive
    branch of the potential, cut at the well minimum (no dump)
    -> the same Langevin on the full potential. ``DNA_assembly_*.gsd``
    frame 0 is the configuration that starts production.

    One ``hoomd.md.pair.table`` is used for both segments. Switching is
    ``pair_coeff.set`` with the full potential and the production rmax.
    HOOMD 2.9.7's ``hoomd.run`` calls ``Integrator.update_forces`` (which
    rebuilds the table via ``setTable``) and then ``nlist.update_rcut``
    (which reads that table's rmax while the table is enabled) before
    the first step of the new segment. The neighbor-list cutoff therefore
    grows from r_well to rmax together with the force. A second table is
    not used: an enabled one would add its energy on top of this one.

    ``potential_energy.csv`` is written every 5000 production steps.
    After the HOOMD timestep the columns are ``potential_energy``,
    ``kinetic_energy``, ``temperature``, and ``pressure`` (names from the
    HOOMD 2.9.7 ``hoomd.analyze.log`` docstring). ``potential_energy`` stays
    in column 1, which ``plot_energy`` plots. The GSD dump (period 50000)
    and this energy log are attached at timestep ``n_init`` with
    ``phase=-1``, so on HOOMD 2.9.7 their first sample is production
    step 0 and the later samples are production steps 50000, 100000, ...
    and 5000, 10000, .... The HOOMD clock already includes initialization:
    production step 0 is ``round(t_init / dt)`` (10000 at the defaults;
    it was 100000 when initialization was a Heyes-Melrose segment at
    dt = 1e-4).

    ``transient_log_steps > 0`` writes ``transient_energy.csv`` over the
    start of production. ``transient_log_init_steps > 0`` prepends that
    many steps from the end of initialization to the same file, so the
    switch can be read across consecutive rows. Default 0 for both leaves
    initialization unlogged and production as one ``hoomd.run`` after the
    initialization segment. The logger uses ``phase=-1`` and is attached
    at the first timestep it should record. With the default period of 1
    that is every step from ``n_init - transient_log_init_steps`` through
    ``n_init + transient_log_steps - 1``. The Langevin update itself is
    unchanged by the split: HOOMD 2.9.7 draws the noise from the seed,
    the particle, and the timestep.

    With period 1, timestep ``n_init`` is still the repulsive branch on
    the end-of-initialization positions. ``hoomd.run`` installs the new
    table and neighbor-list cutoff before the step loop, but the analyzer
    runs before the integrator recomputes forces, so that row is the
    repulsive energy. Timestep ``n_init + 1`` is the first full-potential
    row. ``potential_energy.csv`` has the same one-step lag: its first
    row, at production step 0, is the repulsive branch.

    Forward model
    -------------
    The defaults of this function, together with those of
    ``scattering.convert_to_SAXS_fft``, are the forward model shared by
    inverse design and mapping training (ground truth and BO): call both
    with defaults and pass only ``N``, ``density`` and the potential
    parameters. Changing a default changes the forward model.

      - modified_lj table potential; dynamic cutoff
        ``t_tol_lj = 0.1 / U_0`` (``tail_energy_cut``), no fixed rmax,
        ``rmax < L/2`` enforced; ``rmin = rmin_k * r0`` with ``rmin_k = 0.5``
      - repulsive-branch initialization: ``t_init = 10`` with the
        production Langevin (``dt = 1e-3``, gamma = 1, ``kT = 1``), cut
        at the well minimum
      - initial lattice shuffled by ``numpy.random.default_rng(seed)``
        (``seed = 42``). Before 2026-10 this shuffle was unseeded, so the
        same ``seed`` did not repeat the configuration that enters
        initialization. This NumPy generator is not HOOMD's, so the draw
        does not consume the Langevin stream.
      - Langevin production: ``kT = 1``, gamma = 1, ``dt = 1e-3``,
        ``steps = 22_500_000`` (22,500 tau; one GSD frame every 50,000
        production steps -> 450 frames). ``seed = 42`` seeds that
        Langevin for both segments
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
        Reference length: equilibrium distance scale; well minimum is at r = r0.
    n, m : float
        Repulsive and attractive exponents (n > m > 0).
    outdir : str
        Directory to write artifacts (gsd, csv).
    potential : {"modified_lj"}
        Pair potential.  Only ``"modified_lj"`` (the default) is accepted;
        it is recorded in the returned dict.
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
    init_offset : float
        Added to rmin when checking that the initial lattice spacing is
        wide enough (``min_dist = rmin + init_offset``). Default 0.1.
    device : {"gpu","cpu"}
        HOOMD context device mode.
    seed : int
        Seed for the lattice shuffle (``numpy.random.default_rng``) and
        the Langevin thermostat, which covers initialization and
        production. The shuffle used to be unseeded, so a fixed ``seed``
        did not fix the initial configuration. It does now. HOOMD's
        generator is separate and is not advanced by the NumPy draw.
    plot : bool
        Controls whether potential and energy plots are generated.
    rmax : float, optional
        If set, use this value as the pair-potential table cutoff instead of
        the analytically derived rmax from ``_compute_table_bounds``.
        Every cutoff mode must satisfy ``rmax < L/2`` (minimum image),
        checked before HOOMD starts; see ``resolve_table_bounds``.
    rmin_k : float
        ``rmin = rmin_k * r0``. Default 0.5 (0.65 on
        2026-10-07, 0.7 before). Must lie in (0, 1).
    t_init : float
        Repulsive-branch Langevin time, in the same units as production
        (D = kT/γ = 1). Default 10. The timestep is ``dt``, not a
        separate initialization step. ``t_init = 0`` skips the segment.
        The step count is ``round(t_init / dt)``.
    transient_log_steps : int
        If positive, write ``transient_energy.csv`` for this many steps
        at the start of production, then disable that logger. Default 0
        leaves production as one ``hoomd.run`` after initialization and
        does not, by itself, write the file. With the default period of 1
        the rows are timesteps ``[n_init, n_init + transient_log_steps)``.
        The endpoint itself is not in this file.
    transient_log_period : int
        Period of ``transient_energy.csv``, in steps, counted from the
        timestep where that logger is attached. Default 1 records every
        step in the window.
    transient_log_init_steps : int
        If positive, the same ``transient_energy.csv`` also covers this
        many steps at the end of initialization (still on the repulsive
        branch). Default 0 leaves initialization out of the file, which
        is the previous logging behaviour aside from the timestep offset.
        Must be <= ``round(t_init / dt)``.
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
        t_init, n_init_steps, rmin_k, r_well, n_pairs_below_rmin,
        min_pair_distance
        (``transient_energy_csv`` is None when both transient windows are
        0. ``t_tol_lj`` is the tolerance actually used, None when rmax is
        fixed; ``tail_energy_cut`` is None unless the
        dynamic cutoff was used; ``L`` is the cubic box length;
        ``n_init_steps`` is the HOOMD timestep of production step 0;
        ``r_well`` is the cut (r0);
        ``n_pairs_below_rmin`` and ``min_pair_distance`` are measured on
        the configuration that starts production, i.e. GSD frame 0,
        under periodic boundaries.)

    Raises
    ------
    ValueError
        If ``potential`` is not ``"modified_lj"``, if the derived rmin >= rmax,
        if the well minimum is not strictly inside (rmin, rmax), if cutoff
        arguments conflict, if rmax >= L/2, if ``rmin_k`` is outside (0, 1),
        or if the transient log arguments are out of range.
    """
    if potential != "modified_lj":
        raise ValueError(
            f"Unknown potential {potential!r}. Choose from: ['modified_lj']"
        )
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
    if t_init < 0 or dt <= 0:
        raise ValueError(
            f"t_init must be >= 0 and dt must be > 0 (got t_init={t_init}, dt={dt})."
        )
    n_init = int(np.round(float(t_init) / float(dt)))
    if transient_log_init_steps < 0 or transient_log_init_steps > n_init:
        raise ValueError(
            f"transient_log_init_steps ({transient_log_init_steps}) must lie "
            f"in [0, {n_init}] (t_init={t_init}, dt={dt})."
        )

    os.makedirs(outdir, exist_ok=True)

    # --- Analytically derived table bounds ---
    bounds = resolve_table_bounds(
        U_0, n, m, r0,
        N=N, density=density, t_tol_lj=t_tol_lj,
        tail_energy_cut=tail_energy_cut, rmax=rmax,
        rmin_k=rmin_k,
    )
    rmin, rmax = bounds["rmin"], bounds["rmax"]
    if bounds["rmax_fixed"]:
        cut_desc = "rmax fixed"
    elif bounds["tail_energy_cut"] is not None:
        cut_desc = (f"dynamic: t_tol_lj={bounds['t_tol_lj']:.6g} "
                    f"= {bounds['tail_energy_cut']:g}/U_0")
    else:
        cut_desc = f"t_tol_lj={bounds['t_tol_lj']}"
    r_well = float(r0)
    if not (rmin < r_well < rmax):
        raise ValueError(
            f"Repulsive-branch cutoff r_well={r_well:.4f} must lie strictly "
            f"between rmin={rmin:.4f} and rmax={rmax:.4f}."
        )
    k_desc = f", rmin_k={rmin_k:g}"
    print(f"Table bounds: rmin={rmin:.4f}, rmax={rmax:.4f}, "
          f"r_well={r_well:.4f} ({cut_desc}{k_desc}; L/2={bounds['L'] / 2:.4f})")

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
    # Same Langevin for initialization and production: dt, gamma, kT, seed.
    hoomd.md.integrate.mode_standard(dt=dt)
    langevin = hoomd.md.integrate.langevin(group=group_all, kT=kT, seed=seed)
    langevin.set_gamma('A', gamma=1.0)

    # One table. Coefficients are replaced at the switch; the next
    # hoomd.run rebuilds the table and the neighbor-list cutoff together.
    table = hoomd.md.pair.table(width=width, nlist=nl, name="pair")
    coeff = dict(U_0=U_0, n=n, m=m, r0=r0)

    if plot:
        out_png = os.path.join(outdir, "potential_plot.png")
        plot_pair_potential(rmin, rmax, width, U_0, n, m, r0, out_png,
                            modified_LJ)

    if n_init:
        table.pair_coeff.set(
            'A', 'A',
            rmin=rmin, rmax=float(r_well),
            func=_repulsive_branch,
            coeff=dict(coeff),
        )
        n_quiet = n_init - int(transient_log_init_steps)
        print(f"Repulsive-branch initialization: t_init={t_init:g} "
              f"({n_init} steps), dt={dt}, cut at r_well={r_well:.4f}, langevin")
        if n_quiet:
            hoomd.run(n_quiet)
    else:
        print(f"Skipping repulsive-branch initialization (t_init={t_init:g})")

    transient_csv = None
    dense_log = None
    if transient_log_init_steps or transient_log_steps:
        transient_csv = os.path.join(outdir, "transient_energy.csv")
        dense_log = _attach_thermo_log(
            transient_csv, period=transient_log_period,
            log_max_force=log_max_force, system=system,
            phase=_LOG_PHASE_FROM_NOW,
        )
    if transient_log_init_steps:
        print(f"Transient log: last {transient_log_init_steps} initialization "
              f"steps every {transient_log_period} -> {transient_csv}")
        hoomd.run(int(transient_log_init_steps))

    # Production frame 0. Read through the particle proxy: under numpy 2
    # the HOOMD 2.9.7 snapshot position buffer is wrong.
    positions_ready = np.array([p.position for p in system.particles])
    n_pairs_below_rmin, min_pair_distance = count_close_pairs(
        positions_ready, L, rmin
    )
    print(f"Pairs with r < rmin={rmin:.4f} at the start of production: "
          f"{n_pairs_below_rmin} (min pair distance {min_pair_distance:.4f})")

    table.pair_coeff.set(
        'A', 'A',
        rmin=rmin, rmax=rmax,
        func=modified_LJ,
        coeff=coeff,
    )

    # Attached at timestep n_init. phase=-1 makes frame 0 and the first
    # energy row that timestep, then every period after it.
    gsd_period = 50000
    energy_period = 5000
    ts = time.localtime()
    timestamp = f"{ts.tm_year:02d}{ts.tm_mon:02d}{ts.tm_mday:02d}{ts.tm_hour:02d}{ts.tm_min:02d}{ts.tm_sec:02d}"
    gsd_path = os.path.join(outdir, f"DNA_assembly_{timestamp}.gsd")
    gsd_dump = hoomd.dump.gsd(
        filename=gsd_path, period=gsd_period, group=group_all, overwrite=True,
        phase=_LOG_PHASE_FROM_NOW,
    )

    energy_csv = os.path.join(outdir, "potential_energy.csv")
    _attach_thermo_log(energy_csv, period=energy_period, log_max_force=log_max_force,
                       system=system, phase=_LOG_PHASE_FROM_NOW)

    # The dense logger is removed by ending its segment. HOOMD 2.9.7
    # Langevin noise depends on (seed, particle, timestep), not on how
    # many times hoomd.run was called, so the split does not change the
    # trajectory. transient_log_steps == 0 keeps production as one hoomd.run.
    print(f"Running {steps} steps with {N} spheres at number density "
          f"{density:.6g} particles/σ³ using potential='{potential}' "
          f"(dt={dt}, production step 0 at timestep {n_init})")
    if transient_log_steps:
        print(f"Transient log: first {transient_log_steps} production steps "
              f"every {transient_log_period} -> {transient_csv}")
        hoomd.run(transient_log_steps)
    if dense_log is not None:
        dense_log.disable()
    remaining = steps - transient_log_steps
    if remaining:
        hoomd.run(remaining)
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
        "t_init"      : t_init,
        "n_init_steps": n_init,
        "rmin_k"      : float(rmin_k),
        "r_well"      : r_well,
        "n_pairs_below_rmin": n_pairs_below_rmin,
        "min_pair_distance" : min_pair_distance,
    }


# ---------------------------------------------------------------------------
# Plotting Block from original codebase; could be moved to a separate file
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd


def plot_pair_potential(rmin, rmax, width, U_0, n, m, r0, out_png,
                        potential_fn):
    """Plot U(r) for any potential that follows the HOOMD table-function API."""
    r_vals = np.linspace(rmin, rmax, width)
    U, _ = potential_fn(r_vals, rmin, rmax, U_0, n, m, r0)
    label = f"n={n}, m={m}, r0={r0}, U0={U_0}"
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


def _attach_thermo_log(filename, period, log_max_force, system, phase=0):
    """Tab-separated HOOMD log of the thermo columns, optionally max force.

    ``max_force`` is a Python callback, not a built-in quantity. It runs
    on every row this logger writes. Leave ``log_max_force`` false on a
    long run: each row walks all particles from Python. ``phase=-1``
    starts the log on the timestep where it is attached.
    """
    logger = hoomd.analyze.log(
        filename=filename,
        quantities=_energy_log_quantities(log_max_force),
        period=period,
        overwrite=True,
        phase=int(phase),
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
