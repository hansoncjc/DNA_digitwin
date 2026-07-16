# datasets.py
import numpy as np
from pathlib import Path


class ExperimentalParams:
    """
    Holds experimental inputs (flat, explicit) and lightweight helpers.

    Stock / mixing (units)
    ----------------------
    C_stock : float
        Silica stock concentration (mg/mL). Default: 60.0
    V_stock : float
        Volume of stock solution added (mL). Default: 0.333
    V_total : float
        Final total volume (mL). Default: 10.0
    rho_si : float
        Silica density (g/cm^3). Default: 2.2   (used by Dataset.rho_N; converted to mg/mL)

    Solution chemistry / sequence
    -----------------------------
    C_NaCl : float
        NaCl concentration (mM). Default: 10.0
    C_chol : float
        DNA–cholesterol count per siNP (molecules/particle). Default: 100.0
    b_bridge : float
        Bridge concentration ratio respective to Chol concentration(dimensionless), where C_bridge = b_bridge * C_chol.
        Default: 0.5 (tunable).
    L_poly : float
        Length of ssDNA spacer (nt). Default: 10.0
    L_bridge : float or None
        Length of DNA bridge (nt). No default; set when used in mappings.
    L_HBP : float
        Length of hybridizing base pairs (bp). Default: 18.0

    Geometry
    --------
    d_si : float
        Silica NP diameter (nm). Default: 24.6
    t_b : float
        Lipid bilayer thickness (nm). Default: 3.6

    Optional context
    ----------------
    temperature_C : float or None
    extra : dict
    """

    def __init__(self,
                 C_stock=60.0, V_stock=0.333, V_total=10.0, rho_si=2.2,
                 C_NaCl=10.0, C_chol=100.0, b_bridge=0.5,
                 L_poly=10.0, L_bridge=None, L_HBP=18.0,
                 d_si=24.6, t_b=3.6, dna_coverage = None,
                 temperature_C=None, extra=None):
        # Stock / mixing
        self.C_stock = C_stock
        self.V_stock = V_stock
        self.V_total = V_total
        self.rho_si  = rho_si

        # Solution chemistry / sequence
        self.C_NaCl   = C_NaCl
        self.C_chol   = C_chol
        self.b_bridge = b_bridge  # C_bridge = b_bridge * C_chol
        self.L_poly   = L_poly
        self.L_bridge = L_bridge  # may be None; required by r0 mapping
        self.L_HBP    = L_HBP

        # Geometry
        self.d_si = d_si
        self.t_b  = t_b

        # Optional context
        self.dna_coverage  = dna_coverage
        self.temperature_C = temperature_C
        self.extra = {} if extra is None else extra

    @property
    def C_bridge(self):
        """
        Generative bridge concentration (molecules/siNP).

        Returns
        -------
        float
            C_bridge = b_bridge * C_chol
        """
        return self.b_bridge * self.C_chol


class SimulationParams:
    """
    Holds simulation parameters (shared and sample-specific).
    """
    def __init__(self, density=None, n=12.0, m=6.0, U0=None, r0=None,
                 N=None, steps=None, dt=None, kT=None):
        self.density = density
        self.n = n
        self.m = m
        self.U0 = U0
        self.r0 = r0
        self.N = N
        self.steps = steps
        self.dt = dt
        self.kT = kT


class Dataset:
    """
    Container for one dataset: experimental params, simulation params, data path, and weight.

    Parameters
    ----------
    id : str
        Short identifier, e.g., "d1".
    exp_path : str or pathlib.Path
        Path to experimental I(q) .npy (shape (N,2) as [q, I]).
    exp : ExperimentalParams or None
        If None, defaults are used.
    sim : SimulationParams or None
        If None, defaults are used (n=12, m=6).
    weight : float
        Loss weight for this dataset in global aggregation. Default: 1.0
    out_dir : str or pathlib.Path or None
        Optional output directory for artifacts.

    Methods
    -------
    load_exp_curve(trim_tail=0) -> np.ndarray
        Lazy-load the experimental [q, I] and optionally drop last `trim_tail` rows.
    rho_N(alpha=1.0) -> float
        Number density = alpha * theoretical_base, with theoretical_base computed here.
    r0_sigma(k=0.76, LC_ss=0.63, LC_ds=0.34) -> float
        Map experimental geometry/sequence to r0/σ using your formula.
    U0_from_gaussian(A, mu_c, mu_b, sigma_c, sigma_b, K_s) -> float
        Separable Gaussian in (C_chol, b_bridge) times a linear salt prefactor
        S = 1 + K_s * C_NaCl (K_s is the salt-response slope k_s, 1/mM).
    """

    def __init__(self, id, exp_path, exp=None, sim=None, weight=1.0, out_dir=None, datatype="sq"):
        self.id = id
        self.exp_path = Path(exp_path)
        self.exp = exp if exp is not None else ExperimentalParams()
        self.sim = sim if sim is not None else SimulationParams()
        self.weight = float(weight)
        self.out_dir = out_dir
        self.datatype = datatype
        self._exp_curve_cache = None  # in-memory cache for the .npy

    def load_exp_curve(self, trim_tail=200):
        """
        Load the experimental curve from .npy (first two columns as [q, I]) and cache it.

        Parameters
        ----------
        trim_tail : int, default 200
            Number of rows to drop from the end after loading.

        Returns
        -------
        numpy.ndarray
            Array of shape (N', 2): [q, I].
        """
        if trim_tail < 0:
            raise ValueError("trim_tail must be >= 0")
        if self._exp_curve_cache is None:
            if not self.exp_path.exists():
                raise FileNotFoundError(f"Experimental data not found: {self.exp_path}")
            arr = np.load(self.exp_path)
            if arr.ndim != 2 or arr.shape[1] < 2:
                raise ValueError(f"Expected (N, 2+) array in {self.exp_path}, got {arr.shape}")
            self._exp_curve_cache = arr[:, :2].astype(np.float64, copy=False)
        arr = self._exp_curve_cache
        if trim_tail and arr.shape[0] > trim_tail:
            return arr[:-trim_tail]
        return arr

    def rho_N(self, alpha=3.0):
        """
        Compute simulation number density as alpha * theoretical_base.

        Theoretical base is computed here (not in ExperimentalParams):
            base = (6/pi) * (C_stock / (rho_si * 1000)) * (V_stock / V_total)

        Parameters
        ----------
        alpha : float, default 3.0
            Global coefficient to be fitted in Stage 1. Default value is 3.0. 

        Returns
        -------
        float
            density = alpha * base
        """
        rho_si_mg_per_ml = self.exp.rho_si * 1000.0  # g/cm^3 → mg/mL
        base = (6.0 / np.pi) * (
            (self.exp.C_stock / rho_si_mg_per_ml) * (self.exp.V_stock / self.exp.V_total)
        )
        return float(alpha) * float(base)

    def r0_sigma(self, k=0.76, LC_ss=0.63, LC_ds=0.34):
        """
        Compute r0 in units of σ from experimental geometry/sequence.

        Formula
        -------
            r0/σ = 1 + k * ( 2*(t_b + LC_ss*L_poly) + LC_ds*(2*L_HBP + L_bridge) ) / d_si

        Defaults
        --------
        k     = 0.76        (dimensionless; fitted in Stage 2)
        LC_ss = 0.63 nm/nt  (contour length per nt of ssDNA)
        LC_ds = 0.34 nm/bp  (contour length per bp of dsDNA)

        Returns
        -------
        float
            r0 in σ units.

        Raises
        ------
        ValueError
            If L_bridge is None (required by the formula).
        """
        if self.exp.L_bridge is None:
            raise ValueError("L_bridge is required to compute r0 but is None.")
        num = 2.0 * (self.exp.t_b + LC_ss * self.exp.L_poly) + LC_ds * (2.0 * self.exp.L_HBP + self.exp.L_bridge)
        return 1.0 + float(k) * (num / self.exp.d_si)

    def U0_from_gaussian(self, A=2.0, mu_c=100.0, mu_b=0.5,
                         sigma_c=10.0, sigma_b=0.2, K_s=0.05):
        """
        Compute U0 from a separable Gaussian in (C_chol, b_bridge) multiplied
        by a monotonic linear salt prefactor.

        Mapping Function
        -------
            S  = 1 + K_s * C_NaCl                         # salt prefactor
            U0 = A * S * exp( - ((C_chol   - mu_c)^2) / (2*sigma_c^2)
                              - ((b_bridge - mu_b)^2) / (2*sigma_b^2) )

        ``K_s`` is the linear salt-response slope ``k_s`` (units 1/mM). Salt is
        decoupled from cholesterol: the Gaussian uses the raw ``C_chol``.
        At ``C_NaCl = 0``, ``S = 1`` (no salt keeps the full cholesterol-driven
        well; sufficient C_chol can still aggregate). Increasing salt only
        raises U0, never collapses it, so there is no re-entrant high-salt
        dispersion (the failure mode of the previous ``C_chol_eff``-inside-
        Gaussian form; see Salt_Concentration.md).

        Defaults
        --------
            A       = 2.0
            mu_c    = 100.0
            mu_b    = 0.5
            sigma_c = 10.0
            sigma_b = 0.2
            K_s     = 0.05

        Parameters
        ----------
        A : float
            Zero-salt amplitude of U0.
        mu_c : float
            Center for the cholesterol count `C_chol` (molecules/particle).
        mu_b : float
            Center for the bridge-loading ratio `b_bridge` (dimensionless).
        sigma_c : float
            Std. dev. for the C_chol term (same unit as C_chol); must be > 0.
        sigma_b : float
            Std. dev. for bridge-loading term (dimensionless b_bridge);
            must be > 0.
        K_s : float
            Linear salt-response slope k_s (1/mM); fractional increase of U0
            per mM NaCl. Should be >= 0 to keep the prefactor monotonic.

        Returns
        -------
        float
            U0 value.
        """
        if sigma_c <= 0 or sigma_b <= 0:
            raise ValueError("sigma_c and sigma_b must be > 0.")

        C_chol   = float(self.exp.C_chol)
        C_NaCl   = float(self.exp.C_NaCl)
        b_bridge = float(self.exp.b_bridge)

        salt_gate = 1.0 + float(K_s) * C_NaCl

        term_c = ((C_chol   - float(mu_c)) ** 2) / (2.0 * (float(sigma_c) ** 2))
        term_b = ((b_bridge - float(mu_b)) ** 2) / (2.0 * (float(sigma_b) ** 2))

        return float(A) * salt_gate * np.exp(- (term_c + term_b))

    def _autofill_sim_from_default(self,
                                    *,
                                    alpha=3.0,
                                    k=0.76,
                                    A=2.0, mu_c=100.0, sigma_c=10.0,
                                    mu_b=0.5, sigma_b=0.2,
                                    K_s=0.05):
        """
        Fill self.sim.{density,r0,U0} from mappings if any of them is None.
        Does NOT overwrite fields that are already set.
        """
        if getattr(self.sim, "density", None) is None:
            self.sim.density = self.rho_N(alpha=alpha)
        if getattr(self.sim, "r0", None) is None:
            self.sim.r0 = self.r0_sigma(k=k)
        if getattr(self.sim, "U0", None) is None:
            self.sim.U0 = self.U0_from_gaussian(A=A, mu_c=mu_c, sigma_c=sigma_c,
                                                mu_b=mu_b, sigma_b=sigma_b,
                                                K_s=K_s)
    @classmethod
    def from_dict(cls, d):
        """
        Expected keys:
        {
            "id": "itr0",
            "exp_path": "path/to/curve.npy",
            "experimental": { ... ExperimentalParams ... },
            "simulation":   { ... SimulationParams ... }   # optional/partial
            "mapping": {                                  # optional overrides for auto-fill
            "alpha": 3.0,
            "k": 0.76,
            "A": 2.0, "mu_c": 100.0, "sigma_c": 10.0,
            "mu_b": 0.5, "sigma_b": 0.2,
            "K_s": 0.05
            },
            "weight": 1.0,
            "out_dir": "Results/itr0"
        }
        """
        exp = ExperimentalParams(**d.get("experimental", {}))

        sim_dict = d.get("simulation", {})
        sim = SimulationParams(**sim_dict) if isinstance(sim_dict, dict) else SimulationParams()

        ds = cls(
            id=d["id"],
            exp_path=d["exp_path"],
            exp=exp,
            sim=sim,
            weight=d.get("weight", 1.0),
            out_dir=d.get("out_dir"),
            datatype=d.get("datatype", "sq"),
        )

        # Auto-fill any missing sim fields, allowing per-dataset overrides via "mapping"
        ds._autofill_sim_from_default(**d.get("mapping", {}))
        return ds
