# datasets.py
import numpy as np
from pathlib import Path

# Floor on C_chol_eff denominator (same units as C_NaCl, K_s*C_chol) to avoid 0/0 when C_NaCl=C_chol=0.
_CHOL_EFF_EPS = 1e-6

class ExperimentalParams:
    """
    Holds experimental inputs (flat, explicit) and lightweight helpers.

    Stock / mixing (units), NOT GONNA USE THESE RIGHT NOW
    ----------------------
    C_stock_au : float
        Gold stock concentration (M). Default: 1E-9
    V_stock_au : float
        Volume of stock solution added (uL). Default: 800
    V_total : float
        Final total volume (uL). Default: 1000.0
    rho_au : float
        Gold density (g/cm^3). Default: 19.32

    Solution chemistry / sequence
    -----------------------------
    C_NaCl : float
        NaCl concentration (mM). Default: 10.0
    C_AuNP : float
        AuNP concentration (M). Default: 8E-10

    Geometry
    --------
    r_au : float
        Gold NP radius (nm). Default: 10.0
    t_c : float
        Citrate layer thickness (nm). Default: 0.5

    Optional context
    ----------------
    temperature_C : float or None
    extra : dict
    """

    def __init__(self,
                 C_NaCl=10.0, C_AuNP=8E-10,
                 r_au=10.0, t_c=0.5,
                 temperature_C=None, extra=None):

        # Solution chemistry / sequence
        self.C_NaCl   = C_NaCl
        self.C_AuNP   = C_AuNP

        # Geometry
        self.r_au = r_au
        self.t_c  = t_c

        # Optional context
        self.temperature_C = temperature_C
        self.extra = {} if extra is None else extra


class SimulationParams:
    """
    Holds simulation parameters (shared and sample-specific).
    """
    def __init__(self, phi=0.01, A=None, debye_length=None,
                 z=None, N=None, steps=None, dt=None, kT=None):
        self.phi = phi
        self.A = A
        self.debye_length = debye_length
        self.z = z
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
        If None, defaults are used.
    weight : float
        Loss weight for this dataset in global aggregation. Default: 1.0
    out_dir : str or pathlib.Path or None
        Optional output directory for artifacts.

    Methods
    -------
    load_exp_curve(trim_tail=0) -> np.ndarray
        Lazy-load the experimental [q, I] and optionally drop last `trim_tail` rows.
    deb_length(w1=1.0, w2=0.0) -> float
        Compute debye length from C_NaCl using the formula: w1 * base + w2, where base is the theoretical debye length.
    z(w3=-0.01, w4=0.025) -> float
        Compute zeta potential from C_NaCl using the formula: z = w3 * C_NaCl + w4
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

    def deb_length(self, w1 = 1.0, w2 = 0.0):
        """
        Compute simulation debye length as w1 * theoretical_base + w2.

        Theoretical base is computed here (not in ExperimentalParams):
            base = (((80.0 * 8.85E-12 * 8.314 * 298.15)**0.5) / 96485) * (2 * C_NaCl)**(-0.5)

        Parameters
        ----------
        w1 : float, default 1.0
            Global coefficient to be fitted in Stage 1. Default value is 1.0.
        w2 : float, default 0.0
            Global coefficient to be fitted in Stage 1. Default value is 0.0.

        Returns
        -------
        float
            debye length = w1 * base + w2
        """
        base = (((80.0 * 8.854E-12 * 8.314 * 298)**0.5) / 96485) * (2 * self.exp.C_NaCl)**(-0.5)
        return float(w1) * float(base) + float(w2)

    def z(self, w3 = -0.001, w4 = 0.045):
        """
        Compute zeta potential in units of V from NaCl concentration.

        Formula
        -------
            z = w3 * C_NaCl + w4  

        Defaults
        --------
        w3 = -0.01       (V/mM; fitted in Stage 1) CHANGE TO 1
        w4 = 0.025       (V; fitted in Stage 1) CHANGE TO 0

        Returns
        -------
        float
            zeta potential in V.
        """
        return float(w3) * float(self.exp.C_NaCl) + float(w4)


    def _autofill_sim_from_default(self, *, w1=1.0, w2=0.0, w3=-0.01, w4=0.025):
        """
        Fill self.sim.{debye_length,zeta_potential} from mappings if any of them is None.
        Does NOT overwrite fields that are already set.
        """
        if getattr(self.sim, "debye_length", None) is None:
            self.sim.debye_length = self.deb_length(w1=w1, w2=w2)
        if getattr(self.sim, "zeta_potential", None) is None:
            self.sim.zeta_potential = self.z(w3=w3, w4=w4)
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
            "w1": 1.0,
            "w2": 0.0,
            "w3": -0.01,
            "w4": 0.025
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
