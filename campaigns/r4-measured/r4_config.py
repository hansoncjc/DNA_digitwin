"""
R4: train the physics-based mapping on six measured SAXS curves (2026-10-16).

Data: the 10 mM NaCl subset of the Huat room-temperature collection (as in the
vault note 2026-05-08 BO_Huat_10mMSubset): L_bridge 20 at C_anchor 20, 60,
100, 140 and C_anchor 100 at L_bridge 40, 80. The measured I(q) files
([q, I, sigma]) are converted to S(q) inside the objective (datatype "iq"):
``trim_tail`` drops the last 200 I(q) rows (q above about 0.075 A^-1), then
``scattering.extract_exp_sq`` divides by the polydisperse sphere form factor
``formfactors/sasmodels_sphere_fit.txt``, normalized in q 0.02-0.03 A^-1
(the value fixed in bo.py and parallel/worker.py).

Settings decided with the author on 2026-10-16:
- Search box and initial point: R3.2 (``train_config.PARAM_CFG``), K_s free.
  At 10 mM, A and K_s enter only as A (1 + 10 K_s).
- C_anchor = 20 is kept (R3.1 left it out). About half of the box maps it
  below U0 = 0.5; such points fail check_mapped_params, are logged and not
  fed to the GP.
- Loss: shift_rmse defaults except
  ``intensity_scale = "none"``: the measured S(q) does not level off at 1 in
  the asymptote band (0.085, 0.100), and after trim_tail = 200 it has no
  points there;
  ``dispersed_delta = 0.25``: with this extraction the first peaks of the six
  curves reach s_max - 1 = 0.44-0.76 (C_anchor 100, L_bridge 40 is 0.44 and
  would be called dispersed at the default 0.5); flat simulated curves lie at
  0.002-0.089 (task 0b), so 0.25 is in that gap.
  R3 and R6 keep the defaults.
- One BO point per evaluation (no batch BO); the six conditions run as six
  GPU jobs at once (gpu-a40). At most 100 successful evaluations.
- Forward model: as R3.2 (function defaults, N 5000, density 0.005, CPU S(q)).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r3.2-physics-validate"))

from train_config import DENSITY, N, PARAM_CFG  # noqa: E402,F401

C_NACL = 10.0   # mM, all six curves

# id, file (in the data folder), L_bridge (bp), C_anchor (molecules/particle),
# md5 of the I(q) file as downloaded on 2026-10-16.
CONDITIONS = [
    {"id": "d0", "file": "chol_020_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 20.0,
     "md5": "2acc951739f38ec24478cb4e7c44fb74"},
    {"id": "d1", "file": "chol_060_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 60.0,
     "md5": "4b57b359dd33244a10bf2af62c25ccd6"},
    {"id": "d2", "file": "chol_100_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 100.0,
     "md5": "29b0822ee86f07dfaae71e9401c22723"},
    {"id": "d3", "file": "chol_140_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 140.0,
     "md5": "944f4192fdc2f1d3dc9b762e14cc8925"},
    {"id": "d4", "file": "chol_100_nacl_10_bridge_40.npy", "L_bridge": 40.0, "C_anchor": 100.0,
     "md5": "14479e91412f3c9c8714795e4b489da3"},
    {"id": "d5", "file": "chol_100_nacl_10_bridge_80.npy", "L_bridge": 80.0, "C_anchor": 100.0,
     "md5": "d6c9900d437b2acfad52c2b80c219b24"},
]

DATATYPE = "iq"
TRIM_TAIL = 200          # I(q) rows dropped from the end before extraction
EXTRACT_Q_WINDOW = (0.02, 0.03)   # extract_exp_sq normalization (fixed in bo.py / worker.py)
FORMFACTOR = "formfactors/sasmodels_sphere_fit.txt"   # relative to the repo root

METRIC = "shift_rmse"
METRIC_KWARGS = {"intensity_scale": "none", "dispersed_delta": 0.25}

BO_SEED = 0       # torch seed of run_bo; simulations use the run_simulation default seed (42)
MAX_EVALS = 100   # successful evaluations in total, resumes included
