"""
R4: train the physics-based mapping on six measured SAXS curves (2026-10-16).

Data: six room-temperature curves of the Huat collection, chosen as a cross
around L_bridge 20, C_anchor 100, NaCl 10 mM:
  loading:       C_anchor 60, 100, 140   (L_bridge 20, 10 mM)
  salt:          NaCl 10, 20, 40 mM      (L_bridge 20, C_anchor 100)
  bridge length: L_bridge 20, 80         (C_anchor 100, 10 mM)
The raw I(q) files ([q, I, sigma]) are copied unchanged into ``data/`` and
read from there (md5 below). The objective converts them to S(q)
(datatype "iq"): ``trim_tail`` drops the last 200 I(q) rows (q above about
0.075 A^-1), then ``scattering.extract_exp_sq`` divides by the polydisperse
sphere form factor ``formfactors/sasmodels_sphere_fit.txt``, normalized in
q 0.02-0.03 A^-1 (the value fixed in bo.py and parallel/worker.py).

Left out of the collection (2026-10-16, all bridged curves checked with this
extraction): C_anchor 20 (the R3.2 box maps it below U0 = 0.5 over part of
the box at every salt level, and at 20-40 mM it has no peak near q 0.013);
NaCl 0 (no peak near q 0.013; the first-peak rule picks a feature near
q 0.03, close to the first form-factor minimum); L_bridge 40 (smoothed
s_max - 1 = 0.44, below dispersed_delta). Every condition here maps inside
check_mapped_params over the whole box.

Settings decided with the author on 2026-10-16:
- Search box and initial point: R3.2 (``train_config.PARAM_CFG``), K_s free;
  the three salt levels at C_anchor 100 separate A and K_s.
- Loss: shift_rmse defaults except ``intensity_scale = "none"``: the measured
  S(q) does not level off at 1 in the asymptote band (0.085, 0.100), and
  after trim_tail = 200 it has no points there. ``dispersed_delta`` is the
  default 0.5, written out so it can be changed here.
- One BO point per evaluation (no batch BO); the six conditions run as six
  GPU jobs at once (gpu-a40). At most 100 successful evaluations.
- Forward model: as R3.2 (function defaults, N 5000, density 0.005, CPU S(q)).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "r3.2-physics-validate"))

from train_config import DENSITY, N, PARAM_CFG  # noqa: E402,F401

DATA_DIR = HERE / "data"

# id, file (in DATA_DIR), L_bridge (bp), C_anchor (molecules/particle),
# C_NaCl (mM), md5 of the raw I(q) file.
CONDITIONS = [
    {"id": "d0", "file": "chol_060_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 60.0,
     "C_NaCl": 10.0, "md5": "4b57b359dd33244a10bf2af62c25ccd6"},
    {"id": "d1", "file": "chol_100_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 100.0,
     "C_NaCl": 10.0, "md5": "29b0822ee86f07dfaae71e9401c22723"},
    {"id": "d2", "file": "chol_140_nacl_10_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 140.0,
     "C_NaCl": 10.0, "md5": "944f4192fdc2f1d3dc9b762e14cc8925"},
    {"id": "d3", "file": "chol_100_nacl_20_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 100.0,
     "C_NaCl": 20.0, "md5": "19f1456fb9038bdaa39a499daf5199e3"},
    {"id": "d4", "file": "chol_100_nacl_40_bridge_20.npy", "L_bridge": 20.0, "C_anchor": 100.0,
     "C_NaCl": 40.0, "md5": "cd183883615a43ab120ebfd7845c5eb5"},
    {"id": "d5", "file": "chol_100_nacl_10_bridge_80.npy", "L_bridge": 80.0, "C_anchor": 100.0,
     "C_NaCl": 10.0, "md5": "d6c9900d437b2acfad52c2b80c219b24"},
]

DATATYPE = "iq"
TRIM_TAIL = 200          # I(q) rows dropped from the end before extraction
EXTRACT_Q_WINDOW = (0.02, 0.03)   # extract_exp_sq normalization (fixed in bo.py / worker.py)
FORMFACTOR = "formfactors/sasmodels_sphere_fit.txt"   # relative to the repo root

METRIC = "shift_rmse"
METRIC_KWARGS = {"intensity_scale": "none", "dispersed_delta": 0.5}

BO_SEED = 0       # torch seed of run_bo; simulations use the run_simulation default seed (42)
MAX_EVALS = 100   # successful evaluations in total, resumes included
