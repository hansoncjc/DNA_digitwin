"""R6.10 decisions agreed with the author; all scientific inputs are bundled."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUTS = HERE / 'inputs'
RUNS = ('r6.4', 'r6.5')
OUT_ROOT = Path('/gscratch/zeelab/hanson/codes/DNA_lipid_silica/testing/inverse/r6/r6.10-translate')
BOUNDS = {'U0': (0.5, 8.0), 'r0': (2.0, 3.5), 'n': (5.5, 15.0), 'm_rel': (0.0, 1.0)}
D_SI = 24.6
SALTS = (0., 20., 40., 60.)
DISPLAY_SALT_MAX = 100.
CHECK_SALT = 20.
SEED = 42  # Existing forward-model default; one seed, no repeat statistics.
GRID_SIZE = 121
MULTISTART = 12
POSTERIOR_ATOL = 1e-7
POSTERIOR_RTOL = 1e-5
