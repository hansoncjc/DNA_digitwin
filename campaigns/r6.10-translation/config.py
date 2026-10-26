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

# x-space rerun (advisor meeting 2026-10-21): minimize mu(f(x)) directly over
# x = (L_bridge, C_anchor, C_NaCl). The box only makes the grid finite; it
# must enclose f^-1(R6 box) in L_bridge and C_anchor at every salt (checked at
# run time), so it adds no constraint beyond nonnegativity and the R6 box.
# Salt 0-100 mM is the scope of the salt direction (the display range above).
X_NAMES = ('L_bridge', 'C_anchor', 'C_NaCl')
X_BOX = {'L_bridge': (0., 250.), 'C_anchor': (0., 800.), 'C_NaCl': (0., DISPLAY_SALT_MAX)}
X_GRID = (161, 161, 21)  # L_bridge, C_anchor, C_NaCl; salt step 5 mM includes CHECK_SALT
SLICE_GRID = 241  # per axis, figure-data slices only
FLAT_TOL = 1e-6  # loss units: slice minima within this of the 3D minimum are equivalent
SAME_MU_TOL = 1e-6  # loss units: same minimum as the first translation
SAME_X_TOL = 1e-3  # distance in x normalized by X_BOX widths
OUT_X = OUT_ROOT / 'translation_x'
