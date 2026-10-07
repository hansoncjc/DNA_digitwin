"""Pin the forward model: the defaults of run_simulation and convert_to_SAXS_fft.

A failure here means the forward model shared by inverse design and mapping
training has changed; update the docstrings and README together with it.
"""
import inspect

import simulation
from scattering import convert_to_SAXS_fft
from simulation import run_simulation


def _defaults(fn):
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_simulation_defaults():
    d = _defaults(run_simulation)
    assert d["potential"] == "modified_lj"
    assert d["N"] == 5000
    assert d["dt"] == 1e-3
    assert d["steps"] == 22_500_000
    assert d["kT"] == 1.0
    assert d["seed"] == 42
    assert d["t_init"] == 10.0
    assert d["rmin_k"] == 0.65
    assert d["transient_log_steps"] == 0
    assert d["transient_log_period"] == 1
    assert d["transient_log_init_steps"] == 0
    assert d["log_max_force"] is False
    assert "t_rand" not in d and "dt_hs" not in d and "hs_sigma_follows_rmin" not in d
    assert d["rmax"] is None
    assert d["t_tol_lj"] is None
    assert d["tail_energy_cut"] is None
    assert simulation.DEFAULT_TAIL_ENERGY_CUT == 0.1
    assert simulation.DEFAULT_RMIN_K == 0.65


def test_saxsfft_defaults():
    d = _defaults(convert_to_SAXS_fft)
    assert d["N_grid"] == 600
    assert d["frames"] == "last:100"
    assert d["step"] == 5
    assert d["trim"] == slice(3, -3)
    assert d["particle_diameter"] == 24.6
