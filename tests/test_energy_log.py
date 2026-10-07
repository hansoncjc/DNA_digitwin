"""Energy-log columns. The HOOMD write itself is not run here."""
import numpy as np
import pytest

from simulation import _energy_log_quantities, _energy_log_xy, plot_energy, run_simulation


def test_potential_energy_stays_the_first_logged_quantity():
    assert _energy_log_quantities(False) == [
        "potential_energy",
        "kinetic_energy",
        "temperature",
        "pressure",
    ]
    with_force = _energy_log_quantities(True)
    assert with_force[0] == "potential_energy"
    assert with_force[-1] == "max_force"


def _write_log(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_reader_uses_potential_energy_when_later_columns_exist(tmp_path):
    path = tmp_path / "potential_energy.csv"
    _write_log(
        path,
        "timestep\tpotential_energy\tkinetic_energy\ttemperature\tpressure",
        [f"{1000 + i}\t{-i}\t{50 + i}\t1.0\t0.1" for i in range(10)],
    )
    time, energy = _energy_log_xy(str(path))
    assert np.allclose(time, [1006, 1007, 1008, 1009])
    assert np.allclose(energy, [-6, -7, -8, -9])


def test_reader_accepts_the_two_column_log(tmp_path):
    path = tmp_path / "potential_energy.csv"
    _write_log(
        path,
        "timestep\tpotential_energy",
        [f"{i}\t{i * 10}" for i in range(8)],
    )
    time, energy = _energy_log_xy(str(path))
    assert np.allclose(time, [6, 7])
    assert np.allclose(energy, [60, 70])


def test_plot_energy_ignores_added_columns(tmp_path):
    csv_path = tmp_path / "potential_energy.csv"
    _write_log(
        csv_path,
        "timestep\tpotential_energy\tkinetic_energy\ttemperature\tpressure",
        [f"{i}\t{i}\t999\t1\t0" for i in range(10)],
    )
    out = tmp_path / "potential_energy_plot.png"
    plot_energy(str(csv_path), str(out))
    assert out.is_file() and out.stat().st_size > 0


def test_transient_log_steps_cannot_exceed_the_run(tmp_path):
    with pytest.raises(ValueError, match="transient_log_steps"):
        run_simulation(
            0.005, 3.0, 2.5, 12, 6, str(tmp_path),
            N=8, steps=10, plot=False, transient_log_steps=11,
        )


def test_transient_log_period_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="transient_log_period"):
        run_simulation(
            0.005, 3.0, 2.5, 12, 6, str(tmp_path),
            steps=10, plot=False, transient_log_period=0,
        )
