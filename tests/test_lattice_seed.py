"""Lattice permutation is fixed by seed. No HOOMD."""
import numpy as np
import pytest

from simulation import generate_lattice_positions


def test_same_seed_same_order():
    a = generate_lattice_positions(30, 12.0, 1.2, seed=42)
    generate_lattice_positions(30, 12.0, 1.2, seed=99)
    b = generate_lattice_positions(30, 12.0, 1.2, seed=42)
    assert np.array_equal(a, b)


def test_different_seed_different_order():
    a = generate_lattice_positions(30, 12.0, 1.2, seed=42)
    b = generate_lattice_positions(30, 12.0, 1.2, seed=43)
    assert not np.array_equal(a, b)


def test_seed_ignores_the_legacy_global_rng():
    np.random.seed(0)
    a = generate_lattice_positions(30, 12.0, 1.2, seed=7)
    np.random.seed(1)
    b = generate_lattice_positions(30, 12.0, 1.2, seed=7)
    assert np.array_equal(a, b)


def test_returned_sites_belong_to_the_cubic_lattice():
    N, L, rmin, offset = 30, 12.0, 1.2, 0.1
    pos = generate_lattice_positions(N, L, rmin, offset=offset, seed=1)
    assert pos.shape == (N, 3)
    n_side = int(np.ceil(N ** (1 / 3)))
    spacing = L / n_side
    coords = np.linspace(-L / 2 + spacing / 2, L / 2 - spacing / 2, n_side)
    grid = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
    keys = {tuple(np.round(row, 10)) for row in grid}
    assert all(tuple(np.round(row, 10)) in keys for row in pos)


def test_box_too_small_raises():
    with pytest.raises(ValueError, match="Box too small"):
        generate_lattice_positions(1000, 2.0, 1.0, seed=0)
