# Inverse-design targets

Generation scripts and settings for the target S(q) curves used by
`inverse.py` campaigns (filled by the target-regeneration task). Targets use
the forward model's saxs-fft settings (`N_grid = 600`, `trim = slice(6, -6)`);
`inverse.py` checks the `N_grid` and records each target's md5 at the first
start of a run.

## Targets (2026-10-12)

| Directory | Use | Structure | N | L_box (σ) | q1 (Å⁻¹) |
|---|---|---|---|---|---|
| `fcc_rho0.10_D45.7_pad2.2/` | R6.4 | FCC crystallite, ρ* = 0.10, D = 45.7 σ | 5017 | 100.54 | 0.01289 |
| `fcc_rho0.05_D57.6_pad1.738/` | R6.5 | FCC crystallite, ρ* = 0.05, D = 57.6 σ | 5017 | 100.11 | 0.01022 |
| `fluid_hs_rho0.763944_D23.21_pad4.309/` | R6.6 | Hard-sphere fluid droplets, φ = 0.4, 16 frames | 5000 | 100.0 | 0.02717 |

Each directory holds `target_sq.npy` (columns q in Å⁻¹, S), the configuration
GSD and `meta.json` (construction parameters, md5s, saxs-fft environment and the
`shift_rmse` target checks). FCC crystallites: spherical cut of a bulk lattice,
i.i.d. Gaussian displacements σ_disp = 0.10 σ, seed 0, one realization. The
ρ* = 0.05 crystallite is the ρ* = 0.10 one scaled by (0.10/0.05)^(1/3) (same N
and displacement draws, σ_disp kept at 0.10 σ).

## S(q) device

S(q) of all three was computed on an A40 GPU with `a40_targets.py` (saxs-fft
0.3.1, `53aa855`). Repeats on the same or another GPU model (A40, L40) agree to
~1e-14, but a CPU computation of the same configuration assigns some
bin-edge q-grid points to the neighbouring shell (`shift_rmse` ~1e-3 for these
targets, up to ~1e-2). Simulated S(q) should therefore also be computed on GPU.
