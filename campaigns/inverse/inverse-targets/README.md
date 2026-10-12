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

FCC crystallites: spherical cut of a bulk lattice, i.i.d. Gaussian
displacements σ_disp = 0.10 σ, seed 0, one realization. The ρ* = 0.05
crystallite is the ρ* = 0.10 one scaled by (0.10/0.05)^(1/3) (same N and
displacement draws, σ_disp kept at 0.10 σ).

## Layout

```
<target>/
  fcc_crystallite.gsd | fluid_droplets.gsd   configuration
  meta.json                                  parameters, md5s, both S(q), checks
  gpu/target_sq.npy, gpu/a40_info.json       S(q) on GPU (A40), a40_targets.py
  cpu/target_sq.npy, cpu/cpu_info.json       S(q) on CPU (Xeon Gold 6230R), cpu_targets.py
```

`target_sq.npy` has columns q (Å⁻¹) and S. Both scripts use saxs-fft 0.3.1
(`53aa855`) with the same settings and differ only in the device.

## Which file to use

Use the S(q) computed on the same kind of device as the simulated S(q).
The GPU and CPU versions of one configuration differ in how q-grid points lying
exactly on a shell boundary are assigned, which changes a few bins:

| Target | `shift_rmse`(cpu, gpu) |
|---|---|
| FCC ρ* = 0.10 | 7.7e-4 |
| FCC ρ* = 0.05 | 1.3e-2 |
| HS fluid | 4.1e-4 |

GPU repeats on A40 and L40 agree to ~1e-14. CPU results also depend on the CPU
model: another cluster node gave `shift_rmse` 1.7e-3 against this CPU version
for FCC ρ* = 0.10.
