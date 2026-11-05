# R7.0: measured I(q), the form-factor file, and S(q) extraction

First step of the measured-curve metric work (2026-11-05): check the inputs of
the current I(q) -> S(q) extraction before changing the metric. Nothing in the
BO path changes here; `sq_extraction.py` writes a second S(q) next to the
current one so the two can be compared.

Run on the six R4 curves (outputs in `out/r4/`):

```bash
python campaigns/r7.0-sq-extraction/sq_extraction.py \
    --ff formfactors/sasmodels_sphere_fit.txt \
    --out campaigns/r7.0-sq-extraction/out/r4 --old-trim-tail 200 \
    campaigns/r4-measured/data/*.npy
```

Input files are `.npy` or text with columns `q, I[, sigma]` (header lines are
skipped). Tests: `tests/test_r70_sq_extraction.py`.

## Findings on the R4 curves

**The form-factor file is a SasView `sphere` curve with its background.**
`formfactors/sasmodels_sphere_fit.txt` (md5 `f7be827a6350dd6094eefdcf46e7c5ba`)
is reproduced to 1e-13 in ln by `scale * P(q; R = 134.21 A, pd = 0.100) + 0.001`,
with P the sasmodels sphere, Gaussian radius polydispersity, 3 sigma / 35
points. 0.001 is SasView's default `background`. It is 14 % of the divisor at
q = 0.075 and 24 % at q = 0.090; no resolution smearing is in the file. Its q
grid is the union of measured grids (167-184 of each R4 file's q values are in
it): it is the SasView fit of bare silica NPs at 2 mg/mL (confirmed by the
author, 2026-11-05). The background belongs to that measurement and should
not be in the divisor. Its diameter, 26.8 nm, differs from `d_si = 24.6` nm,
the silica diameter used as the simulation length unit (`datasets.physics_map`,
saxs-fft q conversion); where 24.6 nm comes from is not recorded here.

**The R4 I(q) are two detector distances merged.** Below
q_merge = 0.0263-0.0301 A^-1 (different in each file) the q step is
1.70e-4 A^-1, above it 3.40e-4. With 75 um pixels (Eiger2 R 1M) and Cu K-alpha
one pixel per bin gives sample-detector distances of 1799 mm and 900 mm.
sigma/I drops by a factor 4-6 at the merge: the end of the long-distance data is
the noisiest part of each curve. The intensity ratio of the two segments at the
merge cannot be fixed from the merged data: a line fit on each side gives
0.86-1.37, a quadratic fit up to 0.4 different for the same file, because the
second structure-peak group and the first form-factor minimum sit at the
merge. The unmerged per-distance files (with their overlap) are needed for that.

**The data favour a smaller radius than the file.** Fitting
`c_k P(q; R, pd) + B_k` to all six curves above the first minimum (S taken as 1
there, R and pd shared, c_k and B_k >= 0 per curve):

| fit window (A^-1) | R (A) | pd | chi2_red |
|---|---|---|---|
| 0.033-0.09 / 0.12 / 0.143 | 128.5 / 128.1 / 127.8 | 0.110-0.111 | 2.4-3.8 |
| 0.040-0.09 / 0.12 / 0.143 | 129.6 / 129.1 / 128.8 | 0.100-0.110 | 1.9-3.8 |
| 0.045-0.09 / 0.12 / 0.143 | 131.5 / 132.0 / 132.4 | 0.098-0.102 | 1.3-2.8 |

Per-file fits in 0.033-0.12 give R = 127.7-128.7 A, pd = 0.109-0.113. The
file's 134.2 A is outside every window's value.

The extrema of I q^4 do not depend on that fit (`iq4_*` columns of
`summary.csv`, ratio to the bare-silica model in parentheses):

| | min 1 | max 2 | min 2 | max 3 |
|---|---|---|---|---|
| bare silica, R 134.2 A | 0.0326 | 0.0443 | 0.0559 | 0.0674 |
| six R4 curves | 0.0334-0.0347 (1.025-1.066) | 0.0457-0.0470 (1.033-1.063) | 0.0566-0.0588 (1.012-1.051) | 0.0671-0.0719 (0.995-1.066) |

23 of the 24 ratios are above 1, median about 1.04: an apparent radius about
4 % below the bare silica. A coating does not move all extrema alike: with a
supported bilayer (10 A water gap, 9 A heads, 28 A tails) with or without a
DNA layer of 60-100 A, the shift is -4 % to +7 % for single extrema, but the
second minimum moves by -0.5 % to 0 % in every case tried, against +1.2 % to
+5.1 % measured. A smaller silica (another batch) or a q calibration that
differs between the bare-silica and the R4 measurements would move all
extrema alike; the bare-silica I(q) and the calibration of both sessions are
needed to tell them apart. A 5 % radius error matters most
just below the first minimum, where |d ln P / d ln q| = 8-12 (1.6 at the first
structure peak).

**A flat background is left in every I(q).** I q^4 rises above q ~ 0.07. The
fitted B is 14-30 % of I at q = 0.075. Divided by P it grows roughly as q^4,
which is what `trim_tail = 200` removed.

**Effect on S(q)** (`out/r4/summary.csv`, current -> refit; peak 1 in
0.006-0.02, group 2 between 0.02 and the refitted first minimum 0.0349, tail
mean in 0.06-0.075):

| file | peak 1 | group 2 | tail |
|---|---|---|---|
| chol_060_nacl_10_bridge_20 | 1.37 -> 1.45 | 1.69 -> 1.29 | 1.13 -> 1.12 |
| chol_100_nacl_10_bridge_20 | 1.59 -> 1.73 | 1.45 -> 1.21 | 1.31 -> 1.14 |
| chol_140_nacl_10_bridge_20 | 1.52 -> 1.94 | 1.53 -> 1.53 | 0.94 -> 1.12 |
| chol_100_nacl_20_bridge_20 | 1.45 -> 1.44 | 1.79 -> 1.20 | 1.19 -> 1.12 |
| chol_100_nacl_40_bridge_20 | 1.35 -> 1.26 | 2.11 -> 1.19 | 1.28 -> 1.13 |
| chol_100_nacl_10_bridge_80 | 1.76 -> 1.90 | 1.60 -> 1.11 | 1.27 -> 1.13 |

Group 2 is above peak 1 in four files with the current extraction (one a tie,
1.53 vs 1.52) and in none after the refit; the dip after the first minimum (q 0.035-0.04) is gone;
the tail level spreads over 1.12-1.14 instead of 0.94-1.31. Peak 1 changes by
up to 28 % because the current extraction scales P to the data in
q 0.02-0.03, inside group 2; the refit scales at S -> 1 above the minimum.

**Peak widths.** First-peak FWHM (Gaussian on a linear baseline, refitted S):
2.8-4.3e-3 A^-1, FWHM/q1 = 0.26-0.31. For comparison, the saved FCC inverse
targets have FWHM/q1 = 0.04 (about two saxs-fft bins of 2.55e-4), the HS fluid
target 0.27. Instrument resolution (`resolution_fwhm`, beam size on the
detector + one pixel + K-alpha doublet) at 1800 mm is 0.7e-3, 1.6e-3 and
3.2e-3 A^-1 for a 0.3, 0.7 and 1.4 mm beam (twice that at 900 mm, above the
merge). Which collimation was used decides whether the instrument is a small
or a large part of the measured width.

## Limits

- S = 1 above the first minimum is an approximation; chi2_red 2-4 shows
  structure left there, and the fitted R moves by 4 A with the window.
- The low-q segment is not rescaled at the merge (see above).
- Resolution is not applied to either curve.
- The output S(q) files have columns `q, S, sigma_S` on the measured q grid
  with no trimming; nothing reads them yet.
