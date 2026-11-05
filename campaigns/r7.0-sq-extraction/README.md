# R7.0: measured I(q), the form factor, and S(q) extraction

First step of the measured-curve metric work (2026-11-05): check the inputs of
the I(q) -> S(q) extraction before changing the metric. Nothing in the BO path
changes here; `sq_extraction.py` writes a second S(q) next to the current one
so the two can be compared. Tests: `tests/test_r70_sq_extraction.py`.

## Run

Full Huat room-temperature collection (`Experimental_Data/Huat/SAXS_data_rt_Collection`,
42 I(q) files `[q, I, sigma]`, uploaded 2026-11-05). Step 1 fits the form
factor to the 17 dispersed unbridged samples and writes the empirical
correction; step 2 extracts S(q) for the other 25 with R, pd and the
correction held:

```bash
D=Experimental_Data/Huat/SAXS_data_rt_Collection
REF=$(ls $D/*_bridge_00.npy | grep -v -E "chol_020_nacl_(20|30|40)_bridge_00")
SMP=$(ls $D/*.npy | grep -v -F "$REF")
python campaigns/r7.0-sq-extraction/sq_extraction.py --ff formfactors/sasmodels_sphere_fit.txt \
    --out campaigns/r7.0-sq-extraction/out/reference --fit-window 0.015 0.16 --no-scan --write-correction $REF
python campaigns/r7.0-sq-extraction/sq_extraction.py --ff formfactors/sasmodels_sphere_fit.txt \
    --out campaigns/r7.0-sq-extraction/out/samples --fix-ff 123.4908 0.12684 \
    --ff-correction campaigns/r7.0-sq-extraction/out/reference/ff_correction.npy --old-trim-tail 200 $SMP
```

`out/samples/*_sq.npy` are `[q, S, sigma_S]` on the measured q grid, not
trimmed; nothing reads them yet. `out/r4/` is the first pass on the six R4
curves (R and pd fitted on those curves, no correction), kept for reference.

In each `--out`: `summary.csv` (one row per file), `fit.json` (fits, form-factor
audit, extrema, artifact measure), `sq_compare.png` (current vs new S),
`iq_audit.png` (I q^4 and sigma/I), `ff_audit.png`; `out/reference` also has
`ff_correction.npy` (`q, r, raw mean, raw error`) and `ff_correction.png`.
`--old-trim-tail 200` is only for the comparison with the current extraction.

## The data

Two instrument configurations, read from the q step (75 um pixels, Cu K-alpha):

- 20 files merged from 1800 mm (q step 1.70e-4) and 900 mm (3.40e-4) at
  q = 0.022-0.030: every bridged sample at 10-40 mM NaCl (18, including
  L_bridge 40 and 80) and two unbridged ones. q up to 0.143-0.150.
- 20 files at 900 mm only, q up to 0.16-0.20: 17 unbridged samples and the
  bridged chol 20/60/100 at 0 mM. I(0.05) is about 8 times lower than in the
  merged files and sigma/I is 13-17 % at q = 0.05 (2-4 % merged).
- `chol_020_nacl_40_bridge_00`: 900 mm with one irregular step;
  `chol_140_nacl_00_bridge_20`: 1800 mm only, q <= 0.116, sigma/I 37 % at
  q = 0.05 (unusable above q ~ 0.03).

`other data/`: the three `_scaled` files are the originals times 0.594 (sigma
unchanged); `chol_100_nacl_10_bridge_20_sweep` is a second merged measurement
of that sample, with form-factor extrema within 2 % of the first.

Unbridged samples (`bridge_00`): 17 of 20 are dispersed, S = 1.00-1.02 in
q 0.012-0.03 after a per-file sphere fit (chi2_red 1.1-1.7). chol 20 at 20,
30 and 40 mM are not (contact peaks near q 0.03, 0.045, 0.07; chi2_red 9-15)
and are left out of the reference. All dispersed samples rise below
q ~ 0.012 (median S 1.6-1.9 in 0.004-0.012): a low-q excess that has nothing
to do with bridging.

## Findings

**The form-factor file does not describe these particles.**
`formfactors/sasmodels_sphere_fit.txt` (md5 `f7be827a6350dd6094eefdcf46e7c5ba`,
same as the author's copy) is reproduced to 1e-13 in ln by the SasView
`sphere`: R = 134.21 A, Gaussian pd 0.100 (3 sigma, 35 points), plus
SasView's default background 0.001 (14 % of the divisor at q = 0.075, 24 % at
0.090). It is the fit of bare silica at 2 mg/mL (author, 2026-11-05). In all
42 files the extrema of I q^4 lie above that model's: ratio median 1.05-1.08
for the dispersed samples, 1.03-1.05 for the bridged merged ones
(`iq4_*_vs_file` columns). The dispersed samples, which have no structure,
show it too, so it is the reference that differs (another batch, the
calibration of that session, or the coating), not structure.

**Form factor from the dispersed samples.** A joint sphere fit to the 17
(q 0.015-0.16, shared R and pd, per-file scale and background) gives
R = 123.49 A, pd = 0.127, chi2_red = 1.39: diameter 24.7 nm, the
`d_si = 24.6` nm used as the simulation length unit. With pd held at 0.127,
the fitted R depends on the window but not on the group:

| window (A^-1) | dispersed (17) | bridged, 900 mm, 0 mM (3) | bridged, merged (18) |
|---|---|---|---|
| 0.033-0.12 | 125.9 | 126.3 | 126.0 |
| 0.045-0.14 | 128.9 | 129.3 | 129.5 |

So the two configurations agree, and a uniform Gaussian-polydisperse sphere is
not exact for the coated particles. The noisy 900 mm data do not show it
(chi2 ~ 1); the merged data (sigma/I 2-4 %) do, as a residual of +-20 % in S
around the first minimum.

**Empirical correction.** `ff_correction.npy` is the weighted mean over the
17 dispersed samples of (I - B)/(c P_sphere), smoothed (15 points), applied
in q 0.015-0.075 with a 0.005 linear taper at each end (above 0.075 the
dispersed data are dominated by background). It is up to +30 % near
q 0.045. P = P_sphere * r. Mean S over the 25 samples on a common q grid
(`mean_s_artifact`: structure peaks sit at different q and partly average
out, a form-factor error does not), RMS of (mean S - 1) in q 0.026-0.045:

| divisor | all 25 | merged bridged 18 | tail S, q 0.06-0.075 |
|---|---|---|---|
| current extraction (file as is, scaled in 0.02-0.03) | 0.353 | 0.358 | 0.91-1.31 |
| sphere 134.21 / 0.100, no background | 0.430 | 0.427 | 1.13-1.20 |
| sphere 123.49 / 0.127 | 0.112 | 0.112 | 1.14-1.22 |
| sphere 123.49 / 0.127 x correction | 0.061 | 0.067 | 1.01-1.08 |

(tail range without `chol_140_nacl_00_bridge_20`). With the correction no
curve has the group below the first minimum above its first peak.

**The first-peak height is not fixed by the data.** The scale c is set above
the first minimum, so any error in the shape of P between low and high q moves
S at the first peak. For the six 10 mM curves the first-peak maximum is:

| file | current | sphere 134.2 | sphere 123.5 | with correction |
|---|---|---|---|---|
| chol_020_nacl_10_bridge_20 | 1.07 | 1.44 | 1.55 | 1.79 |
| chol_060_nacl_10_bridge_20 | 1.37 | 1.44 | 1.55 | 1.79 |
| chol_100_nacl_10_bridge_20 | 1.59 | 1.71 | 1.86 | 2.14 |
| chol_100_nacl_10_bridge_40 | 1.44 | 1.74 | 1.96 | 2.24 |
| chol_100_nacl_10_bridge_80 | 1.76 | 1.81 | 2.06 | 2.38 |
| chol_140_nacl_10_bridge_20 | 1.52 | 1.93 | 2.07 | 2.39 |

a factor 1.3-1.7 between defensible choices. Peak positions do not move.

**The merged I(q).** Below the merge the q step is 1.70e-4, above 3.40e-4;
sigma/I drops by a factor 4-6 at the merge. The intensity ratio of the two
segments at the merge cannot be fixed from merged data: line and quadratic
fits on each side give 0.86-1.37 and differ by up to 0.4 for the same file,
because structure and the first minimum sit there. The unmerged files are
needed for that.

**Peak widths.** First-peak FWHM/q1 = 0.26-0.33 in the bridged samples with a
peak near q 0.013 (`fwhm1_over_q1`); the saved FCC inverse targets have 0.04,
the HS fluid target 0.27. Resolution (`resolution_fwhm`) at 1800 mm is
0.7e-3, 1.6e-3 and 3.2e-3 A^-1 FWHM for a 0.3, 0.7 and 1.4 mm beam on the
detector, against a measured FWHM of 2.8-4.3e-3; the collimation used is not
known here. `q1` and `fwhm1` are meaningless for curves without a peak in
q 0.006-0.02 (0 mM; chol 20 at 20-40 mM).

## Limits

- S = 1 is assumed in the scale/background window (0.033-0.12) of each sample;
  chi2_red 2.2 for the 25 samples shows structure left there.
- The correction is the mean of 17 noisy curves (sigma/I 13-17 % each at
  q 0.05) and stops at q 0.075.
- The S(q) in `out/reference` are extracted without the correction (with it
  they would be 1 by construction).
- The two merged segments are not rescaled; resolution is applied to nothing.
