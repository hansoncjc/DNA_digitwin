# Inverse-design targets

Generation scripts and settings for the target S(q) curves used by
`inverse.py` campaigns (filled by the target-regeneration task). Targets use
the forward model's saxs-fft settings (`N_grid = 600`, `trim = slice(6, -6)`);
`inverse.py` checks the `N_grid` and records each target's md5 at the first
start of a run.
