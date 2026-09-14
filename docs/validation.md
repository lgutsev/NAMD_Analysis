# Validation against the supplied archives

Version 0.1 was checked against the campaign archives that motivated the
package. The archives themselves are not redistributed here; the paths below
are the folder names inside them.

| Archive | Contents used |
| --- | --- |
| `X_OutPackNAMD_FAPI_MD_NAMD_2026_Mar23_2026` | `FAPI_001_BCF_PCBM_A/B/C`, 6-state basis (BMIN 976, BMAX 981) |
| `X_OutPackNAMD_FAPI_MD_NAMD_Feb6_2025` | `FAPI_001_DISORD`, `_PEAI`, `_EHACl`, `_D3`, `EXAMPLE` |
| `X_OutPackNAMD_FAPI_MD_NAMD_Mar10_2025` | seven `*_D3` runs, including `_BCF_D3` and `_PEAI-EHACl_D3` |
| `Step2_NAMD_Instructions` | `SHPROP_avg.sh`, `plot_SHPROP_data.py` (the legacy averaging and fitting path) |
| `Step2B_Phonon_Analysis`, `VACF/` | `spectral_density_*.txt` for four systems, plus the scripts that produced them |

`inventory` found 16 run directories across the unpacked archives, 11 of them
carrying a legacy `fitting_results_final.txt`.

## What the audit found in the archived couplings

Every run reads cleanly: `EIGTXT` and `NATXT` agree on the frame count, the
NAC matrices are antisymmetric with a zero diagonal to machine precision, and
`INICON` matches `NSAMPLE`. Three findings are worth carrying into any paper
that uses these runs.

**The coupling files carry an engineered 0.6 eV ceiling.** In all three
`FAPI_001_BCF_PCBM` runs the largest off-diagonal `|NAC|` is exactly
600.000 meV, and that value is attained by many samples at once — 328, 282 and
234 of the 59 970 off-diagonal samples in runs A, B and C. A maximum reached
by hundreds of samples did not come out of the dynamics; some upstream step
put it there.

*Version 0.2 described this as accidental clipping and called every statistic
built on those samples a lower bound. That reading was wrong, and 0.3 corrects
it.* The ceiling was engineered on purpose, following guidance from
Dr. Oleg Prezhdo, and the physics behind it is the discrete electronic
timestep. For a step `dt`, a finite-difference NAC cannot meaningfully exceed

```
E_dt = ħ / dt       ħ = 0.6582119569 eV fs       dt = 1 fs  ->  ħ/dt = 0.658 eV
```

In the workflow that produced these couplings, ~0.6 eV was treated as a
warning/safety region and values beyond roughly 0.66 eV were considered
numerical errors of the finite-time-step evaluation and zeroed by the later
filtering logic. Archived files may carry the 0.6 eV ceiling depending on the
exact preprocessing version. The correct statement is therefore:

> The coupling distribution reaches the numerical-safety region associated
> with the finite electronic timestep. For dt = 1 fs, ħ/dt ≈ 0.658 eV.
> Couplings approaching this scale should be treated as numerically
> pathological rather than interpreted as arbitrarily large physical matrix
> elements. Repeated values at exactly 0.600 eV are consistent with an
> intentionally imposed upstream NAC safety ceiling.

Note that 600 meV is 0.912 × ħ/dt: the ceiling sits inside the numerical
warning region, which is exactly where it was placed.

Whether that ceiling makes the affected statistics lower bounds depends on the
upstream rule, and the rule is not recoverable from a NATXT file. A policy
that **zeroes or rejects** a pathological sample does not truncate otherwise
valid values, so it does not turn the surviving means into lower bounds; only
a policy that **clips** valid couplings at the limit does. Version 0.3 refuses
to make that claim unless a declared `nac_policy` says `action_above_limit`
is `clip`. The declared campaign policy
(`examples/nac_policy_bcf_campaign.json`) says `zero`, so the audit reports
the ceiling and its extent and makes no lower-bound claim.

The ceiling is concentrated where it matters most. In run A the strongest
pair, states 2-3 (bands 978-979), sits at it in 154 of 1999 frames, so its
95th percentile is also exactly 600 meV: more than 5% of that pair's history
is at the numerical-safety value. That fraction belongs beside the pair's mean
`|NAC|` of 187 meV whenever that mean is quoted — not because the mean is a
bound, but because a coupling that spends 5% of its history in the numerical
warning region is telling you the finite-difference evaluation was straining,
and a rate estimated from it inherits that. The older two-state runs show no
ceiling and nothing near ħ/dt (`FAPI_001_DISORD`: mean `|NAC|` 0.105 meV, max
1.556 meV = 0.0024 × ħ/dt), so this is specific to the interface campaign.

The audit reports all of it and undoes none of it. `pairs.csv` carries
`samples_at_global_max`, `samples_at_engineered_ceiling`,
`max_abs_nac_over_hbar_dt`, `fraction_above_0p8_hbar_dt` and
`fraction_above_0p9_hbar_dt` so the affected pairs can be identified and
placed on the ħ/dt scale.

**`NSW` in `inp` disagrees with the coupling files.** Every run declares
`NSW = 1998` while `EIGTXT` and `NATXT` hold 1999 frames. CA-NAC normally
writes `NSW - 1`. This is reported as a `mismatch` check rather than an
error, because the coupling files are self-consistent; it means `inp` is not
a reliable record of the trajectory those couplings came from.

**Pair statistics are dominated by rare spikes.** For several weakly coupled
pairs the mean `|NAC|` exceeds the 95th percentile — in run A, pair 0-4 has a
mean of 0.030 meV against a 95th percentile of 0.0015 meV, with 17 samples
above 0.5 meV. The mean is carried by a handful of frames. This is why
`pairs.csv` reports the mean, RMS, 95th percentile, maximum and
above-threshold count separately, and why the samples must not be treated as
independent events.

## What the audit reproduced about the legacy fits

The legacy path (`SHPROP_avg.sh` then `plot_SHPROP_data.py`) fitted
`exp(-t/A)` to the CBM column normalized by its own maximum, over the full
0-10 ns window, with no window control and no diagnostics beyond R². Running
`inventory` over the archives separates two distinct failures.

*Fits that failed visibly.* `FAPI_001_BCF_PCBM_A` reports
`A = 0.0001 ns, R² = -17.43` and `_B` reports `A = 0.0027 ns, R² = -79.44`.
A negative R² means the fit is worse than a horizontal line through the data.
`inventory` flags both. Version 0.1 refuses these outright: the traces do not
decay monotonically over the window, and `fit_single_exponential` raises
rather than returning a number.

*Fits that failed invisibly.* The older campaigns report high R² with
lifetimes far outside the simulated window — `FAPI_001_DISORD_EHACl_D3` gives
`A = 4569.97 ns, R² = 0.9969`, and `_D3` gives `A = 1544.15 ns, R² = 0.9989`,
both from a 10 ns window (the tabulated fit in each `fitting_results_final.txt`
ends at 10.000000 ns). At 4570 ns the fitted curve falls by 0.2% across the
whole window; almost any slowly decaying function fits that well. Version 0.1
still returns the number but attaches two warnings — that the window spans
less than one τ so the lifetime is an extrapolation, and that the population
falls by less than 20% inside the window. `tests/test_fitting.py` encodes
this case directly.

`FAPI_001_BCF_PCBM_C` (`A = 0.6722 ns, R² = 0.9035`) sits between the two and
is flagged on R² alone.

## Populations

No `SHPROP.*` files were included in the supplied archives, so the population
path was validated on synthetic sets with known answers rather than on the
campaign output: exact conservation, recovery of an analytic lifetime to six
decimal places, the between-file SEM against a hand-computed value, and
rejection of mismatched shapes and non-identical time grids.

One check is aimed squarely at the legacy path. `SHPROP_avg.sh` averages
**only column 4** and copies the rest from the first file. A file produced
that way is not an averaged multistate population history, and
`test_every_column_is_averaged_not_only_one` fails if this package ever
regresses to that behaviour. Re-average from the original `SHPROP.*` files
rather than reusing an old `data.txt`.

## Multistate kinetics

No campaign `SHPROP.*` files were supplied, so `kinetics` was validated on
synthetic sets propagated from a known rate matrix over four groups
(CBM, BCF, PCBM, VBM) with the scheme
`CBM->BCF, BCF->CBM, BCF->PCBM, PCBM->VBM` and true rates
8.0, 2.0, 3.0, 0.5 ns⁻¹.

*Noiseless data.* All four rates are recovered to six decimal places, R² = 1,
population is conserved to 1e-10, and no warnings are raised.

*Five noisy SHPROP files, end to end through the CLI.* Recovered
7.956, 1.965, 2.991, 0.4946 ns⁻¹ against true values of 8.0, 2.0, 3.0, 0.5 —
every rate within 1.8% — with R² = 0.99985 and a Jacobian condition number
of 75.

*The identifiability test has teeth.* Given the same noisy data, a `dense`
scheme (all 12 transitions, of which 4 are real) reaches the **same** R² as
the correct sparse scheme. It buys nothing and hides everything: the Jacobian
condition number rises to 6e19 and the undetermined rates come back
`identified: false` with their degeneracy partners named. (`dense` is every
ordered pair: twelve transitions on four groups, six on three.)
A `dense` fit that looks excellent by R² alone is exactly the failure this
check exists to catch.

*The sink is a real channel.* Across the sweep, collected + still-in-the-
interface sums to 1.000000 at every escape rate, confirming that escaped
population leaves the dynamics rather than being counted twice. At
`k_esc = 0`, nothing is collected and the sink-free solution is unchanged.
On the synthetic system the crossover where collection overtakes
recombination falls between `k_esc` ≈ 0.46 and 1.2 ns⁻¹, i.e. an escape time
of roughly 1-2 ns.

*The bootstrap is narrower than the truth.* With five files differing only by
independent noise, the 95% interval for `CBM->BCF` came back [7.92, 7.98],
which excludes the true 8.0. That is the documented behaviour, not a bug: the
interval measures the spread between the supplied files, and those files share
everything except their noise. It is reported with that caveat attached and
must not be quoted as an ensemble error bar.

*The quoted standard errors are a lower bound.* Over 200 noise realisations,
the empirical spread of each fitted rate was compared against the standard
error the module reports. With P(0) held exact, the reported error is correct
to slightly conservative (0.65-0.91x the empirical spread). With P(0) read
from the noisy data — what the code actually does — it understates by 1.6x to
5.9x even after the P(0) nuisance direction is marginalized out. The ratio is
reported in the module and in the CLI summary rather than papered over, and
the bootstrap is recommended instead whenever more than one file exists.

## Defects found by reviewing the kinetics module

The module was reviewed independently after release. Four defects were
confirmed and fixed; each has a regression test.

**A pseudo-inverse marked unconstrained rates as identified.** This was the
worst possible failure for this module, since refusing to quote undetermined
rates is its entire purpose. The covariance used `pinv(J^T J)`, and a
pseudo-inverse assigns *zero* variance to a direction the data does not
constrain, rather than infinite. A rate the residuals are completely blind to
therefore came back with relative standard error 0.0 and `identified: true`.
The correlation guard could not catch it either: with a standard error of
zero the correlation is NaN, which the degeneracy loop skips.

It was not an exotic corner. It fired whenever a declared group was never
populated, or a rate was driven to the optimizer bound — routine for an
over-declared scheme or a late fit window. Measured over 100 noise
realisations on a window starting at 2.5 ns, **77 rates were reported
identified with a standard error of exactly zero, 71 of them off the true
value by more than 50%**. One such rate was fitted at 6.7e-10 ns⁻¹ against a
true 3.0 and stamped `identified: true`.

Blind directions are now detected from the Jacobian columns and given
infinite variance, and each rejected rate carries a stated reason. After the
fix, together with the threshold change below, **zero of 400 rates in the
same experiment are wrongly reported as identified**, while all 160 rates in
a well-posed control remain identified and within 15% of truth.

**The identifiability threshold was far too loose.** It admitted any rate with
relative standard error below 1.0 — a 100% error. Given that the standard
error is itself a lower bound by up to 6x, that admitted rates known to
nothing better than an order of magnitude. It is now 0.1. Calibration across
a threshold sweep showed this costs nothing: every correctly recovered rate
in the well-posed control sits orders of magnitude below it.

**The ill-conditioning warning was silent in the worst case.** The guard read
`if np.isfinite(condition) and condition > 1e8`. A singular Gram matrix gives
`cond = inf`, and `np.isfinite(inf)` is False, so a condition number of 1e9
warned while a perfectly singular one did not. Of 43 singular fits in the
100-realisation experiment, none produced a diagnostic. The report also wrote
`jacobian_condition_number: null` for infinity, indistinguishable from "not
computed". Now the warning fires on non-finite values, a separate
rank-deficiency warning was added, and the report carries explicit
`jacobian_is_singular` and `jacobian_rank_deficient` flags.

**The bootstrap ignored `--weight-by-sem`.** The point estimate was a weighted
least-squares fit while the interval printed beside it came from unweighted
refits, so the two were different estimators. On a test with heteroscedastic
noise the reported rate for `BCF->CBM` was 1.9589 while its own interval was
[1.9730, 2.0224] — the rate fell outside its own confidence interval, with
nothing flagging it. `bootstrap_rates` now takes the weights and returns
convergence diagnostics, so an interval built from a subset of resamples says
so instead of being reported as if every draw succeeded.

Four smaller issues were fixed alongside: a group with zero between-file SEM
received the largest weight in the problem (the floor is now the median
positive spread, not the maximum); `_plain` crashed on a 0-dimensional numpy
array; the femtosecond-to-nanosecond conversion multiplied by `1/1e6` instead
of dividing by `1e6`, which dropped the frame sitting exactly on a requested
window boundary; and `--sink-rates` accepted an empty or non-finite list,
producing a silent no-op sweep.

Three documentation errors were corrected: the recovered rates are within
1.8% of truth, not 1.1%; `dense` on four groups is twelve transitions, not
six; and `examples/interface_six_state.json` described different columns in
its notes than it declared in its groups block.

## VACF and spectra

The four supplied `spectral_density_*.txt` files share a frequency grid
(2499 points, 6.6712 cm⁻¹ spacing, 0-16 670 cm⁻¹), so they can be compared
pointwise. Over a 0-800 cm⁻¹ analysis range:

| System | Centroid (cm⁻¹) | 0-50 cm⁻¹ share | Peaks |
| --- | --- | --- | --- |
| `FAPI_001_DISORD` | 364.4 | 8.9% | 5 |
| `FAPI_001_DISORD_PEAI` | 307.4 | 27.5% | 4 |
| `FAPI_001_DISORD_EHACl` | 261.3 | 21.6% | 4 |
| `FAPI_001_DISORD_PEAI-EHACl` | 386.7 | 10.4% | 6 |

These are measurements of the supplied curves, not conclusions. The files
record only frequency and intensity, so the transform convention, window and
smoothing behind them are unknown to the reader and are reported as such.

The trajectory path was validated against synthetic XDATCARs with known
frequencies: a 120 cm⁻¹ oscillator is recovered inside one frequency bin, two
oscillators at 100 and 250 cm⁻¹ are resolved as separate peaks, and the FFT
autocorrelation reproduces the direct double-loop estimator to 1e-10.

### Two differences from the scripts in `Step2B_Phonon_Analysis`

Both are deliberate, and both change the numbers.

**Velocities are Cartesian.** `spectral_density_multi.py` and
`vacf_analysis_full.py` difference the raw `Direct` coordinates from XDATCAR
and divide by the timestep. Fractional differences are not velocities: they
mix the three cell axes, they carry the wrong units, and they record an
atom crossing a periodic boundary as a jump of nearly a full cell. On a
synthetic trajectory with a modest drift, the raw difference produces a peak
velocity of 9.997 Å/fs against 0.0044 Å/fs after minimum-image unwrapping —
a factor of 2300, injected as broadband noise. This package folds each
displacement into the minimum image and multiplies by the cell.
`--no-unwrap` reproduces the old behaviour for comparison only.

**Rigid translation is removed by default.** Centre-of-mass drift is not a
vibration, but it adds a non-decaying term to the VACF that lands at the
lowest resolvable frequencies — exactly where the 0-50 and 0-200 cm⁻¹
dynamic-disorder metrics are read. On a synthetic trajectory carrying a
0.002 Å/fs drift, a spurious line appears at 33 cm⁻¹ and vanishes once the
centre-of-mass velocity is subtracted. Use `--keep-com` to leave it in.

A third difference is a change of definition rather than a correction. The
default spectrum here is the cosine transform of the VACF, the usual
vibrational density of states; the earlier scripts squared the modulus of the
transform, which is a different quantity with different line shapes and
different relative intensities. `--convention power` reproduces them. Note
that the cosine transform of a truncated, windowed VACF can ring negative;
`describe` reports the negative sample count in the analysis range and
declines to compute a centroid when it occurs.

## Defects found by reviewing v0.1

Three defects in the first release were found by re-deriving its numerics and
fixed; each now has a regression test.

**XDATCAR negative scale factor.** VASP reads a negative value on line 2 of a
POSCAR-style header as the *target cell volume* in Å³, not as a multiplier.
v0.1 multiplied by it. On a 2 Å cube with `-64.0`, that produced a cell of
volume 2 097 152 Å³ instead of 64 Å³ — a factor of 32 768, flipped in sign.
Frequencies would have survived this (they depend on the time axis) but every
velocity, VACF magnitude and spectral intensity would have been wrong by a
large constant. Now handled by the volume convention, with the degenerate case
rejected.

**Non-finite numpy scalars broke strict JSON.** `_plain` unwrapped numpy
scalars and returned early, skipping the finiteness check that Python floats
went through. A numpy NaN — which `describe` produces for the centroid of a
spectrum that rings negative — reached `json.dumps` and was written as a bare
`NaN` token, which the JSON specification does not allow and many parsers
reject. Now unwrapped first and checked, with `allow_nan=False` as a backstop
so anything that slips through fails loudly at write time rather than quietly
in the reader.

**Gaussian smoothing differed from scipy at the edges.** `gaussian_smooth`
claimed to match `gaussian_filter1d` and did so in the interior to machine
precision, but used numpy's `reflect` padding where scipy's default `reflect`
is numpy's `symmetric`. The edge samples differed by up to 0.1 on a unit-scale
signal. For a spectrum, the edge is the lowest-frequency bins — exactly where
the dynamic-disorder metrics are read. Now matches to 1e-12 across the whole
array, verified against scipy at four widths.

One thing checked and found correct: the FFT autocorrelation is free of
circular wraparound even at `max_lag == nsteps`, where the final lag has a
single time origin. It agrees with the direct double loop to 1e-14.

## Version 0.3: corrections and new checks

### The NAC ceiling is no longer described as accidental clipping

See the corrected reading above. `nac_clipping` is gone; in its place the
audit emits `nac_timestep_limit`, which ties the coupling distribution to
`ħ/dt` and reports `hbar_over_dt_eV`, `hbar_over_dt_meV`,
`global_max_nac_meV`, `global_max_over_hbar_dt` and the counts and fractions
above 0.8 and 0.9 × ħ/dt, and `nac_repeated_ceiling`, which reports a
magnitude shared exactly by many samples as an engineered upstream ceiling.
The lower-bound claim is now made only when a declared `nac_policy` says the
upstream rule *clipped* otherwise valid couplings, and never on a magnitude
alone. Tests assert the exact absence of the old wording.

### Canonical master SHPROP

`average-shprop` converts original `SHPROP.*` histories into a reproducible
`SHPROP.master`. Validated against synthetic sets with known answers: every
population column averaged (never one column with the rest copied), the time
grid verified identical and copied bit-for-bit, mismatched grids and lengths
rejected rather than interpolated or truncated, duplicate paths and
byte-identical files rejected, a single non-conserving input blocking the
master rather than being buried in the mean, an identical extra column copied
and a differing one refused without an explicit policy, and the written file
read back through the ordinary reader and re-checked before the command
returns. Every source file is fingerprinted with path, size, SHA-256,
modification time, row and column counts and its own conservation
diagnostics.

### Identifiability: two new checks and a dimensionality correction

*Null space.* An SVD of the rate Jacobian now names the combinations of rates
that leave the residuals unchanged, and each rate carries
`null_space_participation` = ‖V_null[j, :]‖. This catches a case the earlier
checks missed. Fitting `A->B, B->C, C->B` to data containing no C→B flow at
all, v0.2 returned `C->B = 1.7e-8 ns⁻¹` with `identified: true` and a standard
error of essentially zero: the column is small but not exactly zero, so the
blind-column test did not fire, and `pinv` assigns **zero** variance to a
direction it discards rather than infinite. The rank tolerance is taken at
`√ε` rather than `ε` precisely because the covariance inverts `JᵀJ`, whose
condition number is the square of J's; directions below that are the ones
`pinv` will discard. That rate is now refused, with infinite variance and a
stated reason, while the two rates the data does determine stay identified and
exact.

*Optimizer bounds.* `least_squares(...).active_mask` is read, and a rate
stopping on a bound is reported (`at_optimizer_bound`, `optimizer_bound`) and
refused — even when its linearized standard error looks perfectly ordinary,
because a boundary solution is not an interior optimum. `bound_decades` makes
the allowed range explicit so the case is testable deterministically rather
than by accident.

*Bootstrap.* Intervals are now taken over the resamples in which the rate was
actually identified, and reported at all only when the rate was identified in
at least 80% of the successful resamples. On the dense scheme — which
converges on essentially every resample while determining almost nothing —
v0.2 produced narrow percentile bands for rates the data never fixed; those
are now suppressed with the counts and the reason recorded, and the raw
optimizer percentiles kept separately as diagnostics.

*Statistical dimensionality.* The residual variance behind the covariance is
scaled by `(n_times − 1) × (G − 1)` when the populations are complete and
conserved, instead of `n_times × G`. The conservation direction was never free
to disagree with the model, and the conditioned `P(0)` row is not an
observation about the rates. This is the same subspace `compare-schemes`
already scored through Helmert contrasts; a regression test confirms the sum
of squares is identical in the redundant and contrast representations to
machine precision, so only the count changes. The standard errors get wider,
which is the right direction — they remain a local linearization and a lower
bound, and the bootstrap is still preferable when more than one file exists.

### What is still not done

The original `SHPROP.*` histories for the BCF/PCBM campaign are still not
present in the supplied archives. The reanalysis those files would support —
canonical master, reproduced population curves, the historical
single-exponential fit kept for comparison only, a corrected single-decay fit,
physically justified state groups, sparse candidate schemes ranked by the
descriptive information criteria, identifiability throughout, and eigenvalue
timescales reported separately from individual rates — is what `average-shprop`
plus the existing commands are built to run. **No campaign population figure
or lifetime has been reproduced by this release.** Everything above is
validated against synthetic inputs with known answers.

## Running the checks

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

CI runs the suite on Python 3.9, 3.11 and 3.13, plus a Ruff pass over `src`
and `tests`.

247 tests, covering table and XDATCAR parsing (including the negative scale
factor on triclinic cells), namelist coercion, the audit checks including the
timestep-limit and engineered-ceiling diagnostics, canonical master SHPROP
generation, population conservation and averaging, fit recovery and rejection,
VACF estimators against the direct double loop, smoothing against scipy where
the pad radius exceeds the array, spectrum conventions, strict-JSON report
serialization, recovery of a known rate matrix, refusal to identify a rate the
residuals are blind to, detection of an unidentifiable over-parameterized
scheme, bootstrap weighting and convergence diagnostics, sink conservation,
and end-to-end CLI runs that assert on the written reports and figures. The
0.3 additions are the timestep-limit and engineered-ceiling diagnostics,
declared NAC policies, canonical master SHPROP generation and provenance, SVD
null-space participation from hand-built Jacobians through to per-rate flags,
optimizer-bound refusal, identifiability-aware bootstrap suppression, and
conservation-subspace observation counts agreeing between `kinetics` and
`compare-schemes`.

## Version 0.2 plan completion

The full suite passed 178 tests at 0.2 (165 existing plus 13 new). New known-answer
checks cover conditional graph ranking, conservation-subspace observation
counts, run comparisons on exact common times, rejection of duplicate graphs
and inconsistent populations, reproducible whole-file exponential bootstrap,
missing spread and failed resamples, covariance-preserving group SEM, launcher
manifest import, and end-to-end command reports and plots. Synthetic curves
exercise both correct and deliberately wrong kinetic schemes. These checks
validate implementation behavior, not model validity for the BCF system.

The supplied archives still lack original SHPROP histories. No paper population
figures or lifetimes have been independently reproduced by this update.
