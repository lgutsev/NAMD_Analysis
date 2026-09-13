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

**The coupling files are capped at 0.6 eV.** In all three `FAPI_001_BCF_PCBM`
runs the largest off-diagonal `|NAC|` is exactly 600.000 meV, and that value
is attained by many samples at once — 328, 282 and 234 of the 59 970
off-diagonal samples in runs A, B and C. A maximum reached by hundreds of
samples is a cap applied before the file was written, not a measurement.

It is concentrated where it matters most. In run A the strongest pair, states
2-3 (bands 978-979), sits at the cap in 154 of 1999 frames, so its 95th
percentile is also exactly 600 meV: more than 5% of that pair's history is a
bound rather than a value. Its mean `|NAC|` of 187 meV is therefore a lower
bound, and any rate estimated from it inherits that. The older two-state runs
show no capping (`FAPI_001_DISORD`: mean `|NAC|` 0.105 meV, max 1.556 meV),
so this is specific to the interface campaign. The audit reports the cap; it
does not undo it, and `pairs.csv` carries a `samples_at_global_max` column so
the affected pairs can be identified.

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

## Running the checks

```bash
python -m unittest discover -s tests -v
```

107 tests, covering table and XDATCAR parsing, namelist coercion, the audit
checks including cap detection, population conservation and averaging, fit
recovery and rejection, VACF estimators, spectrum conventions, and end-to-end
CLI runs that assert on the written reports and figures.
