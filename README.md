# NAMD Analysis

Analyze saved CA-NAC and Hefei-NAMD results with explicit units, state mappings,
fit windows, and input provenance. This package is independent of
[NAMD_Launcher](https://github.com/lgutsev/NAMD_Launcher): the launcher prepares
and runs calculations; this package reads their output. Existing manually
prepared campaigns work too.

Version 0.2 provides:

- Campaign inventory and identification of failed historical single-exponential fits.
- EIGTXT/NATXT dimension and run-setting audits, energy-gap statistics, and
  pair-resolved NAC statistics with explicit units, including detection of
  coupling files that were capped before they were written.
- SHPROP population averaging across **all mapped groups**, matched time-grid
  checks, population conservation checks, and between-file standard errors.
- Group populations, net changes, finite-window population integrals, and
  survival when a complete normalized state map is explicitly declared.
- Optional single-exponential decay fitting with explicit time windows,
  residual diagnostics, poor-fit flags, extrapolation warnings, and optional
  whole-file bootstrap intervals.
- Multistate kinetic fitting: a Markovian rate matrix over the declared
  groups, uncertainty on every rate, an identifiability test that refuses to
  quote rates the data did not determine, and an optional extraction sink.
- VACF and phonon spectral density: descriptive comparison of existing
  `spectral_density_*.txt` files, and computation from an XDATCAR with
  Cartesian minimum-image velocities, optional mass weighting, segment
  averaging, and an explicit transform convention.
- Run/initial-state comparison on common saved times and candidate kinetic
  graph ranking by descriptive AIC/AICc/BIC.
- JSON reports, CSV tables, PNG/PDF plots, SHA-256 fingerprints and imported
  launcher manifests.

No raw data are modified, clipped, reordered, or silently renormalized. Output
folders must be new, or `--overwrite` must be passed explicitly. No DFT, NAC
calculation, job submission, or electronic propagation is performed.

The command is installed as both `namd-analysis` and `inamd-analysis`; the
two are the same program.

## Install

```bash
git clone https://github.com/lgutsev/NAMD_Analysis.git
cd NAMD_Analysis
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Start with existing inputs

Unpack your archives locally into separate directories, preserving their dates.
Keep full simulation data outside this Git repository.

```bash
inamd-analysis inventory /path/to/unpacked/campaigns --out results/inventory

inamd-analysis audit /path/to/FAPI_001_BCF_PCBM_A \
  --nac-unit eV --dt-fs 1 --threshold-mev 0.5 \
  --out results/pcbm_A_audit
```

The audit expects `EIGTXT` with one frame per row and one energy (eV) per state,
and real `NATXT` with one flattened N×N matrix per row. Both must have the same
frame count. `inp` is optional; when present its band window is checked and
POTIM can supply the timestep. Headerless numeric tables and `#` comments are
supported, including Fortran D exponents. Complex or packed triangular NAC
formats are not supported.

**Choose the NAC unit from the producing code, not its magnitude.** Choices:
`eV`, `meV`, or derivative coupling `fs^-1` (converted using hbar). The
hbar/dt comparison is only a diagnostic scale, not an automatic artifact filter
or universal bound. Confirm the timestep used to produce the NAC file; an
explicit `--dt-fs` overrides POTIM.

`pairs.csv` distinguishes mean gap, RMS gap, mean absolute NAC, RMS NAC,
95th percentile, samples above threshold, samples sitting at the file's global
maximum, the sample sum in meV, and the rectangular time integral in meV fs.
Samples are not independent burst events.

The audit also reports whether the coupling file was **capped** before it was
written. When many off-diagonal samples share the file's largest magnitude
exactly, that magnitude is a bound rather than a measurement, and every mean,
RMS and integral built on those samples is a lower bound. The cap is reported,
never undone. The archived `FAPI_001_BCF_PCBM` runs are capped at 0.6 eV; see
[validation](docs/validation.md).

## Analyze populations

First inspect a raw SHPROP file and identify its time and population columns.
All column numbers in the JSON configuration are **zero-based table columns**,
not VASP band numbers. `examples/two_state.json` assumes time in column 0,
a metadata/energy column 1, VBM in column 2, and CBM in column 3.

```bash
inamd-analysis populations \
  --files '/path/to/run/SHPROP.*' \
  --config examples/two_state.json \
  --fit-group CBM --fit-start-ns 0 --fit-end-ns 10 \
  --out results/two_state
```

`examples/interface_six_state.json` illustrates grouping three PCBM states:
VBM, BCF, PCBM1–3, CBM. **Verify this assignment against your actual band
characters and file layout.** It is not an automatic physical assignment and
cannot track states whose spatial character changes. Copy/edit the config
for each system. A file previously averaged by a script that changed only one
column cannot be treated as an averaged multistate population history.

```bash
inamd-analysis populations \
  --files '/path/to/interface/SHPROP.*' \
  --config examples/interface_six_state.json \
  --out results/interface_populations
```

Groups must be disjoint. Set `complete_population: true` only if they exhaust
the normalized single-electron population. `recombined_group: "VBM"` then
permits survival = 1 - VBM, under the physical interpretation that VBM
population represents recombination. Partial maps are supported without a
survival claim. No interpolation or truncation of different time grids occurs.
Each input file receives equal weight; do not mix pre-averaged and individual
files or simulations with different statistical weights.

Outputs include `report.json`, `populations.csv`, `populations.png`, and
`populations.pdf`; a requested fit also writes `fit.csv`. SEM is omitted for
one file. It is a between-file estimate, not uncertainty from independent MD
sampling, and can be too small for correlated initial conditions.

## Analyze phonon spectra

Two entry points. The first describes and compares spectral density files you
already have, on whatever grid they were written:

```bash
namd-analysis vacf-spectra \
  --files '/path/to/VACF/spectral_density_*.txt' \
  --range 0:800 --bands 0:50,50:100,100:200,200:400,400:800 \
  --reference FAPI_001_DISORD --xlim 0:150 \
  --out results/spectra
```

It writes band integrals, in-range band fractions, the spectral centroid, a
peak table with FWHM, and — only when every file shares a frequency grid —
differences against the chosen reference. Band *fractions* are normalized
inside the analysis range and are comparable between systems; raw integrals
are not, unless the spectra share convention, normalization, trajectory length
and atom count. A two-column file records none of those, so they are reported
as unknown.

The second computes the spectrum from a trajectory:

```bash
namd-analysis vacf-trajectory \
  --xdatcar '/path/to/*/XDATCAR_FINAL' \
  --dt-fs 1 --max-lag 5000 --segment-length 10000 \
  --mass-weight --convention cosine --xlim 0:800 \
  --out results/vacf
```

Velocities are **Cartesian**: each fractional displacement from XDATCAR is
folded into the minimum image and multiplied by the cell before differencing.
Differencing raw fractional coordinates, as the older scripts in
`Step2B_Phonon_Analysis` do, mixes the cell axes and records a boundary
crossing as a jump of nearly a full cell. Rigid centre-of-mass translation is
removed by default, because it is not a vibration but does add intensity at
the lowest resolvable frequencies — exactly where low-frequency dynamic
disorder is read off. `--no-unwrap` and `--keep-com` restore the old behaviour
for comparison only.

`--convention cosine` (the default) is the cosine transform of the VACF, the
usual vibrational density of states. `--convention power` is the squared
modulus used by the earlier scripts: a different quantity, with different line
shapes and relative intensities. The cosine transform of a truncated, windowed
VACF can ring negative; the report counts negative samples in the analysis
range and declines to report a centroid when they occur.

Frequency resolution is set by `--max-lag`, not by the trajectory length:
`max_lag = 5000` at 1 fs gives 6.67 cm⁻¹ bins. `--segment-length` averages
VACFs over consecutive non-overlapping segments and keeps each one, but
segments cut from a single trajectory are not independent samples, and their
spread measures drift along the run rather than statistical error.

Outputs include `report.json`, `vacf_<label>.csv`, a
`spectral_density_<label>.txt` in the same two-column format as the older
scripts, per-system VACF plots, and a combined `spectra.png`/`.pdf`.

## What a fitted lifetime means

The optional model is P(t) = P(t_start) exp[-(t-t_start)/tau], with the starting
population fixed to the first sample in the selected window. It has no offset
and does not normalize to the maximum. It is suitable only for a resolved
single decay. Plateau, sequential trapping, and competing-channel dynamics
need a different kinetic model. A high R² does not establish a unique physical
mechanism or a precise extrapolated lifetime. Whole-file bootstrap intervals are available with `--bootstrap 200`; they
measure between-file spread and do not account for shared-trajectory bias. Rising or flat populations are rejected as decay fits.

Population loss from BCF is not automatically recombination or extraction.
Inspect its destinations. Net PCBM accumulation is not directional flux or
collected charge. A single-exponential fit infers no forward/backward rates;
for those, state a kinetic model explicitly with `kinetics` below, and read
what that model can and cannot claim.

## Fit a rate model

`kinetics` fits a Markovian master equation over the declared groups,

```
dP/dt = K P      K_ij = rate j -> i  (i != j)      K_jj = -sum_{i != j} K_ij
```

so every column of K sums to zero and population is conserved. The initial
condition is taken from the data, not fitted. You must name the transitions
you are willing to allow:

```bash
namd-analysis kinetics \
  --files '/path/to/interface/SHPROP.*' \
  --config examples/interface_kinetics.json \
  --scheme 'CBM->BCF,BCF->CBM,BCF->PCBM,PCBM->VBM' \
  --bootstrap 200 \
  --out results/kinetics
```

`--scheme` takes the presets `dense`, `sequential` and `reversible`, or an
explicit list as above. The state map must set `complete_population: true`: a
master equation over a subset of states is not a closed system, and the
command refuses rather than fitting one.

**Two rates in opposite directions are usually not separately determined.**
Population curves constrain the *eigenvalues* of K much better than its
entries, and many different forward/backward pairs reproduce the same P(t).
Every rate therefore carries a standard error and a degeneracy check: a rate
whose relative standard error exceeds 0.1, or which correlates above 0.95 with
another, or which the residuals are simply blind to, is reported with
`identified: false` and a stated reason, and must not be quoted. The
eigenvalue timescales are reported separately, because those are what the
data actually constrains.

The check has teeth. On the package's own synthetic test, a `dense` scheme —
every ordered pair of groups, so twelve transitions on these four groups —
reaches the same R² as the correct sparse one while its Jacobian condition
number rises to 6e19 and the rates it cannot determine come back
`identified: false` with their degeneracy partners named. On a three-group
version of the same test, three of the six dense rates are rejected.

The `±` values are a **linearized lower bound**. Because P(0) is read from a
noisy sample rather than fitted, this package's own Monte Carlo measures the
true spread at up to six times the quoted standard error (the nuisance
direction is marginalized out, which narrows the gap but does not close it).
Prefer the bootstrap whenever you have more than one file.

`--bootstrap N` resamples whole input files with replacement and refits,
using the same weighting as the point estimate, and giving a percentile
interval. That measures the spread between the files you
supplied — which share a trajectory and often correlated initial conditions —
so it can be narrower than the true uncertainty, and it says nothing about
whether the Markovian model is right at all.

### The extraction sink

`--sink-group PCBM --sink-rates 0.01:1000:25` adds an absorbing channel that
drains a group at rate `k_esc` and sweeps it on a log grid:

```
dP_PCBM/dt|escape = -k_esc P_PCBM      dP_collected/dt = +k_esc P_PCBM
```

The sink is part of the propagated system, so escaped population leaves the
dynamics and cannot return or recombine — it is not an integral taken over
the sink-free solution afterwards.

Be clear about what this is. The transfer rates were fitted to data with no
extraction in it, and `k_esc` is physics you supply; nothing in the interface
calculation determines it. The sweep answers a counterfactual — *given these
transfer rates, how fast would onward transport have to be to outcompete
return and recombination?* — and the crossover it reports is a requirement on
the ETL, not a measured extraction efficiency.

Outputs: `report.json`, `rates.csv`, `kinetics_curves.csv`, a data-versus-model
figure with residuals, and `sink_sweep.csv` plus its figure when a sink is
requested.

## Tests and development

```bash
python -m unittest discover -s tests -v
```

165 tests cover table and XDATCAR parsing (including VASP's negative
target-volume scale factor), namelist coercion, audit checks including cap
detection, conservation, all-column averaging/SEM, malformed and mismatched
inputs, analytic exponential recovery, long extrapolations, legacy failed
fits, VACF estimators against the direct double loop, recovery of known
oscillator frequencies, strict-JSON report serialization, recovery of a known
rate matrix, refusal to identify a rate the residuals cannot see, detection of
an unidentifiable over-parameterized scheme, bootstrap weighting, and CLI
report/figure generation. See [validation](docs/validation.md)
for the supplied archive audit and [scope](docs/scope.md) for next steps.

## Scientific software credit

The underlying dynamics and couplings are produced by
[Hefei-NAMD](https://github.com/QijingZheng/Hefei-NAMD),
[CA-NAC](https://github.com/WeibinChu/CA-NAC), and the electronic-structure
software used in the original campaign. Cite those methods in publications.
This repository implements analysis and does not redistribute those engines,
VASP potential files, or the uploaded research archives.

## Run comparisons and remaining plans

See [comparison commands and intervals](docs/comparisons.md),
[plan status](docs/scope.md), and [hopping-history requirements](docs/hopping_histories.md).
