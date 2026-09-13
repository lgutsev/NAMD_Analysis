# NAMD Analysis

Analyze saved CA-NAC and Hefei-NAMD results with explicit units, state mappings,
fit windows, and input provenance. This package is independent of
[NAMD_Launcher](https://github.com/lgutsev/NAMD_Launcher): the launcher prepares
and runs calculations; this package reads their output. Existing manually
prepared campaigns work too.

Version 0.1 provides:

- Campaign inventory and identification of failed historical single-exponential fits.
- EIGTXT/NATXT dimension and run-setting audits, energy-gap statistics, and
  pair-resolved NAC statistics with explicit units.
- SHPROP population averaging across **all mapped groups**, matched time-grid
  checks, population conservation checks, and between-file standard errors.
- Group populations, net changes, finite-window population integrals, and
  survival when a complete normalized state map is explicitly declared.
- Optional single-exponential decay fitting with explicit time windows,
  residual diagnostics, poor-fit flags, and extrapolation warnings.
- JSON reports, CSV tables, PNG/PDF population plots, and SHA-256 fingerprints.

No raw data are modified, clipped, reordered, or silently renormalized. Output
folders must be new. No DFT, NAC calculation, job submission, or electronic
propagation is performed.

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
95th percentile, samples above threshold, the sample sum in meV, and the
rectangular time integral in meV fs. Samples are not independent burst events.

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

## What a fitted lifetime means

The optional model is P(t) = P(t_start) exp[-(t-t_start)/tau], with the starting
population fixed to the first sample in the selected window. It has no offset
and does not normalize to the maximum. It is suitable only for a resolved
single decay. Plateau, sequential trapping, and competing-channel dynamics
need a different kinetic model. A high R² does not establish a unique physical
mechanism or a precise extrapolated lifetime. Parameter confidence intervals
are not yet implemented. Rising or flat populations are rejected as decay fits.

Population loss from BCF is not automatically recombination or extraction.
Inspect its destinations. Net PCBM accumulation is not directional flux or
collected charge. This release does **not** infer forward/backward rates,
first-passage yields, or extraction efficiencies from averaged populations.

## Tests and development

```bash
python -m unittest discover -s tests -v
```

Tests cover conservation, all-column averaging/SEM, malformed and mismatched
inputs, analytic exponential recovery, long extrapolations, legacy failed
fits, and CLI report/figure generation. See [validation](docs/validation.md)
for the supplied archive audit and [scope](docs/scope.md) for next steps.

## Scientific software credit

The underlying dynamics and couplings are produced by
[Hefei-NAMD](https://github.com/QijingZheng/Hefei-NAMD),
[CA-NAC](https://github.com/WeibinChu/CA-NAC), and the electronic-structure
software used in the original campaign. Cite those methods in publications.
This repository implements analysis and does not redistribute those engines,
VASP potential files, or the uploaded research archives.
