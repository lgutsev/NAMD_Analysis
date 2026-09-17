# NAMD Analysis

Analyze saved CA-NAC and Hefei-NAMD results with explicit units, state mappings,
fit windows, and input provenance. This package is independent of
[NAMD_Launcher](https://github.com/lgutsev/NAMD_Launcher): the launcher prepares
and runs calculations; this package reads their output. Existing manually
prepared campaigns work too.

Version 0.5.2 provides:

- Campaign inventory and identification of failed historical single-exponential fits.
- EIGTXT/NATXT dimension and run-setting audits, energy-gap statistics, and
  pair-resolved NAC statistics with explicit units, read against `ħ/dt`, the
  energy scale set by the electronic timestep, with an optional declared
  upstream NAC handling policy.
- Canonical master SHPROP generation from the original `SHPROP.*` histories,
  with every source file fingerprinted and every population column averaged.
- SHPROP population averaging across **all mapped groups**, matched time-grid
  checks, population conservation checks, and between-file standard errors.
- Group populations, net changes, finite-window population integrals, and
  survival when a complete normalized state map is explicitly declared.
- Optional single-exponential decay fitting with explicit time windows,
  residual diagnostics, poor-fit flags, extrapolation warnings, and optional
  whole-file bootstrap intervals.
- Multistate kinetic fitting: a Markovian rate matrix over the declared
  groups, uncertainty on every rate, and four independent identifiability
  tests — blind Jacobian columns, relative standard error, pairwise
  correlation, and SVD null-space participation — plus refusal of any rate
  that stopped on an optimizer bound, identifiability-aware bootstrap
  intervals, and an optional extraction sink.
- Transient population observables over user-declared windows: peak, time of
  peak, endpoints and time-integrated population, so an early event that ends
  before a later fit window cannot be reported as an absence of change.
- First-passage branching: P(reach one group set before another) solved from
  the fitted generator, with the competing-channel ratio given only where it is
  exact, and the whole observable bootstrapped under the identifiability tests.
- Counterfactual extraction-versus-recombination competition, including the
  onward escape rate that would be required for extraction to win.
- A reviewer-facing table, one row per interface configuration, where every
  cell is labelled observed, model-inferred or counterfactual.
- VACF and phonon spectral density: descriptive comparison of existing
  `spectral_density_*.txt` files, and computation from an XDATCAR with
  Cartesian minimum-image velocities, optional mass weighting, segment
  averaging, and an explicit transform convention.
- Run/initial-state comparison on common saved times and candidate kinetic
  graph ranking by descriptive AIC/AICc/BIC.
- Frame-dependent subsystem character from PROCAR projections, giving a
  projection-weighted *diagonal* subsystem population, so a band that exchanges
  subsystem character mid-trajectory is followed rather than
  mislabelled; each original SHPROP is projected before averaging, only the
  electronic frames actually visited are parsed, and a cheap preflight checks
  the whole configuration before any projection is read.
- One-command campaign preparation: `character-prepare` reads a campaign
  directory and writes `state_map.json`, `atom_groups.json`,
  `projection_manifest.json`, an audit of how every value was decided, and
  optionally a Slurm script that preflights before it computes. Ambiguity is
  refused rather than resolved.
- Bounded-memory SHPROP reading: preflight costs the same on a 900 MB history
  as on a small one, and the analysis streams each history in row chunks into a
  running ensemble mean instead of building a whole-campaign stack.
- Early-vs-late regime analysis: two declared windows characterized separately,
  then one global kinetic model scored against early-plus-late. The early
  window defaults to 0-100 ps; the late window is never defaulted, because no
  manuscript fit window is encoded here.
- Adiabatic character events: gap, |NAC| and frame-resolved fragment character
  synchronized on the resolved MD frame, with crossing events classified
  without calling a character swap a surface hop or automatic charge transfer.
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
maximum, the fractions of samples above 0.8 and 0.9 × ħ/dt, the pair maximum
as a multiple of ħ/dt, the sample sum in meV, and the rectangular time
integral in meV fs. Samples are not independent burst events.

### Couplings and the electronic timestep

The audit reads the coupling distribution against **ħ/dt**, the energy scale
set by the discrete electronic timestep:

```
E_dt = ħ / dt       ħ = 0.6582119569 eV fs       dt = 1 fs  ->  ħ/dt = 0.658 eV
```

Couplings approaching this scale should be treated as **numerically
pathological** — the finite-difference evaluation of the NAC has broken down
over one step — rather than interpreted as arbitrarily large physical matrix
elements. The `nac_timestep_limit` check reports ħ/dt in eV and meV, the
global maximum as a multiple of it, and how many samples sit above 0.8 and
0.9 × ħ/dt. Nothing is filtered, rescaled or rejected on account of it.

A magnitude shared **exactly** by many samples is a separate observation,
reported by `nac_repeated_ceiling`. It did not come out of the dynamics, so
some upstream step put it there: repeated values at 0.600 eV are consistent
with an intentionally imposed upstream NAC safety ceiling. This is not
described as accidental clipping or as a defect.

What that ceiling did to the statistics depends on the upstream rule, and the
rule is not recoverable from a NATXT file. Declare it if you know it:

```bash
inamd-analysis audit /path/to/FAPI_001_BCF_PCBM_A \
  --nac-unit eV --dt-fs 1 \
  --nac-policy examples/nac_policy_bcf_campaign.json \
  --out results/pcbm_A_audit
```

```json
{
  "nac_policy": {
    "warning_threshold_eV": 0.6,
    "numerical_limit": "hbar_over_dt",
    "reject_above_eV": 0.66,
    "action_above_limit": "zero"
  }
}
```

Only `action_above_limit: "clip"` — truncation of otherwise valid couplings —
makes the affected means, RMS values and integrals lower bounds, and only then
does the report say so. A policy that zeroes or rejects a pathological sample
does not, and the audit will not claim otherwise. Without a declared policy
nothing about upstream handling is assumed for the dataset. The policy is a
statement about the workflow that produced the file; it is fingerprinted into
the report but never verified against the numbers.

## Build a canonical master SHPROP

Before analyzing anything, turn the original `SHPROP.*` histories into one
canonical mean file:

```bash
inamd-analysis average-shprop \
  --files '/path/to/run/SHPROP.*' \
  --config examples/two_state.json \
  --out results/master_shprop
```

Outputs are `SHPROP.master` (a plain numeric table, readable by every other
command here), `report.json`, `population_sem.csv` and `input_files.csv`.

For every declared population column,

```
Pbar_i(t) = (1/N) sum_r P_i,r(t)
```

with **equal weight per file**. Every population column is averaged, not one
of them. This matters: the historical `SHPROP_avg.sh` averaged a single
selected column and copied the rest from whichever file came first, so the
averaged column and the copied ones describe different things and nothing in
the file says which is which. `test_every_population_column_is_averaged` fails
if this package ever reproduces that.

Nothing else is touched:

- **Time column.** Verified identical in every input and copied through
  exactly. Differing grids are an error — no interpolation, no truncation to a
  common length, no time shifting.
- **Other columns** (the running energy in column 1, for instance). Copied
  only when every input agrees on them exactly. When they disagree, master
  generation **refuses**, because the first file's copy is not the average of
  anything and the mean of a per-trajectory quantity need not be meaningful.
  Pass `--average-extra-columns` to state explicitly that averaging them is
  what you want, or declare the column in the state map.
- **Conservation.** With `complete_population: true`, the populations are
  checked in every input file *independently* as well as in the result, so
  averaging cannot bury a bad file inside a healthy-looking mean.
- **Duplicates.** The same path twice, or two byte-identical files, is an
  error: equal weighting makes a repeat a silent reweighting.

`report.json` records, for every source file, its path, size, SHA-256,
modification time, row and column counts and its own conservation
diagnostics, plus the exact averaging rule, the state-map fingerprint, the
software version, the CLI arguments and any launcher manifests found beside
the inputs.

Between-file SEM goes to `population_sem.csv`, not into `SHPROP.master`, which
stays SHPROP-compatible. Grouped SEM sums states into the physical group
*within each file first* and then takes the spread across files, preserving
within-group covariance; it is never reconstructed from marginal state SEMs.
As everywhere else here, that spread measures the supplied files, which
commonly share one MD trajectory — it is not an independent ensemble error bar.

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

## Charge transfer: what the populations show and what a model infers

Three commands answer the interfacial-transfer question, and they are
deliberately separated by how much modelling each one requires.

### What the populations actually show

```bash
namd-analysis transient-populations   --files '/path/to/run/SHPROP.*'   --config state_map.json   --transient-window 0:0.1 --late-window 0.1:10   --out results/transient
```

Reports, per group and per window: initial, final, peak, time of peak, minimum,
net change and time-integrated population. `--window NAME=START:END` is
repeatable for arbitrary windows; either end may be left blank.

This exists because a late-window fit cannot see an early transient. A group
can rise to substantial occupation and fall back before the fit window opens,
and the fit will then report it as flat. When the early peak clearly exceeds
anything in the late window, the report says so in as many words.

**The windows are yours.** There is no default transient window. Where one
regime ends is a property of the system; run this command first and read it off
the data.

A time-integrated population is the area under an occupation curve, in
population x ns. It is not a flux and not an amount transferred.

### What a kinetic model infers

```bash
namd-analysis branching   --files '/path/to/run/SHPROP.*'   --config state_map.json   --scheme 'CBM->BCF,BCF->PCBM,BCF->VBM'   --source BCF --success PCBM --failure VBM   --bootstrap 200   --out results/branching
```

Solves the backward equation on the fitted generator for the probability that a
carrier starting in `--source` reaches `--success` before `--failure`. This is
what accumulated population cannot tell you: back-transfer erases accumulation,
so a carrier can reach the acceptor, return and leave again while the final
occupancy records none of it. Both sets accept several groups.

The simple competing-channel ratio `k_S / (k_S + k_F)` is printed **only** when
every exit from the source lands directly in one of the two outcome sets —
otherwise a carrier can leave and come back, the ratio is not the branching
probability, and the reason it is withheld is stated.

Every result is graded `identified`, `weakly_identified` or `not_identifiable`
from the rates it actually depends on. `--bootstrap` resamples whole files,
refits, re-runs the v0.3 identifiability tests and only then takes the branch;
if fewer than 80% of successful resamples produce an identifiable branch, the
interval is suppressed and the counts are reported rather than the interval
being quietly narrowed by the draws that were dropped.

### What would have to be true

```bash
namd-analysis extraction-competition   --files '/path/to/run/SHPROP.*'   --config state_map.json   --scheme 'CBM->BCF,BCF->PCBM,BCF->VBM'   --sink-group PCBM --escape-rates 0.001:1000:40   --out results/competition
```

Adds an absorbing onward-escape channel at an assumed rate and competes it
against recombination by first passage, then finds the escape rate at which the
two yields are equal.

Call that a **required onward escape timescale**. It is not a measured
extraction time. The escape rate is supplied by you, the transfer rates were
fitted to data containing no extraction at all, and the simulated cell has no
electrode and no long-range transport.

### One table for a campaign

```bash
namd-analysis reviewer-summary comparison_manifest.json --out results/reviewer
```

Runs the whole chain per configuration and writes `reviewer_branching.csv`,
one row each, plus a `report.json` with provenance, scheme comparison and every
identifiability verdict. See [examples/bcf_pcbm](examples/bcf_pcbm/) for a
worked recipe and templates.

**An empty cell means the quantity was unavailable or its underlying rates were
not identifiable. It is never a zero.**

### Three kinds of number

Every reported quantity is labelled:

| class | meaning |
| --- | --- |
| `observed_from_SHPROP` | read off the populations |
| `model_inferred` | from the fitted rate matrix |
| `counterfactual` | propagated under a condition never simulated |

A first-passage probability is never a device extraction efficiency. A sink
yield is never measured. And no fitted rate is an event count: averaged SHPROP
populations do not record individual hops, so directional counts such as
`BCF->PCBM` against `PCBM->BCF` cannot be recovered from them at all. See
[docs/hopping_histories.md](docs/hopping_histories.md).

## Populations that follow the state, not the column

Every other command in this package labels a state by its SHPROP column for
the whole trajectory. That is wrong whenever an adiabatic band exchanges
spatial character partway through: near a crossing, the band that was BCF-like
becomes PCBM-like, the column keeps its old label, and the reported population
becomes a mixture of two physical things.

`character-populations` fixes that by reading the character from the
electronic structure itself. For every NAMD time step it finds the MD frame
that generated it, reads the PROCAR subsystem projection for that frame, and
reweights the adiabatic populations accordingly:

```
P_g(t) = sum over states i of  P_i(t) * w_ig[frame(t)]
```

### What that number is

A **projection-weighted diagonal subsystem population** — never called simply
"the physical population" here, because two approximations stand between it and
the actual occupation of a subsystem, and both are properties of the input
files rather than choices you can tune.

*It has no coherences in it.* The true subsystem occupation is
`Tr[rho P_g] = sum_i rho_ii <i|P_g|i> + sum_{i!=j} rho_ij <j|P_g|i>`. SHPROP
stores only the diagonal `rho_ii`, and a PROCAR stores only diagonal,
band-by-band projections, so the second sum is not in the inputs at all. It is
omitted, not estimated, and nothing reported here bounds how big it is. That is
the natural companion to surface-hopping populations, which are themselves
diagonal — but it is an approximation to the occupation, not the occupation.

*The weights are a conditional share.* `w_ig = W_ig / sum_g W_ig` says what
fraction of the weight that landed inside a declared group was `g`, not what
fraction of the band was `g`. Interstitial and unassigned weight is divided
away by that normalization. `captured_projection` is what tells you how much
there was to begin with — see below.

The full statement travels with the numbers, in `report.json` under
`population_definition`, and the CSV column is named
`projection_weighted_diagonal_population`.

### Generate the configuration instead of writing it

```bash
namd-analysis character-prepare   --shprop-dir /path/to/NuTest   --projection-dir /path/to/FAPI_001_BCF_PCBM_A   --preset bcf_pcbm --campaign A   --frame-mode dish-cyclic   --write-sbatch --run-preflight
```

Reads the SHPROP headers and a bounded sample of rows, walks the numbered frame
directories, and writes `state_map.json`, `atom_groups.json`,
`projection_manifest.json`, `prepare_report.json` and a batch script — then
runs the ordinary preflight against exactly those files.

It prints what was **inferred** from the files separately from what came from a
**preset** (a declaration a human established for that campaign) and what still
needs **you**. It refuses to name a subsystem, place an atom boundary, choose
between two readings of a table, guess the frame mode, or pick a cycle length
when directory coverage and `NSW - 1` disagree. Details:
[docs/campaign_preparation.md](docs/campaign_preparation.md).

### What you need

| input | what it is |
| --- | --- |
| original `SHPROP.*` histories | **not** `SHPROP.master` — see the warning below |
| a state map | one declared population column per basis state, BMIN through BMAX |
| a projection manifest | JSON mapping MD frame number to PROCAR file |
| an atom-group map | JSON assigning one-based PROCAR ion indices to subsystems |

Templates: `examples/projection_manifest.template.json` and
`examples/atom_groups.template.json`.

### Check before you run

```bash
namd-analysis character-preflight   --files '/path/to/run/SHPROP.*'   --config state_map.json   --projection-manifest projection.json   --atom-groups atoms.json   --frame-mode dish-cyclic
```

Reads the SHPROP headers, the manifest and one PROCAR *header*. It parses no
projections and computes no populations, so it is fast enough to run every
time. It reports the basis window, the distinct `NAMDTINI` values, the frames
each history will visit, how many PROCARs will actually be parsed, the ion and
band counts of a representative PROCAR, and whether the atom groups cover the
structure. It collects every problem rather than stopping at the first, and
exits non-zero if any would block the run.

### The smallest working run

```bash
namd-analysis character-populations   --files '/path/to/run/SHPROP.*'   --config state_map.json   --projection-manifest projection.json   --atom-groups atoms.json   --frame-mode dish-cyclic   --out results/character
```

### Never pass SHPROP.master

Pass the **original** histories. Projection must happen per history and only
then be averaged, because histories with different `NAMDTINI` start at
different MD frames and therefore visit different PROCAR frames. Averaging
first destroys the alignment, and the result will look perfectly plausible
while being wrong. `average-shprop` is for the fixed-column path; it is not an
input to this one.

### Choosing `--frame-mode`

- `linear` — time step *t* uses MD frame `NAMDTINI + t - 1`. Use this when the
  trajectory walks forward through a long MD run and never wraps.
- `dish-cyclic` — the frame wraps around a fixed period:
  `mod(t + NAMDTINI - 1, period)`, with 0 mapping to `period`. Use this for
  engines that recycle a short MD segment across many trajectories.

Pick the one your engine actually implements. Getting it wrong shifts every
frame assignment and the populations will still sum to one, so nothing will
look broken. Run the preflight and check the reported first and last frames
against what you expect.

### `cycle_length`

In `dish-cyclic` mode the wrap period comes from `NSW - 1` in the SHPROP
header. An archived campaign whose engine bookkeeping does not match the
number of saved electronic frames can override it with `"cycle_length"` in the
projection manifest. The explicit value wins, and the disagreement is recorded
in `shprop_alignment.csv` and flagged by preflight rather than being silently
absorbed. Only set it if you know why the header is wrong.

### Reading `captured_projection`

PROCAR weights are normalized across the declared subsystems, so
`captured_projection` is how much of the band actually landed inside any
declared PAW sphere before that division. Values near one mean the band is
well described by your groups. Values well below one mean most of the band is
somewhere else — diffuse, vacuum, or in atoms you did not assign — and its
normalized character is then mostly an artefact of dividing a small number.

The report gives the min, 1st, 5th, 50th and 95th percentiles, per-band
medians and minima, the worst individual frame/band samples, and the fraction
below your `min_projection_weight`. These are **diagnostics**: nothing is
discarded, repaired or reweighted on their basis. Decide for yourself whether
a band with poor capture belongs in the analysis.

### What a character swap is and is not

`character_swaps.csv` records where a band's dominant subsystem changes from
one MD frame to the next. That is a statement about the **electronic
structure along the MD trajectory**: the orbital moved.

It is **not** a surface hop. A nonadiabatic hop is a change of *which state
the carrier occupies*, and it lives in the SHPROP populations, not in the
PROCAR projections. A band can change character with no hop at all, and a
carrier can hop with no change of character. Do not report swap counts as
transfer events.

Every swap statistic in every output — the rows of `character_swaps.csv`, the
`report.json` summary, and the per-band counts in
`projection_quality_by_band.csv` — is built from one shared definition of an
examined transition, so the per-band counts sum to the campaign total and no
two of them can disagree about what adjacency means. Each row carries a
`resolution`:

- `adjacent` — consecutive MD frames; the change is located exactly.
- `across_gap` — only the frames your histories visit are parsed, so a change
  seen between two non-adjacent frames happened somewhere in between. The count
  is a lower bound.
- `cycle_wrap` — in `dish-cyclic` mode the frames form a ring, and the
  `period -> 1` step is taken by the dynamics like any other. Scanning frames in
  ascending order would miss it, so it is examined separately. Treat it as an
  artefact of cyclic re-use unless your trajectory is genuinely periodic: the
  nuclear geometry jumps there, because the mapping restarts the MD run rather
  than continuing it.

The wrap step is only examined when a history actually takes it, counted from
the resolved frame series rather than guessed from which frames happen to be
loaded. When it is not examined the report says which of the reasons
applies; it is never passed over in silence.

### Large histories

Archived histories run to hundreds of megabytes each. Preflight reads only
headers, row counts and each file's first and last row, so its memory does not
grow with file size. The analysis streams each history in row chunks and folds
them into a running ensemble mean and variance rather than building an
`(nfiles, nrows, ncolumns)` stack.

Chunking changes residency only — the mean, the standard error and every
validation are identical at any chunk size, which is pinned by test. Tune it if
you want to:

- `--shprop-chunk-rows N` — rows held at once.
- `--shprop-io-mode auto|stream|memory` — `auto` picks from file size, `stream`
  always chunks, `memory` reads each whole table in one chunk. There is one
  code path; `memory` is a chunk the size of the table, not a second
  implementation.
- `--accumulator-memmap-dir DIR` — spill the running mean/variance to
  memory-mapped files when they are large. Without it they stay in RAM.

### Outputs

`character_populations.csv`, `fixed_vs_projected.csv` and its summary
`fixed_vs_projected_summary.csv` (max, RMS and integrated absolute difference
per group, with the time of worst disagreement — this is what tells you
whether dynamic character changed the conclusion), `character_swaps.csv`,
`projection_quality_by_band.csv`, `projection_character.csv`,
`shprop_alignment.csv`, a figure, and `report.json` with full provenance.

### Early and late regimes

```bash
namd-analysis regime-analysis   --files 'run/SHPROP.*'   --config state_map.json   --late-window 0.1:10   --scheme 'BCF->PCBM,PCBM->VBM'   --out results/regimes
```

Characterizes the first 100 ps separately from the slow regime and asks whether
one kinetic description is adequate for both. The early default is a choice,
not a constant; the late window is **required**, because no manuscript fit
window is encoded in this repository and inventing one would put a number
nobody chose into the science. Details:
[docs/early_late_regimes.md](docs/early_late_regimes.md).

### Adiabatic character events

```bash
namd-analysis character-crossings   --projection-character results/character/projection_character.csv   --eigtxt /path/to/EIGTXT --natxt /path/to/NATXT --dt-fs 1   --out results/crossings
```

Reuses the character outputs rather than reparsing PROCARs, and synchronizes
gap, coupling and fragment character on the resolved MD frame. **A character
swap is not a surface hop, and neither is automatically charge transfer** --
staying on one adiabatic state through an avoided crossing changes the fragment
identity, while hopping between two can preserve it. Why, with the two-state
model behind it: [docs/adiabatic_vs_diabatic.md](docs/adiabatic_vs_diabatic.md).

Theory and conventions: [docs/state_character.md](docs/state_character.md).
Generating the configuration: [docs/campaign_preparation.md](docs/campaign_preparation.md). A worked campaign: [examples/bcf_pcbm/FAPI_001_A](examples/bcf_pcbm/FAPI_001_A).

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
Every rate therefore carries a standard error and **five independent checks**,
any one of which sets `identified: false` with a stated reason. A rate so
marked must not be quoted; the eigenvalue timescales are reported separately,
because those are what the data actually constrains.

| Check | Fires when |
| --- | --- |
| Blind column | the residuals do not move at all when the rate moves |
| Relative standard error | it exceeds 0.1, and that error is itself a lower bound |
| Pairwise correlation | it exceeds 0.95 with another rate |
| **Null-space participation** | the rate's parameter axis lies in the numerical null space of the Jacobian |
| **Optimizer bound** | the optimum sits on the edge of the allowed range |

The last two are new in 0.3 and neither replaces the others.

*Null space.* A rate can be unidentifiable even when its own Jacobian column
is far from zero and it correlates strongly with no single other rate: it is
enough that some **combination** containing it leaves the residuals unchanged.
An SVD of the rate Jacobian names those combinations. The report carries
`jacobian_rank`, `jacobian_nullity`, `null_space_tolerance` and, per rate,
`null_space_participation` = ‖V_null[j, :]‖ — zero when the axis is entirely
inside the range space, one when the data is blind to that rate alone, about
0.71 for each member of a two-way degeneracy. Above 0.1 the rate is refused.

The rank tolerance is `σ_max · max(shape) · √ε`, not `· ε`. The covariance is
obtained by inverting `JᵀJ`, whose condition number is the *square* of J's, so
the effective precision of that solve is `√ε`. Directions below that are ones
the pseudo-inverse discards — and a pseudo-inverse reports **zero** variance
for a discarded direction rather than infinite, which would stamp exactly the
least determined rates `identified: true`. Such rates now get infinite
variance, so no rate is ever printed as `0 ± 0`.

*Optimizer bound.* Rates are fitted as bounded log-rates. A result sitting on
a bound is not an interior optimum, and the local quadratic picture behind the
covariance does not describe it — however small the resulting standard error
looks. `at_optimizer_bound` and `optimizer_bound` (`lower`/`upper`/`null`)
report it, and the rate is refused.

*Statistical dimensionality.* With a complete conserved population basis only
`G-1` of the `G` residual coordinates per time are free — the same subspace
`compare-schemes` scores through Helmert contrasts, which are orthonormal, so
the sum of squares is unchanged and only the count differs. The conditioned
`P(0)` row is excluded too, since the model is started there rather than
fitting it. The residual variance is scaled by that count, so the redundant
conservation direction is no longer counted as independent information. The
result is wider, more honest standard errors — which remain a local
approximation and a lower bound, not an exact interval.

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

**The interval is identifiability-aware.** An optimizer that converges is not
the same thing as data that determines a rate: keeping the value from every
converged resample produces a tight-looking percentile band around a number
the data never fixed. For each rate the bootstrap therefore records how many
resamples converged, in how many of those the rate was `identified`, and the
resulting fraction. An interval is reported only when the bootstrap as a whole
produced enough successful fits **and** the rate was identified in at least
80% of them, and it is then taken over the identified resamples alone.
Otherwise `bootstrap_ci_per_ns` is `null` and the report says

> Bootstrap interval suppressed because the transition was identifiable in
> only X% of successful resamples.

The dropped draws are counted, never silently discarded; the raw percentiles
over every converged draw are kept in the diagnostics for inspection and are
explicitly not an inferential interval. `rates.csv` carries
`bootstrap_ci_status` and `bootstrap_identified_fraction` alongside the
interval bounds.

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

## Known limits carried forward

**Averaged populations are not hopping histories.** Directional event counts —
perovskite→BCF, BCF→perovskite, BCF→PCBM, PCBM→BCF — cannot be reconstructed
from averaged SHPROP data, and this package does not attempt to. A flat
acceptor population can hide forward and backward exchange in equal measure.
Fitted kinetic rates are model parameters, not counted events. Event-resolved
analysis needs per-trajectory logging from the engine; see
[hopping histories](docs/hopping_histories.md) for exactly what would have to
be recorded.

**A fixed state map cannot follow a state whose character changes.** Grouping
by table column means that when adiabatic states exchange spatial character
near a crossing, the column keeps its label and the group population becomes a
mixture. Every command *except* `character-populations` has this limitation.
State character is never inferred from column number, energy or coupling
magnitude, and no nearest-energy band tracking is attempted as a substitute —
those would produce a confident wrong answer rather than an honest limit.

`character-populations` is the way out, and it needs real upstream input:
per-frame PROCAR projections. See the section above.

## Tests and development

```bash
python -m unittest discover -s tests -v
ruff check src tests
```

Tested on Python 3.9, 3.11 and 3.13 in CI; those are the versions the
`requires-python = ">=3.9"` declaration is actually backed by.

320 tests cover table and XDATCAR parsing (including VASP's negative
target-volume scale factor), namelist coercion, audit checks including the
timestep-limit and engineered-ceiling diagnostics and declared NAC policies,
canonical master SHPROP generation with full source provenance, conservation,
all-column averaging/SEM, malformed and mismatched
inputs, analytic exponential recovery, long extrapolations, legacy failed
fits, VACF estimators against the direct double loop, recovery of known
oscillator frequencies, strict-JSON report serialization, recovery of a known
rate matrix, refusal to identify a rate the residuals cannot see, detection of
an unidentifiable over-parameterized scheme, SVD null-space
participation, refusal of rates pinned to an optimizer bound,
identifiability-aware bootstrap suppression, conservation-subspace
observation counts, bootstrap weighting, and CLI report/figure generation. See [validation](docs/validation.md)
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
