# BCF / PCBM interface campaign

A recipe for taking the original `SHPROP.*` histories of configurations A, B
and C through the whole chain and out to one reviewer-facing table.

Nothing here is filled in for you. The state maps deliberately contain **no
column numbers** and the windows in the manifest are placeholders. Both depend
on the actual files, and guessing either produces confident numbers about the
wrong states.

## What the chain answers, and what it does not

The question is whether the simulated interface supplies a viable transfer
branch toward the acceptor, or whether the carrier returns and recombines.
The finite cell contains no electrode and no long-range transport, so none of
this measures device collection. Every number falls into one of three classes,
and the reports say which:

| class | meaning |
| --- | --- |
| `observed_from_SHPROP` | read off the populations: peaks, endpoints, integrals |
| `model_inferred` | from the fitted rate matrix: rates, first-passage probabilities |
| `counterfactual` | propagated under an assumed escape rate that was never simulated |

A first-passage probability is **not** an extraction efficiency. A sink yield
is **not** measured. A crossover escape rate is a *requirement on the transport
layer*, not an observed extraction time.

## Step 0 — populate the templates

Copy each `state_map_X.template.json` to `state_map_X.json` and fill in
`population_columns` and `groups` from the real files and their provenance.
Loading an unedited template fails with `group 'CBM' is empty`; that is
intentional.

Verify every column against the band characters you computed. A fixed column
map cannot follow a state whose spatial character changes during the
trajectory, which is exactly what happens near a crossing. See
[docs/scope.md](../../docs/scope.md).

## Step 1 — canonical populations

```bash
namd-analysis average-shprop \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --out results/A/master
```

Averages **every** population column, checks that all inputs share one time
grid, verifies conservation per file, and fingerprints every source. Do not
reuse an old `data.txt` from the legacy shell script: it averaged one column
and copied the rest.

## Step 2 — look at the populations before fitting anything

```bash
namd-analysis transient-populations \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --transient-window 0:0.1 --late-window 0.1:10 \
  --out results/A/transient
```

Choose the windows **from this output**, not from another configuration. Read
where the early regime actually ends. The peak, the time of the peak and the
time-integrated population are reported for every group, so a transient that
ends before a later fit window opens cannot be reported as an absence of
change.

A time-integrated population is the area under an occupation curve. It is not
a flux and not an amount transferred.

## Step 3 — let the data rank the candidate schemes

```bash
namd-analysis compare-schemes \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --schemes candidate_schemes.json \
  --criterion aicc \
  --out results/A/schemes
```

`candidate_schemes.json` holds four graphs, from minimal to expanded. They are
candidates, not conclusions. A lower information criterion ranks them
descriptively on these observations; it does not establish a mechanism. A
scheme with an excellent R² whose rates are non-identifiable is not a better
answer, and the comparison reports both.

## Step 4 — fit, and check what the fit actually determined

```bash
namd-analysis kinetics \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --scheme 'CBM->BCF,BCF->PCBM,BCF->VBM' \
  --fit-start-ns 0.1 --bootstrap 200 \
  --out results/A/kinetics
```

Read the `identified` column before reading any rate. A rate marked
`identified: false` carries the reason it was rejected and must not be quoted.
Quote the eigenvalue timescales instead.

## Step 5 — the branch

```bash
namd-analysis branching \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --scheme 'CBM->BCF,BCF->PCBM,BCF->VBM' \
  --source BCF --success PCBM --failure VBM \
  --bootstrap 200 \
  --out results/A/branching
```

Reports P(acceptor before recombination | BCF) by first passage, which is what
accumulated population cannot tell you: back-transfer erases accumulation. The
simple competing-channel ratio is printed only when every exit from the source
lands directly in one of the two outcome sets; otherwise the reason it is
unavailable is stated.

The bootstrap resamples whole files, refits, re-runs the identifiability tests
and only then takes the branch. If too few resamples produce an identifiable
branch, the interval is suppressed and the count is reported.

## Step 6 — the counterfactual competition

```bash
namd-analysis extraction-competition \
  --files '/path/to/A/SHPROP.*' \
  --config state_map_A.json \
  --scheme 'CBM->BCF,BCF->PCBM,BCF->VBM' \
  --sink-group PCBM --escape-rates 0.001:1000:40 \
  --out results/A/competition
```

Adds an absorbing onward-escape channel at an assumed rate and competes it
against recombination by first passage. The crossover is the escape rate
onward transport *would have to* reach. Name it a required onward escape
timescale, never a measured extraction time.

## Step 7 — one table for all three configurations

```bash
cp comparison_manifest.template.json comparison_manifest.json
# edit paths, state maps and windows
namd-analysis reviewer-summary comparison_manifest.json --out results/reviewer
```

Writes `reviewer_branching.csv`, one row per configuration, plus a
`report.json` carrying provenance, the scheme comparison, the identifiability
verdicts and every note. **An empty cell means the quantity was unavailable or
its underlying rates were not identifiable. It is never a zero.**

## Reporting the result honestly

Run the same predefined workflow for A, B and C and report what comes out. If
the acceptor branch is substantial, report it. If recombination dominates,
report that. If the branch is not identifiable from averaged populations,
report that instead — it is a real finding about what this data can support,
not a failure.

Do not adjust windows, schemes or thresholds to move the number.

## What this cannot do

Averaged SHPROP populations do not record individual hops, so nothing here
counts `BCF->PCBM` against `PCBM->BCF` events. The branching probability is a
rate-model inference from ensemble populations. Event-resolved analysis needs
per-trajectory logging from the engine; see [docs/scope.md](../../docs/scope.md).
