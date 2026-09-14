# Comparing runs and kinetic schemes

## Separate systems or initial electronic states

```bash
namd-analysis compare-runs \
  --manifest examples/initial_state_comparison.json \
  --start-ns 0 --end-ns 10 --out results/initial_states
```

Copy the example and replace file patterns, map paths and initial-state
annotations. Relative paths are resolved against the manifest's directory,
not the shell's working directory. Each run is averaged separately and can
use its own state map and file count. No new trajectories are generated.

The report records the declared preparation and the actual first saved
population for every run. That first sample may already include electronic
relaxation and is not assumed to equal the prepared state. Every overlay and
difference uses the **exact intersection of saved absolute times** and common
group names. No interpolation, automatic time alignment, normalization or
zero-filling of absent groups occurs. Excluded sample counts and nonshared
groups are reported. The full-run summaries and common-window differences
have distinct fields in the report. Identical group names must denote the
same physical observable; labels do not establish that automatically.

Outputs: `comparison.png`, `comparison.pdf`, long-form `curves.csv`,
`differences.csv`, and provenance-bearing `report.json`. The CSV difference
integrals have units population × ns. They are not extraction yields.
Different initial states probe conditional questions, so a larger PCBM
population in a PCBM-initialized run does not demonstrate improved transfer.

## Candidate kinetic schemes

```bash
namd-analysis compare-schemes \
  --files '/path/to/SHPROP.*' --config examples/interface_kinetics.json \
  --schemes examples/candidate_schemes.json --criterion aicc \
  --fit-start-ns 0.1 --fit-end-ns 10 --out results/schemes
```

Supply at least two distinct candidate graphs as a JSON object mapping names
to the schemes accepted by `kinetics`. All fits use the same complete state
map, saved time samples, window, initial population, unweighted objective and
optimizer multistarts. Duplicate graphs are rejected. A failed candidate is
reported without aborting other candidates. No sink is included in this
comparison because there is no sink in the observed closed-system data.

`schemes.csv` ranks the candidates by the chosen AIC, AICc or BIC; all three
scores and delta scores are reported. Complete fits and identifiability
flags remain in `report.json`. A low score never overrides an unidentifiable
rate or demonstrates a unique mechanism.

The score assumes a shared unknown isotropic Gaussian residual variance in
the G−1 dimensional population-conservation subspace. An orthonormal Helmert
contrast removes the redundant population direction. With the initial
population conditioned on, its exactly matched row is omitted: n=(T−1)(G−1)
and k=number of fitted rates + one variance parameter. The common additive
likelihood constant is omitted. AICc is unavailable when n<=k+1. The full
rate count is retained for singular fits; an unidentifiable parameter does
not make a model free of complexity. Numerically exact residuals use an
explicit machine-precision variance floor and are flagged.

These are **descriptive conditional scores**, not calibrated evidence:
correlated time samples, noisy P0, active rate boundaries and singular fits
violate regular information-criterion assumptions. Do not interpret delta
scores as posterior model probabilities or automatically select a mechanism.
There is no automatic claim that eigenvalue timescales are certain either.

## Single-exponential intervals

```bash
namd-analysis populations \
  --files '/path/to/SHPROP.*' --config examples/two_state.json \
  --fit-group CBM --fit-start-ns 0 --fit-end-ns 10 \
  --bootstrap 200 --bootstrap-seed 7 --out results/decay_ci
```

Whole files are resampled with replacement, then averaged and fitted with
exactly the point estimate's window and fixed-prefactor model. The first
sample is recomputed for every resample, so its variability is retained.
The default 95% percentile interval is reported under `fit_uncertainty`.
At least 20 resamples are required; at least 20 and 80% of requested fits
must succeed for an interval to be emitted. Failures and poor-fit counts
are recorded. Poor-fit replicates are retained and flagged, not silently
removed to narrow an interval. A single file or identical files provides no
spread estimate and produces no interval. Time rows are never treated as
independent bootstrap units. Shared-trajectory bias remains unmeasured.

## Import launcher provenance

All commands automatically read adjacent `*_manifest.json` files and
`.namdforge/state.json` when they fingerprint data inputs. Add a campaign
manifest explicitly with repeatable `--launcher-manifest /path/file.json`.
A run-comparison entry can also list `launcher_manifests` relative to its
comparison manifest. Inputs in each report include a fingerprint of the
state-map JSON and parsed CLI options.

The report fingerprints and embeds the upstream JSON object; malformed
JSON is recorded as invalid. A 10 MiB size limit avoids embedding huge
manifests. No artifact path inside imported JSON is followed or executed.
This preserves upstream claims, not an independent verification of them.
No search above the input directory occurs, so parent campaign manifests
must be supplied explicitly. Existing commands keep their data formats.
