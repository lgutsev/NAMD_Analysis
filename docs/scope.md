# Scope

## What this package does

It reads results that already exist on disk and describes them: campaign
inventory, coupling audits, SHPROP population averaging and group analysis,
optional single-exponential fits with explicit windows and diagnostics, and
VACF / phonon spectral density analysis. Every command writes a JSON report
carrying the input SHA-256 fingerprints, the options used and the checks that
ran.

## What it does not do

- No DFT, no NAC calculation, no job submission, no electronic propagation.
  Preparing and running calculations is [NAMD_Launcher](https://github.com/lgutsev/NAMD_Launcher)'s
  job; this package reads the output.
- No modification of raw inputs. Nothing is clipped, reordered, interpolated,
  truncated or renormalized in place.
- No inference of state character. Group membership is declared by the user
  and validated for consistency, never guessed from energies or couplings.
- No first-passage yields or measured extraction efficiencies. Forward and
  backward rates are available only through an explicitly declared kinetic
  model (`kinetics`), which is an assumption about the dynamics rather than a
  measurement of them; see below.
- No universal confidence claims from a single trajectory. Both single-decay
  and multistate fits can bootstrap whole files; shared-trajectory bias and
  correlated initial conditions remain outside those intervals.

## Known limits

**A fixed state map cannot follow a moving state.** Grouping is by table
column. If a state's spatial character changes during the trajectory — which
is exactly what happens near a trivial crossing — the column keeps its label
and the group population becomes a mixture. Nothing here detects that.

This is a stated limit, not a gap to be filled heuristically. State character
is never inferred from column number, energy or coupling magnitude, and no
nearest-energy band tracking is attempted as a substitute: either would
produce a confident wrong answer where the honest output is a declared
limitation. The architecture a real fix would need is

```
frame -> electronic state -> physical character / projection
      -> time-dependent physical state mapping
      -> BCF / PCBM / perovskite populations
```

fed by genuine upstream input — orbital projections, spatial localization,
fragment charge or projection analysis, or wavefunction overlap tracking. None
of that exists in the supplied archives, so no file format or API for
frame-dependent state maps is designed here. The current fixed-map behaviour
is explicit and stays explicit.

**Between-file SEM is not an ensemble error bar.** It measures the spread
between the SHPROP files given, which share a trajectory and are often
launched from correlated initial conditions. It can be far too small.

**The cosine transform can ring negative.** The default spectral density is
the cosine transform of a truncated, windowed VACF, so it can go negative
near sharp features. Band fractions and centroids computed across a negative
region are unreliable; `describe` reports the negative sample count and
returns `NaN` for a centroid rather than a misleading number.

**Segment spread is not statistical error.** Segments cut from one trajectory
are not independent samples. Their spread measures drift along the run.

**Spectrum files carry no provenance.** A two-column
`spectral_density_*.txt` records frequency and intensity only. Convention,
window, smoothing, trajectory length and atom count are all unknown to the
reader, which is why raw integrals are not comparable across files unless you
know those matched.

**A Markovian rate matrix is assumed, not demonstrated.** `kinetics` fits
constant rates on a fixed graph. Memory effects, inhomogeneity across initial
conditions, and states whose character changes mid-trajectory all break that
picture, and none of them announce themselves as a bad fit. A high R² means
the model *can* reproduce the curves, not that the mechanism is right.

**Individual rates are frequently unidentifiable.** Population curves
constrain the eigenvalues of K far better than its entries. The module applies
five independent tests — a Jacobian column the residuals are blind to,
relative standard error above 0.1, correlation above 0.95 with another rate,
participation above 0.1 in the numerical null space of the Jacobian, and a
result sitting on an optimizer bound — and marks such rates `identified: false`
with a stated reason. None of the five replaces the others: a rate can join a
blind *combination* while its own column is far from zero and it correlates
strongly with no single partner, and a boundary solution can carry a small
standard error while being no estimate at all. Those numbers exist in the
output so the diagnostic can be audited; they are not results. The eigenvalue
timescales are the quantity to quote.

**Bootstrap intervals cannot make an unidentified rate look precise.** An
interval is reported only when the rate was identified in at least 80% of the
successful resamples, and it is taken over those resamples alone. Below that,
`bootstrap_ci_per_ns` is null and the count is reported. A converged optimizer
is not determined data.

**The coupling ceiling in the archived interface runs was engineered, not
accidental.** Repeated values at exactly 0.6 eV are consistent with an
intentional upstream NAC safety ceiling placed inside the numerical-safety
region of `ħ/dt` (0.658 eV at dt = 1 fs). Whether it turned any statistic into
a lower bound depends on the upstream rule, which a NATXT file does not
record; it has to be declared, and it is not assumed.

**The extraction sink is a counterfactual.** `k_esc` is supplied by the user,
the transfer rates behind it were fitted to data containing no extraction, and
the sweep answers "how fast would onward transport have to be" rather than
"how much charge was collected". It is a requirement on the ETL, not a
measurement of one.

**The quoted standard errors are a lower bound.** They are a local
linearization, and because P(0) is read from a noisy sample rather than fitted,
the measured spread runs up to six times larger (docs/validation.md). The P(0)
nuisance direction is marginalized out, which narrows the gap without closing
it. Prefer the bootstrap when more than one file is available.

**The bootstrap resamples files, not trajectories.** It captures the spread
between the SHPROP files supplied, which share a trajectory. On the package's
own synthetic test the resulting interval is narrower than the true error, and
it can exclude the true rate when the inputs carry a common bias.

## Plan status after version 0.3

New in 0.3:

- `average-shprop`: canonical master SHPROP from the original histories, every
  population column averaged, the time grid verified and copied exactly, a
  conservative policy for other columns, per-input conservation checks and full
  SHA-256 provenance.
- `ħ/dt` interpretation of coupling magnitudes, engineered-ceiling reporting
  separated from accidental clipping, and an optional declared `nac_policy`.
- SVD null-space rate identifiability and refusal of rates pinned to an
  optimizer bound.
- Identifiability-aware bootstrap intervals with per-rate counts and explicit
  suppression.
- Covariance scaled by the conservation subspace, consistent with the Helmert
  contrasts `compare-schemes` already uses.
- CI on Python 3.9, 3.11 and 3.13 plus a lightweight Ruff pass.

Implemented in 0.2:

- `compare-runs`: separately average and compare systems or initial states on
  common saved times, with initial populations and nonshared groups reported.
- `compare-schemes`: explicitly declared candidate graphs ranked by descriptive
  AIC/AICc/BIC with full fits and identifiability flags retained.
- `populations --bootstrap`: whole-file percentile intervals for a single decay,
  including failed-fit counts and missing-spread cases.
- Automatic adjacent launcher-manifest import plus explicit campaign manifests.
- Covariance-preserving group SEM and per-file conservation validation.

Still dependent on unavailable data or another repository:

1. **BCF figure reproduction:** original SHPROP histories and verified state maps
   are absent from the supplied archives. Saved fitted curves cannot replace
   them. `average-shprop` plus the existing commands are what that reanalysis
   would run, in order: canonical master, reproduced population curves, the
   historical single-exponential fit kept for comparison only, a corrected
   single-decay fit and an honest look at whether a single lifetime means
   anything, physically justified state groups, sparse candidate schemes,
   descriptive AIC/AICc/BIC ranking, identifiability throughout, and eigenvalue
   timescales reported separately from individual rates. A lowest information
   criterion will not establish a mechanism, and no fitted rate will be called
   an observed hopping rate.
2. **Directional event analysis:** the inspected public classic engine writes
   averaged SHPROP, not individual hop histories. See
   [engine inspection](hopping_histories.md) for the exact source revision,
   limitations, and required logging. The installed DISH/DEV engine still needs
   verification before instrumenting it through the launcher.
3. **New initial-state propagations:** the analyzer compares results but does not
   generate them; prepare these with the installed engine via NAMD_Launcher.

Usage and statistical assumptions are documented in
[comparisons](comparisons.md). A lowest information-criterion score is not
proof of a mechanism, and bootstrapping correlated files does not produce an
independent MD ensemble.

## Relationship to NAMD_Launcher

| NAMD_Launcher | This package |
| --- | --- |
| Prepare inputs and submit jobs | Assign states to groups and validate the map |
| Generate and collect energies / NACs | Audit them, and document corrections such as caps |
| Configure and launch Hefei-NAMD | Analyze populations and competing channels |
| Audit jobs, preserve provenance | Fit kinetics with explicit windows and diagnostics |
| Basic output summaries | Compare systems and generate publication figures |
| — | Fit multistate kinetics and test whether the rates are identifiable |
| Collect raw SHPROP histories | Average them into a canonical, fingerprinted master |
| Record the NAC handling policy used | Read couplings against ħ/dt and report a declared ceiling |

Inputs are accepted directly, whether or not a launcher produced them. When a
launcher manifest is present beside the inputs, its content and fingerprint
are imported. Parent campaign manifests can be passed explicitly. Upstream
claims are preserved without executing or trusting embedded artifact paths.
