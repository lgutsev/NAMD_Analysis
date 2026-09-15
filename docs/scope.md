# Scope

## What this package does

It reads results that already exist on disk and describes them: campaign
inventory, coupling audits, SHPROP population averaging and group analysis,
optional single-exponential fits with explicit windows and diagnostics,
frame-dependent projection-weighted subsystem populations when genuine PROCAR
projections are supplied, and VACF / phonon spectral density analysis. Every command writes a
JSON report carrying the input SHA-256 fingerprints, the options used and the
checks that ran.

## What it does not do

- No DFT, no NAC calculation, no job submission, no electronic propagation.
  Preparing and running calculations is [NAMD_Launcher](https://github.com/lgutsev/NAMD_Launcher)'s
  job; this package reads the output.
- No modification of raw inputs. Nothing is clipped, reordered, interpolated,
  truncated or renormalized in place.
- No heuristic inference of state character. Fixed group membership is declared
  by the user. Frame-dependent character is used only when genuine projections
  are supplied explicitly; it is never guessed from energies or couplings.
- No measured extraction efficiencies. First-passage probabilities come from an
  explicitly declared kinetic model, and extraction-sink yields are
  counterfactual; neither is a device collection measurement.
- No universal confidence claims from a single trajectory. Both single-decay
  and multistate fits can bootstrap whole files; shared-trajectory bias and
  correlated initial conditions remain outside those intervals.

## Known limits

**Three kinds of number, never interchangeable.** Every reported quantity
carries one of `observed_from_SHPROP`, `model_inferred` or `counterfactual`.
Projection-weighted subsystem populations remain in the first class: they are
formed directly from saved SHPROP populations and explicitly supplied PROCAR
weights, without fitting a kinetic model. They are nevertheless a *diagonal*
approximation to subsystem occupation -- SHPROP records no coherences and a
PROCAR no cross-band projections, so the off-diagonal term of `Tr[rho P_g]` is
absent from the inputs and omitted from the result. The approximation lives in
the input files rather than in a fitted model, which is why the classification
does not change; the qualification is stated wherever the number appears. See
[state character](state_character.md). Rates and first-passage probabilities
come from a fitted model and inherit all of its assumptions. Extraction yields
are propagated under an escape rate that was never simulated. A first-passage
probability is not a device extraction efficiency, a sink yield is not
measured, and a crossover escape rate is a requirement on the transport layer
rather than an observed extraction time. The simulated cell contains no
electrode and no long-range transport.

**A time-integrated population is not a flux.** It is the area under an
occupation curve, in population x ns. Population that arrives, leaves and
arrives again is counted every time it is present, and population that never
moves contributes as much as population that turns over continuously. It does
not measure an amount transferred.

**A branching probability is a rate-model inference, not an event count.**
`P(acceptor before recombination)` is computed from the fitted generator by
solving the backward equation. It says what the fitted model implies, not how
many carriers actually made a transition. Averaged SHPROP populations cannot
answer the second question at all; see [hopping histories](hopping_histories.md).

**The competing-channel ratio is only sometimes the branching probability.**
`k_S / (k_S + k_F)` equals the first-passage probability only when every exit
from the source lands directly in one of the two outcome sets. With any other
exit the carrier can leave, return and try again. The ratio is withheld in that
case and the reason is stated.

**A branch inherits every identifiability limit of its rates.** Each branching
result is graded `identified`, `weakly_identified` or `not_identifiable` from
the rates it actually depends on -- the edges leaving the transient states
reachable from the source. A rate behind an absorbing state cannot affect the
answer and is not counted against it. When the grade is `not_identifiable` the
reviewer table leaves the cell empty rather than printing a number, and the
bootstrap interval is suppressed unless the branch was identifiable in at least
80% of successful resamples.

**Windows are yours, not the package's.** There is no default transient window
and no default late window. Where one regime ends and the next begins is a
property of the system, and encoding one campaign's choice as a constant would
silently misdescribe another.

**A fixed state map still cannot follow a moving state.** The ordinary
`populations`, `kinetics`, `branching` and reviewer workflows group fixed SHPROP
table columns. If an adiabatic state's spatial character changes during the
trajectory, a fixed column map becomes a mixture.

Version 0.5 adds an explicit alternative, `character-populations`, for datasets
that retain genuine frame-resolved orbital projections. For every original
SHPROP history it evaluates

```
P_g^r(t) = sum_i P_i^r(t) w_ig[f_r(t)]
```

with `w_ig` obtained from declared PROCAR atom groups and `f_r(t)` selected from
that SHPROP file's own `NAMDTINI` and the explicitly chosen frame-alignment
rule. Only after that per-file projection are histories averaged. This is
important: projecting an already averaged `SHPROP.master` would discard the
per-file phase information.

The feature does **not** infer character from column number, nearest energy,
energy continuity or coupling magnitude. Multiple k-points/spin components are
rejected rather than combined implicitly. Bands must match `BMIN..BMAX`
explicitly, and missing projection frames are fatal. The normalized subsystem
character is accompanied by the raw captured PAW projection so weak projection
quality remains visible, because normalization over the declared groups divides
that missing weight away. See [state character](state_character.md).

A dominant-character change of an adiabatic band is not itself a surface hop.
It says the chemical/spatial character of the eigenstate changed; individual
accepted hopping events still require engine-level trajectory histories.

**Between-file SEM is not an ensemble error bar.** It measures the spread
between the SHPROP files given, which share a trajectory and are often launched
from correlated initial conditions. It can be far too small.

**The cosine transform can ring negative.** The default spectral density is
the cosine transform of a truncated, windowed VACF, so it can go negative near
sharp features. Band fractions and centroids computed across a negative region
are unreliable; `describe` reports the negative sample count and returns `NaN`
for a centroid rather than a misleading number.

**Segment spread is not statistical error.** Segments cut from one trajectory
are not independent samples. Their spread measures drift along the run.

**Spectrum files carry no provenance.** A two-column
`spectral_density_*.txt` records frequency and intensity only. Convention,
window, smoothing, trajectory length and atom count are all unknown to the
reader, which is why raw integrals are not comparable across files unless you
know those matched.

**A Markovian rate matrix is assumed, not demonstrated.** `kinetics` fits
constant rates on a fixed graph. Memory effects, inhomogeneity across initial
conditions, and states whose character changes mid-trajectory can break that
picture, and none of them announce themselves as a bad fit. A high R² means
the model *can* reproduce the curves, not that the mechanism is right.
Projection-weighted populations can reduce one source of state-label ambiguity,
but they do not by themselves prove Markovian kinetics.

**Individual rates are frequently unidentifiable.** Population curves constrain
the eigenvalues of K far better than its entries. The module applies five
independent tests — a Jacobian column the residuals are blind to, relative
standard error above 0.1, correlation above 0.95 with another rate,
participation above 0.1 in the numerical null space of the Jacobian, and a
result sitting on an optimizer bound — and marks such rates `identified: false`
with a stated reason. None of the five replaces the others. Those numbers exist
in the output so the diagnostic can be audited; they are not results. The
eigenvalue timescales are the quantity to quote.

**Bootstrap intervals cannot make an unidentified rate look precise.** An
interval is reported only when the rate was identified in at least 80% of the
successful resamples, and it is taken over those resamples alone. Below that,
`bootstrap_ci_per_ns` is null and the count is reported. A converged optimizer
is not determined data.

**The coupling ceiling in the archived interface runs was engineered, not
accidental.** Repeated values at exactly 0.6 eV are consistent with an
intentional upstream NAC safety ceiling placed inside the numerical-safety
region of `ħ/dt` (0.658 eV at dt = 1 fs). Whether it turned any statistic into
a lower bound depends on the upstream rule, which a NATXT file does not record;
it has to be declared, and it is not assumed.

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

## Plan status after version 0.5

New in 0.5:

- `character-populations`: projection-weight each original SHPROP history with
  frame-resolved PROCAR subsystem character before averaging.
- SHPROP header metadata preservation for `NAMDTINI`, `NSW`, `BMIN` and `BMAX`.
- Strict single-k-point PROCAR ionic-projection parsing with exact VASP-band
  matching and no nearest-energy tracking.
- Explicit linear or DISH-cyclic frame alignment, including a manifest
  `cycle_length` override for archived campaigns whose saved electronic frame
  count differs from header bookkeeping; mismatches are retained in provenance.
- Projection-quality diagnostics, dominant-character swaps, fixed-vs-projected
  population comparison, full input fingerprints and publication plots.

New in 0.4:

- transient population observables and explicit early/late regime comparison;
- first-passage branching with identifiability-aware bootstrap uncertainty;
- extraction-versus-recombination sink competition and crossover escape time;
- reviewer-facing campaign summary with observed/model-inferred/counterfactual
  labels.

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

1. **BCF campaign reanalysis:** the package can now perform both fixed-state and
   projection-weighted analyses, but the actual original SHPROP histories,
   verified state map and frame-resolved PROCAR files still have to be supplied
   to run the campaign. Saved fitted curves cannot replace them.
2. **Directional event analysis:** the inspected public classic engine writes
   averaged SHPROP, not individual hop histories. See
   [engine inspection](hopping_histories.md) for the exact source revision,
   limitations, and required logging. The installed DISH/DEV engine still needs
   verification before instrumenting it through the launcher.
3. **New initial-state propagations:** the analyzer compares results but does not
   generate them; prepare these with the installed engine via NAMD_Launcher.

Usage and statistical assumptions are documented in
[comparisons](comparisons.md). A lowest information-criterion score is not proof
of a mechanism, and bootstrapping correlated files does not produce an
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
| Collect frame-resolved PROCARs/projections | Projection-weight diagonal subsystem populations |
| Record the NAC handling policy used | Read couplings against ħ/dt and report a declared ceiling |

Inputs are accepted directly, whether or not a launcher produced them. When a
launcher manifest is present beside the inputs, its content and fingerprint
are imported. Parent campaign manifests can be passed explicitly. Upstream
claims are preserved without executing or trusting embedded artifact paths.
