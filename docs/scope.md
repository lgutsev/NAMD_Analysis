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
- No forward/backward rates, first-passage yields, or extraction
  efficiencies. Averaged populations do not contain that information; see
  below.
- No parameter confidence intervals on fits. Only R², residual statistics and
  window diagnostics.

## Known limits in version 0.1

**A fixed state map cannot follow a moving state.** Grouping is by table
column. If a state's spatial character changes during the trajectory — which
is exactly what happens near a trivial crossing — the column keeps its label
and the group population becomes a mixture. Nothing here detects that.

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

## Next steps, in order

1. **Reproduce the BCF paper's existing figures.** This is the first concrete
   target: re-average the original `SHPROP.*` files, rebuild each population
   figure, and fix the lifetime definitions that the legacy script left
   ambiguous. Until that baseline is in place, nothing further is worth
   building.
2. **Pathway resolution.** Distinguish perovskite → PCBM, BCF → PCBM,
   PCBM → BCF and recombination as separate channels rather than reading net
   populations. This needs hopping histories, not averaged populations; the
   engine must be made to write them, and whether it can is an open question
   to settle before designing the analysis.
3. **Initial-state comparison.** A perovskite-initialized electron, a
   BCF-initialized one and a PCBM-initialized one answer different
   conditional questions. This package can define and compare the runs; the
   runs themselves go through the launcher.
4. **Extraction sink as an explicit model extension.** Adding
   `dP_PCBM/dt|escape = -k_esc P_PCBM` and sweeping `k_esc` asks how fast
   onward transport must be to outcompete return and recombination. Two
   conditions: the sink must actually remove population from the subsequent
   dynamics rather than being integrated afterwards, and `k_esc` must be
   labelled everywhere as physics supplied by the user, not determined by the
   interface calculation.
5. **Confidence intervals on fitted lifetimes**, and a multi-exponential or
   sequential kinetic model for traces a single exponential cannot describe.

## Relationship to NAMD_Launcher

| NAMD_Launcher | This package |
| --- | --- |
| Prepare inputs and submit jobs | Assign states to groups and validate the map |
| Generate and collect energies / NACs | Audit them, and document corrections such as caps |
| Configure and launch Hefei-NAMD | Analyze populations and competing channels |
| Audit jobs, preserve provenance | Fit kinetics with explicit windows and diagnostics |
| Basic output summaries | Compare systems and generate publication figures |

Inputs are accepted directly, whether or not a launcher produced them. When a
launcher manifest is present its provenance should be imported alongside this
package's own fingerprints; that import is not yet implemented.
