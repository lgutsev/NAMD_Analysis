# Is band mixing robust to the projection's missing weight?

[Issue #4](https://github.com/lgutsev/NAMD_Analysis/issues/4): in campaign A,
band 981 is the one band with apparent BCF/PCBM mixing, and it is also the
band whose PAW projection captures least of it (median `captured_projection`
0.445; 1633 of 1999 frames below the 0.5 reporting threshold). Two readings
fit that:

1. band 981 is genuinely delocalized or hybridized, and its low capture is
   real interstitial amplitude; or
2. its mixing is an artefact of normalizing a small captured weight.

This page describes three tools that separate those readings. They do not
change or "repair" anything. `projection_character.csv` from
`character-populations` remains the authoritative result of that method.
Everything here reads it and writes new files.

| part | command | runs where | what it answers |
| --- | --- | --- | --- |
| audit | `character-robustness` | login node, seconds | does the mixing survive the absolute weights, and how open does the projection leave the answer? |
| focus report | `character-robustness --focus-band 981` | same run | the six diagnostic panels for band 981, with the Issue #4 frames and the crossing window marked |
| full-space check | `character-fullspace-prepare`, then `character-fullspace-compare` | VASP + Bader on a handful of frames | where the weight the projection missed actually is |

## Three facts the audit is built on

**Normalization cannot change a ranking or a ratio.** `w_ig = W_ig /
captured` divides every declared fragment by the same number, so the
dominant declared fragment and the BCF:PCBM ratio are the same before and
after. "Dominant before normalization" therefore means the largest of the raw
fragments *and* the uncaptured weight `1 - captured`. It differs from the
normalized dominant exactly when more of the band is uncaptured than any one
fragment holds. The declared-only argmax is reported too, as
`dominant_raw_declared_only`, and the invariance is counted on the data
(`declared_argmax_invariance_violations`, expected zero).

**What normalization decides is where the missing weight goes.** It
allocates the uncaptured `1 - captured` to fragments in proportion to what
they already hold. A full-space partition is a different allocation. If each
fragment's projected weight lies in that fragment's region, its full-space
fraction satisfies `W_g <= f_g <= W_g + (1 - captured)`. For the mixing
metric `min(f_BCF, f_PCBM)` this gives three numbers per sample, always
ordered:

| column | allocation |
| --- | --- |
| `pair_min_worst_case` | none of the uncaptured weight on BCF or PCBM: `min(W_BCF, W_PCBM)` |
| `pair_min_normalized` | proportional, as `character-populations` does: `min(w_BCF, w_PCBM)` |
| `pair_min_best_case` | as much of it on the pair as the budget allows |

At a threshold τ, a sample is then *mixed under every allocation* (worst ≥ τ),
*mixed under some allocation* (best ≥ τ > worst), or *not mixed under any
allocation*. The bracket is an idealisation, not a measurement.
`bracket_assumptions` in every report states its assumptions: unit-normalized
bands, projector weight assigned to its own fragment, and nothing known about
the rest.

**Low capture and mixing are different axes.** A band can have low capture
and a single dominant fragment, or high capture and genuine two-fragment
weight. The audit reports them separately and never infers one from the
other. The synthetic tests pin four cases
(`tests/test_projection_robustness.py::FourScenarioTests`):

| case | normalized mixed? | survives raw weights? | capture |
| --- | --- | --- | --- |
| genuinely mixed raw weights | yes | yes, under every allocation | high |
| tiny raw weights, strongly normalized | yes | no; the uncaptured weight dominates | very low |
| high-capture pure state | no | no, under any allocation | high |
| low-capture state without mixing | no | open: mixed under *some* allocation | low |

The last row matters. When 70% of a band is unseen, the projection can't
rule out mixing, and the audit says so. It still doesn't call that band
mixed.

**A correlation does not decide Issue #4.** Both readings predict that
low capture goes with apparent mixing: genuine delocalization puts weight
into the interstitial and onto several fragments at once, and an artefact
inflates small weights most where the denominator is smallest. The
correlations are reported (Pearson and Spearman over frames, with no
p-value, because consecutive MD frames are autocorrelated). What bears on
the question is whether mixing survives the raw weights, and ultimately the
full-space measurement.

## PROCAR print precision

VASP prints each ion's `tot` to three decimals (F7.3), and prints the band's
grand total on a `tot` row that it summed *before* rounding. A band spread
thinly over many ions loses weight to that rounding: an ion at 0.0004 prints
as 0.000. The character analysis sums the printed per-ion values, so part of
a delocalized band's low capture could be a printing effect, not
interstitial density. With 324 perovskite ions, the rounding bound is up to
±0.16 of the band.

`character-robustness --projection-manifest ... --atom-groups ...` re-reads
the marked frames' PROCARs and writes `procar_resolution.csv`: per band and
group, the sum of printed values, the ions printed as zero, the interval the
true sum must lie in, and the printed grand total minus the sum. Where that
difference exceeds one print step, it is the weight rounding moved. Where it
doesn't, the file can't distinguish "little was lost" from "the `tot` row was
summed after rounding", and the report says so. `read_procar_ion_totals` now
records the `tot` row and the print precision alongside what it already
returned. The values the character analysis reads are unchanged.

## LORBIT, RWIGS and what a sensitivity test would test

| LORBIT | projection | does an RWIGS sweep test it? |
| --- | --- | --- |
| ≥ 10 | onto the PAW projector functions; **RWIGS is ignored by VASP** | no: the PROCARs would come out unchanged |
| < 10 | integration inside Wigner-Seitz spheres of radius RWIGS | yes, partially |

A PROCAR doesn't say which it was: LORBIT 1 and 11 write the same
lm-decomposed layout, and 0 and 10 the same l-decomposed one. The value is
read from a production frame's INCAR, OUTCAR or `vasprun.xml`, which must
agree. Every report carries it under `projection_method`, including
`rwigs_sweep_tests_these_weights`. With nothing to read it from, the method
is recorded as `undetermined`, and no conclusion here depends on assuming it.

The full-space check below is the sensitivity test that is valid for both
methods.

## Running the audit on campaign A

```bash
ROOT=/ddnB/work/lgutsev/MD/FAPI_MD_NAMD_2026
namd-analysis character-robustness \
  --projection-character $ROOT/NuTest/A/projection_character.csv \
  --character-report $ROOT/NuTest/character_test_A_1030646/report.json \
  --focus-band 981 \
  --mark issue4_low_capture=1273,1286,1302 \
  --profile examples/bcf_pcbm/production_profile.json --configuration A \
  --projection-manifest $ROOT/NuTest/A/projection_manifest.json \
  --atom-groups $ROOT/NuTest/A/atom_groups.json \
  --out $ROOT/FAPI_001_BCF_PCBM_A/analysis/robustness_981_$(date +%Y%m%d)
```

`--profile/--configuration A` marks `crossing_A = 1320:1331`, A's own mixing
window. **B1–B4 are windows of configuration B's trajectory.** B shares A's
atom indexing but not its coordinates, so frame 1488 of B is not frame 1488
of A, and B1–B4 are never drawn on an A figure. To audit B, pass B's
projection table and `--configuration B`; B1–B4 are then marked, each under
its own name. Band numbers are also per configuration. Use the focus bands
from B's own provenance, not 981 by default.

Outputs:

| file | contents |
| --- | --- |
| `robustness_samples.csv` | every frame × band: `captured_projection`, `total_projection`, `W_g` and `w_g` per fragment, uncaptured weight, amplification `1/captured`, dominant before/after normalization, the three pair minima, pair balance, and `allocation_class_at_<τ>` per threshold: whether the mixing survives every allocation of the uncaptured weight, only some, or none |
| `robustness_by_band.csv` | per band: capture median/min/percentiles, remainder, invariance check, pair-minimum medians, Spearman/Pearson of capture against each normalized fraction and against the mixing metric |
| `mixing_sensitivity.csv` | mixed-sample counts over the full grid of normalized threshold × minimum capture × minimum raw weight on each pair fragment, per band and campaign-wide. Every row states how many samples its capture condition set aside |
| `robustness_summary.json` | everything above, plus capture strata, the allocation classes per threshold, the `question` block, `projection_method`, provenance |
| `band_981_focus.{csv,json,png,pdf}` | the focus report: every frame of band 981 with its window and mark labels, and a figure with capture vs frame, raw and normalized weights vs frame, capture vs normalized BCF, capture vs normalized PCBM, and raw BCF vs raw PCBM |
| `band_981_mixing_sensitivity.{png,pdf}` | mixed-sample count against the normalized threshold, one line per raw minimum |
| `procar_resolution.csv` | the print-precision check on the marked frames |

The marked frames are the Issue #4 frames (`--mark`), and automatic marks
chosen by stated, deterministic rules: the lowest-capture frames, the
highest-capture frames (the controls), the median-capture frame, and the
most-mixed frames. Each mark records its rule.

### Reading the answer to "mixed only when the absolute weight is small?"

`robustness_summary.json` → `question` lays out, for band 981 at each
threshold:

- `n_mixed_normalized`: samples mixed after normalization, the Issue #4
  count;
- `n_that_survive_in_raw_weights`: of those, how many clear the same
  threshold in `W` (the worst-case allocation);
- `n_in_lowest_capture_stratum`: how many sit in the band's lowest-capture
  quartile;
- `median_capture_mixed` against `median_capture_not_mixed`.

If normalization created the mixing, mixed samples fail the raw test and
cluster in the lowest stratum. If the raw weights carry it, they survive and
occur across strata. `mixing_sensitivity.csv` shows the same thing in
finer detail.

No threshold is preferred. The capture grid always includes 0, meaning
every sample, and the existing 0.5 reporting threshold. A capture minimum
*conditions* a count, and each row says how many samples it set aside.

Band-981 capture sits near 0.45, so a normalized fraction of 0.10 is a raw
weight near 0.045. The raw-minimum grid runs from 0 to 0.10 to cover that
scale.

## The full-space check

The projection can't say where the uncaptured ~55% of band 981 is. A
band-resolved density can. The check integrates `|ψ_981(r)|²` over a fixed
partition of the whole cell, sums it into fragments, and compares.

1. **Prepare** (no VASP):

   ```bash
   namd-analysis character-fullspace-prepare \
     --projection-character $ROOT/NuTest/A/projection_character.csv \
     --projection-manifest $ROOT/NuTest/A/projection_manifest.json \
     --atom-groups $ROOT/NuTest/A/atom_groups.json \
     --band 981 --bands 981,977,978 \
     --include issue4_low_capture=1273,1286,1302 \
     --profile examples/bcf_pcbm/production_profile.json --configuration A \
     --account loni_perovsk27 --partition <a VASP partition> \
     --out $ROOT/FAPI_001_BCF_PCBM_A/analysis/fullspace_981
   ```

   Frames are chosen by stated rules, and every reason a frame was chosen is
   recorded. The rules cover the worst-capture frames (the Issue #4 frames
   are also included by name), the frames where PROCAR reports the most
   BCF/PCBM mixing, a high-capture control, a median-capture frame, and, for
   each crossing window of this configuration, its most-mixed frame. Bands
   977 and 978 are well captured and nearly pure, and exchange character
   inside `crossing_A`. They are the *method control*: if the full-space
   partition reproduces PROCAR for them but not for 981, the difference is
   specific to 981.

   It writes `INCAR.scf` and `INCAR.parchg` per frame, derived from the
   production INCAR. Every change is listed with its reason in
   `validation_manifest.json`, and frames whose INCARs differ are refused.
   It also writes `frames.tsv`, the production PROCAR values for every
   sample, the projection method, and `run_fullspace_validation.sbatch`.

2. **Run** (`sbatch run_fullspace_validation.sbatch`, with `VASP_CMD` set,
   and `bader` and `chgsum.pl` available). One array task per frame:
   - a single-point SCF with the production settings, including LORBIT, so
     its PROCAR (`PROCAR.rerun`) can be checked against the production one;
   - `LPARD` band-decomposed densities from that WAVECAR;
   - Bader basins of `AECCAR0 + AECCAR2`, with each band's PARCHG integrated
     over them (`bader PARCHG -ref CHGCAR_sum -vac off` → `ACF_<band>.dat`).
     The basins come from the frame's total density, not from the band being
     measured, so the partition is fixed before the band is integrated.

3. **Compare**:

   ```bash
   namd-analysis character-fullspace-compare \
     --campaign $ROOT/FAPI_001_BCF_PCBM_A/analysis/fullspace_981 \
     --voronoi --out $ROOT/FAPI_001_BCF_PCBM_A/analysis/fullspace_981_compare
   ```

   `--voronoi` adds a second, independent partition: nearest-atom cells
   computed from the PARCHG files themselves. `fullspace_comparison.csv`
   gives, per frame, band and partition:
   - the full-space fraction of each fragment beside `w_g` and `W_g`;
   - `uncaptured_allocated_to_g`, where the weight PROCAR missed actually
     went, beside `uncaptured_proportional_share_g`, where normalization
     put it;
   - the full-space pair minimum beside its worst, normalized and best-case
     PROCAR values, and their ratio;
   - whether the full-space fractions fall inside the allocation bracket;
   - how closely the rerun reproduced the production projection.

## What the numbers can settle, and how

The task is settled by one of these, and the code encodes none of them.
`fullspace_summary.json` → `readings` gives, for each threshold, over the
validation frames PROCAR calls mixed: how many are also mixed in full space,
and the median and range of the full-space/normalized ratio. Frames PROCAR
does not call mixed are counted too, so mixing that appears only in full
space is visible.

| statement | supported where |
| --- | --- |
| band 981 remains comparably BCF/PCBM-mixed under the spatial partition | PROCAR-mixed frames stay mixed at the same threshold, and the ratio is near one |
| the qualitative mixing remains but its magnitude changes substantially | they stay mixed at a lower threshold, but the ratio departs substantially from one |
| the apparent mixing largely disappears once the missing density is included | few stay mixed at any threshold, and the ratio is small |

"Near one" and "substantially" are deliberately not given a number in the
code. Read them off the ratios and the control bands.

Before relying on a comparison, check:

- `rerun_max_abs_raw_difference` is small. If not, the validation's SCF did
  not reproduce the production projection, and band 981 of the rerun may be
  a different state from the band the character analysis used.
- `fullspace_integral` is as expected. A PARCHG is the plane-wave part of
  `|ψ|²` on the FFT grid, and how VASP treats the one-centre PAW terms in it
  depends on version and settings. The integral is recorded so a departure
  is visible.
- The control bands 977/978 agree between PROCAR and full space. If they
  don't, the partition, not band 981, is the first suspect.
- The Bader and Voronoi partitions agree. Where they disagree, the answer
  depends on the partition, and that is itself the finding.

Every expected output that is missing is listed in `missing_outputs`. Its
sample is absent from the comparison and is not estimated.
