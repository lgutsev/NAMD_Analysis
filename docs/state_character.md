# Frame-dependent subsystem character

Version 0.5 can combine SHPROP populations with time-dependent subsystem
projections from PROCAR.  This addresses a limitation of fixed state maps: an
adiabatic band index can exchange BCF, PCBM, or perovskite character during an
MD trajectory.

For SHPROP history `r`, the reported subsystem population is

```text
P_g^r(t) = sum_i P_i^r(t) w_ig[f_r(t)]
```

where `w_ig` is the normalized PROCAR projection of adiabatic state `i` onto
subsystem `g`, and `f_r(t)` is the electronic-structure frame actually used by
that SHPROP history.  The ensemble mean is formed only after this projection:

```text
Pbar_g(t) = (1/N) sum_r P_g^r(t)
```

Do **not** projection-weight `SHPROP.master`.  Different SHPROP files can carry
different `NAMDTINI` values, so averaging the adiabatic populations first
removes the phase needed to select the correct PROCAR frame.

## What this quantity is, and what it is not

It is a **projection-weighted diagonal subsystem population**.  The
documentation and the reports do not call it "the physical population" without
that qualification, because two approximations sit between it and the actual
occupation of a subsystem, and both come from what the input files contain
rather than from any choice made here.

**It drops the coherences.**  The subsystem occupation of the propagated
electronic state is

```text
Tr[rho(t) P_g] = sum_i rho_ii(t) <i|P_g|i>  +  sum_{i != j} rho_ij(t) <j|P_g|i>
```

SHPROP records only the diagonal `rho_ii`.  A PROCAR records only diagonal,
band-by-band projections, so `<j|P_g|i>` for `i != j` is not in the inputs
either.  The second sum is therefore **omitted, not estimated**, and nothing in
these reports bounds its size.  What is computed is the first sum alone.

When does that matter?  When two states of different subsystem character are
strongly and persistently coherent, the true subsystem occupation oscillates
about the diagonal value and the diagonal result misses that oscillation.  In
surface hopping the population dynamics is itself formulated on the diagonal, so
the diagonal population is the natural companion observable — but it is an
approximation to the subsystem occupation, not the thing itself.

**The weights are a conditional share, not a fraction of the band.**

```text
w_ig = W_ig / sum_g W_ig
```

`W_ig` is PAW-sphere weight, which does not sum to one over any structure:
interstitial density and any undeclared atom fall outside every sphere.  The
normalization divides that missing weight away, so `w_ig` answers "of the weight
that landed inside a declared group, what share was `g`", not "what fraction of
band `i` is `g`".  `captured_projection` reports how much weight there was to
normalize, which is why it is reported for every frame and band rather than
folded away.  See [Projection normalization and quality](#projection-normalization-and-quality).

The full statement is carried in `report.json` under `population_definition`, so
it travels with the numbers.  The population columns are named
`projection_weighted_diagonal_population` in the CSVs for the same reason.

## Inputs

`character-populations` needs the original SHPROP files, their ordinary state
map, a projection manifest, and an atom-group map.

The state map must declare one population column per adiabatic state in the
same order as `BMIN..BMAX`.  `BMIN`, `BMAX`, `NAMDTINI`, and (for cyclic DISH)
`NSW` are read from each SHPROP header.  Missing metadata is an error rather
than something inferred from filenames or population shapes.

A projection manifest can list files explicitly:

```json
{
  "frames": [
    {"frame": 1, "procar": "frames/0001/PROCAR"},
    {"frame": 2, "procar": "frames/0002/PROCAR"}
  ],
  "cycle_length": 2
}
```

or use a format pattern:

```json
{
  "procar_pattern": "frames/{frame}/PROCAR",
  "first_frame": 1,
  "last_frame": 1999,
  "frame_step": 1,
  "cycle_length": 1999
}
```

Relative paths are resolved from the manifest directory.  For cyclic DISH,
`cycle_length` is optional but important when the number of saved electronic
frames is known to differ from `NSW - 1` in an archived input/header.  When it
is supplied it is used as the electronic-frame period and the mismatch with the
header-derived period is recorded in `shprop_alignment.csv` rather than hidden.

The atom-group map uses one-based PROCAR ion indices. Ranges are accepted so a
large perovskite slab does not require hundreds of explicit integers:

```json
{
  "groups": {
    "perovskite": ["1-135"],
    "BCF": ["136-166"],
    "PCBM": ["167-236"]
  },
  "complete_atoms": true,
  "min_projection_weight": 0.5
}
```

With `complete_atoms: true`, every PROCAR ion must appear exactly once across
the declared groups.

## Frame alignment

The alignment rule is explicit because it is part of the physics provenance.

For the inspected public DISH implementation the electronic frame is

```text
RTTIME = mod(tion + NAMDTINI - 1, NSW - 1)
if RTTIME == 0: RTTIME = NSW - 1
```

Use:

```bash
namd-analysis character-populations \
  --files 'run/SHPROP.*' \
  --config state_map.json \
  --projection-manifest projection_manifest.json \
  --atom-groups atom_groups.json \
  --frame-mode dish-cyclic \
  --out results/character
```

Use `--frame-mode linear` only when the producing engine uses
`frame = NAMDTINI + tion - 1`.  The package never guesses between these modes.
For a private or modified engine, verify its frame rule before interpreting the
result.

## Before you run: preflight

```bash
namd-analysis character-preflight --files ... --config ...   --projection-manifest ... --atom-groups ... --frame-mode dish-cyclic
```

or `character-populations --preflight`.  Both take the same arguments and run
the same planning code the real analysis runs, so what preflight checks is
exactly what the run will do.

Preflight reads the SHPROP headers, the projection manifest, and the *header*
of one representative PROCAR.  It parses no projection data and produces no
populations.  It reports the basis window and how it compares with the declared
population columns, the distinct `NAMDTINI` values and row counts, the cycle
period from both sources and whether they disagree, the electronic frames each
history will visit, how many PROCARs will actually be parsed, the representative
PROCAR's ion/band/k-point/spin structure, and whether the atom groups cover it.

Problems are collected rather than raised, so one run lists everything that is
wrong.  The command exits non-zero if anything would block the analysis.

## How SHPROP files are read

Preflight reads headers, row counts and each history's first and last row. No
table is materialized, so its memory does not grow with file size -- on a
47 MiB history the scan peaks at 0.2 MiB against 159 MiB for a whole-file read.
That matters: a campaign of five ~889 MB histories exhausted memory during
preflight before this.

The analysis streams each history in row chunks and folds them into a running
ensemble mean and variance (Welford), so no `(nfiles, nrows, ncolumns)` stack
is ever built. The projection-weighted and fixed-column populations come from
the same pass, so a history is read once.

Chunking changes residency only. The mean, the between-file standard error and
every validation -- time-grid identity, strict increase across chunk joins,
population range, conservation -- are identical at any chunk size, and the
cyclic wrap maps the same whether it falls inside a chunk or on a boundary.
Tests pin all of that.

| flag | effect |
| --- | --- |
| `--shprop-chunk-rows N` | rows held at once |
| `--shprop-io-mode auto` | chunk size chosen from the largest history (default) |
| `--shprop-io-mode stream` | always chunk |
| `--shprop-io-mode memory` | one chunk per history, the whole table at once |
| `--accumulator-memmap-dir DIR` | spill the running mean/variance to memory maps when large |

There is one code path: `memory` is a chunk the size of the table, not a second
implementation, so the two cannot drift apart.

Per-history projected populations are retained only when they are small enough
to be worth keeping; the mean and standard error never depend on them.

## Which PROCAR frames are parsed

The required electronic frames are computed from the SHPROP metadata before any
PROCAR is opened, and only those files are parsed.  A cyclic campaign commonly
declares one PROCAR per MD frame while the trajectories revisit a much shorter
cycle; parsing the whole manifest to use a fraction of it is the dominant cost.
On a synthetic 400-frame manifest (115 MB) whose histories visit a 20-frame
cycle, this parses 20 files instead of 400.

Within each file, only the bands in the `BMIN..BMAX` window are retained.  Other
bands are walked, so the band count and ion-table structure are still checked,
but their projections are never parsed or stored.  Files are read as a stream
rather than loaded whole.

Selection never weakens validation.  The manifest is still checked in full:
every declared PROCAR must exist, frame numbers must be unique and positive, and
a declared `cycle_length` must be completely covered by the manifest.  Dropping
those checks to save time would let a genuine gap go unnoticed until it changed
a number.

`report.json` records both the declared manifest range and the subset consumed,
under `frame_consumption`.

## PROCAR restrictions

The built-in reader is deliberately narrow.  It reads the per-ion `tot`
projection for every band and requires one k-point. Multiple spin components
are rejected instead of being selected or combined silently.  For SOC-style
files, the first ionic table after a band is interpreted as the scalar charge
projection.

The required `BMIN..BMAX` bands must exist in every PROCAR.  Bands are matched
by their VASP band number, never by nearest energy, and the numbers are read
from the file rather than assumed to run `1..NBANDS`.

Dialects, and what the reader does with each:

| dialect | behaviour |
| --- | --- |
| plain single-k-point, `LORBIT=10` or `11` | read |
| lm-decomposed (many orbital columns) | read; the `tot` column is located by name |
| scientific notation, wide whitespace, CRLF, UTF-8 BOM | read |
| `LORBIT=12` (charge table then phase table) | first table read, phase table skipped |
| SOC (total, mx, my, mz tables per band) | first table read, magnetisation tables skipped |
| multiple k-points | **rejected** |
| multiple spin components | **rejected** |
| truncated, duplicated or out-of-order ion rows | **rejected** |
| band count disagreeing with the header | **rejected** |

The dangerous case is the second ionic table: reading a phase or magnetisation
table as though it were the charge projection would produce plausible,
signed-wrong weights.  The scanner only resumes on the next `band` line, so the
extra tables are skipped, and `tests/test_character_hardening.py` pins that.

## Projection normalization and quality

For a band `i`, raw subsystem projections are summed over the declared ions and
normalized as

```text
w_ig = W_ig / sum_g W_ig
```

The raw summed projection is retained as `captured_projection`.  This matters
because PAW-sphere projections need not sum to one.  A low captured projection
means that the normalized chemical character is less trustworthy even though
its normalized weights still sum to one.  The configured
`min_projection_weight` is therefore a quality-reporting threshold, not an
automatic renormalization or deletion rule.

`report.json` carries, under `projection_quality`: the minimum, 1st, 5th, 50th
and 95th percentiles of the captured projection; per-band medians, minima and
the frame at which each minimum occurs; the worst individual frame/band samples
with their source file; the worst frames by their minimum across bands; and the
count and fraction below the threshold, globally and per band.  Under
`dominance` it carries, per band, the fraction of frames on which each subsystem
is dominant, the most common character, the dominant-character swap counts
split by resolution (`adjacent_swaps`, `changes_across_a_frame_gap`,
`cycle_wrap_swaps`), the median dominant weight, and the mixed fraction under
the dominance threshold.  Those counts use the transition set described in
[What counts as a swap](#what-counts-as-a-swap), so they agree with
`character_swaps.csv` by construction.

All of these are diagnostics.  No sample is discarded, repaired or reweighted on
account of them.  A captured projection of exactly zero is the one exception: it
would divide by zero, so it is a loud error naming the file and the bands.

## Outputs

`character_populations.csv` contains the projection-weighted diagonal
subsystem populations described above, in a column named
`projection_weighted_diagonal_population`, with between-file SEM.
`projection_character.csv` contains the long-form frame/band/subsystem
character and raw projection quality.
`character_swaps.csv` records changes in the dominant subsystem character of a
fixed adiabatic band. `shprop_alignment.csv` records the exact `NAMDTINI`,
`NSW`, header-derived cycle length, cycle length actually used, basis window,
and projection frames used by every SHPROP file.

`projection_quality_by_band.csv` summarizes capture and dominance per band.

When a fixed state-map group has the same name as a projection group,
`fixed_vs_projected.csv` compares the two definitions directly and
`fixed_vs_projected_summary.csv` reduces that to one row per group: the maximum
absolute difference and the time at which it occurs, the RMS difference, the
integrated absolute difference, and the mean signed difference.  This is the
quickest way to see whether dynamic character changes the interpretation at all.
Groups present in only one of the two maps are listed rather than silently
dropped.

`shprop_alignment.csv` additionally records the first five and last five
electronic frames each history uses, the number of unique frames, the wrap count
in cyclic mode, and whether the period came from `NSW-1` or from an explicit
manifest override.

## What counts as a swap

There is **one** definition of an examined transition, and every swap statistic
in every output is built from it: the rows of `character_swaps.csv`, the summary
under `character_swaps` in `report.json`, and the per-band counts under
`dominance` and in `projection_quality_by_band.csv`.  Summing the per-band
`dominant_character_swaps` reproduces the campaign total exactly, and no
statistic can quietly use a different notion of adjacency from another.

A transition carries a `resolution`:

| `resolution` | meaning |
| --- | --- |
| `adjacent` | consecutive MD frames; a change here is located exactly |
| `across_gap` | the frames between were never loaded; the change happened *somewhere inside* the gap |
| `cycle_wrap` | the `period -> 1` step of a cyclic campaign |

`across_gap` exists because only the frames the histories visit are loaded, so
two consecutive rows of the projection need not be adjacent MD frames.  The
summary reports how many MD frames were skipped.  Swap counts are therefore a
lower bound on the number of character changes along the full MD trajectory.
When every examined frame is adjacent -- the usual case for a short cyclic
campaign -- the summary says so and the count is exact over the range examined.

### The cyclic wrap

Loaded frames are held in ascending order, so a plain scan over consecutive
entries would compare `1 -> 2 -> ... -> period` and stop.  In a cyclic campaign
that misses one step: the trajectory continues from `period` back to frame `1`,
and a character change there is a real change in the state the dynamics
occupies.  It is examined, and labelled `cycle_wrap` rather than `adjacent`,
because the nuclear geometry does **not** evolve continuously across it — the
cyclic mapping restarts the MD run rather than continuing it, so the change
reflects that discontinuity as much as any physical evolution.  Read a
`cycle_wrap` swap as an artefact of cyclic re-use unless the trajectory is
genuinely periodic.

The step is only examined when a history actually takes it.  This is counted
from the resolved per-history frame series, not deduced from which frames
happen to be loaded: a campaign can load both frame `1` and frame `period`
without any single history stepping between them.

When it is *not* examined, the reason is stated rather than left silent.
`report.json` carries `cycle_wrap` under both `character_swaps` and `dominance`,
and the CLI prints one line for it.  The `state` field is one of:

| `state` | meaning |
| --- | --- |
| `examined` | a history takes the step and it was compared |
| `not_traversed` | no history reaches `period` and continues; the step is not part of this campaign |
| `not_cyclic_or_not_declared` | linear mode, or a projection series loaded without an alignment plan |
| `traversal_unknown` | a period is known but traversal was not supplied, so the step is excluded rather than guessed |
| `frames_not_loaded` | the wrap is traversed but one of the two frames is absent from the projection |
| `degenerate_period` | the period is 1, so the wrap would compare frame 1 with itself |

A dominant-character swap is **not** a surface hop.  It is a change in the
chemical character of an adiabatic eigenstate.  Likewise, projection-weighted
PCBM population is not device extraction: the finite NAMD cell still contains
no electrode or irreversible long-range transport.
