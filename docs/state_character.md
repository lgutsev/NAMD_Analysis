# Frame-dependent physical state character

Version 0.5 can combine SHPROP populations with time-dependent subsystem
projections from PROCAR.  This addresses a limitation of fixed state maps: an
adiabatic band index can exchange BCF, PCBM, or perovskite character during an
MD trajectory.

For SHPROP history `r`, the physical population of subsystem `g` is

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
is dominant, the most common character, the number of dominant-character swaps,
the median dominant weight, and the mixed fraction under the dominance
threshold.

All of these are diagnostics.  No sample is discarded, repaired or reweighted on
account of them.  A captured projection of exactly zero is the one exception: it
would divide by zero, so it is a loud error naming the file and the bands.

## Outputs

`character_populations.csv` contains projection-weighted physical populations
and between-file SEM. `projection_character.csv` contains the long-form
frame/band/subsystem character and raw projection quality.
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

A dominant-character swap is **not** a surface hop.  It is a change in the
chemical character of an adiabatic eigenstate.  Likewise, projection-weighted
PCBM population is not device extraction: the finite NAMD cell still contains
no electrode or irreversible long-range transport.
