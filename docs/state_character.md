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

## PROCAR restrictions

The built-in reader is deliberately narrow.  It reads the per-ion `tot`
projection for every band and requires one k-point. Multiple spin components
are rejected instead of being selected or combined silently.  For SOC-style
files, the first ionic table after a band is interpreted as the scalar charge
projection.

The required `BMIN..BMAX` bands must exist in every PROCAR.  Bands are matched
by their VASP band number, never by nearest energy.

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

## Outputs

`character_populations.csv` contains projection-weighted physical populations
and between-file SEM. `projection_character.csv` contains the long-form
frame/band/subsystem character and raw projection quality.
`character_swaps.csv` records changes in the dominant subsystem character of a
fixed adiabatic band. `shprop_alignment.csv` records the exact `NAMDTINI`,
`NSW`, header-derived cycle length, cycle length actually used, basis window,
and projection frames used by every SHPROP file.

When a fixed state-map group has the same name as a projection group,
`fixed_vs_projected.csv` compares the two definitions directly.  A large
mismatch is evidence that fixed band labels are materially changing the
physical interpretation.

A dominant-character swap is **not** a surface hop.  It is a change in the
chemical character of an adiabatic eigenstate.  Likewise, projection-weighted
PCBM population is not device extraction: the finite NAMD cell still contains
no electrode or irreversible long-range transport.
