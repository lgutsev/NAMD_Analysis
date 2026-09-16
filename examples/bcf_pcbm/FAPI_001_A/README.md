# FAPI_001_BCF_PCBM_A

The confirmed configuration for campaign **A**: a FAPI (001) slab with BCF and
PCBM, 446 atoms, a six-state NAMD basis.

These two files are what `namd-analysis character-prepare --preset bcf_pcbm
--campaign A` writes. They are here so the values can be read, diffed and
corrected without running anything, and so that a campaign that is *not* A has
something concrete to be written against.

```bash
namd-analysis character-prepare \
  --shprop-dir /path/to/NuTest \
  --projection-dir /path/to/FAPI_001_BCF_PCBM_A \
  --preset bcf_pcbm --campaign A \
  --frame-mode dish-cyclic \
  --out .
```

## Nominal state order

SHPROP columns, zero-based:

| column | state |
| --- | --- |
| 0 | time |
| 1 | an energy-like quantity |
| 2 | VBM |
| 3 | BCF |
| 4 | PCBM1 |
| 5 | PCBM2 |
| 6 | PCBM3 |
| 7 | CBM |

**This order is a property of the run that wrote the file.** Nothing in a
SHPROP table identifies which state a column holds; `character-prepare` infers
only that columns 2–7 are *the population block* (they lie in [0,1] and sum to
one, and the energy column does not), never which is which. Confirm the order
against the run's `inp`/`INICON` before trusting any fixed-column number.

## Why PCBM1/2/3 collapse to `PCBM`

They are three separate basis states but one physical subsystem. Keeping them
apart in the fixed map would report three populations that no measurement
distinguishes, and summing them loses nothing: the state map groups them.

```
VBM  -> [2]
BCF  -> [3]
PCBM -> [4, 5, 6]
CBM  -> [7]
```

`complete_population` is true and `recombined_group` is `VBM`, so survival is
defined as `1 - P(VBM)`.

## Why the dynamic groups are coarser

The PROCAR projection partitions **atoms**, not states:

```
perovskite / BCF / PCBM
```

There is no `VBM` or `CBM` group, and there cannot be one. Both the valence
band maximum and the conduction band minimum are perovskite-localized in this
structure, so atomic localization alone cannot separate them — a band sitting
on the perovskite atoms is perovskite, and which band edge it is requires
energy ordering, which this workflow deliberately does not use for state
identity. Adding a `VBM` group here would be a guess dressed as a measurement.

## The fixed map is reference-only

The fixed map assigns each column to one group for the whole trajectory. That
is exactly the assumption the character workflow exists to test: an adiabatic
band can exchange BCF/PCBM/perovskite character mid-trajectory, at which point
the column keeps its old label and the population it reports becomes a mixture.

Run both and compare. `fixed_vs_projected_summary.csv` reduces the difference
to one row per shared group name — `BCF` and `PCBM` here, since `VBM` and `CBM`
have no projection counterpart. A large difference means the fixed labelling
was wrong somewhere; a small one means it was adequate for this campaign.
Neither is a defect, and neither number is a transfer rate.

## Atom partition

446 atoms, disjoint, covering 1..446 exactly:

| group | one-based PROCAR ions | count |
| --- | --- | --- |
| perovskite | 2–28, 134–268, 283–363, 364–417, 420–446 | 324 |
| BCF | 1, 29–46, 119–133 | 34 |
| PCBM | 47–118, 269–282, 418–419 | 88 |

The ranges are not contiguous per subsystem because the structure was not built
in subsystem order. `character-prepare` checks this partition against the
representative PROCAR's ion count and refuses if they differ — it does not trim
or rescale. `min_projection_weight` is 0.5, a reporting threshold only: nothing
is discarded or renormalized on account of it.

## Do not reuse this for B or C

Campaign letters elsewhere in this repository refer to differently built
structures. The state order and the atom partition here were established from
the structure that was actually built for A; neither transfers. The preset
registry contains only A for that reason, and asking for an unregistered
campaign is an error rather than a fallback to this one.
