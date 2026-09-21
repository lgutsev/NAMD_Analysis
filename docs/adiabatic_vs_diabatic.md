# Adiabatic states, fragment character, and what a band label hides

This is the interpretive basis for the state-character workflow. It exists
because five things are routinely conflated, and the manuscript's argument
depends on keeping them apart.

## Five distinct quantities

| quantity | what it is | where it lives |
| --- | --- | --- |
| **adiabatic state population** | `P_i(t) = ρ_ii(t)` — which eigenstate the carrier occupies | SHPROP |
| **adiabatic-state character exchange** | the fragment composition `w_ig` of a **fixed band index** changing with time | PROCAR, frame by frame |
| **nonadiabatic hop** | population moving from state `i` to state `j` | SHPROP populations — *not* a PROCAR |
| **fragment population** | `P_g(t) = Σ_i P_i(t) w_ig[f(t)]` | what `character-populations` reports |
| **true diabatic transformation** | a unitary that removes the derivative coupling | **not performed anywhere in this package** |

The fourth is closer to a diabatic reading than a band-resolved one, but it is
**not a diabatization**. A formal transformation needs the off-diagonal
elements of the diabatic Hamiltonian and the coherences `ρ_ij`; SHPROP records
only the diagonal `ρ_ii`, and a PROCAR only diagonal, band-by-band
projections. Those terms are absent from the inputs, so they are omitted rather
than reconstructed.

## The two-state avoided crossing

Take two diabatic fragment states — say `|A⟩` localized on BCF and `|B⟩` on
PCBM — coupled by `V`. The adiabatic eigenstates are

```
|φ₁⟩ =  cos θ |A⟩ + sin θ |B⟩
|φ₂⟩ = −sin θ |A⟩ + cos θ |B⟩
```

with the mixing angle set by

```
tan 2θ = 2V / (E_A − E_B)
```

and the adiabatic gap

```
ΔE = √((E_A − E_B)² + 4V²)     →     ΔE_min = 2V   at resonance
```

As nuclear motion sweeps the diabatic detuning `E_A − E_B` through zero:

| far below resonance | at resonance | far above |
| --- | --- | --- |
| `θ → 0` | `θ = π/4` | `θ → π/2` |
| `φ₁ ≈ A`, `φ₂ ≈ B` | both are 50/50 mixtures | `φ₁ ≈ B`, `φ₂ ≈ A` |

**The band indices did not move. The orbitals they label swapped.**

A PROCAR projection onto A's atoms measures approximately `cos²θ` for `φ₁` and
`sin²θ` for `φ₂`. So the character curves this workflow reads are tracking
`θ(t)` sweeping through the crossing — which is exactly why a fixed band label
obscures physical transfer: **the label is constant while `θ` is not.**

That is the rigorous basis for the manuscript's claim. A fixed-column state map
assigns "band 978 = PCBM" for the whole trajectory. If `θ` sweeps, band 978 is
PCBM before the crossing and BCF after it, and the fixed map reports the
population of *whichever orbital that index happens to name at each moment* as
though it were one physical thing.

## Why the two failure modes run opposite ways

This is the part that is easy to get backwards.

**Staying on one adiabatic state through the crossing changes the fragment
identity.** The trajectory never hops; `P₁` stays at 1. But `φ₁` was A-like
before and is B-like after, so the charge has physically moved from BCF to
PCBM. *No hop, transfer occurred.*

**Hopping between the two adiabatic states at the crossing can preserve the
fragment identity.** Population moves from `φ₁` to `φ₂` exactly where both are
even mixtures. In the diabatic picture the charge stayed put. *A hop occurred,
no transfer.*

So:

- a **character swap is not a surface hop**;
- a **surface hop is not charge transfer**;
- and neither one alone licenses the phrase "charge-transfer event".

`character-crossings` enforces this on **one observable**: the
projection-weighted fragment population `P_g = Σ_i P_i w_ig`, and whether it
moved. Three outcomes for a character swap, and no others:

| outcome | label |
| --- | --- |
| no SHPROP population supplied — nothing was evaluated | `character_swap_population_not_evaluated` |
| `\|ΔP_g\| ≤ tolerance` — evaluated, and it did not move | `character_swap_without_projected_fragment_change` |
| `\|ΔP_g\| > tolerance` — evaluated, and it moved | `character_swap_with_projected_fragment_change` |

The first row must never be merged into the second. The second is a *measured
absence*: a population was read, and it stayed put. The first is no measurement
at all — a configuration-level scan reads `projection_character.csv`, `EIGTXT`
and `NATXT`, none of which carries a SHPROP population, so the question was
never asked. A run producing only the first row has said nothing whatever about
`P_g`, and no sentence of the form "no change accompanied these exchanges" may
be written from it. A null `projected_fragment_population_change` means **not
evaluated**, never zero; the `population_evaluated` column says so in its own
cell.

### Why `P_g` and not one term of the split

`P_g` is the projection of the occupied density onto a fragment, summed over
**every** band of the SHPROP basis. A re-ordering of band labels therefore
leaves it exactly invariant. A change in `P_g` is consequently *not* a
labelling artifact — it is a change in where the occupied density sits.

This corrects an earlier reading of this analysis, and the correction matters:

> **A character-driven change in `P_g` is not "no charge moved".**

If an occupied adiabatic state turns from BCF-like to PCBM-like while its
population `P_i` is unchanged, its density has moved from one fragment to the
other in real space. That is the first of the two failure modes above — the
adiabatic passage through an avoided crossing, no hop anywhere, and the charge
moved. An earlier
version of the classifier called exactly that case "no fragment transfer"
whenever `ΔP^pop ≈ 0`, on the reasoning that character evolution is a mere
relabelling. It is not, and that label is gone.

## Adiabatic states vs physical fragments: two different maps

The campaign uses **two deliberately different groupings**, and they are not
interchangeable.

**The nominal six-state adiabatic view**, useful for the band-resolved picture:

```
VBM, BCF, PCBM1, PCBM2, PCBM3, CBM
```

collapsed for the fixed-column comparison to `VBM / BCF / PCBM / CBM`.

**The dynamic fragment view**, from the PROCAR atom projection:

```
perovskite, BCF, PCBM
```

There is no `VBM` or `CBM` fragment group, and there cannot be one. Both band
edges are perovskite-localized, so **atomic projection cannot distinguish VBM
from CBM** — a band sitting on the perovskite atoms is perovskite, and which
edge it is requires energy ordering, which this workflow deliberately does not
use for state identity. Adding a `VBM` projection group would be a guess
dressed as a measurement.

This is why `fixed_vs_projected_summary.csv` compares only the shared names
(`BCF`, `PCBM`) and lists `VBM` and `CBM` as having no projection counterpart
rather than inventing one.

## What a difference between the two maps means

A large fixed-vs-projected difference means the fixed labelling was wrong
somewhere — that a band index stopped naming the orbital the map says it names.
It is **not** a flux, a transfer rate, or an extraction. Both series are
diagonal in the adiabatic basis, so the difference is about labelling alone and
says nothing about the coherences neither one contains.

## The symmetric split: what it describes, and what it cannot

Each step's change in `P_g` splits exactly, using the symmetric midpoint form:

```
ΔP_g   =   ΔP_g^pop                    +   ΔP_g^char
       =   Σ_i [P_i(t) − P_i(t−1)] · ½[w_ig(t) + w_ig(t−1)]
                                       +   Σ_i ½[P_i(t) + P_i(t−1)] · [w_ig(t) − w_ig(t−1)]
```

`ΔP_g^pop` is occupation moving between states at fixed character.
`ΔP_g^char` is an occupied state's own character evolving at fixed occupation.

**Both move the occupied density, and for physical reasons.** Neither is "the
real one" and neither is bookkeeping about labels. They are written to
`per_history_events.csv` as `occupation_redistribution` and
`character_evolution`, for the fragment the row names in `dominant_fragment`,
so that `ΔP_g = occupation + character` closes on that one fragment.

**The split never decides whether a change occurred.** That is `|ΔP_g|`'s job,
and only `|ΔP_g|`'s. The split only *describes* a change already established,
through the `decomposition_descriptor` column:

| description | meaning |
| --- | --- |
| `occupation_dominated` | ≥70% of the absolute movement is `ΔP^pop` |
| `character_dominated` | ≥70% of it is `ΔP^char` |
| `mixed` | neither term carries it alone |
| `no_movement` | neither term moved |

These are descriptions of **a bookkeeping convention**, not physical branching
fractions and not mechanisms. The split is one of infinitely many exact splits,
so its proportions are a choice of accounting rather than a measurement of two
competing processes. An event described as `character_dominated` moved the
projected occupied density **exactly as much** as an `occupation_dominated` one
did; what differs is the accounting, not the physics.

What the analysis genuinely cannot resolve is the *microscopic* process. No
surface-hopping record is an input here, so nothing above counts hops,
establishes that a hop occurred, or yields a hopping rate.

### The direction descriptor

`swap_direction_matches_projected_change` asks whether `P_g` moved the way the
dominance swap points — the fragment it moved to gaining what the one it left
lost. It runs on **`ΔP_g` itself**, never on one term of the split: testing the
occupation term alone would ask a question about occupation and report the
answer as though it had been about charge.

It is a **descriptor**, not a verdict. A swap and a population change can
co-occur without either driving the other, so agreement is not evidence of a
mechanism and disagreement is not evidence against one — and neither changes
the classification, which rests on `|ΔP_g|` alone. It is `None` where the
question is not well posed: no band's dominant fragment moved, or the swap
names **no single direction** — the usual symmetric exchange, one band going
BCF→PCBM while the other goes PCBM→BCF, where movement either way would match
one of the two.

## Couplings near ħ/dt

A coupling approaching `ħ/dt` (0.658 eV at 1 fs) is **numerically
pathological** — the finite-difference evaluation of the NAC has broken down
over one step — and is never read as a measurement of an arbitrarily large
physical matrix element. A magnitude shared *exactly* by many samples did not
come out of the dynamics: some upstream step put it there, and repeated values
near 0.6 eV are consistent with an intentionally imposed NAC safety ceiling.
This is not described as accidental clipping or as a defect. Nothing is
filtered, rescaled or rejected on account of it.

## Reading the outputs

| file | what it answers |
| --- | --- |
| `crossing_metrics.csv` | raw `ΔE_ij`, `\|NAC_ij\|` and dominant character per frame — re-threshold freely |
| `crossing_events.csv` | flagged events with every raw metric and the classification |
| `crossing_summary.md` | the counts, the thresholds, the sensitivity sweep, and the vocabulary |
| `crossing_overlay.png` | gap, coupling and character on one frame axis, swaps marked |
| `character_heatmap.png` | band against frame, one panel per fragment |

Everything is synchronized on the **resolved MD frame**, never on a SHPROP row
index. Histories with different `NAMDTINI` visit different frames at the same
row, and a cyclic mapping wraps them; a row-number correlation would be wrong
in a way that still produces plausible numbers.

## Per history, along its own trajectory

```bash
namd-analysis character-crossings \
  --projection-character results/character/projection_character.csv \
  --eigtxt EIGTXT --natxt NATXT --dt-fs 1 \
  --shprop 'run/SHPROP.*' \
  --state-map state_map.json \
  --projection-manifest projection_manifest.json \
  --frame-mode dish-cyclic \
  --late-window 0.1:10 \
  --out results/crossings
```

Given `--shprop`, each history is classified **on its own trajectory**, and the
fragment population attached to an event is that history's own

```
P_g^(r)(t) = Σ_i P_i^(r)(t) w_ig[f_r(t)]
```

— the same projection-weighted diagonal contraction `character-populations`
performs, but kept per history, because an event has to be classified on the
history that produced it. `SHPROP.master` is not a valid input here either.

There is no ensemble "population at frame *f*" to classify against. Under a
cyclic mapping one history revisits a frame many times, at a different
population each time, and two histories occupy different frames at the same
row. What *is* well defined is one history's own trajectory: at step *t* it
occupies frame `f_r(t)` with population `P^(r)(t)`. Consecutive entries are
therefore consecutive **time steps of that history**, and the frames they
occupy are whatever the resolved mapping says — adjacent, wrapped, or repeated.
A step that stays on one electronic frame is skipped: the character cannot have
changed, whatever the population did.

**Classification comes first, aggregation second.** Two histories can exchange
character in opposite directions at the same frame, and their mean shows
nothing — a rise and a matching fall average to a flat population and to no
transfer at all. `per_history_events.csv` and the `per_history` block of
`report.json` keep the per-history counts beside the totals, so cancellation is
visible rather than silent.

| field | meaning |
| --- | --- |
| `distinct_namdtini` | the starts actually found, so a shared mapping is obvious if it happens |
| `first_frame` / `last_frame` | where each history opened and closed — under a wrap, `last < first` |
| `by_classification` | that history's own counts, before any sum |
| `totals_by_classification` | the sum of those counts, and nothing else |

## Early against late

`--early-window` defaults to `0:0.1` ns (0–100 ps); **`--late-window` is
required and never defaulted**, for the reason given in
[early_late_regimes.md](early_late_regimes.md) — no manuscript fit window is
encoded anywhere in this repository. Without one, `early_vs_late_events` is
`null` rather than a comparison against an invented window.

The comparison splits the *same* per-history classifications by the window each
event fell in. The windows are of different length, so **a larger count in one
is not by itself a higher rate**: divide by the window duration before
comparing, and remember these are flagged metric crossings rather than measured
transitions.
