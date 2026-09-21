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

`character-crossings` enforces this. An event is only classified as
`character_swap_with_fragment_population_change` when the fragment population
actually moved in the corresponding direction — see
[below](#did-the-charge-move-the-way-the-character-did) for what that test is
and when it cannot be run. Where the character exchanged and the population did
not follow, the classification is `character_swap_without_fragment_transfer`
and the summary says outright that these are not charge-transfer events. Where
it moved the *other* way, it is
`character_swap_with_unrelated_population_change`.

Where **no fragment population was supplied at all**, the classification is
`character_swap_population_not_evaluated`. That is a fourth, distinct outcome,
and it is not a weaker form of the second: a configuration-level scan reads
`projection_character.csv`, `EIGTXT` and `NATXT`, none of which carries a
SHPROP population, so the transfer question was never asked. Such a run may
report neither transfer nor its absence, and the summary refuses to state one.
`fragment_population_change = null` in `crossing_events.csv` and `report.json`
means **not evaluated**, never zero; the `population_evaluated` column says so
in its own cell.

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

## Did the charge move the way the character did?

A population that moves at the same step as a swap has not thereby moved
*because* of it. **Nothing is called transfer unless the fragment the dominance
moved to gained the occupation the one it left lost.**

### The total cannot answer this

`P_g = Σ_i P_i w_ig` moves when the **weights** move. A band-index swap changes
`w_ig` by construction, so it shifts `P_g` mechanically *at fixed occupation* —
no charge has to go anywhere. Testing the total against the swap direction
would therefore confirm "transfer" at very nearly every swap, which is the
opposite of what this module is for.

So each step's change is split first, exactly:

```
ΔP_g   =   ΔP_g^pop                    +   ΔP_g^char
       =   Σ_i [P_i(t) − P_i(t−1)] w_ig[f(t)]
                                       +   Σ_i P_i(t−1) [w_ig[f(t)] − w_ig[f(t−1)]]
```

`ΔP_g^pop` is occupation moving between states at fixed character — the part
that can mean charge transfer. `ΔP_g^char` is the character moving under fixed
occupation — the part a swap produces on its own. Both are written to
`per_history_events.csv` as `population_driven_change` and
`character_driven_change`, and **the direction test runs on `ΔP_g^pop` alone.**

This needs the per-band populations `P_i(t)`, which only a single SHPROP
history carries. `detect_events`, given a pre-contracted total, therefore never
claims a direction — it reports `character_swap_with_undetermined_direction`
and says why.

| outcome | label |
| --- | --- |
| occupation moved the way the swap did | `character_swap_with_fragment_population_change` |
| it moved, but not that way | `character_swap_with_unrelated_population_change` |
| occupation did not move at all — the whole change was `ΔP^char` | `character_swap_without_fragment_transfer` |
| the direction could not be tested | `character_swap_with_undetermined_direction` |
| **no population was supplied, so nothing was evaluated** | `character_swap_population_not_evaluated` |

The third row is the case the split exists to catch: the *total* fragment
population can move a long way at a swap while `ΔP^pop` is zero, because the
weights moved and the occupation did not. Before the split that looked
identical to transfer.

The **last** row is the one that must never be merged into the third. The
third is a measured absence — a population was read, and it did not move. The
last is no measurement at all. A run that produces only the last row has said
nothing whatever about fragment transfer, and no sentence of the form "no
transfer accompanied these exchanges" may be written from it.

The `character_swap_with_undetermined_direction` row is an honest verdict, not
a fallback. It covers three cases: only a total was available; the character
changed without any band's dominant
fragment moving, so there is no direction; or the swap names **no single
direction** — a *simultaneous* symmetric exchange, one band going BCF→PCBM
while the other goes PCBM→BCF, where occupation moving either way would match
one of the two. That is undecidable, and is reported as undecided rather than
resolved in favour of transfer.

When the two flips land a frame apart, each names a single direction and each
is tested on its own. One steady occupation ramp then agrees with one flip and
contradicts the other — which is the point: reporting both as transfer would
count one exchange twice.

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
