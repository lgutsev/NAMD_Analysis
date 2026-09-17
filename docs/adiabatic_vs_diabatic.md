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
actually moved in the corresponding direction. Where the character exchanged
and the population did not follow, the classification is
`character_swap_without_fragment_transfer` and the summary says outright that
these are not charge-transfer events.

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
