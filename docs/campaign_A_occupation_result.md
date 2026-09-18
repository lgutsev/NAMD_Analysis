# Campaign A: what the occupation decomposition actually says

Real data. One history, **`SHPROP.25`** (`NAMDTINI = 25`), 10,000,000 steps over
a recycled 1999-frame nuclear trajectory (**5003 passes**), contracted against
the PROCAR character of all six basis states.

> **n = 1.** The campaign's five histories start at 37/171/425/848/1625. This one
> starts at 25 and begins with its entire population on band 981. Everything
> below describes *this* history. It is not an ensemble result and must not be
> quoted as one.

---

## 1. Directly observed

**The decomposition closes to machine precision.**

```
max | ΔP_g − (occupation_redistribution + character_evolution) |  =  3.33e-16
```

**The projection-weighted diagonal subsystem population** over 10 ns:

| fragment | initial | final | net | min | max |
| --- | --- | --- | --- | --- | --- |
| perovskite | 0.0000 | 0.0660 | +0.0660 | 0.0000 | 1.0000 |
| **BCF** | 0.0000 | **0.7800** | **+0.7800** | 0.0000 | 0.9911 |
| **PCBM** | 1.0000 | **0.1540** | **−0.8460** | 0.0000 | 1.0000 |

The history starts with all population on band 981, which at frame 25 is
PCBM-character, and ends with 78% on band 977, the BCF state. **The net motion
is PCBM → BCF**, i.e. population accumulating on BCF — not leaving it.

---

## 2. Inferred from the decomposition

### Over the whole run

| fragment | net | occupation_redistribution | character_evolution | occupation share |
| --- | --- | --- | --- | --- |
| perovskite | +0.0660 | −0.6988 | +0.7648 | 0.477 |
| **BCF** | **+0.7800** | **+0.6183** | +0.1617 | **0.793** |
| **PCBM** | **−0.8460** | **+0.0805** | **−0.9265** | **0.095** |

Two results, and they point opposite ways:

- **BCF's gain is genuinely occupational.** 0.618 of the +0.780 is population
  moving onto the BCF-localized state at fixed character.
- **PCBM's loss is *not* occupation loss.** Occupation moves slightly **toward**
  PCBM (+0.0805). The −0.846 is character evolution: the states carrying the
  population became less PCBM-like. A fixed-band reading would call this
  "PCBM population collapses"; most of it is relabelling.

### At the 977/978 crossing, per pass

Summing the episode across all 5003 passes is meaningless — it adds 5003
re-encounters of one nuclear geometry and produces numbers larger than a
population can be. **Per pass** (12 steps each, 5002 complete passes):

| fragment | net / pass | occupation | character | occ. share | passes gaining | losing |
| --- | --- | --- | --- | --- | --- | --- |
| perovskite | −0.001565 ± 0.000072 | −0.000001 | −0.001563 | 0.001 | 0 | 5002 |
| **BCF** | **+0.001825 ± 0.004550** | +0.000024 | +0.001801 | **0.013** | 4956 | 46 |
| **PCBM** | **−0.000260 ± 0.004556** | −0.000022 | −0.000238 | 0.086 | 81 | 4921 |

**The crossing does not increase PCBM.** PCBM *decreases* on 4921 of 5002
passes, and the change is **98.7% character evolution** for BCF and 91.4% for
PCBM. Occupation contributes ~1% of the movement at the crossing.

### The crossing is largely undone within the same pass

| fragment | whole pass | episode 1320–1331 | rest of cycle |
| --- | --- | --- | --- |
| perovskite | +0.0000132 | −0.0015643 | +0.0015775 |
| **BCF** | **+0.0001559** | +0.0018247 | −0.0016687 |
| PCBM | −0.0001691 | −0.0002603 | +0.0000912 |

The episode hands BCF **+0.00182** and the rest of the cycle takes back
**−0.00167**: about **91% of the crossing's BCF gain is reversed** before the
pass ends. What survives is +0.000156 per pass, and *that* residual is
**79.3% occupational**. Accumulated over 5003 passes it is the +0.78 in §1.

So the mechanism this history supports is: **a small, occupation-driven
residual deposits on BCF on each pass and accumulates**, while the large
character swing at the crossing itself is nearly reversible and moves no charge.

---

## 3. Fixed-band vs dynamic projection

| window | quantity | fixed | dynamic | difference |
| --- | --- | --- | --- | --- |
| late 0.1–10 ns | PCBM net | −0.0090 | −0.0090 | **0.0000** |
| late 0.1–10 ns | PCBM **range** | 0.1480 | **0.8415** | **5.7×** |
| late 0.1–10 ns | BCF net | −0.0570 | −0.0570 | 0.0000 |
| late 0.1–10 ns | BCF **range** | 0.1760 | **0.8461** | 4.8× |

**The "PCBM population remains approximately constant" statement survives as a
statement about net change** — both readings give −0.0090 over the manuscript
window, and the slopes agree to 2e-5/ns. **It does not survive as a statement
that the population is static**: under dynamic character the PCBM population
sweeps a range 5.7× larger, because the character of the occupied states moves
as the nuclei do.

In the early window the two readings disagree by a full **1.0** at t = 0: the
fixed map calls band 981 "CBM" while the projection finds it PCBM-character at
frame 25. That is the fixed labelling being wrong at that frame, which is what
the comparison exists to detect.

---

## 4. What still cannot be claimed

- **No hop record exists.** `occupation_redistribution` is a redistribution of
  SHPROP populations. It is **not** a hop count, a hopping rate, or evidence
  that hops occurred.
- **No transfer rate** follows from any of this.
- **n = 1.** One history, one starting state. The per-pass spread (±0.00455 on a
  mean of +0.00182) is larger than the mean; only the *sign* is consistent
  (4956/5002). Five histories will not simply reproduce it.
- **The 5003 passes are not independent samples.** They are re-encounters of one
  recycled nuclear trajectory. No standard error over passes is quoted, and none
  should be.
- **Not a diabatization.** The coherence term is absent from SHPROP and PROCAR
  alike; the quantity is the projection-weighted **diagonal** subsystem
  population throughout.
- **The zero-field NAMD does not simulate the device field.**

---

## The conclusion this history licenses

> In `SHPROP.25`, the crossing region produces a large but mostly reversible
> fragment-character excursion and no persistent BCF→PCBM transfer; the
> long-time net change instead favours BCF accumulation. **Whether this
> behaviour is representative of Campaign A must be determined from the full
> ensemble of 300 SHPROP histories across three runs.**

That is the whole of it. This document is **validation of the machinery and one
realization**, not a mechanistic result.

### Retracted

An earlier version of this section said "the 977/978 crossing is **not** the
BCF→PCBM transfer event." **That is retracted.** It generalizes from n = 1 to
Campaign A, which one history cannot support. The crossing was not the transfer
event *in this history*; whether it is in others is unmeasured.

No statement of the form "BCF is predominantly a reservoir", "PCBM transfer
occurs in a minority/majority of histories", or "the crossing manifold is
largely reversible" may be made until all 300 histories have been analysed. See
[ensemble_hierarchy.md](ensemble_hierarchy.md) for the level at which R3.2 is
answerable.

### On the two decomposition terms

The occupation/character percentages quoted above are an **unbiased bookkeeping
convention**, not an observable decomposition of the Hamiltonian dynamics. The
symmetric midpoint split is exact and endpoint-unbiased, but it is one of
infinitely many exact splits, and its relative percentages are **not physical
branching fractions**. They say how the *chosen* bookkeeping apportions a
change. Do not report them as a mechanism ratio.
