# BCF-mediated transfer: what this analysis can and cannot establish

The mechanism under revision is

```
perovskite  ->  BCF-associated intermediate  ->  PCBM
```

with BCF a **long-lived metastable intermediate, not a terminal trap**. The
long residence time gives the carrier repeated opportunities to encounter
BCF/PCBM mixed states and move population into PCBM. Under device operating
conditions the built-in or applied field can then favour further separation and
extraction of population already in the PCBM manifold.

**The zero-field NAMD does not simulate that field.** The extraction step is an
operating-device implication, not a simulated result, and is never written as
one.

## The evidence hierarchy

| tier | evidence | status |
| --- | --- | --- |
| 1 | NAMD population kinetics; long-lived BCF-associated population | **primary** — unchanged, and still the basis of the mechanism |
| 2 | Frame-resolved PROCAR character: BCF/PCBM mixing and exchange, avoided-crossing-like behaviour | **supporting, microscopic** |
| 3 | Per-history occupation-driven BCF loss with concomitant PCBM gain | **only if the data show it** |
| 4 | Field-driven extraction under bias | **not accessible** — device implication |

Tier 1 carries the mechanism. Tiers 2 and 3 are corroboration, never a
replacement, and never an independent transfer-rate measurement.

## Four things that are not the same

A revision is at risk of eliding these, and a referee will not be.

1. **BCF–PCBM electronic mixing** — the instantaneous adiabatic states have
   both fragments in their composition.
2. **Occupation redistribution into PCBM** — population actually moves between
   states, with character held fixed.
3. **Threshold-defined BCF→PCBM transfer events** — (2) counted against an
   arbitrary cutoff.
4. **Device-level extraction under bias** — outside the simulation entirely.

Only 1–3 are accessible here. Only 4 involves the field.

## The decomposition, and why the total cannot be used

The reported quantity is the projection-weighted **diagonal** subsystem
population, per history *r*:

```
P_g^(r)(t) = sum_i P_i^(r)(t) w_ig[f_r(t)]
```

This moves for two independent reasons, and they must be separated before any
direction is assigned:

```
dP_g  =  dP_g^pop                                  +  dP_g^char
      =  sum_i [P_i(t) - P_i(t-1)] w_ig[f(t)]      +  sum_i P_i(t-1) [w_ig[f(t)] - w_ig[f(t-1)]]
```

- `dP_g^pop` — **occupation-driven**: population moving between states at fixed
  character. This is the part that can mean charge transfer.
- `dP_g^char` — **character-driven**: the character moving under fixed
  occupation. A band-index relabelling produces this *mechanically*, with no
  charge going anywhere.

The split is exact and is computed during the streaming pass. **A transfer
direction may be assigned only from `dP^pop`.** Using the total would report
transfer at essentially every avoided crossing, because the crossing itself
moves the weights.

> A character swap is not a surface hop, and a surface hop is not charge
> transfer. Staying on one adiabatic state through an avoided crossing
> **changes** the fragment identity — no hop, but the charge moved. Hopping
> between two at the crossing can **preserve** it — a hop, but no transfer.

## Operational threshold counting

Surface-hopping studies count transfer events by thresholding donor and
acceptor populations — donor above 0.9 falling below 0.1, or the acceptor
equivalent. Toldo *et al.* set out both that transition-count approach and the
population-decay alternative, and are explicit that **the threshold is
arbitrary**:

> J. M. Toldo, M. T. do Casal, E. Ventura, S. A. do Monte and M. Barbatti,
> *Phys. Chem. Chem. Phys.* **25**, 8293–8316 (2023).

They also note that **population decay is preferable when the charge is
delocalized**, because a delocalized carrier may never place 0.9 of its
population on one fragment and so never registers a transition at all.

`namd_analysis.transfer` implements this with three disciplines:

- **The cutoff is swept, never single.** `(0.9, 0.1)`, `(0.8, 0.2)`, `(0.7, 0.3)`
  by default. A count that changes across the sweep is a count the threshold
  chose, and `threshold_dependent` says so.
- **The raw continuous change is primary.** Net donor loss, net acceptor gain,
  and the concomitant overlap of the two are reported whether or not any
  threshold is ever crossed. The counts are a *secondary diagnostic*.
- **Only occupation is consumed.** Every entry point takes the occupation-driven
  trajectory. A relabelling cannot register as transfer, structurally.

The acceptor must confirm: a donor emptying into some third fragment is not
counted as transfer into this one. Re-arming is required, so jitter around the
cutoff cannot count one decay many times.

**Zero events is not evidence of no transfer.** It may mean the carrier is
delocalized, which is exactly the regime where the decay fit is the observable
that still works.

## Allowed claims

- BCF acts as a long-lived metastable intermediate rather than a terminal trap
  — *on tier-1 kinetics*.
- The instantaneous adiabatic states exchange BCF and PCBM character along the
  trajectory, with avoided-crossing-like behaviour.
- Where `dP^pop` shows BCF loss with concomitant PCBM gain, that is BCF→PCBM
  population transfer consistent with charge transfer.
- Extended BCF residence provides repeated opportunities for population to
  enter the PCBM manifold.
- Under operating bias, the device field can favour further separation and
  extraction of population already in PCBM — **flagged as an implication**.

## Prohibited claims

- **No formal diabatization.** This is a projection-weighted analysis of
  adiabatic NAMD trajectories. Coherences are absent from the inputs and an
  explicit diabatic Hamiltonian is never constructed. Do not write
  "diabatic states", "diabatized", or "diabatic populations".
- **No character swap called charge transfer.** Not without `dP^pop`.
- **No transfer rate** from event counts. These are flagged events against an
  arbitrary cutoff, not a rate.
- **No field effect claimed as simulated.** The NAMD is zero-field.
- **No threshold count reported alone**, and none reported without its sweep.
- **No claim that a fixed band label tracks a fragment.** Campaign A shows 154
  dominance swaps in 1999 frames; the labels move.

## The Campaign A caveat that governs the wording

From the real run (job `1030646`, 1999 frames, six bands):

| band | perovskite | BCF | PCBM | median capture |
| --- | --- | --- | --- | --- |
| 976 | 0.9945 | 0.0000 | 0.0055 | 0.554 |
| 977 | 0.0006 | 0.9968 | 0.0026 | 0.527 |
| 978 | 0.0000 | 0.0028 | 0.9972 | 0.507 |
| 979 | 0.0000 | 0.0001 | 0.9999 | 0.506 |
| 980 | 0.0010 | 0.0002 | 0.9988 | 0.512 |
| **981** | 0.7820 | **0.1022** | **0.1159** | **0.445** |

Frame/band samples with BCF **and** PCBM both above 0.10 of captured weight:
**19 of 11994 (0.16%)** — and **17 of those 19 are band 981**, the band whose
projection is least determined (1633 of its 1999 frames below the capture
threshold; see issue #4).

So **"persistent BCF/PCBM mixing" is not supported by Campaign A.** *Static*
mixed character — both fragments simultaneously on one state, in the mean — is
rare and sits on the least reliable band.

**The character exchange is a different quantity, and it is clean.** Running
`character-crossings` on the real `EIGTXT`/`NATXT` resolves the BCF–PCBM
avoided crossing completely, on bands **977 and 978**, whose capture (0.50–0.53)
is at the campaign median and therefore *not* subject to the band-981 caveat:

| frame | 977: perov / BCF / PCBM | 978: perov / BCF / PCBM | gap (eV) | \|NAC\| (eV) |
| --- | --- | --- | --- | --- |
| 1321 | 0.000 / **1.000** / 0.000 | 0.000 / 0.006 / **0.994** | 0.04710 | 0.1476 |
| 1322 | 0.000 / 0.977 / 0.023 | 0.000 / 0.031 / 0.969 | 0.02847 | **0.6000** |
| **1323** | 0.000 / 0.091 / **0.909** | 0.000 / **0.905** / 0.095 | 0.01752 | 0.2963 |
| 1325 | 0.000 / 0.016 / 0.984 | 0.000 / 0.994 / 0.006 | 0.03869 | 0.0546 |
| 1327 | 0.000 / 0.091 / 0.909 | 0.000 / 0.917 / 0.083 | 0.01957 | **0.6000** |
| **1328** | 0.000 / **0.773** / 0.227 | 0.000 / 0.240 / **0.760** | **0.01360** | **0.6000** |
| 1330 | 0.000 / **1.000** / 0.000 | 0.000 / 0.006 / **0.994** | 0.06268 | 0.0436 |

The two states exchange character at frame 1323, remain exchanged for four
frames, and exchange back at 1328 where the gap reaches its minimum of
**13.6 meV**. Character is pure (>0.99) on either side. This is a single,
well-resolved passage in and out of an avoided crossing — exactly the two-state
model in [adiabatic_vs_diabatic.md](adiabatic_vs_diabatic.md), found in real
data.

`DEPHTIME` independently supports a strongly interacting pair: BCF↔PCBM (977–978)
dephases in **5.95 fs**, against 86–151 fs for the PCBM–PCBM pairs.

### The frame axes were verified, not assumed

`EIGTXT`/`NATXT` were written months before the PROCAR character run. They are
the same campaign, and the shared frame axis is checkable rather than taken on
trust: at an avoided crossing, mixing is maximal where the gap is minimal, so
character (from PROCARs) and gap (from `EIGTXT`) must co-vary *only* at the
correct offset.

Correlating band 977's BCF/PCBM mixing against `−log(gap)` over all 1999 frames
and scanning the offset gives a maximum at **exactly offset 0** (r = +0.265,
n = 1999; next best −1 at +0.237). Locally, maximum mixing and minimum gap both
fall on frame **1328**. Only the 977–978 pair can run this test — 978/979/980
are all PCBM-dominated, and 976–977 and 980–981 have no mixing variance — but
that is the pair the mechanism rests on.

A naive version of this check, using the gap at all 154 dominance-swap frames,
is **confounded** and should not be used: most swaps sit on band 981, whose only
neighbour gap (980–981) has median 0.62 eV, which inflates every offset alike
and produces a flat, uninformative scan.

Campaign-wide, **28 of the 154 dominance swaps connect BCF and PCBM directly**
(16 BCF→PCBM, 12 PCBM→BCF). The 16:12 asymmetry is far too small to carry a
directional claim.

### Two caveats that must travel with this

- **The coupling at the crossing is clamped.** `|NAC|` is exactly **0.6000 eV**
  at frames 1322, 1327 and 1328 — including the gap minimum. Campaign-wide,
  **328 samples share that magnitude exactly**, at 91.2% of ħ/dt (0.658 eV at
  1 fs). A value repeated exactly did not come out of the dynamics: it is an
  imposed safety ceiling. **The coupling strength at the crossing cannot be
  quoted as a measured value**, and no claim may rest on its magnitude. The gap
  and the character are unaffected.
- **The crossing is re-encountered, not repeated.** Each history is 10,000,000
  steps over a 1999-frame cycle — about **5000 passes** of this same crossing.
  That is the recycled MD trajectory, not 5000 independent crossing events, and
  must not be described as though it were.

The defensible statement is therefore **a well-resolved BCF/PCBM avoided
crossing that the trajectory revisits**, not persistent mixing. Wording is in
[manuscript_wording.md](manuscript_wording.md).

## Running it

Occupation-driven populations come from `character-crossings --shprop`, which
needs the per-band SHPROP populations. The character outputs alone are not
enough: they give tier 2, never tier 3.
