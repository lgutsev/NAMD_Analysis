# Campaign A: the 977/978 BCF–PCBM crossing

Real data, job `1030646`, 1999 frames, six states, run with the campaign's own
`EIGTXT`/`NATXT` and the `projection_character.csv` the character run wrote.

Three sections, deliberately: **what is observed**, **what is inferred from the
decomposition**, and **what cannot be claimed without hop histories**. A
statement in one section must not be quoted as though it came from another.

---

## 1. Directly observed from SHPROP + PROCAR

These are read from the files. No model, no fit, no decomposition.

### The states are essentially pure away from the crossing

Mean normalized character over 1999 frames:

| band | perovskite | BCF | PCBM |
| --- | --- | --- | --- |
| 976 | 0.9945 | 0.0000 | 0.0055 |
| 977 | 0.0006 | **0.9968** | 0.0026 |
| 978 | 0.0000 | 0.0028 | **0.9972** |
| 979 | 0.0000 | 0.0001 | 0.9999 |
| 980 | 0.0010 | 0.0002 | 0.9988 |
| 981 | 0.7820 | 0.1022 | 0.1159 |

977 is the BCF state and 978 the PCBM state. 979 and 980 are also PCBM.

### One BCF/PCBM episode in the whole trajectory

`find_episodes` merges a passage that exchanges character and exchanges back
into **one** episode, because it is one encounter with the crossing:

| episode | frames | min gap | max \|NAC\| | 977 | 978 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1320–1331 | **0.01360 eV** | 0.6000 | BCF → BCF | PCBM → PCBM |

Frame by frame:

| frame | gap (eV) | 977 BCF/PCBM | 978 BCF/PCBM |
| --- | --- | --- | --- |
| 1321 | 0.0471 | **1.000** / 0.000 | 0.006 / **0.994** |
| 1322 | 0.0285 | 0.977 / 0.023 | 0.031 / 0.969 |
| **1323** | 0.0175 | 0.091 / **0.909** | **0.905** / 0.095 |
| 1325 | 0.0387 | 0.016 / 0.984 | 0.994 / 0.006 |
| 1327 | 0.0196 | 0.092 / 0.909 | 0.917 / 0.083 |
| **1328** | **0.0136** | **0.773** / 0.227 | 0.240 / **0.760** |
| 1330 | 0.0627 | **1.000** / 0.000 | 0.006 / **0.994** |

The character is exchanged at 1323, held for four frames, and exchanged back at
1328 where the gap reaches its minimum. **Net character over the episode is
unchanged**: it is a there-and-back passage, not a permanent relabelling.

### The exchange is not an artifact of normalization

`w_ig = W_ig / Σ_g W_ig` could in principle manufacture a swap through its
denominator. Checked against the **raw, unnormalized** PAW-sphere weights over
frames 1315–1335:

| band | raw `W_BCF` | raw `W_PCBM` | captured range |
| --- | --- | --- | --- |
| 977 | 0.524 → **0.008** | 0.000 → **0.498** | 0.503–0.527 (4.6%) |
| 978 | 0.000 → **0.514** | 0.505 → **0.003** | 0.501–0.526 (4.9%) |

Both the normalized and the raw weights swap, and they **agree**. The captured
fraction moves by under 5% across the window, far too little to produce a swap
of this size. **The exchange is in the numerator.**

### The crossing is *not* isolated from its neighbours

| quantity | value |
| --- | --- |
| 977–978 minimum gap | 0.01360 eV |
| closest approach of **976** | 1.10374 eV (81× the pair gap) |
| closest approach of **979** | **0.00401 eV (0.3× the pair gap)** |
| isolation ratio | **0.3** |

**Band 979 comes three times closer to the pair than the pair comes to
itself.** A strict two-state treatment of frames 1315–1335 is therefore *not*
justified, and the SI must not describe it as an isolated two-level system.

This does **not** invalidate the fragment-level reading, and the reason is
specific: 979 is itself PCBM (0.9999). A near-degeneracy between two
PCBM-localized states redistributes population *within* the PCBM manifold and
cannot move it between BCF and PCBM. The **state-level** picture is
three-state; the **fragment-level** picture is unaffected. Say it that way.

### Couplings at the crossing are bounded, not measured

`|NAC| = 0.6000 eV` exactly at frames 1322, 1327 and 1328 — including the gap
minimum. Campaign-wide, 328 samples share that value exactly, at 91.2% of
ħ/Δt (0.658 eV at 1 fs). **This is an imposed ceiling, set deliberately**, not
a measurement. The exchange at frame 1323 carries `|NAC| = 0.2963` eV, which is
below the ceiling and therefore a genuine value.

Of 4245 flagged events campaign-wide, 164 (3.9%) carry a clamped coupling and
only **11 (0.3%)** are flagged by the coupling alone. No conclusion here rests
on a clamped magnitude.

---

## 2. Inferred from the decomposition

Arithmetic on observed quantities — exact, but a *decomposition*, not a
measurement of a mechanism.

Each step's change in the projection-weighted fragment population splits
exactly, using the symmetric midpoint form:

```
Δ(P_i w_i) = ΔP_i · (w_i^{t+1} + w_i^t)/2  +  (P_i^{t+1} + P_i^t)/2 · Δw_i
             └──── occupation_redistribution ────┘  └──── character_evolution ────┘
```

summed over states. Both terms are reported per fragment, per step and summed
over the episode. An endpoint-biased split is equally exact in the sum but
assigns up to ~0.5 of a population differently between the terms, so the
symmetric form is used and neither endpoint is privileged.

An episode is then classified as `character_following_transfer`,
`occupation_redistribution_dominated`, `mixed`, or `no_net_fragment_transfer`,
by which term carries ≥70% of the absolute acceptor movement.

> **Status: not yet computed for Campaign A.** Both terms need the per-band
> SHPROP populations `P_i(t)`. Those are the five 868 MB histories on LONI and
> were not available here. The episode above is therefore `not_classified`, and
> **the question of whether `P_PCBM` genuinely increases across the 977/978
> crossing is open.** It is answered by
> `examples/bcf_pcbm/FAPI_001_A/run_crossings_A.sbatch`.

---

## 3. What cannot be claimed without hop histories

**No hop record is an input to this analysis.** Surface-hopping trajectories
record which adiabatic state each trajectory occupied at each step; that file
is not consumed anywhere here.

Consequently:

- `occupation_redistribution` is a redistribution of SHPROP populations. It
  **must not** be called hopping, a hop count, or a hopping rate, and it is not
  evidence that hops occurred. The code says so in every note it emits.
- No transfer **rate** may be derived from episode counts.
- Threshold-defined event counts are a secondary diagnostic against an
  arbitrary cutoff, swept over (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), and are
  never reported alone.
- The zero-field NAMD does not simulate the device field. Extraction under bias
  is an operating-device implication, not a result.
- This is **not a formal diabatization**: coherences are absent from the inputs
  and no diabatic Hamiltonian is constructed.

### The recycled trajectory is not independent sampling

Each history is 10,000,000 steps over a **1999-frame cycle** — roughly **5000
passes** of this same crossing. Those are re-encounters with one nuclear
configuration, not 5000 independent crossing events. Per-history counts are
reported beside the ensemble totals and are never multiplied up into a rate.

---

## Reproducing

```bash
namd-analysis character-crossings \
  --projection-character character_test_A_1030646/projection_character.csv \
  --eigtxt EIGTXT --natxt NATXT --nac-unit eV --dt-fs 1 \
  --out crossings_A
```

Sections 1 and the isolation/raw-weight checks need only the above. Section 2
additionally needs `--shprop ... --state-map ... --frame-mode dish-cyclic`.
