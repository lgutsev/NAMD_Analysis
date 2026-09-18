# The levels, and what may be said at each

```
passes  ⊂  SHPROP histories  ⊂  campaign
```

**Campaigns A, B and C are distinct interface configurations, not
replicates.** They are different physical systems, each with its own band
map, atom partition, cycle length and crossing manifold. Nothing averages
or pools them, and their agreement is *not* a reproducibility check: a
difference between campaigns is a result about the interfaces.

The nesting is what keeps the statistics honest. Collapsing it is the easiest
way to manufacture significance that is not there.

| level | unit | what it varies | independent? |
| --- | --- | --- | --- |
| pass | one traversal of the recycled 1999-frame trajectory | nothing — the same nuclei, again | **no** |
| history | one SHPROP | electronic initial condition (`NAMDTINI`) and the hopping realization | of each other, **at fixed nuclei** |
| campaign | 100 histories on one interface configuration | the interface itself | **not a statistical level** — a separate physical case |

Three consequences, enforced in `namd_analysis.ensemble`:

1. **No standard error over passes.** A history of 10,000,000 steps on a
   1999-frame cycle contains ~5003 re-encounters of one geometry. Summing a
   per-step quantity over them produces numbers larger than a population can
   be — which is how the error announces itself if you try. Everything
   episode-level is reported **per pass**.
2. **100 histories are not 100 nuclear configurations.** A spread across the
   histories of one configuration is a spread over *electronic* initial
   conditions with the nuclei held fixed. `summarize_run` labels it
   `std_between_histories` and states that it is not an uncertainty on the
   system.
3. **Campaigns are never pooled.** `compare_runs` reports each campaign's own
   median, the spread between campaigns, and whether they agree in sign. It
   computes no mean across them, because a mean over three different physical
   systems is not a measurement of anything. **No grand 300-history average is
   reported as a result.**

## What each level answers

**Per history** — `per_history.csv`, one row per SHPROP:
`P_g(t)` for perovskite/BCF/PCBM, the symmetric decomposition summed over the
trajectory, the crossing-episode response per pass, the late-window net and
range, and the decomposition residual (which must stay at machine precision).

**Per configuration** — `run_summary.json`: mean, median, quantiles (5/25/50/75/95),
sign fractions with the tolerance that defined them, between-history spread,
and a tally of episode verdicts
(`persistent_acceptor_gain`, `persistent_donor_gain`,
`essentially_reversible`, `no_clear_direction`).

**Across configurations** — `across_configurations.json` plus the comparison
table. This is a **comparison, not a statistical level**: each configuration's
own summary already licenses statements about it, and this table only sets
them side by side. One column per configuration, nothing pooled:

| Quantity | Campaign A | Campaign B | Campaign C |
| --- | --- | --- | --- |
| histories with net PCBM gain | x/100 | x/100 | x/100 |
| histories with net BCF gain | … | … | … |
| histories with net PCBM loss | … | … | … |
| histories with net BCF loss | … | … | … |
| median ΔP_PCBM | … | … | … |
| median ΔP_BCF | … | … | … |
| late-window net ΔP_PCBM | … | … | … |
| late-window net ΔP_BCF | … | … | … |
| fixed vs dynamic discrepancy | net …, range ×… | … | … |
| dominant decomposition term | … | … | … |

produced for both the full trajectory and the `t ≥ 100 ps` window, with
`ensemble_curves.json` carrying the median and quantile band per
configuration.

Counts are written `x/N`, not as percentages, so the denominator travels with
them — and the note attached says the fraction is of *electronic initial
conditions*, not of nuclear configurations.

## Statements that require a full configuration ensemble

None of the following may be made from a handful of histories:

- "BCF is predominantly a reservoir."
- "PCBM transfer occurs in a minority / majority of histories."
- "The crossing manifold is largely reversible."

Each is a claim about a *distribution*, and each is made **per configuration**.
**Campaign A's 100 histories license Campaign A statements on their own** — B
and C are not a prerequisite. They are separate physical systems that get
their own statements, and a difference between configurations is itself a
finding. One history can only ever say what that history did. `docs/campaign_A_occupation_result.md` reports `SHPROP.25` and is
explicitly labelled as machinery validation plus one realization — and carries
a retraction of an earlier sentence that generalized from it.

## On the two decomposition terms

`occupation_redistribution` and `character_evolution` come from an exact,
endpoint-unbiased split:

```
Δ(P_i w_i) = ΔP_i (w_i^{t+1} + w_i^t)/2  +  (P_i^{t+1} + P_i^t)/2 Δw_i
```

summed over the basis. It closes to machine precision, and the symmetric form
privileges neither endpoint — an asymmetric form is equally exact in the sum
but shifts up to ~0.5 of a population between the two terms.

**Their relative sizes are a bookkeeping convention, not physical branching
fractions.** The split is one of infinitely many exact splits; it is not a
decomposition of the Hamiltonian dynamics into two mechanisms. Report both
terms, report ensemble statistics for both, and do not read the percentages as
a mechanism ratio. `BOOKKEEPING_NOTE` ships with every configuration summary
saying so.

## Running it

```bash
# One array task per configuration; histories stream inside a task. Each task
# passes --fixed-state-map, so the output carries the fixed-vs-dynamic
# statistics and figures.
sbatch examples/bcf_pcbm/run_ensemble_production.sbatch

# Optional: set the configurations side by side. This is a comparison, not a
# prerequisite — each configuration's own output is already interpretable.
namd-analysis character-ensemble --combine \
    ens_A_*/run_summary.json \
    ens_B_*/run_summary.json \
    ens_C_*/run_summary.json \
    --combine-curves A=ens_A_JOBID B=ens_B_JOBID C=ens_C_JOBID \
    --out ens_across_configurations
```

Measured cost: **73.6 s per 10M-row history**, so ~2.0 h for 100 serial, ~6.1 h
for 300; ~1.5 GiB peak, independent of history count, because they stream one
at a time. Per-history rows are flushed as produced, so a crash late in a run
keeps everything before it.

**B and C need their own provenance.** Required: band mapping, SHPROP state
order, atom partition, projection character, the nominal fixed state map, and
the cycle length. The production script refuses to start when any of those is
missing, rather than silently inheriting Campaign A's.

**The crossing window is optional.** If a configuration's manifold has not been
located yet, the populations, the symmetric decomposition and the
fixed-vs-dynamic comparison all still run; every history simply reports
`episode_verdict = not_classified`. Locate the manifold separately and rerun
with `--episode-window` to add episode classification afterwards.
