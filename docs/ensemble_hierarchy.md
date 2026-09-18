# The three levels, and what may be said at each

```
passes  ⊂  SHPROP histories  ⊂  runs
```

The nesting is what keeps the statistics honest. Collapsing it is the easiest
way to manufacture significance that is not there.

| level | unit | what it varies | independent? |
| --- | --- | --- | --- |
| pass | one traversal of the recycled 1999-frame trajectory | nothing — the same nuclei, again | **no** |
| history | one SHPROP | electronic initial condition (`NAMDTINI`) and the hopping realization | of each other, **at fixed nuclei** |
| run | 100 histories on one nuclear/electronic run | the nuclear trajectory | **yes** — this is the reproducibility unit |

Three consequences, enforced in `namd_analysis.ensemble`:

1. **No standard error over passes.** A history of 10,000,000 steps on a
   1999-frame cycle contains ~5003 re-encounters of one geometry. Summing a
   per-step quantity over them produces numbers larger than a population can
   be — which is how the error announces itself if you try. Everything
   episode-level is reported **per pass**.
2. **100 histories are not 100 nuclear configurations.** A spread across the
   histories of one run is a spread over *electronic* initial conditions with
   the nuclei held fixed. `summarize_run` labels it
   `std_between_histories` and states that it is not an uncertainty on the
   system.
3. **Three runs means n = 3.** `compare_runs` reports the per-run medians, the
   **range** of those medians, and whether the runs agree in sign. It does not
   compute a standard error on three numbers. A grand mean over all 300 exists
   but is not the headline, because it hides exactly the between-run variation
   that is the actual check.

## What each level answers

**Per history** — `per_history.csv`, one row per SHPROP:
`P_g(t)` for perovskite/BCF/PCBM, the symmetric decomposition summed over the
trajectory, the crossing-episode response per pass, the late-window net and
range, and the decomposition residual (which must stay at machine precision).

**Per run** — `run_summary.json`: mean, median, quantiles (5/25/50/75/95),
sign fractions with the tolerance that defined them, between-history spread,
and a tally of episode verdicts
(`persistent_acceptor_gain`, `persistent_donor_gain`,
`essentially_reversible`, `no_clear_direction`).

**Across runs** — `across_runs.json` plus the reviewer table:

| | Run A | Run B | Run C |
| --- | --- | --- | --- |
| histories with net PCBM gain | x/100 | y/100 | z/100 |
| histories with net BCF gain | … | … | … |
| histories with net PCBM loss | … | … | … |
| histories with net BCF loss | … | … | … |
| median ΔP_PCBM | … | … | … |
| median ΔP_BCF | … | … | … |

produced for both the full trajectory and the `t ≥ 100 ps` window, with
`ensemble_curves.json` carrying the median and quantile band per run.

Counts are written `x/N`, not as percentages, so the denominator travels with
them — and the note attached says the fraction is of *electronic initial
conditions*, not of nuclear configurations.

## Statements that require the full ensemble

None of the following may be made from fewer than all three runs:

- "BCF is predominantly a reservoir."
- "PCBM transfer occurs in a minority / majority of histories."
- "The crossing manifold is largely reversible."

Each is a claim about a *distribution*. One history can only ever say what that
history did. `docs/campaign_A_occupation_result.md` reports `SHPROP.25` and is
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
a mechanism ratio. `BOOKKEEPING_NOTE` ships with every run summary saying so.

## Running it

```bash
# One array task per run; histories stream inside a task.
sbatch examples/bcf_pcbm/run_ensemble_production.sbatch

# Then gather the three runs — the only level that licenses a Campaign claim.
namd-analysis character-ensemble --combine \
    ens_runA_*/run_summary.json \
    ens_runB_*/run_summary.json \
    ens_runC_*/run_summary.json \
    --out ens_across_runs
```

Measured cost: **73.6 s per 10M-row history**, so ~2.0 h for 100 serial, ~6.1 h
for 300; ~1.5 GiB peak, independent of history count, because they stream one
at a time. Per-history rows are flushed as produced, so a crash late in a run
keeps everything before it.

**Runs B and C need their own provenance** — band mapping, SHPROP state order,
atom partition, cycle length, and the location of their own crossing manifold.
The production script refuses to start on a run whose cycle length or episode
window has not been set, rather than silently inheriting Campaign A's.
