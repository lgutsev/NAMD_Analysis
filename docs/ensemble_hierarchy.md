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

**Per episode** — `episode_per_history.csv` (long form: one row per history,
episode and fragment) and the named `<episode>__<group>_<quantity>` columns of
`per_history.csv`. A configuration may have several crossing regions;
Configuration B has four. Each keeps its own net, occupation and
character-evolution response per pass, and its own verdict.

**Per configuration** — `run_summary.json`: mean, median, quantiles (5/25/50/75/95),
sign fractions with the tolerance that defined them, between-history spread,
a tally of episode verdicts
(`persistent_acceptor_gain`, `persistent_donor_gain`,
`essentially_reversible`, `no_clear_direction`), and an `episodes` block
carrying each named window separately.

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

**Every run records which implementation produced it.**
`run_summary.json` carries `environment.code`: the package version and, when
the package is imported from a git working tree, the commit SHA, branch and a
`dirty` flag. The A/B/C comparison is only valid if the three configurations
were analysed by the *same* implementation, and "same version" is a weaker
claim than "same commit" — an editable checkout can move between array tasks.
`dirty = true` means uncommitted changes were present, so the commit alone
does not identify what ran; the production script prints the record and warns
before it starts.

**B and C need their own provenance.** Required: band mapping, SHPROP state
order, atom partition, projection character, the nominal fixed state map, and
the cycle length. The production script refuses to start when any of those is
missing, rather than silently inheriting Campaign A's.

**The crossing window is optional.** If a configuration's manifold has not been
located yet, the populations, the symmetric decomposition and the
fixed-vs-dynamic comparison all still run; every history simply reports
`episode_verdict = not_classified`. Locate the manifold separately and rerun
with `--episode` to add episode classification afterwards.

## Several crossing windows in one configuration

A configuration may contain more than one BCF/PCBM mixing region.
Configuration B contains four. They are passed by name and reported by name:

```bash
namd-analysis character-ensemble \
    --episode crossing_B1=1488:1492 \
    --episode crossing_B2=1687:1696 \
    --episode crossing_B3=1801:1811 \
    --episode crossing_B4=1846:1855 \
    ...
```

Two things about this, and they pull in opposite directions from each other.

**All of them are evaluated in one streamed pass.** Each ~900 MB SHPROP is read
once, the symmetric midpoint decomposition is formed once, and each window is a
frame mask over that single result. Four regions cost one read of the archive,
not four, and no window can change another window's numbers.

**None of them is combined with any other.** B1–B4 are four regions of the
*same* recycled nuclear trajectory. They are not four replicates, not four
independent samples, and not four repeat measurements of one quantity. There is
no level at which averaging them is defined, and nothing computes such an
average — not `run_summary.json`, not `per_history.csv`, not the
across-configuration comparison. A pooled "Campaign B episode response" would
describe no region at all.

The **legacy single `--episode-window`** is unchanged and still supported: it
keeps the original unprefixed `episode_*` columns and the run-level
`episode_verdict`. When only named windows are given, that run-level verdict
stays `not_classified` — promoting one of four regions to "the" episode would
be a choice with no basis in the data.

### Control windows

`--control-episode NAME=FIRST:LAST` records a window as an explicit **control**.
A control window is a range of frames chosen for comparison. It is **not** an
avoided crossing, **not** a character-exchange region and **not** a BCF/PCBM
transfer event, and every output records it as `role = control` so it cannot
later be read back as one.

Configuration C has **no** clear BCF/PCBM character-exchange event. That
emptiness is the finding, not a gap waiting to be filled. Its closest-approach
window 1628:1632 is available as `closest_approach_C`, as a control and only as
a control.
