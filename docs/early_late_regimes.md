# Early and late regimes

A fit that begins after a fast transient cannot see it. If an acceptor
population rises and relaxes inside the first hundred picoseconds, a
late-window fit reports the relaxed value and says nothing about the transfer
that produced it — not because the transfer did not happen, but because the
window did not look.

```bash
namd-analysis regime-analysis \
  --files 'run/SHPROP.*' \
  --config state_map.json \
  --late-window 0.1:10 \
  --scheme 'BCF->PCBM,PCBM->VBM' \
  --out results/regimes
```

`early-late-analysis` is the same command.

## The windows

| window | default | why |
| --- | --- | --- |
| early | `0:0.1` ns (0–100 ps) | the interval under examination — a **choice**, not something the data selected |
| late | **none** | see below |

**The late window is required and never defaulted.** No manuscript fit window
is encoded anywhere in this repository. The shipped
`examples/bcf_pcbm/comparison_manifest.template.json` says so in as many words:
the windows *"are not universal constants and are not defaulted anywhere in the
code"*. Inventing one here would put a number nobody chose into the science.

A test asserts the repository still encodes none, so that if a real window is
ever committed the test fails and the CLI can start defaulting to it.

Where one regime ends and the next begins is a property of the system. Vary the
boundary and see whether the conclusions move.

## What each regime reports

**Observed** — read from the histories, per group:

- initial and final population, net change
- peak and minimum, and the time of each
- integrated population over the window

**Model-inferred** — conditional on the declared kinetic graph and the
Markovian constant-rate assumption:

- fitted rates with standard errors and timescales
- relaxation timescales from the eigenvalues of `K`
- fit quality: R², per-group RMS residual
- identifiability status, rank deficiency, Jacobian nullity
- bootstrap intervals where more than one history is available

Nothing assumes the early regime is single-exponential, or that it shares the
late regime's rate matrix. **That assumption is tested, not imposed.**

A window that cannot constrain the declared graph reports `not_fitted` with the
reason. It is not widened, and no edge is dropped until a number appears. An
unidentified rate **is not a small rate** — the data do not constrain it.

## One model, or one per regime

The command scores a single global kinetic model against early-plus-late using
the same conditional-Gaussian information criterion on `G−1` orthonormal
Helmert contrasts that `compare-schemes` uses, so the numbers are the same kind
of thing. `--criterion aic|aicc|bic`, default `aicc`.

> **This is model comparison, not proof of a mechanistic transition.**
> A piecewise description scoring better says one constant-rate matrix does not
> describe both windows equally well. It does not say *what* changed, and it
> does not locate a transition: **the boundary was supplied, not fitted.**

The piecewise model has strictly more parameters and fits a partition of the
same data. The criterion penalizes the parameter count but *not* the freedom to
have chosen the boundary.

If a window cannot be fitted, the comparison reports
`piecewise_incomplete` rather than completing it with invented rates.

## The referee's questions

`early_late_summary.md` answers them directly, each labelled observed,
model-inferred or interpretive:

- Does the acceptor population rise within the early window?
- Does the donor depopulate into the acceptor, the ground state, or both?
- Is there a transient maximum that has already relaxed before the late window
  opens?
- Do different groups dominate the two windows?
- Does the late fit miss early transfer because the acceptor has already
  relaxed?

Every answer is read from populations. **A net change is not a flux,
co-movement is not transfer, and none of it is a hop count.**

## Plots

Early and late are drawn on **independent axes**, so a nanosecond scale cannot
compress a hundred picoseconds into the first pixel. `regime_fits.png` overlays
the fitted curves on the observations, per regime.

## Exit behaviour

The command fails rather than guessing: an empty window, a window with fewer
than two samples, a scheme naming groups that do not exist, or a time grid too
irregular for one propagator are all errors with the numbers attached.
