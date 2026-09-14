# Individual hopping histories

The public classic Hefei-NAMD source was inspected at commit
[`b7c8fc5a27e89b85fbcd7ba67c41b38a99085d4e`](https://github.com/QijingZheng/Hefei-NAMD/tree/b7c8fc5a27e89b85fbcd7ba67c41b38a99085d4e).
In [`src/namd/SurfHop.f90`](https://github.com/QijingZheng/Hefei-NAMD/blob/b7c8fc5a27e89b85fbcd7ba67c41b38a99085d4e/src/namd/SurfHop.f90),
`runSH` updates `cstat` inside the `NTRAJ` loop, accumulates state counts in
`sh_pops`, and divides by NTRAJ. The subsequent SHPROP writer outputs those
averaged populations. It does not write individual accepted-hop histories.
This observation applies to that public revision, not every DISH/DEV fork or
the user's uninspected installed binary.

Consequently, directional hopping counts cannot be recovered from these
SHPROP files. A flat acceptor population could hide forward/backward exchange.
The fitted kinetic rates are model-inferred rates, not measured event counts.
No history parser has been fabricated for data the engine does not emit.

For the installed engine, the next step belongs in NAMD_Launcher/engine
integration: identify the exact engine commit and output options. If no event
logging exists, add opt-in logging of trajectory ID, initial MD frame, seed,
initial active state, accepted transition time, source and destination state,
and a termination/censoring record for **every** trajectory, including those
with no hops. Rejected/frustrated attempts must be distinguished from accepted
transitions. Stable physical group assignments or frame-dependent state
character are needed to map state hops to interfacial events. Parallel logging
must preserve unique trajectory IDs and avoid race conditions.

Only after such records are available can actual forward/backward counts and
first-arrival fractions be computed with a correct denominator. First arrival
at PCBM remains distinct from collection at an electrode. Implementing that
instrumentation in this separate analyzer would violate its analysis-only
boundary; no changes to scientific engine code are included here.

## What is deliberately absent

No event-history reconstruction is fabricated from averaged populations, at
any version. Concretely, none of

- perovskite -> BCF
- BCF -> perovskite
- BCF -> PCBM
- PCBM -> BCF

can be counted from `SHPROP.master` or from any average this package produces,
including the canonical one `average-shprop` writes: averaging populations
destroys exactly the per-trajectory record such a count needs. A rate that
`kinetics` fits is a parameter of a declared Markovian model conditional on
the declared groups, the declared graph, constant-rate assumptions and the fit
window. It is not a hop count, and the reports say so in those terms.

The full record a real event analysis requires is listed above: trajectory ID,
seed, starting MD frame, initial active state, transition time, source and
destination state, accepted versus frustrated hop, termination/censoring
state, and the trajectories that made no accepted hop at all. Without every
one of those the denominator is wrong. That belongs in engine/launcher
integration, not here.
