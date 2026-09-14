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
