# Manuscript wording

Copy-paste text for the revision. The rationale, equations and the
allowed/prohibited claim list are in
[transfer_mechanism.md](transfer_mechanism.md).

**Status of the evidence as written.** Tier 3 — occupation-driven BCF→PCBM
transfer — **has not been measured yet**. It requires
`character-crossings --shprop` against the Campaign A SHPROP histories, which
has not been run. Everything below is therefore written at the level Campaign A
currently supports, with the tier-3 upgrade marked. Do not promote the wording
before the numbers exist.

---

## 1. Main text (one or two sentences)

Keep the main-text revision minimal. The detailed analysis belongs in the SI.

### 1a. Supported by Campaign A now

> Frame-resolved state-character analysis resolves an avoided crossing between
> the BCF- and PCBM-localized states, at which the two states exchange fragment
> character across a minimum adiabatic gap of ~14 meV, indicating that the
> BCF-associated intermediate retains electronic access to the PCBM manifold
> (Supplementary Note S*x*, Fig. S*x*). This extended residence time provides
> repeated opportunities for population to enter the PCBM manifold, after which
> the device electrostatic field can favour further charge separation and
> extraction.

The second sentence is the mechanism statement and rests on the existing
kinetics; the field clause is explicitly an operating-device implication, not a
simulated result.

### 1b. Only if the per-history analysis shows the projected population moving

Replace the first sentence with:

> Frame-resolved state-character analysis resolves exchange of BCF and PCBM
> character along the trajectory and identifies a concomitant change in the
> projection-weighted fragment population consistent with BCF-to-PCBM charge
> transfer (Supplementary Note S*x*, Fig. S*x*).

### 1c. If it does not

> The projection analysis confirms BCF/PCBM character exchange but does not
> uniquely resolve individual BCF→PCBM transfer events; it is therefore used as
> supporting evidence for electronic coupling rather than as an independent
> transfer-rate measurement.

---

## 2. Supporting Information — mechanistic subsection

> **S*x*. Frame-resolved state character and the origin of fragment-population
> change.**
>
> Fixed adiabatic-state labels can obscure the chemical identity of the state
> they name. To test this, the ion-projected character of each state in the
> NAMD basis was evaluated at every molecular-dynamics frame and combined with
> the surface-hopping populations of each individual trajectory, giving the
> projection-weighted **diagonal** fragment population
> P_g^(r)(t) = Σ_i P_i^(r)(t) w_ig[f_r(t)] for history *r*. This quantity is
> diagonal in the adiabatic basis and is not exact fragment charge: the
> surface-hopping output records no electronic coherences and the ion
> projection carries no cross-band terms, so the off-diagonal contributions to
> Tr[ρ P̂_g] are absent from the inputs, and projection weight falling outside
> the declared fragments is unassigned. Each history was
> analysed on its own resolved frame mapping and classified before any ensemble
> average was taken, because histories initiated at different frames occupy
> different geometries at the same trajectory step.
>
> The analysis resolves an avoided crossing between the BCF- and PCBM-localized
> states of the basis. Away from it each state is essentially pure (>0.99 of the
> captured weight on a single fragment); approaching it the adiabatic gap closes
> to a minimum of 13.6 meV, the two states mix, and their fragment character is
> exchanged, before separating again to pure character on the far side. The
> exchange is therefore episodic and well resolved rather than continuous, and
> the anticorrelated variation of the BCF and PCBM contributions across these
> frames is that expected for passage through an avoided crossing. The
> independently computed decoherence times are consistent with a strongly
> interacting pair: the BCF/PCBM pair dephases on a ~6 fs timescale, against
> 86–151 fs for pairs of PCBM-localized states.
>
> Because the nuclear trajectory is cycled during the nonadiabatic dynamics,
> each trajectory samples this crossing repeatedly; this reflects re-encounter
> of the same nuclear configuration rather than a corresponding number of
> independent crossing events.
>
> Because P_g is summed over the full adiabatic basis, it is invariant to any
> re-ordering of state labels, so a change in it is not an artefact of
> labelling. **A change in P_g was accordingly taken as the observable**, and
> BCF-to-PCBM movement of projected population was identified from concomitant
> BCF loss and PCBM gain in P_g, subject to the diagonal approximation noted
> above.
>
> For transparency the change at each step was additionally decomposed exactly,
> using a symmetric midpoint form, into an occupation-driven contribution,
> ΔP_g^pop = Σ_i [P_i(t) − P_i(t−1)] · ½{w_ig[f(t)] + w_ig[f(t−1)]}, and a
> character-driven contribution,
> ΔP_g^char = Σ_i ½{P_i(t) + P_i(t−1)} · {w_ig[f(t)] − w_ig[f(t−1)]}, with
> ΔP_g = ΔP_g^pop + ΔP_g^char. The midpoint form is endpoint-unbiased; an
> endpoint-weighted split is equally exact in the sum but apportions up to half
> a step's change differently between the two terms. **The character-driven
> contribution is not a relabelling**: it corresponds to adiabatic passage, in
> which a state occupied throughout changes its fragment composition, and can
> therefore reflect a spatial redistribution of the occupied density. The
> decomposition is one of infinitely many exact decompositions and is reported
> as a bookkeeping description of the observed change, **not as a branching
> fraction between distinct physical mechanisms**, and neither contribution is
> equated with a quantity of charge transferred. Because no surface-hopping
> trajectory record was used, no statement is made about nonadiabatic hop
> counts or hopping rates.
>
> In addition, transfer events were counted using donor/acceptor population
> thresholds, following the operational definitions used in the surface-hopping
> literature.<sup>[Toldo2023]</sup> Because such a threshold is necessarily
> arbitrary, counts were evaluated over several donor/acceptor cutoffs —
> (0.9, 0.1), (0.8, 0.2) and (0.7, 0.3) — and are reported alongside the raw
> continuous population change rather than in place of it. Threshold counts are
> used here as a secondary diagnostic only. Where the carrier is delocalized
> across fragments, a population-decay description is more appropriate than a
> transition count, and both are reported.<sup>[Toldo2023]</sup>
>
> This analysis is a projection-weighted characterization of adiabatic NAMD
> trajectories. It is **not a formal diabatization**: electronic coherences are
> absent from the surface-hopping populations and the ion-projected weights
> alike, and no explicit diabatic Hamiltonian is constructed. The results are
> used to characterize electronic mixing and the origin of fragment-population
> change, not to extract transfer rates.

### Caveat paragraph — include it

> Nonadiabatic couplings in this dataset are subject to an imposed upper bound:
> a single magnitude (0.600 eV, 91% of ħ/Δt for the 1 fs electronic timestep)
> recurs exactly across 328 samples, including at the frames of closest
> approach. Coupling magnitudes at the crossing are therefore reported as
> bounded rather than measured, and no conclusion here rests on their size; the
> adiabatic gaps and the fragment character are unaffected.

> The ion-projected weights capture approximately half of the PAW-sphere weight
> of each state (campaign median 0.511), the remainder lying outside the
> declared atomic spheres; the reported character is the normalized
> distribution of the captured fraction. This affects the highest state of the
> basis most strongly, and static mixed BCF/PCBM character is concentrated on
> that state. The conclusions drawn here rest instead on the character
> *exchange* between the BCF- and PCBM-localized states, which occurs on states
> whose captured fraction is at the campaign median and is therefore robust to
> the normalization.

**Do not omit this.** It is also what lets the claim stand: the exchange sits
on the well-captured states, so the caveat bounds the weaker observation
without touching the one the mechanism actually uses.

---

## 3. Figure caption

> **Fig. S*x*.** Frame-resolved fragment character of the adiabatic states in
> the NAMD basis. (**a**) Normalized ion-projected weight on the perovskite,
> BCF and PCBM fragments for each state, as a function of molecular-dynamics
> frame. (**b**) Frames at which the dominant fragment character of a state
> changes, with BCF↔PCBM exchanges highlighted; the strong anticorrelation of
> BCF and PCBM weight across these frames is characteristic of passage through
> an avoided crossing. (**c**) Decomposition of the change in
> projection-weighted fragment population into occupation-driven (ΔP^pop) and
> character-driven (ΔP^char) contributions, using an endpoint-unbiased
> symmetric midpoint split; the decomposition is an exact bookkeeping
> description rather than a branching fraction between mechanisms, and
> ΔP^char is not a relabelling. (**d**) Threshold-defined BCF→PCBM
> transfer counts for donor/acceptor cutoffs (0.9, 0.1), (0.8, 0.2) and
> (0.7, 0.3), evaluated on the occupation-driven component and shown with the
> continuous change in projection-weighted fragment population; the cutoff is
> an arbitrary reporting convention, the counts are a secondary diagnostic, and
> a passage carried by ΔP^char is not registered by this counter.

---

## 4. Reviewer response

> We have added a frame-resolved state-character analysis to test whether fixed
> adiabatic-state labels obscure BCF/PCBM mixing in our simulations. Projecting
> each state in the NAMD basis onto the perovskite, BCF and PCBM fragments at
> every molecular-dynamics frame, and combining these weights with the
> surface-hopping populations of each individual trajectory, confirms that the
> adiabatic states exchange BCF and PCBM character along the trajectory, with
> the anticorrelated weight variation expected near an avoided crossing.
>
> We were careful to distinguish character exchange from a nonadiabatic hop,
> and both from the observable we report. Because the projection-weighted
> fragment population is summed over the full adiabatic basis, it is invariant
> to any re-ordering of state labels; a change in it therefore reflects
> movement of the occupied density and not a change of labelling. We used that
> change as the observable, and identified BCF-to-PCBM movement from
> concomitant BCF loss and PCBM gain in it. We additionally decomposed the
> change at each step exactly into occupation-driven and character-driven
> contributions, using an endpoint-unbiased symmetric midpoint split. The
> character-driven contribution is not a relabelling — it corresponds to
> adiabatic passage, where a continuously occupied state changes its fragment
> composition, and can therefore reflect a spatial redistribution of the
> occupied density. We report the decomposition as an exact bookkeeping
> description of the observed change rather than as a branching fraction
> between mechanisms, we do not equate either contribution with a quantity of
> charge transferred, and we make no claim about hop counts, since no
> surface-hopping trajectory record enters the analysis. We
> additionally report threshold-based transfer counts following the operational
> definitions used in the surface-hopping literature (Toldo *et al.*, *Phys.
> Chem. Chem. Phys.* **25**, 8293–8316, 2023), evaluated over several
> donor/acceptor cutoffs because any single threshold is arbitrary, and
> alongside the raw continuous population change.
>
> These results are presented in the Supporting Information and **do not alter
> the main kinetic model**, which remains the primary evidence for the proposed
> mechanism. We emphasize that the analysis is a projection-weighted
> characterization of adiabatic trajectories rather than a formal
> diabatization, and that it is used as supporting evidence for electronic
> coupling and for the identity of the states involved, not as an independent
> measurement of transfer rates.

---

## Citation

```bibtex
@article{Toldo2023,
  author  = {Toldo, Josene M. and do Casal, Mariana T. and Ventura, Elizete and
             do Monte, Silmar A. and Barbatti, Mario},
  title   = {Surface hopping modeling of charge and energy transfer in active
             environments},
  journal = {Phys. Chem. Chem. Phys.},
  volume  = {25},
  pages   = {8293--8316},
  year    = {2023},
  doi     = {10.1039/D3CP00247K}
}
```

Verify the author list and DOI against the publisher record before submission —
it is reproduced here from the citation supplied with the task, not from the
article itself.
