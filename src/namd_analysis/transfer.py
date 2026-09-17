"""Operational donor -> acceptor transfer counting, on occupation only.

Surface-hopping studies commonly count charge-transfer events by thresholding
donor and acceptor populations -- donor above 0.9 falling below 0.1, or the
equivalent acceptor rise.  Toldo, Mukherjee, Mai and Barbatti, *Phys. Chem.
Chem. Phys.* **25**, 8293-8316 (2023) set out both that transition-count
approach and the population-decay alternative, and are explicit that **the
threshold is arbitrary**: it is a reporting convention, not a measurement.
They also note that population decay is the better observable when the charge
is delocalized, because a delocalized carrier may never put 0.9 of its
population on one fragment and so never registers a "transition" at all.

Two disciplines follow from that, and this module enforces both.

**Thresholds are swept, never single.**  One cutoff produces one number with no
way to tell a robust count from an artifact of where the line was drawn.  A
count that collapses across :data:`DEFAULT_THRESHOLD_PAIRS` is a count the
threshold chose.  The raw continuous change is reported alongside and is the
primary quantity; the counts are a secondary diagnostic.

**Only occupation may define a transfer direction.**  The projection-weighted
fragment population ``P_g = sum_i P_i w_ig`` moves when the *weights* move, so
a band-index relabelling shifts it with no charge going anywhere.  Every
function here consumes the **occupation-driven** component

    dP_g^pop(t) = sum_i [P_i(t) - P_i(t-1)] w_ig[f(t)]

accumulated into a trajectory, never the total.  See
:class:`~namd_analysis.crossings.HistoryPopulation` for the split, which is
exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "TOLDO_2023",
    "DEFAULT_THRESHOLD_PAIRS",
    "TransferError",
    "ThresholdCount",
    "ContinuousChange",
    "PopulationDecay",
    "TransferAnalysis",
    "occupation_driven_trajectory",
    "count_threshold_events",
    "continuous_change",
    "population_decay",
    "analyze_transfer",
]

TOLDO_2023 = (
    "J. M. Toldo, M. T. do Casal, E. Ventura, S. A. do Monte and M. Barbatti, "
    "Phys. Chem. Chem. Phys. 25, 8293-8316 (2023)"
)

#: ``(donor_high, acceptor_low)`` pairs swept together.  The first is the
#: convention most often quoted; none of them is privileged.
DEFAULT_THRESHOLD_PAIRS: Tuple[Tuple[float, float], ...] = (
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
)

ARBITRARY_NOTE = (
    "the threshold is arbitrary and is a reporting convention, not a "
    f"measurement ({TOLDO_2023}). Counts are reported across a sweep so that a "
    "count which depends on the cutoff is visible as such, and the continuous "
    "population change is reported alongside as the primary quantity"
)

DELOCALIZED_NOTE = (
    "a delocalized carrier may never place the donor threshold's worth of "
    "population on one fragment, so a transition count can be zero while "
    "occupation is demonstrably moving. Population decay is the better "
    f"observable in that regime ({TOLDO_2023}); zero events is not evidence of "
    "no transfer"
)


class TransferError(ValueError):
    """Raised when the inputs cannot support a transfer statement."""


@dataclass
class ThresholdCount:
    """Events at one ``(donor_high, acceptor_low)`` cutoff."""

    donor_high: float
    acceptor_low: float
    n_events: int
    #: Index of each accepted crossing, on the supplied time axis.
    event_indices: List[int] = field(default_factory=list)
    donor_ever_above: bool = False
    acceptor_ever_above: bool = False

    def as_row(self) -> List[Any]:
        return [
            self.donor_high, self.acceptor_low, self.n_events,
            self.donor_ever_above, self.acceptor_ever_above,
        ]


@dataclass
class ContinuousChange:
    """The raw, threshold-free statement of what occupation did."""

    donor_initial: float
    donor_final: float
    donor_net: float
    acceptor_initial: float
    acceptor_final: float
    acceptor_net: float
    #: Largest single-step occupation-driven move, either fragment.
    max_step: float
    #: Occupation-driven donor loss that coincided with acceptor gain.
    concomitant: float

    @property
    def donor_loss_with_acceptor_gain(self) -> bool:
        return self.donor_net < 0.0 and self.acceptor_net > 0.0


@dataclass
class PopulationDecay:
    """Mono-exponential decay of the donor occupation, or why there is none."""

    fitted: bool
    tau_ns: Optional[float] = None
    amplitude: Optional[float] = None
    offset: Optional[float] = None
    r_squared: Optional[float] = None
    reason: str = ""


@dataclass
class TransferAnalysis:
    donor: str
    acceptor: str
    continuous: ContinuousChange
    thresholds: List[ThresholdCount]
    decay: PopulationDecay
    note: str = ARBITRARY_NOTE

    @property
    def counts(self) -> List[int]:
        return [t.n_events for t in self.thresholds]

    @property
    def threshold_dependent(self) -> bool:
        """Do the counts disagree across the sweep?"""
        return len(set(self.counts)) > 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "donor": self.donor,
            "acceptor": self.acceptor,
            "continuous": {
                "donor_initial": self.continuous.donor_initial,
                "donor_final": self.continuous.donor_final,
                "donor_net": self.continuous.donor_net,
                "acceptor_initial": self.continuous.acceptor_initial,
                "acceptor_final": self.continuous.acceptor_final,
                "acceptor_net": self.continuous.acceptor_net,
                "max_single_step": self.continuous.max_step,
                "concomitant_donor_loss_acceptor_gain":
                    self.continuous.concomitant,
                "donor_loss_with_acceptor_gain":
                    self.continuous.donor_loss_with_acceptor_gain,
            },
            "threshold_sweep": [
                {
                    "donor_high": t.donor_high,
                    "acceptor_low": t.acceptor_low,
                    "n_events": t.n_events,
                    "donor_ever_reached_high": t.donor_ever_above,
                    "acceptor_ever_reached_high": t.acceptor_ever_above,
                }
                for t in self.thresholds
            ],
            "threshold_dependent": self.threshold_dependent,
            "population_decay": {
                "fitted": self.decay.fitted,
                "tau_ns": self.decay.tau_ns,
                "amplitude": self.decay.amplitude,
                "offset": self.decay.offset,
                "r_squared": self.decay.r_squared,
                "reason": self.decay.reason,
            },
            "citation": TOLDO_2023,
            "note": self.note,
            "delocalization_note": DELOCALIZED_NOTE,
            "quantity": (
                "occupation-driven fragment population only; the "
                "character-driven component is excluded, so a band relabelling "
                "cannot register as transfer"
            ),
        }


def occupation_driven_trajectory(
    initial: Sequence[float], occupation_redistribution: np.ndarray
) -> np.ndarray:
    """Accumulate per-step occupation-driven changes into a trajectory.

    ``occupation_redistribution`` is ``(ntime, ngroup)`` with row 0 zero, as
    :func:`~namd_analysis.crossings.history_fragment_population` returns it.
    The result is the population each fragment *would* have had if its
    character had been frozen at the start: occupation redistribution with the
    weight motion removed.

    It is not the observable fragment population, and is not claimed to be.  It
    is the part of the observable that a transfer direction may be read from.
    """
    driven = np.asarray(occupation_redistribution, dtype=float)
    if driven.ndim != 2:
        raise TransferError("occupation_redistribution must be (ntime, ngroup)")
    start = np.asarray(initial, dtype=float)
    if start.shape != (driven.shape[1],):
        raise TransferError(
            f"initial has {start.shape} values for {driven.shape[1]} groups"
        )
    return start[None, :] + np.cumsum(driven, axis=0)


def count_threshold_events(
    donor: np.ndarray,
    acceptor: np.ndarray,
    pairs: Sequence[Tuple[float, float]] = DEFAULT_THRESHOLD_PAIRS,
) -> List[ThresholdCount]:
    """Count donor-above-high -> donor-below-low crossings, acceptor confirming.

    An event is one downward crossing of the donor through ``acceptor_low``
    that began above ``donor_high``, **and** at which the acceptor has risen
    above ``acceptor_low``.  Requiring the acceptor to confirm is what keeps a
    donor decaying into some third fragment from being counted as transfer to
    this one.

    Re-arming is required: the donor must climb back above ``donor_high``
    before another event can be counted, so one decay cannot be counted many
    times by jitter around the cutoff.
    """
    donor = np.asarray(donor, dtype=float)
    acceptor = np.asarray(acceptor, dtype=float)
    if donor.shape != acceptor.shape:
        raise TransferError(
            f"donor {donor.shape} and acceptor {acceptor.shape} differ"
        )
    if donor.ndim != 1:
        raise TransferError("donor and acceptor must be one-dimensional")

    out: List[ThresholdCount] = []
    for high, low in pairs:
        if not 0.0 <= low < high <= 1.0:
            raise TransferError(
                f"threshold pair ({high}, {low}) must satisfy 0 <= low < high <= 1"
            )
        armed = False
        indices: List[int] = []
        for index in range(donor.size):
            if donor[index] >= high:
                armed = True
            elif armed and donor[index] <= low and acceptor[index] >= low:
                indices.append(int(index))
                armed = False
        out.append(
            ThresholdCount(
                donor_high=float(high),
                acceptor_low=float(low),
                n_events=len(indices),
                event_indices=indices,
                donor_ever_above=bool(np.any(donor >= high)),
                acceptor_ever_above=bool(np.any(acceptor >= high)),
            )
        )
    return out


def continuous_change(
    donor: np.ndarray, acceptor: np.ndarray, per_step: Optional[np.ndarray] = None
) -> ContinuousChange:
    """The threshold-free summary, which is the primary quantity."""
    donor = np.asarray(donor, dtype=float)
    acceptor = np.asarray(acceptor, dtype=float)
    if donor.size == 0:
        raise TransferError("no samples")

    d_step = np.diff(donor)
    a_step = np.diff(acceptor)
    both = np.minimum(np.maximum(-d_step, 0.0), np.maximum(a_step, 0.0))
    max_step = 0.0
    if per_step is not None:
        max_step = float(np.max(np.abs(np.asarray(per_step, dtype=float))))
    elif d_step.size:
        max_step = float(max(np.max(np.abs(d_step)), np.max(np.abs(a_step))))

    return ContinuousChange(
        donor_initial=float(donor[0]),
        donor_final=float(donor[-1]),
        donor_net=float(donor[-1] - donor[0]),
        acceptor_initial=float(acceptor[0]),
        acceptor_final=float(acceptor[-1]),
        acceptor_net=float(acceptor[-1] - acceptor[0]),
        max_step=max_step,
        concomitant=float(both.sum()),
    )


def population_decay(time_ns: np.ndarray, donor: np.ndarray) -> PopulationDecay:
    """Mono-exponential fit of the donor occupation, or a stated refusal.

    Preferred over a transition count when the carrier is delocalized.  A fit
    that does not describe the data is reported as not fitted, with the reason,
    rather than returned with a large residual and no comment.
    """
    from scipy.optimize import least_squares

    time_ns = np.asarray(time_ns, dtype=float)
    donor = np.asarray(donor, dtype=float)
    if time_ns.shape != donor.shape:
        raise TransferError("time and donor differ in shape")
    if donor.size < 4:
        return PopulationDecay(False, reason="fewer than four samples")
    span = float(donor.max() - donor.min())
    if span < 1.0e-6:
        return PopulationDecay(
            False, reason=f"donor occupation is flat (range {span:.2e}); nothing decays"
        )

    t = time_ns - time_ns[0]
    scale = float(t.max()) or 1.0

    def residual(p):
        amplitude, rate, offset = p
        return amplitude * np.exp(-rate * t / scale) + offset - donor

    best = None
    for guess_rate in (0.5, 1.0, 3.0, 10.0):
        try:
            fit = least_squares(
                residual,
                [donor[0] - donor[-1], guess_rate, donor[-1]],
                method="lm",
                max_nfev=5000,
            )
        except Exception:  # noqa: BLE001 - a failed start is not an error
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        return PopulationDecay(False, reason="no optimizer start converged")

    amplitude, rate, offset = best.x
    if rate <= 0.0:
        return PopulationDecay(
            False, reason=f"fitted rate is not positive ({rate:.3e}); this is not a decay"
        )
    residuals = residual(best.x)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((donor - donor.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return PopulationDecay(
        fitted=True,
        tau_ns=float(scale / rate),
        amplitude=float(amplitude),
        offset=float(offset),
        r_squared=float(r2),
    )


def analyze_transfer(
    donor_name: str,
    acceptor_name: str,
    groups: Sequence[str],
    occupation: np.ndarray,
    time_ns: Optional[np.ndarray] = None,
    per_step: Optional[np.ndarray] = None,
    pairs: Sequence[Tuple[float, float]] = DEFAULT_THRESHOLD_PAIRS,
) -> TransferAnalysis:
    """Continuous change, a threshold sweep and a decay fit, in that order.

    ``occupation`` is the ``(ntime, ngroup)`` **occupation-driven** trajectory
    from :func:`occupation_driven_trajectory`, indexed by ``groups``.
    """
    names = list(groups)
    for name in (donor_name, acceptor_name):
        if name not in names:
            raise TransferError(f"group {name!r} is not among {names}")
    if donor_name == acceptor_name:
        raise TransferError("donor and acceptor must differ")

    occupation = np.asarray(occupation, dtype=float)
    donor = occupation[:, names.index(donor_name)]
    acceptor = occupation[:, names.index(acceptor_name)]

    decay = PopulationDecay(False, reason="no time axis supplied")
    if time_ns is not None:
        decay = population_decay(time_ns, donor)

    return TransferAnalysis(
        donor=donor_name,
        acceptor=acceptor_name,
        continuous=continuous_change(donor, acceptor, per_step),
        thresholds=count_threshold_events(donor, acceptor, pairs),
        decay=decay,
    )
