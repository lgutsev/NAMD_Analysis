"""Classification of every number this package reports.

Three kinds of quantity appear in these reports and they must never be read
as the same thing:

``observed_from_SHPROP``
    Read directly from reconstructed population histories, optionally combined
    with explicitly supplied frame-dependent physical-state projections.  It
    depends on the trajectories, the declared state/projection mapping and no
    fitted kinetic model.  A peak population, a time-integrated population, or
    a PROCAR-weighted physical population is of this kind.

``model_inferred``
    Produced by fitting a kinetic model to those populations.  It is
    conditional on the declared groups, the declared kinetic graph, the
    Markovian constant-rate assumption and the fit window.  A transfer rate
    or a first-passage probability is of this kind.  It is not an event count:
    averaged populations do not record individual hops.

``counterfactual``
    Produced by propagating the fitted model under a condition that was never
    simulated, such as an imposed onward escape rate.  It answers "what would
    have to be true", not "what was measured".  An extraction yield at an
    assumed escape rate is of this kind.

Nothing in this module computes anything.  It exists so that a report can
carry the distinction explicitly rather than leaving a reader to guess, and
so that a mislabelled field fails a test instead of reaching a manuscript.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

#: Read from the saved population history without a fitted kinetic model.
OBSERVED = "observed_from_SHPROP"

#: Obtained by fitting a kinetic model to observed populations.
MODEL_INFERRED = "model_inferred"

#: Obtained by propagating a model under an assumed, unsimulated condition.
COUNTERFACTUAL = "counterfactual"

CLASSES = (OBSERVED, MODEL_INFERRED, COUNTERFACTUAL)

#: Phrases that must never be attached to a quantity of the given class.
#: The finite NAMD model contains no electrode and no long-range transport,
#: so no quantity here is a device efficiency or a measured collection.
FORBIDDEN_DESCRIPTIONS = {
    MODEL_INFERRED: (
        "device extraction efficiency",
        "measured extraction",
        "observed hopping rate",
        "hop count",
    ),
    COUNTERFACTUAL: (
        "measured",
        "device extraction efficiency",
        "observed",
    ),
}


class ObservableClassError(ValueError):
    """Raised when a reported quantity carries no valid classification."""


def validate(classes: Mapping[str, str]) -> Dict[str, str]:
    """Check that every field name maps to one of the three classes."""
    bad = {
        name: value for name, value in classes.items() if value not in CLASSES
    }
    if bad:
        raise ObservableClassError(
            f"these fields carry no valid classification {bad}; "
            f"every reported quantity must be one of {CLASSES}"
        )
    return dict(classes)


def uniform(names: Iterable[str], kind: str) -> Dict[str, str]:
    """Classify a group of field names the same way."""
    if kind not in CLASSES:
        raise ObservableClassError(f"{kind!r} is not one of {CLASSES}")
    return {str(name): kind for name in names}


#: The sentence a report carries alongside its classification table.
LEGEND = {
    OBSERVED: (
        "read from reconstructed SHPROP populations, optionally combined with "
        "explicitly supplied frame-dependent physical-state projections; "
        "conditional on the declared mappings and on the trajectories that were "
        "run, but not on a fitted kinetic model"
    ),
    MODEL_INFERRED: (
        "obtained by fitting a kinetic model to those populations; conditional "
        "on the declared groups, the declared kinetic graph, the Markovian "
        "constant-rate assumption and the fit window. Not an event count: "
        "averaged populations do not record individual hops"
    ),
    COUNTERFACTUAL: (
        "obtained by propagating the fitted model under a condition that was "
        "never simulated. It states what would have to be true, not what was "
        "measured. The simulated cell contains no electrode and no long-range "
        "transport, so nothing here is a device collection efficiency"
    ),
}
