"""The upstream NAC handling policy, declared rather than inferred.

A NATXT file records numbers and nothing else.  Whether an upstream step
warned, rejected or zeroed couplings above some magnitude is a property of the
workflow that produced the file, and this package has no way to recover it
from the values.  It therefore has to be *declared*, in a small JSON document
supplied alongside the run:

.. code-block:: json

    {
      "nac_policy": {
        "warning_threshold_eV": 0.6,
        "numerical_limit": "hbar_over_dt",
        "reject_above_eV": 0.66,
        "action_above_limit": "zero"
      }
    }

Nothing here is assumed for a dataset that does not declare it.  An undeclared
run is audited against ``hbar/dt`` alone, which is arithmetic from the
timestep and needs no provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .units import hbar_over_dt_ev

#: What the upstream workflow did with a coupling above its numerical limit.
#: Only ``clip`` replaces an otherwise valid value with the limit; the others
#: discard the sample or leave it alone, which has a different consequence for
#: every statistic computed downstream.
POLICY_ACTIONS = ("zero", "reject", "clip", "keep")

#: The symbolic numerical limit: the energy scale set by the timestep.
HBAR_OVER_DT = "hbar_over_dt"


class NacPolicyError(ValueError):
    """Raised when a declared NAC policy cannot be read as written."""


def _positive(payload: Dict[str, Any], key: str) -> Optional[float]:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NacPolicyError(f"nac_policy.{key} must be a number, got {value!r}")
    if not value > 0:
        raise NacPolicyError(f"nac_policy.{key} must be positive, got {value!r}")
    return float(value)


@dataclass
class NacPolicy:
    """A declared upstream coupling-handling policy."""

    warning_threshold_ev: Optional[float] = None
    numerical_limit: Optional[str] = None
    reject_above_ev: Optional[float] = None
    action_above_limit: Optional[str] = None
    notes: str = ""
    source: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], source: Optional[str] = None) -> "NacPolicy":
        if not isinstance(payload, dict):
            raise NacPolicyError("nac_policy must be a JSON object")
        if "nac_policy" in payload:
            inner = payload["nac_policy"]
            if not isinstance(inner, dict):
                raise NacPolicyError("nac_policy must be a JSON object")
            payload = inner

        limit = payload.get("numerical_limit")
        if limit is not None and limit != HBAR_OVER_DT:
            raise NacPolicyError(
                f"nac_policy.numerical_limit must be {HBAR_OVER_DT!r} or absent; "
                f"got {limit!r}. A numeric limit belongs in reject_above_eV, so "
                "that it is not confused with the timestep scale."
            )
        action = payload.get("action_above_limit")
        if action is not None and action not in POLICY_ACTIONS:
            raise NacPolicyError(
                f"nac_policy.action_above_limit must be one of {POLICY_ACTIONS}, "
                f"got {action!r}"
            )

        policy = cls(
            warning_threshold_ev=_positive(payload, "warning_threshold_eV"),
            numerical_limit=limit,
            reject_above_ev=_positive(payload, "reject_above_eV"),
            action_above_limit=action,
            notes=str(payload.get("notes", "")),
            source=str(source) if source is not None else None,
        )
        if (
            policy.warning_threshold_ev is not None
            and policy.reject_above_ev is not None
            and policy.reject_above_ev < policy.warning_threshold_ev
        ):
            raise NacPolicyError(
                "nac_policy.reject_above_eV is below warning_threshold_eV; the "
                "warning region has to sit below the rejection limit"
            )
        return policy

    @classmethod
    def from_json(cls, path) -> "NacPolicy":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise NacPolicyError(f"{path}: {exc}") from exc
        return cls.from_dict(payload, source=str(path))

    @property
    def truncates_valid_values(self) -> bool:
        """True only when the policy replaces a valid coupling with the limit.

        This is the one case in which a repeated ceiling makes every mean, RMS
        and integral built on the affected samples a lower bound.  Zeroing or
        rejecting a pathological sample does not, and must not be described as
        if it did.
        """
        return self.action_above_limit == "clip"

    def limit_ev(self, dt_fs: float) -> Optional[float]:
        """The declared numerical limit in eV, resolved against the timestep."""
        if self.numerical_limit == HBAR_OVER_DT:
            return hbar_over_dt_ev(dt_fs)
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "declared": True,
            "source": self.source,
            "warning_threshold_eV": self.warning_threshold_ev,
            "numerical_limit": self.numerical_limit,
            "reject_above_eV": self.reject_above_ev,
            "action_above_limit": self.action_above_limit,
            "notes": self.notes,
            "provenance": (
                "declared by the user for this dataset; it describes what the "
                "upstream workflow did and is not verified against the file"
            ),
        }


#: Reported when no policy was declared.  The audit still reports hbar/dt,
#: which follows from the timestep alone.
UNDECLARED = {
    "declared": False,
    "note": (
        "no upstream NAC handling policy was declared for this dataset, so none "
        "is assumed. Repeated extreme values are reported as observed, and are "
        "not attributed to any particular upstream rule."
    ),
}
