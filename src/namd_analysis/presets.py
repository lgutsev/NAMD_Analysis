"""Campaign presets: declarations a human made, recorded so they can be reused.

A preset is not inference.  Every value here was established by looking at the
structure that was actually built and at the order the states were written in,
and none of it can be recovered from a SHPROP table or a PROCAR by itself.
Applying a preset to a campaign it was not written for would relabel the
science silently, so a preset names the campaign it describes, states the ion
count it assumes, and is checked against the files before it is used.

What a preset never does is invent a subsystem name or an atom boundary.  If a
campaign is not in this registry, the preparation command asks for the atom
grouping rather than guessing one from element symbols or from contiguity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class PresetError(ValueError):
    """Raised when a preset does not match the campaign it was applied to."""


@dataclass
class CampaignPreset:
    """One campaign's declared state order and physical atom partition."""

    preset: str
    campaign: str
    name: str
    description: str
    #: Zero-based SHPROP columns, in basis order BMIN..BMAX.  The preparation
    #: command still infers the column block from the table and refuses to
    #: proceed if the two disagree.
    population_columns: List[int]
    #: Nominal fixed-column groups, for comparison only.
    groups: Dict[str, List[int]]
    complete_population: bool
    recombined_group: Optional[str]
    time_column: int
    time_unit: str
    #: Physical atom partition, one-based PROCAR ion indices, range strings
    #: allowed.  Coarser than the state map on purpose.
    atom_groups: Dict[str, List[Any]]
    n_ions: int
    complete_atoms: bool = True
    min_projection_weight: float = 0.5
    notes: str = ""
    state_order: List[str] = field(default_factory=list)

    def state_map_payload(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "time_column": self.time_column,
            "time_unit": self.time_unit,
            "population_columns": list(self.population_columns),
            "groups": {name: list(cols) for name, cols in self.groups.items()},
            "complete_population": self.complete_population,
            "recombined_group": self.recombined_group,
            "notes": self.notes,
        }

    def atom_groups_payload(self) -> Dict[str, Any]:
        return {
            "groups": {name: list(spec) for name, spec in self.atom_groups.items()},
            "complete_atoms": self.complete_atoms,
            "min_projection_weight": self.min_projection_weight,
        }


#: Why PCBM1/2/3 collapse, and why the dynamic groups are coarser than the
#: state map.  Repeated into every generated file so the distinction survives
#: being copied out of this repository.
BCF_PCBM_NOTES = (
    "Nominal fixed-column map for FAPI/BCF/PCBM interfaces, for comparison only. "
    "The column order is a property of the run that wrote SHPROP, not something "
    "this package can verify; confirm it against the run's inp/INICON before "
    "trusting any fixed-column number. PCBM1/PCBM2/PCBM3 are separate basis "
    "states but one physical subsystem, so they are grouped. The dynamic PROCAR "
    "groups are coarser still -- perovskite / BCF / PCBM -- because atomic "
    "localization cannot separate VBM from CBM when both are perovskite-localized. "
    "Where the fixed map and the projection disagree is exactly what the "
    "character workflow exists to show."
)

_A = CampaignPreset(
    preset="bcf_pcbm",
    campaign="A",
    name="FAPI_001_BCF_PCBM_A",
    description="FAPI (001) slab with BCF and PCBM, six-state basis",
    population_columns=[2, 3, 4, 5, 6, 7],
    groups={"VBM": [2], "BCF": [3], "PCBM": [4, 5, 6], "CBM": [7]},
    complete_population=True,
    recombined_group="VBM",
    time_column=0,
    time_unit="fs",
    state_order=["VBM", "BCF", "PCBM1", "PCBM2", "PCBM3", "CBM"],
    atom_groups={
        "perovskite": ["2-28", "134-268", "283-363", "364-417", "420-446"],
        "BCF": [1, "29-46", "119-133"],
        "PCBM": ["47-118", "269-282", "418-419"],
    },
    n_ions=446,
    complete_atoms=True,
    min_projection_weight=0.5,
    notes=BCF_PCBM_NOTES,
)

#: Registered campaigns, keyed by ``(preset, campaign)``.  B and C exist as
#: campaign labels elsewhere in this repository but their structures were built
#: differently; adding them here requires the same confirmed atom partition
#: that A has, not a copy of A's.
PRESETS: Dict[str, Dict[str, CampaignPreset]] = {"bcf_pcbm": {"A": _A}}


def preset_names() -> List[str]:
    return sorted(PRESETS)


def campaign_names(preset: str) -> List[str]:
    return sorted(PRESETS.get(preset, {}))


def load_preset(preset: str, campaign: Optional[str]) -> CampaignPreset:
    """Fetch a registered campaign, refusing to substitute a different one."""
    if preset not in PRESETS:
        raise PresetError(
            f"unknown preset {preset!r}; registered presets are {preset_names()}"
        )
    campaigns = PRESETS[preset]
    if campaign is None:
        if len(campaigns) == 1:
            return next(iter(campaigns.values()))
        raise PresetError(
            f"preset {preset!r} covers campaigns {sorted(campaigns)}; name one with "
            "--campaign. Campaigns differ in state order and atom partition, so "
            "one cannot stand in for another."
        )
    if campaign not in campaigns:
        raise PresetError(
            f"preset {preset!r} has no campaign {campaign!r}; registered campaigns "
            f"are {sorted(campaigns)}. A campaign's state order and atom partition "
            "were established from the structure that was actually built; they are "
            "not transferable between campaigns, so nothing is substituted here."
        )
    return campaigns[campaign]
