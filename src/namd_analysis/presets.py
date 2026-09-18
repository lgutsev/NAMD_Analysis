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
from typing import Any, Dict, List, Optional, Tuple


class PresetError(ValueError):
    """Raised when a preset does not match the campaign it was applied to."""


@dataclass
class CampaignPreset:
    """One campaign's declared state order and physical atom partition."""

    preset: str
    campaign: str
    name: str
    description: str
    #: Zero-based SHPROP columns, in basis order.  The preparation command
    #: still infers the column block from the table and refuses to proceed if
    #: the two disagree.
    population_columns: List[int]
    #: Exact VASP band numbers, one per population column, in the same order.
    #: This is campaign provenance: production SHPROP files are plain numeric
    #: tables and need not carry BMIN/BMAX at all, so the basis cannot be
    #: recovered from them. An empty list means the campaign is registered but
    #: its basis has not been established yet, and preparation refuses rather
    #: than borrowing another campaign's.
    band_numbers: List[int]
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
            "band_numbers": list(self.band_numbers),
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
    band_numbers=[976, 977, 978, 979, 980, 981],
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

#: Registered campaigns, keyed by ``(preset, campaign)``.
#:
#: B and C are campaign labels used elsewhere in this repository, but their
#: structures were built differently: a different band order, a different atom
#: partition, possibly a different cycle provenance.  None of that is
#: recoverable from A, so they are deliberately absent until their own
#: provenance is supplied.  :func:`load_preset` says so by name rather than
#: falling back to A.
PRESETS: Dict[str, Dict[str, CampaignPreset]] = {"bcf_pcbm": {"A": _A}}

#: Campaigns we know exist but whose provenance has not been established.
#: Naming them here turns "unknown campaign" into "known campaign, unresolved
#: provenance", which is a different and more useful error.
UNRESOLVED_CAMPAIGNS: Dict[str, Dict[str, str]] = {
    "bcf_pcbm": {
        "B": (
            "campaign B was built from a different structure than A. Its band "
            "order, atom partition and cycle provenance have not been supplied, "
            "and none of them transfer from A"
        ),
        "C": (
            "campaign C was built from a different structure than A. Its band "
            "order, atom partition and cycle provenance have not been supplied, "
            "and none of them transfer from A"
        ),
    }
}


def bands_for_campaign_name(name: str) -> Optional[Tuple[List[int], str]]:
    """Exact VASP bands for a state map whose ``name`` matches a registered campaign.

    Returns ``(bands, source)`` or ``None``.  A registered campaign with an
    empty band list returns ``None``: registered is not the same as resolved.
    """
    for preset, campaigns in PRESETS.items():
        for campaign, entry in campaigns.items():
            if entry.name == name and entry.band_numbers:
                return list(entry.band_numbers), f"preset_{preset}_{campaign}"
    return None


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
        pending = UNRESOLVED_CAMPAIGNS.get(preset, {}).get(campaign)
        if pending is not None:
            raise PresetError(
                f"preset {preset!r} campaign {campaign!r} is known but unresolved: "
                f"{pending}. Supply its band order and atom partition before "
                "preparing it."
            )
        raise PresetError(
            f"preset {preset!r} has no campaign {campaign!r}; registered campaigns "
            f"are {sorted(campaigns)}. A campaign's state order and atom partition "
            "were established from the structure that was actually built; they are "
            "not transferable between campaigns, so nothing is substituted here."
        )
    return campaigns[campaign]
