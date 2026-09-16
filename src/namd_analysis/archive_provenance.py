"""Compatibility shim for the real-archive provenance model.

The resolvers this module used to install by replacing functions at import time
now live where they belong, and are always in force:

* :func:`namd_analysis.character.resolve_band_numbers`
* :func:`namd_analysis.character.resolve_namdtini`
* :func:`namd_analysis.character.resolve_cycle_period`
* :attr:`namd_analysis.populations.StateMap.band_numbers`
* :data:`namd_analysis.presets.CampaignPreset.band_numbers`

Patching them in had a defect worth recording, because it is the reason this
module is now a shim.  ``install()`` replaced ``character.preflight_report``
and friends, but ``character_cli`` binds those names with ``from .character
import ...`` at its own import time.  Import ``namd_analysis.character_cli``
before ``namd_analysis.dispatch`` and the CLI kept the *unpatched* function, so
whether the provenance model applied at all depended on import order -- and a
library caller importing ``namd_analysis.character`` directly never got it.
The post-hoc report patching also mislabelled provenance: a history with both a
``NAMDTINI`` header and a matching filename suffix was reported as
``SHPROP_filename_suffix`` when the metadata was what had actually been used.

``install()`` remains as a no-op so existing callers do not break.
"""

from __future__ import annotations

from .character import (  # noqa: F401  (re-exported for compatibility)
    resolve_band_numbers,
    resolve_cycle_period,
    resolve_namdtini,
)
from .presets import bands_for_campaign_name  # noqa: F401

__all__ = [
    "resolve_band_numbers",
    "resolve_cycle_period",
    "resolve_namdtini",
    "bands_for_campaign_name",
    "install",
]


def install() -> None:
    """No-op.  The provenance model is part of the normal APIs now."""
    return None
