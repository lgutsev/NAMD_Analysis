"""Analysis of saved CA-NAC / Hefei-NAMD output and VACF phonon spectra.

This package reads results that already exist on disk.  It performs no DFT,
no NAC calculation, no job submission and no electronic propagation, and it
never modifies, clips, reorders or renormalizes the raw inputs.
"""

__version__ = "0.7.1"

__all__ = ["__version__"]
