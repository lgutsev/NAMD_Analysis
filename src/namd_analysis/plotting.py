"""Figures.  Every curve drawn here is a quantity also written to CSV."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .fitting import ExponentialFit  # noqa: E402
from .populations import GroupSeries  # noqa: E402
from .vacf import Spectrum, VacfResult  # noqa: E402

FORMATS = ("png", "pdf")


def _save(figure, stem: Path) -> List[Path]:
    written = []
    for extension in FORMATS:
        path = stem.with_suffix(f".{extension}")
        figure.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def plot_populations(
    time_ns,
    series: Sequence[GroupSeries],
    stem: Path,
    title: str = "",
    survival=None,
    fit: Optional[ExponentialFit] = None,
) -> List[Path]:
    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    for group in series:
        line, = axes.plot(time_ns, group.values, label=group.name, linewidth=1.6)
        if group.sem is not None:
            axes.fill_between(
                time_ns,
                group.values - group.sem,
                group.values + group.sem,
                color=line.get_color(),
                alpha=0.2,
                linewidth=0,
            )
    if survival is not None:
        axes.plot(time_ns, survival, label="survival", linestyle="--",
                  color="black", linewidth=1.2)
    if fit is not None:
        import numpy as np

        window = np.linspace(fit.window[0], fit.window[1], 400)
        curve = fit.amplitude * np.exp(-(window - fit.window[0]) / fit.tau)
        axes.plot(
            window,
            curve,
            linestyle=":",
            color="red",
            linewidth=1.6,
            label=f"fit {fit.group}: tau = {fit.tau:.4g} {fit.tau_unit}",
        )
    axes.set_xlabel("Time (ns)")
    axes.set_ylabel("Population")
    if title:
        axes.set_title(title)
    axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes.legend(frameon=True)
    return _save(figure, stem)


def plot_spectra(
    spectra: Sequence[Spectrum],
    stem: Path,
    xlim: Tuple[float, float] = (0.0, 800.0),
    title: str = "Phonon spectral density",
    normalize: bool = False,
) -> List[Path]:
    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    for spectrum in spectra:
        values = spectrum.intensity
        if normalize:
            ceiling = float(max(abs(values.max()), abs(values.min())))
            if ceiling > 0:
                values = values / ceiling
        axes.plot(spectrum.frequency_cm1, values,
                  label=spectrum.label or "spectrum", linewidth=1.4)
    axes.set_xlim(*xlim)
    axes.set_xlabel(r"Frequency (cm$^{-1}$)")
    axes.set_ylabel("Spectral density" + (" (scaled to 1)" if normalize else " (arb. units)"))
    axes.set_title(title)
    axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes.legend(frameon=True)
    return _save(figure, stem)


def plot_vacf(vacf: VacfResult, stem: Path, title: str = "Velocity autocorrelation") -> List[Path]:
    figure, axes = plt.subplots(figsize=(7.5, 5.0))
    axes.plot(vacf.lags_fs, vacf.normalized, linewidth=1.4, color="tab:blue")
    if vacf.segment_values is not None:
        for segment in vacf.segment_values:
            axes.plot(
                vacf.lags_fs,
                segment / segment[0],
                linewidth=0.6,
                color="tab:blue",
                alpha=0.25,
            )
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("Lag (fs)")
    axes.set_ylabel("Normalized VACF")
    axes.set_title(title)
    axes.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    return _save(figure, stem)
