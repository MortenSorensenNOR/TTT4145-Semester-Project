"""Plotting utilities for signal visualization and analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

if TYPE_CHECKING:
    from modules.modulation import Modulator


def plot_iq(
    *signals: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    sample_rate: float | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot I and Q components of complex signals."""
    if labels is None:
        labels = [f"Signal {i + 1}" for i in range(len(signals))]

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for sig, label in zip(signals, labels, strict=True):
        if sample_rate:
            t = np.arange(len(sig)) / sample_rate * 1e6  # microseconds
            ax_i.plot(t, np.real(sig), label=label)
            ax_q.plot(t, np.imag(sig), label=label)
            ax_q.set_xlabel("Time [us]")
        else:
            ax_i.plot(np.real(sig), label=label)
            ax_q.plot(np.imag(sig), label=label)
            ax_q.set_xlabel("Sample")

    ax_i.set_ylabel("I (Real)")
    ax_q.set_ylabel("Q (Imag)")
    ax_i.legend()
    ax_q.legend()
    ax_i.grid(visible=True, alpha=0.3)
    ax_q.grid(visible=True, alpha=0.3)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, (ax_i, ax_q)


def plot_constellation(
    *signals: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot constellation diagram of complex signals."""
    if labels is None:
        labels = [f"Signal {i + 1}" for i in range(len(signals))]

    fig, ax = plt.subplots(figsize=(6, 6))

    for sig, label in zip(signals, labels, strict=True):
        ax.scatter(np.real(sig), np.imag(sig), s=10, alpha=0.6, label=label)

    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.axis("equal")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_constellation_confidence(
    rx_symbols: np.ndarray,
    llrs: np.ndarray,
    tx_symbols: np.ndarray | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot constellation with symbols colored by soft decision confidence."""
    # Confidence = minimum |LLR| across bits for each symbol
    confidence = np.min(np.abs(llrs), axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot received symbols colored by confidence
    scatter = ax.scatter(
        np.real(rx_symbols),
        np.imag(rx_symbols),
        c=confidence,
        cmap="plasma",
        alpha=0.7,
        s=15,
        label="Received",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Confidence (min |LLR|)")

    # Overlay ideal constellation points
    if tx_symbols is not None:
        unique_tx = np.unique(tx_symbols)
        ax.scatter(
            np.real(unique_tx),
            np.imag(unique_tx),
            c="white",
            edgecolors="black",
            s=100,
            marker="s",
            linewidths=1.5,
            label="Ideal",
            zorder=10,
        )

    # Decision boundaries
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("I (Real)")
    ax.set_ylabel("Q (Imag)")
    ax.legend(loc="upper right")
    ax.grid(visible=True, alpha=0.3)
    ax.axis("equal")

    if title:
        ax.set_title(title)
    else:
        ax.set_title("Constellation with Soft Decision Confidence")

    plt.tight_layout()
    return fig, ax


def plot_spectrum(
    *signals: np.ndarray,
    sample_rate: float = 1.0,
    labels: list[str] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot power spectrum of complex signals."""
    if labels is None:
        labels = [f"Signal {i + 1}" for i in range(len(signals))]

    fig, ax = plt.subplots(figsize=(10, 4))

    for sig, label in zip(signals, labels, strict=True):
        n = len(sig)
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1 / sample_rate)) / 1e6  # MHz
        spectrum = np.fft.fftshift(np.fft.fft(sig))
        power_db = 20 * np.log10(np.abs(spectrum) + np.finfo(float).tiny)
        power_db -= np.max(power_db)  # normalize to 0 dB peak
        ax.plot(freq, power_db, label=label)

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Power [dB]")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    ax.set_ylim(bottom=-60, top=5)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_llr_heatmap(
    modulator: Modulator,
    sigma_sq: float = 0.1,
    grid_size: int = 100,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot heatmaps showing LLR values across the complex plane."""
    extent = 1.5
    re = np.linspace(-extent, extent, grid_size)
    im = np.linspace(-extent, extent, grid_size)
    re_grid, im_grid = np.meshgrid(re, im)
    symbols_grid = (re_grid + 1j * im_grid).flatten()

    llrs = modulator.symbols2bits_soft(symbols_grid, sigma_sq=sigma_sq)
    llr_bit0 = llrs[:, 0].reshape(grid_size, grid_size)
    llr_bit1 = llrs[:, 1].reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(
        llr_bit0,
        extent=[-extent, extent, -extent, extent],
        origin="lower",
        cmap="RdBu",
        aspect="equal",
    )
    axes[0].axvline(x=0, color="k", linestyle="--", linewidth=1, label="Decision boundary")
    axes[0].plot(np.real(modulator.symbol_mapping), np.imag(modulator.symbol_mapping), "ko", markersize=8)
    axes[0].set_xlabel("Real")
    axes[0].set_ylabel("Imaginary")
    axes[0].set_title("LLR for Bit 0 (I)\nBlue: likely 0, Red: likely 1")
    plt.colorbar(im0, ax=axes[0], label="LLR")

    im1 = axes[1].imshow(
        llr_bit1,
        extent=[-extent, extent, -extent, extent],
        origin="lower",
        cmap="RdBu",
        aspect="equal",
    )
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=1, label="Decision boundary")
    axes[1].plot(np.real(modulator.symbol_mapping), np.imag(modulator.symbol_mapping), "ko", markersize=8)
    axes[1].set_xlabel("Real")
    axes[1].set_ylabel("Imaginary")
    axes[1].set_title("LLR for Bit 1 (Q)\nBlue: likely 0, Red: likely 1")
    plt.colorbar(im1, ax=axes[1], label="LLR")

    confidence = np.minimum(np.abs(llr_bit0), np.abs(llr_bit1))
    im2 = axes[2].imshow(
        confidence,
        extent=[-extent, extent, -extent, extent],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    axes[2].axvline(x=0, color="w", linestyle="--", linewidth=1)
    axes[2].axhline(y=0, color="w", linestyle="--", linewidth=1)
    axes[2].plot(np.real(modulator.symbol_mapping), np.imag(modulator.symbol_mapping), "wo", markersize=8)
    axes[2].set_xlabel("Real")
    axes[2].set_ylabel("Imaginary")
    axes[2].set_title(f"Min Confidence |LLR| (σ²={sigma_sq})\nDark = uncertain")
    plt.colorbar(im2, ax=axes[2], label="|LLR|")

    plt.tight_layout()
    return fig, axes
