"""Plots the constellations of the modulation schemes."""

import matplotlib.pyplot as plt
import numpy as np

from modules.modulation import BPSK, QPSK, EightPSK, QAM


def plot_constellation(mod, ax, title):
    """Helper function to plot a constellation."""
    symbols = mod.symbol_mapping
    ax.scatter(np.real(symbols), np.imag(symbols))
    ax.set_title(title)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    for i, symbol in enumerate(symbols):
        ax.annotate(f'{mod.bit_mapping[i]}', (np.real(symbol), np.imag(symbol)))


def main():
    """Create and save constellation plots."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # BPSK
    bpsk = BPSK()
    plot_constellation(bpsk, axs[0, 0], "BPSK Constellation")

    # QPSK
    qpsk = QPSK()
    plot_constellation(qpsk, axs[0, 1], "QPSK Constellation")

    # 8-PSK
    eight_psk = EightPSK()
    plot_constellation(eight_psk, axs[1, 0], "8-PSK Constellation")

    # 16-QAM
    qam16 = QAM(16)
    plot_constellation(qam16, axs[1, 1], "16-QAM Constellation")

    fig.tight_layout()
    plt.savefig("examples/constellations.png")
    print("Saved constellation plot to examples/constellations.png")


if __name__ == "__main__":
    main()
