"""Constellation verification test for PlutoSDR loopback.

Verifies BPSK, QPSK, and QAM16 constellations appear correctly after TX->RX.
Plots received vs ideal constellations and computes EVM.
"""

from pathlib import Path

import adi
import matplotlib.pyplot as plt
import numpy as np

from modules.modulation import BPSK, QAM, QPSK
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from pluto.config import SPS
from pluto.loopback import setup_pluto, transmit_and_receive

PLOT_DIR = "examples/data"
RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
N_SYMBOLS = 500


def estimate_phase_offset(rx_symbols: np.ndarray, tx_symbols: np.ndarray) -> float:
    """Estimate carrier phase offset using data-aided correlation."""
    return np.angle(np.sum(rx_symbols * np.conj(tx_symbols)))


def compute_evm(rx_symbols: np.ndarray, constellation: np.ndarray) -> float:
    """Compute Error Vector Magnitude as a percentage."""
    indices = np.argmin(
        np.abs(rx_symbols.reshape(-1, 1) - constellation.reshape(1, -1)),
        axis=1,
    )
    ideal = constellation[indices]
    error = rx_symbols - ideal
    ref_power = np.mean(np.abs(ideal) ** 2)
    error_power = np.mean(np.abs(error) ** 2)
    return 100.0 * np.sqrt(error_power / ref_power)


def normalize_and_extract(rx_filtered: np.ndarray, n_symbols: int, sps: int) -> np.ndarray:
    """Downsample and amplitude-normalize received symbols."""
    symbols = rx_filtered[::sps][:n_symbols]
    power = np.mean(np.abs(symbols) ** 2)
    if power > 0:
        symbols = symbols / np.sqrt(power)
    return symbols


def test_constellation(
    modulator: BPSK | QPSK | QAM, h_rrc: np.ndarray, sdr: adi.Pluto,
) -> tuple[np.ndarray, float]:
    """Test a single modulation scheme and return (rx_symbols, evm)."""
    rng = np.random.default_rng()
    bits = rng.integers(0, 2, N_SYMBOLS * modulator.bits_per_symbol)
    tx_symbols = modulator.bits2symbols(bits)
    tx_symbols_norm = tx_symbols / np.sqrt(np.mean(np.abs(tx_symbols) ** 2))

    guard = np.zeros(100, dtype=complex)
    frame = np.concatenate([guard, tx_symbols, guard])
    tx_signal = upsample_and_filter(frame, SPS, h_rrc)

    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=3)
    rx_filtered = np.convolve(rx_raw, h_rrc, mode="same")

    corr = np.correlate(rx_filtered, upsample_and_filter(tx_symbols[:50], SPS, h_rrc), mode="valid")
    offset = np.argmax(np.abs(corr))

    rx_symbols = normalize_and_extract(rx_filtered[offset:], N_SYMBOLS, SPS)

    phase_offset = estimate_phase_offset(rx_symbols, tx_symbols_norm)
    rx_symbols = rx_symbols * np.exp(-1j * phase_offset)

    constellation = modulator.symbol_mapping / np.sqrt(np.mean(np.abs(modulator.symbol_mapping) ** 2))
    evm = compute_evm(rx_symbols, constellation)

    return rx_symbols, evm


def main() -> None:
    """Run constellation verification test for all modulation schemes."""
    sdr = setup_pluto()
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)

    modulators = [
        (BPSK(), "BPSK"),
        (QPSK(), "QPSK"),
        (QAM(16), "QAM16"),
    ]

    _fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (modulator, name) in zip(axes, modulators, strict=False):
        rx_symbols, evm = test_constellation(modulator, h_rrc, sdr)

        constellation = modulator.symbol_mapping / np.sqrt(np.mean(np.abs(modulator.symbol_mapping) ** 2))
        ax.scatter(np.real(rx_symbols), np.imag(rx_symbols), alpha=0.5, s=10, label="RX")
        ax.scatter(np.real(constellation), np.imag(constellation), c="red", s=100, marker="x", label="Ideal")
        ax.set_title(f"{name} (EVM={evm:.1f}%)")
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(visible=True, alpha=0.3)
        ax.legend()

    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    filepath = Path(PLOT_DIR) / "constellation_test.png"
    plt.savefig(filepath, dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
