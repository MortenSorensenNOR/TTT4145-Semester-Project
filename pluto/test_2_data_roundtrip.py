"""Data roundtrip test for PlutoSDR loopback.

Verifies data survives the round trip by comparing TX bits vs RX bits.
Uses QPSK modulation with RRC pulse shaping.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "examples"

from modules.modulation import QPSK
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from pluto.loopback import setup_pluto, transmit_and_receive, SPS

RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
N_BITS = 2000


def estimate_phase_offset(rx_symbols: np.ndarray, tx_symbols: np.ndarray) -> float:
    """Estimate carrier phase offset using data-aided correlation."""
    return np.angle(np.sum(rx_symbols * np.conj(tx_symbols)))


def main():
    print("Setting up PlutoSDR for data roundtrip test...")
    sdr = setup_pluto()
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    qpsk = QPSK()

    tx_bits = np.random.randint(0, 2, N_BITS)
    tx_symbols = qpsk.bits2symbols(tx_bits)
    tx_symbols_norm = tx_symbols / np.sqrt(np.mean(np.abs(tx_symbols) ** 2))
    n_symbols = len(tx_symbols)

    guard = np.zeros(100, dtype=complex)
    frame = np.concatenate([guard, tx_symbols, guard])
    tx_signal = upsample_and_filter(frame, SPS, h_rrc)

    print(f"Transmitting {N_BITS} bits ({n_symbols} QPSK symbols)...")
    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=3)
    rx_filtered = np.convolve(rx_raw, h_rrc, mode="same")

    template = upsample_and_filter(tx_symbols[:50], SPS, h_rrc)
    corr = np.correlate(rx_filtered, template, mode="valid")
    offset = np.argmax(np.abs(corr))

    rx_symbols_raw = rx_filtered[offset::SPS][:n_symbols]

    power = np.mean(np.abs(rx_symbols_raw) ** 2)
    if power > 0:
        rx_symbols_raw = rx_symbols_raw / np.sqrt(power)

    phase_offset = estimate_phase_offset(rx_symbols_raw, tx_symbols_norm)
    rx_symbols = rx_symbols_raw * np.exp(-1j * phase_offset)
    print(f"Phase offset: {np.degrees(phase_offset):.1f}°")

    rx_bits = qpsk.symbols2bits(rx_symbols).flatten()

    if len(rx_bits) != len(tx_bits):
        print(f"Length mismatch: TX={len(tx_bits)}, RX={len(rx_bits)}")
        rx_bits = rx_bits[:len(tx_bits)]

    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)

    print(f"TX bits: {len(tx_bits)}")
    print(f"RX bits: {len(rx_bits)}")
    print(f"Bit errors: {errors}")
    print(f"BER: {ber:.6f}")

    if ber == 0:
        print("SUCCESS: All bits recovered correctly!")
    elif ber < 0.01:
        print("GOOD: BER < 1% (minor errors)")
    else:
        print("WARNING: High BER detected")

    # Align TX signal for comparison (need to account for filtering delay)
    filter_delay = (RRC_NUM_TAPS - 1) // 2
    tx_baseband = upsample_and_filter(tx_symbols_norm, SPS, h_rrc)
    rx_baseband = rx_filtered[offset : offset + len(tx_baseband)]
    rx_baseband = rx_baseband / np.sqrt(np.mean(np.abs(rx_baseband) ** 2))
    rx_baseband = rx_baseband * np.exp(-1j * phase_offset)

    fig = plt.figure(figsize=(14, 12))

    # Row 1: I/Q baseband comparison
    n_plot = min(400, len(tx_baseband))
    t = np.arange(n_plot) / SPS

    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(t, np.real(tx_baseband[:n_plot]), 'b-', linewidth=0.8, label="TX")
    ax1.plot(t, np.real(rx_baseband[:n_plot]), 'r-', linewidth=0.8, alpha=0.7, label="RX")
    ax1.plot(np.arange(0, n_plot, SPS) / SPS, np.real(tx_symbols_norm[:n_plot//SPS]), 'bo', markersize=4)
    ax1.set_title("In-Phase (I) Component")
    ax1.set_xlabel("Symbol")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, np.imag(tx_baseband[:n_plot]), 'b-', linewidth=0.8, label="TX")
    ax2.plot(t, np.imag(rx_baseband[:n_plot]), 'r-', linewidth=0.8, alpha=0.7, label="RX")
    ax2.plot(np.arange(0, n_plot, SPS) / SPS, np.imag(tx_symbols_norm[:n_plot//SPS]), 'bo', markersize=4)
    ax2.set_title("Quadrature (Q) Component")
    ax2.set_xlabel("Symbol")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Row 2: Eye diagrams
    n_eye_symbols = min(200, n_symbols - 10)
    eye_len = 2 * SPS  # 2 symbol periods per trace

    ax3 = fig.add_subplot(3, 2, 3)
    t_eye = np.linspace(-1, 1, eye_len)
    for i in range(n_eye_symbols):
        start = i * SPS
        if start + eye_len <= len(rx_baseband):
            ax3.plot(t_eye, np.real(rx_baseband[start:start + eye_len]), 'b-', alpha=0.1, linewidth=0.5)
    ax3.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax3.set_title("Eye Diagram - In-Phase (I)")
    ax3.set_xlabel("Symbol Period")
    ax3.set_ylabel("Amplitude")
    ax3.set_xlim(-1, 1)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    for i in range(n_eye_symbols):
        start = i * SPS
        if start + eye_len <= len(rx_baseband):
            ax4.plot(t_eye, np.imag(rx_baseband[start:start + eye_len]), 'b-', alpha=0.1, linewidth=0.5)
    ax4.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax4.set_title("Eye Diagram - Quadrature (Q)")
    ax4.set_xlabel("Symbol Period")
    ax4.set_ylabel("Amplitude")
    ax4.set_xlim(-1, 1)
    ax4.grid(True, alpha=0.3)

    # Row 3: Constellations
    ax5 = fig.add_subplot(3, 2, 5)
    constellation = qpsk.symbol_mapping / np.sqrt(np.mean(np.abs(qpsk.symbol_mapping) ** 2))
    ax5.scatter(np.real(rx_symbols_raw), np.imag(rx_symbols_raw), alpha=0.3, s=5, label="RX (raw)")
    ax5.scatter(np.real(constellation), np.imag(constellation), c="red", s=100, marker="x", label="Ideal")
    ax5.set_title(f"Constellation Before Phase Corr ({np.degrees(phase_offset):.1f}°)")
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect("equal")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(3, 2, 6)
    ax6.scatter(np.real(rx_symbols), np.imag(rx_symbols), alpha=0.3, s=5, label="RX (corrected)")
    ax6.scatter(np.real(constellation), np.imag(constellation), c="red", s=100, marker="x", label="Ideal")
    ax6.set_title(f"Constellation After Phase Corr (BER={ber:.6f})")
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect("equal")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, "data_roundtrip_test.png")
    plt.savefig(filepath, dpi=150)
    print(f"Saved plot: {filepath}")
    plt.show()


if __name__ == "__main__":
    main()
