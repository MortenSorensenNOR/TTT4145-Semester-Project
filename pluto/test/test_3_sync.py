"""Synchronization test for PlutoSDR loopback.

Tests preamble detection, timing recovery, and CFO estimation.
Injects real CFO by offsetting TX/RX LO frequencies.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import adi
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from modules.modulation import QPSK
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import Synchronizer, SynchronizerConfig, build_preamble
from pluto.config import CENTER_FREQ, SAMPLE_RATE, SPS
from pluto.loopback import setup_pluto, transmit_and_receive

PLOT_DIR = "examples/data"
RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
N_PAYLOAD_SYMBOLS = 200


def run_sync_test(
    sdr: adi.Pluto, h_rrc: np.ndarray, sync: Synchronizer, injected_cfo: float = 0,
) -> dict:
    """Run a single sync test and return results."""
    rng = np.random.default_rng()
    qpsk = QPSK()
    payload_bits = rng.integers(0, 2, N_PAYLOAD_SYMBOLS * 2)
    payload_symbols = qpsk.bits2symbols(payload_bits)

    preamble = build_preamble(sync.config)
    frame = np.concatenate([preamble, payload_symbols])

    guard = np.zeros(200, dtype=complex)
    tx_frame = np.concatenate([guard, frame, guard])
    tx_signal = upsample_and_filter(tx_frame, SPS, h_rrc)

    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=3)
    rx_filtered = np.convolve(rx_raw, h_rrc, mode="same")

    result = sync.detect_preamble(rx_filtered, SAMPLE_RATE)

    return {
        "success": result.success,
        "reason": result.reason,
        "cfo_estimated": result.cfo_hat_hz,
        "cfo_injected": injected_cfo,
        "timing": result.d_hat,
        "n_peaks": len(result.peak_indices),
        "peak_indices": result.peak_indices,
        "rx_filtered": rx_filtered,
        "tx_signal": tx_signal,
        "sync": sync,
    }


def plot_sync_result(result: dict, title: str) -> "Figure":
    """Plot the synchronization process."""
    rx = result["rx_filtered"]
    sync = result["sync"]
    peaks = result["peak_indices"]
    timing = result["timing"]

    template_short = sync.template_short
    corr = np.abs(np.correlate(rx, template_short, mode="same"))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(np.abs(rx), linewidth=0.5)
    ax.set_title(f"{title} - RX Signal Magnitude")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Magnitude")
    if timing > 0:
        ax.axvline(timing, color="r", linestyle="--", label=f"Timing: {timing}")
        ax.legend()

    ax = axes[1]
    ax.plot(corr, linewidth=0.5)
    for _i, p in enumerate(peaks):
        ax.axvline(p, color="g", alpha=0.7, linestyle="--")
    ax.set_title(f"Short ZC Correlation (found {len(peaks)} peaks)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Correlation")

    ax = axes[2]
    cfo_est = result["cfo_estimated"]
    cfo_inj = result["cfo_injected"]
    n = np.arange(min(5000, len(rx)))
    phase_uncorrected = np.angle(rx[: len(n)])
    phase_corrected = np.angle(rx[: len(n)] * np.exp(-1j * 2 * np.pi * cfo_est / SAMPLE_RATE * n))
    ax.plot(np.unwrap(phase_uncorrected), label="Before CFO correction", alpha=0.7)
    ax.plot(np.unwrap(phase_corrected), label="After CFO correction", alpha=0.7)
    ax.set_title(f"Phase (injected={cfo_inj:+.0f} Hz, estimated={cfo_est:+.0f} Hz)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Phase (rad)")
    ax.legend()

    plt.tight_layout()
    return fig


def test_with_cfo_offset(
    tx_lo: int, rx_lo: int, h_rrc: np.ndarray, sync: Synchronizer,
) -> dict:
    """Test synchronization with specific TX/RX LO settings."""
    injected_cfo = tx_lo - rx_lo  # CFO as seen by receiver

    sdr = setup_pluto(freq_hz=tx_lo)
    sdr.rx_lo = int(rx_lo)

    return run_sync_test(sdr, h_rrc, sync, injected_cfo)


def main() -> None:
    """Run synchronization tests with various CFO offsets over PlutoSDR loopback."""
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync_config = SynchronizerConfig()
    sync = Synchronizer(sync_config, sps=SPS, rrc_taps=h_rrc)

    # Max unambiguous CFO ~ 1/(2*N_short*T_s) = 1/(2*19*4us) ~ 6.6 kHz
    cfo_offsets = [0, 1000, -1000, 3000, -3000, 5000, -5000, 6000, -6000]
    results = []

    for _i, offset in enumerate(cfo_offsets):
        result = test_with_cfo_offset(CENTER_FREQ, CENTER_FREQ - offset, h_rrc, sync)
        label = f"{offset:+d} Hz" if offset != 0 else "Baseline"
        results.append((label, result))

    all_passed = True
    for _name, r in results:
        if r["success"]:
            error = abs(r["cfo_estimated"] - r["cfo_injected"])
            tolerance = 500 + abs(r["cfo_injected"]) * 0.05  # 5% + 500 Hz baseline
            if error >= tolerance:
                all_passed = False
        else:
            all_passed = False

    if all_passed:
        pass
    else:
        pass

    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    plot_indices = [0, 5, 8]  # baseline, +5kHz, -6kHz
    for i in plot_indices:
        if i < len(results):
            name, r = results[i]
            if r["success"]:
                fig = plot_sync_result(r, f"CFO Test: {name}")
                filename = f"sync_test_{name.replace(' ', '_').replace('+', 'p').replace('-', 'n')}.png"
                filepath = Path(PLOT_DIR) / filename
                fig.savefig(filepath, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
