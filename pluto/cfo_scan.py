"""Continuously samples a large buffer, prints and plots the peak FFT bin.

Useful for debugging CFO between two Plutos — tune center_freq to roughly
where the TX carrier is and watch the reported peak drift.

Usage:
    uv run python pluto/cfo_scan.py --uri ip:192.168.2.1 --freq 2400000000
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import adi
from pluto.config import PIPELINE

SAMPLE_RATE = PIPELINE.SAMPLE_RATE
BUF_SIZE = 131072  # 128k samples
PEAK_HISTORY = 200  # number of iterations to keep in the peak-offset time series


def compute_spectrum(samples: np.ndarray, sample_rate: int):
    """Return (freqs_hz, power_db, peak_offset_hz)."""
    window = np.hanning(len(samples))
    spectrum = np.fft.fft(samples * window)
    spectrum = np.fft.fftshift(spectrum)
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1.0 / sample_rate))
    power_db = 20 * np.log10(np.abs(spectrum) / len(samples) + 1e-12)
    peak_bin = int(np.argmax(power_db))
    return freqs, power_db, float(freqs[peak_bin])


def main() -> None:
    parser = argparse.ArgumentParser(description="CFO peak-frequency scanner")
    parser.add_argument("--uri", default="ip:192.168.2.1", help="PlutoSDR URI")
    parser.add_argument("--freq", type=int, required=True, help="Center frequency in Hz")
    parser.add_argument("--gain-mode", default="slow_attack", help="RX gain mode")
    args = parser.parse_args()

    sdr = adi.Pluto(uri=args.uri)
    # sdr.gain_control_mode_chan0 = args.gain_mode
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = 50
    sdr.rx_lo = args.freq
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_buffer_size = BUF_SIZE

    print(f"Connected to {args.uri}")
    print(f"Center freq : {args.freq / 1e6:.6f} MHz")
    print(f"Sample rate : {SAMPLE_RATE / 1e6:.1f} MSPS")
    print(f"Buffer size : {BUF_SIZE} samples  ({BUF_SIZE / SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"FFT res     : {SAMPLE_RATE / BUF_SIZE:.2f} Hz/bin")
    print()

    # Flush stale DMA buffers so AGC settles
    for _ in range(5):
        sdr.rx()

    # --- build figure ---
    fig, (ax_spec, ax_hist) = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle(f"CFO scan  —  center {args.freq / 1e6:.3f} MHz", fontsize=11)

    # Top: live spectrum
    freqs_init = np.fft.fftshift(np.fft.fftfreq(BUF_SIZE, d=1.0 / SAMPLE_RATE))
    (line_spec,) = ax_spec.plot(freqs_init / 1e3, np.full(BUF_SIZE, -120.0), lw=0.8, color="steelblue")
    (peak_dot,) = ax_spec.plot([], [], "r+", ms=10, mew=2, label="peak")
    ax_spec.set_xlim(freqs_init[0] / 1e3, freqs_init[-1] / 1e3)
    ax_spec.set_ylim(-120, 10)
    ax_spec.set_xlabel("Offset from center (kHz)")
    ax_spec.set_ylabel("Power (dB)")
    ax_spec.set_title("Spectrum")
    ax_spec.legend(loc="upper right")
    ax_spec.grid(True, alpha=0.3)
    ax_spec.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Bottom: peak offset history
    peak_history: list[float] = []
    (line_hist,) = ax_hist.plot([], [], color="tomato", lw=1.0)
    ax_hist.axhline(0, color="gray", lw=0.6, ls="--")
    ax_hist.set_xlim(0, PEAK_HISTORY)
    ax_hist.set_xlabel("Iteration")
    ax_hist.set_ylabel("Peak offset (Hz)")
    ax_hist.set_title("Peak offset over time")
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.ion()
    plt.show()

    try:
        while True:
            samples = np.array(sdr.rx(), dtype=np.complex64)
            freqs, power_db, offset_hz = compute_spectrum(samples, SAMPLE_RATE)
            abs_hz = args.freq + offset_hz

            print(
                f"peak offset: {offset_hz:+10.2f} Hz  |  "
                f"absolute: {abs_hz / 1e6:.6f} MHz"
            )

            # Update spectrum line
            line_spec.set_ydata(power_db)
            peak_dot.set_data([offset_hz / 1e3], [power_db[np.argmax(power_db)]])

            # Update history
            peak_history.append(offset_hz)
            if len(peak_history) > PEAK_HISTORY:
                peak_history.pop(0)
            xs = list(range(len(peak_history)))
            line_hist.set_data(xs, peak_history)
            ax_hist.set_xlim(0, max(PEAK_HISTORY, len(peak_history)))
            if len(peak_history) > 1:
                margin = max(50.0, (max(peak_history) - min(peak_history)) * 0.15)
                ax_hist.set_ylim(min(peak_history) - margin, max(peak_history) + margin)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
