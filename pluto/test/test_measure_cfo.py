"""Measure actual carrier frequency offset using FFT."""

import argparse
from typing import cast

import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, RX_BUFFER_SIZE, SAMPLE_RATE


def measure_cfo_fft(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate CFO using FFT peak detection."""
    # Use a large FFT for fine resolution
    n_fft = 2**16

    # Window the signal
    window = np.hanning(len(samples))
    windowed = samples * window

    # Zero-pad and FFT
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / sample_rate))

    # Find peak (excluding DC)
    magnitude = np.abs(spectrum)
    dc_idx = n_fft // 2
    exclude_width = 100  # Exclude +-100 bins around DC
    magnitude[dc_idx - exclude_width : dc_idx + exclude_width] = 0

    peak_idx = np.argmax(magnitude)
    return freqs[peak_idx]



def main() -> None:
    """Measure and display CFO between TX and RX oscillators."""
    parser = argparse.ArgumentParser(description="Measure carrier frequency offset")
    parser.add_argument("--pluto-ip", default="192.168.2.1", help="PlutoSDR IP address (default: %(default)s)")
    args = parser.parse_args()

    sdr = create_pluto(f"ip:{args.pluto_ip}")
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    sdr.rx()  # flush


    cfo_measurements = []

    try:
        while True:
            samples = cast("np.ndarray", sdr.rx())

            cfo = measure_cfo_fft(samples, SAMPLE_RATE)
            cfo_measurements.append(cfo)
            mean_cfo = np.mean(cfo_measurements)
            print(f"CFO: {cfo:+.1f} Hz  (mean: {mean_cfo:+.1f} Hz, n={len(cfo_measurements)})")

    except KeyboardInterrupt:
        if cfo_measurements:
            print(f"\nFinal mean CFO: {np.mean(cfo_measurements):+.1f} Hz")


if __name__ == "__main__":
    main()
