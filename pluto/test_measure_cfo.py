"""Measure actual carrier frequency offset using FFT."""

import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, SAMPLE_RATE, RX_BUFFER_SIZE


def measure_cfo_fft(samples: np.ndarray, sample_rate: float) -> float:
    """Estimate CFO using FFT peak detection."""
    # Use a large FFT for fine resolution
    n_fft = 2 ** 16

    # Window the signal
    window = np.hanning(len(samples))
    windowed = samples * window

    # Zero-pad and FFT
    spectrum = np.fft.fftshift(np.fft.fft(windowed, n_fft))
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, 1 / sample_rate))

    # Find peak (excluding DC)
    magnitude = np.abs(spectrum)
    dc_idx = n_fft // 2
    exclude_width = 100  # Exclude ±100 bins around DC
    magnitude[dc_idx - exclude_width : dc_idx + exclude_width] = 0

    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]

    return peak_freq


def main():
    sdr = create_pluto()
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    sdr.rx()  # flush

    print(f"Listening on {CENTER_FREQ / 1e6:.0f} MHz...")
    print(f"Sample rate: {SAMPLE_RATE / 1e6:.1f} MHz")
    print("Measuring CFO via FFT (Ctrl+C to stop):\n")

    cfo_measurements = []

    try:
        while True:
            samples = sdr.rx()

            # Measure CFO
            cfo = measure_cfo_fft(samples, SAMPLE_RATE)
            cfo_measurements.append(cfo)

            # Running average
            avg_cfo = np.mean(cfo_measurements[-20:])

            print(f"CFO: {cfo:+8.0f} Hz  (avg: {avg_cfo:+8.0f} Hz)")

    except KeyboardInterrupt:
        if cfo_measurements:
            final_avg = np.mean(cfo_measurements)
            print(f"\n\nFinal average CFO: {final_avg:+.0f} Hz")
            print(f"Use: --cfo-offset {int(final_avg)}")


if __name__ == "__main__":
    main()
