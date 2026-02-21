"""Quick RX power level check to verify signal is reaching the receiver."""

import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, SAMPLE_RATE, RX_BUFFER_SIZE

# PlutoSDR ADC is 12-bit, full scale is 2^11 = 2048 per I/Q component
ADC_FULL_SCALE = 2048


def main():
    sdr = create_pluto()
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    sdr.rx()  # flush stale buffer

    print(f"Listening on {CENTER_FREQ / 1e6:.0f} MHz...")
    print(f"ADC full scale: ±{ADC_FULL_SCALE}")
    print("Measuring RX levels (Ctrl+C to stop):\n")

    try:
        while True:
            samples = sdr.rx()
            # Check actual I/Q component ranges
            max_i = np.max(np.abs(np.real(samples)))
            max_q = np.max(np.abs(np.imag(samples)))
            max_component = max(max_i, max_q)

            # Normalized power relative to full scale
            normalized = samples / ADC_FULL_SCALE
            pwr_dbfs = 10 * np.log10(np.mean(np.abs(normalized) ** 2) + 1e-12)

            clipping = "CLIPPING!" if max_component > ADC_FULL_SCALE * 0.95 else ""
            print(f"Max I/Q: {max_component:6.0f} / {ADC_FULL_SCALE}  |  Power: {pwr_dbfs:+6.1f} dBFS  {clipping}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
