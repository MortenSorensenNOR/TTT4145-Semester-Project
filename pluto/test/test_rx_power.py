"""Quick RX power level check to verify signal is reaching the receiver."""

import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, RX_BUFFER_SIZE, SAMPLE_RATE

# PlutoSDR ADC is 12-bit, full scale is 2^11 = 2048 per I/Q component
ADC_FULL_SCALE = 2048


def main() -> None:
    """Measure and display RX signal power levels."""
    sdr = create_pluto()
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    sdr.rx()  # flush stale buffer


    try:
        while True:
            samples = sdr.rx()
            # Check actual I/Q component ranges
            max_i = np.max(np.abs(np.real(samples)))
            max_q = np.max(np.abs(np.imag(samples)))
            max_component = max(max_i, max_q)

            # Normalized power relative to full scale
            normalized = samples / ADC_FULL_SCALE
            10 * np.log10(np.mean(np.abs(normalized) ** 2) + 1e-12)

            "CLIPPING!" if max_component > ADC_FULL_SCALE * 0.95 else ""
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
