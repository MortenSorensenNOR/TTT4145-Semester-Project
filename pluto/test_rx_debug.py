"""Debug RX to see preamble detection and correlation values."""

import numpy as np

from pluto import create_pluto
from pluto.config import (
    CENTER_FREQ,
    SAMPLE_RATE,
    RX_BUFFER_SIZE,
    SYNC_CONFIG,
    SPS,
    RRC_ALPHA,
    RRC_NUM_TAPS,
)
from modules.pulse_shaping import rrc_filter
from modules.synchronization import Synchronizer


def main():
    # Setup SDR
    sdr = create_pluto()
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    # Setup sync
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(SYNC_CONFIG, sps=SPS, rrc_taps=h_rrc)

    sdr.rx()  # flush

    print(f"Listening on {CENTER_FREQ / 1e6:.0f} MHz...")
    print(f"Preamble correlation threshold: {sync.config.peak_threshold}")
    print("Checking for preambles (Ctrl+C to stop):\n")

    try:
        while True:
            samples = sdr.rx()

            # Apply matched filter
            filtered = np.convolve(samples, h_rrc, mode="same")

            # Get correlation metrics from synchronizer
            result = sync.detect_preamble(filtered, SAMPLE_RATE)

            # Also compute raw correlation manually to see values
            short_zc = sync.zc_short
            short_len = len(short_zc) * SPS

            if len(filtered) >= short_len:
                # Correlate with short ZC
                corr = np.abs(np.correlate(filtered[:short_len * 4], np.repeat(short_zc, SPS), mode="valid"))
                max_corr = np.max(corr) if len(corr) > 0 else 0

                # Normalize
                sig_power = np.mean(np.abs(filtered[:short_len * 4]) ** 2)
                ref_power = np.mean(np.abs(np.repeat(short_zc, SPS)) ** 2)
                norm_corr = max_corr / (np.sqrt(sig_power * ref_power) * len(short_zc) * SPS + 1e-12)

                status = "DETECTED!" if result.success else ""
                print(f"Max correlation: {norm_corr:.4f} (threshold: {sync.config.peak_threshold})  {status}")

                if result.success:
                    print(f"  -> CFO estimate: {result.cfo_hat_hz:+.1f} Hz")
                    print(f"  -> Long ZC start: {result.long_zc_start}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
