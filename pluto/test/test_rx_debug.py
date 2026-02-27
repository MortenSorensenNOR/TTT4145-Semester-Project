"""Debug RX to see preamble detection and correlation values."""

import argparse

import numpy as np

from modules.pulse_shaping import rrc_filter
from modules.synchronization import Synchronizer
from pluto import create_pluto
from pluto.config import (
    CENTER_FREQ,
    RRC_ALPHA,
    RRC_NUM_TAPS,
    RX_BUFFER_SIZE,
    SAMPLE_RATE,
    SPS,
    SYNC_CONFIG,
)


def main() -> None:
    """Run RX debug session with live frame decoding output."""
    parser = argparse.ArgumentParser(description="Debug RX preamble detection")
    parser.add_argument("--pluto-ip", default="192.168.2.1", help="PlutoSDR IP address (default: %(default)s)")
    args = parser.parse_args()

    # Setup SDR
    sdr = create_pluto(f"ip:{args.pluto_ip}")
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "slow_attack"

    # Setup sync
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(SYNC_CONFIG, sps=SPS, rrc_taps=h_rrc)

    sdr.rx()  # flush


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
                corr = np.abs(np.correlate(filtered[: short_len * 4], np.repeat(short_zc, SPS), mode="valid"))
                max_corr = np.max(corr) if len(corr) > 0 else 0

                # Normalize
                sig_power = np.mean(np.abs(filtered[: short_len * 4]) ** 2)
                ref_power = np.mean(np.abs(np.repeat(short_zc, SPS)) ** 2)
                max_corr / (np.sqrt(sig_power * ref_power) * len(short_zc) * SPS + 1e-12)


                if result.success:
                    pass
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
