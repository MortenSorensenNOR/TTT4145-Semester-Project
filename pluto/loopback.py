"""Simple PlutoSDR loopback test utilities.

Provides basic TX/RX helpers for loopback testing with a single frequency.
"""

import time
from typing import cast

import adi
import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, DAC_SCALE, SAMPLE_RATE

TX_GAIN = -30
RX_GAIN = 40
RX_BUFFER_SIZE = 2**16


def setup_pluto(
    freq_hz: int = CENTER_FREQ,
    sample_rate: int = SAMPLE_RATE,
    tx_gain: float = TX_GAIN,
    rx_gain: float = RX_GAIN,
) -> adi.Pluto:
    """Configure PlutoSDR for loopback testing with a single LO frequency."""
    sdr = create_pluto()
    sdr.sample_rate = int(sample_rate)
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(freq_hz)
    sdr.rx_lo = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = tx_gain
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    return sdr


def transmit_and_receive(
    sdr: adi.Pluto,
    tx_signal: np.ndarray,
    rx_delay_ms: float = 50,
    n_captures: int = 3,
) -> np.ndarray:
    """Transmit a signal in cyclic mode and capture RX samples.

    1. Start cyclic TX
    2. Wait for settling
    3. Capture n_captures RX buffers
    4. Stop TX
    5. Return concatenated RX samples
    """
    samples = (tx_signal * DAC_SCALE).astype(np.complex64)

    sdr.tx_cyclic_buffer = True
    sdr.tx(samples)

    time.sleep(rx_delay_ms / 1000)
    sdr.rx()  # flush stale buffer

    captures = [sdr.rx() for _ in range(n_captures)]

    sdr.tx_destroy_buffer()

    return np.concatenate(cast("list[np.ndarray]", captures))
