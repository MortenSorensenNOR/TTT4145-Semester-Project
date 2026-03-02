"""Simple PlutoSDR loopback test utilities.

Provides basic TX/RX helpers for loopback testing with a single frequency.
"""

import time
from typing import Any, cast

import adi
import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, DAC_SCALE, SAMPLE_RATE

TX_GAIN = np.float32(-30) # Changed to np.float32
RX_GAIN = np.float32(40) # Changed to np.float32
RX_BUFFER_SIZE = 2**16


def setup_pluto(
    freq_hz: np.float32 = CENTER_FREQ, # Changed type hint
    sample_rate: np.float32 = SAMPLE_RATE, # Changed type hint
    tx_gain: np.float32 = TX_GAIN, # Changed type hint
    rx_gain: np.float32 = RX_GAIN, # Changed type hint
    uri: str = "ip:192.168.2.1",
) -> adi.Pluto:
    """Configure PlutoSDR for loopback testing with a single LO frequency."""
    sdr = create_pluto(uri)
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
    tx_signal: np.ndarray[np.complex64, Any], # Added type hint
    rx_delay_ms: np.float32 = np.float32(50), # Changed type hint and default
    n_captures: int = 3,
) -> np.ndarray[np.complex64, Any]: # Added return type hint
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

    time.sleep(rx_delay_ms / np.float32(1000)) # Explicitly cast to np.float32
    sdr.rx()  # flush stale buffer

    captures = [sdr.rx().astype(np.complex64) for _ in range(n_captures)] # Ensure np.complex64

    sdr.tx_destroy_buffer()

    return np.concatenate(cast("list[np.ndarray[np.complex64, Any]]", captures)) # Ensure np.complex64