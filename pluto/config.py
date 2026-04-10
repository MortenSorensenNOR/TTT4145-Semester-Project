"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

from typing import TYPE_CHECKING

from modules.pipeline import PipelineConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    import adi

DAC_SCALE = 2**14
RX_GAIN = 70.0
DEFAULT_TX_GAIN = -50
RX_BUFFER_SIZE = 2**14  # Smaller buffer = lower latency
TX_BUFFER_SIZE = 2**14
NODE_SRC = 0
NODE_DST = 0
MAX_PACKET_SIZE_BYTES = 1500

# FDD frequency pair for bidirectional bridge mode
FREQ_A_TO_B = 2_400_000_000
FREQ_B_TO_A = 2_450_000_000

PIPELINE = PipelineConfig()

def configure_rx(
    sdr: adi.Pluto,
    *,
    freq: int,
    sample_rate: int,
    buffer_size: int,
    gain_mode: str = "slow_attack",
) -> None:
    """Apply standard RX settings to an SDR."""
    sdr.gain_control_mode_chan0 = gain_mode
    sdr.rx_lo = int(freq)
    sdr.sample_rate = sample_rate
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = buffer_size

def configure_tx(
    sdr: adi.Pluto,
    *,
    freq: int,
    gain: float,
    sample_rate: int,
    buffer_size: int,
    cyclic: bool = False,
) -> None:
    """Apply standard TX settings to an SDR."""
    sdr.sample_rate = sample_rate
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = gain
    sdr.tx_cyclic_buffer = cyclic
    sdr.tx_buffer_size = buffer_size

