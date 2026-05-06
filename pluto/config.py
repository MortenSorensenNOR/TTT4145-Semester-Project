"""Shared radio configuration for PlutoSDR transmit and receive."""

from typing import TYPE_CHECKING
from modules.pipeline import PipelineConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    import adi

DAC_SCALE = 2**14
RX_GAIN = 70.0
DEFAULT_TX_GAIN = -50
RX_BUFFER_SIZE = 2**14
TX_BUFFER_SIZE = 2**14
KERNEL_BUFFERS_COUNT = 8

PIPELINE = PipelineConfig()

# FDD frequency pairs for bidirectional bridge mode. Apps select between them
# via --video (default: network).
NETWORK_FREQ_A_TO_B = 2_470_000_000
NETWORK_FREQ_B_TO_A = 2_475_000_000
VIDEO_FREQ_A_TO_B   = 2_327_000_000
VIDEO_FREQ_B_TO_A   = 2_390_000_000


def get_node_freqs(node: str, *, video: bool = False) -> dict[str, int]:
    """Return {'tx': ..., 'rx': ...} for `node` in the chosen mode."""
    if video:
        a_to_b, b_to_a = VIDEO_FREQ_A_TO_B, VIDEO_FREQ_B_TO_A
    else:
        a_to_b, b_to_a = NETWORK_FREQ_A_TO_B, NETWORK_FREQ_B_TO_A
    if node == "A":
        return {"tx": a_to_b, "rx": b_to_a}
    if node == "B":
        return {"tx": b_to_a, "rx": a_to_b}
    raise ValueError(f"node must be 'A' or 'B', got {node!r}")


def configure_rx(
    sdr: adi.Pluto,
    *,
    freq: int,
    sample_rate: int = PIPELINE.SAMPLE_RATE,
    buffer_size: int = RX_BUFFER_SIZE,
    gain_mode: str = "slow_attack",
    gain: float = 40.0,
    kernel_buffers_count: int = KERNEL_BUFFERS_COUNT,
) -> None:
    sdr.gain_control_mode_chan0 = gain_mode
    if gain_mode == "manual":
        sdr.rx_hardwaregain_chan0 = float(gain)
    sdr.rx_lo = int(freq)
    sdr.sample_rate = sample_rate
    sdr.rx_rf_bandwidth = int(sample_rate)
    sdr.rx_buffer_size = buffer_size
    sdr._rxadc.set_kernel_buffers_count(kernel_buffers_count)

def configure_tx(
    sdr: adi.Pluto,
    *,
    freq: int,
    gain: float,
    sample_rate: int = PIPELINE.SAMPLE_RATE,
    cyclic: bool = False,
    kernel_buffers_count: int = KERNEL_BUFFERS_COUNT,
) -> None:
    sdr.sample_rate = sample_rate
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = gain
    sdr.tx_cyclic_buffer = cyclic
    sdr._txdac.set_kernel_buffers_count(kernel_buffers_count)
