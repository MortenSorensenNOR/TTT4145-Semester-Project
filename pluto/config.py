"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

import time
import numpy as np

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
# Depth of the libiio kernel-side DMA ring. pyadi's default is 4; bumping it
# gives more headroom for USB scheduling jitter before the Pluto's DMA
# overruns/underruns.
KERNEL_BUFFERS_COUNT = 8
NODE_SRC = 0
NODE_DST = 0
MAX_PACKET_SIZE_BYTES = 1500

# FDD frequency pairs for bidirectional bridge mode. Apps select between them
# via --video (default: network).
VIDEO_FREQ_A_TO_B   = 2_327_000_000
VIDEO_FREQ_B_TO_A   = 2_390_000_000
NETWORK_FREQ_A_TO_B = 2_470_000_000
NETWORK_FREQ_B_TO_A = 2_475_000_000


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

# Split-radio layout: each node dedicates one Pluto to TX and another to RX
# because a single USB-2 Pluto cannot sustain 4 Msps full-duplex. The actual
# per-node TX/RX IP assignment lives in pluto/setup.json (see pluto.setup_config)
# so that swapping cables doesn't require code edits.

PIPELINE = PipelineConfig()

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
    """Apply standard RX settings to an SDR.

    Args:
        gain_mode: AD9361 AGC mode — "slow_attack", "fast_attack", "hybrid",
                   or "manual". slow_attack is fine for dense / continuous
                   traffic but drifts during silence between sparse bursts
                   (gain ramps up while waiting, next packet then clips at the
                   ADC and the constellation widens 3–5×). For sparse-traffic
                   links (e.g. UDP video at low bitrate, control traffic, ARP)
                   use "manual" with a fixed --rx-gain.
        gain:      Fixed RX hardware gain in dB when gain_mode="manual"
                   (AD9361 range ~0–71 dB; ignored in any auto mode).
    """
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
    # buffer_size: int,
    cyclic: bool = False,
    kernel_buffers_count: int = KERNEL_BUFFERS_COUNT,
) -> None:
    """Apply standard TX settings to an SDR."""

    sdr.sample_rate = sample_rate
    sdr.tx_rf_bandwidth = int(sample_rate)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = gain
    sdr.tx_cyclic_buffer = cyclic
    sdr._txdac.set_kernel_buffers_count(kernel_buffers_count)
