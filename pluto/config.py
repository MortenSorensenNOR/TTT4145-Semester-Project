"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from modules.channel_coding import CodeRates
from modules.costas_loop import CostasConfig
from modules.frame_constructor import ModulationSchemes
from modules.modulation import BPSK, QAM, QPSK, EightPSK, Modulator
from modules.pilots import PilotConfig
from modules.synchronization import SynchronizerConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    import adi

SAMPLE_RATE = np.float32(5_336_000) # Changed to np.float32
CENTER_FREQ = np.float32(2_400_000_000) # Changed to np.float32
SPS = 8
SPAN = 8
RRC_ALPHA = np.float32(0.35)
RRC_NUM_TAPS = 2 * SPS * SPAN + 1
DAC_SCALE = np.float32(2**14)
RX_GAIN = np.float32(70.0)
MOD_SCHEME = ModulationSchemes.QPSK
CODING_RATE = CodeRates.THREE_QUARTER_RATE  # Higher rate = more throughput (needs good SNR)
DEFAULT_TX_GAIN = np.float32(-50.0)
RX_BUFFER_SIZE = 2**20  # Smaller buffer = lower latency
NODE_SRC = 0
NODE_DST = 0

# FDD frequency pair for bidirectional bridge mode
FREQ_A_TO_B = np.float32(2_400_000_000) # Changed to np.float32
FREQ_B_TO_A = np.float32(2_450_000_000) # Changed to np.float32

SYNC_CONFIG = SynchronizerConfig()
PILOT_CONFIG = PilotConfig()
COSTAS_CONFIG = CostasConfig()


@dataclass
class PipelineConfig:
    """Toggle individual pipeline stages on/off for testing."""

    pulse_shaping: bool = True
    pilots: bool = True
    costas_loop: bool = False
    channel_coding: bool = True
    interleaving: bool = True
    cfo_correction: bool = True
    pilot_config: PilotConfig = field(default_factory=PilotConfig)

    def __post_init__(self) -> None:
        """Validate pipeline configuration. Making sure costas and pilots are not running at the same time."""
        if self.pilots and self.costas_loop:
            msg = "Pilots and Costas loop are mutually exclusive"
            raise ValueError(msg)


PIPELINE = PipelineConfig()


def configure_rx(
    sdr: adi.Pluto,
    *,
    freq: np.float32 = CENTER_FREQ, # Changed type hint
    gain_mode: str = "slow_attack",
) -> None:
    """Apply standard RX settings to an SDR."""
    sdr.gain_control_mode_chan0 = gain_mode
    sdr.rx_lo = int(freq)
    sdr.sample_rate = int(SAMPLE_RATE) # Explicitly cast to int
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE


def configure_tx(
    sdr: adi.Pluto,
    *,
    freq: np.float32 = CENTER_FREQ, # Changed type hint
    gain: np.float32 = DEFAULT_TX_GAIN, # Changed type hint
    cyclic: bool = False,
) -> None:
    """Apply standard TX settings to an SDR."""
    sdr.sample_rate = int(SAMPLE_RATE) # Explicitly cast to int
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = gain
    sdr.tx_cyclic_buffer = cyclic


_modulator_cache: dict[ModulationSchemes, Callable[[], Modulator]] = {} # type: ignore


def get_modulator(scheme: ModulationSchemes) -> Modulator:
    """Return a (cached) modulator instance for the given scheme."""
    if scheme not in _modulator_cache:
        factories: dict[ModulationSchemes, Callable[[], Modulator]] = {
            ModulationSchemes.BPSK: BPSK,
            ModulationSchemes.QPSK: QPSK,
            ModulationSchemes.QAM16: lambda: QAM(16),
            ModulationSchemes.PSK8: EightPSK,
        }
        if scheme not in factories:
            msg = f"Unknown modulation scheme: {scheme}"
            raise ValueError(msg)
        _modulator_cache[scheme] = factories[scheme]()
    return _modulator_cache[scheme]