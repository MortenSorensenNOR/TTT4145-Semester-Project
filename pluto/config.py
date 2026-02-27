"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from modules.channel_coding import CodeRates
from modules.costas_loop import CostasConfig
from modules.frame_constructor import ModulationSchemes
from modules.modulation import BPSK, QAM, QPSK, Modulator
from modules.pilots import PilotConfig
from modules.synchronization import SynchronizerConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    import adi

SAMPLE_RATE = 1_000_000
CENTER_FREQ = 2_400_000_000
SPS = 4
RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
DAC_SCALE = 2**14
RX_GAIN = 70.0
MOD_SCHEME = ModulationSchemes.QPSK
CODING_RATE = CodeRates.THREE_QUARTER_RATE  # Higher rate = more throughput (needs good SNR)
DEFAULT_TX_GAIN = -10
RX_BUFFER_SIZE = 2**14  # Smaller buffer = lower latency
NODE_SRC = 0
NODE_DST = 0

# FDD frequency pair for bidirectional bridge mode
FREQ_A_TO_B = 2_400_000_000
FREQ_B_TO_A = 2_450_000_000

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
    freq: int = CENTER_FREQ,
    gain_mode: str = "slow_attack",
) -> None:
    """Apply standard RX settings to an SDR."""
    sdr.gain_control_mode_chan0 = gain_mode
    sdr.rx_lo = int(freq)
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE


def configure_tx(
    sdr: adi.Pluto,
    *,
    freq: int = CENTER_FREQ,
    gain: float = DEFAULT_TX_GAIN,
) -> None:
    """Apply standard TX settings to an SDR."""
    sdr.sample_rate = SAMPLE_RATE
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(freq)
    sdr.tx_hardwaregain_chan0 = gain


_modulator_cache: dict[ModulationSchemes, Modulator] = {}


def get_modulator(scheme: ModulationSchemes) -> Modulator:
    """Return a (cached) modulator instance for the given scheme."""
    if scheme not in _modulator_cache:
        factories: dict[ModulationSchemes, Callable[[], Modulator]] = {
            ModulationSchemes.BPSK: BPSK,
            ModulationSchemes.QPSK: QPSK,
            ModulationSchemes.QAM16: lambda: QAM(16),
            ModulationSchemes.QAM64: lambda: QAM(64),
        }
        if scheme not in factories:
            msg = f"Unknown modulation scheme: {scheme}"
            raise ValueError(msg)
        _modulator_cache[scheme] = factories[scheme]()
    return _modulator_cache[scheme]
