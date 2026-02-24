"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modules.channel_coding import CodeRates
from modules.costas_loop import CostasConfig
from modules.frame_constructor import ModulationSchemes
from modules.modulation import BPSK, QAM, QPSK, Modulator
from modules.pilots import PilotConfig
from modules.synchronization import SynchronizerConfig

if TYPE_CHECKING:
    from collections.abc import Callable

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
    costas_loop: bool = False  # Not yet implemented correctly! Cannot be used!!!
    channel_coding: bool = True
    interleaving: bool = True
    cfo_correction: bool = True
    pilot_config: PilotConfig | None = None

    def __post_init__(self) -> None:
        """Validate pipeline configuration. Making sure costas and pilots are not running at the same time."""
        if self.pilots and self.costas_loop:
            msg = "Pilots and Costas loop are mutually exclusive"
            raise ValueError(msg)
        if self.pilot_config is None:
            self.pilot_config = PilotConfig()


PIPELINE = PipelineConfig()

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
