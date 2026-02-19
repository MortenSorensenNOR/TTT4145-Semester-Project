"""Shared radio configuration for PlutoSDR transmit and receive."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

# FDD frequency pair for bidirectional bridge mode
FREQ_A_TO_B = 2_400_000_000
FREQ_B_TO_A = 2_450_000_000

SYNC_CONFIG = SynchronizerConfig()
PILOT_CONFIG = PilotConfig()

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
