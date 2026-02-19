"""Shared radio configuration for PlutoSDR transmit and receive."""

from modules.frame_constructor import ModulationSchemes
from modules.modulation import BPSK, QAM, QPSK, Modulator

SAMPLE_RATE = 1_000_000
CENTER_FREQ = 2_400_000_000
SPS = 4
RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
DAC_SCALE = 2**14


def get_modulator(scheme: ModulationSchemes) -> Modulator:
    """Return a modulator instance for the given scheme."""
    modulators = {
        ModulationSchemes.BPSK: BPSK,
        ModulationSchemes.QPSK: QPSK,
        ModulationSchemes.QAM16: lambda: QAM(16),
        ModulationSchemes.QAM64: lambda: QAM(64),
    }
    if scheme not in modulators:
        msg = f"Unknown modulation scheme: {scheme}"
        raise ValueError(msg)
    return modulators[scheme]()
