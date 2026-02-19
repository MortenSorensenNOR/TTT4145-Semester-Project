"""Utility functions for radio communication simulation."""

import numpy as np
from numpy.typing import NDArray


def ebn0_to_snr(ebn0_db: float, code_rate: float, bits_per_symbol: int = 2) -> float:
    """Convert Eb/N0 (dB) to SNR per symbol (Es/N0) in dB.

    The relationship is:
        Es/N0 = Eb/N0 * code_rate * bits_per_symbol

    In dB:
        SNR (dB) = Eb/N0 (dB) + 10*log10(code_rate * bits_per_symbol)

    Args:
        ebn0_db: Eb/N0 in dB (energy per information bit over noise PSD).
        code_rate: Channel coding rate (e.g., 0.5 for rate 1/2, 0.833 for rate 5/6).
        bits_per_symbol: Number of bits per modulation symbol (2 for QPSK, 4 for 16-QAM).

    Returns:
        SNR per symbol (Es/N0) in dB.

    Example:
        >>> ebn0_to_snr(3.0, code_rate=0.5, bits_per_symbol=2)  # Rate 1/2 QPSK
        3.0  # No change since 0.5 * 2 = 1
        >>> ebn0_to_snr(3.0, code_rate=5/6, bits_per_symbol=2)  # Rate 5/6 QPSK
        5.22  # +2.22 dB adjustment

    """
    return ebn0_db + 10 * np.log10(code_rate * bits_per_symbol)


def snr_to_ebn0(snr_db: float, code_rate: float, bits_per_symbol: int = 2) -> float:
    """Convert SNR per symbol (Es/N0) in dB to Eb/N0 (dB).

    The relationship is:
        Eb/N0 = Es/N0 / (code_rate * bits_per_symbol)

    In dB:
        Eb/N0 (dB) = SNR (dB) - 10*log10(code_rate * bits_per_symbol)

    Args:
        snr_db: SNR per symbol (Es/N0) in dB.
        code_rate: Channel coding rate (e.g., 0.5 for rate 1/2, 0.833 for rate 5/6).
        bits_per_symbol: Number of bits per modulation symbol (2 for QPSK, 4 for 16-QAM).

    Returns:
        Eb/N0 in dB.

    Example:
        >>> snr_to_ebn0(3.0, code_rate=0.5, bits_per_symbol=2)  # Rate 1/2 QPSK
        3.0  # No change since 0.5 * 2 = 1
        >>> snr_to_ebn0(5.22, code_rate=5/6, bits_per_symbol=2)  # Rate 5/6 QPSK
        3.0  # -2.22 dB adjustment

    """
    return snr_db - 10 * np.log10(code_rate * bits_per_symbol)


def calculate_reference_power(reference_signal: NDArray[np.complex128]) -> float:
    """Calculate reference power from a representative signal.

    Use this to determine the transmit power of your signal chain
    (after modulation, upsampling, pulse shaping, etc.) for accurate
    SNR configuration in the channel model.

    Args:
        reference_signal: A representative signal sample, e.g., one or more
            modulated, upsampled, and pulse-shaped symbols. Should not
            contain leading/trailing zeros or silence periods.

    Returns:
        The average power of the reference signal.

    """
    return float(np.mean(np.abs(reference_signal) ** 2))
