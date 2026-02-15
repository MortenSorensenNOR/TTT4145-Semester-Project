"""Utility functions for radio communication simulation."""

import numpy as np
from numpy.typing import NDArray


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
