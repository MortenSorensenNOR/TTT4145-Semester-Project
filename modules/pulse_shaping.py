"""Pulse shaping filters for signal modulation."""

import numpy as np

# Tolerance for detecting the special t = 1/(4*alpha) point in the RRC formula
RRC_SPECIAL_POINT_TOLERANCE = 1e-8


def rrc_filter(sps: int, alpha: float, num_taps: int = 101) -> np.ndarray:
    """Root raised cosine filter."""
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps

    zero_mask = t == 0
    special_val = 1 / (4 * alpha + 1e-10)
    special_mask = np.abs(np.abs(t) - special_val) < RRC_SPECIAL_POINT_TOLERANCE
    general_mask = ~zero_mask & ~special_mask

    # General case (safe for all t due to +1e-10 in denominator)
    num = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
    den = np.pi * t * (1 - (4 * alpha * t) ** 2)
    general_vals = num / (den + 1e-10)

    h = np.select(
        [zero_mask, special_mask, general_mask],
        [
            1 + alpha * (4 / np.pi - 1),
            alpha
            / np.sqrt(2)
            * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))),
            general_vals,
        ],
    )

    return h / np.sqrt(np.sum(h**2))  # normalize energy


def upsample_and_filter(symbols: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Upsample a symbol-rate signal and apply an RRC filter."""
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, rrc_taps, mode="same")
