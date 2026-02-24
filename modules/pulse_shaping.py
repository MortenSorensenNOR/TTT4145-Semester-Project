"""Pulse shaping filters for signal modulation."""

import numpy as np

# The RRC formula has a 0/0 singularity at |t| = 1/(4*alpha).
# This tolerance detects when a sample lands on that point.
RRC_SPECIAL_POINT_TOLERANCE = 8 * np.finfo(float).eps


def rrc_filter(sps: int, alpha: float, num_taps: int = 101) -> np.ndarray:
    """Root raised cosine filter.

    Source is https://en.wikipedia.org/wiki/Root-raised-cosine_filter
    """
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps

    zero_mask = t == 0
    if alpha > 0:
        special_val = 1 / (4 * alpha)
        special_mask = np.abs(np.abs(t) - special_val) < RRC_SPECIAL_POINT_TOLERANCE
        special_case = (
            alpha
            / np.sqrt(2)
            * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        )
    else:
        special_mask = np.zeros_like(t, dtype=bool)
        special_case = 0.0
    general_mask = ~zero_mask & ~special_mask

    num = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
    den = np.pi * t * (1 - (4 * alpha * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        general_vals = num / den

    h = np.select(
        [zero_mask, special_mask, general_mask],
        [
            1 + alpha * (4 / np.pi - 1),
            special_case,
            general_vals,
        ],
    )

    return h / np.sqrt(np.sum(h**2))  # normalize energy


def upsample_and_filter(symbols: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Upsample a symbol-rate signal and apply an RRC filter."""
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, rrc_taps, mode="same")
