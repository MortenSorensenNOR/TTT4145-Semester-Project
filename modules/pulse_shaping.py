"""Pulse shaping filters for signal modulation."""

import numpy as np
from typing import Any # Added for np.ndarray type hints

# The RRC formula has a 0/0 singularity at |t| = 1/(4*alpha).
# This tolerance detects when a sample lands on that point.
RRC_SPECIAL_POINT_TOLERANCE = np.finfo(np.float32).eps # Changed to np.float32


def rrc_filter(sps: int, alpha: np.float32, num_taps: int = 101) -> np.ndarray[np.float32, Any]: # Changed type hint and return type hint
    """Root raised cosine filter.

    Source is https://en.wikipedia.org/wiki/Root-raised-cosine_filter
    """
    t = (np.arange(num_taps).astype(np.float32) - np.float32(num_taps - 1) / np.float32(2)) / np.float32(sps) # Explicitly cast to np.float32

    zero_mask = t == np.float32(0) # Explicitly cast to np.float32
    if alpha > np.float32(0): # Explicitly cast to np.float32
        special_val = np.float32(1) / (np.float32(4) * alpha) # Explicitly cast to np.float32
        special_mask = np.abs(np.abs(t) - special_val) < RRC_SPECIAL_POINT_TOLERANCE
        special_case = (
            alpha
            / np.sqrt(np.float32(2)) # Explicitly cast to np.float32
            * ((np.float32(1) + np.float32(2) / np.float32(np.pi)) * np.sin(np.float32(np.pi) / (np.float32(4) * alpha)) + (np.float32(1) - np.float32(2) / np.float32(np.pi)) * np.cos(np.float32(np.pi) / (np.float32(4) * alpha))) # Explicitly cast to np.float32
        )
    else:
        special_mask = np.zeros_like(t, dtype=bool)
        special_case = np.float32(0.0) # Explicitly cast to np.float32
    general_mask = ~zero_mask & ~special_mask

    num = np.sin(np.float32(np.pi) * t * (np.float32(1) - alpha)) + np.float32(4) * alpha * t * np.cos(np.float32(np.pi) * t * (np.float32(1) + alpha)) # Explicitly cast to np.float32
    den = np.float32(np.pi) * t * (np.float32(1) - (np.float32(4) * alpha * t) ** np.float32(2)) # Explicitly cast to np.float32
    with np.errstate(divide="ignore", invalid="ignore"):
        general_vals = num / den

    h = np.select(
        [zero_mask, special_mask, general_mask],
        [
            np.float32(1) + alpha * (np.float32(4) / np.float32(np.pi) - np.float32(1)), # Explicitly cast to np.float32
            special_case,
            general_vals,
        ],
    )

    return (h / np.sqrt(np.sum(h**np.float32(2)))).astype(np.float32)  # normalize energy # Explicitly cast to np.float32


def upsample_and_filter(symbols: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray[np.complex64, Any]: # Added return type hint
    """Upsample a symbol-rate signal and apply an RRC filter."""
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, rrc_taps, mode="same")