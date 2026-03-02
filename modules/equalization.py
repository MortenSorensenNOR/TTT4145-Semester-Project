"""Pilot-aided channel estimation and equalization."""

import numpy as np
from typing import Any # Added for np.ndarray type hints

from modules.pilots import PilotConfig, pilot_indices


def estimate_channel(
    pilot_symbols: np.ndarray,
    pilot_idx: np.ndarray,
    n_total: int,
    pilot_value: np.complex64 = np.complex64(1 + 0j), # Changed type hint and default
) -> np.ndarray[np.complex64, Any]: # Added return type hint
    """Estimate the channel at every symbol position from received pilots.

    Computes H at pilot positions via H = rx/known, then interpolates
    magnitude and (unwrapped) phase to all symbol positions.
    """
    h_at_pilots = pilot_symbols / np.complex64(pilot_value)

    all_pos = np.arange(n_total, dtype=np.int32)
    h_mag = np.interp(all_pos, pilot_idx, np.abs(h_at_pilots)).astype(np.float32)
    h_phase = np.interp(all_pos, pilot_idx, np.unwrap(np.angle(h_at_pilots))).astype(np.float32)
    return h_mag * np.exp(np.complex64(1j) * h_phase)


def equalize_mmse(
    symbols: np.ndarray,
    h_est: np.ndarray,
    noise_variance: np.float32 = np.float32(0.01), # Changed type hint and default
) -> np.ndarray[np.complex64, Any]: # Added return type hint
    """Per-symbol MMSE equalization.

    x_hat[n] = y[n] * conj(h[n]) / (|h[n]|^2 + sigma^2)
    """
    return symbols * np.conj(h_est) / (np.abs(h_est) ** np.float32(2) + noise_variance) # Explicitly cast to np.float32


def equalize_zf(
    symbols: np.ndarray,
    h_est: np.ndarray,
    reg: np.float32 = np.float32(1e-6), # Changed type hint and default
) -> np.ndarray[np.complex64, Any]: # Added return type hint
    """Per-symbol zero-forcing equalization with regularization."""
    return symbols * np.conj(h_est) / (np.abs(h_est) ** np.float32(2) + reg) # Explicitly cast to np.float32


def equalize_payload(
    symbols: np.ndarray,
    n_data: int,
    pilot_config: PilotConfig,
    sigma_sq: np.float32 = np.float32(0.01), # Changed type hint and default
    p_idx: np.ndarray | None = None,
) -> np.ndarray[np.complex64, Any]: # Added return type hint
    """High-level equalization: extract pilots, estimate channel, equalize.

    Returns the full stream (pilots still in place) for subsequent phase tracking.
    If p_idx is provided, reuses pre-computed pilot indices.
    """
    if p_idx is None:
        p_idx = pilot_indices(n_data, pilot_config)
    pilot_rx = symbols[p_idx]

    h_est = estimate_channel(pilot_rx, p_idx, len(symbols), pilot_config.pilot_value)
    return equalize_mmse(symbols, h_est, sigma_sq)