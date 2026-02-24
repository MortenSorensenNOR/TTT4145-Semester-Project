"""Pilot symbol insertion, extraction, and pilot-aided phase tracking."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PilotConfig:
    """Configuration for pilot symbol insertion."""

    spacing: int = 16
    pilot_value: complex = 1 + 0j


def n_pilots(n_data: int, config: PilotConfig) -> int:
    """Return the number of pilot symbols inserted for n_data data symbols."""
    if n_data <= 0:
        return 0
    # One pilot every `spacing` data symbols (before each block)
    return (n_data - 1) // config.spacing + 1


def n_total_symbols(n_data: int, config: PilotConfig) -> int:
    """Total symbols (data + pilots) after pilot insertion."""
    return n_data + n_pilots(n_data, config)


def insert_pilots(data_symbols: np.ndarray, config: PilotConfig) -> np.ndarray:
    """Insert known pilot symbols into a data symbol stream.

    A pilot is placed before every block of `spacing` data symbols:
        [P, D, D, ..., D, P, D, D, ..., D, P, ...]
    """
    n_data = len(data_symbols)
    n_total = n_total_symbols(n_data, config)
    out = np.empty(n_total, dtype=complex)

    src = 0  # index into data_symbols
    dst = 0  # index into out
    while src < n_data:
        out[dst] = config.pilot_value
        dst += 1
        block_end = min(src + config.spacing, n_data)
        block_len = block_end - src
        out[dst : dst + block_len] = data_symbols[src:block_end]
        src = block_end
        dst += block_len

    return out


def pilot_indices(n_data: int, config: PilotConfig) -> np.ndarray:
    """Return the indices of pilot symbols within the combined stream."""
    indices = []
    src = 0
    dst = 0
    while src < n_data:
        indices.append(dst)
        dst += 1
        block_end = min(src + config.spacing, n_data)
        dst += block_end - src
        src = block_end
    return np.array(indices, dtype=int)


def data_indices(n_data: int, config: PilotConfig) -> np.ndarray:
    """Return the indices of data symbols within the combined stream."""
    n_total = n_total_symbols(n_data, config)
    p_idx = set(pilot_indices(n_data, config).tolist())
    return np.array([i for i in range(n_total) if i not in p_idx], dtype=int)


def pilot_aided_phase_track(
    symbols: np.ndarray,
    n_data: int,
    config: PilotConfig,
    p_idx: np.ndarray | None = None,
    d_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate and correct phase drift using embedded pilot symbols.

    1. Extract pilots, compute phase error at each pilot position.
    2. Unwrap and interpolate to all symbol positions.
    3. Correct phase, then return data-only symbols.

    If p_idx/d_idx are provided, reuses pre-computed indices.
    """
    if p_idx is None:
        p_idx = pilot_indices(n_data, config)
    if d_idx is None:
        d_idx = data_indices(n_data, config)

    pilot_rx = symbols[p_idx]
    phase_at_pilots = np.angle(pilot_rx * np.conj(config.pilot_value))
    phase_at_pilots = np.unwrap(phase_at_pilots)

    # Interpolate to every symbol position
    all_positions = np.arange(len(symbols))
    phase_all = np.interp(all_positions, p_idx, phase_at_pilots)

    corrected = symbols * np.exp(-1j * phase_all)
    return corrected[d_idx]
