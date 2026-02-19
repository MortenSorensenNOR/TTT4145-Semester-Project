"""Modulation schemes for digital communication."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Modulator(Protocol):
    """Protocol for modulation schemes."""

    bits_per_symbol: int
    symbol_mapping: np.ndarray

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        """Convert bit stream to modulation symbols."""
        ...

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert symbols to hard-decision bits."""
        ...

    def symbols2bits_soft(self, symbols: np.ndarray, sigma_sq: float | None = None) -> np.ndarray:
        """Compute log-likelihood ratios (LLRs) for soft decision decoding."""
        ...

    def estimate_noise_variance(self, symbols: np.ndarray) -> float:
        """Estimate noise variance from received symbols."""
        ...


def estimate_noise_variance(symbols: np.ndarray, constellation: np.ndarray) -> float:
    """Estimate noise variance from received symbols using hard decisions."""
    if len(symbols) == 0:
        return np.finfo(float).eps
    indices = np.argmin(
        np.abs(symbols.reshape(-1, 1) - constellation.reshape(1, -1)),
        axis=1,
    )
    noise = symbols - constellation[indices]
    return float(max(np.mean(np.abs(noise) ** 2), np.finfo(float).eps))


class _ModulatorBase:
    """Shared base providing estimate_noise_variance for all modulators."""

    symbol_mapping: np.ndarray
    bits_per_symbol: int

    def estimate_noise_variance(self, symbols: np.ndarray) -> float:
        """Estimate noise variance from received symbols."""
        return estimate_noise_variance(symbols, self.symbol_mapping)


class BPSK(_ModulatorBase):
    """Binary Phase Shift Keying modulation.

    Source: https://en.wikipedia.org/wiki/Phase-shift_keying#Binary_phase-shift_keying_(BPSK)
    """

    def __init__(self) -> None:
        """Initialize BPSK modulation scheme."""
        self.symbol_mapping = np.array([-1 + 0j, 1 + 0j])
        self.bits_per_symbol = 1
        self.qam_order = 2

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        """Convert bit stream to BPSK symbols."""
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        return self.symbol_mapping[bitstream]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert BPSK symbols to hard-decision bits."""
        if len(symbols) == 0:
            return np.array([], dtype=int).reshape(0, 1)
        return np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1).reshape(-1, 1)

    def symbols2bits_soft(
        self,
        symbols: np.ndarray,
        sigma_sq: float | None = None,
    ) -> np.ndarray:
        """Compute log-likelihood ratios (LLRs) for soft decision decoding.

        LLR convention: positive = more likely 0, negative = more likely 1.

        Source: https://en.wikipedia.org/wiki/Log-likelihood_ratio
        """
        if len(symbols) == 0:
            return np.array([], dtype=float)

        if sigma_sq is None:
            sigma_sq = estimate_noise_variance(symbols, self.symbol_mapping)

        # BPSK: symbol -1 -> bit 0, symbol +1 -> bit 1
        # LLR = -2 * Re(y) / σ²  (positive when Re(y)<0, i.e. bit 0 more likely)
        return (-2.0 * np.real(symbols) / sigma_sq).reshape(-1, 1)


class QPSK(_ModulatorBase):
    """Quadrature Phase Shift Keying modulation (Gray-coded).

    Source: https://en.wikipedia.org/wiki/Phase-shift_keying#Quadrature_phase-shift_keying_(QPSK)
    """

    def __init__(self) -> None:
        """Initialize QPSK modulation scheme."""
        # Gray-coded QPSK: 00 -> -1-1j, 01 -> -1+1j, 10 -> +1-1j, 11 -> +1+1j
        self.symbol_mapping = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)
        self.bit_mapping = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.bits_per_symbol = 2
        self.qam_order = 4

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        """Convert bit stream to QPSK symbols."""
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(-1, 2)
        indices = bitstream[:, 0] * 2 + bitstream[:, 1]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert QPSK symbols to hard-decision bits."""
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1)
        return np.column_stack([indices // 2, indices % 2])

    def symbols2bits_soft(
        self,
        symbols: np.ndarray,
        sigma_sq: float | None = None,
    ) -> np.ndarray:
        """Compute log-likelihood ratios (LLRs) for soft decision decoding."""
        if len(symbols) == 0:
            return np.array([], dtype=float)

        if sigma_sq is None:
            sigma_sq = estimate_noise_variance(symbols, self.symbol_mapping)

        # For this QPSK mapping:
        # bit0=0 when Re<0, bit0=1 when Re>0
        # bit1=0 when Im<0, bit1=1 when Im>0
        # LLR = ln(P(bit=0)/P(bit=1)), so LLR>0 means bit=0
        # LLR(bit0) = -2*√2/σ² * Re(y)
        # LLR(bit1) = -2*√2/σ² * Im(y)
        scale = 2.0 * np.sqrt(2) / sigma_sq
        llr_bit0 = -scale * np.real(symbols)
        llr_bit1 = -scale * np.imag(symbols)

        return np.column_stack([llr_bit0, llr_bit1])


class QAM(_ModulatorBase):
    """Quadrature Amplitude Modulation with Gray coding.

    Source: https://en.wikipedia.org/wiki/Quadrature_amplitude_modulation
    Gray coding: https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
    """

    QPSK_ORDER = 4

    def __init__(self, qam_order: int) -> None:
        """Initialize QAM modulation scheme with Gray-coded bit mapping."""
        bits_per_symbol = int(np.log2(qam_order))

        # Build constellation grid and normalize power
        iq = 2 * np.arange(np.sqrt(qam_order)) - np.sqrt(qam_order) + 1
        q_rep, i_rep = np.meshgrid(iq, iq)
        symbols = i_rep.reshape(qam_order) + 1j * q_rep.reshape(qam_order)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

        # Build Gray code columns using reflected binary construction
        half_grid_size = int(np.sqrt(qam_order) / 2)
        gray_code_column = np.hstack((np.zeros(half_grid_size), np.ones(half_grid_size)))
        for i in range(int(bits_per_symbol / 2 - 1)):
            prev_column = gray_code_column if i == 0 else gray_code_column[-1, :]
            gray_code_column = np.vstack(
                (gray_code_column, np.hstack((prev_column[::2], prev_column[::-2]))),
            )
        gray_code_column = gray_code_column.T
        bit_mapping = np.zeros((qam_order, bits_per_symbol))

        # Assign Gray code bits to I and Q dimensions
        for grid_value in iq:
            in_phase_indices = np.nonzero(i_rep.reshape(qam_order) == grid_value)
            quadrature_indices = np.nonzero(q_rep.reshape(qam_order) == grid_value)

            if qam_order == self.QPSK_ORDER:
                bit_mapping[in_phase_indices, 1] = gray_code_column
                bit_mapping[quadrature_indices, 0] = gray_code_column
            else:
                bit_mapping[in_phase_indices, 1::2] = gray_code_column
                bit_mapping[quadrature_indices, ::2] = gray_code_column

        # Sort by bit-pattern index so symbol_mapping[i] corresponds to bit pattern i
        bit_pattern_to_index = np.sum(
            bit_mapping * 2 ** np.arange(bits_per_symbol - 1, -1, -1),
            axis=1,
            dtype=int,
        )

        self.bit_mapping = bit_mapping[np.argsort(bit_pattern_to_index), :]
        self.symbol_mapping = symbols[np.argsort(bit_pattern_to_index)]
        self.bits_per_symbol = bits_per_symbol
        self.qam_order = qam_order

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        """Convert bit stream to QAM symbols."""
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(
            int(np.size(bitstream) / self.bits_per_symbol),
            self.bits_per_symbol,
        )
        return self.symbol_mapping[
            np.sum(
                bitstream * 2 ** np.arange(self.bits_per_symbol - 1, -1, -1),
                axis=1,
                dtype=int,
            )
        ]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert QAM symbols to hard-decision bits."""
        if len(symbols) == 0:
            return np.array([], dtype=int)
        distance_symbols_to_constellation = np.abs(
            symbols.reshape(np.size(symbols), 1, order="F") - self.symbol_mapping,
        )
        return self.bit_mapping[np.argmin(distance_symbols_to_constellation, axis=1), :]

    def symbols2bits_soft(
        self,
        symbols: np.ndarray,
        sigma_sq: float | None = None,
    ) -> np.ndarray:
        """Compute log-likelihood ratios (LLRs) using max-log-MAP approximation.

        For each bit position, LLR = min distance to constellation point with bit=1
        minus min distance to constellation point with bit=0, scaled by 1/sigma_sq.

        Source: https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation
        """
        if len(symbols) == 0:
            return np.array([], dtype=float)

        if sigma_sq is None:
            sigma_sq = estimate_noise_variance(symbols, self.symbol_mapping)

        distances_sq = (
            np.abs(
                symbols.reshape(-1, 1) - self.symbol_mapping[np.newaxis, :],
            )
            ** 2
        )

        llrs = np.zeros((len(symbols), self.bits_per_symbol))
        for bit_idx in range(self.bits_per_symbol):
            bit_is_zero = self.bit_mapping[:, bit_idx] == 0
            bit_is_one = ~bit_is_zero

            min_dist_zero = np.min(distances_sq[:, bit_is_zero], axis=1)
            min_dist_one = np.min(distances_sq[:, bit_is_one], axis=1)

            # LLR > 0 means bit=0 more likely
            llrs[:, bit_idx] = (min_dist_one - min_dist_zero) / sigma_sq

        return llrs
