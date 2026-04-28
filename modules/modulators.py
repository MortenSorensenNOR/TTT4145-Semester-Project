from dataclasses import dataclass
from typing import runtime_checkable, Protocol
import numpy as np
from numpy.typing import NDArray

try:
    from modules import modulators_ext as _mod_ext
except ImportError:  # pragma: no cover - extension may not be built yet
    _mod_ext = None

EMPTY_COMPLEX = np.empty(0, dtype=np.complex64)
EMPTY_INT = np.empty(0, dtype=np.uint8)


def _build_psk_llr_tables(symbol_mapping: np.ndarray, bits_per_symbol: int) -> tuple:
    """Pack constellation + per-bit subset indices for the C++ LLR kernel.

    Returns (sym_re, sym_im, c0_idx, c1_idx) — c*_idx are int32 (nbits, M/2).
    Caller is expected to keep these alive for the lifetime of the modulator.
    """
    M = symbol_mapping.size
    sym_re = np.ascontiguousarray(symbol_mapping.real, dtype=np.float32)
    sym_im = np.ascontiguousarray(symbol_mapping.imag, dtype=np.float32)
    idx = np.arange(M)
    bits_for_idx = np.stack(
        [(idx >> (bits_per_symbol - 1 - b)) & 1 for b in range(bits_per_symbol)],
        axis=1,
    )
    per_bit = M // 2
    c0_idx = np.empty((bits_per_symbol, per_bit), dtype=np.int32)
    c1_idx = np.empty((bits_per_symbol, per_bit), dtype=np.int32)
    for b in range(bits_per_symbol):
        c0_idx[b] = np.flatnonzero(bits_for_idx[:, b] == 0)
        c1_idx[b] = np.flatnonzero(bits_for_idx[:, b] == 1)
    return sym_re, sym_im, c0_idx, c1_idx

@runtime_checkable
class Modulator(Protocol):
    bits_per_symbol: int
    qam_order: int
    symbol_mapping: np.ndarray

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        """Convert bit stream to modulation symbols."""
        ...

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert symbols to hard-decision bits."""
        ...

    def symbols2llrs(self, symbols: np.ndarray) -> np.ndarray:
        """Convert symbols to soft LLRs (positive ⇒ bit 0). Shape: (N, bits_per_symbol)."""
        ...

class BPSK(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 1
        self.qam_order = 2
        self.symbol_mapping = np.array([-1 + 0j, 1 + 0j], dtype=np.complex64)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if bitstream.size == 0:
            return EMPTY_COMPLEX
        return self.symbol_mapping[bitstream.reshape(-1)]

    def symbols2bits(self, symbols: NDArray[np.complex64]) -> np.ndarray:
        if symbols.size == 0:
            return EMPTY_INT
        # BPSK: decision boundary is the imaginary axis
        bits = np.empty((symbols.size, 1), dtype=np.uint8)
        bits[:, 0] = symbols.real > 0
        return bits

    def symbols2llrs(self, symbols: NDArray[np.complex64]) -> np.ndarray:
        # bit=0 → real=-1, bit=1 → real=+1, so LLR(b=0) ∝ -real.
        if symbols.size == 0:
            return np.empty((0, 1), dtype=np.float32)
        return (-2.0 * symbols.real).astype(np.float32).reshape(-1, 1)

class QPSK(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 2
        self.qam_order = 4
        self.symbol_mapping = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=np.complex64) / np.sqrt(2)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if bitstream.size == 0:
            return EMPTY_COMPLEX
        bitstream = bitstream.reshape(-1, 2)
        indices = (bitstream[:, 0] << 1) | bitstream[:, 1]
        #indices = bitstream[:, 0] * 2 + bitstream[:, 1]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: NDArray[np.complex64]) -> np.ndarray:
        if symbols.size == 0:
            return np.array([], dtype=int)
        # QPSK mapping: [-1-j, -1+j, 1-j, 1+j]/√2 → bit0=sign(re), bit1=sign(im)
        bits = np.empty((symbols.size, 2), dtype=np.uint8)
        bits[:, 0] = symbols.real > 0
        bits[:, 1] = symbols.imag > 0
        return bits

    def symbols2llrs(self, symbols: NDArray[np.complex64]) -> np.ndarray:
        # Gray-coded: bit0 ↔ sign(real), bit1 ↔ sign(imag).
        if symbols.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        out = np.empty((symbols.size, 2), dtype=np.float32)
        out[:, 0] = -2.0 * symbols.real
        out[:, 1] = -2.0 * symbols.imag
        return out

class PSK8(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 3
        self.qam_order = 8
        self.symbol_mapping = np.array([-1 - 1j, -np.sqrt(2) + 0j, 0 + np.sqrt(2)*1j, -1 + 1j,
                                         0 - np.sqrt(2)*1j, 1 - 1j, 1 + 1j, np.sqrt(2)+ 0j], dtype=np.complex64) / np.sqrt(2)
        self._BIN_TO_IDX = np.array([7, 6, 2, 3, 1, 0, 4, 5], dtype=np.uint8)

        # Precomputed real/imag of constellation points + per-bit subset indices
        # for max-log LLR.  Operating on real+imag separately avoids np.abs's sqrt
        # (which **2 then undoes) and lets us compute distances to all 8 points
        # once per symbol instead of 8 times.
        idx = np.arange(8)
        bits_for_idx = np.stack([(idx >> 2) & 1, (idx >> 1) & 1, idx & 1], axis=1)
        self._sym_re = self.symbol_mapping.real.astype(np.float32).copy()
        self._sym_im = self.symbol_mapping.imag.astype(np.float32).copy()
        self._c0_idx = [np.flatnonzero(bits_for_idx[:, b] == 0).astype(np.intp)
                        for b in range(3)]
        self._c1_idx = [np.flatnonzero(bits_for_idx[:, b] == 1).astype(np.intp)
                        for b in range(3)]
        # C++ kernel tables (separate dtype/shape — held alongside the numpy
        # ones so the pure-Python fallback still works if the extension is
        # missing).
        self._llr_tables = _build_psk_llr_tables(self.symbol_mapping, 3)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if bitstream.size == 0:
            return EMPTY_COMPLEX
        bitstream = bitstream.reshape(-1, 3)
        indices = (bitstream[:, 0] << 2) | (bitstream[:, 1] << 1) | bitstream[:, 2]
        #indices = bitstream[:, 0] * 4 + bitstream[:, 1] * 2 + bitstream[:, 2]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if symbols.size == 0:
            return EMPTY_INT
        # Quantize angle to nearest π/4 bin (0..7), then map to constellation index.
        # Constellation phases: idx7=0, idx6=π/4, idx2=π/2, idx3=3π/4,
        #                       idx1=π, idx0=-3π/4, idx4=-π/2, idx5=-π/4

        #bins = np.round(fast_angle(symbols.real, symbols.imag) / (np.pi / 4)).astype(int) & 7
        bins = np.round(np.angle(symbols) / (np.pi / 4)).astype(int) % 8
        indices = self._BIN_TO_IDX[bins]
        bits = np.empty((symbols.size, 3), dtype=np.uint8)
        bits[:, 0] = indices >> 2
        bits[:, 1] = (indices >> 1) & 1
        bits[:, 2] = indices & 1
        return bits

    def symbols2llrs(self, symbols: np.ndarray) -> np.ndarray:
        # max-log MAP: LLR(b) = min_{s∈C1} |r-s|² − min_{s∈C0} |r-s|²
        # (positive ⇒ bit 0 closer ⇒ bit=0 more likely).
        if symbols.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if _mod_ext is not None:
            sr, si, c0, c1 = self._llr_tables
            return _mod_ext.psk_llr_unit_norm(
                np.ascontiguousarray(symbols, dtype=np.complex64),
                sr, si, c0, c1, 3,
            )
        # Numpy fallback: compute squared distance to all 8 constellation
        # points once and take min over the per-bit subsets.
        r = np.ascontiguousarray(symbols, dtype=np.complex64)
        rr = r.real[:, np.newaxis]
        ri = r.imag[:, np.newaxis]
        dr = rr - self._sym_re
        di = ri - self._sym_im
        d2 = dr * dr + di * di
        out = np.empty((r.size, 3), dtype=np.float32)
        for b in range(3):
            out[:, b] = d2[:, self._c1_idx[b]].min(axis=1) - d2[:, self._c0_idx[b]].min(axis=1)
        return out


class PSK16(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 4
        self.qam_order = 16

        # Gray-coded 16-PSK: ring position k ∈ 0..15 sits at angle k·π/8 and
        # carries the bit pattern gray(k) = k ^ (k>>1).  symbol_mapping is
        # indexed by bit-pattern, so we invert: position(bits) = inv_gray(bits).
        bits_idx = np.arange(16, dtype=np.uint16)
        # 4-bit inverse Gray:  b = g ^ (g>>1) ^ (g>>2) ^ (g>>3)
        positions = bits_idx ^ (bits_idx >> 1) ^ (bits_idx >> 2) ^ (bits_idx >> 3)
        angles = positions.astype(np.float64) * (np.pi / 8.0)
        self.symbol_mapping = np.exp(1j * angles).astype(np.complex64)

        # angular bin k (0..15) → bit pattern (Gray code of k).  Used by
        # symbols2bits to map the nearest constellation slot back to bits.
        ring = np.arange(16, dtype=np.uint8)
        self._BIN_TO_IDX = (ring ^ (ring >> 1)).astype(np.uint8)

        # Precomputed real/imag arrays + per-bit subset indices for max-log LLR.
        # See PSK8 for the rationale (avoid sqrt, compute all-point distances once).
        idx = np.arange(16)
        bits_for_idx = np.stack([(idx >> 3) & 1, (idx >> 2) & 1,
                                 (idx >> 1) & 1, idx & 1], axis=1)
        self._sym_re = self.symbol_mapping.real.astype(np.float32).copy()
        self._sym_im = self.symbol_mapping.imag.astype(np.float32).copy()
        self._c0_idx = [np.flatnonzero(bits_for_idx[:, b] == 0).astype(np.intp)
                        for b in range(4)]
        self._c1_idx = [np.flatnonzero(bits_for_idx[:, b] == 1).astype(np.intp)
                        for b in range(4)]
        self._llr_tables = _build_psk_llr_tables(self.symbol_mapping, 4)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if bitstream.size == 0:
            return EMPTY_COMPLEX
        bitstream = bitstream.reshape(-1, 4)
        indices = ((bitstream[:, 0] << 3) | (bitstream[:, 1] << 2)
                   | (bitstream[:, 2] << 1) | bitstream[:, 3])
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if symbols.size == 0:
            return EMPTY_INT
        # 16 angular slots of π/8 each, centred on the constellation points.
        bins = np.round(np.angle(symbols) / (np.pi / 8)).astype(int) % 16
        indices = self._BIN_TO_IDX[bins]
        bits = np.empty((symbols.size, 4), dtype=np.uint8)
        bits[:, 0] = (indices >> 3) & 1
        bits[:, 1] = (indices >> 2) & 1
        bits[:, 2] = (indices >> 1) & 1
        bits[:, 3] = indices & 1
        return bits

    def symbols2llrs(self, symbols: np.ndarray) -> np.ndarray:
        # max-log MAP: LLR(b) = min_{s∈C1} |r-s|² − min_{s∈C0} |r-s|²
        if symbols.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        if _mod_ext is not None:
            sr, si, c0, c1 = self._llr_tables
            return _mod_ext.psk_llr_unit_norm(
                np.ascontiguousarray(symbols, dtype=np.complex64),
                sr, si, c0, c1, 4,
            )
        r = np.ascontiguousarray(symbols, dtype=np.complex64)
        rr = r.real[:, np.newaxis]
        ri = r.imag[:, np.newaxis]
        dr = rr - self._sym_re
        di = ri - self._sym_im
        d2 = dr * dr + di * di
        out = np.empty((r.size, 4), dtype=np.float32)
        for b in range(4):
            out[:, b] = d2[:, self._c1_idx[b]].min(axis=1) - d2[:, self._c0_idx[b]].min(axis=1)
        return out
"""

LUT_SIZE = 128  # 128x128 grid for moderate accuracy
LUT_MAX = LUT_SIZE - 1
LUT_SCALE = (LUT_SIZE - 1) / 3.0  # maps [-1.5,1.5] to [0,LUT_SIZE-1]

class PSK8(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 3
        self.qam_order = 8
        self.symbol_mapping = np.array([-1 - 1j, -np.sqrt(2) + 0j, 0 + np.sqrt(2)*1j, -1 + 1j,
                                        0 - np.sqrt(2)*1j, 1 - 1j, 1 + 1j, np.sqrt(2)+ 0j], dtype=np.complex64)/np.sqrt(2)
        self._BIN_TO_IDX = np.array([7, 6, 2, 3, 1, 0, 4, 5], dtype=np.int8)
        self._lut_size = 128
        self._lut_max = self._lut_size - 1
        self._scale = (self._lut_size - 1) / 3.0
        self._lut = self._build_lut()

    def _build_lut(self):
        levels = np.linspace(-1.5, 1.5, self._lut_size, dtype=np.float32)
        lut = np.empty((self._lut_size, self._lut_size), dtype=np.int8)

        for i, re in enumerate(levels):
            for j, im in enumerate(levels):
                angle = np.arctan2(im, re)   # happens ONCE at init → OK
                bin_ = int(np.round(angle / (np.pi / 4))) & 7
                lut[i, j] = bin_

        return lut

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if bitstream.size == 0:
            return EMPTY_COMPLEX
        bitstream = bitstream.reshape(-1, 3)
        indices = (bitstream[:, 0] << 2) | (bitstream[:, 1] << 1) | bitstream[:, 2]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if symbols.size == 0:
            return EMPTY_INT

        # Map I/Q → LUT indices
        re = symbols.real
        im = symbols.imag

        i = np.clip(((re + 1.5) * self._scale).astype(np.int16), 0, self._lut_max)
        q = np.clip(((im + 1.5) * self._scale).astype(np.int16), 0, self._lut_max)

        # LUT gives angular bin (0..7)
        bins = self._lut[i, q]

        # Match your original mapping exactly
        indices = self._BIN_TO_IDX[bins]

        # Convert to bits
        bits = np.empty((len(symbols), 3), dtype=np.int8)
        bits[:, 0] = (indices >> 2) & 1
        bits[:, 1] = (indices >> 1) & 1
        bits[:, 2] = indices & 1

        return bits

def fast_angle(re, im):
    # Approximate atan2(im, re) scaled to [-pi, pi] Maybe faster on pluto
    abs_im = np.abs(im) + 1e-12  # avoid div0

    r = np.where(re >= 0,
                 (re - abs_im) / (re + abs_im),
                 (re + abs_im) / (abs_im - re))

    angle = np.where(re >= 0,
                     np.pi/4 - (np.pi/4) * r,
                     3*np.pi/4 - (np.pi/4) * r)

    return np.where(im < 0, -angle, angle)
"""
