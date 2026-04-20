from dataclasses import dataclass
from typing import runtime_checkable, Protocol
import numpy as np
from numpy.typing import NDArray

EMPTY_COMPLEX = np.empty(0, dtype=np.complex64)
EMPTY_INT = np.empty(0, dtype=np.uint8)

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

class PSK8(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 3
        self.qam_order = 8
        self.symbol_mapping = np.array([-1 - 1j, -np.sqrt(2) + 0j, 0 + np.sqrt(2)*1j, -1 + 1j,
                                         0 - np.sqrt(2)*1j, 1 - 1j, 1 + 1j, np.sqrt(2)+ 0j], dtype=np.complex64) / np.sqrt(2)
        self._BIN_TO_IDX = np.array([7, 6, 2, 3, 1, 0, 4, 5], dtype=np.uint8)

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
