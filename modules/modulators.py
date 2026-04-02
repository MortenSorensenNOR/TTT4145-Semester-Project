from dataclasses import dataclass
from typing import runtime_checkable, Protocol
import numpy as np

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
        if len(bitstream) == 0:
            return np.array([], dtype=np.complex64)
        return self.symbol_mapping[bitstream.ravel()]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.array([], dtype=int)
        return np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1).reshape(-1, 1)

class QPSK(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 2
        self.qam_order = 4
        self.symbol_mapping = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=np.complex64) / np.sqrt(2)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if len(bitstream) == 0:
            return np.array([], dtype=np.complex64)
        bitstream = bitstream.reshape(-1, 2)
        indices = bitstream[:, 0] * 2 + bitstream[:, 1]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1)
        return np.column_stack([indices // 2, indices % 2])

class PSK8(Modulator):
    def __init__(self) -> None:
        self.bits_per_symbol = 3
        self.qam_order = 8
        self.symbol_mapping = np.array([-1 - 1j, -np.sqrt(2) + 0j, 0 + np.sqrt(2)*1j, -1 + 1j,
                                         0 - np.sqrt(2)*1j, 1 - 1j, 1 + 1j, np.sqrt(2)+ 0j], dtype=np.complex64) / np.sqrt(2)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if len(bitstream) == 0:
            return np.array([], dtype=np.complex64)
        bitstream = bitstream.reshape(-1, 3)
        indices = bitstream[:, 0] * 4 + bitstream[:, 1] * 2 + bitstream[:, 2]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1)
        return np.column_stack([indices // 4, (indices % 4) // 2, indices % 2])

