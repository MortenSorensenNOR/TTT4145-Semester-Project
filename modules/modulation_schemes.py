import numpy as np

class BPSK():
    def __init__(self) -> None:
        self.bits_per_symbol = 1
        self.qam_order = 2
        self.symbol_mapping = np.array([-1 + 0j, 1 + 0j])

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        return self.symbol_mapping[bitstream]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.ndarray([], dtype=int)
        return np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1).reshape(-1, 1)

class QPSK():
    def __init__(self) -> None:
        self.bits_per_symbol = 2
        self.qam_order = 4
        self.symbol_mapping = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(-1, 2)
        indices = bitstream[:, 0] * 2 + bitstream[:, 1]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1)
        return np.column_stack([indices // 2, indices % 2])

class PSK8():
    def __init__(self) -> None:
        self.bits_per_symbol = 3
        self.qam_order = 8
        self.symbol_mapping = np.array([-1 - 1j, -np.sqrt(2) + 0j, 0 + np.sqrt(2)*1j, -1 + 1j,
                                         0 - np.sqrt(2)*1j, 1 - 1j, 1 + 1j, np.sqrt(2)+ 0j]) / np.sqrt(2)

    def bits2symbols(self, bitstream: np.ndarray) -> np.ndarray:
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(-1, 3)
        indices = bitstream[:, 0] * 4 + bitstream[:, 1] * 2 + bitstream[:, 2]
        return self.symbol_mapping[indices]

    def symbols2bits(self, symbols: np.ndarray) -> np.ndarray:
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbol_mapping[None, :]), axis=1)
        return np.column_stack([indices // 4, (indices % 4) // 2, indices % 2])

