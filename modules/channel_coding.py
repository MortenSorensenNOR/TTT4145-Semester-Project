"""Channel coding enums and helpers."""

from enum import Enum
from dataclasses import dataclass
import numpy as np

class CodeRates(Enum):
    """Supported channel coding rates."""
    HALF_RATE = 0
    TWO_THIRDS_RATE = 1
    THREE_QUARTER_RATE = 2
    FIVE_SIXTH_RATE = 3

    @property
    def value_float(self) -> float:
        """Return the numeric code rate value (k/n ratio).

        Returns:
            The code rate as a float (e.g., 0.5 for HALF_RATE, 0.833... for FIVE_SIXTH_RATE).
        """
        rate_values = {
            CodeRates.HALF_RATE: 1 / 2,
            CodeRates.TWO_THIRDS_RATE: 2 / 3,
            CodeRates.THREE_QUARTER_RATE: 3 / 4,
            CodeRates.FIVE_SIXTH_RATE: 5 / 6,
        }
        return rate_values[self]

"""
Golay channel coding for frame header
"""

class Golay:
    """Golay channel coding for the frame header. Since frame header is 24 bits, the header will be encoded in two blocks."""

    def __init__(self) -> None:
        self.block_length = 24
        self.message_length = 12
        self.matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,   1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,   1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,   0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,   0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        ])

        self.P = self.matrix[:, 12:]
        self.H = np.hstack([self.P.T, np.eye(12, dtype=int)])

    def encode(self, message: np.ndarray) -> np.ndarray:
        assert message.shape[0] % self.message_length == 0, "Message must have a length that is a multiple of 12"
        assert np.all((message == 0) | (message == 1)), "Message must be binary"
        blocks = message.reshape(-1, 12)          # shape (num_blocks, 12)
        encoded = (blocks @ self.matrix) % 2      # shape (num_blocks, 24)
        return encoded.flatten()

    def decode(self, received: np.ndarray) -> np.ndarray:
        assert received.shape[0] % self.message_length == 0, "Received signal must have a length that is a multiple of 12"

        decoded = []
        for block_idx in range(received.shape[0] // self.block_length):
            r = received[block_idx*24:(block_idx+1)*24]
            corrected = self._decode_block(r)
            decoded.append(corrected[:12])

        return np.concatenate(decoded)

    def _decode_block(self, block):
        def weight(v):
            return int(np.sum(v))

        def unit(i, n):
            e = np.zeros(n, dtype=int)
            e[i] = 1
            return e

        s = (block @ self.H.T) % 2
        if weight(s) <= 3:
            # Errors in parity bits (last 12 positions)
            e = np.concatenate([np.zeros(12, dtype=int), s])
            return (block + e) % 2

        for i in range(12):
            candidate = (s + self.P.T[:, i]) % 2
            if weight(candidate) <= 2:
                # Error in systematic bit i, plus errors in parity bits
                e = np.concatenate([unit(i, 12), candidate])
                return (block + e) % 2
        sP = (s @ self.P) % 2

        if weight(sP) <= 3:
            # Errors in systematic bits (first 12 positions)
            e = np.concatenate([sP, np.zeros(12, dtype=int)])
            return (block + e) % 2

        for i in range(12):
            candidate = (sP + self.P[i]) % 2
            if weight(candidate) <= 2:
                # Error in parity bit i, plus errors in systematic bits
                e = np.concatenate([candidate, unit(i, 12)])
                return (block + e) % 2

        raise ValueError("More than 3 bit errors in block")
        

"""
LDPC channel coding for payload
"""

class LDPC_BaseMatrix:
    """Base matrices for 802.11 LDPC codes (Tables F-1, F-2, F-3)"""

    # n=648, Z=27
    _N648_R12 = np.array([
        [ 0, -1, -1, -1,  0,  0, -1, -1,  0, -1, -1,  0,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [22,  0, -1, -1, 17, -1,  0,  0, 12, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 6, -1,  0, -1, 10, -1, -1, -1, 24, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 2, -1, -1,  0, 20, -1, -1, -1, 25,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1],
        [23, -1, -1, -1,  3, -1, -1, -1,  0, -1,  9, 11, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1],
        [24, -1, 23,  1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [25, -1, -1, -1,  8, -1, -1, -1,  7, 18, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [13, 24, -1, -1,  0, -1,  8, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [ 7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1],
        [11, -1, -1, -1, 19, -1, -1, -1, 13, -1,  3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [25, -1,  8, -1, 23, 18, -1, 14,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [ 3, -1, -1, -1, 16, -1, -1,  2, 25,  5, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N648_R23 = np.array([
        [25, 26, 14, -1, 20, -1,  2, -1,  4, -1, -1,  8, -1, 16, -1, 18,  1,  0, -1, -1, -1, -1, -1, -1],
        [10,  9, 15, 11, -1,  0, -1,  1, -1, -1, 18, -1,  8, -1, 10, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [16,  2, 20, 26, 21, -1,  6, -1,  1, 26, -1,  7, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [10, 13,  5,  0, -1,  3, -1,  7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1,  0,  0, -1, -1, -1],
        [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1,  0, -1, -1, -1,  0,  0, -1, -1],
        [ 6, 22,  9, 20, -1, 25, -1, 17, -1,  8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1,  0,  0],
        [17, 11, 11, 20, -1, 21, -1, 26, -1,  3, -1, -1, 18, -1, 26, -1,  1, -1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N648_R34 = np.array([
        [16, 17, 22, 24,  9,  3, 14, -1,  4,  2,  7, -1, 26, -1,  2, -1, 21, -1,  1,  0, -1, -1, -1, -1],
        [25, 12, 12,  3,  3, 26,  6, 21, -1, 15, 22, -1, 15, -1,  4, -1, -1, 16, -1,  0,  0, -1, -1, -1],
        [25, 18, 26, 16, 22, 23,  9, -1,  0, -1,  4, -1,  4, -1,  8, 23, 11, -1, -1, -1,  0,  0, -1, -1],
        [ 9,  7,  0,  1, 17, -1, -1,  7,  3, -1,  3, 23, -1, 16, -1, -1, 21, -1,  0, -1, -1,  0,  0, -1],
        [24,  5, 26,  7,  1, -1, -1, 15, 24, 15, -1,  8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1,  0,  0],
        [ 2,  2, 19, 14, 24,  1, 15, 19, -1, 21, -1,  2, -1, 24, -1,  3, -1,  2,  1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N648_R56 = np.array([
        [17, 13,  8, 21,  9,  3, 18, 12, 10,  0,  4, 15, 19,  2,  5, 10, 26, 19, 13, 13,  1,  0, -1, -1],
        [ 3, 12, 11, 14, 11, 25,  5, 18,  0,  9,  2, 26, 26, 10, 24,  7, 14, 20,  4,  2, -1,  0,  0, -1],
        [22, 16,  4,  3, 10, 21, 12,  5, 21, 14, 19,  5, -1,  8,  5, 18, 11,  5,  5, 15,  0, -1,  0,  0],
        [ 7,  7, 14, 14,  4, 16, 16, 24, 24, 10,  1,  7, 15,  6, 10, 26,  8, 18, 21, 14,  1, -1, -1,  0],
    ], dtype=int)

    # n=1296, Z=54
    _N1296_R12 = np.array([
        [40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [50,  1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [39, 50, -1, -1,  4, -1,  2, -1, -1, -1, -1, 49, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [33, -1, -1, 38, 37, -1, -1,  4,  1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1],
        [45, -1, -1, -1,  0, 22, -1, -1, 20, 42, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1],
        [51, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [ 5, -1, 25, -1,  6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1],
        [ 1, -1, 27, -1,  1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [-1, 18, -1, -1, 23, -1, -1,  8,  0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, 19,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1296_R23 = np.array([
        [39, 31, 22, 43, -1, 40,  4, -1, 11, -1, -1, 50, -1, -1, -1,  6,  1,  0, -1, -1, -1, -1, -1, -1],
        [25, 52, 41,  2,  6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [43, 31, 29,  0, 21, -1, 28, -1, -1,  2, -1, -1,  7, -1, 17, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [20, 33, 48, -1,  4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [45,  7, 18, 51, 12, 25, -1, -1, -1, 50, -1,  5, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1],
        [35, 40, 32, 16,  5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [ 9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1,  0,  0],
        [32, 22,  4, 21, 16, -1, -1, -1, 27, 28, -1, -1, 38, -1, -1, -1,  8,  1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1296_R34 = np.array([
        [39, 40, 51, 41,  3, 29,  8, 36, -1, 14, -1,  6, -1, 33, -1, 11, -1,  4,  1,  0, -1, -1, -1, -1],
        [48, 21, 47,  9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1,  0,  0, -1, -1, -1],
        [30, 39, 28, 42, 50, 39,  5, 17, -1,  6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1,  0,  0, -1, -1],
        [29,  0,  1, 43, 36, 30, 47, -1, 49, -1, 47, -1,  3, -1, 35, -1, 34, -1,  0, -1, -1,  0,  0, -1],
        [ 1, 32, 11, 23, 10, 44, 12,  7, -1, 48, -1,  4, -1,  9, -1, 17, -1, 16, -1, -1, -1, -1,  0,  0],
        [13,  7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1,  2, -1, 53, -1,  1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1296_R56 = np.array([
        [48, 29, 37, 52,  2, 16,  6, 14, 53, 31, 34,  5, 18, 42, 53, 31, 45, -1, 46, 52,  1,  0, -1, -1],
        [17,  4, 30,  7, 43, 11, 24,  6, 14, 21,  6, 39, 17, 40, 47,  7, 15, 41, 19, -1, -1,  0,  0, -1],
        [ 7,  2, 51, 31, 46, 23, 16, 11, 53, 40, 10,  7, 46, 53, 33, 35, -1, 25, 35, 38,  0, -1,  0,  0],
        [19, 48, 41,  1, 10,  7, 36, 47,  5, 29, 52, 52, 31, 10, 26,  6,  3,  2, -1, 51,  1, -1, -1,  0],
    ], dtype=int)

    # n=1944, Z=81
    _N1944_R12 = np.array([
        [57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [ 3, -1, 28, -1,  0, -1, -1, -1, 55,  7, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
        [62, 53, -1, -1, 53, -1, -1,  3, 35, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1],
        [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1],
        [ 0, -1, -1, -1,  8, -1, 42, -1, 50, -1, -1,  8, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [69, 79, 79, -1, -1, 56, -1, 52, -1, -1, -1,  0, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1],
        [-1, 45, -1, 70,  0, -1, -1, -1, 77,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [ 2, 56, -1, 57, 35, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
        [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1944_R23 = np.array([
        [61, 75,  4, 63, 56, -1, -1, -1, -1, -1, -1,  8, -1,  2, 17, 25,  1,  0, -1, -1, -1, -1, -1, -1],
        [56, 74, 77, 20, -1, -1, -1, 64, 24,  4, 67, -1,  7, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
        [28, 21, 68, 10,  7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1,  0,  0, -1, -1, -1, -1],
        [48, 38, 43, 78, 76, -1, -1, -1, -1,  5, 36, -1, 15, 72, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
        [40,  2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1,  0, -1, -1, -1,  0,  0, -1, -1],
        [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1,  0,  0, -1],
        [12,  0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1,  0,  0],
        [58,  8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58,  1, -1, -1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1944_R34 = np.array([
        [48, 29, 28, 39,  9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22,  1,  0, -1, -1, -1, -1],
        [ 4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1,  0,  0, -1, -1, -1],
        [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59,  7, -1, -1, 32, -1, -1,  0,  0, -1, -1],
        [ 9, 65, 44,  9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39,  0, -1, -1,  0,  0, -1],
        [ 3, 62,  7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1,  9, -1, 72, -1, -1, -1, -1, -1,  0,  0],
        [26, 75, 33, 21, 69, 59,  3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1,  1, -1, -1, -1, -1,  0],
    ], dtype=int)

    _N1944_R56 = np.array([
        [13, 48, 80, 66,  4, 74,  7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1,  1,  0, -1, -1],
        [69, 63, 74, 56, 64, 77, 57, 65,  6, 16, 51, -1, 64, -1, 68,  9, 48, 62, 54, 27, -1,  0,  0, -1],
        [51, 15,  0, 80, 24, 25, 42, 54, 44, 71, 71,  9, 67, 35, -1, 58, -1, 29, -1, 53,  0, -1,  0,  0],
        [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65,  4, 65, 52, -1,  4, -1, 73, 52,  1, -1, -1,  0],
    ], dtype=int)

    def get_matrix(self, code_rate: CodeRates, N: int) -> np.ndarray:
        matrices = {
            648: {
                CodeRates.HALF_RATE: self._N648_R12,
                CodeRates.TWO_THIRDS_RATE: self._N648_R23,
                CodeRates.THREE_QUARTER_RATE: self._N648_R34,
                CodeRates.FIVE_SIXTH_RATE: self._N648_R56,
            },
            1296: {
                CodeRates.HALF_RATE: self._N1296_R12,
                CodeRates.TWO_THIRDS_RATE: self._N1296_R23,
                CodeRates.THREE_QUARTER_RATE: self._N1296_R34,
                CodeRates.FIVE_SIXTH_RATE: self._N1296_R56,
            },
            1944: {
                CodeRates.HALF_RATE: self._N1944_R12,
                CodeRates.TWO_THIRDS_RATE: self._N1944_R23,
                CodeRates.THREE_QUARTER_RATE: self._N1944_R34,
                CodeRates.FIVE_SIXTH_RATE: self._N1944_R56,
            },
        }
        if N not in matrices:
            raise ValueError(f"Unsupported block length N={N}. Must be 648, 1296, or 1944.")
        if code_rate not in matrices[N]:
            raise ValueError(f"Unsupported code rate {code_rate} for N={N}.")
        return matrices[N][code_rate]

@dataclass(frozen=True)
class LDPCConfig:
    """Configuration for LDPC encoding/decoding.

    Only k (message length) and code_rate are required - n and Z are derived.
    Made frozen (immutable) so it can be used as a cache key.
    """
    k: int  # message length (payload bits)
    code_rate: CodeRates

    @property
    def n(self) -> int:
        """Codeword length derived from k and code_rate."""
        rate_fractions = {
            CodeRates.HALF_RATE: (1, 2),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }
        num, denom = rate_fractions[self.code_rate]
        n = (self.k * denom) // num
        if n not in (648, 1296, 1944):
            raise ValueError(
                f"Invalid k={self.k} for {self.code_rate}: computed n={n} "
                f"is not a valid block length (648, 1296, or 1944)."
            )
        return n

    @property
    def Z(self) -> int:
        """Circulant size derived from n."""
        params = {648: 27, 1296: 54, 1944: 81}
        return params[self.n]


class LDPC:
    """LDPC codec with dynamic configuration support and memoization.

    Supports dynamic selection of code rate and block length at encode/decode time.
    H matrices and adjacency lists are cached to avoid recomputation.
    """

    # Class-level cache: LDPCConfig -> (H, check_neighbors, var_neighbors)
    _cache: dict = {}
    _base_matrix_generator = LDPC_BaseMatrix()

    def get_supported_payload_lengths(self, code_rate: CodeRates = CodeRates.HALF_RATE) -> np.ndarray:
        """Return supported message lengths (k) for LDPC.

        If code_rate is specified, returns k values valid for that rate.
        If None, returns all valid k values across all rates.
        """
        valid_n = [648, 1296, 1944]
        rate_fractions = {
            CodeRates.HALF_RATE: (1, 2),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }

        if code_rate is not None:
            num, denom = rate_fractions[code_rate]
            return np.array([n * num // denom for n in valid_n])

    @classmethod
    def _get_cached_structures(cls, config: LDPCConfig):
        """Get or compute H matrix and adjacency lists for the given configuration.

        Returns:
            Tuple of (H, check_neighbors, var_neighbors)
        """
        if config not in cls._cache:
            H = cls._expand_h(config)
            check_neighbors, var_neighbors = cls._build_adj_list(H)
            cls._cache[config] = (H, check_neighbors, var_neighbors)

        return cls._cache[config]

    @classmethod
    def clear_cache(cls):
        """Clear the cached H matrices and adjacency lists."""
        cls._cache.clear()

    @classmethod
    def get_structures(cls, config: LDPCConfig):
        """Get the H matrix and adjacency lists for the given configuration."""
        return cls._get_cached_structures(config)

    @staticmethod
    def _circulant_multiply(vector, shift, Z):
        """Multiply a Z-length vector by a cyclically shifted identity matrix."""
        if shift == -1:
            return np.zeros(Z, dtype=int)
        return np.roll(vector, -shift)

    def encode(self, message: np.ndarray, config: LDPCConfig) -> np.ndarray:
        """Encode k-bit message to a n-bit codeword.

        Args:
            message: length k bit array
            config: LDPC configuration specifying n and code_rate

        Returns:
            length n codeword [systematic | parity]
        """
        n, k, Z = config.n, config.k, config.Z

        assert len(message) == k, f"Message length {len(message)} != expected {k} ({n}, {k}, {Z}, {config.code_rate})"

        H_base = self._base_matrix_generator.get_matrix(config.code_rate, n)
        num_parity_blocks = H_base.shape[0]
        num_systematic_blocks = H_base.shape[1] - num_parity_blocks

        s_blocks = message.reshape(num_systematic_blocks, Z)
        p_blocks = np.zeros((num_parity_blocks, Z), dtype=int)

        # Extract parity submatrix B (rightmost columns of H_base)
        B = H_base[:, num_systematic_blocks:]

        # Compute λ_i = sum of H[i,j] * s[j] for systematic columns
        lambda_sums = np.zeros((num_parity_blocks, Z), dtype=int)
        for i in range(num_parity_blocks):
            for j in range(num_systematic_blocks):
                shift = H_base[i, j]
                if shift != -1:
                    lambda_sums[i] ^= self._circulant_multiply(s_blocks[j], shift, Z)

        # 802.11 LDPC encoding using the dual-diagonal structure of B
        # Step 1: Compute p[0] by XORing all λ values
        # This works because the first parity column has entries that sum to identity
        p0_sum = np.zeros(Z, dtype=int)
        for i in range(num_parity_blocks):
            p0_sum ^= lambda_sums[i]
        p_blocks[0] = p0_sum

        # Step 2: Solve remaining parity blocks using back-substitution
        # For each row i, solve for the unknown parity block using known values
        for i in range(num_parity_blocks):
            # Find which parity blocks appear in this row
            parity_entries = [(col, B[i, col]) for col in range(num_parity_blocks) if B[i, col] != -1]

            # Start with λ[i]
            result = lambda_sums[i].copy()

            # XOR all known parity contributions
            unknown_col = -1
            for col, shift in parity_entries:
                if col <= i:
                    # This parity block is already computed
                    result ^= self._circulant_multiply(p_blocks[col], shift, Z)
                else:
                    # This is the unknown we're solving for (should be col = i+1 for staircase)
                    unknown_col = col

            # If we found an unknown parity block, store the result
            if unknown_col != -1 and unknown_col < num_parity_blocks:
                # The unknown parity block appears with some shift in the matrix
                # For 802.11 codes, the diagonal entries are always 0 (identity shift)
                p_blocks[unknown_col] = result

        # Combine systematic and parity
        codeword = np.concatenate([s_blocks.flatten(), p_blocks.flatten()])
        return codeword

    def decode(self, llr_channel: np.ndarray, config: LDPCConfig,
               max_iterations: int = 50, alpha: float = 0.75) -> np.ndarray:
        """Decode using min-sum belief propagation.

        Args:
            llr_channel: Channel LLRs (log-likelihood ratios) for each codeword bit.
            config: LDPC configuration specifying n and code_rate.
            max_iterations: Maximum number of belief propagation iterations.
            alpha: Normalization factor for min-sum (0.75 = normalized, 1.0 = standard).
                   Normalized min-sum typically performs ~0.2 dB better.

        Returns:
            Decoded message bits (k bits).
        """
        n, k = config.n, config.k
        H, check_neighbors, var_neighbors = self._get_cached_structures(config)

        assert len(llr_channel) == n
        hard_decision = np.zeros(n, dtype=int)

        num_checks = H.shape[0]
        num_vars = H.shape[1]

        # initialize messages
        # L_v2c[i][j] = message from variable j to check i
        # L_c2v[i][j] = message from check i to variable j
        L_v2c = {}
        L_c2v = {}

        # initialize variable-to-check messages with channel LLRs
        for j in range(num_vars):
            for i in var_neighbors[j]:
                L_v2c[(i, j)] = llr_channel[j]

        # initialize check-to-variable messages to zero
        for i in range(num_checks):
            for j in check_neighbors[i]:
                L_c2v[(i, j)] = 0.0

        # belief propagation iterations
        for iteration in range(max_iterations):
            # check node update (min-sum)
            for i in range(num_checks):
                neighbors = check_neighbors[i]

                for j in neighbors:
                    # compute message to j using all other neighbors
                    other_neighbors = [jj for jj in neighbors if jj != j]
                    if len(other_neighbors) == 0:
                        L_c2v[(i, j)] = 0.0
                        continue

                    # min-sum - product of signs * mimimum magnitude
                    sign = 1
                    min_mag = float('inf')
                    for jj in other_neighbors:
                        msg = L_v2c[(i, jj)]
                        sign *= -1 if msg < 0 else 1
                        mag = abs(msg)
                        min_mag = mag if mag < min_mag else min_mag

                    L_c2v[(i, j)] = alpha * sign * min_mag

            # variable node update
            for j in range(num_vars):
                neighbors = var_neighbors[j]
                for i in neighbors:
                    # sum channel llr and all other incoming check messages
                    total = llr_channel[j]
                    for ii in neighbors:
                        if ii != i:
                            total += L_c2v[(ii, j)]
                    L_v2c[(i, j)] = total

            # compute hard decision
            L_total = np.zeros(num_vars)
            for j in range(num_vars):
                L_total[j] = llr_channel[j]
                for i in var_neighbors[j]:
                    L_total[j] += L_c2v[(i, j)]

            hard_decision = (L_total < 0).astype(int)

            # check if valid codeword
            syndrome = H @ hard_decision % 2
            if np.all(syndrome == 0):
                return hard_decision[:k]  # valid codeword -> extract systematic bits

        # max iterations reached - return best guess
        return hard_decision[:k]

    @classmethod
    def _expand_h(cls, config: LDPCConfig) -> np.ndarray:
        """Expand base matrix to full H matrix."""
        n, Z = config.n, config.Z
        H_base = cls._base_matrix_generator.get_matrix(config.code_rate, n)
        num_block_rows = H_base.shape[0]
        num_block_cols = H_base.shape[1]

        H = np.zeros((num_block_rows * Z, num_block_cols * Z), dtype=int)

        for i in range(num_block_rows):
            for j in range(num_block_cols):
                shift = H_base[i, j]
                if shift != -1:
                    for k in range(Z):
                        H[i * Z + k, j * Z + (k + shift) % Z] = 1

        return H

    @staticmethod
    def _build_adj_list(H: np.ndarray):
        """Build adjacency list from H matrix."""
        num_checks, num_vars = H.shape

        check_neighbors = [[] for _ in range(num_checks)]  # list of vars
        var_neighbors = [[] for _ in range(num_vars)]      # list of checks
        for i in range(num_checks):
            for j in range(num_vars):
                if H[i, j] == 1:
                    check_neighbors[i].append(j)
                    var_neighbors[j].append(i)
        return check_neighbors, var_neighbors
