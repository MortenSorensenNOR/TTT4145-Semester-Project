"""Channel coding enums and helpers."""

from dataclasses import dataclass
from enum import Enum

import numba
import numpy as np
import pyldpc

# Minimum number of edges connected to a check node for meaningful BP update
MIN_CHECK_DEGREE = 2


class CodeRates(Enum):
    """Supported channel coding rates."""

    HALF_RATE = 0
    TWO_THIRDS_RATE = 1
    THREE_QUARTER_RATE = 2
    FIVE_SIXTH_RATE = 3

    @property
    def rate_fraction(self) -> tuple[int, int]:
        """Return the (numerator, denominator) tuple for this code rate."""
        fractions = {
            CodeRates.HALF_RATE: (1, 2),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }
        return fractions[self]

    @property
    def value_float(self) -> float:
        """Return the numeric code rate value (k/n ratio)."""
        num, denom = self.rate_fraction
        return num / denom


class Golay:
    """Golay (24,12) channel coding for the frame header."""

    def __init__(self) -> None:
        """Initialize Golay encoder/decoder with the standard generator matrix."""
        self.block_length = 24
        self.message_length = 12
        self.matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            ],
        )

        self.parity = self.matrix[:, 12:]
        self.h_matrix = np.hstack([self.parity.T, np.eye(12, dtype=int)])

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode a binary message using Golay (24,12) code."""
        if message.shape[0] % self.message_length != 0:
            msg = "Message must have a length that is a multiple of 12"
            raise ValueError(msg)
        if not np.all((message == 0) | (message == 1)):
            msg = "Message must be binary"
            raise ValueError(msg)
        blocks = message.reshape(-1, 12)  # shape (num_blocks, 12)
        encoded = (blocks @ self.matrix) % 2  # shape (num_blocks, 24)
        return encoded.flatten()

    def decode(self, received: np.ndarray) -> np.ndarray:
        """Decode a Golay-encoded signal."""
        if received.shape[0] % self.block_length != 0:
            msg = "Received signal must have a length that is a multiple of 24"
            raise ValueError(msg)

        decoded = []
        for block_idx in range(received.shape[0] // self.block_length):
            r = received[block_idx * 24 : (block_idx + 1) * 24]
            corrected = self._decode_block(r)
            decoded.append(corrected[:12])

        return np.concatenate(decoded)

    def _decode_block(self, block: np.ndarray) -> np.ndarray:
        """Decode a single 24-bit Golay block with error correction."""
        max_correctable = 3
        max_secondary_errors = 2
        half_block = 12

        def weight(v: np.ndarray) -> int:
            return int(np.sum(v))

        def unit(i: int, n: int) -> np.ndarray:
            e = np.zeros(n, dtype=int)
            e[i] = 1
            return e

        s = (block @ self.h_matrix.T) % 2
        if weight(s) <= max_correctable:
            e = np.concatenate([np.zeros(half_block, dtype=int), s])
            return (block + e) % 2

        for i in range(half_block):
            candidate = (s + self.parity.T[:, i]) % 2
            if weight(candidate) <= max_secondary_errors:
                e = np.concatenate([unit(i, half_block), candidate])
                return (block + e) % 2
        s_parity = (s @ self.parity) % 2

        if weight(s_parity) <= max_correctable:
            e = np.concatenate([s_parity, np.zeros(half_block, dtype=int)])
            return (block + e) % 2

        for i in range(half_block):
            candidate = (s_parity + self.parity[i]) % 2
            if weight(candidate) <= max_secondary_errors:
                e = np.concatenate([candidate, unit(i, half_block)])
                return (block + e) % 2

        msg = "More than 3 bit errors in block"
        raise ValueError(msg)


# Base matrices for 802.11 LDPC codes (Tables F-1, F-2, F-3)

# n=648, Z=27
_N648_R12 = np.array(
    [
        [0, -1, -1, -1, 0, 0, -1, -1, 0, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [22, 0, -1, -1, 17, -1, 0, 0, 12, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [6, -1, 0, -1, 10, -1, -1, -1, 24, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, -1, -1, 0, 20, -1, -1, -1, 25, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [23, -1, -1, -1, 3, -1, -1, -1, 0, -1, 9, 11, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [24, -1, 23, 1, 17, -1, 3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [25, -1, -1, -1, 8, -1, -1, -1, 7, 18, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [13, 24, -1, -1, 0, -1, 8, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [11, -1, -1, -1, 19, -1, -1, -1, 13, -1, 3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [25, -1, 8, -1, 23, 18, -1, 14, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [3, -1, -1, -1, 16, -1, -1, 2, 25, 5, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N648_R23 = np.array(
    [
        [25, 26, 14, -1, 20, -1, 2, -1, 4, -1, -1, 8, -1, 16, -1, 18, 1, 0, -1, -1, -1, -1, -1, -1],
        [10, 9, 15, 11, -1, 0, -1, 1, -1, -1, 18, -1, 8, -1, 10, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [16, 2, 20, 26, 21, -1, 6, -1, 1, 26, -1, 7, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [10, 13, 5, 0, -1, 3, -1, 7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1, 0, 0, -1, -1, -1],
        [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [6, 22, 9, 20, -1, 25, -1, 17, -1, 8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, 0, 0],
        [17, 11, 11, 20, -1, 21, -1, 26, -1, 3, -1, -1, 18, -1, 26, -1, 1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N648_R34 = np.array(
    [
        [16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
        [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
        [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, 4, -1, 8, 23, 11, -1, -1, -1, 0, 0, -1, -1],
        [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
        [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
        [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, 2, -1, 24, -1, 3, -1, 2, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N648_R56 = np.array(
    [
        [17, 13, 8, 21, 9, 3, 18, 12, 10, 0, 4, 15, 19, 2, 5, 10, 26, 19, 13, 13, 1, 0, -1, -1],
        [3, 12, 11, 14, 11, 25, 5, 18, 0, 9, 2, 26, 26, 10, 24, 7, 14, 20, 4, 2, -1, 0, 0, -1],
        [22, 16, 4, 3, 10, 21, 12, 5, 21, 14, 19, 5, -1, 8, 5, 18, 11, 5, 5, 15, 0, -1, 0, 0],
        [7, 7, 14, 14, 4, 16, 16, 24, 24, 10, 1, 7, 15, 6, 10, 26, 8, 18, 21, 14, 1, -1, -1, 0],
    ],
    dtype=int,
)

# n=1296, Z=54
_N1296_R12 = np.array(
    [
        [40, -1, -1, -1, 22, -1, 49, 23, 43, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [50, 1, -1, -1, 48, 35, -1, -1, 13, -1, 30, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [39, 50, -1, -1, 4, -1, 2, -1, -1, -1, -1, 49, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [33, -1, -1, 38, 37, -1, -1, 4, 1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [45, -1, -1, -1, 0, 22, -1, -1, 20, 42, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [51, -1, 48, 35, -1, -1, -1, 44, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [47, 11, -1, -1, -1, 17, -1, -1, 51, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [5, -1, 25, -1, 6, -1, 45, -1, 13, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [33, -1, -1, 34, 24, -1, -1, -1, 23, -1, -1, 46, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [1, -1, 27, -1, 1, -1, -1, -1, 38, -1, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [-1, 18, -1, -1, 23, -1, -1, 8, 0, 35, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [49, -1, 17, -1, 30, -1, -1, -1, 34, -1, 19, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1296_R23 = np.array(
    [
        [39, 31, 22, 43, -1, 40, 4, -1, 11, -1, -1, 50, -1, -1, -1, 6, 1, 0, -1, -1, -1, -1, -1, -1],
        [25, 52, 41, 2, 6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [43, 31, 29, 0, 21, -1, 28, -1, -1, 2, -1, -1, 7, -1, 17, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [20, 33, 48, -1, 4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [45, 7, 18, 51, 12, 25, -1, -1, -1, 50, -1, 5, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [35, 40, 32, 16, 5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1, 0, 0],
        [32, 22, 4, 21, 16, -1, -1, -1, 27, 28, -1, -1, 38, -1, -1, -1, 8, 1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1296_R34 = np.array(
    [
        [39, 40, 51, 41, 3, 29, 8, 36, -1, 14, -1, 6, -1, 33, -1, 11, -1, 4, 1, 0, -1, -1, -1, -1],
        [48, 21, 47, 9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1, 0, 0, -1, -1, -1],
        [30, 39, 28, 42, 50, 39, 5, 17, -1, 6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1, 0, 0, -1, -1],
        [29, 0, 1, 43, 36, 30, 47, -1, 49, -1, 47, -1, 3, -1, 35, -1, 34, -1, 0, -1, -1, 0, 0, -1],
        [1, 32, 11, 23, 10, 44, 12, 7, -1, 48, -1, 4, -1, 9, -1, 17, -1, 16, -1, -1, -1, -1, 0, 0],
        [13, 7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1, 2, -1, 53, -1, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1296_R56 = np.array(
    [
        [48, 29, 37, 52, 2, 16, 6, 14, 53, 31, 34, 5, 18, 42, 53, 31, 45, -1, 46, 52, 1, 0, -1, -1],
        [17, 4, 30, 7, 43, 11, 24, 6, 14, 21, 6, 39, 17, 40, 47, 7, 15, 41, 19, -1, -1, 0, 0, -1],
        [7, 2, 51, 31, 46, 23, 16, 11, 53, 40, 10, 7, 46, 53, 33, 35, -1, 25, 35, 38, 0, -1, 0, 0],
        [19, 48, 41, 1, 10, 7, 36, 47, 5, 29, 52, 52, 31, 10, 26, 6, 3, 2, -1, 51, 1, -1, -1, 0],
    ],
    dtype=int,
)

# n=1944, Z=81
_N1944_R12 = np.array(
    [
        [57, -1, -1, -1, 50, -1, 11, -1, 50, -1, 79, -1, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, -1, 28, -1, 0, -1, -1, -1, 55, 7, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [30, -1, -1, -1, 24, 37, -1, -1, 56, 14, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1],
        [62, 53, -1, -1, 53, -1, -1, 3, 35, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [40, -1, -1, 20, 66, -1, -1, 22, 28, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
        [0, -1, -1, -1, 8, -1, 42, -1, 50, -1, -1, 8, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [69, 79, 79, -1, -1, 56, -1, 52, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [65, -1, -1, -1, 38, 57, -1, -1, 72, -1, 27, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [64, -1, -1, -1, 14, 52, -1, -1, 30, -1, -1, 32, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [-1, 45, -1, 70, 0, -1, -1, -1, 77, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [2, 56, -1, 57, 35, -1, -1, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [24, -1, 61, -1, 60, -1, -1, 27, 51, -1, -1, 16, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1944_R23 = np.array(
    [
        [61, 75, 4, 63, 56, -1, -1, -1, -1, -1, -1, 8, -1, 2, 17, 25, 1, 0, -1, -1, -1, -1, -1, -1],
        [56, 74, 77, 20, -1, -1, -1, 64, 24, 4, 67, -1, 7, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [28, 21, 68, 10, 7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [48, 38, 43, 78, 76, -1, -1, -1, -1, 5, 36, -1, 15, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [40, 2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [12, 0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1, 0, 0],
        [58, 8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58, 1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1944_R34 = np.array(
    [
        [48, 29, 28, 39, 9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22, 1, 0, -1, -1, -1, -1],
        [4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1, 0, 0, -1, -1, -1],
        [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59, 7, -1, -1, 32, -1, -1, 0, 0, -1, -1],
        [9, 65, 44, 9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39, 0, -1, -1, 0, 0, -1],
        [3, 62, 7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1, 9, -1, 72, -1, -1, -1, -1, -1, 0, 0],
        [26, 75, 33, 21, 69, 59, 3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1944_R56 = np.array(
    [
        [13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1, 1, 0, -1, -1],
        [69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, -1, 64, -1, 68, 9, 48, 62, 54, 27, -1, 0, 0, -1],
        [51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, -1, 58, -1, 29, -1, 53, 0, -1, 0, 0],
        [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65, 4, 65, 52, -1, 4, -1, 73, 52, 1, -1, -1, 0],
    ],
    dtype=int,
)


def get_ldpc_base_matrix(code_rate: CodeRates, n: int) -> np.ndarray:
    """Return the base matrix for the given code rate and block length."""
    matrices = {
        648: {
            CodeRates.HALF_RATE: _N648_R12,
            CodeRates.TWO_THIRDS_RATE: _N648_R23,
            CodeRates.THREE_QUARTER_RATE: _N648_R34,
            CodeRates.FIVE_SIXTH_RATE: _N648_R56,
        },
        1296: {
            CodeRates.HALF_RATE: _N1296_R12,
            CodeRates.TWO_THIRDS_RATE: _N1296_R23,
            CodeRates.THREE_QUARTER_RATE: _N1296_R34,
            CodeRates.FIVE_SIXTH_RATE: _N1296_R56,
        },
        1944: {
            CodeRates.HALF_RATE: _N1944_R12,
            CodeRates.TWO_THIRDS_RATE: _N1944_R23,
            CodeRates.THREE_QUARTER_RATE: _N1944_R34,
            CodeRates.FIVE_SIXTH_RATE: _N1944_R56,
        },
    }
    if n not in matrices:
        msg = f"Unsupported block length n={n}. Must be 648, 1296, or 1944."
        raise ValueError(msg)
    if code_rate not in matrices[n]:
        msg = f"Unsupported code rate {code_rate} for n={n}."
        raise ValueError(msg)
    return matrices[n][code_rate]


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
        num, denom = self.code_rate.rate_fraction
        n = (self.k * denom) // num
        if n not in (648, 1296, 1944):
            msg = (
                f"Invalid k={self.k} for {self.code_rate}: computed n={n} "
                f"is not a valid block length (648, 1296, or 1944)."
            )
            raise ValueError(
                msg,
            )
        return n

    @property
    def z(self) -> int:
        """Circulant size derived from n."""
        params = {648: 27, 1296: 54, 1944: 81}
        return params[self.n]


# ---------------------------------------------------------------------------
# LDPC module-level caches
# ---------------------------------------------------------------------------
_h_cache: dict[LDPCConfig, np.ndarray] = {}
_encoding_cache: dict[LDPCConfig, tuple[np.ndarray, np.ndarray]] = {}
_decode_cache: dict[LDPCConfig, tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# LDPC public API
# ---------------------------------------------------------------------------


def ldpc_get_supported_payload_lengths(code_rate: CodeRates = CodeRates.HALF_RATE) -> np.ndarray:
    """Return supported message lengths (k) for LDPC with the given code rate."""
    valid_n = [648, 1296, 1944]
    num, denom = code_rate.rate_fraction
    return np.array([n * num // denom for n in valid_n])


def ldpc_get_h_matrix(config: LDPCConfig) -> np.ndarray:
    """Get (or compute and cache) the full H parity-check matrix."""
    if config not in _h_cache:
        _h_cache[config] = _expand_h(config)
    return _h_cache[config]


def ldpc_clear_cache() -> None:
    """Clear all cached LDPC structures."""
    _h_cache.clear()
    _encoding_cache.clear()
    _decode_cache.clear()


def ldpc_encode(message: np.ndarray, config: LDPCConfig) -> np.ndarray:
    """Encode k-bit message to a n-bit codeword."""
    k = config.k
    if len(message) != k:
        msg = f"Message length {len(message)} != expected {k}"
        raise ValueError(msg)

    g_mat, _ = _get_encoding_structures(config)
    codeword = message @ g_mat % 2
    return codeword.astype(int)


@numba.njit(cache=True)
def _check_node_update(
    v2c: np.ndarray,
    c2v: np.ndarray,
    check_order: np.ndarray,
    check_bounds: np.ndarray,
    alpha: float,
) -> None:
    """JIT-compiled check node update (min-sum)."""
    num_checks = len(check_bounds) - 1
    for ci in range(num_checks):
        start = check_bounds[ci]
        end = check_bounds[ci + 1]
        d = end - start
        if d < MIN_CHECK_DEGREE:
            for j in range(start, end):
                c2v[check_order[j]] = 0.0
            continue

        total_sign = 1.0
        min1 = np.inf
        min2 = np.inf
        argmin_local = 0
        for j in range(start, end):
            msg = v2c[check_order[j]]
            if msg >= 0:
                total_sign *= 1.0
            else:
                total_sign *= -1.0
            mag = abs(msg)
            if mag < min1:
                min2 = min1
                min1 = mag
                argmin_local = j - start
            elif mag < min2:
                min2 = mag

        for j in range(start, end):
            msg = v2c[check_order[j]]
            sign = 1.0 if msg >= 0 else -1.0
            sign_excl = total_sign * sign
            min_excl = min1 if (j - start) != argmin_local else min2
            c2v[check_order[j]] = alpha * sign_excl * min_excl


def ldpc_decode(
    llr_channel: np.ndarray,
    config: LDPCConfig,
    max_iterations: int = 50,
    alpha: float = 0.75,
) -> np.ndarray:
    """Decode using min-sum belief propagation."""
    n, k = config.n, config.k
    if len(llr_channel) != n:
        msg = f"Expected {n}, got {len(llr_channel)}"
        raise ValueError(msg)

    (h_permuted, _num_checks, _num_vars, edge_var, check_order, check_bounds) = _get_decode_structures(config)

    llr = llr_channel.astype(np.float64)
    num_edges = len(edge_var)

    # Message arrays indexed by edge
    v2c = llr[edge_var].copy()
    c2v = np.zeros(num_edges, dtype=np.float64)
    hard_decision = (llr_channel < 0).astype(int)

    for _iteration in range(max_iterations):
        # --- Check node update (min-sum, JIT) ---
        _check_node_update(v2c, c2v, check_order, check_bounds, alpha)

        # --- Variable node update ---
        l_total = llr.copy()
        np.add.at(l_total, edge_var, c2v)
        v2c[:] = l_total[edge_var] - c2v

        # Hard decision + syndrome check
        hard_decision = (l_total < 0).astype(int)
        if np.all(h_permuted @ hard_decision % 2 == 0):
            return hard_decision[:k]

    return hard_decision[:k]


# ---------------------------------------------------------------------------
# LDPC internal helpers
# ---------------------------------------------------------------------------


def _get_encoding_structures(config: LDPCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Get or compute the generator matrix G and permuted H matrix."""
    if config in _encoding_cache:
        return _encoding_cache[config]

    h_mat = ldpc_get_h_matrix(config)

    h_permuted, g_transposed = pyldpc.coding_matrix_systematic(h_mat)
    g_mat = g_transposed.T

    g_mat = np.asarray(g_mat, dtype=int)
    h_permuted = np.asarray(h_permuted, dtype=int)

    _encoding_cache[config] = (g_mat, h_permuted)
    return g_mat, h_permuted


def _get_decode_structures(
    config: LDPCConfig,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Get cached edge-based structures for BP decoding."""
    if config in _decode_cache:
        return _decode_cache[config]

    _, h_permuted = _get_encoding_structures(config)
    num_checks, num_vars = h_permuted.shape

    rows, cols = np.nonzero(h_permuted)

    check_order = np.argsort(rows).astype(np.int64)
    sorted_checks = rows[check_order]
    check_bounds = np.searchsorted(sorted_checks, np.arange(num_checks + 1)).astype(np.int64)

    result = (h_permuted, num_checks, num_vars, cols, check_order, check_bounds)
    _decode_cache[config] = result
    return result


def _expand_h(config: LDPCConfig) -> np.ndarray:
    """Expand base matrix to full H matrix using vectorized circulant placement."""
    n, z = config.n, config.z
    h_base = get_ldpc_base_matrix(config.code_rate, n)
    num_block_rows, num_block_cols = h_base.shape

    h_mat = np.zeros((num_block_rows * z, num_block_cols * z), dtype=int)

    bi, bj = np.nonzero(h_base != -1)
    shifts = h_base[bi, bj]

    idx = np.arange(z)
    rows = (bi[:, np.newaxis] * z + idx[np.newaxis, :]).ravel()
    cols = (bj[:, np.newaxis] * z + (idx[np.newaxis, :] + shifts[:, np.newaxis]) % z).ravel()
    h_mat[rows, cols] = 1

    return h_mat
