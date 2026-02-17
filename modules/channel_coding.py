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
    """Golay channel coding for the farme header. Since frame header is 24 bits, the header will be encoded in two blocks."""

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

    def encode(self, message: np.ndarray) -> np.ndarray:
        assert message.shape[0] % self.message_length == 0, "Message must have a length that is a multiple of 12"
        return np.array([])

    def decode(self, llr_channel: np.ndarray) -> np.ndarray:
        assert llr_channel.shape[0] % self.message_length == 0, "Recieved signal must have a length that is a multiple of 12"

        return np.array([])

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

@dataclass
class LDPCConfig:
    n: int # codeword length
    k: int # message length
    Z: int # circulant size (27 for n=648)
    code_rate: CodeRates

class LDPC:
    """Placeholder LDPC codec implementation."""

    def __init__(self, config: LDPCConfig) -> None:
        """Initialize the LDPC codec placeholder."""
        self.config = config
        self.n = config.n
        self.k = config.k
        self.Z = config.Z
        self.code_rate = config.code_rate
        if self.code_rate == CodeRates.HALF_RATE:
            assert(2 * self.k == self.n)
        elif self.code_rate == CodeRates.THREE_QUARTER_RATE:
            assert(self.k / self.n == 0.75)

        assert self.n == 648 and self.Z == 27 and self.code_rate == CodeRates.HALF_RATE, \
            "Only n = 648 with Z=27 with half rate coding is implemented for now"

        self.base_matrix_generator = LDPC_BaseMatrix()
        self.H = self._expand_h()
        self.check_neighbors, self.var_neighbors = self._build_adj_list(self.H)

    def circulant_multiply(self, vector, shift):
        """Multiply a Z-length vector by a cyclically shifted identity matrix."""
        if shift == -1:
            return np.zeros(self.Z, dtype=int)
        return np.roll(vector, -shift)

    def encode(self, message):
        """Encode k-bit message to a n-bit codeword

        message: length k (324) bit array
        returns: length n (648) codeword [systemic | parity]
        """
        assert len(message) == self.k
        assert self.n == 648
        assert self.Z == 27

        H_base = self.base_matrix_generator.get_matrix(self.code_rate, self.n)
        num_parity_blocks = H_base.shape[0]
        num_systemic_blocks = H_base.shape[1] - num_parity_blocks

        s_blocks = message.reshape(num_systemic_blocks, self.Z)
        p_blocks = np.zeros((num_parity_blocks, self.Z), dtype=int)

        # compute partial sum of A[i, j] * s[j] for each row i
        lambda_sums = np.zeros((num_parity_blocks, self.Z), dtype=int)
        for i in range(num_parity_blocks):
            for j in range(num_systemic_blocks):
                shift = H_base[i, j]
                if shift != -1:
                    lambda_sums[i] ^= self.circulant_multiply(s_blocks[j], shift)

        # compute first parity block p[0]
        # 802.11 structure needs to accumulate all lambda_sums
        p0_sum = np.zeros(self.Z, dtype=int)
        for i in range(num_parity_blocks):
            p0_sum ^= lambda_sums[i]
        p_blocks[0] = p0_sum

        # compute remaining parity blocks using back-substitutions
        # p[1] from row 0: lambda[0] + P^1 * p[0] + P^0 * p[1] = 0
        # The shift for p[0] in row 0 is 1 (from base_matrix[0, 12] = 1)
        p_blocks[1] = lambda_sums[0] ^ self.circulant_multiply(p_blocks[0], 1)

        # p[2] to p[6] from rows 1-5 (standard staircase)
        for i in range(2, 7):
            p_blocks[i] = lambda_sums[i-1] ^ p_blocks[i-1]

        # p[7] from row 6: row 6 has p[0] (col 12), p[6] (col 18), p[7] (col 19)
        # lambda[6] + p[0] + p[6] + p[7] = 0 => p[7] = lambda[6] + p[0] + p[6]
        p_blocks[7] = lambda_sums[6] ^ p_blocks[0] ^ p_blocks[6]

        # p[8] to p[11] from rows 7-10 (standard staircase)
        for i in range(8, num_parity_blocks):
            p_blocks[i] = lambda_sums[i-1] ^ p_blocks[i-1]

        # combine systematic and parity
        codeword = np.concatenate([s_blocks.flatten(), p_blocks.flatten()])
        return codeword

    def decode(self, llr_channel: np.ndarray, max_iterations: int = 50, alpha: float = 0.75) -> np.ndarray:
        """Decode using min-sum belief propagation.

        Args:
            llr_channel: Channel LLRs (log-likelihood ratios) for each codeword bit.
            max_iterations: Maximum number of belief propagation iterations.
            alpha: Normalization factor for min-sum (0.75 = normalized, 1.0 = standard).
                   Normalized min-sum typically performs ~0.2 dB better.

        Returns:
            Decoded message bits (k bits).
        """
        assert len(llr_channel) == self.n
        hard_decision = np.zeros(self.n, dtype=int)

        num_checks = self.H.shape[0]
        num_vars   = self.H.shape[1]

        # initialize messages
        # L_v2c[i][j] = message from variable j to check i
        # L_c2v[i][j] = message from check i to variable j
        L_v2c = {}
        L_c2v = {}

        # initialize variable-to-check messages with channel LLRs
        for j in range(num_vars):
            for i in self.var_neighbors[j]:
                L_v2c[(i, j)] = llr_channel[j]

        # initialize check-to-variable messages to zero
        for i in range(num_checks):
            for j in self.check_neighbors[i]:
                L_c2v[(i, j)] = 0.0

        # belief propagation iterations
        for iteration in range(max_iterations):
            # check node update (min-sum)
            for i in range(num_checks):
                neighbors = self.check_neighbors[i]
                
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
                neighbors = self.var_neighbors[j]
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
                for i in self.var_neighbors[j]:
                    L_total[j] += L_c2v[(i, j)]

            hard_decision = (L_total < 0).astype(int)

            # check if valid codeword
            syndrome = self.H @ hard_decision % 2
            if np.all(syndrome == 0):
                return hard_decision[:self.k] # valid codeword -> extract systemic bits

        # max iterations reached - return best guess
        return hard_decision[:self.k]

    def _expand_h(self) -> np.ndarray:
        """Expand base matrix to full H matrix"""
        H_base = self.base_matrix_generator.get_matrix(self.code_rate, self.n)
        num_block_rows = H_base.shape[0]
        num_block_cols = H_base.shape[1]

        H = np.zeros((num_block_rows * self.Z, num_block_cols * self.Z), dtype=int)

        for i in range(num_block_rows):
            for j in range(num_block_cols):
                shift = H_base[i, j]
                if shift != -1:
                    for k in range(self.Z):
                        H[i * self.Z + k, j * self.Z + (k + shift) % self.Z] = 1

        return H

    def _build_adj_list(self, H: np.ndarray):
        """Build adjecency list from H matrix"""
        num_chekcs, num_vars = H.shape

        check_neighbors = [[] for _ in range(num_chekcs)] # list of vars
        var_neighbors   = [[] for _ in range(num_vars)]   # list of checks
        for i in range(num_chekcs):
            for j in range(num_vars):
                if H[i, j] == 1:
                    check_neighbors[i].append(j)
                    var_neighbors[j].append(i)
        return check_neighbors, var_neighbors
