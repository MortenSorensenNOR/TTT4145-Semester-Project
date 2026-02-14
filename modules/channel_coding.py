"""Channel coding enums and helpers."""

from enum import Enum
from dataclasses import dataclass
import numpy as np

class CodeRates(Enum):
    """Supported channel coding rates."""
    HALF_RATE = 1
    THREE_QUARTER_RATE = 2

class BaseMatrix:
    """Base matrix for generating parity matrix"""
    base_matrix = np.array([
        [ 0,   -1,   -1,   -1,    0,    0,   -1,   -1,    0,   -1,   -1,    0,    1,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [22,    0,   -1,   -1,   17,   -1,    0,    0,   12,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [ 6,   -1,    0,   -1,   10,   -1,   -1,   -1,   24,   -1,    0,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [ 2,   -1,   -1,    0,   20,   -1,   -1,   -1,   25,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1,   -1],
        [23,   -1,   -1,   -1,    3,   -1,   -1,   -1,    0,   -1,    9,   11,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1,   -1],
        [24,   -1,   23,    1,   17,   -1,    3,   -1,   10,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1,   -1],
        [25,   -1,   -1,   -1,    8,   -1,   -1,   -1,    7,   18,   -1,   -1,    0,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1,   -1],
        [13,   24,   -1,   -1,    0,   -1,    8,   -1,    6,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1,   -1],
        [ 7,   20,   -1,   16,   22,   10,   -1,   -1,   23,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1,   -1],
        [11,   -1,   -1,   -1,   19,   -1,   -1,   -1,   13,   -1,    3,   17,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   -1],
        [25,   -1,    8,   -1,   23,   18,   -1,   14,    9,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0],
        [ 3,   -1,   -1,   -1,   16,   -1,   -1,    2,   25,    5,   -1,   -1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0]
    ], dtype=int)

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

    def circulant_multiply(self, vector, shift, Z):
        """Multiply a Z-length vector by a cyclically shifted identity matrix"""
        if shift == -1:
            return np.zeros(self.Z, dtype=int)
        return np.roll(vector, shift)

    def encode(self, message):
        """Encode k-bit message to a n-bit codeword

        message: length k (324) bit array
        returns: length n (648) codeword [systemic | parity]
        """
        assert len(message) == self.k

        H_base = BaseMatrix.base_matrix
        num_parity_blocks = H_base.shape[0]
        num_systemic_blocks = H_base.shape[1] - num_parity_blocks

        s_blocks = message.reshape(num_systemic_blocks, self.Z)
        p_blocks = np.zeros((num_parity_blocks, self.Z), dtype=int)

        # compute partial sum of A[i, j] * s[j] for each row i
        lambda_sums = np.zeros((num_parity_blocks, self.Z), dtype=int)
        for i in range(num_parity_blocks):
            for j in range():
        

        raise NotImplementedError

    def decode(self, message):
        """"""
        raise NotImplementedError
