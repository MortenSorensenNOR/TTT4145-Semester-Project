"""Golay (24,12) channel coding for the frame header."""

import numpy as np


class Golay:
    """Golay (24,12) channel coding for the frame header.

    Source: https://en.wikipedia.org/wiki/Binary_Golay_code
    """

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
            ], dtype=np.int32
        )

        self.parity = self.matrix[:, 12:]
        self.h_matrix = np.hstack([self.parity.T, np.eye(12, dtype=np.int32)])

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
        received = received.ravel()
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
