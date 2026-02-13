"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from channel_coding import CodeRates


def int_to_bits(n, length):
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]


class ModulationSchemes(Enum):
    """Supported modulation schemes."""

    BPSK = 1
    QPSK = 2
    QAM16 = 3
    QAM64 = 4


@dataclass
class FrameHeader:
    """Frame header with metadata."""

    length: int
    src: int
    dst: int
    mod_scheme: ModulationSchemes
    crc: int
    crc_passed: bool = True


class FrameHeaderDecoder:
    def __init__(
        self,
        length_bits: int,
        src_bits: int,
        dst_bits: int,
        mod_scheme_bits: int,
        crc_bits: int,
    ):
        self.length_bits = length_bits
        self.src_bits = src_bits
        self.dst_bits = dst_bits
        self.mod_scheme_bits = mod_scheme_bits
        self.crc_bits = crc_bits

        self.header_length = (
            config.length_bits
            + config.src_bits
            + config.dst_bits
            + config.mod_scheme_bits
            + config.crc_bits
        )
        self.header_length = 2 * int(np.ceil(self.header_length / 2))

    def _crc_calc(self, data_bits: str, poly: int = 0b10011) -> int:
        """Calculate CRC checksum."""
        reg = int(data_bits, 2) << 4
        for i in range(len(data_bits) - 1, -1, -1):
            if reg & (1 << (i + 4)):
                reg ^= poly << i
        return reg & 0b1111

    def encode(
        self,
        length: int,
        src: int,
        dst: int,
        mod_scheme: ModulationSchemes,
    ) -> list[int]:
        """Encode frame header."""
        length_bits = int_to_bits(length, self.length_bits)
        src_bits = int_to_bits(src, self.src_bits)
        dst_bits = int_to_bits(dst, self.src_bits)
        mod_scheme_bits = int_to_bits(mod_scheme.value, self.mod_scheme_bits)

        header_data_bits = length_bits + src_bits + dst_bits + mod_scheme_bits + [0]
        data_bits = "".join([str(bit) for bit in header_data_bits])
        crc_bits = self._crc_calc(data_bits)
        header_data_bits += int_to_bits(crc_bits, self.crc_bits)

        return header_data_bits

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        self.idx = 0
        length_bits = self._get_length_bits(header)
        src_bits = self._get_src_dst_bits(header)
        dst_bits = self._get_src_dst_bits(header)
        mod_scheme_bits = self._get_mod_scheme_bits(header)
        crc_bits = self._get_crc_bits(header)

        # Convert bit lists to integers
        length = int("".join(str(b) for b in length_bits), 2)
        src = int("".join(str(b) for b in src_bits), 2)
        dst = int("".join(str(b) for b in dst_bits), 2)
        mod_scheme = ModulationSchemes(
            int("".join(str(b) for b in mod_scheme_bits), 2),
        )
        crc = int("".join(str(b) for b in crc_bits), 2)

        # check crc
        header_data_bits = "".join(
            [
                str(bit)
                for bit in (length_bits + src_bits + dst_bits + mod_scheme_bits + [0])
            ],
        )
        calculated_crc = self._crc_calc(header_data_bits)
        calculated_crc_bits = int_to_bits(calculated_crc, self.crc_bits)

        return FrameHeader(
            length,
            src,
            dst,
            mod_scheme,
            crc,
            crc_passed=(crc_bits == calculated_crc_bits),
        )

    def _get_length_bits(self, header: np.ndarray) -> list[int]:
        """Extract length field from header."""
        length = header[self.idx : self.idx + self.length_bits]
        self.idx += self.length_bits
        return length.tolist() if isinstance(length, np.ndarray) else length

    def _get_src_dst_bits(self, header: np.ndarray) -> list[int]:
        """Extract source/destination field from header."""
        if self.src_bits != self.dst_bits:
            msg = "Source and destination bit widths must be equal"
            raise ValueError(msg)
        sd = header[self.idx : self.idx + self.src_bits]
        self.idx += self.src_bits
        return sd.tolist() if isinstance(sd, np.ndarray) else sd

    def _get_mod_scheme_bits(self, header: np.ndarray) -> list[int]:
        """Extract modulation scheme field from header."""
        scheme = header[self.idx : self.idx + self.mod_scheme_bits]
        self.idx += self.mod_scheme_bits
        return scheme.tolist() if isinstance(scheme, np.ndarray) else scheme

    def _get_crc_bits(self, header: np.ndarray) -> list[int]:
        """Extract CRC field from header."""
        crc = header[self.idx : self.idx + self.crc_bits]
        self.idx += self.crc_bits
        return crc.tolist() if isinstance(crc, np.ndarray) else crc


class FrameConstructor:
    def __init__(
        self,
        data_size: int,  # number of data bits per frame
        code_rate: CodeRates,
        pilots,  # TODO: Figure out this
        payload_length_bits: int = 8,
        src_bits: int = 4,
        dst_bits: int = 4,
        mod_scheme_bits: int = 3,
        crc_bits: int = 4,
    ):
        self.frame_header_generator = FrameHeader(
            payload_length_bits,
            src_bits,
            dst_bits,
            mod_scheme_bits,
            crc_bits,
        )

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data into a frame."""
        raise NotImplementedError

    def decode(self, _frame: np.ndarray) -> np.ndarray:
        """Decode a frame to extract data bits."""
        return np.array([], dtype=int)


# Util
def int_to_bits(n, length):
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]
