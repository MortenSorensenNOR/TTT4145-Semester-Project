"""Construct frames with header data, pilots, and error correction information."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from .channel_coding import CodeRates, LDPC, LDPCConfig


def int_to_bits(n: int, length: int) -> list[int]:
    """Convert an integer to a fixed-width big-endian bit list."""
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


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""
    payload_length_bits: int = 8
    src_bits: int = 4
    dst_bits: int = 4
    mod_scheme_bits: int = 3
    reserved_bits: int = 1
    crc_bits: int = 4
    # TODO: Add coding rate field and reduce src and dst bit lengths, and increase payload length
    header_total_size: int = field(init=False)

    def __post_init__(self):
        self.header_total_size = (
            self.payload_length_bits +
            self.src_bits +
            self.dst_bits +
            self.mod_scheme_bits +
            self.reserved_bits +
            self.crc_bits
        )


class FrameHeaderConstructor:
    """Encode and decode frame header fields."""
    def __init__(
        self,
        length_bits: int,
        src_bits: int,
        dst_bits: int,
        mod_scheme_bits: int,
        crc_bits: int,
    ) -> None:
        """Initialize fixed bit widths for each frame header field."""
        self.length_bits = length_bits
        self.src_bits = src_bits
        self.dst_bits = dst_bits
        self.mod_scheme_bits = mod_scheme_bits
        self.crc_bits = crc_bits

        self.header_length = (
            self.length_bits
            + self.src_bits
            + self.dst_bits
            + self.mod_scheme_bits
            + self.crc_bits
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
        header: FrameHeader
    ) -> list[int]:
        """Encode frame header."""
        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, self.mod_scheme_bits)

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
        self.idx += 1  # skip padding bit
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
    """Build and parse frames based on a configured header format."""

    def __init__(
        self,
        data_size: int,  # number of data bits per frame
        code_rate: CodeRates,
        pilots: Sequence[int] | np.ndarray,
        header_config: FrameHeaderConfig | None = None,
    ) -> None:
        """Initialize frame construction parameters."""
        self.data_size = data_size
        self.code_rate = code_rate
        self.pilots = pilots
        self.header_config = header_config or FrameHeaderConfig()

        self.frame_header_constructor = FrameHeaderConstructor(
            self.header_config.payload_length_bits,
            self.header_config.src_bits,
            self.header_config.dst_bits,
            self.header_config.mod_scheme_bits,
            self.header_config.crc_bits,
        )

        # TODO: Make config not fixed to one length, in order to facilitate multiple header lengths and coding rates
        self.ldpc_config = LDPCConfig(
            n=648,
            k=324,
            Z=27,
            code_rate=CodeRates.HALF_RATE
        )
        self.LDPC = LDPC(self.ldpc_config)

    def encode(self, header: FrameHeader, payload: np.ndarray) -> np.ndarray:
        """Encode data into a frame."""
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = header_bits # TODO: Add header specific channel coding

        assert payload.shape[0] == self.ldpc_config.k # TODO: make this dynamic
        payload_encoded = self.LDPC.encode(payload)

        return np.concatenate([header_encoded, payload_encoded]) 

    def decode(self, _frame: np.ndarray) -> tuple[FrameHeader, np.ndarray]:
        """Decode a frame to extract data bits."""
        header_bits_encoded = _frame[:self.header_config.header_total_size]
        payload_bits_encoded = _frame[self.header_config.header_total_size:]

        # TODO: header specific channel decoding
        header_bits = header_bits_encoded
        header = self.frame_header_constructor.decode(header_bits)
        payload_bits = self.LDPC.decode(payload_bits_encoded)

        return (header, payload_bits)
