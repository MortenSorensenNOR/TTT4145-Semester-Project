"""Construct frames with header data, pilots, and error correction information."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from .channel_coding import CodeRates, LDPC, LDPCConfig, Golay


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
    coding_rate: CodeRates
    crc: int
    crc_passed: bool = True


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""
    payload_length_bits: int = 10
    src_bits: int = 2
    dst_bits: int = 2
    mod_scheme_bits: int = 3
    coding_rate_bits: int = 2
    reserved_bits: int = 1
    crc_bits: int = 4
    header_total_size: int = field(init=False)

    def __post_init__(self):
        self.header_total_size = (
            self.payload_length_bits +
            self.src_bits +
            self.dst_bits +
            self.mod_scheme_bits +
            self.coding_rate_bits +
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
        coding_rate_bits: int,
        crc_bits: int,
    ) -> None:
        """Initialize fixed bit widths for each frame header field."""
        self.length_bits = length_bits
        self.src_bits = src_bits
        self.dst_bits = dst_bits
        self.mod_scheme_bits = mod_scheme_bits
        self.coding_rate_bits = coding_rate_bits
        self.crc_bits = crc_bits

        self.header_length = (
            self.length_bits
            + self.src_bits
            + self.dst_bits
            + self.mod_scheme_bits
            + self.coding_rate_bits
            + self.crc_bits
        )
        raw_length = self.header_length

        self.header_length = 2 * int(np.ceil(self.header_length / 2))
        self.reserved_bits = self.header_length - raw_length

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
    ) -> np.ndarray:
        """Encode frame header."""
        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, self.mod_scheme_bits)
        coding_rate_bits = int_to_bits(header.coding_rate.value, self.coding_rate_bits)

        header_data_bits = length_bits + src_bits + dst_bits + mod_scheme_bits + coding_rate_bits + [0] * self.reserved_bits
        data_bits = "".join([str(bit) for bit in header_data_bits])
        crc_bits = self._crc_calc(data_bits)
        header_data_bits += int_to_bits(crc_bits, self.crc_bits)

        return np.array(header_data_bits, dtype=int)

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        self.idx = 0
        length_bits = self._get_length_bits(header)
        src_bits = self._get_src_dst_bits(header)
        dst_bits = self._get_src_dst_bits(header)
        mod_scheme_bits = self._get_mod_scheme_bits(header)
        coding_rate_bits = self._get_coding_rate_bits(header)
        self.idx += self.reserved_bits  # skip padding bit
        crc_bits = self._get_crc_bits(header)

        # Convert bit lists to integers
        length = int("".join(str(b) for b in length_bits), 2)
        src = int("".join(str(b) for b in src_bits), 2)
        dst = int("".join(str(b) for b in dst_bits), 2)
        mod_scheme = ModulationSchemes(
            int("".join(str(b) for b in mod_scheme_bits), 2),
        )
        coding_rate = CodeRates(
            int("".join(str(b) for b in coding_rate_bits), 2),
        )
        crc = int("".join(str(b) for b in crc_bits), 2)

        # check crc
        header_data_bits = "".join(
            [
                str(bit)
                for bit in (length_bits + src_bits + dst_bits + mod_scheme_bits + coding_rate_bits + [0] * self.reserved_bits)
            ],
        )
        calculated_crc = self._crc_calc(header_data_bits)
        calculated_crc_bits = int_to_bits(calculated_crc, self.crc_bits)

        return FrameHeader(
            length,
            src,
            dst,
            mod_scheme,
            coding_rate,
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

    def _get_coding_rate_bits(self, header: np.ndarray) -> list[int]:
        """Extract modulation scheme field from header."""
        rate = header[self.idx : self.idx + self.coding_rate_bits]
        self.idx += self.coding_rate_bits
        return rate.tolist() if isinstance(rate, np.ndarray) else rate

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
            self.header_config.coding_rate_bits,
            self.header_config.crc_bits,
        )

        # Default LDPC config - can be overridden at encode/decode time
        self.ldpc = LDPC()
        self.golay = Golay()

    def encode(self, header: FrameHeader, payload: np.ndarray) -> np.ndarray:
        """Encode data into a frame."""
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        # find closest packet size to header.length for this coding rate
        payload_lengths = self.ldpc.get_supported_payload_lengths(header.coding_rate)
        closest_payload_length = next((l for l in sorted(payload_lengths) if l >= header.length), None)
        if closest_payload_length is None:
            raise ValueError(f"Unsupported payload length {header.length}. Greater than max length {payload_lengths[-1]}")
        payload_padded = np.concatenate([payload, np.zeros(closest_payload_length - header.length, dtype=int)])

        ldpc_config = LDPCConfig(k=closest_payload_length, code_rate=header.coding_rate)
        payload_encoded = self.ldpc.encode(payload_padded, ldpc_config)

        return np.concatenate([header_encoded, payload_encoded]) 

    def decode(self, _frame: np.ndarray) -> tuple[FrameHeader, np.ndarray]:
        """Decode a frame to extract data bits.

        Args:
            _frame: Input frame, either as hard decision bits (0/1) or LLR values.
                    LLR convention: positive = more likely 0, negative = more likely 1.
        """
        header_encoded = _frame[:self.header_config.header_total_size * 2]
        payload_encoded = _frame[self.header_config.header_total_size * 2:]

        # Detect if input is LLR (floats outside [0,1]) or hard bits
        is_llr = not np.all((np.abs(_frame) <= 1) | (np.isclose(_frame, 0)) | (np.isclose(_frame, 1)))

        # Golay uses hard decision - convert LLR to bits if needed
        if is_llr:
            header_hard = (header_encoded < 0).astype(int)
        else:
            header_hard = header_encoded.astype(int)

        header_bits = self.golay.decode(header_hard)
        header = self.frame_header_constructor.decode(header_bits)
        if header.crc_passed != True:
            raise ValueError("Header did not yield valid crc")

        # LDPC uses soft decision - convert bits to LLR if needed
        if is_llr:
            payload_llr = payload_encoded
        else:
            # Convert hard bits to high-confidence LLRs (0 -> +10, 1 -> -10)
            payload_llr = 10.0 * (1 - 2 * payload_encoded.astype(np.float64))

        # get closest payload length to header payload length for this coding rate
        payload_lengths = self.ldpc.get_supported_payload_lengths(header.coding_rate)
        closest_payload_length = next((l for l in sorted(payload_lengths) if l >= header.length), None)
        if closest_payload_length is None:
            raise ValueError(f"Unsupported payload length {header.length}. Greater than max length {payload_lengths[-1]}")

        ldpc_config = LDPCConfig(k=closest_payload_length, code_rate=header.coding_rate)
        payload_bits = self.ldpc.decode(payload_llr, ldpc_config)
        payload_bits = payload_bits[:header.length]

        return (header, payload_bits)
