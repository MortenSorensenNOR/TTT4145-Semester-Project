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
    crc: int = 0
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


def _bits_to_int(bits: list[int]) -> int:
    """Convert a bit list to an integer."""
    return int("".join(str(b) for b in bits), 2)


def _closest_payload_length(ldpc: LDPC, length: int, code_rate: CodeRates) -> int:
    """Find the smallest supported LDPC payload length >= length."""
    payload_lengths = ldpc.get_supported_payload_lengths(code_rate)
    result = next((k for k in sorted(payload_lengths) if k >= length), None)
    if result is None:
        raise ValueError(
            f"Unsupported payload length {length}. "
            f"Greater than max length {payload_lengths[-1]}"
        )
    return result


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

        raw_length = (
            self.length_bits
            + self.src_bits
            + self.dst_bits
            + self.mod_scheme_bits
            + self.coding_rate_bits
            + self.crc_bits
        )
        self.header_length = 2 * int(np.ceil(raw_length / 2))
        self.reserved_bits = self.header_length - raw_length

    def _crc_calc(self, data_bits: str, poly: int = 0b10011) -> int:
        """Calculate CRC checksum."""
        reg = int(data_bits, 2) << 4
        for i in range(len(data_bits) - 1, -1, -1):
            if reg & (1 << (i + 4)):
                reg ^= poly << i
        return reg & 0b1111

    @staticmethod
    def _extract_field(header: np.ndarray, offset: int, width: int) -> list[int]:
        """Extract a bit field from the header array."""
        bits = header[offset : offset + width]
        return bits.tolist() if isinstance(bits, np.ndarray) else bits

    def encode(self, header: FrameHeader) -> np.ndarray:
        """Encode frame header."""
        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, self.mod_scheme_bits)
        coding_rate_bits = int_to_bits(header.coding_rate.value, self.coding_rate_bits)

        data_bits = (
            length_bits + src_bits + dst_bits
            + mod_scheme_bits + coding_rate_bits
            + [0] * self.reserved_bits
        )
        crc = self._crc_calc("".join(str(b) for b in data_bits))
        data_bits += int_to_bits(crc, self.crc_bits)

        return np.array(data_bits, dtype=int)

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        offset = 0
        field_widths = [
            self.length_bits, self.src_bits, self.dst_bits,
            self.mod_scheme_bits, self.coding_rate_bits,
        ]
        fields: list[list[int]] = []
        for width in field_widths:
            fields.append(self._extract_field(header, offset, width))
            offset += width

        length_bits, src_bits, dst_bits, mod_scheme_bits, coding_rate_bits = fields

        offset += self.reserved_bits
        crc_bits = self._extract_field(header, offset, self.crc_bits)

        length = _bits_to_int(length_bits)
        src = _bits_to_int(src_bits)
        dst = _bits_to_int(dst_bits)
        mod_scheme = ModulationSchemes(_bits_to_int(mod_scheme_bits))
        coding_rate = CodeRates(_bits_to_int(coding_rate_bits))
        crc = _bits_to_int(crc_bits)

        # Verify CRC
        data_str = "".join(
            str(b) for b in (
                length_bits + src_bits + dst_bits
                + mod_scheme_bits + coding_rate_bits
                + [0] * self.reserved_bits
            )
        )
        expected_crc_bits = int_to_bits(self._crc_calc(data_str), self.crc_bits)

        return FrameHeader(
            length, src, dst,
            mod_scheme, coding_rate, crc,
            crc_passed=(crc_bits == expected_crc_bits),
        )


class FrameConstructor:
    """Build and parse frames based on a configured header format."""

    def __init__(
        self,
        header_config: FrameHeaderConfig | None = None,
    ) -> None:
        """Initialize frame construction parameters."""
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

    def encode(self, header: FrameHeader, payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode data into a frame.

        Returns:
            Tuple of (header_encoded, payload_encoded). These are returned separately
            because the header should always be modulated with BPSK, while the payload
            uses a variable modulation scheme specified in the header.
        """
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        k = _closest_payload_length(self.ldpc, header.length, header.coding_rate)
        payload_padded = np.concatenate([
            payload, np.zeros(k - header.length, dtype=int),
        ])

        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_encoded = self.ldpc.encode(payload_padded, ldpc_config)

        return (header_encoded, payload_encoded)

    def decode(
        self,
        header_encoded: np.ndarray,
        payload_encoded: np.ndarray,
    ) -> tuple[FrameHeader, np.ndarray]:
        """Decode a frame to extract data bits.

        Args:
            header_encoded: Encoded header bits (hard decision 0/1 only, since
                           header is always BPSK modulated).
            payload_encoded: Encoded payload, either as hard decision bits (0/1)
                            or LLR values. LLR convention: positive = more likely 0,
                            negative = more likely 1.
        """
        # Golay uses hard decision
        header_hard = header_encoded.astype(int)
        header_bits = self.golay.decode(header_hard)
        header = self.frame_header_constructor.decode(header_bits)
        if not header.crc_passed:
            raise ValueError("Header did not yield valid crc")

        # Detect if payload input is LLR (floats outside [0,1]) or hard bits
        is_llr = not np.all(
            (np.abs(payload_encoded) <= 1)
            | (np.isclose(payload_encoded, 0))
            | (np.isclose(payload_encoded, 1))
        )

        # LDPC uses soft decision - convert bits to LLR if needed
        if is_llr:
            payload_llr = payload_encoded
        else:
            # Convert hard bits to high-confidence LLRs (0 -> +10, 1 -> -10)
            payload_llr = 10.0 * (1 - 2 * payload_encoded.astype(np.float64))

        k = _closest_payload_length(self.ldpc, header.length, header.coding_rate)
        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_bits = self.ldpc.decode(payload_llr, ldpc_config)

        return (header, payload_bits[:header.length])
