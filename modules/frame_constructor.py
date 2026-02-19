"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .channel_coding import (
    CodeRates,
    Golay,
    LDPCConfig,
    ldpc_decode,
    ldpc_encode,
    ldpc_get_supported_payload_lengths,
)


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
    crc_bits: int = 4


def _bits_to_int(bits: list[int]) -> int:
    """Convert a bit list to an integer."""
    return int("".join(str(b) for b in bits), 2)


def _closest_payload_length(length: int, code_rate: CodeRates) -> int:
    """Find the smallest supported LDPC payload length >= length."""
    payload_lengths = ldpc_get_supported_payload_lengths(code_rate)
    result = next((k for k in sorted(payload_lengths) if k >= length), None)
    if result is None:
        msg = f"Unsupported payload length {length}. Greater than max length {payload_lengths[-1]}"
        raise ValueError(msg)
    return result


class FrameHeaderConstructor:
    """Encode and decode frame header fields."""

    def __init__(self, config: FrameHeaderConfig | None = None) -> None:
        """Initialize fixed bit widths for each frame header field."""
        self.config = config or FrameHeaderConfig()

        raw_length = (
            self.config.payload_length_bits
            + self.config.src_bits
            + self.config.dst_bits
            + self.config.mod_scheme_bits
            + self.config.coding_rate_bits
            + self.config.crc_bits
        )
        # Pad to multiple of 12 (Golay block size)
        self.header_length = 12 * int(np.ceil(raw_length / 12))
        self.reserved_bits = self.header_length - raw_length

    @staticmethod
    def _crc_calc(data_bits: list[int], poly: int = 0b10011) -> int:
        """Calculate CRC checksum from a bit list."""
        reg = 0
        for b in data_bits:
            reg = (reg << 1) | b
        reg <<= 4
        for i in range(len(data_bits) - 1, -1, -1):
            if reg & (1 << (i + 4)):
                reg ^= poly << i
        return reg & 0b1111

    def encode(self, header: FrameHeader) -> np.ndarray:
        """Encode frame header."""
        cfg = self.config
        length_bits = int_to_bits(header.length, cfg.payload_length_bits)
        src_bits = int_to_bits(header.src, cfg.src_bits)
        dst_bits = int_to_bits(header.dst, cfg.dst_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, cfg.mod_scheme_bits)
        coding_rate_bits = int_to_bits(header.coding_rate.value, cfg.coding_rate_bits)

        data_bits = length_bits + src_bits + dst_bits + mod_scheme_bits + coding_rate_bits + [0] * self.reserved_bits
        crc = self._crc_calc(data_bits)
        data_bits += int_to_bits(crc, cfg.crc_bits)

        return np.array(data_bits, dtype=int)

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        cfg = self.config
        offset = 0
        field_widths = [
            cfg.payload_length_bits,
            cfg.src_bits,
            cfg.dst_bits,
            cfg.mod_scheme_bits,
            cfg.coding_rate_bits,
        ]
        fields: list[list[int]] = []
        for width in field_widths:
            bits = header[offset : offset + width]
            fields.append(bits.tolist() if isinstance(bits, np.ndarray) else bits)
            offset += width

        length_bits, src_bits, dst_bits, mod_scheme_bits, coding_rate_bits = fields

        offset += self.reserved_bits
        crc_field = header[offset : offset + cfg.crc_bits]
        crc_bits = crc_field.tolist() if isinstance(crc_field, np.ndarray) else crc_field

        length = _bits_to_int(length_bits)
        src = _bits_to_int(src_bits)
        dst = _bits_to_int(dst_bits)
        mod_scheme = ModulationSchemes(_bits_to_int(mod_scheme_bits))
        coding_rate = CodeRates(_bits_to_int(coding_rate_bits))
        crc = _bits_to_int(crc_bits)

        # Verify CRC
        data_for_crc = length_bits + src_bits + dst_bits + mod_scheme_bits + coding_rate_bits + [0] * self.reserved_bits
        expected_crc_bits = int_to_bits(self._crc_calc(data_for_crc), cfg.crc_bits)

        return FrameHeader(
            length,
            src,
            dst,
            mod_scheme,
            coding_rate,
            crc,
            crc_passed=(crc_bits == expected_crc_bits),
        )


class FrameConstructor:
    """Build and parse frames based on a configured header format."""

    PAYLOAD_CRC_BITS = 16

    def __init__(
        self,
        header_config: FrameHeaderConfig | None = None,
    ) -> None:
        """Initialize frame construction parameters."""
        self.header_config = header_config or FrameHeaderConfig()
        self.frame_header_constructor = FrameHeaderConstructor(self.header_config)

        self.golay = Golay()

    @staticmethod
    def _crc16(data_bits: np.ndarray) -> int:
        """Compute CRC-16-CCITT over a bit array (poly 0x11021)."""
        reg = 0xFFFF
        for bit in data_bits:
            reg ^= int(bit) << 15
            reg = (((reg << 1) ^ 0x1021) & 0xFFFF) if (reg & 0x8000) else ((reg << 1) & 0xFFFF)
        return reg

    def encode(self, header: FrameHeader, payload: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode data into a frame."""
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        crc = self._crc16(payload)
        crc_bits = np.array(int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=int)
        payload_with_crc = np.concatenate([payload, crc_bits])

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        payload_padded = np.concatenate(
            [
                payload_with_crc,
                np.zeros(k - len(payload_with_crc), dtype=int),
            ],
        )

        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_encoded = ldpc_encode(payload_padded, ldpc_config)

        return (header_encoded, payload_encoded)

    def decode_header(self, header_encoded: np.ndarray) -> FrameHeader:
        """Decode a Golay-encoded header and verify CRC.

        Args:
            header_encoded: Golay-encoded header bits (hard decisions).

        """
        header_hard = header_encoded.astype(int)
        header_bits = self.golay.decode(header_hard)
        header = self.frame_header_constructor.decode(header_bits)
        if not header.crc_passed:
            msg = "Header did not yield valid crc"
            raise ValueError(msg)
        return header

    def decode_payload(
        self,
        header: FrameHeader,
        payload_encoded: np.ndarray,
        *,
        soft: bool = False,
    ) -> np.ndarray:
        """Decode an LDPC-encoded payload using parameters from a decoded header.

        Args:
            header: Already-decoded frame header (from decode_header).
            payload_encoded: LDPC-encoded payload (hard bits or LLR values).
            soft: If True, treat payload as LLR values. If False (default),
                treat as hard bits and convert to LLR.

        """
        payload_llr = payload_encoded if soft else 10.0 * (1 - 2 * payload_encoded.astype(np.float64))

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_bits = ldpc_decode(payload_llr, ldpc_config)

        data_bits = payload_bits[: header.length]
        crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits.astype(int).tolist())
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)

        return data_bits
