"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from modules.channel_coding import (
    CodeRates,
    Golay,
    LDPCConfig,
    deinterleave,
    interleave,
    ldpc_decode,
    ldpc_encode,
    ldpc_get_supported_payload_lengths,
)


def _int_to_bits(n: int, length: int) -> list[int]:
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
        """Calculate CRC checksum from a bit list.

        Source: https://en.wikipedia.org/wiki/Cyclic_redundancy_check#Computation
        """
        crc_width = poly.bit_length() - 1
        reg = 0
        for b in data_bits:
            reg = (reg << 1) | b
        reg <<= crc_width
        for i in range(len(data_bits) - 1, -1, -1):
            if reg & (1 << (i + crc_width)):
                reg ^= poly << i
        return reg & ((1 << crc_width) - 1)

    def encode(self, header: FrameHeader) -> np.ndarray:
        """Encode frame header."""
        cfg = self.config
        length_bits = _int_to_bits(header.length, cfg.payload_length_bits)
        src_bits = _int_to_bits(header.src, cfg.src_bits)
        dst_bits = _int_to_bits(header.dst, cfg.dst_bits)
        mod_scheme_bits = _int_to_bits(header.mod_scheme.value, cfg.mod_scheme_bits)
        coding_rate_bits = _int_to_bits(header.coding_rate.value, cfg.coding_rate_bits)

        data_bits = length_bits + src_bits + dst_bits + mod_scheme_bits + coding_rate_bits + [0] * self.reserved_bits
        crc = self._crc_calc(data_bits)
        data_bits += _int_to_bits(crc, cfg.crc_bits)

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
        expected_crc_bits = _int_to_bits(self._crc_calc(data_for_crc), cfg.crc_bits)

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

    @property
    def header_encoded_n_bits(self) -> int:
        """Number of bits in the Golay-encoded header (= number of BPSK symbols)."""
        golay_ratio = self.golay.block_length // self.golay.message_length
        return self.frame_header_constructor.header_length * golay_ratio

    def payload_coded_n_bits(self, header: FrameHeader, *, channel_coding: bool = True) -> int:
        """Return the number of coded payload bits for a given header."""
        if not channel_coding:
            raw = header.length + self.PAYLOAD_CRC_BITS
            return raw + (-raw % 12)  # pad to multiple of 12
        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        return LDPCConfig(k=k, code_rate=header.coding_rate).n

    @staticmethod
    def _crc16(data_bits: np.ndarray) -> int:
        """Compute CRC-16-CCITT over a bit array (poly 0x1021).

        Source: https://en.wikipedia.org/wiki/Cyclic_redundancy_check#CRC-16-CCITT
        """
        n = FrameConstructor.PAYLOAD_CRC_BITS
        mask = (1 << n) - 1
        msb = 1 << (n - 1)
        reg = mask
        for bit in data_bits:
            reg ^= int(bit) << (n - 1)
            reg = (((reg << 1) ^ 0x1021) & mask) if (reg & msb) else ((reg << 1) & mask)
        return reg

    def encode(
        self,
        header: FrameHeader,
        payload: np.ndarray,
        *,
        channel_coding: bool = True,
        interleaving: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode data into a frame."""
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        crc = self._crc16(payload)
        crc_bits = np.array(_int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=int)
        payload_with_crc = np.concatenate([payload, crc_bits])

        if not channel_coding:
            n = self.payload_coded_n_bits(header, channel_coding=False)
            payload_encoded = np.concatenate(
                [payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=int)]
            )
            return (header_encoded, payload_encoded)

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        payload_padded = np.concatenate(
            [
                payload_with_crc,
                np.zeros(k - len(payload_with_crc), dtype=int),
            ],
        )

        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_encoded = ldpc_encode(payload_padded, ldpc_config)
        if interleaving:
            payload_encoded = interleave(payload_encoded, ldpc_config.n)

        return (header_encoded, payload_encoded)

    def decode_header(self, header_encoded: np.ndarray) -> FrameHeader:
        """Decode a Golay-encoded header and verify CRC."""
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
        channel_coding: bool = True,
        interleaving: bool = True,
    ) -> np.ndarray:
        """Decode an LDPC-encoded payload using parameters from a decoded header."""
        if not channel_coding:
            if soft:
                payload_bits = (payload_encoded < 0).astype(int)
            else:
                payload_bits = payload_encoded.astype(int)
            data_bits = payload_bits[: header.length]
            crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
            received_crc = _bits_to_int(crc_bits.astype(int).tolist())
            expected_crc = self._crc16(data_bits)
            if received_crc != expected_crc:
                msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
                raise ValueError(msg)
            return data_bits

        payload_llr = payload_encoded if soft else 10.0 * (1 - 2 * payload_encoded.astype(np.float64))

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        if interleaving:
            payload_llr = deinterleave(payload_llr, ldpc_config.n)
        payload_bits = ldpc_decode(payload_llr, ldpc_config)

        data_bits = payload_bits[: header.length]
        crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits.astype(int).tolist())
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)

        return data_bits
