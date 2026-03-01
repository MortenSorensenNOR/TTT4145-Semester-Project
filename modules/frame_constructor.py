"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass, field
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


def int_to_bits(n: int, length: int) -> list[int]:
    """Convert an integer to a fixed-width big-endian bit list."""
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]


class ModulationSchemes(Enum):
    """Supported modulation schemes."""

    BPSK = 0
    QPSK = 1
    QAM16 = 2
    PSK8 = 3


@dataclass
class FrameHeader:
    """Frame header with metadata."""

    length: int
    src: int
    dst: int
    frame_type: int
    mod_scheme: ModulationSchemes
    coding_rate: CodeRates
    sequence_number: int
    crc: int = 0
    crc_passed: bool = True


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""

    payload_length_bits: int = 12  # length in bits
    src_bits: int = 2
    dst_bits: int = 2
    frame_type_bits: int = 2
    mod_scheme_bits: int = 2
    coding_rate_bits: int = 3
    sequence_number_bits: int = 4
    reserved_bits: int = 1
    crc_bits: int = 8
    header_total_size: int = field(init=False)

    def __post_init__(self) -> None:
        """Compute the total header size from all field widths."""
        self.header_total_size = (
            self.payload_length_bits
            + self.src_bits
            + self.dst_bits
            + self.frame_type_bits
            + self.mod_scheme_bits
            + self.coding_rate_bits
            + self.sequence_number_bits
            + self.reserved_bits
            + self.crc_bits
        )


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

    def __init__(self, config: FrameHeaderConfig) -> None:
        """Initialize fixed bit widths for each frame header field."""
        self.length_bits = config.payload_length_bits
        self.src_bits = config.src_bits
        self.dst_bits = config.dst_bits
        self.frame_type_bits = config.frame_type_bits
        self.mod_scheme_bits = config.mod_scheme_bits
        self.coding_rate_bits = config.coding_rate_bits
        self.sequence_number_bits = config.sequence_number_bits
        self.reserved_bits = config.reserved_bits
        self.crc_bits = config.crc_bits

        raw_length = config.header_total_size
        self.header_length = 2 * int(np.ceil(raw_length / 2))

    def _crc_calc(self, data_bits: str, poly: int = 0x07) -> int:
        """Calculate CRC checksum."""
        padded = data_bits.zfill((len(data_bits) + 7) // 8 * 8)
        crc = 0x00
        for i in range(0, len(padded), 8):
            byte = int(padded[i : i + 8], 2)
            crc ^= byte
            for _ in range(8):
                crc = (crc << 1 ^ poly) & 255 if crc & 128 else crc << 1 & 255
        return crc

    def encode(self, header: FrameHeader) -> np.ndarray:
        """Encode frame header."""
        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        frame_type_bits = int_to_bits(header.frame_type, self.frame_type_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, self.mod_scheme_bits)
        coding_rate_bits = int_to_bits(header.coding_rate.value, self.coding_rate_bits)
        sequence_number_bits = int_to_bits(header.sequence_number, self.sequence_number_bits)

        data_bits = (
            length_bits
            + src_bits
            + dst_bits
            + frame_type_bits
            + mod_scheme_bits
            + coding_rate_bits
            + sequence_number_bits
            + [0] * self.reserved_bits
        )
        crc = self._crc_calc("".join(str(b) for b in data_bits))
        data_bits += int_to_bits(crc, self.crc_bits)

        return np.array(data_bits, dtype=int)

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        offset = 0
        field_widths = [
            self.length_bits,
            self.src_bits,
            self.dst_bits,
            self.frame_type_bits,
            self.mod_scheme_bits,
            self.coding_rate_bits,
            self.sequence_number_bits,
        ]
        fields: list[list[int]] = []
        for width in field_widths:
            bits = header[offset : offset + width]
            fields.append(bits.tolist() if isinstance(bits, np.ndarray) else bits)
            offset += width

        (length_bits, src_bits, dst_bits, frame_type_bits, mod_scheme_bits, coding_rate_bits, sequence_number_bits) = (
            fields
        )

        offset += self.reserved_bits
        crc_field = header[offset : offset + self.crc_bits]
        crc_bits = crc_field.tolist() if isinstance(crc_field, np.ndarray) else crc_field

        length = _bits_to_int(length_bits)
        src = _bits_to_int(src_bits)
        dst = _bits_to_int(dst_bits)
        frame_type = _bits_to_int(frame_type_bits)
        mod_scheme = ModulationSchemes(_bits_to_int(mod_scheme_bits))
        coding_rate = CodeRates(_bits_to_int(coding_rate_bits))
        sequence_number = _bits_to_int(sequence_number_bits)
        crc = _bits_to_int(crc_bits)

        # Verify CRC
        data_str = "".join(
            str(b)
            for b in (
                length_bits
                + src_bits
                + dst_bits
                + frame_type_bits
                + mod_scheme_bits
                + coding_rate_bits
                + sequence_number_bits
                + [0] * self.reserved_bits
            )
        )
        expected_crc_bits = int_to_bits(self._crc_calc(data_str), self.crc_bits)

        return FrameHeader(
            length,
            src,
            dst,
            frame_type,
            mod_scheme,
            coding_rate,
            sequence_number,
            crc,
            crc_passed=(crc_bits == expected_crc_bits),
        )


class FrameConstructor:
    """Build and parse frames based on a configured header format."""

    PAYLOAD_CRC_BITS = 16
    PAYLOAD_PAD_MULTIPLE = 12

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
        if not channel_coding or header.coding_rate == CodeRates.NONE:
            raw = header.length + self.PAYLOAD_CRC_BITS
            return raw + (-raw % self.PAYLOAD_PAD_MULTIPLE)
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
        crc_bits = np.array(int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=int)
        payload_with_crc = np.concatenate([payload, crc_bits])

        if not channel_coding or header.coding_rate == CodeRates.NONE:
            n = self.payload_coded_n_bits(header, channel_coding=False)
            payload_encoded = np.concatenate([payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=int)])
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
        if not channel_coding or header.coding_rate == CodeRates.NONE:
            payload_bits = (payload_encoded < 0).astype(int) if soft else payload_encoded.astype(int)
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
