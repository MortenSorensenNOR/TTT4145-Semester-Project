"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any # Added for np.ndarray type hints

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


def int_to_bits(n: np.int32, length: int) -> list[np.int32]: # Changed type hint
    """Convert an integer to a fixed-width big-endian bit list."""
    return [np.int32((n >> (length - 1 - i)) & 1) for i in range(length)] # Ensure np.int32


class ModulationSchemes(Enum):
    """Supported modulation schemes."""

    BPSK = 0
    QPSK = 1
    QAM16 = 2
    PSK8 = 3


@dataclass
class FrameHeader:
    """Frame header with metadata."""

    length: np.int32 # Changed type hint
    src: np.int32 # Changed type hint
    dst: np.int32 # Changed type hint
    frame_type: np.int32 # Changed type hint
    mod_scheme: ModulationSchemes
    coding_rate: CodeRates
    sequence_number: np.int32 # Changed type hint
    crc: np.int32 = np.int32(0) # Changed type hint and default
    crc_passed: bool = True


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""

    payload_length_bits: np.int32 = np.int32(12)  # length in bits # Changed type hint and default
    src_bits: np.int32 = np.int32(2) # Changed type hint and default
    dst_bits: np.int32 = np.int32(2) # Changed type hint and default
    frame_type_bits: np.int32 = np.int32(2) # Changed type hint and default
    mod_scheme_bits: np.int32 = np.int32(2) # Changed type hint and default
    coding_rate_bits: np.int32 = np.int32(3) # Changed type hint and default
    sequence_number_bits: np.int32 = np.int32(4) # Changed type hint and default
    reserved_bits: np.int32 = np.int32(1) # Changed type hint and default
    crc_bits: np.int32 = np.int32(8) # Changed type hint and default
    header_total_size: np.int32 = field(init=False) # Changed type hint

    def __post_init__(self) -> None:
        """Compute the total header size from all field widths."""
        self.header_total_size = np.int32( # Ensure np.int32
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


def _bits_to_int(bits: list[np.int32]) -> np.int32: # Changed type hint
    """Convert a bit list to an integer."""
    return np.int32(int("".join(str(b) for b in bits), 2)) # Ensure np.int32


def _closest_payload_length(length: np.int32, code_rate: CodeRates) -> np.int32: # Changed type hint
    """Find the smallest supported LDPC payload length >= length."""
    payload_lengths = ldpc_get_supported_payload_lengths(code_rate).astype(np.int32)
    result = next((k for k in sorted(payload_lengths) if k >= length), None)
    if result is None:
        msg = f"Unsupported payload length {length}. Greater than max length {payload_lengths[-1]}"
        raise ValueError(msg)
    return np.int32(result) # Ensure np.int32


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

        raw_length = config.header_total_size # Fixed: was self.header_config.header_total_size
        self.header_length = np.int32(2 * np.ceil(np.float32(raw_length) / np.float32(2))) # Ensure np.int32 and np.float32

    def _crc_calc(self, data_bits: str, poly: np.int32 = np.int32(0x07)) -> np.int32: # Changed type hint
        """Calculate CRC checksum."""
        padded = data_bits.zfill((len(data_bits) + np.int32(7)) // np.int32(8) * np.int32(8)) # Ensure np.int32
        crc = np.int32(0x00) # Ensure np.int32
        for i in range(np.int32(0), len(padded), np.int32(8)): # Ensure np.int32
            byte = np.int32(int(padded[i : i + np.int32(8)], 2)) # Ensure np.int32
            crc ^= byte
            for _ in range(np.int32(8)): # Ensure np.int32
                crc = np.int32((crc << np.int32(1) ^ poly) & np.int32(255)) if np.int32(crc) & np.int32(128) else np.int32(crc << np.int32(1)) & np.int32(255) # Ensure np.int32
        return crc

    def encode(self, header: FrameHeader) -> np.ndarray[np.int32, Any]: # Added return type hint
        """Encode frame header."""
        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        frame_type_bits = int_to_bits(header.frame_type, self.frame_type_bits)
        mod_scheme_bits = int_to_bits(np.int32(header.mod_scheme.value), self.mod_scheme_bits) # Ensure np.int32
        coding_rate_bits = int_to_bits(np.int32(header.coding_rate.value), self.coding_rate_bits) # Ensure np.int32
        sequence_number_bits = int_to_bits(header.sequence_number, self.sequence_number_bits)

        data_bits = (
            length_bits
            + src_bits
            + dst_bits
            + frame_type_bits
            + mod_scheme_bits
            + coding_rate_bits
            + sequence_number_bits
            + [np.int32(0)] * self.reserved_bits # Ensure np.int32
        )
        crc = self._crc_calc("".join(str(b) for b in data_bits))
        crc_int_bits = int_to_bits(crc, self.crc_bits) # Use a new variable name
        data_bits += crc_int_bits

        return np.array(data_bits, dtype=np.int32)

    def decode(self, header: np.ndarray) -> FrameHeader: # Added return type hint
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
        fields: list[list[np.int32]] = [] # Changed type hint
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
                + [np.int32(0)] * self.reserved_bits # Ensure np.int32
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

    PAYLOAD_CRC_BITS: np.int32 = np.int32(16) # Changed type hint and default
    PAYLOAD_PAD_MULTIPLE: np.int32 = np.int32(12) # Changed type hint and default

    def __init__(
        self,
        header_config: FrameHeaderConfig | None = None,
    ) -> None:
        """Initialize frame construction parameters."""
        self.header_config = header_config or FrameHeaderConfig()
        self.frame_header_constructor = FrameHeaderConstructor(self.header_config)

        self.golay = Golay()

    @property
    def header_encoded_n_bits(self) -> np.int32: # Changed return type hint
        """Number of bits in the Golay-encoded header (= number of BPSK symbols)."""
        golay_ratio = np.int32(self.golay.block_length // self.golay.message_length) # Ensure np.int32
        return self.frame_header_constructor.header_length * golay_ratio

    def payload_coded_n_bits(self, header: FrameHeader, *, channel_coding: bool = True) -> np.int32: # Changed return type hint
        """Return the number of coded payload bits for a given header."""
        if not channel_coding or header.coding_rate == CodeRates.NONE:
            raw = header.length + self.PAYLOAD_CRC_BITS
            return np.int32(raw + (-raw % self.PAYLOAD_PAD_MULTIPLE)) # Ensure np.int32
        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        return np.int32(LDPCConfig(k=k, code_rate=header.coding_rate).n) # Ensure np.int32

    @staticmethod
    def _crc16(data_bits: np.ndarray[np.int32, Any]) -> np.int32: # Changed type hint and return type hint
        """Compute CRC-16-CCITT over a bit array (poly 0x1021).

        Source: https://en.wikipedia.org/wiki/Cyclic_redundancy_check#CRC-16-CCITT
        """
        n = FrameConstructor.PAYLOAD_CRC_BITS
        mask = np.int32((np.int32(1) << n) - np.int32(1)) # Ensure np.int32
        msb = np.int32(np.int32(1) << (n - np.int32(1))) # Ensure np.int32
        reg = np.int32(mask) # Ensure np.int32
        for bit in data_bits:
            reg ^= np.int32(bit) << (n - np.int32(1)) # Ensure np.int32
            reg = np.int32(((reg << np.int32(1)) ^ np.int32(0x1021)) & mask) if np.int32(reg) & msb else np.int32((reg << np.int32(1)) & mask) # Ensure np.int32
        return reg

    def encode(
        self,
        header: FrameHeader,
        payload: np.ndarray[np.int32, Any], # Changed type hint
        *,
        channel_coding: bool = True,
        interleaving: bool = True,
    ) -> tuple[np.ndarray[np.int32, Any], np.ndarray[np.int32, Any]]: # Added return type hint
        """Encode data into a frame."""
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        crc = self._crc16(payload)
        crc_bits = np.array(int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=np.int32)
        payload_with_crc = np.concatenate([payload, crc_bits])

        if not channel_coding or header.coding_rate == CodeRates.NONE:
            n = self.payload_coded_n_bits(header, channel_coding=False)
            payload_encoded = np.concatenate([payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=np.int32)])
            return (header_encoded, payload_encoded)

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        payload_padded = np.concatenate(
            [
                payload_with_crc,
                np.zeros(k - len(payload_with_crc), dtype=np.int32),
            ],
        )

        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        payload_encoded = ldpc_encode(payload_padded, ldpc_config)
        if interleaving:
            payload_encoded = interleave(payload_encoded, ldpc_config.n)

        return (header_encoded, payload_encoded)

    def decode_header(self, header_encoded: np.ndarray) -> FrameHeader:
        """Decode a Golay-encoded header and verify CRC."""
        header_hard = header_encoded.astype(np.int32)
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
    ) -> np.ndarray[np.int32, Any]: # Added return type hint
        """Decode an LDPC-encoded payload using parameters from a decoded header."""
        if not channel_coding or header.coding_rate == CodeRates.NONE:
            payload_bits = (payload_encoded < np.float32(0)).astype(np.int32) if soft else payload_encoded.astype(np.int32) # Explicitly cast to np.float32
            data_bits = payload_bits[: header.length]
            crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
            received_crc = _bits_to_int(crc_bits.astype(np.int32).tolist())
            expected_crc = self._crc16(data_bits)
            if received_crc != expected_crc:
                msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
                raise ValueError(msg)
            return data_bits

        payload_llr = payload_encoded if soft else np.float32(10.0) * (np.float32(1) - np.float32(2) * payload_encoded.astype(np.float32))

        k = _closest_payload_length(header.length + self.PAYLOAD_CRC_BITS, header.coding_rate)
        ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
        if interleaving:
            payload_llr = deinterleave(payload_llr, ldpc_config.n)
        payload_bits = ldpc_decode(payload_llr, ldpc_config)

        data_bits = payload_bits[: header.length]
        crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits.astype(np.int32).tolist())
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)

        return data_bits