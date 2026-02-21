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
    ldpc_max_k,
    ldpc_max_n,
)


def int_to_bits(n: int, length: int) -> list[int]:
    """Convert an integer to a fixed-width big-endian bit list."""
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]


class ModulationSchemes(Enum):
    """Supported modulation schemes."""

    BPSK = 0
    QPSK = 1
    QAM16 = 2
    QAM64 = 3


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
    """Bit-width configuration for frame header fields.

    Total header size must be a multiple of 12 for Golay encoding.
    """

    payload_length_bits: int = 14  # max ~16K bits payload for multi-block LDPC
    src_bits: int = 2
    dst_bits: int = 2
    frame_type_bits: int = 2
    mod_scheme_bits: int = 2
    coding_rate_bits: int = 3
    sequence_number_bits: int = 4
    reserved_bits: int = 11  # padding to make total 48 bits (4 Golay blocks)
    crc_bits: int = 8
    header_total_size: int = field(init=False)

    def __post_init__(self):
        self.header_total_size = (
            self.payload_length_bits +
            self.src_bits +
            self.dst_bits +
            self.frame_type_bits +
            self.mod_scheme_bits +
            self.coding_rate_bits +
            self.sequence_number_bits +
            self.reserved_bits +
            self.crc_bits
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

    def __init__(
        self,
        config: FrameHeaderConfig
    ) -> None:
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
            byte = int(padded[i:i+8], 2)
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
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
            self.sequence_number_bits
        ]
        fields: list[list[int]] = []
        for width in field_widths:
            bits = header[offset : offset + width]
            fields.append(bits.tolist() if isinstance(bits, np.ndarray) else bits)
            offset += width

        (length_bits, 
         src_bits, 
         dst_bits, 
         frame_type_bits, 
         mod_scheme_bits, 
         coding_rate_bits, 
         sequence_number_bits) = fields

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
            str(b) for b in (
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
    MAX_LDPC_BLOCKS = 8  # Max blocks per frame (limits frame length)

    def __init__(
        self,
        header_config: FrameHeaderConfig | None = None,
    ) -> None:
        """Initialize frame construction parameters."""
        self.header_config = header_config or FrameHeaderConfig()
        self.frame_header_constructor = FrameHeaderConstructor(self.header_config)

        self.golay = Golay()

    def _num_ldpc_blocks(self, payload_bits: int, code_rate: CodeRates) -> int:
        """Compute number of LDPC blocks needed for the payload."""
        max_k = ldpc_max_k(code_rate)
        return int(np.ceil(payload_bits / max_k))

    def max_payload_bits(self, code_rate: CodeRates) -> int:
        """Return max payload bits (excluding CRC) for multi-block LDPC."""
        max_k = ldpc_max_k(code_rate)
        return self.MAX_LDPC_BLOCKS * max_k - self.PAYLOAD_CRC_BITS

    @property
    def header_encoded_n_bits(self) -> int:
        """Number of bits in the Golay-encoded header (= number of BPSK symbols)."""
        golay_ratio = self.golay.block_length // self.golay.message_length
        return self.frame_header_constructor.header_length * golay_ratio

    def payload_coded_n_bits(self, header: FrameHeader, *, channel_coding: bool = True) -> int:
        """Return the number of coded payload bits for a given header.

        For multi-block LDPC, this returns the total bits across all blocks.
        """
        if not channel_coding or header.coding_rate == CodeRates.NONE:
            raw = header.length + self.PAYLOAD_CRC_BITS
            return raw + (-raw % 12)  # pad to multiple of 12

        payload_with_crc = header.length + self.PAYLOAD_CRC_BITS
        max_k = ldpc_max_k(header.coding_rate)
        max_n = ldpc_max_n(header.coding_rate)

        # Check if single block suffices
        if payload_with_crc <= max_k:
            k = _closest_payload_length(payload_with_crc, header.coding_rate)
            return LDPCConfig(k=k, code_rate=header.coding_rate).n

        # Multi-block: use max-size blocks for all
        n_blocks = self._num_ldpc_blocks(payload_with_crc, header.coding_rate)
        return n_blocks * max_n

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
        """Encode data into a frame.

        For payloads exceeding a single LDPC block, uses multi-block encoding
        with the largest available block size (n=1944).
        """
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        crc = self._crc16(payload)
        crc_bits = np.array(int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=int)
        payload_with_crc = np.concatenate([payload, crc_bits])

        if not channel_coding or header.coding_rate == CodeRates.NONE:
            n = self.payload_coded_n_bits(header, channel_coding=False)
            payload_encoded = np.concatenate(
                [payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=int)]
            )
            return (header_encoded, payload_encoded)

        max_k = ldpc_max_k(header.coding_rate)

        # Single block case
        if len(payload_with_crc) <= max_k:
            k = _closest_payload_length(len(payload_with_crc), header.coding_rate)
            payload_padded = np.concatenate(
                [payload_with_crc, np.zeros(k - len(payload_with_crc), dtype=int)]
            )
            ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
            payload_encoded = ldpc_encode(payload_padded, ldpc_config)
            if interleaving:
                payload_encoded = interleave(payload_encoded, ldpc_config.n)
            return (header_encoded, payload_encoded)

        # Multi-block: split into max_k chunks, encode each
        n_blocks = self._num_ldpc_blocks(len(payload_with_crc), header.coding_rate)
        ldpc_config = LDPCConfig(k=max_k, code_rate=header.coding_rate)

        # Pad payload to exact multiple of max_k
        total_bits = n_blocks * max_k
        payload_padded = np.concatenate(
            [payload_with_crc, np.zeros(total_bits - len(payload_with_crc), dtype=int)]
        )

        # Encode each block
        encoded_blocks = []
        for i in range(n_blocks):
            block = payload_padded[i * max_k : (i + 1) * max_k]
            encoded = ldpc_encode(block, ldpc_config)
            if interleaving:
                encoded = interleave(encoded, ldpc_config.n)
            encoded_blocks.append(encoded)

        payload_encoded = np.concatenate(encoded_blocks)
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
        """Decode an LDPC-encoded payload using parameters from a decoded header.

        Handles multi-block LDPC payloads by decoding each block separately.
        """
        if not channel_coding or header.coding_rate == CodeRates.NONE:
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

        payload_with_crc_len = header.length + self.PAYLOAD_CRC_BITS
        max_k = ldpc_max_k(header.coding_rate)
        max_n = ldpc_max_n(header.coding_rate)

        # Single block case
        if payload_with_crc_len <= max_k:
            k = _closest_payload_length(payload_with_crc_len, header.coding_rate)
            ldpc_config = LDPCConfig(k=k, code_rate=header.coding_rate)
            if interleaving:
                payload_llr = deinterleave(payload_llr, ldpc_config.n)
            payload_bits = ldpc_decode(payload_llr, ldpc_config)
        else:
            # Multi-block decoding
            n_blocks = self._num_ldpc_blocks(payload_with_crc_len, header.coding_rate)
            ldpc_config = LDPCConfig(k=max_k, code_rate=header.coding_rate)

            decoded_blocks = []
            for i in range(n_blocks):
                block_llr = payload_llr[i * max_n : (i + 1) * max_n]
                if interleaving:
                    block_llr = deinterleave(block_llr, max_n)
                decoded = ldpc_decode(block_llr, ldpc_config)
                decoded_blocks.append(decoded)

            payload_bits = np.concatenate(decoded_blocks)

        data_bits = payload_bits[: header.length]
        crc_bits = payload_bits[header.length : header.length + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits.astype(int).tolist())
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)

        return data_bits
