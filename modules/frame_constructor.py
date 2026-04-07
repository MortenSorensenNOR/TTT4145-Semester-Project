"""Construct frames with header data, pilots, and error correction information."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from modules.golay import Golay

def int_to_bits(n: int, length: int) -> np.ndarray:
    return np.array([(n >> (length - 1 - i)) & 1 for i in range(length)], dtype=np.int8)

class ModulationSchemes(Enum):
    """Supported modulation schemes."""
    BPSK = 0
    QPSK = 1
    PSK8 = 2


@dataclass
class FrameHeader:
    """Frame header with metadata."""
    length: int # payload length in bytes
    src: int
    dst: int
    frame_type: int
    mod_scheme: ModulationSchemes
    sequence_number: int
    crc: int = field(default=0, compare=False) # just used for comparing in tests.
    crc_passed: bool = True


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""
    payload_length_bits: int = 12  # length in bits
    src_bits: int = 2
    dst_bits: int = 2
    frame_type_bits: int = 2
    mod_scheme_bits: int = 2
    sequence_number_bits: int = 4
    reserved_bits: int = 4
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
            + self.sequence_number_bits
            + self.reserved_bits
            + self.crc_bits
        )

def _bits_to_int(bits) -> int:
    arr = np.asarray(bits, dtype=np.int32)
    powers = 1 << np.arange(len(arr) - 1, -1, -1, dtype=np.int32)
    return int(arr @ powers)

class FrameHeaderConstructor:
    """Encode and decode frame header fields."""

    def __init__(self, config: FrameHeaderConfig) -> None:
        self.length_bits = config.payload_length_bits
        self.src_bits = config.src_bits
        self.dst_bits = config.dst_bits
        self.frame_type_bits = config.frame_type_bits
        self.mod_scheme_bits = config.mod_scheme_bits
        self.sequence_number_bits = config.sequence_number_bits
        self.reserved_bits = config.reserved_bits
        self.crc_bits = config.crc_bits
        raw_length = config.header_total_size
        self.header_length = 2 * int(np.ceil(raw_length / 2))

        # Precompute slice boundaries once
        widths = [
            self.length_bits,
            self.src_bits,
            self.dst_bits,
            self.frame_type_bits,
            self.mod_scheme_bits,
            self.sequence_number_bits,
        ]
        self._offsets = np.concatenate([[0], np.cumsum(widths)]).astype(int)
        # where the CRC field starts (skip reserved bits)
        self._crc_start = int(self._offsets[-1]) + self.reserved_bits

    def _crc_calc(self, data_bits: np.ndarray, poly: int = 0x07) -> int:
        """Calculate CRC checksum over a bit array."""
        bits = np.asarray(data_bits, dtype=np.uint8)
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        crc = 0x00
        for i in range(0, len(bits), 8):
            byte = int(np.packbits(bits[i:i+8])[0])
            crc ^= byte
            for _ in range(8):
                crc = (crc << 1 ^ poly) & 255 if crc & 128 else crc << 1 & 255
        return crc

    def encode(self, header: FrameHeader) -> np.ndarray:
        """Encode frame header."""
        data_bits = np.concatenate([
            int_to_bits(header.length, self.length_bits),
            int_to_bits(header.src, self.src_bits),
            int_to_bits(header.dst, self.dst_bits),
            int_to_bits(header.frame_type, self.frame_type_bits),
            int_to_bits(header.mod_scheme.value, self.mod_scheme_bits),
            int_to_bits(header.sequence_number, self.sequence_number_bits),
            np.zeros(self.reserved_bits, dtype=np.int8),
        ])

        crc = self._crc_calc(data_bits)
        crc_bits = int_to_bits(crc, self.crc_bits)
        return np.concatenate([data_bits, crc_bits])

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        o = self._offsets  # shorthand

        # All slices stay as ndarrays — no .tolist() needed
        length_bits        = header[o[0]:o[1]]
        src_bits           = header[o[1]:o[2]]
        dst_bits           = header[o[2]:o[3]]
        frame_type_bits    = header[o[3]:o[4]]
        mod_scheme_bits    = header[o[4]:o[5]]
        sequence_number_bits = header[o[5]:o[6]]

        crc_bits = header[self._crc_start : self._crc_start + self.crc_bits]

        # Convert to ints using numpy dot product (fast _bits_to_int)
        length          = _bits_to_int(length_bits)
        src             = _bits_to_int(src_bits)
        dst             = _bits_to_int(dst_bits)
        frame_type      = _bits_to_int(frame_type_bits)
        mod_scheme      = ModulationSchemes(_bits_to_int(mod_scheme_bits))
        sequence_number = _bits_to_int(sequence_number_bits)
        crc             = _bits_to_int(crc_bits)

        # Build the data region for CRC verification — no string join needed
        data_bits = np.concatenate([
            length_bits,
            src_bits,
            dst_bits,
            frame_type_bits,
            mod_scheme_bits,
            sequence_number_bits,
            np.zeros(self.reserved_bits, dtype=np.int8),
        ])

        expected_crc = self._crc_calc(data_bits)
        expected_crc_bits = int_to_bits(expected_crc, self.crc_bits)

        return FrameHeader(
            length,
            src,
            dst,
            frame_type,
            mod_scheme,
            sequence_number,
            crc,
            crc_passed=bool(np.array_equal(crc_bits, expected_crc_bits)),
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

    def payload_coded_n_bits(self, header: FrameHeader) -> int:
        """Return the number of coded payload bits for a given header."""
        raw = header.length*8 + self.PAYLOAD_CRC_BITS
        return raw + (-raw % self.PAYLOAD_PAD_MULTIPLE)

    @staticmethod
    def _crc16(data_bits: np.ndarray) -> int:
        bits = data_bits.flatten().astype(np.int8)
        # Pad to byte boundary
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.int8)])
        bytes_ = np.packbits(bits)  # converts bit array → byte array
        
        reg = 0xFFFF
        for byte in bytes_:
            reg ^= int(byte) << 8
            for _ in range(8):
                reg = ((reg << 1) ^ 0x1021) & 0xFFFF if reg & 0x8000 else (reg << 1) & 0xFFFF
        return reg

    def encode(
        self,
        header: FrameHeader,
        payload: np.ndarray,
        *,
        interleaving: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode data into a frame."""
        payload = payload.ravel()
        header_bits = self.frame_header_constructor.encode(header)
        header_encoded = self.golay.encode(header_bits)

        crc = self._crc16(payload)
        crc_bits = int_to_bits(crc, self.PAYLOAD_CRC_BITS)
        payload_with_crc = np.concatenate([payload, crc_bits])

        n = self.payload_coded_n_bits(header)
        payload_encoded = np.concatenate([payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=np.int8)])
        return (header_encoded, payload_encoded)

    def decode_header(self, header_encoded: np.ndarray) -> FrameHeader:
        """Decode a Golay-encoded header and verify CRC."""
        header_hard = header_encoded.astype(np.int8)
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
        payload_bits = payload_encoded#(payload_encoded < 0).astype(np.int8) if soft else payload_encoded.astype(np.int8)
        data_bits = payload_bits[: header.length*8]
        crc_bits = payload_bits[header.length*8 : header.length*8 + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits)
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)
        return data_bits
