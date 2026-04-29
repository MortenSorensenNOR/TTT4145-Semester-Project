"""Construct frames with header data, pilots, and error correction information."""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

try:
    from modules.frame_constructor import frame_constructor_ext as _ext
    _USE_EXT = True
    logger.info("Loaded frame_constructor_ext pybind11 C++ extension.")
except ImportError:
    _USE_EXT = False
    logger.warning(
        "frame_constructor_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python setup.py build_ext --inplace"
    )


def int_to_bits(n: int, length: int) -> list[int]:
    """Convert an integer to a fixed-width big-endian bit list."""
    if _USE_EXT:
        return _ext.int_to_bits(n, length)
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]


def _bits_to_int(bits: list[int]) -> int:
    """Convert a bit list to an integer."""
    if _USE_EXT:
        return _ext.bits_to_int(bits)
    return int("".join(str(b) for b in bits), 2)


class ModulationSchemes(Enum):
    """Supported modulation schemes."""
    BPSK = 0
    QPSK = 1
    PSK8 = 2
    PSK16 = 3


DEFAULT_CODING_RATE = 3  # CodeRates.FIVE_SIXTH_RATE.value — kept as int to avoid an import cycle

@dataclass
class FrameHeader:
    """Frame header with metadata."""
    length: int  # payload length in bytes
    src: int
    dst: int
    frame_type: int
    mod_scheme: ModulationSchemes
    sequence_number: int
    coding_rate: int = DEFAULT_CODING_RATE  # 2-bit field carrying CodeRates.value
    crc: int = field(default=0, compare=False)
    crc_passed: bool = True

    @classmethod
    def _from_ext(cls, ext_hdr: "_ext.FrameHeader") -> "FrameHeader":
        """Construct from a C++ extension FrameHeader object."""
        return cls(
            length=ext_hdr.length,
            src=ext_hdr.src,
            dst=ext_hdr.dst,
            frame_type=ext_hdr.frame_type,
            mod_scheme=ModulationSchemes(int(ext_hdr.mod_scheme)),
            sequence_number=ext_hdr.sequence_number,
            coding_rate=ext_hdr.coding_rate,
            crc=ext_hdr.crc,
            crc_passed=ext_hdr.crc_passed,
        )

    def _to_ext(self) -> "_ext.FrameHeader":
        """Convert to a C++ extension FrameHeader object."""
        return _ext.FrameHeader(
            length=self.length,
            src=self.src,
            dst=self.dst,
            frame_type=self.frame_type,
            mod_scheme=_ext.ModulationSchemes(self.mod_scheme.value),
            sequence_number=self.sequence_number,
            coding_rate=self.coding_rate,
            crc=self.crc,
            crc_passed=self.crc_passed,
        )


@dataclass
class FrameHeaderConfig:
    """Bit-width configuration for frame header fields."""
    payload_length_bits: int = 11
    src_bits: int = 2
    dst_bits: int = 2
    frame_type_bits: int = 1
    mod_scheme_bits: int = 3
    sequence_number_bits: int = 7
    coding_rate_bits: int = 2
    crc_bits: int = 8
    header_total_size: int = field(init=False)

    use_golay: bool = False

    def __post_init__(self) -> None:
        """Compute the total header size from all field widths."""
        self.header_total_size = (
            self.payload_length_bits
            + self.src_bits
            + self.dst_bits
            + self.frame_type_bits
            + self.mod_scheme_bits
            + self.sequence_number_bits
            + self.coding_rate_bits
            + self.crc_bits
        )

    def _to_ext(self) -> "_ext.FrameHeaderConfig":
        """Convert to a C++ extension FrameHeaderConfig object."""
        cfg = _ext.FrameHeaderConfig()
        cfg.payload_length_bits   = self.payload_length_bits
        cfg.src_bits              = self.src_bits
        cfg.dst_bits              = self.dst_bits
        cfg.frame_type_bits       = self.frame_type_bits
        cfg.mod_scheme_bits       = self.mod_scheme_bits
        cfg.sequence_number_bits  = self.sequence_number_bits
        cfg.coding_rate_bits      = self.coding_rate_bits
        cfg.crc_bits              = self.crc_bits
        cfg.use_golay             = self.use_golay
        return cfg


class FrameHeaderConstructor:
    """Encode and decode frame header fields."""

    def __init__(self, config: FrameHeaderConfig) -> None:
        """Initialize fixed bit widths for each frame header field."""
        self._config = config

        if _USE_EXT:
            self._ext = _ext.FrameHeaderConstructor(config._to_ext())
        else:
            self._ext = None

        self.length_bits          = config.payload_length_bits
        self.src_bits             = config.src_bits
        self.dst_bits             = config.dst_bits
        self.frame_type_bits      = config.frame_type_bits
        self.mod_scheme_bits      = config.mod_scheme_bits
        self.sequence_number_bits = config.sequence_number_bits
        self.coding_rate_bits     = config.coding_rate_bits
        self.crc_bits             = config.crc_bits

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
        if _USE_EXT:
            return self._ext.encode(header._to_ext())

        length_bits = int_to_bits(header.length, self.length_bits)
        src_bits = int_to_bits(header.src, self.src_bits)
        dst_bits = int_to_bits(header.dst, self.dst_bits)
        frame_type_bits = int_to_bits(header.frame_type, self.frame_type_bits)
        mod_scheme_bits = int_to_bits(header.mod_scheme.value, self.mod_scheme_bits)
        sequence_number_bits = int_to_bits(header.sequence_number, self.sequence_number_bits)
        coding_rate_bits = int_to_bits(header.coding_rate, self.coding_rate_bits)

        data_bits = (
            length_bits
            + src_bits
            + dst_bits
            + frame_type_bits
            + mod_scheme_bits
            + sequence_number_bits
            + coding_rate_bits
        )
        crc = self._crc_calc("".join(str(b) for b in data_bits))
        data_bits += int_to_bits(crc, self.crc_bits)

        return np.array(data_bits, dtype=int)

    def decode(self, header: np.ndarray) -> FrameHeader:
        """Decode frame header."""
        if _USE_EXT:
            return FrameHeader._from_ext(self._ext.decode(header.astype(np.int32)))

        header = header.reshape(-1)
        offset = 0
        field_widths = [
            self.length_bits,
            self.src_bits,
            self.dst_bits,
            self.frame_type_bits,
            self.mod_scheme_bits,
            self.sequence_number_bits,
            self.coding_rate_bits,
        ]
        fields: list[list[int]] = []
        for width in field_widths:
            bits = header[offset : offset + width]
            fields.append(bits.tolist() if isinstance(bits, np.ndarray) else bits)
            offset += width

        (length_bits, src_bits, dst_bits, frame_type_bits,
         mod_scheme_bits, sequence_number_bits, coding_rate_bits) = fields

        crc_field = header[offset : offset + self.crc_bits]
        crc_bits = crc_field.tolist() if isinstance(crc_field, np.ndarray) else crc_field

        length = _bits_to_int(length_bits)
        src = _bits_to_int(src_bits)
        dst = _bits_to_int(dst_bits)
        frame_type = _bits_to_int(frame_type_bits)
        mod_scheme = ModulationSchemes(_bits_to_int(mod_scheme_bits))
        sequence_number = _bits_to_int(sequence_number_bits)
        coding_rate = _bits_to_int(coding_rate_bits)
        crc = _bits_to_int(crc_bits)

        data_str = "".join(
            str(b)
            for b in (
                length_bits
                + src_bits
                + dst_bits
                + frame_type_bits
                + mod_scheme_bits
                + sequence_number_bits
                + coding_rate_bits
            )
        )
        expected_crc_bits = int_to_bits(self._crc_calc(data_str), self.crc_bits)

        return FrameHeader(
            length, src, dst, frame_type, mod_scheme, sequence_number,
            coding_rate, crc, crc_passed=(crc_bits == expected_crc_bits),
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

        if _USE_EXT:
            self._ext = _ext.FrameConstructor(self.header_config._to_ext())
        else:
            self._ext = None
            from modules.golay import Golay
            self.golay = Golay()

    @property
    def header_encoded_n_bits(self) -> int:
        """Number of bits in the encoded header."""
        if _USE_EXT:
            return self._ext.header_encoded_n_bits()
        golay_ratio = self.golay.block_length // self.golay.message_length
        return self.frame_header_constructor.header_length * golay_ratio

    def payload_coded_n_bits(self, header: FrameHeader) -> int:
        """Return the number of coded payload bits for a given header."""
        if _USE_EXT:
            return self._ext.payload_coded_n_bits(header._to_ext())
        raw = header.length * 8 + self.PAYLOAD_CRC_BITS
        return raw + (-raw % self.PAYLOAD_PAD_MULTIPLE)

    @staticmethod
    def _crc16(data_bits: np.ndarray) -> int:
        """Compute CRC-16-CCITT over a bit array (poly 0x1021)."""
        n = FrameConstructor.PAYLOAD_CRC_BITS
        mask = (1 << n) - 1
        msb = 1 << (n - 1)
        reg = mask
        for bit in data_bits.flatten():
            reg ^= int(bit) << (n - 1)
            reg = (((reg << 1) ^ 0x1021) & mask) if (reg & msb) else ((reg << 1) & mask)
        return reg

    def encode(
        self,
        header: FrameHeader,
        payload: np.ndarray,
        *,
        interleaving: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode data into a frame."""
        if _USE_EXT:
            h, p = self._ext.encode(header._to_ext(), payload.ravel().astype(np.int32))
            # The C++ encoder allocates header_length bits (rounded up to even)
            # but only writes header_total_size bits — any odd-length header
            # leaves the trailing slot uninitialized.  Zero-fill so it doesn't
            # blow up bits2symbols downstream.
            raw_bits = self.header_config.header_total_size
            if raw_bits < len(h):
                h[raw_bits:] = 0
            return (h, p)

        payload = payload.ravel()
        header_bits = self.frame_header_constructor.encode(header)
        if self.header_config.use_golay:
            header_encoded = self.golay.encode(header_bits)
        else:
            header_encoded = header_bits
        crc = self._crc16(payload)
        crc_bits = np.array(int_to_bits(crc, self.PAYLOAD_CRC_BITS), dtype=int)
        payload_with_crc = np.concatenate([payload, crc_bits])

        n = self.payload_coded_n_bits(header)
        payload_encoded = np.concatenate([payload_with_crc, np.zeros(n - len(payload_with_crc), dtype=int)])
        return (header_encoded, payload_encoded)

    def decode_header(self, header_encoded: np.ndarray) -> FrameHeader:
        """Decode a header and verify CRC."""
        if _USE_EXT:
            return FrameHeader._from_ext(self._ext.decode_header(header_encoded.astype(np.int32)))

        header_hard = header_encoded.astype(int)
        if self.header_config.use_golay:
            header_bits = self.golay.decode(header_hard)
        else:
            header_bits = header_hard
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
        """Decode payload and verify CRC-16."""
        if _USE_EXT:
            return self._ext.decode_payload(header._to_ext(), payload_encoded.astype(np.float64), soft)

        payload_bits = (payload_encoded < 0).astype(int) if soft else payload_encoded.astype(int)
        data_bits = payload_bits[: header.length * 8]
        crc_bits = payload_bits[header.length * 8 : header.length * 8 + self.PAYLOAD_CRC_BITS]
        received_crc = _bits_to_int(crc_bits.astype(int).tolist())
        expected_crc = self._crc16(data_bits)
        if received_crc != expected_crc:
            msg = f"Payload CRC-16 mismatch: got 0x{received_crc:04X}, expected 0x{expected_crc:04X}"
            raise ValueError(msg)
        return data_bits

