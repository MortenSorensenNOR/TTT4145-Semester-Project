"""Tests for FrameHeaderConstructor encode/decode roundtrip and CRC detection."""

from collections.abc import Sequence

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modules.channel import ChannelConfig, ChannelModel
from modules.channel_coding import CodeRates
from modules.frame_constructor import (
    FrameHeader,
    FrameHeaderConfig,
    FrameHeaderConstructor,
    ModulationSchemes,
)

LENGTH_BITS = 10
SRC_BITS = 2
DST_BITS = 2
FRAME_TYPE_BITS = 2
MOD_SCHEME_BITS = 3
CODING_RATE_BITS = 3
SEQUENCE_NUMBER_BITS = 4
CRC_BITS = 8

LENGTH_MAX = (1 << LENGTH_BITS) - 1
SRC_MAX = (1 << SRC_BITS) - 1
DST_MAX = (1 << DST_BITS) - 1
FRAME_TYPE_MAX = (1 << FRAME_TYPE_BITS) - 1
SEQUENCE_NUMBER_MAX = (1 << SEQUENCE_NUMBER_BITS) - 1

PADDING_BIT_POS = LENGTH_BITS + SRC_BITS + DST_BITS + FRAME_TYPE_BITS + MOD_SCHEME_BITS + CODING_RATE_BITS + SEQUENCE_NUMBER_BITS

HIGH_SNR_DB = 30.0
LOW_SNR_DB = -5.0
PHASE_OFFSET_RAD = np.pi / 4
CHANNEL_SEED = 42
CHANNEL_SEED_ALT = 123
MAX_HYPOTHESIS_EXAMPLES = 50


def make_header_constructor() -> FrameHeaderConstructor:
    """Create a FrameHeaderConstructor with default test config."""
    config = FrameHeaderConfig(
        payload_length_bits=LENGTH_BITS,
        src_bits=SRC_BITS,
        dst_bits=DST_BITS,
        frame_type_bits=FRAME_TYPE_BITS,
        mod_scheme_bits=MOD_SCHEME_BITS,
        coding_rate_bits=CODING_RATE_BITS,
        sequence_number_bits=SEQUENCE_NUMBER_BITS,
        crc_bits=CRC_BITS,
    )
    return FrameHeaderConstructor(config)


def _make_header(
    length: int,
    src: int,
    dst: int,
    mod_scheme: ModulationSchemes,
    coding_rate: CodeRates,
    frame_type: int = 0,
    sequence_number: int = 0,
) -> FrameHeader:
    """Create a FrameHeader with the given fields and zero CRC."""
    return FrameHeader(
        length=length,
        src=src,
        dst=dst,
        frame_type=frame_type,
        mod_scheme=mod_scheme,
        coding_rate=coding_rate,
        sequence_number=sequence_number,
        crc=0,
    )


class TestFrameHeaderConstructor:
    """Tests for basic header encode/decode roundtrip and CRC burst error detection."""

    header_constructor = make_header_constructor()

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
    )
    def test_roundtrip(
        self,
        length: int,
        src: int,
        dst: int,
        mod_scheme: ModulationSchemes,
        coding_rate: CodeRates,
    ) -> None:
        """Encode then decode a header and verify all fields match."""
        header = _make_header(length, src, dst, mod_scheme, coding_rate)
        encoded = self.header_constructor.encode(header)
        decoded = self.header_constructor.decode(np.array(encoded))

        np.testing.assert_equal(decoded.length, length)
        np.testing.assert_equal(decoded.src, src)
        np.testing.assert_equal(decoded.dst, dst)
        np.testing.assert_equal(decoded.mod_scheme, mod_scheme)
        np.testing.assert_equal(decoded.coding_rate, coding_rate)
        if not decoded.crc_passed:
            pytest.fail("CRC did not pass after clean roundtrip")

    @given(data=st.data())
    def test_crc_detects_burst_errors(
        self,
        data: st.DataObject,
    ) -> None:
        """CRC should detect burst errors up to CRC_BITS in length."""
        length = data.draw(st.integers(min_value=0, max_value=LENGTH_MAX))
        src = data.draw(st.integers(min_value=0, max_value=SRC_MAX))
        dst = data.draw(st.integers(min_value=0, max_value=DST_MAX))
        mod_scheme = data.draw(st.sampled_from(ModulationSchemes))
        coding_rate = data.draw(st.sampled_from(CodeRates))
        header = _make_header(length, src, dst, mod_scheme, coding_rate)
        encoded = self.header_constructor.encode(header)

        padding_bit_pos = PADDING_BIT_POS

        burst_len = data.draw(st.integers(min_value=1, max_value=CRC_BITS))

        max_start = len(encoded) - burst_len
        valid_starts = [i for i in range(max_start + 1) if padding_bit_pos not in range(i, i + burst_len)]
        burst_start = data.draw(st.sampled_from(valid_starts))

        for i in range(burst_start, burst_start + burst_len):
            encoded[i] ^= 1

        try:
            decoded = self.header_constructor.decode(np.array(encoded))
            if decoded.crc_passed:
                pytest.fail("CRC passed despite burst error injection")
        except ValueError:
            pass


def _bits_to_bpsk(bits: np.ndarray | Sequence[int]) -> np.ndarray:
    """Convert bits to BPSK symbols (0 -> +1, 1 -> -1)."""
    return 1 - 2 * np.array(bits, dtype=np.float64)


def _bpsk_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Convert BPSK symbols back to bits via hard decision."""
    return (np.real(symbols) < 0).astype(int)


class TestFrameHeaderWithChannel:
    """Test header encoding/decoding through a channel with noise."""

    header_constructor = make_header_constructor()

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
    )
    @settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
    def test_roundtrip_high_snr(
        self,
        length: int,
        src: int,
        dst: int,
        mod_scheme: ModulationSchemes,
        coding_rate: CodeRates,
    ) -> None:
        """Header should survive high SNR AWGN channel."""
        header = _make_header(length, src, dst, mod_scheme, coding_rate)
        encoded_bits = self.header_constructor.encode(header)

        tx_symbols = _bits_to_bpsk(encoded_bits)

        channel = ChannelModel(ChannelConfig(snr_db=HIGH_SNR_DB, seed=CHANNEL_SEED))
        rx_symbols = channel.apply(tx_symbols)

        rx_bits = _bpsk_to_bits(rx_symbols)

        decoded = self.header_constructor.decode(rx_bits)

        np.testing.assert_equal(decoded.length, length)
        np.testing.assert_equal(decoded.src, src)
        np.testing.assert_equal(decoded.dst, dst)
        np.testing.assert_equal(decoded.mod_scheme, mod_scheme)
        np.testing.assert_equal(decoded.coding_rate, coding_rate)
        if not decoded.crc_passed:
            pytest.fail("CRC did not pass after high-SNR channel roundtrip")

    @pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0])
    def test_roundtrip_various_snr(self, snr_db: float) -> None:
        """Test header at various SNR levels with fixed seed for reproducibility."""
        header = _make_header(
            length=100,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE,
        )
        encoded_bits = self.header_constructor.encode(header)

        tx_symbols = _bits_to_bpsk(encoded_bits)

        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=CHANNEL_SEED_ALT))
        rx_symbols = channel.apply(tx_symbols)

        rx_bits = _bpsk_to_bits(rx_symbols)
        decoded = self.header_constructor.decode(rx_bits)

        np.testing.assert_equal(decoded.length, header.length)
        np.testing.assert_equal(decoded.src, header.src)
        np.testing.assert_equal(decoded.dst, header.dst)
        np.testing.assert_equal(decoded.mod_scheme, header.mod_scheme)
        np.testing.assert_equal(decoded.coding_rate, header.coding_rate)
        if not decoded.crc_passed:
            pytest.fail("CRC did not pass at SNR={snr_db}")

    def test_header_with_phase_offset(self) -> None:
        """Header should survive channel with phase offset when using coherent detection."""
        header = _make_header(
            length=512,
            src=3,
            dst=0,
            mod_scheme=ModulationSchemes.QAM16,
            coding_rate=CodeRates.THREE_QUARTER_RATE,
        )
        encoded_bits = self.header_constructor.encode(header)
        tx_symbols = _bits_to_bpsk(encoded_bits)

        channel = ChannelModel(
            ChannelConfig(
                snr_db=HIGH_SNR_DB,
                initial_phase_rad=PHASE_OFFSET_RAD,
                seed=CHANNEL_SEED,
            ),
        )
        rx_symbols = channel.apply(tx_symbols)

        rx_compensated = rx_symbols * np.exp(-1j * PHASE_OFFSET_RAD)
        rx_bits = _bpsk_to_bits(rx_compensated)

        decoded = self.header_constructor.decode(rx_bits)

        np.testing.assert_equal(decoded.length, header.length)
        np.testing.assert_equal(decoded.coding_rate, header.coding_rate)
        if not decoded.crc_passed:
            pytest.fail("CRC did not pass after phase-offset channel roundtrip")

    def test_crc_detects_noise_errors(self) -> None:
        """At low SNR, bit errors should be detected by CRC."""
        header = _make_header(
            length=100,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.HALF_RATE,
        )
        encoded_bits = self.header_constructor.encode(header)
        tx_symbols = _bits_to_bpsk(encoded_bits)

        channel = ChannelModel(ChannelConfig(snr_db=LOW_SNR_DB, seed=CHANNEL_SEED))
        rx_symbols = channel.apply(tx_symbols)

        rx_bits = _bpsk_to_bits(rx_symbols)

        try:
            decoded = self.header_constructor.decode(rx_bits)
            if decoded.crc_passed:
                pytest.fail("CRC passed despite heavy noise corruption")
        except ValueError:
            pass
