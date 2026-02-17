import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from modules.channel_coding import CodeRates
from modules.frame_constructor import (
    FrameHeader,
    FrameHeaderConfig,
    FrameHeaderConstructor,
    ModulationSchemes,
)
from modules.channel import ChannelConfig, ChannelModel

LENGTH_BITS = 10
SRC_BITS = 2
DST_BITS = 2
MOD_SCHEME_BITS = 3
CODING_RATE_BITS = 2
CRC_BITS = 4

LENGTH_MAX = (1 << LENGTH_BITS) - 1
SRC_MAX = (1 << SRC_BITS) - 1
DST_MAX = (1 << DST_BITS) - 1

PADDING_BIT_POS = LENGTH_BITS + SRC_BITS + DST_BITS + MOD_SCHEME_BITS + CODING_RATE_BITS


def make_header_constructor() -> FrameHeaderConstructor:
    return FrameHeaderConstructor(
        length_bits=LENGTH_BITS,
        src_bits=SRC_BITS,
        dst_bits=DST_BITS,
        mod_scheme_bits=MOD_SCHEME_BITS,
        coding_rate_bits=CODING_RATE_BITS,
        crc_bits=CRC_BITS,
    )


class TestFrameHeaderConstructor:
    header_constructor = make_header_constructor()

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
    )
    def test_roundtrip(self, length, src, dst, mod_scheme, coding_rate):
        header = FrameHeader(
            length=length,
            src=src,
            dst=dst,
            mod_scheme=mod_scheme,
            coding_rate=coding_rate,
            crc=0,
        )
        encoded = self.header_constructor.encode(header)
        decoded = self.header_constructor.decode(np.array(encoded))

        assert decoded.length == length
        assert decoded.src == src
        assert decoded.dst == dst
        assert decoded.mod_scheme == mod_scheme
        assert decoded.coding_rate == coding_rate
        assert decoded.crc_passed

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
        data=st.data(),
    )
    def test_crc_detects_burst_errors(self, length, src, dst, mod_scheme, coding_rate, data):
        header = FrameHeader(
            length=length,
            src=src,
            dst=dst,
            mod_scheme=mod_scheme,
            coding_rate=coding_rate,
            crc=0,
        )
        encoded = self.header_constructor.encode(header)

        padding_bit_pos = PADDING_BIT_POS

        # CRC guarantees detection of burst errors up to CRC_BITS
        burst_len = data.draw(st.integers(min_value=1, max_value=CRC_BITS))

        # Pick a valid start position that avoids the padding bit
        # and doesn't go out of bounds
        max_start = len(encoded) - burst_len
        valid_starts = [
            i for i in range(max_start + 1)
            if padding_bit_pos not in range(i, i + burst_len)
        ]
        burst_start = data.draw(st.sampled_from(valid_starts))

        # Flip consecutive bits (burst error)
        for i in range(burst_start, burst_start + burst_len):
            encoded[i] ^= 1

        try:
            decoded = self.header_constructor.decode(np.array(encoded))
            # If decode succeeds, CRC should fail
            assert not decoded.crc_passed
        except ValueError:
            # Corruption caused invalid enum value - also counts as detection
            pass


class TestFrameHeaderWithChannel:
    """Test header encoding/decoding through a channel with noise."""

    header_constructor = make_header_constructor()

    def _bits_to_bpsk(self, bits: list[int]) -> np.ndarray:
        """Convert bits to BPSK symbols (0 -> +1, 1 -> -1)."""
        return 1 - 2 * np.array(bits, dtype=np.float64)

    def _bpsk_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """Convert BPSK symbols back to bits via hard decision."""
        return (np.real(symbols) < 0).astype(int)

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
    )
    @settings(max_examples=50)
    def test_roundtrip_high_snr(self, length, src, dst, mod_scheme, coding_rate):
        """Header should survive high SNR AWGN channel."""
        header = FrameHeader(
            length=length,
            src=src,
            dst=dst,
            mod_scheme=mod_scheme,
            coding_rate=coding_rate,
            crc=0,
        )
        encoded_bits = self.header_constructor.encode(header)

        # Modulate to BPSK
        tx_symbols = self._bits_to_bpsk(encoded_bits)

        # Pass through high SNR channel
        channel = ChannelModel(ChannelConfig(snr_db=30.0, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        # Demodulate
        rx_bits = self._bpsk_to_bits(rx_symbols)

        # Decode header
        decoded = self.header_constructor.decode(rx_bits)

        assert decoded.length == length
        assert decoded.src == src
        assert decoded.dst == dst
        assert decoded.mod_scheme == mod_scheme
        assert decoded.coding_rate == coding_rate
        assert decoded.crc_passed

    @pytest.mark.parametrize("snr_db", [20.0, 15.0, 10.0])
    def test_roundtrip_various_snr(self, snr_db):
        """Test header at various SNR levels with fixed seed for reproducibility."""
        header = FrameHeader(
            length=100,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )
        encoded_bits = self.header_constructor.encode(header)

        tx_symbols = self._bits_to_bpsk(encoded_bits)

        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=123))
        rx_symbols = channel.apply(tx_symbols)

        rx_bits = self._bpsk_to_bits(rx_symbols)
        decoded = self.header_constructor.decode(rx_bits)

        # At these SNR levels with BPSK, we should have no errors
        assert decoded.length == header.length
        assert decoded.src == header.src
        assert decoded.dst == header.dst
        assert decoded.mod_scheme == header.mod_scheme
        assert decoded.coding_rate == header.coding_rate
        assert decoded.crc_passed

    def test_header_with_phase_offset(self):
        """Header should survive channel with phase offset when using coherent detection."""
        header = FrameHeader(
            length=512,
            src=3,
            dst=0,
            mod_scheme=ModulationSchemes.QAM16,
            coding_rate=CodeRates.THREE_QUARTER_RATE,
            crc=0,
        )
        encoded_bits = self.header_constructor.encode(header)
        tx_symbols = self._bits_to_bpsk(encoded_bits)

        # Channel with phase offset (but high SNR)
        channel = ChannelModel(ChannelConfig(
            snr_db=30.0,
            enable_phase_offset=True,
            initial_phase_rad=np.pi / 4,  # 45 degree phase shift
            seed=42,
        ))
        rx_symbols = channel.apply(tx_symbols)

        # Compensate for known phase offset before demodulation
        rx_compensated = rx_symbols * np.exp(-1j * np.pi / 4)
        rx_bits = self._bpsk_to_bits(rx_compensated)

        decoded = self.header_constructor.decode(rx_bits)

        assert decoded.length == header.length
        assert decoded.coding_rate == header.coding_rate
        assert decoded.crc_passed

    def test_crc_detects_noise_errors(self):
        """At low SNR, bit errors should be detected by CRC."""
        header = FrameHeader(
            length=100,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )
        encoded_bits = self.header_constructor.encode(header)
        tx_symbols = self._bits_to_bpsk(encoded_bits)

        # Very low SNR to guarantee errors
        channel = ChannelModel(ChannelConfig(snr_db=-5.0, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        rx_bits = self._bpsk_to_bits(rx_symbols)

        try:
            decoded = self.header_constructor.decode(rx_bits)
            # With this much noise, CRC should almost certainly fail
            # (or values are corrupted beyond valid enum range)
            assert not decoded.crc_passed
        except ValueError:
            # Invalid enum value due to corruption - expected
            pass


class TestFrameHeaderConfig:
    """Test FrameHeaderConfig dataclass."""

    def test_default_config_total_size(self):
        config = FrameHeaderConfig()
        expected = (
            config.payload_length_bits +
            config.src_bits +
            config.dst_bits +
            config.mod_scheme_bits +
            config.coding_rate_bits +
            config.reserved_bits +
            config.crc_bits
        )
        assert config.header_total_size == expected

    def test_custom_config(self):
        config = FrameHeaderConfig(
            payload_length_bits=12,
            src_bits=4,
            dst_bits=4,
            mod_scheme_bits=3,
            coding_rate_bits=3,
            reserved_bits=2,
            crc_bits=8,
        )
        assert config.header_total_size == 12 + 4 + 4 + 3 + 3 + 2 + 8
