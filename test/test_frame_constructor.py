"""Tests for FrameConstructor with Golay header encoding and LDPC payload coding."""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck

from modules.frame_constructor import (
    FrameConstructor,
    FrameHeader,
    FrameHeaderConfig,
    ModulationSchemes,
)
from modules.channel_coding import CodeRates, Golay
from modules.channel import ChannelConfig, ChannelModel
from modules.modulation import QPSK
from modules.pulseshaping import PulseShaper


@pytest.fixture
def frame_constructor():
    """Default FrameConstructor with rate-1/2 LDPC."""
    return FrameConstructor(
        data_size=324,
        code_rate=CodeRates.HALF_RATE,
        pilots=np.array([]),
    )


@pytest.fixture
def random_payload():
    """324-bit random payload matching LDPC k=324."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 2, size=324, dtype=int)


@pytest.fixture
def sample_header():
    """Sample frame header for testing."""
    return FrameHeader(
        length=100,
        src=1,
        dst=2,
        mod_scheme=ModulationSchemes.QPSK,
        coding_rate=CodeRates.HALF_RATE,
        crc=0,
    )


@pytest.fixture
def qpsk():
    """QPSK modulator/demodulator."""
    return QPSK()


@pytest.fixture
def pulse_shaper():
    """Pulse shaper for symbol transmission."""
    return PulseShaper(sps=4, alpha=0.35, taps=101)


class TestFrameConstructorEncoding:
    """Test frame encoding produces correct output structure."""

    def test_encoded_frame_length(self, frame_constructor, sample_header, random_payload):
        """Encoded frame should have correct total length: Golay header (48) + LDPC payload (648)."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # Header: 24 bits -> 48 bits after Golay (rate 1/2)
        # Payload: 324 bits -> 648 bits after LDPC (rate 1/2)
        expected_length = 48 + 648
        assert encoded.shape[0] == expected_length

    def test_encoded_frame_is_binary(self, frame_constructor, sample_header, random_payload):
        """Encoded frame should contain only 0s and 1s."""
        encoded = frame_constructor.encode(sample_header, random_payload)
        assert np.all((encoded == 0) | (encoded == 1))

    def test_different_payloads_different_frames(self, frame_constructor, sample_header):
        """Different payloads should produce different encoded frames."""
        payload1 = np.zeros(324, dtype=int)
        payload2 = np.ones(324, dtype=int)

        encoded1 = frame_constructor.encode(sample_header, payload1)
        encoded2 = frame_constructor.encode(sample_header, payload2)

        assert not np.array_equal(encoded1, encoded2)

    def test_different_headers_different_frames(self, frame_constructor, random_payload):
        """Different headers should produce different encoded frames."""
        header1 = FrameHeader(
            length=100, src=1, dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE, crc=0,
        )
        header2 = FrameHeader(
            length=200, src=3, dst=0,
            mod_scheme=ModulationSchemes.QAM16,
            coding_rate=CodeRates.THREE_QUARTER_RATE, crc=0,
        )

        encoded1 = frame_constructor.encode(header1, random_payload)
        encoded2 = frame_constructor.encode(header2, random_payload)

        # At minimum the header portion should differ
        assert not np.array_equal(encoded1[:48], encoded2[:48])


class TestGolayHeaderCoding:
    """Test Golay encoding/decoding for frame headers."""

    def test_golay_roundtrip_no_errors(self):
        """Golay encode/decode should recover original message with no errors."""
        golay = Golay()
        message = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)

        encoded = golay.encode(message)
        decoded = golay.decode(encoded)

        assert np.array_equal(decoded, message)

    def test_golay_corrects_1_error(self):
        """Golay should correct single bit errors."""
        golay = Golay()
        message = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
        encoded = golay.encode(message)

        # Introduce 1 error at various positions
        for error_pos in [0, 5, 11, 15, 23]:
            corrupted = encoded.copy()
            corrupted[error_pos] ^= 1
            decoded = golay.decode(corrupted)
            assert np.array_equal(decoded, message), f"Failed to correct error at position {error_pos}"

    def test_golay_corrects_2_errors(self):
        """Golay should correct double bit errors."""
        golay = Golay()
        message = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        encoded = golay.encode(message)

        # Introduce 2 errors
        error_positions = [(0, 5), (3, 20), (11, 23)]
        for pos1, pos2 in error_positions:
            corrupted = encoded.copy()
            corrupted[pos1] ^= 1
            corrupted[pos2] ^= 1
            decoded = golay.decode(corrupted)
            assert np.array_equal(decoded, message), f"Failed to correct errors at positions {pos1}, {pos2}"

    def test_golay_corrects_3_errors(self):
        """Golay should correct triple bit errors (max correction capability)."""
        golay = Golay()
        message = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
        encoded = golay.encode(message)

        # Introduce 3 errors
        corrupted = encoded.copy()
        corrupted[0] ^= 1
        corrupted[12] ^= 1
        corrupted[23] ^= 1

        decoded = golay.decode(corrupted)
        assert np.array_equal(decoded, message)

    def test_golay_fails_on_4_errors(self):
        """Golay should fail or miscorrect with 4+ errors."""
        golay = Golay()
        message = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
        encoded = golay.encode(message)

        # Introduce 4 errors
        corrupted = encoded.copy()
        corrupted[0] ^= 1
        corrupted[1] ^= 1
        corrupted[2] ^= 1
        corrupted[3] ^= 1

        # Should either raise an error or return wrong result
        try:
            decoded = golay.decode(corrupted)
            # If it doesn't raise, the result should be wrong
            assert not np.array_equal(decoded, message)
        except ValueError:
            pass  # Expected: "More than 3 bit errors in block"

    @given(st.binary(min_size=12, max_size=12))
    @settings(max_examples=50)
    def test_golay_roundtrip_random_messages(self, message_bytes):
        """Property-based test: Golay roundtrip works for any 12-bit message."""
        golay = Golay()
        # Convert bytes to 12 bits
        message = np.array([int(b) for b in format(message_bytes[0], '08b') + format(message_bytes[1] >> 4, '04b')], dtype=int)

        encoded = golay.encode(message)
        decoded = golay.decode(encoded)

        assert np.array_equal(decoded, message)

    def test_golay_multi_block_encoding(self):
        """Golay should handle multi-block messages (like 24-bit headers -> 2 blocks)."""
        golay = Golay()
        # 24-bit message (2 blocks of 12 bits)
        message = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                           0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)

        encoded = golay.encode(message)
        assert encoded.shape[0] == 48  # 2 blocks * 24 bits

        decoded = golay.decode(encoded)
        assert np.array_equal(decoded, message)


class TestFrameConstructorRoundtrip:
    """Test full frame encode/decode roundtrip."""

    def test_roundtrip_no_channel(self, frame_constructor, sample_header, random_payload):
        """Frame should survive encode/decode with no channel impairments."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # Convert to hard bits for decode (simulating perfect reception)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert decoded_header.length == sample_header.length
        assert decoded_header.src == sample_header.src
        assert decoded_header.dst == sample_header.dst
        assert decoded_header.mod_scheme == sample_header.mod_scheme
        assert decoded_header.coding_rate == sample_header.coding_rate
        assert decoded_header.crc_passed
        assert np.array_equal(decoded_payload, random_payload)

    @given(
        length=st.integers(min_value=0, max_value=1023),
        src=st.integers(min_value=0, max_value=3),
        dst=st.integers(min_value=0, max_value=3),
        mod_scheme=st.sampled_from(ModulationSchemes),
        coding_rate=st.sampled_from(CodeRates),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_roundtrip_various_headers(self, frame_constructor, length, src, dst, mod_scheme, coding_rate):
        """Roundtrip should work for various header field values."""
        header = FrameHeader(
            length=length, src=src, dst=dst,
            mod_scheme=mod_scheme, coding_rate=coding_rate, crc=0,
        )
        payload = np.zeros(324, dtype=int)

        encoded = frame_constructor.encode(header, payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert decoded_header.length == length
        assert decoded_header.src == src
        assert decoded_header.dst == dst
        assert decoded_header.mod_scheme == mod_scheme
        assert decoded_header.coding_rate == coding_rate


class TestFrameConstructorWithChannel:
    """Test frame construction through simulated channel using QPSK modulation."""

    def test_header_survives_high_snr_qpsk(self, frame_constructor, sample_header, random_payload, qpsk):
        """Header with Golay coding should survive high SNR AWGN channel with QPSK."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # QPSK modulate
        tx_symbols = qpsk.bits2symbols(encoded)

        # Pass through high SNR channel
        channel = ChannelModel(ChannelConfig(snr_db=20.0, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        # Hard decision demodulation
        rx_bits = qpsk.symbols2bits(rx_symbols).flatten()

        # Decode frame
        decoded_header, decoded_payload = frame_constructor.decode(rx_bits)

        assert decoded_header.length == sample_header.length
        assert decoded_header.src == sample_header.src
        assert decoded_header.dst == sample_header.dst
        assert decoded_header.crc_passed

    def test_frame_with_qpsk_soft_decision(self, frame_constructor, sample_header, random_payload, qpsk):
        """Frame decode should work with QPSK soft decision (LLR) output."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # QPSK modulate
        tx_symbols = qpsk.bits2symbols(encoded)

        # Pass through channel
        channel = ChannelModel(ChannelConfig(snr_db=15.0, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        # Soft decision demodulation (LLR)
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        rx_llr = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq).flatten()

        # Decode frame with LLR input
        decoded_header, decoded_payload = frame_constructor.decode(rx_llr)

        assert decoded_header.length == sample_header.length
        assert decoded_header.crc_passed
        # With LDPC soft decoding at 15dB, should have very few errors
        bit_errors = np.sum(decoded_payload != random_payload)
        assert bit_errors < 10, f"Too many bit errors: {bit_errors}"

    def test_header_golay_corrects_channel_errors(self, frame_constructor, sample_header, random_payload, qpsk):
        """Golay should correct bit errors introduced by noisy QPSK channel."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # QPSK modulate
        tx_symbols = qpsk.bits2symbols(encoded)

        # Test multiple seeds at moderate SNR
        success_count = 0
        for seed in range(10):
            channel = ChannelModel(ChannelConfig(snr_db=10.0, seed=seed))
            rx_symbols = channel.apply(tx_symbols)
            rx_bits = qpsk.symbols2bits(rx_symbols).flatten()

            try:
                decoded_header, _ = frame_constructor.decode(rx_bits)
                if decoded_header.length == sample_header.length and decoded_header.crc_passed:
                    success_count += 1
            except ValueError:
                pass  # Too many errors for Golay

        # At 10dB SNR with QPSK, Golay should succeed most of the time
        assert success_count >= 7, f"Only {success_count}/10 successful decodes at 10dB SNR"

    @pytest.mark.parametrize("snr_db", [20.0, 15.0, 12.0])
    def test_full_frame_qpsk_hard_decision(self, frame_constructor, sample_header, random_payload, qpsk, snr_db):
        """Full frame through QPSK channel with hard decision decoding."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        tx_symbols = qpsk.bits2symbols(encoded)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=42))
        rx_symbols = channel.apply(tx_symbols)
        rx_bits = qpsk.symbols2bits(rx_symbols).flatten()

        decoded_header, decoded_payload = frame_constructor.decode(rx_bits)

        assert decoded_header.length == sample_header.length
        assert decoded_header.crc_passed

    @pytest.mark.parametrize("snr_db", [12.0, 10.0, 8.0])
    def test_full_frame_qpsk_soft_decision(self, frame_constructor, sample_header, random_payload, qpsk, snr_db):
        """Full frame through QPSK channel with soft decision (LLR) decoding."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        tx_symbols = qpsk.bits2symbols(encoded)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        # Soft decision
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        rx_llr = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq).flatten()

        decoded_header, decoded_payload = frame_constructor.decode(rx_llr)

        assert decoded_header.length == sample_header.length
        assert decoded_header.crc_passed

    def test_frame_with_pulse_shaping(self, frame_constructor, sample_header, random_payload, qpsk, pulse_shaper):
        """Full frame through pulse-shaped QPSK channel."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # QPSK modulate
        tx_symbols = qpsk.bits2symbols(encoded)

        # Upsample and pulse shape
        sps = pulse_shaper.sps
        tx_upsampled = np.zeros(len(tx_symbols) * sps, dtype=complex)
        tx_upsampled[::sps] = tx_symbols
        tx_shaped = pulse_shaper.shape(tx_upsampled)

        # Pass through channel
        channel = ChannelModel(ChannelConfig(snr_db=15.0, seed=42))
        rx_shaped = channel.apply(tx_shaped)

        # Matched filter and downsample
        rx_filtered = pulse_shaper.shape(rx_shaped)
        rx_symbols = rx_filtered[::sps]

        # Soft decision demodulation
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        rx_llr = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq).flatten()

        # Decode
        decoded_header, decoded_payload = frame_constructor.decode(rx_llr)

        assert decoded_header.length == sample_header.length
        assert decoded_header.crc_passed

    def test_soft_vs_hard_decision_performance(self, frame_constructor, sample_header, qpsk):
        """Soft decision should outperform hard decision at moderate SNR."""
        payload = np.random.default_rng(seed=123).integers(0, 2, size=324, dtype=int)
        encoded = frame_constructor.encode(sample_header, payload)
        tx_symbols = qpsk.bits2symbols(encoded)

        snr_db = 8.0
        hard_errors = []
        soft_errors = []

        for seed in range(5):
            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=seed))
            rx_symbols = channel.apply(tx_symbols)

            # Hard decision
            rx_bits_hard = qpsk.symbols2bits(rx_symbols).flatten()
            try:
                _, decoded_hard = frame_constructor.decode(rx_bits_hard)
                hard_errors.append(np.sum(decoded_hard != payload))
            except ValueError:
                hard_errors.append(324)  # All errors if decode fails

            # Soft decision
            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            rx_llr = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq).flatten()
            try:
                _, decoded_soft = frame_constructor.decode(rx_llr)
                soft_errors.append(np.sum(decoded_soft != payload))
            except ValueError:
                soft_errors.append(324)

        # Soft decision should have fewer or equal errors on average
        assert np.mean(soft_errors) <= np.mean(hard_errors) + 5


class TestFrameConstructorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zeros_payload(self, frame_constructor, sample_header):
        """Frame with all-zeros payload should encode/decode correctly."""
        payload = np.zeros(324, dtype=int)
        encoded = frame_constructor.encode(sample_header, payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert np.array_equal(decoded_payload, payload)

    def test_all_ones_payload(self, frame_constructor, sample_header):
        """Frame with all-ones payload should encode/decode correctly."""
        payload = np.ones(324, dtype=int)
        encoded = frame_constructor.encode(sample_header, payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert np.array_equal(decoded_payload, payload)

    def test_alternating_bits_payload(self, frame_constructor, sample_header):
        """Frame with alternating bits pattern should encode/decode correctly."""
        payload = np.array([i % 2 for i in range(324)], dtype=int)
        encoded = frame_constructor.encode(sample_header, payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert np.array_equal(decoded_payload, payload)

    def test_max_header_values(self, frame_constructor, random_payload):
        """Frame with maximum header field values should encode/decode correctly."""
        header = FrameHeader(
            length=1023,  # max for 10 bits
            src=3,        # max for 2 bits
            dst=3,        # max for 2 bits
            mod_scheme=ModulationSchemes.QAM64,
            coding_rate=CodeRates.FIVE_SIXTH_RATE,
            crc=0,
        )
        encoded = frame_constructor.encode(header, random_payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert decoded_header.length == 1023
        assert decoded_header.src == 3
        assert decoded_header.dst == 3

    def test_min_header_values(self, frame_constructor, random_payload):
        """Frame with minimum header field values should encode/decode correctly."""
        header = FrameHeader(
            length=0,
            src=0,
            dst=0,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )
        encoded = frame_constructor.encode(header, random_payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert decoded_header.length == 0
        assert decoded_header.src == 0
        assert decoded_header.dst == 0

    def test_payload_size_mismatch_raises(self, frame_constructor, sample_header):
        """Encoding with wrong payload size should raise assertion error."""
        wrong_size_payload = np.zeros(100, dtype=int)

        with pytest.raises(AssertionError):
            frame_constructor.encode(sample_header, wrong_size_payload)


class TestFrameConstructorConfig:
    """Test FrameConstructor configuration options."""

    def test_custom_header_config(self):
        """FrameConstructor should respect custom header configuration."""
        custom_config = FrameHeaderConfig(
            payload_length_bits=12,
            src_bits=4,
            dst_bits=4,
            mod_scheme_bits=3,
            coding_rate_bits=3,
            reserved_bits=2,
            crc_bits=8,
        )

        fc = FrameConstructor(
            data_size=324,
            code_rate=CodeRates.HALF_RATE,
            pilots=np.array([]),
            header_config=custom_config,
        )

        assert fc.header_config.payload_length_bits == 12
        assert fc.header_config.header_total_size == 36

    def test_frame_constructor_with_pilots(self):
        """FrameConstructor should accept pilot symbols configuration."""
        pilots = np.array([1, 0, 1, 0, 1, 0, 1, 0])

        fc = FrameConstructor(
            data_size=324,
            code_rate=CodeRates.HALF_RATE,
            pilots=pilots,
        )

        assert np.array_equal(fc.pilots, pilots)
