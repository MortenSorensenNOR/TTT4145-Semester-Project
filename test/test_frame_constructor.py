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

# --- FIXTURES ---

@pytest.fixture
def frame_constructor():
    """Default FrameConstructor with rate-1/2 LDPC."""
    return FrameConstructor(
        data_size=324,
        code_rate=CodeRates.HALF_RATE,
        pilots=np.array([]),
    )

@pytest.fixture
def sample_header():
    """Sample frame header with a specific length for testing padding."""
    return FrameHeader(
        length=100,  # Explicitly using a non-block-aligned length to test padding
        src=1,
        dst=2,
        mod_scheme=ModulationSchemes.QPSK,
        coding_rate=CodeRates.HALF_RATE,
        crc=0,
    )

@pytest.fixture
def random_payload(sample_header):
    """Random payload that EXACTLY matches the header length."""
    rng = np.random.default_rng(seed=42)
    # This was the bug: previously it was hardcoded to 324 regardless of header.length
    return rng.integers(0, 2, size=sample_header.length, dtype=int)

@pytest.fixture
def qpsk():
    return QPSK()

@pytest.fixture
def pulse_shaper():
    return PulseShaper(sps=4, alpha=0.35, taps=101)


# --- TEST CLASSES ---

class TestFrameConstructorEncoding:
    """Test frame encoding produces correct output structure."""

    def test_encoded_frame_length(self, frame_constructor, sample_header, random_payload):
        """Encoded frame should have correct total length: Golay header (48) + LDPC payload."""
        encoded = frame_constructor.encode(sample_header, random_payload)

        # For header.length=100 at HALF_RATE, closest LDPC block is 324.
        # LDPC output (rate 1/2) = 324 * 2 = 648.
        # Total = 48 (Header) + 648 (Payload) = 696.
        expected_payload_block = 324 
        expected_length = 48 + (expected_payload_block * 2)
        
        assert encoded.shape[0] == expected_length, f"Expected {expected_length}, got {encoded.shape[0]}"

    def test_encoded_frame_is_binary(self, frame_constructor, sample_header, random_payload):
        """Encoded frame should contain only 0s and 1s."""
        encoded = frame_constructor.encode(sample_header, random_payload)
        assert np.all((encoded == 0) | (encoded == 1))


class TestFrameConstructorRoundtrip:
    """Test full frame encode/decode roundtrip."""

    def test_roundtrip_no_channel(self, frame_constructor, sample_header, random_payload):
        """Frame should survive encode/decode with no channel impairments."""
        encoded = frame_constructor.encode(sample_header, random_payload)
        decoded_header, decoded_payload = frame_constructor.decode(encoded)

        assert decoded_header.length == sample_header.length
        assert np.array_equal(decoded_payload, random_payload)

    @given(
        # Test a range of lengths to ensure padding logic is hit
        length=st.integers(min_value=1, max_value=324), 
        src=st.integers(min_value=0, max_value=3),
        mod_scheme=st.sampled_from(ModulationSchemes),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_roundtrip_various_lengths(self, frame_constructor, length, src, mod_scheme):
        """Property-based test to ensure any payload length up to 324 works."""
        header = FrameHeader(
            length=length, src=src, dst=0,
            mod_scheme=mod_scheme, coding_rate=CodeRates.HALF_RATE, crc=0,
        )
        rng = np.random.default_rng()
        payload = rng.integers(0, 2, size=length, dtype=int)

        encoded = frame_constructor.encode(header, payload)
        _, decoded_payload = frame_constructor.decode(encoded)

        assert len(decoded_payload) == length
        assert np.array_equal(decoded_payload, payload)


class TestDynamicPayloadLength:
    """Test dynamic payload length selection and padding."""

    @pytest.mark.parametrize("payload_length,expected_padded_k", [
        (10, 324),   # Tiny payload pads to first block
        (324, 324),  # Perfect fit
        (325, 648),  # One bit over triggers next block
        (648, 648),  # Perfect fit
        (700, 972),  # Mid-range padding
    ])
    def test_payload_padding_logic(self, frame_constructor, payload_length, expected_padded_k):
        """Verify that payloads are padded to the correct LDPC k-dimension."""
        rng = np.random.default_rng()
        # Use explicit dtype to match typical decoder outputs
        payload = rng.integers(0, 2, size=payload_length, dtype=int)
        header = FrameHeader(
            length=payload_length, src=1, dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE, crc=0,
        )

        encoded = frame_constructor.encode(header, payload)

        # Total bits = 48 (header) + (expected_padded_k / rate)
        expected_encoded_length = 48 + (expected_padded_k * 2)
        assert encoded.shape[0] == expected_encoded_length

        # Decode and verify
        _, decoded_payload = frame_constructor.decode(encoded)
        
        # 1. Check length
        assert len(decoded_payload) == payload_length
        
        # 2. Check content (Flatten and cast to ensure exact comparison)
        # This handles cases where one might be (N,) and the other (N, 1)
        np.testing.assert_array_equal(
            decoded_payload.flatten().astype(int), 
            payload.flatten().astype(int)
        )

class TestGolayHeaderCoding:
    """Standard Golay tests remain valid."""
    
    def test_golay_roundtrip_no_errors(self):
        golay = Golay()
        message = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
        encoded = golay.encode(message)
        decoded = golay.decode(encoded)
        assert np.array_equal(decoded, message)

    def test_golay_multi_block_encoding(self):
        golay = Golay()
        # 24-bit message = 2 blocks
        message = np.random.randint(0, 2, 24)
        encoded = golay.encode(message)
        assert len(encoded) == 48
        assert np.array_equal(golay.decode(encoded), message)


class TestFrameConstructorWithChannel:
    """Verifies that the system works over a simulated AWGN channel."""

    def test_full_frame_qpsk_soft_decision(self, frame_constructor, sample_header, random_payload, qpsk):
        """Test with LLRs at 15dB."""
        encoded = frame_constructor.encode(sample_header, random_payload)
        tx_symbols = qpsk.bits2symbols(encoded)

        channel = ChannelModel(ChannelConfig(snr_db=15.0, seed=42))
        rx_symbols = channel.apply(tx_symbols)

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        rx_llr = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq).flatten()

        decoded_header, decoded_payload = frame_constructor.decode(rx_llr)

        assert decoded_header.crc_passed
        assert np.array_equal(decoded_payload, random_payload)

    def test_unsupported_payload_length_raises(self, frame_constructor):
        """Ensure we raise an error if the payload is too big for the LDPC tables."""
        header = FrameHeader(
            length=5000, # Way too big
            src=1, dst=2, mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE, crc=0,
        )
        payload = np.zeros(5000, dtype=int)

        with pytest.raises(ValueError, match="Unsupported payload length"):
            frame_constructor.encode(header, payload)
