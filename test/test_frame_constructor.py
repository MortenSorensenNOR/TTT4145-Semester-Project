"""Tests for FrameConstructor with Golay header encoding and LDPC payload coding."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from modules.channel import ChannelConfig, ChannelModel
from modules.channel_coding import CodeRates, Golay
from modules.frame_constructor import (
    FrameConstructor,
    FrameHeader,
    ModulationSchemes,
)
from modules.modulation import QPSK

# --- Constants ---

GOLAY_ENCODED_BITS = 48
GOLAY_MESSAGE_BITS = 24
GOLAY_BLOCK_SIZE = 12
FIRST_LDPC_K = 324
SECOND_LDPC_K = 648
THIRD_LDPC_K = 972
SNR_DB = 15.0
CHANNEL_SEED = 42
RNG_SEED = 42
MAX_PAYLOAD_LENGTH = 324
UNSUPPORTED_PAYLOAD_LENGTH = 5000

# --- FIXTURES ---


@pytest.fixture
def frame_constructor() -> FrameConstructor:
    """Create default FrameConstructor."""
    return FrameConstructor()


@pytest.fixture
def sample_header() -> FrameHeader:
    """Create sample frame header with a specific length for testing padding."""
    return FrameHeader(
        length=100,
        src=1,
        dst=2,
        mod_scheme=ModulationSchemes.QPSK,
        coding_rate=CodeRates.HALF_RATE,
        crc=0,
    )


@pytest.fixture
def random_payload(sample_header: FrameHeader) -> np.ndarray:
    """Create random payload that EXACTLY matches the header length."""
    rng = np.random.default_rng(seed=RNG_SEED)
    return rng.integers(0, 2, size=sample_header.length, dtype=int)


@pytest.fixture
def qpsk() -> QPSK:
    """Create QPSK modulator instance."""
    return QPSK()


# --- TEST CLASSES ---


class TestFrameConstructorEncoding:
    """Test frame encoding produces correct output structure."""

    def test_encoded_frame_length(
        self,
        frame_constructor: FrameConstructor,
        sample_header: FrameHeader,
        random_payload: np.ndarray,
    ) -> None:
        """Encoded frame should have correct total length: Golay header (48) + LDPC payload."""
        header_encoded, payload_encoded = frame_constructor.encode(sample_header, random_payload)

        np.testing.assert_equal(header_encoded.shape[0], GOLAY_ENCODED_BITS)

        expected_payload_length = FIRST_LDPC_K * 2

        np.testing.assert_equal(payload_encoded.shape[0], expected_payload_length)

    def test_encoded_frame_is_binary(
        self,
        frame_constructor: FrameConstructor,
        sample_header: FrameHeader,
        random_payload: np.ndarray,
    ) -> None:
        """Encoded frame should contain only 0s and 1s."""
        header_encoded, payload_encoded = frame_constructor.encode(sample_header, random_payload)
        if not np.all((header_encoded == 0) | (header_encoded == 1)):
            pytest.fail("Header contains non-binary values")
        if not np.all((payload_encoded == 0) | (payload_encoded == 1)):
            pytest.fail("Payload contains non-binary values")


class TestFrameConstructorRoundtrip:
    """Test full frame encode/decode roundtrip."""

    def test_roundtrip_no_channel(
        self,
        frame_constructor: FrameConstructor,
        sample_header: FrameHeader,
        random_payload: np.ndarray,
    ) -> None:
        """Frame should survive encode/decode with no channel impairments."""
        header_encoded, payload_encoded = frame_constructor.encode(sample_header, random_payload)
        decoded_header = frame_constructor.decode_header(header_encoded)
        decoded_payload = frame_constructor.decode_payload(decoded_header, payload_encoded)

        np.testing.assert_equal(decoded_header.length, sample_header.length)
        np.testing.assert_array_equal(decoded_payload, random_payload)

    @given(
        length=st.integers(min_value=1, max_value=MAX_PAYLOAD_LENGTH),
        src=st.integers(min_value=0, max_value=3),
        mod_scheme=st.sampled_from(ModulationSchemes),
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_roundtrip_various_lengths(
        self,
        frame_constructor: FrameConstructor,
        length: int,
        src: int,
        mod_scheme: ModulationSchemes,
    ) -> None:
        """Property-based test to ensure any payload length up to 324 works."""
        header = FrameHeader(
            length=length,
            src=src,
            dst=0,
            mod_scheme=mod_scheme,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )
        rng = np.random.default_rng()
        payload = rng.integers(0, 2, size=length, dtype=int)

        header_encoded, payload_encoded = frame_constructor.encode(header, payload)
        decoded_header = frame_constructor.decode_header(header_encoded)
        decoded_payload = frame_constructor.decode_payload(decoded_header, payload_encoded)

        np.testing.assert_equal(len(decoded_payload), length)
        np.testing.assert_array_equal(decoded_payload, payload)


class TestDynamicPayloadLength:
    """Test dynamic payload length selection and padding."""

    @pytest.mark.parametrize(
        ("payload_length", "expected_padded_k"),
        [
            (10, FIRST_LDPC_K),
            (FIRST_LDPC_K, SECOND_LDPC_K),
            (325, SECOND_LDPC_K),
            (SECOND_LDPC_K, THIRD_LDPC_K),
            (700, THIRD_LDPC_K),
        ],
    )
    def test_payload_padding_logic(
        self,
        frame_constructor: FrameConstructor,
        payload_length: int,
        expected_padded_k: int,
    ) -> None:
        """Verify that payloads are padded to the correct LDPC k-dimension."""
        rng = np.random.default_rng()
        payload = rng.integers(0, 2, size=payload_length, dtype=int)
        header = FrameHeader(
            length=payload_length,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )

        header_encoded, payload_encoded = frame_constructor.encode(header, payload)

        np.testing.assert_equal(header_encoded.shape[0], GOLAY_ENCODED_BITS)
        expected_payload_encoded_length = expected_padded_k * 2
        np.testing.assert_equal(payload_encoded.shape[0], expected_payload_encoded_length)

        decoded_header = frame_constructor.decode_header(header_encoded)
        decoded_payload = frame_constructor.decode_payload(decoded_header, payload_encoded)

        np.testing.assert_equal(len(decoded_payload), payload_length)
        np.testing.assert_array_equal(decoded_payload.flatten().astype(int), payload.flatten().astype(int))


class TestGolayHeaderCoding:
    """Standard Golay tests remain valid."""

    def test_golay_roundtrip_no_errors(self) -> None:
        """Golay encode then decode should recover the original message."""
        golay = Golay()
        message = np.array([1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
        encoded = golay.encode(message)
        decoded = golay.decode(encoded)
        np.testing.assert_array_equal(decoded, message)

    def test_golay_multi_block_encoding(self) -> None:
        """Multi-block Golay encoding should produce correct length and decode perfectly."""
        golay = Golay()
        rng = np.random.default_rng(RNG_SEED)
        message = rng.integers(0, 2, size=GOLAY_MESSAGE_BITS)
        encoded = golay.encode(message)
        np.testing.assert_equal(len(encoded), GOLAY_ENCODED_BITS)
        np.testing.assert_array_equal(golay.decode(encoded), message)


class TestFrameConstructorWithChannel:
    """Verifies that the system works over a simulated AWGN channel."""

    def test_full_frame_qpsk_soft_decision(
        self,
        frame_constructor: FrameConstructor,
        sample_header: FrameHeader,
        random_payload: np.ndarray,
        qpsk: QPSK,
    ) -> None:
        """Test with LLRs at 15dB. Note: In real usage, header would use BPSK."""
        header_encoded, payload_encoded = frame_constructor.encode(sample_header, random_payload)

        tx_header_symbols = qpsk.bits2symbols(header_encoded)
        tx_payload_symbols = qpsk.bits2symbols(payload_encoded)

        channel = ChannelModel(ChannelConfig(snr_db=SNR_DB, seed=CHANNEL_SEED))
        rx_header_symbols = channel.apply(tx_header_symbols)
        rx_payload_symbols = channel.apply(tx_payload_symbols)

        rx_header_bits = qpsk.symbols2bits(rx_header_symbols).flatten()

        sigma_sq = qpsk.estimate_noise_variance(rx_payload_symbols)
        rx_payload_llr = qpsk.symbols2bits_soft(rx_payload_symbols, sigma_sq=sigma_sq).flatten()

        decoded_header = frame_constructor.decode_header(rx_header_bits)
        decoded_payload = frame_constructor.decode_payload(decoded_header, rx_payload_llr, soft=True)

        if not decoded_header.crc_passed:
            pytest.fail("CRC check failed after channel transmission")
        np.testing.assert_array_equal(decoded_payload, random_payload)

    def test_unsupported_payload_length_raises(self, frame_constructor: FrameConstructor) -> None:
        """Ensure we raise an error if the payload is too big for the LDPC tables."""
        header = FrameHeader(
            length=UNSUPPORTED_PAYLOAD_LENGTH,
            src=1,
            dst=2,
            mod_scheme=ModulationSchemes.QPSK,
            coding_rate=CodeRates.HALF_RATE,
            crc=0,
        )
        payload = np.zeros(UNSUPPORTED_PAYLOAD_LENGTH, dtype=int)

        with pytest.raises(ValueError, match="Unsupported payload length"):
            frame_constructor.encode(header, payload)
