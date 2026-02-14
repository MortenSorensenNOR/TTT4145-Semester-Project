import pytest
import numpy as np
from modules.channel_coding import LDPC, LDPCConfig, CodeRates


@pytest.fixture
def ldpc():
    """Create LDPC codec instance."""
    config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
    return LDPC(config)


@pytest.fixture
def random_message():
    """Generate random 324-bit message."""
    np.random.seed(42)  # reproducibility
    return np.random.randint(0, 2, size=324)


class TestLDPCEncoding:
    """Tests for LDPC encoder."""

    def test_codeword_length(self, ldpc, random_message):
        """Encoded codeword should be n bits."""
        codeword = ldpc.encode(random_message)
        assert len(codeword) == 648

    def test_systematic_bits_preserved(self, ldpc, random_message):
        """First k bits of codeword should be the original message."""
        codeword = ldpc.encode(random_message)
        assert np.array_equal(codeword[:324], random_message)

    def test_valid_codeword(self, ldpc, random_message):
        """H @ codeword should be zero (mod 2)."""
        codeword = ldpc.encode(random_message)
        syndrome = ldpc.H @ codeword % 2
        assert np.all(syndrome == 0)

    def test_different_messages_different_codewords(self, ldpc):
        """Different messages should produce different codewords."""
        msg1 = np.zeros(324, dtype=int)
        msg2 = np.ones(324, dtype=int)
        cw1 = ldpc.encode(msg1)
        cw2 = ldpc.encode(msg2)
        assert not np.array_equal(cw1, cw2)


class TestLDPCDecoding:
    """Tests for LDPC decoder."""

    def test_decode_no_noise(self, ldpc, random_message):
        """Perfect channel should decode perfectly."""
        codeword = ldpc.encode(random_message)
        
        # BPSK: 0 -> +1, 1 -> -1
        tx = 1 - 2 * codeword
        
        # Large LLRs (no noise, high confidence)
        llr = 10 * tx  # positive for 0, negative for 1
        
        decoded = ldpc.decode(llr, max_iterations=20)
        assert np.array_equal(decoded, random_message)

    @pytest.mark.parametrize("snr_db", [4.0, 5.0, 6.0])
    def test_decode_with_noise(self, ldpc, random_message, snr_db):
        """Should decode with few/no errors at reasonable SNR."""
        codeword = ldpc.encode(random_message)
        
        # BPSK modulation
        tx = 1 - 2 * codeword
        
        # AWGN channel
        snr_linear = 10 ** (snr_db / 10)
        noise_std = 1 / np.sqrt(2 * snr_linear)
        np.random.seed(123)
        rx = tx + np.random.randn(648) * noise_std
        
        # Channel LLRs
        llr = 2 * rx / (noise_std ** 2)
        
        decoded = ldpc.decode(llr, max_iterations=50)
        bit_errors = np.sum(random_message != decoded)
        
        # At 4+ dB, rate 1/2 LDPC should correct most/all errors
        assert bit_errors < 10, f"Too many errors ({bit_errors}) at {snr_db} dB"

    def test_decode_returns_k_bits(self, ldpc, random_message):
        """Decoded message should be k bits."""
        codeword = ldpc.encode(random_message)
        tx = 1 - 2 * codeword
        llr = 10 * tx
        
        decoded = ldpc.decode(llr)
        assert len(decoded) == 324


class TestLDPCStructure:
    """Tests for LDPC matrix structure."""

    def test_h_matrix_dimensions(self, ldpc):
        """H matrix should be (n-k) x n."""
        assert ldpc.H.shape == (324, 648)

    def test_h_matrix_sparse(self, ldpc):
        """H matrix should be sparse (low density)."""
        density = np.sum(ldpc.H) / (324 * 648)
        assert density < 0.05  # less than 5% ones

    def test_adjacency_lists_consistent(self, ldpc):
        """Adjacency lists should match H matrix."""
        for i, neighbors in enumerate(ldpc.check_neighbors):
            for j in neighbors:
                assert ldpc.H[i, j] == 1
        
        for j, neighbors in enumerate(ldpc.var_neighbors):
            for i in neighbors:
                assert ldpc.H[i, j] == 1
