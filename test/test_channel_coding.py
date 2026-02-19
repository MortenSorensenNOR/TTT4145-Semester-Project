import pytest
import numpy as np
from modules.channel_coding import LDPC, LDPCConfig, CodeRates, LDPC_BaseMatrix
from modules.util import ebn0_to_snr


# LDPC configuration parameters for all supported codes
LDPC_CONFIGS = [
    # (N, Z, code_rate, k, num_parity_rows)
    (648, 27, CodeRates.HALF_RATE, 324, 12),
    (648, 27, CodeRates.TWO_THIRDS_RATE, 432, 8),
    (648, 27, CodeRates.THREE_QUARTER_RATE, 486, 6),
    (648, 27, CodeRates.FIVE_SIXTH_RATE, 540, 4),
    (1296, 54, CodeRates.HALF_RATE, 648, 12),
    (1296, 54, CodeRates.TWO_THIRDS_RATE, 864, 8),
    (1296, 54, CodeRates.THREE_QUARTER_RATE, 972, 6),
    (1296, 54, CodeRates.FIVE_SIXTH_RATE, 1080, 4),
    (1944, 81, CodeRates.HALF_RATE, 972, 12),
    (1944, 81, CodeRates.TWO_THIRDS_RATE, 1296, 8),
    (1944, 81, CodeRates.THREE_QUARTER_RATE, 1458, 6),
    (1944, 81, CodeRates.FIVE_SIXTH_RATE, 1620, 4),
]


@pytest.fixture
def ldpc():
    """Create LDPC codec instance."""
    return LDPC()


@pytest.fixture
def ldpc_config():
    """Create default LDPC configuration."""
    return LDPCConfig(k=324, code_rate=CodeRates.HALF_RATE)


@pytest.fixture
def base_matrix():
    """Create LDPC base matrix generator."""
    return LDPC_BaseMatrix()


@pytest.fixture
def random_message():
    """Generate random 324-bit message."""
    np.random.seed(42)  # reproducibility
    return np.random.randint(0, 2, size=324)


class TestLDPCEncoding:
    """Tests for LDPC encoder."""

    def test_codeword_length(self, ldpc, ldpc_config, random_message):
        """Encoded codeword should be n bits."""
        codeword = ldpc.encode(random_message, ldpc_config)
        assert len(codeword) == 648

    def test_systematic_bits_preserved(self, ldpc, ldpc_config, random_message):
        """First k bits of codeword should be the original message."""
        codeword = ldpc.encode(random_message, ldpc_config)
        assert np.array_equal(codeword[:324], random_message)

    def test_valid_codeword(self, ldpc, ldpc_config, random_message):
        """H_permuted @ codeword should be zero (mod 2)."""
        codeword = ldpc.encode(random_message, ldpc_config)
        # Use the permuted H matrix that matches the encoder
        _, H_permuted = ldpc._get_encoding_structures(ldpc_config)
        syndrome = H_permuted @ codeword % 2
        assert np.all(syndrome == 0)

    def test_different_messages_different_codewords(self, ldpc, ldpc_config):
        """Different messages should produce different codewords."""
        msg1 = np.zeros(324, dtype=int)
        msg2 = np.ones(324, dtype=int)
        cw1 = ldpc.encode(msg1, ldpc_config)
        cw2 = ldpc.encode(msg2, ldpc_config)
        assert not np.array_equal(cw1, cw2)


class TestLDPCDecoding:
    """Tests for LDPC decoder."""

    def test_decode_no_noise(self, ldpc, ldpc_config, random_message):
        """Perfect channel should decode perfectly."""
        codeword = ldpc.encode(random_message, ldpc_config)

        # BPSK: 0 -> +1, 1 -> -1
        tx = 1 - 2 * codeword

        # Large LLRs (no noise, high confidence)
        llr = 10 * tx  # positive for 0, negative for 1

        decoded = ldpc.decode(llr, ldpc_config, max_iterations=20)
        assert np.array_equal(decoded, random_message)

    @pytest.mark.parametrize("ebn0_db", [4.0, 5.0, 6.0])
    def test_decode_with_noise(self, ldpc, ldpc_config, random_message, ebn0_db):
        """Should decode with few/no errors at reasonable Eb/N0.

        Uses Eb/N0 (energy per info bit / noise PSD) for accurate performance
        comparison across different code rates. For rate 1/2 BPSK, Eb/N0 = Es/N0.
        """
        codeword = ldpc.encode(random_message, ldpc_config)
        code_rate = CodeRates.HALF_RATE.value_float  # 0.5

        # BPSK modulation (1 bit per symbol)
        tx = 1 - 2 * codeword

        # Convert Eb/N0 to Es/N0 (SNR per symbol)
        # For BPSK: bits_per_symbol = 1, so Es/N0 = Eb/N0 * code_rate
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=1)

        # AWGN channel
        snr_linear = 10 ** (snr_db / 10)
        noise_std = 1 / np.sqrt(2 * snr_linear)
        np.random.seed(123)
        rx = tx + np.random.randn(648) * noise_std

        # Channel LLRs
        llr = 2 * rx / (noise_std ** 2)

        decoded = ldpc.decode(llr, ldpc_config, max_iterations=50)
        bit_errors = np.sum(random_message != decoded)

        # At Eb/N0 >= 4 dB, rate 1/2 LDPC should correct most/all errors
        assert bit_errors < 10, f"Too many errors ({bit_errors}) at Eb/N0={ebn0_db} dB"

    def test_decode_returns_k_bits(self, ldpc, ldpc_config, random_message):
        """Decoded message should be k bits."""
        codeword = ldpc.encode(random_message, ldpc_config)
        tx = 1 - 2 * codeword
        llr = 10 * tx

        decoded = ldpc.decode(llr, ldpc_config)
        assert len(decoded) == 324


class TestLDPCStructure:
    """Tests for LDPC matrix structure."""

    def test_h_matrix_dimensions(self, ldpc, ldpc_config):
        """H matrix should be (n-k) x n."""
        H, _, _ = ldpc.get_structures(ldpc_config)
        assert H.shape == (324, 648)

    def test_h_matrix_sparse(self, ldpc, ldpc_config):
        """H matrix should be sparse (low density)."""
        H, _, _ = ldpc.get_structures(ldpc_config)
        density = np.sum(H) / (324 * 648)
        assert density < 0.05  # less than 5% ones

    def test_adjacency_lists_consistent(self, ldpc, ldpc_config):
        """Adjacency lists should match H matrix."""
        H, check_neighbors, var_neighbors = ldpc.get_structures(ldpc_config)

        for i, neighbors in enumerate(check_neighbors):
            for j in neighbors:
                assert H[i, j] == 1

        for j, neighbors in enumerate(var_neighbors):
            for i in neighbors:
                assert H[i, j] == 1


class TestLDPCBaseMatrix:
    """Tests for all LDPC base matrices from 802.11 standard."""

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_matrix_dimensions(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Base matrix should have correct dimensions for each configuration."""
        matrix = base_matrix.get_matrix(code_rate, N)
        assert matrix.shape == (num_rows, 24), \
            f"Expected ({num_rows}, 24), got {matrix.shape}"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_matrix_values_in_range(self, base_matrix, N, Z, code_rate, k, num_rows):
        """All matrix values should be -1 or in range [0, Z-1]."""
        matrix = base_matrix.get_matrix(code_rate, N)
        assert np.all((matrix >= -1) & (matrix < Z)), \
            f"Matrix values out of range for N={N}, Z={Z}"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_parity_structure(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Parity portion should have staircase/dual-diagonal structure."""
        matrix = base_matrix.get_matrix(code_rate, N)
        num_info_cols = 24 - num_rows
        parity_part = matrix[:, num_info_cols:]

        # Check that parity part has appropriate structure (not all -1)
        non_neg_count = np.sum(parity_part >= 0)
        assert non_neg_count >= num_rows, \
            f"Parity part has too few non-negative entries"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_expanded_matrix_dimensions(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Expanded H matrix should have correct dimensions."""
        matrix = base_matrix.get_matrix(code_rate, N)

        # Expand base matrix to full H
        H = np.zeros((num_rows * Z, 24 * Z), dtype=int)
        for i in range(num_rows):
            for j in range(24):
                shift = matrix[i, j]
                if shift != -1:
                    for idx in range(Z):
                        H[i * Z + idx, j * Z + (idx + shift) % Z] = 1

        expected_rows = N - k  # n - k parity check equations
        expected_cols = N
        assert H.shape == (expected_rows, expected_cols), \
            f"Expected ({expected_rows}, {expected_cols}), got {H.shape}"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_expanded_matrix_sparsity(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Expanded H matrix should be sparse (LDPC property)."""
        matrix = base_matrix.get_matrix(code_rate, N)

        # Expand base matrix
        H = np.zeros((num_rows * Z, 24 * Z), dtype=int)
        for i in range(num_rows):
            for j in range(24):
                shift = matrix[i, j]
                if shift != -1:
                    for idx in range(Z):
                        H[i * Z + idx, j * Z + (idx + shift) % Z] = 1

        density = np.sum(H) / H.size
        assert density < 0.05, f"H matrix too dense: {density:.3f}"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_each_row_has_connections(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Each row in base matrix should have at least 2 non-negative entries."""
        matrix = base_matrix.get_matrix(code_rate, N)
        for i in range(num_rows):
            row_connections = np.sum(matrix[i] >= 0)
            assert row_connections >= 2, \
                f"Row {i} has only {row_connections} connections"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_each_column_has_connections(self, base_matrix, N, Z, code_rate, k, num_rows):
        """Each column in base matrix should have at least 1 non-negative entry."""
        matrix = base_matrix.get_matrix(code_rate, N)
        for j in range(24):
            col_connections = np.sum(matrix[:, j] >= 0)
            assert col_connections >= 1, \
                f"Column {j} has no connections"


class TestLDPCBaseMatrixCodeRateConsistency:
    """Verify code rate calculations are consistent."""

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_code_rate_matches_dimensions(self, N, Z, code_rate, k, num_rows):
        """Verify k/n matches the expected code rate."""
        actual_rate = k / N
        expected_rates = {
            CodeRates.HALF_RATE: 0.5,
            CodeRates.TWO_THIRDS_RATE: 2/3,
            CodeRates.THREE_QUARTER_RATE: 0.75,
            CodeRates.FIVE_SIXTH_RATE: 5/6,
        }
        expected = expected_rates[code_rate]
        assert abs(actual_rate - expected) < 0.01, \
            f"Rate mismatch: k/n={actual_rate:.3f}, expected {expected:.3f}"

    @pytest.mark.parametrize("N,Z,code_rate,k,num_rows", LDPC_CONFIGS)
    def test_parity_bits_match(self, N, Z, code_rate, k, num_rows):
        """Verify n - k = num_rows * Z."""
        parity_bits = N - k
        expected_parity = num_rows * Z
        assert parity_bits == expected_parity, \
            f"Parity bits mismatch: {parity_bits} != {expected_parity}"
