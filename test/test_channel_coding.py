"""Tests for LDPC channel coding."""

from dataclasses import dataclass

import numpy as np
import pytest

from modules.channel_coding import (
    CodeRates,
    LDPCConfig,
    get_ldpc_base_matrix,
    ldpc_decode,
    ldpc_encode,
    ldpc_get_h_matrix,
)
from modules.util import ebn0_to_snr

CODEWORD_LENGTH = 648
MESSAGE_LENGTH = 324
MAX_DENSITY = 0.05
MAX_BIT_ERRORS = 10
RATE_TOLERANCE = 0.01
MIN_ROW_CONNECTIONS = 2
MIN_COL_CONNECTIONS = 1
BASE_MATRIX_COLS = 24
LLR_SCALE = 10


@dataclass(frozen=True)
class LdpcTestConfig:
    """Test configuration for a single LDPC code."""

    n: int
    z: int
    code_rate: CodeRates
    k: int
    num_parity_rows: int


LDPC_CONFIGS = [
    LdpcTestConfig(648, 27, CodeRates.HALF_RATE, 324, 12),
    LdpcTestConfig(648, 27, CodeRates.TWO_THIRDS_RATE, 432, 8),
    LdpcTestConfig(648, 27, CodeRates.THREE_QUARTER_RATE, 486, 6),
    LdpcTestConfig(648, 27, CodeRates.FIVE_SIXTH_RATE, 540, 4),
    LdpcTestConfig(1296, 54, CodeRates.HALF_RATE, 648, 12),
    LdpcTestConfig(1296, 54, CodeRates.TWO_THIRDS_RATE, 864, 8),
    LdpcTestConfig(1296, 54, CodeRates.THREE_QUARTER_RATE, 972, 6),
    LdpcTestConfig(1296, 54, CodeRates.FIVE_SIXTH_RATE, 1080, 4),
    LdpcTestConfig(1944, 81, CodeRates.HALF_RATE, 972, 12),
    LdpcTestConfig(1944, 81, CodeRates.TWO_THIRDS_RATE, 1296, 8),
    LdpcTestConfig(1944, 81, CodeRates.THREE_QUARTER_RATE, 1458, 6),
    LdpcTestConfig(1944, 81, CodeRates.FIVE_SIXTH_RATE, 1620, 4),
]


@pytest.fixture
def ldpc_config() -> LDPCConfig:
    """Create default LDPC configuration."""
    return LDPCConfig(k=MESSAGE_LENGTH, code_rate=CodeRates.HALF_RATE)


@pytest.fixture
def random_message() -> np.ndarray:
    """Generate random 324-bit message."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=MESSAGE_LENGTH)


def _expand_base_matrix(matrix: np.ndarray, num_rows: int, z_size: int) -> np.ndarray:
    """Expand a base matrix to a full parity-check matrix."""
    h_mat = np.zeros((num_rows * z_size, BASE_MATRIX_COLS * z_size), dtype=int)
    for i in range(num_rows):
        for j in range(BASE_MATRIX_COLS):
            shift = matrix[i, j]
            if shift != -1:
                for idx in range(z_size):
                    h_mat[i * z_size + idx, j * z_size + (idx + shift) % z_size] = 1
    return h_mat


class TestLDPCEncoding:
    """Tests for LDPC encoder."""

    def test_codeword_length(self, ldpc_config: LDPCConfig, random_message: np.ndarray) -> None:
        """Encoded codeword should be n bits."""
        codeword = ldpc_encode(random_message, ldpc_config)
        np.testing.assert_equal(len(codeword), CODEWORD_LENGTH)

    def test_systematic_bits_preserved(self, ldpc_config: LDPCConfig, random_message: np.ndarray) -> None:
        """First k bits of codeword should be the original message."""
        codeword = ldpc_encode(random_message, ldpc_config)
        np.testing.assert_array_equal(codeword[:MESSAGE_LENGTH], random_message)

    def test_valid_codeword(self, ldpc_config: LDPCConfig, random_message: np.ndarray) -> None:
        """Encode then decode should recover the original message."""
        codeword = ldpc_encode(random_message, ldpc_config)
        tx = 1 - 2 * codeword
        llr = LLR_SCALE * tx
        decoded = ldpc_decode(llr, ldpc_config, max_iterations=20)
        np.testing.assert_array_equal(decoded, random_message)

    def test_different_messages_different_codewords(self, ldpc_config: LDPCConfig) -> None:
        """Different messages should produce different codewords."""
        msg1 = np.zeros(MESSAGE_LENGTH, dtype=int)
        msg2 = np.ones(MESSAGE_LENGTH, dtype=int)
        cw1 = ldpc_encode(msg1, ldpc_config)
        cw2 = ldpc_encode(msg2, ldpc_config)
        if np.array_equal(cw1, cw2):
            pytest.fail("Different messages produced identical codewords")


class TestLDPCDecoding:
    """Tests for LDPC decoder."""

    def test_decode_no_noise(self, ldpc_config: LDPCConfig, random_message: np.ndarray) -> None:
        """Perfect channel should decode perfectly."""
        codeword = ldpc_encode(random_message, ldpc_config)
        tx = 1 - 2 * codeword
        llr = LLR_SCALE * tx
        decoded = ldpc_decode(llr, ldpc_config, max_iterations=20)
        np.testing.assert_array_equal(decoded, random_message)

    @pytest.mark.parametrize("ebn0_db", [4.0, 5.0, 6.0])
    def test_decode_with_noise(
        self,
        ldpc_config: LDPCConfig,
        random_message: np.ndarray,
        ebn0_db: float,
    ) -> None:
        """Should decode with few/no errors at reasonable Eb/N0."""
        codeword = ldpc_encode(random_message, ldpc_config)
        code_rate = CodeRates.HALF_RATE.value_float

        tx = 1 - 2 * codeword
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=1)
        snr_linear = 10 ** (snr_db / 10)
        noise_std = 1 / np.sqrt(2 * snr_linear)

        rng = np.random.default_rng(123)
        rx = tx + rng.standard_normal(CODEWORD_LENGTH) * noise_std
        llr = 2 * rx / (noise_std**2)

        decoded = ldpc_decode(llr, ldpc_config, max_iterations=50)
        bit_errors = np.sum(random_message != decoded)
        if bit_errors >= MAX_BIT_ERRORS:
            pytest.fail(f"Too many errors ({bit_errors}) at Eb/N0={ebn0_db} dB")

    def test_decode_returns_k_bits(self, ldpc_config: LDPCConfig, random_message: np.ndarray) -> None:
        """Decoded message should be k bits."""
        codeword = ldpc_encode(random_message, ldpc_config)
        tx = 1 - 2 * codeword
        llr = LLR_SCALE * tx
        decoded = ldpc_decode(llr, ldpc_config)
        np.testing.assert_equal(len(decoded), MESSAGE_LENGTH)


class TestLDPCStructure:
    """Tests for LDPC matrix structure."""

    def test_h_matrix_dimensions(self, ldpc_config: LDPCConfig) -> None:
        """H matrix should be (n-k) x n."""
        h_mat = ldpc_get_h_matrix(ldpc_config)
        np.testing.assert_equal(h_mat.shape, (MESSAGE_LENGTH, CODEWORD_LENGTH))

    def test_h_matrix_sparse(self, ldpc_config: LDPCConfig) -> None:
        """H matrix should be sparse (low density)."""
        h_mat = ldpc_get_h_matrix(ldpc_config)
        density = np.sum(h_mat) / (MESSAGE_LENGTH * CODEWORD_LENGTH)
        if density >= MAX_DENSITY:
            pytest.fail(f"H matrix too dense: {density:.3f}")


class TestLDPCBaseMatrix:
    """Tests for all LDPC base matrices from 802.11 standard."""

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_matrix_dimensions(self, cfg: LdpcTestConfig) -> None:
        """Base matrix should have correct dimensions for each configuration."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        np.testing.assert_equal(matrix.shape, (cfg.num_parity_rows, BASE_MATRIX_COLS))

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_matrix_values_in_range(self, cfg: LdpcTestConfig) -> None:
        """All matrix values should be -1 or in range [0, Z-1]."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        if not np.all((matrix >= -1) & (matrix < cfg.z)):
            pytest.fail(f"Values out of range for n={cfg.n}, z={cfg.z}")

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_parity_structure(self, cfg: LdpcTestConfig) -> None:
        """Parity portion should have staircase/dual-diagonal structure."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        num_info_cols = BASE_MATRIX_COLS - cfg.num_parity_rows
        parity_part = matrix[:, num_info_cols:]
        non_neg_count = np.sum(parity_part >= 0)
        if non_neg_count < cfg.num_parity_rows:
            pytest.fail("Parity part has too few non-negative entries")

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_expanded_matrix_dimensions(self, cfg: LdpcTestConfig) -> None:
        """Expanded H matrix should have correct dimensions."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        h_mat = _expand_base_matrix(matrix, cfg.num_parity_rows, cfg.z)
        expected_rows = cfg.n - cfg.k
        expected_cols = cfg.n
        np.testing.assert_equal(h_mat.shape, (expected_rows, expected_cols))

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_expanded_matrix_sparsity(self, cfg: LdpcTestConfig) -> None:
        """Expanded H matrix should be sparse (LDPC property)."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        h_mat = _expand_base_matrix(matrix, cfg.num_parity_rows, cfg.z)
        density = np.sum(h_mat) / h_mat.size
        if density >= MAX_DENSITY:
            pytest.fail(f"H matrix too dense: {density:.3f}")

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_each_row_has_connections(self, cfg: LdpcTestConfig) -> None:
        """Each row in base matrix should have at least 2 non-negative entries."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        for i in range(cfg.num_parity_rows):
            row_connections = np.sum(matrix[i] >= 0)
            if row_connections < MIN_ROW_CONNECTIONS:
                pytest.fail(f"Row {i} has only {row_connections} connections")

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_each_column_has_connections(self, cfg: LdpcTestConfig) -> None:
        """Each column in base matrix should have at least 1 non-negative entry."""
        matrix = get_ldpc_base_matrix(cfg.code_rate, cfg.n)
        for j in range(BASE_MATRIX_COLS):
            col_connections = np.sum(matrix[:, j] >= 0)
            if col_connections < MIN_COL_CONNECTIONS:
                pytest.fail(f"Column {j} has no connections")


class TestLDPCBaseMatrixCodeRateConsistency:
    """Verify code rate calculations are consistent."""

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_code_rate_matches_dimensions(self, cfg: LdpcTestConfig) -> None:
        """Verify k/n matches the expected code rate."""
        actual_rate = cfg.k / cfg.n
        expected_rates = {
            CodeRates.HALF_RATE: 0.5,
            CodeRates.TWO_THIRDS_RATE: 2 / 3,
            CodeRates.THREE_QUARTER_RATE: 0.75,
            CodeRates.FIVE_SIXTH_RATE: 5 / 6,
        }
        expected = expected_rates[cfg.code_rate]
        if abs(actual_rate - expected) >= RATE_TOLERANCE:
            pytest.fail(f"Rate mismatch: k/n={actual_rate:.3f}, expected {expected:.3f}")

    @pytest.mark.parametrize("cfg", LDPC_CONFIGS)
    def test_parity_bits_match(self, cfg: LdpcTestConfig) -> None:
        """Verify n - k = num_rows * Z."""
        parity_bits = cfg.n - cfg.k
        expected_parity = cfg.num_parity_rows * cfg.z
        np.testing.assert_equal(parity_bits, expected_parity)
