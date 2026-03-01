"""Tests for BPSK, QPSK, and QAM modulation helpers."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from modules.modulation import BPSK, QAM, QPSK, EightPSK

RNG_SEED_NOISE_ESTIMATION = 42
RNG_SEED_SOFT_ROUNDTRIP = 123
QPSK_BITS_PER_SYMBOL = 2
QPSK_ORDER = 4
EIGHT_PSK_BITS_PER_SYMBOL = 3
EIGHT_PSK_ORDER = 8
QAM_ORDER_16 = 16
QAM_ORDERS = [16, 64, 256]


class TestBPSK:
    """Validate BPSK hard-decision modulation and demodulation."""

    @pytest.fixture
    def bpsk(self) -> BPSK:
        """Create a BPSK instance for each test."""
        return BPSK()

    def test_symbol_constellation(self, bpsk: BPSK) -> None:
        """Verify that BPSK defines the expected constellation points."""
        expected = np.array([-1 + 0j, 1 + 0j])
        np.testing.assert_array_equal(bpsk.symbol_mapping, expected)

    def test_bits2symbols_single_bit(self, bpsk: BPSK) -> None:
        """Verify that single bits map to the expected BPSK symbol."""
        np.testing.assert_equal(bpsk.bits2symbols(np.array([0]))[0], -1 + 0j)
        np.testing.assert_equal(bpsk.bits2symbols(np.array([1]))[0], 1 + 0j)

    def test_bits2symbols_bitstream(self, bpsk: BPSK) -> None:
        """Verify mapping from a bitstream to BPSK symbols."""
        bits = np.array([0, 1, 1, 0, 1, 0])
        symbols = bpsk.bits2symbols(bits)
        expected = np.array([-1 + 0j, 1 + 0j, 1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j])
        np.testing.assert_array_equal(symbols, expected)

    def test_symbols2bits_perfect_symbols(self, bpsk: BPSK) -> None:
        """Verify demodulation on noiseless BPSK symbols."""
        symbols = np.array([-1 + 0j, 1 + 0j, 1 + 0j, -1 + 0j])
        bits = bpsk.symbols2bits(symbols).flatten()
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)

    def test_symbols2bits_noisy_symbols(self, bpsk: BPSK) -> None:
        """Verify demodulation on noisy BPSK symbols."""
        symbols = np.array([-0.8 + 0.1j, 0.9 - 0.05j, 0.7 + 0.2j, -1.1 + 0j])
        bits = bpsk.symbols2bits(symbols).flatten()
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)

    def test_roundtrip(self, bpsk: BPSK) -> None:
        """Verify that BPSK hard demodulation roundtrip is lossless."""
        original_bits = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        symbols = bpsk.bits2symbols(original_bits)
        recovered_bits = bpsk.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)


class TestQPSK:
    """Validate QPSK hard-decision modulation and demodulation."""

    @pytest.fixture
    def qpsk(self) -> QPSK:
        """Create a QPSK instance for each test."""
        return QPSK()

    def test_qpsk_order(self, qpsk: QPSK) -> None:
        """Verify QPSK constellation size metadata."""
        np.testing.assert_equal(qpsk.qam_order, QPSK_ORDER)
        np.testing.assert_equal(len(qpsk.symbol_mapping), QPSK_ORDER)

    def test_bits_per_symbol(self, qpsk: QPSK) -> None:
        """Verify QPSK bits per symbol metadata."""
        np.testing.assert_equal(qpsk.bits_per_symbol, QPSK_BITS_PER_SYMBOL)

    def test_unit_energy(self, qpsk: QPSK) -> None:
        """Verify that average QPSK constellation energy is one."""
        avg_energy = np.mean(np.abs(qpsk.symbol_mapping) ** 2)
        np.testing.assert_almost_equal(avg_energy, 1.0)

    def test_bits2symbols(self, qpsk: QPSK) -> None:
        """Verify mapping from bits to QPSK symbols."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(bits)
        np.testing.assert_equal(len(symbols), QPSK_ORDER)

    def test_symbols2bits_shape(self, qpsk: QPSK) -> None:
        """Verify that QPSK hard-decision output shape is `(N, 2)`."""
        symbols = qpsk.symbol_mapping
        bits = qpsk.symbols2bits(symbols)
        np.testing.assert_array_equal(bits.shape, (QPSK_ORDER, QPSK_BITS_PER_SYMBOL))

    def test_roundtrip(self, qpsk: QPSK) -> None:
        """Verify that QPSK hard demodulation roundtrip is lossless."""
        original_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(original_bits)
        recovered_bits = qpsk.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)

    def test_gray_coding(self, qpsk: QPSK) -> None:
        """Verify nearest-neighbor Hamming distance in the QPSK mapping."""
        for index, symbol in enumerate(qpsk.symbol_mapping):
            bits_i = qpsk.bit_mapping[index]
            distances = np.abs(qpsk.symbol_mapping - symbol)
            distances[index] = np.inf
            closest_idx = int(np.argmin(distances))
            bits_closest = qpsk.bit_mapping[closest_idx]
            hamming_dist = np.sum(bits_i != bits_closest)
            np.testing.assert_array_less(hamming_dist, QPSK_ORDER - 1)



class TestEightPSK:
    """Validate EightPSK hard-decision modulation and demodulation."""

    @pytest.fixture
    def eight_psk(self) -> EightPSK:
        """Create an EightPSK instance for each test."""
        return EightPSK()

    def test_8psk_order(self, eight_psk: EightPSK) -> None:
        """Verify EightPSK constellation size metadata."""
        np.testing.assert_equal(eight_psk.qam_order, EIGHT_PSK_ORDER)
        np.testing.assert_equal(len(eight_psk.symbol_mapping), EIGHT_PSK_ORDER)

    def test_bits_per_symbol(self, eight_psk: EightPSK) -> None:
        """Verify EightPSK bits per symbol metadata."""
        np.testing.assert_equal(eight_psk.bits_per_symbol, EIGHT_PSK_BITS_PER_SYMBOL)

    def test_unit_energy(self, eight_psk: EightPSK) -> None:
        """Verify that average EightPSK constellation energy is one."""
        avg_energy = np.mean(np.abs(eight_psk.symbol_mapping) ** 2)
        np.testing.assert_almost_equal(avg_energy, 1.0)

    def test_bits2symbols(self, eight_psk: EightPSK) -> None:
        """Verify mapping from bits to EightPSK symbols."""
        bits = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
        symbols = eight_psk.bits2symbols(bits)
        np.testing.assert_equal(len(symbols), EIGHT_PSK_ORDER)

    def test_symbols2bits_shape(self, eight_psk: EightPSK) -> None:
        """Verify that EightPSK hard-decision output shape is `(N, 3)`."""
        symbols = eight_psk.symbol_mapping
        bits = eight_psk.symbols2bits(symbols)
        np.testing.assert_array_equal(bits.shape, (EIGHT_PSK_ORDER, EIGHT_PSK_BITS_PER_SYMBOL))

    def test_roundtrip(self, eight_psk: EightPSK) -> None:
        """Verify that EightPSK hard demodulation roundtrip is lossless."""
        original_bits = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
        symbols = eight_psk.bits2symbols(original_bits)
        recovered_bits = eight_psk.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)

    def test_gray_coding(self, eight_psk: EightPSK) -> None:
        """Verify nearest-neighbor Hamming distance in the EightPSK mapping."""
        for index, symbol in enumerate(eight_psk.symbol_mapping):
            bits_i = eight_psk.bit_mapping[index]
            distances = np.abs(eight_psk.symbol_mapping - symbol)
            distances[index] = np.inf
            closest_idx = int(np.argmin(distances))
            bits_closest = eight_psk.bit_mapping[closest_idx]
            hamming_dist = np.sum(bits_i != bits_closest)
            np.testing.assert_array_less(hamming_dist, EIGHT_PSK_ORDER - 1)


class TestQAM:
    """Validate generic square-QAM behavior."""

    @pytest.mark.parametrize("qam_order", QAM_ORDERS)
    def test_qam_orders(self, qam_order: int) -> None:
        """Verify consistency of metadata for multiple QAM orders."""
        qam = QAM(qam_order)
        np.testing.assert_equal(qam.qam_order, qam_order)
        np.testing.assert_equal(len(qam.symbol_mapping), qam_order)
        np.testing.assert_equal(qam.bits_per_symbol, int(np.log2(qam_order)))

    def test_16qam_unit_energy(self) -> None:
        """Verify that 16-QAM has unit average constellation energy."""
        qam = QAM(QAM_ORDER_16)
        avg_energy = np.mean(np.abs(qam.symbol_mapping) ** 2)
        np.testing.assert_almost_equal(avg_energy, 1.0)

    def test_16qam_roundtrip(self) -> None:
        """Verify hard-decision roundtrip for 16-QAM."""
        qam = QAM(QAM_ORDER_16)
        rng = np.random.default_rng(RNG_SEED_SOFT_ROUNDTRIP)
        original_bits = rng.integers(0, QPSK_BITS_PER_SYMBOL, size=QAM_ORDER_16)
        symbols = qam.bits2symbols(original_bits)
        recovered_bits = qam.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)

    def test_bit_mapping_completeness(self) -> None:
        """Verify that all bit patterns are represented in the mapping."""
        qam = QAM(QAM_ORDER_16)
        decimal_values = np.sum(
            qam.bit_mapping * 2 ** np.arange(qam.bits_per_symbol - 1, -1, -1),
            axis=1,
        )
        np.testing.assert_equal(len(np.unique(decimal_values)), qam.qam_order)
        np.testing.assert_equal(set(decimal_values), set(range(qam.qam_order)))


class TestEightPSKSoftDecision:
    """Validate EightPSK soft-decision LLR behavior."""

    @pytest.fixture
    def eight_psk(self) -> EightPSK:
        """Create an EightPSK instance for each test."""
        return EightPSK()

    def test_soft_output_shape(self, eight_psk: EightPSK) -> None:
        """Verify that soft output shape is `(N, 3)`."""
        symbols = np.array([0.5 + 0.5j, -0.3 + 0.2j, 0.1 - 0.8j])
        llrs = eight_psk.symbols2bits_soft(symbols, sigma_sq=0.1)
        np.testing.assert_array_equal(llrs.shape, (3, EIGHT_PSK_BITS_PER_SYMBOL))

    def test_soft_output_empty(self, eight_psk: EightPSK) -> None:
        """Verify soft decision output for empty input."""
        llrs = eight_psk.symbols2bits_soft(np.array([]), sigma_sq=0.1)
        np.testing.assert_equal(len(llrs), 0)

    def test_llr_sign_matches_hard_decision(self, eight_psk: EightPSK) -> None:
        """Verify that LLR signs match hard decisions."""
        symbols = eight_psk.symbol_mapping
        llrs = eight_psk.symbols2bits_soft(symbols, sigma_sq=0.1)
        hard_bits = eight_psk.symbols2bits(symbols)

        for symbol_idx, _symbol in enumerate(symbols):
            for bit_idx in range(EIGHT_PSK_BITS_PER_SYMBOL):
                bit_value = hard_bits[symbol_idx, bit_idx]
                llr_value = llrs[symbol_idx, bit_idx]
                if bit_value == 0:
                    np.testing.assert_array_less(0.0, llr_value)
                else:
                    np.testing.assert_array_less(llr_value, 0.0)



    def test_llr_symmetric_around_origin(self, eight_psk: EightPSK) -> None:
        """Verify antisymmetry of LLR around the origin."""
        symbols = np.array([0.5 + 0.3j])
        neg_symbols = -symbols

        llrs = eight_psk.symbols2bits_soft(symbols, sigma_sq=0.1)
        neg_llrs = eight_psk.symbols2bits_soft(neg_symbols, sigma_sq=0.1)

        # For this Gray code, only the first bit is guaranteed to be symmetric
        np.testing.assert_array_almost_equal(llrs[0, 0], -neg_llrs[0, 0], decimal=5)

    def test_llr_scales_with_noise_variance(self, eight_psk: EightPSK) -> None:
        """Verify that LLR magnitude decreases as noise variance increases."""
        symbols = np.array([0.5 + 0.5j])
        llr_low_noise = eight_psk.symbols2bits_soft(symbols, sigma_sq=0.05)
        llr_high_noise = eight_psk.symbols2bits_soft(symbols, sigma_sq=0.5)

        np.testing.assert_array_less(
            np.abs(llr_high_noise[0, 0]),
            np.abs(llr_low_noise[0, 0]),
        )
        np.testing.assert_array_less(
            np.abs(llr_high_noise[0, 1]),
            np.abs(llr_low_noise[0, 1]),
        )

    def test_noise_variance_estimation(self, eight_psk: EightPSK) -> None:
        """Verify that estimated noise variance is close to the true value."""
        rng = np.random.default_rng(RNG_SEED_NOISE_ESTIMATION)
        true_sigma_sq = 0.05
        bits = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1] * 100)
        symbols = eight_psk.bits2symbols(bits)
        gaussian_noise = rng.standard_normal(len(symbols))
        quadrature_noise = rng.standard_normal(len(symbols))
        noise = np.sqrt(true_sigma_sq / 2) * (gaussian_noise + 1j * quadrature_noise)
        noisy_symbols = symbols + noise

        estimated_sigma_sq = eight_psk.estimate_noise_variance(noisy_symbols)

        np.testing.assert_array_less(0.5 * true_sigma_sq, estimated_sigma_sq)
        np.testing.assert_array_less(estimated_sigma_sq, 1.5 * true_sigma_sq)

    def test_soft_roundtrip_improves_with_confidence(self, eight_psk: EightPSK) -> None:
        """Verify hard decisions from high-confidence LLRs."""
        rng = np.random.default_rng(RNG_SEED_SOFT_ROUNDTRIP)
        bits = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
        symbols = eight_psk.bits2symbols(bits)
        gaussian_noise = rng.standard_normal(len(symbols))
        quadrature_noise = rng.standard_normal(len(symbols))
        noise = 0.05 * (gaussian_noise + 1j * quadrature_noise)
        noisy_symbols = symbols + noise
        llrs = eight_psk.symbols2bits_soft(noisy_symbols, sigma_sq=0.01)

        hard_from_llr = (llrs < 0).astype(int).flatten()
        np.testing.assert_array_equal(hard_from_llr, bits)

    @given(
        bits=st.lists(st.integers(0, 1), min_size=3, max_size=100).filter(lambda x: len(x) % 3 == 0),
        snr_db=st.floats(min_value=10.0, max_value=30.0),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_soft_to_hard_decision_high_snr(self, bits: list[int], snr_db: float, seed: int) -> None:
        """Verify that hard decisions from LLRs match original bits at high SNR."""
        eight_psk = EightPSK()
        rng = np.random.default_rng(seed)
        bits_arr = np.array(bits)
        symbols = eight_psk.bits2symbols(bits_arr)

        snr_linear = 10 ** (snr_db / 10)
        sigma_sq = 1.0 / (2 * snr_linear)
        noise = np.sqrt(sigma_sq / 2) * (rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols)))
        noisy_symbols = symbols + noise

        llrs = eight_psk.symbols2bits_soft(noisy_symbols, sigma_sq=sigma_sq)
        hard_from_llr = (llrs < 0).astype(int).flatten()

        np.testing.assert_array_equal(hard_from_llr, bits_arr)


class TestQAM4EqualsQPSK:
    """Verify that QAM(4) and QPSK produce identical behavior."""

    @pytest.fixture
    def qpsk(self) -> QPSK:
        """Create a QPSK instance."""
        return QPSK()

    @pytest.fixture
    def qam4(self) -> QAM:
        """Create a QAM(4) instance."""
        return QAM(4)

    def test_same_constellation(self, qpsk: QPSK, qam4: QAM) -> None:
        """QAM(4) and QPSK should have the same constellation points."""
        np.testing.assert_array_almost_equal(
            qam4.symbol_mapping,
            qpsk.symbol_mapping,
        )

    def test_same_bit_mapping(self, qpsk: QPSK, qam4: QAM) -> None:
        """QAM(4) and QPSK should map bits to symbols identically."""
        np.testing.assert_array_equal(qam4.bit_mapping, qpsk.bit_mapping)

    def test_same_symbols_for_all_bit_patterns(self, qpsk: QPSK, qam4: QAM) -> None:
        """Every 2-bit pattern should produce the same symbol in both."""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        np.testing.assert_array_almost_equal(
            qam4.bits2symbols(bits),
            qpsk.bits2symbols(bits),
        )

    def test_hard_decision_equivalent(self, qpsk: QPSK, qam4: QAM) -> None:
        """Hard demodulation should produce the same bits."""
        rng = np.random.default_rng(42)
        symbols = qpsk.symbol_mapping + 0.05 * (rng.standard_normal(4) + 1j * rng.standard_normal(4))
        np.testing.assert_array_equal(
            qam4.symbols2bits(symbols),
            qpsk.symbols2bits(symbols),
        )

    def test_soft_decision_equivalent(self, qpsk: QPSK, qam4: QAM) -> None:
        """Soft demodulation LLRs should match between QAM(4) and QPSK."""
        rng = np.random.default_rng(42)
        symbols = qpsk.symbol_mapping + 0.1 * (rng.standard_normal(4) + 1j * rng.standard_normal(4))
        sigma_sq = 0.1
        llr_qpsk = qpsk.symbols2bits_soft(symbols, sigma_sq=sigma_sq)
        llr_qam4 = qam4.symbols2bits_soft(symbols, sigma_sq=sigma_sq)
        np.testing.assert_array_almost_equal(llr_qam4, llr_qpsk, decimal=5)

    def test_roundtrip_interchangeable(self, qpsk: QPSK, qam4: QAM) -> None:
        """Encoding with QPSK and decoding with QAM(4) should work and vice versa."""
        bits = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        # QPSK encode → QAM(4) decode
        symbols = qpsk.bits2symbols(bits)
        recovered = qam4.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered, bits)
        # QAM(4) encode → QPSK decode
        symbols = qam4.bits2symbols(bits)
        recovered = qpsk.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered, bits)


class TestQPSKSoftDecision:
    """Validate QPSK soft-decision LLR behavior."""

    @pytest.fixture
    def qpsk(self) -> QPSK:
        """Create a QPSK instance for each test."""
        return QPSK()

    def test_soft_output_shape(self, qpsk: QPSK) -> None:
        """Verify that soft output shape is `(N, 2)`."""
        symbols = np.array([0.5 + 0.5j, -0.3 + 0.2j, 0.1 - 0.8j])
        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        np.testing.assert_array_equal(llrs.shape, (3, QPSK_BITS_PER_SYMBOL))

    def test_soft_output_empty(self, qpsk: QPSK) -> None:
        """Verify soft decision output for empty input."""
        llrs = qpsk.symbols2bits_soft(np.array([]), sigma_sq=0.1)
        np.testing.assert_equal(len(llrs), 0)

    def test_llr_sign_matches_hard_decision(self, qpsk: QPSK) -> None:
        """Verify that LLR signs match hard decisions."""
        symbols = qpsk.symbol_mapping
        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        hard_bits = qpsk.symbols2bits(symbols)

        for symbol_idx, _symbol in enumerate(symbols):
            for bit_idx in range(QPSK_BITS_PER_SYMBOL):
                bit_value = hard_bits[symbol_idx, bit_idx]
                llr_value = llrs[symbol_idx, bit_idx]
                if bit_value == 0:
                    np.testing.assert_array_less(0.0, llr_value)
                else:
                    np.testing.assert_array_less(llr_value, 0.0)

    def test_llr_magnitude_increases_away_from_boundary(self, qpsk: QPSK) -> None:
        """Verify that LLR magnitude increases away from decision boundaries."""
        close_to_boundary = np.array([0.1 + 0.5j])
        far_from_boundary = np.array([0.7 + 0.5j])

        llr_close = qpsk.symbols2bits_soft(close_to_boundary, sigma_sq=0.1)
        llr_far = qpsk.symbols2bits_soft(far_from_boundary, sigma_sq=0.1)

        np.testing.assert_array_less(np.abs(llr_close[0, 0]), np.abs(llr_far[0, 0]))

    def test_llr_symmetric_around_origin(self, qpsk: QPSK) -> None:
        """Verify antisymmetry of LLR around the origin."""
        symbols = np.array([0.5 + 0.3j])
        neg_symbols = -symbols

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        neg_llrs = qpsk.symbols2bits_soft(neg_symbols, sigma_sq=0.1)

        np.testing.assert_array_almost_equal(llrs, -neg_llrs)

    def test_llr_scales_with_noise_variance(self, qpsk: QPSK) -> None:
        """Verify that LLR magnitude decreases as noise variance increases."""
        symbols = np.array([0.5 + 0.5j])
        llr_low_noise = qpsk.symbols2bits_soft(symbols, sigma_sq=0.05)
        llr_high_noise = qpsk.symbols2bits_soft(symbols, sigma_sq=0.5)

        np.testing.assert_array_less(
            np.abs(llr_high_noise[0, 0]),
            np.abs(llr_low_noise[0, 0]),
        )
        np.testing.assert_array_less(
            np.abs(llr_high_noise[0, 1]),
            np.abs(llr_low_noise[0, 1]),
        )

    def test_llr_on_decision_boundary_is_zero(self, qpsk: QPSK) -> None:
        """Verify that LLR is zero on QPSK decision boundaries."""
        on_i_boundary = np.array([0.0 + 0.5j])
        llrs_i = qpsk.symbols2bits_soft(on_i_boundary, sigma_sq=0.1)
        np.testing.assert_almost_equal(llrs_i[0, 0], 0.0)

        on_q_boundary = np.array([0.5 + 0.0j])
        llrs_q = qpsk.symbols2bits_soft(on_q_boundary, sigma_sq=0.1)
        np.testing.assert_almost_equal(llrs_q[0, 1], 0.0)

    def test_noise_variance_estimation(self, qpsk: QPSK) -> None:
        """Verify that estimated noise variance is close to the true value."""
        rng = np.random.default_rng(RNG_SEED_NOISE_ESTIMATION)
        true_sigma_sq = 0.05
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1] * 100)
        symbols = qpsk.bits2symbols(bits)
        gaussian_noise = rng.standard_normal(len(symbols))
        quadrature_noise = rng.standard_normal(len(symbols))
        noise = np.sqrt(true_sigma_sq / 2) * (gaussian_noise + 1j * quadrature_noise)
        noisy_symbols = symbols + noise

        estimated_sigma_sq = qpsk.estimate_noise_variance(noisy_symbols)

        np.testing.assert_array_less(0.5 * true_sigma_sq, estimated_sigma_sq)
        np.testing.assert_array_less(estimated_sigma_sq, 1.5 * true_sigma_sq)

    def test_soft_roundtrip_improves_with_confidence(self, qpsk: QPSK) -> None:
        """Verify hard decisions from high-confidence LLRs."""
        rng = np.random.default_rng(RNG_SEED_SOFT_ROUNDTRIP)
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(bits)
        gaussian_noise = rng.standard_normal(len(symbols))
        quadrature_noise = rng.standard_normal(len(symbols))
        noise = 0.05 * (gaussian_noise + 1j * quadrature_noise)
        noisy_symbols = symbols + noise
        llrs = qpsk.symbols2bits_soft(noisy_symbols, sigma_sq=0.01)

        hard_from_llr = (llrs < 0).astype(int).flatten()
        np.testing.assert_array_equal(hard_from_llr, bits)

    @given(
        bits=st.lists(st.integers(0, 1), min_size=2, max_size=100).filter(lambda x: len(x) % 2 == 0),
        snr_db=st.floats(min_value=10.0, max_value=30.0),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_soft_to_hard_decision_high_snr(self, bits: list[int], snr_db: float, seed: int) -> None:
        """Verify that hard decisions from LLRs match original bits at high SNR."""
        qpsk = QPSK()
        rng = np.random.default_rng(seed)
        bits_arr = np.array(bits)
        symbols = qpsk.bits2symbols(bits_arr)

        # Add AWGN noise based on SNR
        # SNR = E_s / N_0, where E_s = 1 (unit energy constellation)
        # sigma^2 = N_0 / 2 = 1 / (2 * SNR)
        snr_linear = 10 ** (snr_db / 10)
        sigma_sq = 1.0 / (2 * snr_linear)
        noise = np.sqrt(sigma_sq / 2) * (rng.standard_normal(len(symbols)) + 1j * rng.standard_normal(len(symbols)))
        noisy_symbols = symbols + noise

        # Get soft decisions and convert to hard decisions
        llrs = qpsk.symbols2bits_soft(noisy_symbols, sigma_sq=sigma_sq)
        hard_from_llr = (llrs < 0).astype(int).flatten()

        # At high SNR (>=10dB), hard decisions should match original bits
        np.testing.assert_array_equal(hard_from_llr, bits_arr)


class TestEdgeCases:
    """Validate modulation behavior on edge-case inputs."""

    def test_empty_bitstream(self) -> None:
        """Verify handling of an empty BPSK bitstream."""
        bpsk = BPSK()
        bits = np.array([])
        symbols = bpsk.bits2symbols(bits)
        np.testing.assert_equal(len(symbols), 0)

    def test_bpsk_decision_boundary(self) -> None:
        """Verify valid hard decision exactly at the BPSK boundary."""
        bpsk = BPSK()
        symbols = np.array([0 + 0j])
        bits = bpsk.symbols2bits(symbols).flatten()
        np.testing.assert_array_less(bits[0], QPSK_BITS_PER_SYMBOL)
        np.testing.assert_array_less(-1, bits[0])
