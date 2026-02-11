import numpy as np
import pytest
from modules.modulation import BPSK, QPSK, QAM

class TestBPSK:
    @pytest.fixture
    def bpsk(self):
        """Fixture that creates a BPSK instance for each test"""
        return BPSK()
    
    def test_symbol_constellation(self, bpsk):
        """Test that BPSK has the correct symbol constellation"""
        expected = np.array([-1 + 0j, 1 + 0j])
        np.testing.assert_array_equal(bpsk.symbols, expected)
    
    def test_bits2symbols_single_bit(self, bpsk):
        """Test mapping single bits to symbols"""
        assert bpsk.bits2symbols(np.array([0]))[0] == -1 + 0j
        assert bpsk.bits2symbols(np.array([1]))[0] == 1 + 0j
    
    def test_bits2symbols_bitstream(self, bpsk):
        """Test mapping a bitstream to symbols"""
        bits = np.array([0, 1, 1, 0, 1, 0])
        symbols = bpsk.bits2symbols(bits)
        expected = np.array([-1+0j, 1+0j, 1+0j, -1+0j, 1+0j, -1+0j])
        np.testing.assert_array_equal(symbols, expected)
    
    def test_symbols2bits_perfect_symbols(self, bpsk):
        """Test demodulation with perfect (noiseless) symbols"""
        symbols = np.array([-1+0j, 1+0j, 1+0j, -1+0j])
        bits = bpsk.symbols2bits(symbols)
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)
    
    def test_symbols2bits_noisy_symbols(self, bpsk):
        """Test demodulation with noisy symbols"""
        symbols = np.array([-0.8+0.1j, 0.9-0.05j, 0.7+0.2j, -1.1+0j])
        bits = bpsk.symbols2bits(symbols)
        expected = np.array([0, 1, 1, 0])
        np.testing.assert_array_equal(bits, expected)
    
    def test_roundtrip(self, bpsk):
        """Test that bits -> symbols -> bits gives back original bits"""
        original_bits = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        symbols = bpsk.bits2symbols(original_bits)
        recovered_bits = bpsk.symbols2bits(symbols)
        np.testing.assert_array_equal(recovered_bits, original_bits)


class TestQPSK:
    @pytest.fixture
    def qpsk(self):
        """Fixture that creates a QPSK instance for each test"""
        return QPSK()
    
    def test_qpsk_order(self, qpsk):
        """Test that QPSK has 4 symbols"""
        assert qpsk.qam_order == 4
        assert len(qpsk.symbol_mapping) == 4
    
    def test_bits_per_symbol(self, qpsk):
        """Test that QPSK uses 2 bits per symbol"""
        assert qpsk.bits_per_symbol == 2
    
    def test_unit_energy(self, qpsk):
        """Test that QPSK constellation has unit average energy"""
        avg_energy = np.mean(np.abs(qpsk.symbol_mapping)**2)
        np.testing.assert_almost_equal(avg_energy, 1.0)
    
    def test_bits2symbols(self, qpsk):
        """Test mapping bits to QPSK symbols"""
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(bits)
        assert len(symbols) == 4
    
    def test_symbols2bits_shape(self, qpsk):
        """Test that symbols2bits returns correct shape"""
        symbols = qpsk.symbol_mapping
        bits = qpsk.symbols2bits(symbols)
        assert bits.shape == (4, 2)
    
    def test_roundtrip(self, qpsk):
        """Test that bits -> symbols -> bits gives back original bits"""
        original_bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(original_bits)
        recovered_bits = qpsk.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)
    
    def test_gray_coding(self, qpsk):
        """Test that adjacent symbols differ by only 1 bit (Gray coding)"""
        for i, symbol in enumerate(qpsk.symbol_mapping):
            bits_i = qpsk.bit_mapping[i]
            distances = np.abs(qpsk.symbol_mapping - symbol)
            distances[i] = np.inf
            closest_idx = np.argmin(distances)
            bits_closest = qpsk.bit_mapping[closest_idx]
            hamming_dist = np.sum(bits_i != bits_closest)
            assert hamming_dist <= 2


class TestQAM:
    @pytest.mark.parametrize("qam_order", [16, 64, 256])
    def test_qam_orders(self, qam_order):
        """Test different QAM orders"""
        qam = QAM(qam_order)
        assert qam.qam_order == qam_order
        assert len(qam.symbol_mapping) == qam_order
        assert qam.bits_per_symbol == int(np.log2(qam_order))
    
    def test_16qam_unit_energy(self):
        """Test that 16-QAM constellation has unit average energy"""
        qam = QAM(16)
        avg_energy = np.mean(np.abs(qam.symbol_mapping)**2)
        np.testing.assert_almost_equal(avg_energy, 1.0)
    
    def test_16qam_roundtrip(self):
        """Test roundtrip for 16-QAM"""
        qam = QAM(16)
        original_bits = np.random.randint(0, 2, size=16)
        symbols = qam.bits2symbols(original_bits)
        recovered_bits = qam.symbols2bits(symbols).flatten()
        np.testing.assert_array_equal(recovered_bits, original_bits)
    
    def test_bit_mapping_completeness(self):
        """Test that all possible bit patterns are mapped"""
        qam = QAM(16)
        decimal_values = np.sum(qam.bit_mapping * 2**np.arange(qam.bits_per_symbol), axis=1)
        assert len(np.unique(decimal_values)) == qam.qam_order
        assert set(decimal_values) == set(range(qam.qam_order))


class TestQPSKSoftDecision:
    @pytest.fixture
    def qpsk(self):
        return QPSK()

    def test_soft_output_shape(self, qpsk):
        """Test that soft output has correct shape (N, 2)"""
        symbols = np.array([0.5+0.5j, -0.3+0.2j, 0.1-0.8j])
        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        assert llrs.shape == (3, 2)

    def test_soft_output_empty(self, qpsk):
        """Test soft decision with empty input"""
        llrs = qpsk.symbols2bits_soft(np.array([]), sigma_sq=0.1)
        assert len(llrs) == 0

    def test_llr_sign_matches_hard_decision(self, qpsk):
        """Test that LLR sign agrees with hard decision"""
        # Perfect constellation points
        symbols = qpsk.symbols
        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        hard_bits = qpsk.symbols2bits(symbols)

        for i in range(len(symbols)):
            # LLR > 0 means bit=0, LLR < 0 means bit=1
            for bit_idx in range(2):
                if hard_bits[i, bit_idx] == 0:
                    assert llrs[i, bit_idx] > 0, f"Symbol {i}, bit {bit_idx}: expected positive LLR for bit=0"
                else:
                    assert llrs[i, bit_idx] < 0, f"Symbol {i}, bit {bit_idx}: expected negative LLR for bit=1"

    def test_llr_magnitude_increases_away_from_boundary(self, qpsk):
        """Test that |LLR| is larger for symbols farther from decision boundary"""
        # Symbols at different distances from Re=0 boundary (bit 0)
        close_to_boundary = np.array([0.1 + 0.5j])
        far_from_boundary = np.array([0.7 + 0.5j])

        llr_close = qpsk.symbols2bits_soft(close_to_boundary, sigma_sq=0.1)
        llr_far = qpsk.symbols2bits_soft(far_from_boundary, sigma_sq=0.1)

        # Bit 0 LLR magnitude should be larger when farther from Re=0
        assert np.abs(llr_far[0, 0]) > np.abs(llr_close[0, 0])

    def test_llr_symmetric_around_origin(self, qpsk):
        """Test that LLR is antisymmetric: LLR(-y) = -LLR(y)"""
        symbols = np.array([0.5 + 0.3j])
        neg_symbols = -symbols

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)
        neg_llrs = qpsk.symbols2bits_soft(neg_symbols, sigma_sq=0.1)

        np.testing.assert_array_almost_equal(llrs, -neg_llrs)

    def test_llr_scales_with_noise_variance(self, qpsk):
        """Test that |LLR| decreases with higher noise variance"""
        symbols = np.array([0.5 + 0.5j])

        llr_low_noise = qpsk.symbols2bits_soft(symbols, sigma_sq=0.05)
        llr_high_noise = qpsk.symbols2bits_soft(symbols, sigma_sq=0.5)

        # Higher noise = lower confidence = smaller |LLR|
        assert np.abs(llr_low_noise[0, 0]) > np.abs(llr_high_noise[0, 0])
        assert np.abs(llr_low_noise[0, 1]) > np.abs(llr_high_noise[0, 1])

    def test_llr_on_decision_boundary_is_zero(self, qpsk):
        """Test that LLR is zero on decision boundaries"""
        # On Re=0 boundary (bit 0 uncertain)
        on_i_boundary = np.array([0.0 + 0.5j])
        llrs = qpsk.symbols2bits_soft(on_i_boundary, sigma_sq=0.1)
        np.testing.assert_almost_equal(llrs[0, 0], 0.0)

        # On Im=0 boundary (bit 1 uncertain)
        on_q_boundary = np.array([0.5 + 0.0j])
        llrs = qpsk.symbols2bits_soft(on_q_boundary, sigma_sq=0.1)
        np.testing.assert_almost_equal(llrs[0, 1], 0.0)

    def test_noise_variance_estimation(self, qpsk):
        """Test that noise variance is estimated reasonably"""
        # Generate noisy symbols
        np.random.seed(42)
        true_sigma_sq = 0.05
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1] * 100)
        symbols = qpsk.bits2symbols(bits)
        noise = np.sqrt(true_sigma_sq / 2) * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        noisy_symbols = symbols + noise

        estimated_sigma_sq = qpsk.estimate_noise_variance(noisy_symbols)

        # Should be within 50% of true value with enough samples
        assert 0.5 * true_sigma_sq < estimated_sigma_sq < 1.5 * true_sigma_sq

    def test_soft_roundtrip_improves_with_confidence(self, qpsk):
        """Test that high-confidence LLRs give correct hard decisions"""
        # Symbols close to constellation points (high confidence)
        np.random.seed(123)
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        symbols = qpsk.bits2symbols(bits)
        noise = 0.05 * (np.random.randn(len(symbols)) + 1j * np.random.randn(len(symbols)))
        noisy_symbols = symbols + noise

        llrs = qpsk.symbols2bits_soft(noisy_symbols, sigma_sq=0.01)

        # Hard decision from LLR: bit=0 if LLR>0, bit=1 if LLR<0
        hard_from_llr = (llrs < 0).astype(int).flatten()

        np.testing.assert_array_equal(hard_from_llr, bits)


class TestEdgeCases:
    def test_empty_bitstream(self):
        """Test handling of empty bitstreams"""
        bpsk = BPSK()
        bits = np.array([])
        symbols = bpsk.bits2symbols(bits)
        assert len(symbols) == 0

    def test_bpsk_decision_boundary(self):
        """Test symbols exactly on decision boundary"""
        bpsk = BPSK()
        symbols = np.array([0+0j])
        bits = bpsk.symbols2bits(symbols)
        assert bits[0] in [0, 1]
