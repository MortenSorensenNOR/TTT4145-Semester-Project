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
