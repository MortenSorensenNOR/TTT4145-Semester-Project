"""Integration tests for the full TX/RX chain.

Tests the complete pipeline:
  Message bits → LDPC encode → QPSK modulate → Pulse shape →
  Channel (AWGN) → Matched filter → QPSK soft demod → LDPC decode → Recovered bits
"""

import numpy as np
import pytest
from scipy import signal

from modules.channel_coding import LDPC, LDPCConfig, CodeRates
from modules.modulation import QPSK
from modules.pulseshaping import PulseShaper, rrc_filter
from modules.channel import ChannelModel, ChannelConfig
from modules.util import ebn0_to_snr


class TestFullChain:
    """Integration tests for the complete TX/RX chain."""

    @pytest.fixture
    def ldpc(self):
        """Create LDPC codec."""
        config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
        return LDPC(config)

    @pytest.fixture
    def qpsk(self):
        """Create QPSK modulator."""
        return QPSK()

    @pytest.fixture
    def pulse_shaper(self):
        """Create pulse shaper (4 samples per symbol, alpha=0.35)."""
        return PulseShaper(sps=4, alpha=0.35, taps=101)

    @pytest.fixture
    def random_message(self):
        """Generate random 324-bit message."""
        np.random.seed(42)
        return np.random.randint(0, 2, 324)

    def test_chain_no_channel(self, ldpc, qpsk, random_message):
        """Test TX/RX chain without channel impairments (sanity check)."""
        # TX
        codeword = ldpc.encode(random_message)
        symbols = qpsk.bits2symbols(codeword)

        # RX (no channel, just demodulate)
        # Use very small noise variance for soft decision
        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=1e-6)
        llr_flat = llrs.flatten()

        decoded = ldpc.decode(llr_flat, max_iterations=20)

        assert np.array_equal(decoded, random_message)

    def test_chain_awgn_only(self, ldpc, qpsk, random_message):
        """Test TX/RX chain with AWGN channel (no pulse shaping)."""
        # TX
        codeword = ldpc.encode(random_message)
        symbols = qpsk.bits2symbols(codeword)

        # Channel (AWGN only)
        # Use Eb/N0 for accurate channel coding performance measurement
        ebn0_db = 5.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=123))
        rx_symbols = channel.apply(symbols)

        # RX
        # Estimate noise variance from received symbols
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        decoded = ldpc.decode(llr_flat, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        assert bit_errors < 5, f"Too many bit errors: {bit_errors}"

    def test_chain_with_pulse_shaping(self, ldpc, qpsk, pulse_shaper, random_message):
        """Test full chain with pulse shaping and matched filtering."""
        sps = pulse_shaper.sps

        # TX
        codeword = ldpc.encode(random_message)
        symbols = qpsk.bits2symbols(codeword)

        # Upsample and pulse shape
        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols
        tx_signal = pulse_shaper.shape(upsampled)

        # Channel (AWGN)
        # Use Eb/N0 for accurate channel coding performance measurement
        ebn0_db = 6.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(
            snr_db=snr_db,
            sample_rate=1e6,
            seed=456,
        ))
        rx_signal = channel.apply(tx_signal)

        # RX: Matched filter (same RRC filter)
        matched = signal.convolve(rx_signal, pulse_shaper.pulse_shape, mode='same')

        # Downsample at optimal timing (center of symbol)
        rx_symbols = matched[::sps]

        # Soft demodulation
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        # LDPC decode
        decoded = ldpc.decode(llr_flat, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        assert bit_errors < 10, f"Too many bit errors: {bit_errors}"

    @pytest.mark.parametrize("ebn0_db", [3.0, 4.0, 5.0, 6.0, 8.0])
    def test_ber_vs_ebn0(self, ldpc, qpsk, ebn0_db):
        """Test BER performance across different Eb/N0 values.

        Uses Eb/N0 (energy per info bit / noise PSD) for accurate performance
        comparison. This properly accounts for the coding rate when measuring
        channel coding efficiency.
        """
        np.random.seed(789)
        message = np.random.randint(0, 2, 324)

        # TX
        codeword = ldpc.encode(message)
        symbols = qpsk.bits2symbols(codeword)

        # Channel - convert Eb/N0 to SNR per symbol
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=int(ebn0_db * 100)))
        rx_symbols = channel.apply(symbols)

        # RX
        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        decoded = ldpc.decode(llrs.flatten(), max_iterations=50)

        bit_errors = np.sum(decoded != message)

        # Expected: BER should decrease with increasing Eb/N0
        # At Eb/N0 >= 5 dB, rate 1/2 LDPC should have very few errors
        if ebn0_db >= 5.0:
            assert bit_errors < 10, f"Too many errors ({bit_errors}) at Eb/N0={ebn0_db} dB"
        # Just verify it runs at lower Eb/N0 (BER may be higher)

    def test_multiple_frames(self, ldpc, qpsk):
        """Test decoding multiple frames in sequence."""
        np.random.seed(999)
        n_frames = 5
        ebn0_db = 6.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)

        total_errors = 0
        total_bits = 0

        for frame_idx in range(n_frames):
            message = np.random.randint(0, 2, 324)

            # TX
            codeword = ldpc.encode(message)
            symbols = qpsk.bits2symbols(codeword)

            # Channel (different seed per frame)
            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=frame_idx))
            rx_symbols = channel.apply(symbols)

            # RX
            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc.decode(llrs.flatten(), max_iterations=50)

            total_errors += np.sum(decoded != message)
            total_bits += len(message)

        avg_ber = total_errors / total_bits
        assert avg_ber < 0.01, f"Average BER too high: {avg_ber:.4f}"


class TestComponentInterfaces:
    """Test that components interface correctly."""

    def test_ldpc_output_matches_qpsk_input(self):
        """LDPC codeword length should be even for QPSK (2 bits/symbol)."""
        config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
        ldpc = LDPC(config)
        assert ldpc.n % 2 == 0, "Codeword length must be even for QPSK"

    def test_qpsk_llr_shape_matches_ldpc_input(self):
        """QPSK soft output shape should match LDPC decoder input."""
        qpsk = QPSK()
        config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
        ldpc = LDPC(config)

        # 648 bits = 324 QPSK symbols
        n_symbols = ldpc.n // 2
        symbols = np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)

        assert llrs.shape == (n_symbols, 2)
        assert llrs.flatten().shape[0] == ldpc.n

    def test_pulse_shaper_preserves_symbol_count(self):
        """Pulse shaper should preserve symbol timing."""
        sps = 4
        ps = PulseShaper(sps=sps, alpha=0.35, taps=101)

        n_symbols = 100
        symbols = np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)

        upsampled = np.zeros(n_symbols * sps, dtype=complex)
        upsampled[::sps] = symbols
        shaped = ps.shape(upsampled)

        # After matched filter and downsampling, should get same number of symbols
        matched = signal.convolve(shaped, ps.pulse_shape, mode='same')
        recovered = matched[::sps]

        assert len(recovered) == n_symbols
