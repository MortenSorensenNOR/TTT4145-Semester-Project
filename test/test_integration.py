"""Integration tests for the full TX/RX chain.

Tests the complete pipeline:
  Message bits -> LDPC encode -> QPSK modulate -> Pulse shape ->
  Channel (AWGN) -> Matched filter -> QPSK soft demod -> LDPC decode -> Recovered bits
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import signal

from modules.channel import ChannelConfig, ChannelModel
from modules.channel_coding import LDPC, CodeRates, LDPCConfig
from modules.modulation import QPSK
from modules.pulse_shaping import PulseShaper
from modules.util import ebn0_to_snr

MESSAGE_LENGTH = 324
MAX_BIT_ERRORS_AWGN = 5
MAX_BIT_ERRORS_PULSE = 10
HIGH_EBN0_THRESHOLD = 5.0
MAX_AVG_BER = 0.01
N_SYMBOLS_TEST = 100


class TestFullChain:
    """Integration tests for the complete TX/RX chain."""

    @pytest.fixture
    def ldpc(self) -> LDPC:
        """Create LDPC codec."""
        return LDPC()

    @pytest.fixture
    def ldpc_config(self) -> LDPCConfig:
        """Create default LDPC configuration."""
        return LDPCConfig(k=MESSAGE_LENGTH, code_rate=CodeRates.HALF_RATE)

    @pytest.fixture
    def qpsk(self) -> QPSK:
        """Create QPSK modulator."""
        return QPSK()

    @pytest.fixture
    def pulse_shaper(self) -> PulseShaper:
        """Create pulse shaper (4 samples per symbol, alpha=0.35)."""
        return PulseShaper(sps=4, alpha=0.35, taps=101)

    @pytest.fixture
    def random_message(self) -> NDArray[np.int_]:
        """Generate random 324-bit message."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 2, size=MESSAGE_LENGTH)

    def test_chain_no_channel(
        self,
        ldpc: LDPC,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test TX/RX chain without channel impairments (sanity check)."""
        codeword = ldpc.encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=1e-6)
        llr_flat = llrs.flatten()

        decoded = ldpc.decode(llr_flat, ldpc_config, max_iterations=20)

        np.testing.assert_array_equal(decoded, random_message)

    def test_chain_awgn_only(
        self,
        ldpc: LDPC,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test TX/RX chain with AWGN channel (no pulse shaping)."""
        codeword = ldpc.encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        ebn0_db = 5.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=123))
        rx_symbols = channel.apply(symbols)

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        decoded = ldpc.decode(llr_flat, ldpc_config, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        if bit_errors >= MAX_BIT_ERRORS_AWGN:
            pytest.fail(f"Too many bit errors: {bit_errors}")

    def test_chain_with_pulse_shaping(
        self,
        ldpc: LDPC,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        pulse_shaper: PulseShaper,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test full chain with pulse shaping and matched filtering."""
        sps = pulse_shaper.sps

        codeword = ldpc.encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        upsampled = np.zeros(len(symbols) * sps, dtype=complex)
        upsampled[::sps] = symbols
        tx_signal = pulse_shaper.shape(upsampled)

        ebn0_db = 6.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(
            ChannelConfig(
                snr_db=snr_db,
                sample_rate=1e6,
                seed=456,
            ),
        )
        rx_signal = channel.apply(tx_signal)

        matched = signal.convolve(rx_signal, pulse_shaper.pulse_shape, mode="same")

        rx_symbols = matched[::sps]

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        decoded = ldpc.decode(llr_flat, ldpc_config, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        if bit_errors >= MAX_BIT_ERRORS_PULSE:
            pytest.fail(f"Too many bit errors: {bit_errors}")

    @pytest.mark.parametrize("ebn0_db", [3.0, 4.0, 5.0, 6.0, 8.0])
    def test_ber_vs_ebn0(
        self,
        ldpc: LDPC,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        ebn0_db: float,
    ) -> None:
        """Test BER performance across different Eb/N0 values.

        Uses Eb/N0 (energy per info bit / noise PSD) for accurate performance
        comparison. This properly accounts for the coding rate when measuring
        channel coding efficiency.
        """
        rng = np.random.default_rng(789)
        message = rng.integers(0, 2, size=MESSAGE_LENGTH)

        codeword = ldpc.encode(message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=int(ebn0_db * 100)))
        rx_symbols = channel.apply(symbols)

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        decoded = ldpc.decode(llrs.flatten(), ldpc_config, max_iterations=50)

        bit_errors = np.sum(decoded != message)

        if ebn0_db >= HIGH_EBN0_THRESHOLD and bit_errors >= MAX_BIT_ERRORS_PULSE:
            pytest.fail(
                f"Too many errors ({bit_errors}) at Eb/N0={ebn0_db} dB",
            )

    def test_multiple_frames(
        self,
        ldpc: LDPC,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
    ) -> None:
        """Test decoding multiple frames in sequence."""
        rng = np.random.default_rng(999)
        n_frames = 5
        ebn0_db = 6.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)

        total_errors = 0
        total_bits = 0

        for frame_idx in range(n_frames):
            message = rng.integers(0, 2, size=MESSAGE_LENGTH)

            codeword = ldpc.encode(message, ldpc_config)
            symbols = qpsk.bits2symbols(codeword)

            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=frame_idx))
            rx_symbols = channel.apply(symbols)

            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc.decode(llrs.flatten(), ldpc_config, max_iterations=50)

            total_errors += np.sum(decoded != message)
            total_bits += len(message)

        avg_ber = total_errors / total_bits
        if avg_ber >= MAX_AVG_BER:
            pytest.fail(f"Average BER too high: {avg_ber:.4f}")


class TestComponentInterfaces:
    """Test that components interface correctly."""

    def test_ldpc_output_matches_qpsk_input(self) -> None:
        """LDPC codeword length should be even for QPSK (2 bits/symbol)."""
        config = LDPCConfig(k=MESSAGE_LENGTH, code_rate=CodeRates.HALF_RATE)
        if config.n % 2 != 0:
            pytest.fail("Codeword length must be even for QPSK")

    def test_qpsk_llr_shape_matches_ldpc_input(self) -> None:
        """QPSK soft output shape should match LDPC decoder input."""
        qpsk = QPSK()
        config = LDPCConfig(k=MESSAGE_LENGTH, code_rate=CodeRates.HALF_RATE)

        n_symbols = config.n // 2
        rng = np.random.default_rng(42)
        symbols = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)

        np.testing.assert_equal(llrs.shape, (n_symbols, 2))
        np.testing.assert_equal(llrs.flatten().shape[0], config.n)

    def test_pulse_shaper_preserves_symbol_count(self) -> None:
        """Pulse shaper should preserve symbol timing."""
        sps = 4
        ps = PulseShaper(sps=sps, alpha=0.35, taps=101)

        n_symbols = N_SYMBOLS_TEST
        rng = np.random.default_rng(42)
        symbols = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)

        upsampled = np.zeros(n_symbols * sps, dtype=complex)
        upsampled[::sps] = symbols
        shaped = ps.shape(upsampled)

        matched = signal.convolve(shaped, ps.pulse_shape, mode="same")
        recovered = matched[::sps]

        np.testing.assert_equal(len(recovered), n_symbols)
