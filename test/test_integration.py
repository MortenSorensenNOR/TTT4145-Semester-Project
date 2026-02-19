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
from modules.channel_coding import CodeRates, LDPCConfig, ldpc_decode, ldpc_encode, ldpc_get_supported_payload_lengths
from modules.frame_constructor import FrameConstructor, FrameHeader, FrameHeaderConstructor, ModulationSchemes
from modules.modulation import BPSK, QPSK
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import Synchronizer, SynchronizerConfig, build_preamble
from modules.util import bits_to_text, ebn0_to_snr, text_to_bits
from pluto.config import get_modulator

MESSAGE_LENGTH = 324
MAX_BIT_ERRORS_AWGN = 5
MAX_BIT_ERRORS_PULSE = 10
HIGH_EBN0_THRESHOLD = 5.0
MAX_AVG_BER = 0.01
N_SYMBOLS_TEST = 100


class TestFullChain:
    """Integration tests for the complete TX/RX chain."""

    @pytest.fixture
    def ldpc_config(self) -> LDPCConfig:
        """Create default LDPC configuration."""
        return LDPCConfig(k=MESSAGE_LENGTH, code_rate=CodeRates.HALF_RATE)

    @pytest.fixture
    def qpsk(self) -> QPSK:
        """Create QPSK modulator."""
        return QPSK()

    @pytest.fixture
    def random_message(self) -> NDArray[np.int_]:
        """Generate random 324-bit message."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 2, size=MESSAGE_LENGTH)

    def test_chain_no_channel(
        self,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test TX/RX chain without channel impairments (sanity check)."""
        codeword = ldpc_encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        llrs = qpsk.symbols2bits_soft(symbols, sigma_sq=1e-6)
        llr_flat = llrs.flatten()

        decoded = ldpc_decode(llr_flat, ldpc_config, max_iterations=20)

        np.testing.assert_array_equal(decoded, random_message)

    def test_chain_awgn_only(
        self,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test TX/RX chain with AWGN channel (no pulse shaping)."""
        codeword = ldpc_encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        ebn0_db = 5.0
        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=123))
        rx_symbols = channel.apply(symbols)

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        decoded = ldpc_decode(llr_flat, ldpc_config, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        if bit_errors >= MAX_BIT_ERRORS_AWGN:
            pytest.fail(f"Too many bit errors: {bit_errors}")

    def test_chain_with_pulse_shaping(
        self,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        random_message: NDArray[np.int_],
    ) -> None:
        """Test full chain with pulse shaping and matched filtering."""
        sps = 4
        pulse = rrc_filter(sps=sps, alpha=0.35, num_taps=101)

        codeword = ldpc_encode(random_message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        tx_signal = upsample_and_filter(symbols, sps, pulse)

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

        matched = signal.convolve(rx_signal, pulse, mode="same")

        rx_symbols = matched[::sps]

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        llr_flat = llrs.flatten()

        decoded = ldpc_decode(llr_flat, ldpc_config, max_iterations=50)
        bit_errors = np.sum(decoded != random_message)

        if bit_errors >= MAX_BIT_ERRORS_PULSE:
            pytest.fail(f"Too many bit errors: {bit_errors}")

    @pytest.mark.parametrize("ebn0_db", [3.0, 4.0, 5.0, 6.0, 8.0])
    def test_ber_vs_ebn0(
        self,
        ldpc_config: LDPCConfig,
        qpsk: QPSK,
        ebn0_db: float,
    ) -> None:
        """Test BER performance across different Eb/N0 values."""
        rng = np.random.default_rng(789)
        message = rng.integers(0, 2, size=MESSAGE_LENGTH)

        codeword = ldpc_encode(message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)

        code_rate = CodeRates.HALF_RATE.value_float
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)
        channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=int(ebn0_db * 100)))
        rx_symbols = channel.apply(symbols)

        sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
        llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
        decoded = ldpc_decode(llrs.flatten(), ldpc_config, max_iterations=50)

        bit_errors = np.sum(decoded != message)

        if ebn0_db >= HIGH_EBN0_THRESHOLD and bit_errors >= MAX_BIT_ERRORS_PULSE:
            pytest.fail(
                f"Too many errors ({bit_errors}) at Eb/N0={ebn0_db} dB",
            )

    def test_multiple_frames(
        self,
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

            codeword = ldpc_encode(message, ldpc_config)
            symbols = qpsk.bits2symbols(codeword)

            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=frame_idx))
            rx_symbols = channel.apply(symbols)

            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc_decode(llrs.flatten(), ldpc_config, max_iterations=50)

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
        pulse = rrc_filter(sps=sps, alpha=0.35, num_taps=101)

        n_symbols = N_SYMBOLS_TEST
        rng = np.random.default_rng(42)
        symbols = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)

        shaped = upsample_and_filter(symbols, sps, pulse)

        matched = signal.convolve(shaped, pulse, mode="same")
        recovered = matched[::sps]

        np.testing.assert_equal(len(recovered), n_symbols)


class TestFullPipeline:
    """End-to-end tests covering sync, framing, and decoding.

    Mirrors the actual pluto/transmit.py -> channel -> pluto/receive.py path.
    """

    SPS = 4
    RRC_ALPHA = 0.35
    RRC_NUM_TAPS = 101
    SAMPLE_RATE = 1e6
    DELAY_PADDING = 2000

    @pytest.fixture
    def rrc(self) -> np.ndarray:
        """RRC pulse shared by TX and RX."""
        return rrc_filter(self.SPS, self.RRC_ALPHA, self.RRC_NUM_TAPS)

    @pytest.fixture
    def sync_config(self) -> SynchronizerConfig:
        """Synchronizer configuration."""
        return SynchronizerConfig()

    def _transmit(
        self,
        text: str,
        mod_scheme: ModulationSchemes,
        coding_rate: CodeRates,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> np.ndarray:
        """Simulate the TX pipeline (mirrors pluto/transmit.py)."""
        payload_bits = text_to_bits(text)
        header = FrameHeader(
            length=len(payload_bits),
            src=0,
            dst=0,
            mod_scheme=mod_scheme,
            coding_rate=coding_rate,
        )
        fc = FrameConstructor()
        header_encoded, payload_encoded = fc.encode(header, payload_bits)

        header_symbols = BPSK().bits2symbols(header_encoded)
        payload_symbols = get_modulator(mod_scheme).bits2symbols(payload_encoded)

        preamble = build_preamble(sync_config)
        frame = np.concatenate([preamble, header_symbols, payload_symbols])

        tx_signal = upsample_and_filter(frame, self.SPS, rrc)
        return np.concatenate([np.zeros(self.DELAY_PADDING, dtype=complex), tx_signal, np.zeros(self.DELAY_PADDING, dtype=complex)])

    def _receive(
        self,
        rx_raw: np.ndarray,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> str | None:
        """Simulate the RX pipeline (mirrors pluto/receive.py)."""
        sync = Synchronizer(sync_config, sps=self.SPS, rrc_taps=rrc)
        bpsk = BPSK()
        fc = FrameConstructor()
        header_n_symbols = FrameHeaderConstructor().header_length * 2
        zc_long_ref = sync.zc_long

        # Matched filter
        rx_filtered = np.convolve(rx_raw, rrc, mode="same")

        # Sync
        result = sync.detect_preamble(rx_filtered, self.SAMPLE_RATE)
        if not result.success:
            return None

        # CFO correction
        n_vec = np.arange(len(rx_filtered))
        rx_corr = rx_filtered * np.exp(-1j * 2 * np.pi * result.cfo_hat_hz / self.SAMPLE_RATE * n_vec)

        # Phase correction via long ZC
        zc_start = result.timing_hat
        zc_rx = rx_corr[zc_start :: self.SPS][: len(zc_long_ref)]
        if len(zc_rx) == len(zc_long_ref):
            phase_hat = np.angle(np.sum(zc_rx * np.conj(zc_long_ref)))
            rx_corr = rx_corr * np.exp(-1j * phase_hat)

        # Downsample
        data_start = result.timing_hat + sync_config.n_long * self.SPS
        symbols = rx_corr[data_start :: self.SPS]

        if len(symbols) < header_n_symbols:
            return None

        # Amplitude normalization (from known-power header)
        header_power = np.mean(np.abs(symbols[:header_n_symbols]) ** 2)
        if header_power > 0:
            symbols = symbols / np.sqrt(header_power)
        header_hard = bpsk.symbols2bits(symbols[:header_n_symbols])
        try:
            header = fc.decode_header(header_hard)
        except ValueError:
            return None

        # Payload parameters
        modulator = get_modulator(header.mod_scheme)
        supported_k = ldpc_get_supported_payload_lengths(header.coding_rate)
        k = int(min(kk for kk in supported_k if kk >= header.length + fc.PAYLOAD_CRC_BITS))
        n_coded = LDPCConfig(k=k, code_rate=header.coding_rate).n
        n_payload_symbols = n_coded // modulator.bits_per_symbol

        if len(symbols) < header_n_symbols + n_payload_symbols:
            return None

        # Soft demod + decode
        payload_symbols = symbols[header_n_symbols : header_n_symbols + n_payload_symbols]
        sigma_sq = modulator.estimate_noise_variance(payload_symbols)
        payload_llrs = modulator.symbols2bits_soft(payload_symbols, sigma_sq=sigma_sq).flatten()

        try:
            payload_bits = fc.decode_payload(header, payload_llrs, soft=True)
        except ValueError:
            return None

        return bits_to_text(payload_bits)

    def test_pipeline_no_impairments(
        self,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> None:
        """Full pipeline with no channel impairments (sanity check)."""
        text = "Hello!"
        tx = self._transmit(text, ModulationSchemes.QPSK, CodeRates.HALF_RATE, rrc, sync_config)
        decoded = self._receive(tx, rrc, sync_config)
        assert decoded == text

    def test_pipeline_awgn(
        self,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> None:
        """Full pipeline through AWGN channel at comfortable SNR."""
        text = "Test AWGN"
        tx = self._transmit(text, ModulationSchemes.QPSK, CodeRates.HALF_RATE, rrc, sync_config)

        channel = ChannelModel(ChannelConfig(snr_db=15.0, sample_rate=self.SAMPLE_RATE, seed=42))
        rx = channel.apply(tx)

        decoded = self._receive(rx, rrc, sync_config)
        assert decoded == text

    def test_pipeline_awgn_with_cfo(
        self,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> None:
        """Full pipeline through AWGN + CFO channel."""
        text = "CFO test"
        tx = self._transmit(text, ModulationSchemes.QPSK, CodeRates.HALF_RATE, rrc, sync_config)

        channel = ChannelModel(
            ChannelConfig(snr_db=15.0, sample_rate=self.SAMPLE_RATE, cfo_hz=2000.0, seed=99),
        )
        rx = channel.apply(tx)

        decoded = self._receive(rx, rrc, sync_config)
        assert decoded == text

    def test_pipeline_text_roundtrip(
        self,
        rrc: np.ndarray,
        sync_config: SynchronizerConfig,
    ) -> None:
        """Text message survives the full TX -> channel -> RX pipeline."""
        text = "Hello, PlutoSDR!"
        tx = self._transmit(text, ModulationSchemes.QPSK, CodeRates.HALF_RATE, rrc, sync_config)

        channel = ChannelModel(
            ChannelConfig(
                snr_db=12.0,
                sample_rate=self.SAMPLE_RATE,
                cfo_hz=1000.0,
                delay_samples=50,
                seed=77,
            ),
        )
        rx = channel.apply(tx)

        decoded = self._receive(rx, rrc, sync_config)
        assert decoded == text
