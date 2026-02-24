"""Performance timing tests for identifying bottlenecks in the radio communication pipeline.

Run all performance tests:
    uv run pytest test/test_performance.py -v -s

Run specific test class:
    uv run pytest test/test_performance.py::TestComponentTiming -v -s

Run single test:
    uv run pytest test/test_performance.py::TestPipelineTiming::test_full_rx_pipeline -v -s
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy import signal

from modules.channel import ChannelConfig, ChannelModel
from modules.channel_coding import (
    CodeRates,
    Golay,
    LDPCConfig,
    _decode_cache,
    _encoding_cache,
    _h_cache,
    ldpc_clear_cache,
    ldpc_decode,
    ldpc_encode,
)
from modules.costas_loop import apply_costas_loop
from modules.equalization import equalize_payload
from modules.modulation import BPSK, QAM, QPSK
from modules.pilots import PilotConfig, insert_pilots, pilot_aided_phase_track
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import Synchronizer, SynchronizerConfig

# ============================================================================
# Timing Infrastructure
# ============================================================================


@dataclass
class TimingEntry:
    """Single timing measurement."""

    name: str
    duration_ms: float


@dataclass
class TimingReport:
    """Collection of timing measurements with pretty printing."""

    title: str
    entries: list[TimingEntry] = field(default_factory=list)

    def add(self, name: str, duration_ms: float) -> None:
        self.entries.append(TimingEntry(name, duration_ms))

    @property
    def total_ms(self) -> float:
        return sum(e.duration_ms for e in self.entries)

    def print(self, sort_by_duration: bool = True) -> None:
        """Print timing report with percentage bars."""
        if not self.entries:
            return

        total = self.total_ms
        entries = self.entries
        if sort_by_duration:
            entries = sorted(entries, key=lambda e: e.duration_ms, reverse=True)

        name_width = 40

        for entry in entries:
            pct = (entry.duration_ms / total * 100) if total > 0 else 0
            bar_len = int(pct / 100 * 25)
            "#" * bar_len
            entry.name[:name_width].ljust(name_width)



@contextmanager
def timed_section(report: TimingReport, name: str) -> Generator[None]:
    """Context manager for timing a code section and adding to report."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        report.add(name, elapsed_ms)


def time_function(func, *args, iterations: int = 1, **kwargs) -> float:
    """Time a function call and return duration in milliseconds."""
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
    return elapsed_ms, result


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def rrc_taps():
    """RRC filter taps for pulse shaping."""
    return rrc_filter(sps=4, alpha=0.35, num_taps=101)


@pytest.fixture
def synchronizer(rrc_taps):
    """Pre-configured synchronizer."""
    config = SynchronizerConfig()
    return Synchronizer(config, sps=4, rrc_taps=rrc_taps)


@pytest.fixture
def pilot_config():
    """Pilot configuration."""
    return PilotConfig(spacing=16)


# ============================================================================
# Tier 1: Likely Bottlenecks
# ============================================================================


class TestComponentTiming:
    """Time individual components in isolation."""

    def test_ldpc_encode_timing(self, rng) -> None:
        """Time LDPC encoding: cold cache (first call) vs warm cache (subsequent)."""
        cold_report = TimingReport("LDPC ENCODE TIMING - COLD CACHE (matrix construction)")
        warm_report = TimingReport("LDPC ENCODE TIMING - WARM CACHE (encoding only)")

        # Test representative configs for each block size
        test_configs = [
            (324, CodeRates.HALF_RATE),  # n=648
            (648, CodeRates.HALF_RATE),  # n=1296
            (972, CodeRates.HALF_RATE),  # n=1944
        ]

        # Clear cache to measure cold performance
        ldpc_clear_cache()

        for k, code_rate in test_configs:
            config = LDPCConfig(k=k, code_rate=code_rate)
            message = rng.integers(0, 2, size=k).astype(np.int8)

            # Cold: first call builds matrices
            elapsed_ms, _ = time_function(ldpc_encode, message, config, iterations=1)
            rate_str = code_rate.name.replace("_RATE", "").replace("_", "/").lower()
            cold_report.add(f"k={k:4d} n={config.n} r={rate_str}", elapsed_ms)

            # Warm: subsequent calls use cache
            elapsed_ms, _ = time_function(ldpc_encode, message, config, iterations=100)
            warm_report.add(f"k={k:4d} n={config.n} r={rate_str}", elapsed_ms)

        cold_report.print(sort_by_duration=False)
        warm_report.print(sort_by_duration=False)

    def test_ldpc_decode_timing(self, rng) -> None:
        """Time LDPC decoding with JIT warmup.

        Uses pure noise LLRs to prevent early convergence, ensuring
        the decoder runs the full max_iterations.
        """
        report = TimingReport("LDPC DECODE TIMING (forced full iterations)")

        k = 324
        config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)

        # JIT warmup (first call compiles)
        warmup_llrs = rng.normal(0, 1, config.n)
        _ = ldpc_decode(warmup_llrs, config, max_iterations=10)

        # Use pure noise LLRs - decoder will never converge, forcing full iterations
        noise_llrs = rng.normal(0, 1, config.n)

        # Now time different iteration counts
        for max_iter in [10, 25, 50]:
            elapsed_ms, _ = time_function(ldpc_decode, noise_llrs, config, max_iterations=max_iter, iterations=5)
            report.add(f"max_iter={max_iter}", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_synchronization_timing(self, synchronizer, rrc_taps, rng) -> None:
        """Time preamble detection."""
        report = TimingReport("SYNCHRONIZATION TIMING")

        sps = 4
        sample_rate = 1e6

        # Generate a preamble signal
        preamble = synchronizer.preamble
        preamble_upsampled = upsample_and_filter(preamble, sps, rrc_taps)

        # Add some payload symbols after preamble
        payload_symbols = (rng.random(200) > 0.5).astype(np.complex128) * 2 - 1
        payload_upsampled = upsample_and_filter(payload_symbols, sps, rrc_taps)

        # Combine and add noise
        tx_signal = np.concatenate([preamble_upsampled, payload_upsampled])
        rx_signal = tx_signal + rng.normal(0, 0.1, len(tx_signal)) * (1 + 1j)

        # Matched filter
        filtered = signal.convolve(rx_signal, rrc_taps, mode="same")

        # Time synchronization for different signal lengths
        for length_factor in [1, 2, 4]:
            test_signal = np.tile(filtered, length_factor)
            elapsed_ms, _ = time_function(synchronizer.detect_preamble, test_signal, sample_rate, iterations=3)
            report.add(f"signal_len={len(test_signal)}", elapsed_ms)

        report.print()

    def test_pulse_shaping_timing(self, rrc_taps, rng) -> None:
        """Time pulse shaping for different symbol counts."""
        report = TimingReport("PULSE SHAPING TIMING")

        sps = 4

        for n_symbols in [100, 500, 1000, 2000]:
            symbols = rng.standard_normal(n_symbols) + 1j * rng.standard_normal(n_symbols)
            elapsed_ms, _ = time_function(upsample_and_filter, symbols, sps, rrc_taps, iterations=10)
            report.add(f"{n_symbols} symbols", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_matched_filter_timing(self, rrc_taps, rng) -> None:
        """Time matched filtering for different signal lengths."""
        report = TimingReport("MATCHED FILTER TIMING")

        for n_samples in [1000, 5000, 10000, 20000]:
            rx_signal = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            elapsed_ms, _ = time_function(signal.convolve, rx_signal, rrc_taps, mode="same", iterations=10)
            report.add(f"{n_samples} samples", elapsed_ms)

        report.print(sort_by_duration=False)

    # ========================================================================
    # Tier 2: Medium Priority
    # ========================================================================

    def test_modulation_timing(self, rng) -> None:
        """Time modulation/demodulation for different schemes."""
        report = TimingReport("MODULATION TIMING")

        n_bits = 1000

        modulators = [
            ("BPSK", BPSK()),
            ("QPSK", QPSK()),
            ("16-QAM", QAM(16)),
            ("64-QAM", QAM(64)),
        ]

        for name, mod in modulators:
            bits_per_symbol = mod.bits_per_symbol
            n_bits_aligned = (n_bits // bits_per_symbol) * bits_per_symbol
            bits = rng.integers(0, 2, size=n_bits_aligned).astype(np.int8)

            # Time encoding
            elapsed_ms, symbols = time_function(mod.bits2symbols, bits, iterations=100)
            report.add(f"{name} encode", elapsed_ms)

            # Add noise for soft demod
            noisy = symbols + rng.normal(0, 0.1, len(symbols)) * (1 + 1j)

            # Time soft decoding
            elapsed_ms, _ = time_function(mod.symbols2bits_soft, noisy, sigma_sq=0.1, iterations=100)
            report.add(f"{name} soft decode", elapsed_ms)

        report.print()

    def test_channel_model_timing(self, rng) -> None:
        """Time channel model with different impairments."""
        report = TimingReport("CHANNEL MODEL TIMING")

        n_samples = 10000
        tx_signal = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)

        # AWGN only
        config_awgn = ChannelConfig(snr_db=20.0, seed=42)
        channel_awgn = ChannelModel(config_awgn)
        elapsed_ms, _ = time_function(channel_awgn.apply, tx_signal, iterations=10)
        report.add("AWGN only", elapsed_ms)

        # AWGN + CFO
        config_cfo = ChannelConfig(snr_db=20.0, cfo_hz=100.0, seed=42)
        channel_cfo = ChannelModel(config_cfo)
        elapsed_ms, _ = time_function(channel_cfo.apply, tx_signal, iterations=10)
        report.add("AWGN + CFO", elapsed_ms)

        # AWGN + multipath
        config_mp = ChannelConfig(
            snr_db=20.0,
            enable_multipath=True,
            multipath_delays_samples=(0.0, 2.0, 4.0),
            multipath_gains_db=(0.0, -3.0, -6.0),
            seed=42,
        )
        channel_mp = ChannelModel(config_mp)
        elapsed_ms, _ = time_function(channel_mp.apply, tx_signal, iterations=10)
        report.add("AWGN + multipath", elapsed_ms)

        # Full channel (AWGN + CFO + multipath)
        config_full = ChannelConfig(
            snr_db=20.0,
            cfo_hz=100.0,
            enable_multipath=True,
            multipath_delays_samples=(0.0, 2.0, 4.0),
            multipath_gains_db=(0.0, -3.0, -6.0),
            seed=42,
        )
        channel_full = ChannelModel(config_full)
        elapsed_ms, _ = time_function(channel_full.apply, tx_signal, iterations=10)
        report.add("AWGN + CFO + multipath", elapsed_ms)

        report.print()

    # ========================================================================
    # Tier 3: Lower Priority (usually fast)
    # ========================================================================

    def test_pilot_operations_timing(self, pilot_config, rng) -> None:
        """Time pilot insertion and phase tracking."""
        report = TimingReport("PILOT OPERATIONS TIMING")

        n_data = 500
        data_symbols = rng.standard_normal(n_data) + 1j * rng.standard_normal(n_data)

        # Time pilot insertion
        elapsed_ms, symbols_with_pilots = time_function(insert_pilots, data_symbols, pilot_config, iterations=100)
        report.add("insert_pilots", elapsed_ms)

        # Add phase rotation for phase tracking test
        phase = np.linspace(0, np.pi / 4, len(symbols_with_pilots))
        rotated = symbols_with_pilots * np.exp(1j * phase)

        # Time phase tracking
        elapsed_ms, _ = time_function(pilot_aided_phase_track, rotated, n_data, pilot_config, iterations=100)
        report.add("pilot_aided_phase_track", elapsed_ms)

        report.print()

    def test_equalization_timing(self, pilot_config, rng) -> None:
        """Time channel equalization."""
        report = TimingReport("EQUALIZATION TIMING")

        n_data = 500
        data_symbols = rng.standard_normal(n_data) + 1j * rng.standard_normal(n_data)
        symbols_with_pilots = insert_pilots(data_symbols, pilot_config)

        # Simulate channel distortion
        h = 0.8 + 0.2j  # Simple flat fading
        distorted = symbols_with_pilots * h + rng.normal(0, 0.1, len(symbols_with_pilots)) * (1 + 1j)

        elapsed_ms, _ = time_function(equalize_payload, distorted, n_data, pilot_config, sigma_sq=0.01, iterations=100)
        report.add(f"equalize_payload (n_data={n_data})", elapsed_ms)

        report.print()

    def test_costas_loop_timing(self, rng) -> None:
        """Time Costas loop phase tracking."""
        report = TimingReport("COSTAS LOOP TIMING")

        qpsk = QPSK()

        for n_symbols in [200, 500, 1000]:
            bits = rng.integers(0, 2, size=n_symbols * 2).astype(np.int8)
            symbols = qpsk.bits2symbols(bits)

            # Add phase offset
            phase = np.linspace(0, np.pi / 2, n_symbols)
            rotated = symbols * np.exp(1j * phase)

            elapsed_ms, _ = time_function(apply_costas_loop, rotated, qpsk, iterations=10)
            report.add(f"{n_symbols} symbols", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_golay_timing(self, rng) -> None:
        """Time Golay encoding/decoding."""
        report = TimingReport("GOLAY (24,12) TIMING")

        golay = Golay()

        # Golay works on 12-bit messages, can correct up to 3 errors per 24-bit block
        for n_messages in [1, 10, 50]:
            message = rng.integers(0, 2, size=12 * n_messages).astype(np.int8)

            elapsed_ms, encoded = time_function(golay.encode, message, iterations=100)
            report.add(f"encode {n_messages} msg(s)", elapsed_ms)

            # Add up to 2 errors per 24-bit block (safe for Golay correction)
            received = encoded.copy()
            for block_idx in range(n_messages):
                block_start = block_idx * 24
                # Add 2 random errors per block (Golay can correct up to 3)
                error_positions = rng.choice(24, size=2, replace=False) + block_start
                received[error_positions] ^= 1

            elapsed_ms, _ = time_function(golay.decode, received, iterations=100)
            report.add(f"decode {n_messages} msg(s)", elapsed_ms)

        report.print()


# ============================================================================
# Pipeline Timing Tests
# ============================================================================


class TestPipelineTiming:
    """Time complete TX/RX flows."""

    def test_full_tx_pipeline(self, rng, rrc_taps, pilot_config) -> None:
        """Time step-by-step TX breakdown."""
        report = TimingReport("TX PIPELINE TIMING BREAKDOWN")

        # Configuration
        k = 324
        ldpc_config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)
        qpsk = QPSK()
        sps = 4

        # Generate message
        message = rng.integers(0, 2, size=k).astype(np.int8)

        # Step 1: LDPC encode
        with timed_section(report, "LDPC encode"):
            codeword = ldpc_encode(message, ldpc_config)

        # Step 2: Modulation
        with timed_section(report, "QPSK modulation"):
            symbols = qpsk.bits2symbols(codeword)

        # Step 3: Pilot insertion
        with timed_section(report, "Pilot insertion"):
            symbols_with_pilots = insert_pilots(symbols, pilot_config)

        # Step 4: Pulse shaping
        with timed_section(report, "Pulse shaping (upsample + RRC)"):
            upsample_and_filter(symbols_with_pilots, sps, rrc_taps)

        report.print()

    def test_full_rx_pipeline(self, rng, rrc_taps, pilot_config, synchronizer) -> None:
        """Time step-by-step RX breakdown (mirrors pluto/receive.py)."""
        report = TimingReport("RX PIPELINE TIMING BREAKDOWN")

        # Configuration
        k = 324
        ldpc_config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)
        qpsk = QPSK()
        sps = 4
        sample_rate = 1e6

        # ====================================================================
        # Generate TX signal with preamble
        # ====================================================================
        message = rng.integers(0, 2, size=k).astype(np.int8)
        codeword = ldpc_encode(message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)
        symbols_with_pilots = insert_pilots(symbols, pilot_config)

        # Add preamble
        preamble = synchronizer.preamble
        full_frame = np.concatenate([preamble, symbols_with_pilots])
        tx_signal = upsample_and_filter(full_frame, sps, rrc_taps)

        # Add channel impairments
        channel = ChannelModel(ChannelConfig(snr_db=15.0, cfo_hz=50.0, seed=42))
        rx_signal = channel.apply(tx_signal)

        # JIT warmup for LDPC decode
        warmup_llrs = rng.normal(0, 1, ldpc_config.n)
        _ = ldpc_decode(warmup_llrs, ldpc_config, max_iterations=5)

        # ====================================================================
        # RX Pipeline
        # ====================================================================

        # Step 1: Matched filter
        with timed_section(report, "Matched filter"):
            filtered = signal.convolve(rx_signal, rrc_taps, mode="same")

        # Step 2: Preamble detection + CFO estimation
        with timed_section(report, "Preamble detection + CFO est"):
            sync_result = synchronizer.detect_preamble(filtered, sample_rate)

        if not sync_result.success:
            pass

        # Step 3: CFO correction
        with timed_section(report, "CFO correction"):
            t = np.arange(len(filtered)) / sample_rate
            cfo_correction = np.exp(-1j * 2 * np.pi * sync_result.cfo_hat_hz * t)
            corrected = filtered * cfo_correction

        # Step 4: Symbol extraction (downsample)
        with timed_section(report, "Symbol extraction (downsample)"):
            # Extract symbols after preamble
            start_idx = sync_result.long_zc_start + len(preamble) * sps
            n_payload_symbols = len(symbols_with_pilots)
            rx_symbols = corrected[start_idx : start_idx + n_payload_symbols * sps : sps]

        if len(rx_symbols) < n_payload_symbols:
            # Pad if needed for timing purposes
            rx_symbols = np.pad(rx_symbols, (0, n_payload_symbols - len(rx_symbols)))

        # Step 5: Channel equalization
        n_data = len(symbols)
        with timed_section(report, "Channel equalization"):
            equalized = equalize_payload(rx_symbols, n_data, pilot_config)

        # Step 6: Phase tracking (pilot-aided)
        with timed_section(report, "Phase tracking (pilot-aided)"):
            phase_corrected = pilot_aided_phase_track(equalized, n_data, pilot_config)

        # Step 7: Soft demodulation
        with timed_section(report, f"Soft demodulation ({qpsk.__class__.__name__})"):
            sigma_sq = qpsk.estimate_noise_variance(phase_corrected)
            llrs = qpsk.symbols2bits_soft(phase_corrected, sigma_sq=sigma_sq)

        # Step 8: LDPC decode
        with timed_section(report, "LDPC decode (max_iter=50)"):
            llrs_flat = llrs.flatten()
            if len(llrs_flat) < ldpc_config.n:
                llrs_flat = np.pad(llrs_flat, (0, ldpc_config.n - len(llrs_flat)))
            ldpc_decode(llrs_flat[: ldpc_config.n], ldpc_config, max_iterations=50)

        report.print()

    def test_tx_vs_rx_comparison(self, rng, rrc_taps, pilot_config, synchronizer) -> None:
        """Compare TX and RX timing to show asymmetry."""
        k = 324
        ldpc_config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)
        qpsk = QPSK()
        sps = 4
        sample_rate = 1e6

        # JIT warmup
        warmup_llrs = rng.normal(0, 1, ldpc_config.n)
        _ = ldpc_decode(warmup_llrs, ldpc_config, max_iterations=5)

        # Time TX
        message = rng.integers(0, 2, size=k).astype(np.int8)

        tx_start = time.perf_counter()
        codeword = ldpc_encode(message, ldpc_config)
        symbols = qpsk.bits2symbols(codeword)
        symbols_with_pilots = insert_pilots(symbols, pilot_config)
        preamble = synchronizer.preamble
        full_frame = np.concatenate([preamble, symbols_with_pilots])
        tx_signal = upsample_and_filter(full_frame, sps, rrc_taps)
        (time.perf_counter() - tx_start) * 1000

        # Apply channel
        channel = ChannelModel(ChannelConfig(snr_db=15.0, cfo_hz=50.0, seed=42))
        rx_signal = channel.apply(tx_signal)

        # Time RX
        rx_start = time.perf_counter()
        filtered = signal.convolve(rx_signal, rrc_taps, mode="same")
        sync_result = synchronizer.detect_preamble(filtered, sample_rate)
        t = np.arange(len(filtered)) / sample_rate
        corrected = filtered * np.exp(-1j * 2 * np.pi * sync_result.cfo_hat_hz * t)
        start_idx = sync_result.long_zc_start + len(preamble) * sps
        n_payload_symbols = len(symbols_with_pilots)
        rx_symbols = corrected[start_idx : start_idx + n_payload_symbols * sps : sps]
        if len(rx_symbols) < n_payload_symbols:
            rx_symbols = np.pad(rx_symbols, (0, n_payload_symbols - len(rx_symbols)))
        n_data = len(symbols)
        equalized = equalize_payload(rx_symbols, n_data, pilot_config)
        phase_corrected = pilot_aided_phase_track(equalized, n_data, pilot_config)
        llrs = qpsk.symbols2bits_soft(phase_corrected, sigma_sq=0.1)
        llrs_flat = llrs.flatten()
        if len(llrs_flat) < ldpc_config.n:
            llrs_flat = np.pad(llrs_flat, (0, ldpc_config.n - len(llrs_flat)))
        ldpc_decode(llrs_flat[: ldpc_config.n], ldpc_config, max_iterations=50)
        (time.perf_counter() - rx_start) * 1000



# ============================================================================
# Scaling Behavior Tests
# ============================================================================


class TestScalingBehavior:
    """Test how timing scales with input parameters."""

    def test_ldpc_decode_early_vs_forced(self, rng) -> None:
        """Compare LDPC decode time: early convergence vs forced full iterations.

        This test demonstrates why we must use noise-only LLRs for accurate
        worst-case timing: good LLRs converge in few iterations.
        """
        report = TimingReport("LDPC DECODE: EARLY CONVERGENCE vs FORCED FULL")

        k = 324
        config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)
        message = rng.integers(0, 2, size=k).astype(np.int8)
        codeword = ldpc_encode(message, config)

        # JIT warmup
        warmup_llrs = rng.normal(0, 1, config.n)
        _ = ldpc_decode(warmup_llrs, config, max_iterations=5)

        # High SNR LLRs - will converge in few iterations
        good_llrs = (2 * codeword.astype(np.float64) - 1) * 10  # Very clean
        elapsed_ms, _ = time_function(ldpc_decode, good_llrs, config, max_iterations=50, iterations=10)
        report.add("High SNR (converges early)", elapsed_ms)

        # Medium SNR LLRs - may converge
        medium_llrs = (2 * codeword.astype(np.float64) - 1) * 2 + rng.normal(0, 1, config.n)
        elapsed_ms, _ = time_function(ldpc_decode, medium_llrs, config, max_iterations=50, iterations=10)
        report.add("Medium SNR (may converge)", elapsed_ms)

        # Pure noise - will NOT converge, runs all 50 iterations
        noise_llrs = rng.normal(0, 1, config.n)
        elapsed_ms, _ = time_function(ldpc_decode, noise_llrs, config, max_iterations=50, iterations=10)
        report.add("Pure noise (forced 50 iter)", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_ldpc_decode_scaling(self, rng) -> None:
        """Time LDPC decode vs iteration count.

        Uses pure noise LLRs to force full iterations (no early convergence).
        """
        report = TimingReport("LDPC DECODE SCALING (iterations, forced full)")

        k = 324
        config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)

        # JIT warmup
        warmup_llrs = rng.normal(0, 1, config.n)
        _ = ldpc_decode(warmup_llrs, config, max_iterations=5)

        # Pure noise - will not converge, forces full iterations
        noise_llrs = rng.normal(0, 1, config.n)

        for max_iter in [10, 25, 50, 100]:
            elapsed_ms, _ = time_function(ldpc_decode, noise_llrs, config, max_iterations=max_iter, iterations=5)
            report.add(f"max_iter={max_iter}", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_ldpc_decode_payload_scaling(self, rng) -> None:
        """Time LDPC decode vs payload size and code rate.

        Uses pure noise LLRs to force full iterations.
        Tests all three block sizes (n=648, 1296, 1944) with various code rates.
        """
        report = TimingReport("LDPC DECODE SCALING (size & rate, forced full)")

        # JIT warmup with smallest config
        warmup_config = LDPCConfig(k=324, code_rate=CodeRates.HALF_RATE)
        warmup_llrs = rng.normal(0, 1, warmup_config.n)
        _ = ldpc_decode(warmup_llrs, warmup_config, max_iterations=10)

        # Test all three block sizes with multiple code rates
        test_configs = [
            # n=648 (small block)
            (324, CodeRates.HALF_RATE),  # k=324, n=648, r=1/2
            (540, CodeRates.FIVE_SIXTH_RATE),  # k=540, n=648, r=5/6
            # n=1296 (medium block)
            (648, CodeRates.HALF_RATE),  # k=648, n=1296, r=1/2
            (972, CodeRates.THREE_QUARTER_RATE),  # k=972, n=1296, r=3/4
            # n=1944 (large block)
            (972, CodeRates.HALF_RATE),  # k=972, n=1944, r=1/2
            (1296, CodeRates.TWO_THIRDS_RATE),  # k=1296, n=1944, r=2/3
            (1458, CodeRates.THREE_QUARTER_RATE),  # k=1458, n=1944, r=3/4
        ]

        for k, code_rate in test_configs:
            config = LDPCConfig(k=k, code_rate=code_rate)
            # Pure noise - forces full iterations
            noise_llrs = rng.normal(0, 1, config.n)

            elapsed_ms, _ = time_function(ldpc_decode, noise_llrs, config, max_iterations=50, iterations=3)
            rate_str = code_rate.name.replace("_RATE", "").replace("_", "/").lower()
            report.add(f"k={k:4d} n={config.n} r={rate_str}", elapsed_ms)

        report.print(sort_by_duration=False)

    def test_fft_convolution_scaling(self, rrc_taps, rng) -> None:
        """Compare direct vs FFT convolution scaling."""
        report = TimingReport("CONVOLUTION SCALING (direct vs FFT)")

        for n_samples in [1000, 5000, 10000, 50000]:
            rx_signal = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)

            # Direct convolution (scipy default for small signals)
            elapsed_direct, _ = time_function(signal.convolve, rx_signal, rrc_taps, mode="same", iterations=5)
            report.add(f"n={n_samples} (direct)", elapsed_direct)

            # FFT convolution
            elapsed_fft, _ = time_function(signal.fftconvolve, rx_signal, rrc_taps, mode="same", iterations=5)
            report.add(f"n={n_samples} (fft)", elapsed_fft)

        report.print(sort_by_duration=False)

    def test_synchronizer_signal_length_scaling(self, synchronizer, rrc_taps, rng) -> None:
        """Time synchronization vs signal length."""
        report = TimingReport("SYNCHRONIZATION SCALING (signal length)")

        sps = 4
        sample_rate = 1e6

        # Generate base signal with preamble
        preamble = synchronizer.preamble
        preamble_upsampled = upsample_and_filter(preamble, sps, rrc_taps)
        payload = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        payload_upsampled = upsample_and_filter(payload, sps, rrc_taps)
        base_signal = np.concatenate([preamble_upsampled, payload_upsampled])
        base_signal += rng.normal(0, 0.1, len(base_signal)) * (1 + 1j)

        # Matched filter
        filtered_base = signal.convolve(base_signal, rrc_taps, mode="same")

        for multiplier in [1, 2, 4, 8]:
            test_signal = np.tile(filtered_base, multiplier)
            elapsed_ms, _ = time_function(synchronizer.detect_preamble, test_signal, sample_rate, iterations=3)
            report.add(f"len={len(test_signal)}", elapsed_ms)

        report.print(sort_by_duration=False)


# ============================================================================
# Memory Usage Tests
# ============================================================================


class TestMemoryUsage:
    """Measure memory usage of cached structures."""

    def test_ldpc_cache_memory(self, rng) -> None:
        """Measure memory used by LDPC encoder/decoder caches."""

        def get_array_memory(arr) -> int:
            """Get memory in bytes for numpy array or sparse matrix."""
            if hasattr(arr, "nbytes"):
                return arr.nbytes
            if hasattr(arr, "data"):  # sparse matrix
                return arr.data.nbytes + arr.indices.nbytes + arr.indptr.nbytes
            return 0

        def format_bytes(b: int) -> str:
            """Format bytes as human-readable string."""
            if b < 1024:
                return f"{b} B"
            if b < 1024 * 1024:
                return f"{b / 1024:.1f} KB"
            return f"{b / (1024 * 1024):.2f} MB"

        # Clear caches first
        ldpc_clear_cache()

        # ALL valid LDPC configurations (3 block sizes × 4 code rates = 12 configs)
        test_configs = [
            # n=648 (all 4 rates)
            (324, CodeRates.HALF_RATE),
            (432, CodeRates.TWO_THIRDS_RATE),
            (486, CodeRates.THREE_QUARTER_RATE),
            (540, CodeRates.FIVE_SIXTH_RATE),
            # n=1296 (all 4 rates)
            (648, CodeRates.HALF_RATE),
            (864, CodeRates.TWO_THIRDS_RATE),
            (972, CodeRates.THREE_QUARTER_RATE),
            (1080, CodeRates.FIVE_SIXTH_RATE),
            # n=1944 (all 4 rates)
            (972, CodeRates.HALF_RATE),
            (1296, CodeRates.TWO_THIRDS_RATE),
            (1458, CodeRates.THREE_QUARTER_RATE),
            (1620, CodeRates.FIVE_SIXTH_RATE),
        ]


        total_h = 0
        total_encoding = 0
        total_decode = 0

        for k, code_rate in test_configs:
            config = LDPCConfig(k=k, code_rate=code_rate)
            message = rng.integers(0, 2, size=k).astype(np.int8)

            # Trigger cache population
            ldpc_encode(message, config)
            noise_llrs = rng.normal(0, 1, config.n)
            _ = ldpc_decode(noise_llrs, config, max_iterations=5)

            # Measure H matrix cache
            h_mem = get_array_memory(_h_cache.get(config, np.array([])))
            total_h += h_mem

            # Measure encoding cache (G matrix + H_permuted)
            enc_entry = _encoding_cache.get(config)
            enc_mem = 0
            if enc_entry:
                enc_mem = sum(get_array_memory(arr) for arr in enc_entry)
            total_encoding += enc_mem

            # Measure decode cache
            dec_entry = _decode_cache.get(config)
            dec_mem = 0
            if dec_entry:
                for item in dec_entry:
                    if hasattr(item, "nbytes") or hasattr(item, "data"):
                        dec_mem += get_array_memory(item)
            total_decode += dec_mem

            code_rate.name.replace("_RATE", "").replace("_", "/").lower()
            h_mem + enc_mem + dec_mem

        total_h + total_encoding + total_decode

        # Cache size counts


# ============================================================================
# Summary Test
# ============================================================================


class TestPerformanceSummary:
    """Generate a summary of all component timings."""

    def test_all_components_summary(self, rng, rrc_taps, pilot_config, synchronizer) -> None:
        """Quick summary of all major components."""
        report = TimingReport("ALL COMPONENTS SUMMARY (single run)")

        k = 324
        ldpc_config = LDPCConfig(k=k, code_rate=CodeRates.HALF_RATE)
        qpsk = QPSK()
        sps = 4
        sample_rate = 1e6

        # Generate test data
        message = rng.integers(0, 2, size=k).astype(np.int8)

        # JIT warmup for LDPC
        warmup_llrs = rng.normal(0, 1, ldpc_config.n)
        _ = ldpc_decode(warmup_llrs, ldpc_config, max_iterations=5)

        # LDPC encode
        with timed_section(report, "LDPC encode"):
            codeword = ldpc_encode(message, ldpc_config)

        # LDPC decode
        llrs = (2 * codeword.astype(np.float64) - 1) * 4 + rng.normal(0, 0.5, ldpc_config.n)
        with timed_section(report, "LDPC decode (50 iter)"):
            _ = ldpc_decode(llrs, ldpc_config, max_iterations=50)

        # Modulation
        with timed_section(report, "QPSK modulate"):
            symbols = qpsk.bits2symbols(codeword)

        with timed_section(report, "QPSK soft demod"):
            _ = qpsk.symbols2bits_soft(symbols, sigma_sq=0.1)

        # Pilots
        with timed_section(report, "Pilot insert"):
            symbols_with_pilots = insert_pilots(symbols, pilot_config)

        with timed_section(report, "Phase track (pilot)"):
            _ = pilot_aided_phase_track(symbols_with_pilots, len(symbols), pilot_config)

        # Pulse shaping
        preamble = synchronizer.preamble
        full_frame = np.concatenate([preamble, symbols_with_pilots])
        with timed_section(report, "Pulse shape"):
            tx_signal = upsample_and_filter(full_frame, sps, rrc_taps)

        # Channel
        channel = ChannelModel(ChannelConfig(snr_db=15.0, cfo_hz=50.0, seed=42))
        with timed_section(report, "Channel apply"):
            rx_signal = channel.apply(tx_signal)

        # Matched filter
        with timed_section(report, "Matched filter"):
            filtered = signal.convolve(rx_signal, rrc_taps, mode="same")

        # Synchronization
        with timed_section(report, "Preamble detect"):
            _ = synchronizer.detect_preamble(filtered, sample_rate)

        # Equalization
        n_data = len(symbols)
        with timed_section(report, "Equalize"):
            _ = equalize_payload(symbols_with_pilots, n_data, pilot_config)

        # Costas loop
        with timed_section(report, "Costas loop"):
            _ = apply_costas_loop(symbols, qpsk)

        # Golay
        golay = Golay()
        golay_msg = rng.integers(0, 2, size=12).astype(np.int8)
        with timed_section(report, "Golay encode"):
            golay_enc = golay.encode(golay_msg)
        with timed_section(report, "Golay decode"):
            _ = golay.decode(golay_enc)

        report.print()
