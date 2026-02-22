"""Tests for synchronization algorithms - frame detection and CFO estimation.

Uses matched-filter (cross-correlation) based detection for robust performance
at low SNR. This is the standard approach used in WiFi, LTE, and 5G systems.
"""

import logging
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest

from modules.channel import (
    ChannelConfig,
    ChannelModel,
)
from modules.synchronization import Synchronizer, SynchronizerConfig, generate_zadoff_chu
from modules.modulation import Modulator, BPSK, QPSK
from modules.costas_loop import apply_costas_loop, _costas_loop_iteration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Moved here for pytest debug output

# Explicitly set the logging level for modules.costas_loop to DEBUG
SAMPLE_RATE = 1e6
DELAY_PADDING = 10000

# Tolerance constants for test assertions
MAX_DELAY_ERROR_SAMPLES = 2
MAX_CFO_ERROR_HZ = 200
MAX_TIMING_ERROR_SAMPLES = 3
ZC_SEQUENCE_LENGTH = 61

config = SynchronizerConfig()
sync = Synchronizer(config)


def run_sync(
    *,
    actual_delay: float,
    actual_cfo: float,
    snr: float,
    seed: int,
) -> dict[str, object]:
    """Run one full synchronization trial. Returns a results dict."""
    channel_config = ChannelConfig(
        sample_rate=SAMPLE_RATE,
        snr_db=snr,
        seed=seed,
        cfo_hz=actual_cfo,
        initial_phase_rad=np.random.default_rng(seed).uniform(0, 2 * np.pi),
        delay_samples=actual_delay,
    )
    channel = ChannelModel(channel_config)

    tx = np.concatenate([sync.preamble, np.zeros(DELAY_PADDING, dtype=complex)])
    rx = channel.apply(tx)

    result = sync.detect_preamble(rx, SAMPLE_RATE)

    if not result.success:
        return {"success": False, "reason": result.reason}

    true_zc_long_start = actual_delay + config.n_short_reps * config.n_short

    return {
        "success": True,
        "delay_true": actual_delay,
        "delay_hat": result.d_hat,
        "delay_error": result.d_hat - actual_delay,
        "cfo_true_hz": actual_cfo,
        "cfo_hat_hz": result.cfo_hat_hz,
        "cfo_error_hz": result.cfo_hat_hz - actual_cfo,
        "timing_true": true_zc_long_start,
        "timing_hat": result.long_zc_start,
        "timing_error": result.long_zc_start - true_zc_long_start,
        "snr": snr,
        "seed": seed,
    }


class TestZadoffChu:
    """Tests for Zadoff-Chu sequence generation."""

    def test_sequence_length(self) -> None:
        """Generated sequence should have the requested length."""
        seq = generate_zadoff_chu(u=7, n_zc=ZC_SEQUENCE_LENGTH)
        np.testing.assert_equal(len(seq), ZC_SEQUENCE_LENGTH)

    def test_constant_amplitude(self) -> None:
        """Zadoff-Chu sequences have constant amplitude."""
        seq = generate_zadoff_chu(u=7, n_zc=ZC_SEQUENCE_LENGTH)
        np.testing.assert_allclose(np.abs(seq), 1.0, atol=1e-10)

    def test_different_roots_are_different(self) -> None:
        """Different root indices should produce different sequences."""
        seq1 = generate_zadoff_chu(u=7, n_zc=ZC_SEQUENCE_LENGTH)
        seq2 = generate_zadoff_chu(u=11, n_zc=ZC_SEQUENCE_LENGTH)
        if np.allclose(seq1, seq2):
            pytest.fail("Sequences with different roots should not be equal")


class TestSynchronizer:
    """Tests for preamble detection and CFO estimation."""

    def test_preamble_length(self) -> None:
        """Preamble should be n_short_reps * n_short + n_long samples."""
        expected = config.n_short_reps * config.n_short + config.n_long
        np.testing.assert_equal(len(sync.preamble), expected)

    def test_detection_no_impairments(self) -> None:
        """Detection should succeed with no channel impairments."""
        result = run_sync(actual_delay=0, actual_cfo=0, snr=30.0, seed=42)
        if not result["success"]:
            pytest.fail("Detection failed with no impairments")

    def test_detection_with_delay(self) -> None:
        """Detection should succeed and estimate delay correctly."""
        result = run_sync(actual_delay=500, actual_cfo=0, snr=30.0, seed=42)
        if not result["success"]:
            pytest.fail("Detection failed with delay")
        if abs(cast("float", result["delay_error"])) > MAX_DELAY_ERROR_SAMPLES:
            pytest.fail(
                f"Delay error {result['delay_error']} exceeds tolerance {MAX_DELAY_ERROR_SAMPLES}",
            )

    def test_cfo_estimation_high_snr(self) -> None:
        """CFO estimate should be accurate at high SNR."""
        result = run_sync(actual_delay=100, actual_cfo=5000.0, snr=20.0, seed=42)
        if not result["success"]:
            pytest.fail("Detection failed at high SNR")
        if abs(cast("float", result["cfo_error_hz"])) >= MAX_CFO_ERROR_HZ:
            pytest.fail(
                f"CFO error {result['cfo_error_hz']} Hz exceeds tolerance {MAX_CFO_ERROR_HZ} Hz",
            )

    @pytest.mark.parametrize("snr", [5.0, 10.0, 20.0])
    def test_detection_at_various_snr(self, snr: float) -> None:
        """Detection should succeed at moderate-to-high SNR."""
        result = run_sync(actual_delay=1000, actual_cfo=1000.0, snr=snr, seed=123)
        if not result["success"]:
            pytest.fail(f"Detection failed at SNR={snr}")

    def test_fine_timing_accuracy(self) -> None:
        """Fine timing (long ZC) should be within a few samples."""
        result = run_sync(actual_delay=200, actual_cfo=0, snr=20.0, seed=42)
        if not result["success"]:
            pytest.fail("Detection failed for fine timing test")
        if abs(cast("float", result["timing_error"])) > MAX_TIMING_ERROR_SAMPLES:
            pytest.fail(
                f"Timing error {result['timing_error']} exceeds tolerance {MAX_TIMING_ERROR_SAMPLES}",
            )


# =============================================================================
# Costas Loop Tests
# =============================================================================

TEST_SYMBOLS_LEN = 1000
COSTAS_ALPHA = 0.1
COSTAS_BETA = 0.02
MAX_PHASE_ERROR_RAD = 0.2 # Tolerance for phase lock (e.g., ~11.45 degrees)
COSTAS_SETTLING_SYMBOLS = 10  # Number of symbols after which Costas loop should be locked for direct tests
COSTAS_PIPELINE_LOCK_THRESHOLD = 10 # Number of symbols for pipeline simulation to consider lock "acquired"


@pytest.fixture(params=[BPSK(), QPSK()])
def modulator_instance(request) -> Modulator:
    """Fixture to provide BPSK and QPSK modulator instances."""
    return request.param


def _simulate_costas_signal(
    modulator: Modulator,
    num_symbols: int,
    initial_phase_rad: float,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates modulated symbols with a phase offset and AWGN."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_symbols * modulator.bits_per_symbol)
    base_symbols = modulator.bits2symbols(bits)

    # Apply initial phase offset
    phase_shifted_symbols = base_symbols * np.exp(1j * initial_phase_rad)

    # Add AWGN
    # Assuming average symbol energy is 1 for BPSK/QPSK with default modulator scaling
    symbol_energy = np.mean(np.abs(base_symbols) ** 2)
    noise_power = symbol_energy / (10 ** (snr_db / 10))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), num_symbols) + 1j * rng.normal(0, np.sqrt(noise_power / 2), num_symbols))
    noisy_symbols = phase_shifted_symbols + noise
    return noisy_symbols, base_symbols


class TestCostasLoop:
    """Tests for the Costas loop implementation."""

    def test_costas_loop_perfect_lock_bpsk_qpsk(self, modulator_instance: Modulator) -> None:
        """Test that Costas loop can perfectly lock on a noiseless signal with phase offset."""
        mod = modulator_instance
        initial_phase = np.pi / 4  # 45 degrees offset
        noisy_symbols, base_symbols = _simulate_costas_signal(mod, TEST_SYMBOLS_LEN, initial_phase, snr_db=100.0, seed=42)

        corrected_symbols, _ = apply_costas_loop(noisy_symbols, mod, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)

        # Check phase error after lock (using the latter half of symbols for stable lock)
        # We need to compute the phase difference between corrected and original symbols
        # np.angle(A * conj(B)) gives the angle from B to A
        phase_errors = np.angle(corrected_symbols * np.conj(base_symbols))

        # Resolve phase ambiguity: Costas loops for M-PSK can lock to
        # phase_offset + k * (2*pi/M). We check if the final phase error,
        # after normalizing for this ambiguity, is within tolerance.
        if isinstance(modulator_instance, BPSK):
            # For BPSK, ambiguities are 0 and pi. Wrap error to [-pi/2, pi/2)
            wrapped_errors = (phase_errors + np.pi/2) % np.pi - np.pi/2
        elif isinstance(modulator_instance, QPSK):
            # For QPSK, ambiguities are 0, pi/2, pi, 3pi/2. Wrap error to [-pi/4, pi/4)
            wrapped_errors = (phase_errors + np.pi/4) % (np.pi/2) - np.pi/4
        else:
            wrapped_errors = phase_errors # Fallback
        
        final_phase_error = np.mean(np.abs(wrapped_errors[TEST_SYMBOLS_LEN // 2 :]))

        if final_phase_error >= MAX_PHASE_ERROR_RAD:
            pytest.fail(
                f"Final phase error too large for {modulator_instance.__class__.__name__}: {final_phase_error:.3f} rad"
            )

    @pytest.mark.parametrize("initial_phase", [np.pi / 8, np.pi / 3, 2 * np.pi / 3, 5 * np.pi / 6])
    def test_costas_loop_lock_on_speed(self, modulator_instance: Modulator, initial_phase: float) -> None:
        """Test how quickly the Costas loop locks onto different initial phase offsets."""
        mod = modulator_instance
        # Use a slightly longer sequence to ensure enough time for locking
        num_symbols = TEST_SYMBOLS_LEN * 2
        noisy_symbols, base_symbols = _simulate_costas_signal(mod, num_symbols, initial_phase, snr_db=20.0, seed=123)
        corrected_symbols, _ = apply_costas_loop(noisy_symbols, mod, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)

        # Check if the phase error is within tolerance after LOCK_THRESHOLD_SYMBOLS
        # We'll consider the loop locked if the average absolute phase error in the
        # symbols after COSTAS_SETTLING_SYMBOLS is below the tolerance.
        phase_errors = np.angle(corrected_symbols[COSTAS_SETTLING_SYMBOLS:] * np.conj(base_symbols[COSTAS_SETTLING_SYMBOLS:]))
        
        # Resolve phase ambiguity before calculating mean error
        if isinstance(modulator_instance, BPSK):
            wrapped_errors = (phase_errors + np.pi/2) % np.pi - np.pi/2
        elif isinstance(modulator_instance, QPSK):
            wrapped_errors = (phase_errors + np.pi/4) % (np.pi/2) - np.pi/4
        else:
            wrapped_errors = phase_errors

        mean_abs_phase_error = np.mean(np.abs(wrapped_errors))

        if mean_abs_phase_error >= MAX_PHASE_ERROR_RAD:
            pytest.fail(
                f"Costas loop for {modulator_instance.__class__.__name__} failed to lock quickly for initial_phase={initial_phase:.2f} rad (mean phase error: {mean_abs_phase_error:.3f} rad)"
            )

    @pytest.mark.parametrize("snr", [15.0, 10.0, 5.0, 0.0])
    def test_costas_loop_error_tolerance_snr(self, modulator_instance: Modulator, snr: float) -> None:
        """Test Costas loop performance at various SNR levels."""
        mod = modulator_instance
        initial_phase = np.pi / 8  # Small fixed offset
        noisy_symbols, base_symbols = _simulate_costas_signal(mod, TEST_SYMBOLS_LEN, initial_phase, snr_db=snr, seed=456)

        corrected_symbols, _ = apply_costas_loop(noisy_symbols, mod, alpha=COSTAS_ALPHA, beta=COSTAS_BETA)

        # Evaluate performance based on average phase error in the latter half of symbols
        phase_errors = np.angle(corrected_symbols[TEST_SYMBOLS_LEN // 2 :] * np.conj(base_symbols[TEST_SYMBOLS_LEN // 2 :]))
        
        # Resolve phase ambiguity before calculating mean error
        if isinstance(mod, BPSK):
            wrapped_errors = (phase_errors + np.pi/2) % np.pi - np.pi/2
        elif isinstance(mod, QPSK):
            wrapped_errors = (phase_errors + np.pi/4) % (np.pi/2) - np.pi/4
        else:
            wrapped_errors = phase_errors

        mean_abs_phase_error = np.mean(np.abs(wrapped_errors))


        # Define expected performance thresholds. These are heuristic and might need tuning.
        # For SNRs >= 10dB, we expect reasonably good performance
        if snr >= 10.0:
            if mean_abs_phase_error >= MAX_PHASE_ERROR_RAD:
                pytest.fail(
                    f"Costas loop for {mod.__class__.__name__} at {snr=:.1f} dB failed to maintain lock (mean phase error: {mean_abs_phase_error:.3f} rad)"
                )
        elif snr >= 5.0:  # Between 5dB and 10dB, performance might degrade, allow larger error
            if mean_abs_phase_error >= (MAX_PHASE_ERROR_RAD * 2):
                pytest.fail(
                    f"Costas loop for {mod.__class__.__name__} at {snr=:.1f} dB showed excessive phase error (mean phase error: {mean_abs_phase_error:.3f} rad)"
                )
        else:  # Below 5dB, it's expected to struggle more, just ensure it doesn't completely diverge with NaN
            if np.isnan(mean_abs_phase_error):
                pytest.fail(
                    f"Costas loop for {mod.__class__.__name__} at {snr=:.1f} dB diverged (NaN phase error)"
                )
    
# =============================================================================
# Sweep helpers (used by __main__ for detailed analysis)
# =============================================================================

def _simulate_and_track_costas_loop(
    modulator: Modulator,
    num_symbols: int,
    initial_phase_rad: float,
    snr_db: float,
    seed: int,
    alpha: float = COSTAS_ALPHA, # Updated default
    beta: float = COSTAS_BETA, # Updated default
) -> dict[str, np.ndarray | float | Modulator]:
    """Simulates a Costas loop run and explicitly collects phase_estimate history.

    This helper function uses the core `_costas_loop_iteration` logic from the production module,
    but wraps it to track and return the phase history for plotting.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_symbols * modulator.bits_per_symbol)
    base_symbols = modulator.bits2symbols(bits)

    # Apply initial phase offset
    phase_shifted_symbols = base_symbols * np.exp(1j * initial_phase_rad)

    # Add AWGN
    symbol_energy = np.mean(np.abs(base_symbols) ** 2)
    noise_power = symbol_energy / (10 ** (snr_db / 10))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), num_symbols) + 1j * rng.normal(0, np.sqrt(noise_power / 2), num_symbols))
    noisy_symbols = phase_shifted_symbols + noise

    n = len(noisy_symbols)
    phase_estimate = 0.0
    integrator = 0.0
    phase_estimate_history = np.zeros(n, dtype=float)
    
    for i, sym in enumerate(noisy_symbols):
        _, phase_estimate, integrator = _costas_loop_iteration(
            sym, modulator, phase_estimate, integrator, alpha, beta
        )
        phase_estimate_history[i] = phase_estimate

    return {
        "modulator": modulator,
        "initial_phase_rad": initial_phase_rad,
        "snr_db": snr_db,
        "phase_estimate_history": phase_estimate_history,
    }


def _run_sweep(
    cfo_values: list[float],
    snr_values: list[float],
    delays: list[int],
    n_seeds: int = 10,
) -> list[dict[str, object]]:
    """Run a parameter sweep of synchronization trials.

    Iterates over all combinations of CFO, SNR, delay, and random seeds.
    """
    results: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=n_seeds).tolist()

    for cfo in cfo_values:
        for snr in snr_values:
            for delay in delays:
                for seed in seeds:
                    r = run_sync(
                        actual_delay=delay,
                        actual_cfo=cfo,
                        snr=snr,
                        seed=seed,
                    )
                    r.update({"cfo_sweep": cfo, "snr_sweep": snr, "delay_sweep": delay})
                    results.append(r)
    return results


def _print_summary(results: list[dict[str, object]]) -> None:
    """Print a summary of sweep results broken down by SNR.
    Currently focuses on Zadoff-Chu synchronization results.
    """
    successful = [r for r in results if r["success"]]
    if not successful:
        logger.info("No successful ZC synchronization trials.")
        return

    logger.info("--- Zadoff-Chu Synchronization Summary ---")
    for snr in sorted(cast("set[float]", {r["snr_sweep"] for r in results})):
        snr_all = [r for r in results if r["snr_sweep"] == snr]
        snr_ok = [r for r in successful if r["snr_sweep"] == snr]
        cfo_errs = [abs(cast("float", r["cfo_error_hz"])) for r in snr_ok]
        tim_errs = [abs(cast("float", r["timing_error"])) for r in snr_ok]
        logger.info(
            "SNR=%5.1f dB: %d/%d detected, CFO err mean=%.1f Hz max=%.1f Hz, "
            "timing err mean=%.1f max=%.1f samples",
            snr,
            len(snr_ok),
            len(snr_all),
            np.mean(cfo_errs),
            np.max(cfo_errs),
            np.mean(tim_errs),
            np.max(tim_errs),
        )


def _plot_results(results: list[dict[str, object]]) -> None:
    """Plot synchronization sweep results: CFO error, timing error vs SNR and CFO."""
    successful = [r for r in results if r["success"]]
    snrs = sorted(cast("set[float]", {r["snr_sweep"] for r in results}))
    cfos = sorted(cast("set[float]", {r["cfo_sweep"] for r in results}))

    _fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # CFO error vs SNR
    ax = axes[0]
    cfo_err_by_snr = [[r["cfo_error_hz"] for r in successful if r["snr_sweep"] == s] for s in snrs]
    ax.boxplot(cfo_err_by_snr, tick_labels=[str(s) for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("CFO error (Hz)")
    ax.set_title("CFO estimation error vs SNR")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)

    # Timing error vs SNR (long ZC detection)
    ax = axes[1]
    tim_err_by_snr = [[r["timing_error"] for r in successful if r["snr_sweep"] == s] for s in snrs]
    ax.boxplot(tim_err_by_snr, tick_labels=[str(s) for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Timing error (samples)")
    ax.set_title("Long ZC timing error vs SNR")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)

    # CFO error vs actual CFO
    ax = axes[2]
    cfo_err_by_cfo = [[r["cfo_error_hz"] for r in successful if r["cfo_sweep"] == c] for c in cfos]
    ax.boxplot(cfo_err_by_cfo, tick_labels=[f"{c / 1e3:.0f}k" for c in cfos])
    ax.set_xlabel("Actual CFO (kHz)")
    ax.set_ylabel("CFO error (Hz)")
    ax.set_title("CFO estimation error vs CFO magnitude")
    ax.axhline(0, color="r", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("examples/sync_test_results.png", dpi=150)
    plt.show(block=False)


def _run_costas_pipeline(
    cfo_values: list[float],
    snr_values: list[float],
    modulator: Modulator,
    num_payload_symbols: int = 2000,
) -> list[dict[str, object]]:
    """
    Run a sweep simulating the full pipeline: ZC sync -> Costas loop.
    """
    results: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    sps = 1  # For simplicity, assuming symbol rate processing after timing recovery
    if sps != 1:
        raise NotImplementedError("This simulation assumes sps=1 for the Costas loop part.")

    for cfo in cfo_values:
        for snr in snr_values:
            seed = rng.integers(0, 2**31)
            delay = 100  # Fixed delay for this test is fine

            # --- Full pipeline simulation ---
            # 1. Generate payload and prepend preamble
            bits = rng.integers(0, 2, size=num_payload_symbols * modulator.bits_per_symbol)
            payload_symbols = modulator.bits2symbols(bits)
            tx_frame = np.concatenate([sync.preamble, payload_symbols])

            # 2. Apply channel impairments
            channel_config = ChannelConfig(
                sample_rate=SAMPLE_RATE,
                snr_db=snr,
                seed=seed,
                cfo_hz=cfo,
                delay_samples=delay,
            )
            channel = ChannelModel(channel_config)
            rx_frame = channel.apply(tx_frame)

            # 3. Run ZC preamble detection for timing and coarse CFO
            zc_result = sync.detect_preamble(rx_frame, SAMPLE_RATE)
            
            res: dict[str, object] = {
                "cfo_sweep": cfo,
                "snr_sweep": snr,
                "zc_success": zc_result.success,
                "cfo_true_hz": cfo,
                "cfo_hat_hz": zc_result.cfo_hat_hz,
                "cfo_error_hz": zc_result.cfo_hat_hz - cfo,
                "timing_error": zc_result.long_zc_start - (delay + len(sync.preamble)),
            }
            logger.debug(f"Trial (CFO={cfo}, SNR={snr}, Seed={seed}): ZC success: {zc_result.success}")

            if not zc_result.success:
                logger.debug(f"ZC detection failed for (CFO={cfo}, SNR={snr}). Reason: {zc_result.reason}")
                results.append(res)
                continue

            # This block is reached IF zc_result.success is True.
            # 4. Coarse CFO correction and payload extraction
            # In a real receiver, we'd use the fine timing estimate. Here we simplify.
            payload_start_idx = len(sync.preamble) + delay
            # Adjust payload_end_idx to ensure it doesn't exceed rx_frame length
            payload_end_idx = min(payload_start_idx + len(payload_symbols), len(rx_frame))
            
            if payload_end_idx <= payload_start_idx: # This means no payload or frame too short
                res["costas_success"] = False
                res["reason"] = "Payload segment too short or invalid after ZC success."
                logger.debug(f"Payload segment too short or invalid for (CFO={cfo}, SNR={snr}). "
                             f"payload_start_idx={payload_start_idx}, payload_end_idx={payload_end_idx}, "
                             f"len(rx_frame)={len(rx_frame)}")
                results.append(res)
                continue

            rx_payload = rx_frame[payload_start_idx:payload_end_idx]

            try:
                # 5. Seed Costas loop with ZC CFO estimate
                # Convert CFO from Hz to radians per symbol
                cfo_hat_rad_per_symbol = 2 * np.pi * zc_result.cfo_hat_hz / (SAMPLE_RATE / sps)
                
                _corrected_symbols, phase_estimates = apply_costas_loop(
                    rx_payload,
                    modulator,
                    alpha=COSTAS_ALPHA,
                    beta=COSTAS_BETA,
                    initial_freq_offset_rad_per_symbol=cfo_hat_rad_per_symbol,
                )
                
                # --- NEW: Check for actual lock success based on phase error ---
                # Compare corrected symbols to the original payload symbols after the lock threshold
                # Need to use the original payload_symbols (base_symbols) for comparison
                # Note: payload_symbols were generated before channel impairments
                
                # Ensure we have enough symbols to check after lock threshold
                if len(_corrected_symbols) <= LOCK_THRESHOLD_SYMBOLS:
                    raise ValueError(f"Not enough corrected symbols ({len(_corrected_symbols)}) for lock-in check with threshold {LOCK_THRESHOLD_SYMBOLS}")

                # Calculate phase errors after the lock threshold
                # np.angle(A * conj(B)) gives the angle from B to A
                
                # The payload_symbols variable refers to the original full payload.
                # We need to slice it to match the actual length of rx_payload (and thus _corrected_symbols)
                # This ensures the operands for the complex multiplication have compatible shapes.
                original_payload_segment = payload_symbols[:len(_corrected_symbols)]
                
                # Calculate the actual instantaneous phase offset at the input to the Costas loop
                actual_input_phase_offset = np.angle(rx_payload * np.conj(original_payload_segment))

                phase_errors = np.angle(
                    _corrected_symbols[LOCK_THRESHOLD_SYMBOLS:] * np.conj(original_payload_segment[LOCK_THRESHOLD_SYMBOLS:])
                )
                mean_abs_phase_error = np.mean(np.abs(phase_errors))

                if mean_abs_phase_error >= MAX_PHASE_ERROR_RAD:
                    logger.warning(
                        f"Costas loop failed to lock to spec for (CFO={cfo}, SNR={snr}). "
                        f"Mean phase error: {mean_abs_phase_error:.3f} rad exceeds tolerance {MAX_PHASE_ERROR_RAD:.3f} rad"
                    )
                    res["met_lock_spec"] = False # Add a new field to indicate if it met spec
                else:
                    logger.debug(f"Costas loop successfully locked to spec for (CFO={cfo}, SNR={snr}). Mean abs phase error: {mean_abs_phase_error:.3f} rad.")
                    res["met_lock_spec"] = True # Met spec

                # 6. Analyze residual frequency error from the locked Costas loop
                # The integrator term converges to the residual frequency offset
                # We can estimate it from the slope of the phase estimate history
                # Use the latter half of the symbols for a stable estimate
                if len(phase_estimates) < 2: # Need at least 2 points for polyfit
                    # This should ideally not happen if apply_costas_loop returns proper phase_estimates
                    logger.warning(f"Not enough phase estimates for residual CFO analysis for (CFO={cfo}, SNR={snr}).")
                    residual_freq_hz = np.nan # Indicate invalid result
                    residual_cfo_error_hz = np.nan
                else:
                    stable_phase_est = phase_estimates[len(phase_estimates) // 2:]
                    symbol_indices = np.arange(len(stable_phase_est))
                    # polyfit returns [slope, intercept]
                    residual_freq_rad_per_symbol = np.polyfit(symbol_indices, stable_phase_est, 1)[0]
                    
                    # Convert back to Hz
                    residual_freq_hz = residual_freq_rad_per_symbol * (SAMPLE_RATE / sps) / (2 * np.pi)
                    residual_cfo_error_hz = residual_freq_hz - (cfo - zc_result.cfo_hat_hz)
                
                res.update({
                    "costas_success": True, # Always True if it reaches here without an exception, for plotting
                    "costas_phase_history": phase_estimates,
                    "actual_input_phase_offset": actual_input_phase_offset, # NEW: Store actual input phase offset
                    "residual_cfo_hz": residual_freq_hz,
                    "residual_cfo_error_hz": residual_cfo_error_hz,
                    "mean_abs_phase_error": mean_abs_phase_error,
                })

            except Exception as e:
                res["costas_success"] = False
                res["reason"] = f"Error during Costas loop processing: {e}"
                logger.error(f"Costas loop error for (CFO={cfo}, SNR={snr}): {e}")
            finally:
                results.append(res) # Always append result, even if Costas failed

    return results


def _print_costas_summary(results: list[dict[str, object]]) -> None:
    """Print a summary of Costas loop sweep results broken down by SNR."""
    successful_costas = [r for r in results if r.get("costas_success")]
    if not successful_costas:
        logger.info("No successful Costas loop trials.")
        return

    logger.info("\n--- Costas Loop Synchronization Summary (seeded by ZC estimate) ---")
    for snr in sorted(cast("set[float]", {r["snr_sweep"] for r in successful_costas})):
        snr_ok = [r for r in successful_costas if r["snr_sweep"] == snr]
        met_spec_count = sum(1 for r in snr_ok if r.get("met_lock_spec", False)) # Count trials that met the lock spec
        residual_cfo_errs = [abs(cast(float, r["residual_cfo_error_hz"])) for r in snr_ok if r.get("met_lock_spec", False)]
        mean_abs_phase_errors = [cast(float, r["mean_abs_phase_error"]) for r in snr_ok if r.get("met_lock_spec", False)]

        # Handle cases where no trials met the spec to avoid division by zero for mean/max
        if not residual_cfo_errs:
            mean_cfo_err = np.nan
            max_cfo_err = np.nan
        else:
            mean_cfo_err = np.mean(residual_cfo_errs)
            max_cfo_err = np.max(residual_cfo_errs)
        
        if not mean_abs_phase_errors:
            mean_phase_err = np.nan
            max_phase_err = np.nan
        else:
            mean_phase_err = np.mean(mean_abs_phase_errors)
            max_phase_err = np.max(mean_abs_phase_errors)

        logger.info(
            "SNR=%5.1f dB: %d/%d trials met lock spec, Residual CFO err (met spec only) mean=%.1f Hz max=%.1f Hz, "
            "Mean Abs Phase Error (met spec only) mean=%.3f rad max=%.3f rad",
            snr,
            met_spec_count,
            len(snr_ok),
            mean_cfo_err,
            max_cfo_err,
            mean_phase_err,
            max_phase_err,
        )

def _plot_costas_results(costas_results: list[dict[str, object]]) -> None:
    """Plot Costas loop phase trajectories, organized by CFO and SNR."""
    costas_ok = [r for r in costas_results if r.get("costas_success")]
    if not costas_ok:
        logger.info("No successful Costas loop results to plot.")
        return

    unique_cfos = sorted(list(set(r["cfo_sweep"] for r in costas_ok)))
    unique_snrs = sorted(list(set(r["snr_sweep"] for r in costas_ok)))

    # Create a grid of subplots
    fig, axes = plt.subplots(len(unique_cfos), len(unique_snrs), figsize=(12, 10), sharex=True, sharey=True, squeeze=False)

    # Plot for each (CFO, SNR) combination
    for i, cfo in enumerate(unique_cfos):
        for j, snr in enumerate(unique_snrs):
            ax = axes[i, j]
            ax_trials = [r for r in costas_ok if r["cfo_sweep"] == cfo and r["snr_sweep"] == snr]
            
            # Assuming one trial per (CFO, SNR) combination due to the sweeps
            if ax_trials:
                r = ax_trials[0] # Get the single result for this subplot
                phase_hist = cast("np.ndarray", r["costas_phase_history"])
                actual_phase = cast("np.ndarray", r["actual_input_phase_offset"])
                met_spec = r.get("met_lock_spec", False)

                plot_len = min(len(phase_hist), 500) 
                
                ax.plot(phase_hist[:plot_len], color='blue', label="Estimated Phase")
                ax.plot(actual_phase[:plot_len], color='red', linestyle='--', label="Actual Phase")
                
                ax.set_title(f"CFO={cfo/1e3:.0f}kHz, SNR={snr:.0f}dB (Met Spec: {met_spec})")
                ax.grid(True, which="both", linestyle="--", alpha=0.5)
                ax.set_ylim([-np.pi, np.pi]) # Consistent Y-axis limits
                ax.axhline(0, color='grey', linestyle=':', linewidth=0.8) # Add 0-line

                # Add thresholds if not met, for better context
                if not met_spec:
                    ax.axhline(MAX_PHASE_ERROR_RAD, color='green', linestyle=':', alpha=0.7, label='Max Error Thresh')
                    ax.axhline(-MAX_PHASE_ERROR_RAD, color='green', linestyle=':', alpha=0.7)
            else:
                ax.set_title(f"CFO={cfo/1e3:.0f}kHz, SNR={snr:.0f}dB (No Data)")
                ax.grid(True, which="both", linestyle="--", alpha=0.5)


    # Set common labels for the entire figure
    fig.supxlabel("Symbol Index")
    fig.supylabel("Phase (rad)")

    # Create a single legend for the entire figure if there are labels
    # Use dummy lines to collect unique labels for a global legend
    handles = []
    labels = []
    
    # Collect handles/labels from one of the subplots for the global legend
    # This assumes all subplots will show these general labels if data is present
    if costas_ok:
        # Create dummy plots for legend entries
        dummy_ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[]) # Invisible axes
        handles.append(dummy_ax.plot([], [], color='blue', linestyle='-')[0])
        labels.append("Estimated Phase")
        handles.append(dummy_ax.plot([], [], color='red', linestyle='--')[0])
        labels.append("Actual Phase")
        # Only add threshold label if it's relevant (i.e., if there are cases where it's not met)
        if any(not r.get("met_lock_spec", True) for r in costas_ok):
            handles.append(dummy_ax.plot([], [], color='green', linestyle=':')[0])
            labels.append("Max Error Thresh")

    if handles: # Only add legend if there are items to legend
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.05, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust layout to make space for legend
    plt.savefig("examples/costas_phase_trajectories.png", dpi=300)
    plt.show(block=False)


if __name__ == "__main__":
    # Test at lower SNRs to verify robustness for Zadoff-Chu
    _zc_cfo_values = [1000.0, 5000.0, 12000.0, 20000.0]
    _zc_snr_values = [0.0, 5.0, 10.0, 20.0]
    _zc_delays = [100, 1000, 4444, 8000]

    _zc_results = _run_sweep(_zc_cfo_values, _zc_snr_values, _zc_delays, n_seeds=10)
    _print_summary(_zc_results)
    _plot_results(_zc_results)

    
    # --- Costas Loop Pipeline Analysis ---
    # Define sweep parameters for Costas pipeline
    _costas_cfo_values = [1000.0, 5000.0]  # Smaller range for illustration
    _costas_snr_values = [5.0, 10.0, 20.0]
    _costas_modulator = BPSK()

    # Run the full pipeline sweep
    _costas_pipeline_results = _run_costas_pipeline(
        _costas_cfo_values, 
        _costas_snr_values, 
        _costas_modulator,
    )

    # Print and plot summaries for Costas loop
    _print_costas_summary(_costas_pipeline_results)
    _plot_costas_results(_costas_pipeline_results)
    
    # Keep the window open for all plots
    #plt.show(block=True)
