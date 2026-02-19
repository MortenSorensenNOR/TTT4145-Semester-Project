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

logger = logging.getLogger(__name__)

# Fixed system parameters
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
# Sweep helpers (used by __main__ for detailed analysis)
# =============================================================================


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
    """Print a summary of sweep results broken down by SNR."""
    successful = [r for r in results if r["success"]]
    if not successful:
        logger.info("No successful trials.")
        return

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


if __name__ == "__main__":
    # Test at lower SNRs to verify robustness
    _cfo_values = [1000.0, 5000.0, 12000.0, 20000.0]
    _snr_values = [0.0, 5.0, 10.0, 20.0]
    _delays = [100, 1000, 4444, 8000]

    _results = _run_sweep(_cfo_values, _snr_values, _delays, n_seeds=10)
    _print_summary(_results)
    _plot_results(_results)
