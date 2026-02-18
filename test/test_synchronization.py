"""Tests for synchronization algorithms - frame detection and CFO estimation.

Uses matched-filter (cross-correlation) based detection for robust performance
at low SNR. This is the standard approach used in WiFi, LTE, and 5G systems.
"""
import numpy as np
import pytest
import matplotlib.pyplot as plt
from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
)
from modules.synchronization import Synchronizer, SynchronizerConfig, ZadoffChu

# Fixed system parameters
SAMPLE_RATE = 1e6
DELAY_PADDING = 10000

config = SynchronizerConfig()
sync = Synchronizer(config)


def run_sync(actual_delay: float, actual_cfo: float,
             snr: float, seed: int) -> dict:
    """Run one full synchronization trial. Returns a results dict."""
    request = ProfileRequest(
        sample_rate=SAMPLE_RATE,
        snr_db=snr,
        seed=seed,
        overrides=ProfileOverrides(
            cfo_hz=actual_cfo,
            phase_offset_rad=np.random.default_rng(seed).uniform(0, 2 * np.pi),
            delay_samples=actual_delay,
        ),
    )
    channel = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=request)

    tx = np.concatenate([sync.preamble, np.zeros(DELAY_PADDING, dtype=complex)])
    rx = channel.apply(tx)

    result = sync.detect_preamble(rx, SAMPLE_RATE)

    if not result.success:
        return {"success": False, "reason": result.reason}

    true_zc_long_start = actual_delay + config.N_SHORT_REPS * config.N_SHORT

    return {
        "success": True,
        "delay_true": actual_delay,
        "delay_hat": result.d_hat,
        "delay_error": result.d_hat - actual_delay,
        "cfo_true_hz": actual_cfo,
        "cfo_hat_hz": result.cfo_hat_hz,
        "cfo_error_hz": result.cfo_hat_hz - actual_cfo,
        "timing_true": true_zc_long_start,
        "timing_hat": result.timing_hat,
        "timing_error": result.timing_hat - true_zc_long_start,
        "snr": snr,
        "seed": seed,
    }


# =============================================================================
# Pytest-discoverable tests
# =============================================================================


class TestZadoffChu:
    """Tests for Zadoff-Chu sequence generation."""

    def test_sequence_length(self):
        """Generated sequence should have the requested length."""
        zc = ZadoffChu()
        seq = zc.generate(u=7, N_ZC=61)
        assert len(seq) == 61

    def test_constant_amplitude(self):
        """Zadoff-Chu sequences have constant amplitude."""
        zc = ZadoffChu()
        seq = zc.generate(u=7, N_ZC=61)
        np.testing.assert_allclose(np.abs(seq), 1.0, atol=1e-10)

    def test_different_roots_are_different(self):
        """Different root indices should produce different sequences."""
        zc = ZadoffChu()
        seq1 = zc.generate(u=7, N_ZC=61)
        seq2 = zc.generate(u=11, N_ZC=61)
        assert not np.allclose(seq1, seq2)


class TestSynchronizer:
    """Tests for preamble detection and CFO estimation."""

    def test_preamble_length(self):
        """Preamble should be N_SHORT_REPS * N_SHORT + N_LONG samples."""
        expected = config.N_SHORT_REPS * config.N_SHORT + config.N_LONG
        assert len(sync.preamble) == expected

    def test_detection_no_impairments(self):
        """Detection should succeed with no channel impairments."""
        result = run_sync(actual_delay=0, actual_cfo=0, snr=30.0, seed=42)
        assert result["success"]

    def test_detection_with_delay(self):
        """Detection should succeed and estimate delay correctly."""
        result = run_sync(actual_delay=500, actual_cfo=0, snr=30.0, seed=42)
        assert result["success"]
        assert abs(result["delay_error"]) <= 2

    def test_cfo_estimation_high_snr(self):
        """CFO estimate should be accurate at high SNR."""
        result = run_sync(actual_delay=100, actual_cfo=5000.0, snr=20.0, seed=42)
        assert result["success"]
        assert abs(result["cfo_error_hz"]) < 200

    @pytest.mark.parametrize("snr", [5.0, 10.0, 20.0])
    def test_detection_at_various_snr(self, snr):
        """Detection should succeed at moderate-to-high SNR."""
        result = run_sync(actual_delay=1000, actual_cfo=1000.0, snr=snr, seed=123)
        assert result["success"]

    def test_fine_timing_accuracy(self):
        """Fine timing (long ZC) should be within a few samples."""
        result = run_sync(actual_delay=200, actual_cfo=0, snr=20.0, seed=42)
        assert result["success"]
        assert abs(result["timing_error"]) <= 3


# =============================================================================
# Sweep helpers (used by __main__ for detailed analysis)
# =============================================================================


def run_sweep(cfo_values, snr_values, delays, n_seeds=10) -> list[dict]:
    results = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**31, size=n_seeds).tolist()

    total = len(cfo_values) * len(snr_values) * len(delays) * n_seeds
    done = 0
    for cfo in cfo_values:
        for snr in snr_values:
            for delay in delays:
                for seed in seeds:
                    r = run_sync(delay, cfo, snr, seed)
                    r.update({"cfo_sweep": cfo, "snr_sweep": snr, "delay_sweep": delay})
                    results.append(r)
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{total} trials done")
    return results


def print_summary(results: list[dict]):
    successful = [r for r in results if r["success"]]
    print(f"\n{'='*60}")
    print(f"Total trials : {len(results)}")
    print(f"Successful   : {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    if not successful:
        return

    cfo_errors = np.array([r["cfo_error_hz"] for r in successful])
    delay_errors = np.array([r["delay_error"] for r in successful])
    timing_errors = np.array([r["timing_error"] for r in successful])

    print(f"\nCFO error (Hz)     mean={np.mean(cfo_errors):+.2f}  std={np.std(cfo_errors):.2f}  max|e|={np.max(np.abs(cfo_errors)):.2f}")
    print(f"Delay error (samp) mean={np.mean(delay_errors):+.2f}  std={np.std(delay_errors):.2f}  max|e|={np.max(np.abs(delay_errors)):.2f}")
    print(f"Timing error (samp) mean={np.mean(timing_errors):+.2f}  std={np.std(timing_errors):.2f}  max|e|={np.max(np.abs(timing_errors)):.2f}")

    # Break down by SNR
    print(f"\n{'SNR':>6} | {'CFO err std':>12} | {'Timing err std':>14} | {'Success%':>8}")
    print("-" * 50)
    for snr in sorted(set(r["snr_sweep"] for r in results)):
        sub = [r for r in successful if r["snr_sweep"] == snr]
        if not sub:
            continue
        ce = np.std([r["cfo_error_hz"] for r in sub])
        te = np.std([r["timing_error"] for r in sub])
        tot = len([r for r in results if r["snr_sweep"] == snr])
        print(f"{snr:>6.1f} | {ce:>12.2f} | {te:>14.4f} | {100*len(sub)/tot:>7.1f}%")


def plot_results(results: list[dict]):
    successful = [r for r in results if r["success"]]
    snrs = sorted(set(r["snr_sweep"] for r in results))
    cfos = sorted(set(r["cfo_sweep"] for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # CFO error vs SNR
    ax = axes[0]
    cfo_err_by_snr = [[r["cfo_error_hz"] for r in successful if r["snr_sweep"] == s] for s in snrs]
    ax.boxplot(cfo_err_by_snr, tick_labels=[str(s) for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("CFO error (Hz)")
    ax.set_title("CFO estimation error vs SNR")
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    # Timing error vs SNR (long ZC detection)
    ax = axes[1]
    tim_err_by_snr = [[r["timing_error"] for r in successful if r["snr_sweep"] == s] for s in snrs]
    ax.boxplot(tim_err_by_snr, tick_labels=[str(s) for s in snrs])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Timing error (samples)")
    ax.set_title("Long ZC timing error vs SNR")
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    # CFO error vs actual CFO
    ax = axes[2]
    cfo_err_by_cfo = [[r["cfo_error_hz"] for r in successful if r["cfo_sweep"] == c] for c in cfos]
    ax.boxplot(cfo_err_by_cfo, tick_labels=[f"{c/1e3:.0f}k" for c in cfos])
    ax.set_xlabel("Actual CFO (kHz)")
    ax.set_ylabel("CFO error (Hz)")
    ax.set_title("CFO estimation error vs CFO magnitude")
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("examples/sync_test_results.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # Test at lower SNRs to verify robustness
    cfo_values = [1000.0, 5000.0, 12000.0, 20000.0]
    snr_values = [0.0, 5.0, 10.0, 20.0]
    delays = [100, 1000, 4444, 8000]

    print("Running synchronization sweep (matched filter approach)...")
    results = run_sweep(cfo_values, snr_values, delays, n_seeds=10)
    print_summary(results)
    plot_results(results)
