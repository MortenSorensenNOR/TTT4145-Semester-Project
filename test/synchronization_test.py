"""Tests for synchronization algorithms - frame detection and CFO estimation.

Uses matched-filter (cross-correlation) based detection for robust performance
at low SNR. This is the standard approach used in WiFi, LTE, and 5G systems.
"""
import numpy as np
import matplotlib.pyplot as plt
from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
)
from modules.synchronization import ZadofChu

# Fixed system parameters
SAMPLE_RATE = 1e6
N_SHORT = 19
N_LONG = 139
ZC_ROOT = 7
N_SHORT_REPS = 8  # Number of short ZC repetitions for CFO averaging
DELAY_PADDING = 10000

zc = ZadofChu()
zc_short = zc.generate(ZC_ROOT, N_SHORT)
zc_long = zc.generate(ZC_ROOT, N_LONG)


def matched_filter(rx: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Cross-correlation based matched filter. Returns correlation magnitude."""
    # Use scipy-style 'valid' correlation via FFT for efficiency
    n_fft = len(rx)
    template_padded = np.zeros(n_fft, dtype=complex)
    template_padded[:len(template)] = template

    corr = np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(template_padded)))
    return corr


def detect_preamble(rx: np.ndarray, zc_short: np.ndarray, zc_long: np.ndarray,
                    n_short: int, n_short_reps: int, sample_rate: float) -> dict:
    """
    Matched-filter based preamble detection with multiple short ZC repetitions.

    1. Cross-correlate with zc_short to find all repetition peaks
    2. Estimate CFO from phase differences between adjacent peaks (averaged)
    3. Correct CFO
    4. Cross-correlate with zc_long for fine timing

    Returns dict with timing and CFO estimates.
    """
    # Stage 1: Matched filter with short ZC
    corr_short = matched_filter(rx, zc_short)
    corr_mag = np.abs(corr_short)

    # Find the global maximum - this is robust even at low SNR
    global_max_idx = np.argmax(corr_mag)

    # Find which repetition the global max belongs to, then locate all peaks
    # Work backwards to find the first peak
    peak_indices = []

    # Start from global max and search backwards for earlier peaks
    current_idx = global_max_idx
    while current_idx >= n_short:
        search_start = current_idx - n_short - 2
        search_end = current_idx - n_short + 3
        if search_start < 0:
            break
        prev_region = corr_mag[search_start:search_end]
        if len(prev_region) > 0 and np.max(prev_region) > 0.5 * corr_mag[global_max_idx]:
            current_idx = search_start + np.argmax(prev_region)
        else:
            break

    # current_idx is now at (or near) the first peak
    first_peak_idx = current_idx

    # Now find all n_short_reps peaks going forward
    peak_indices = [first_peak_idx]
    for i in range(1, n_short_reps):
        search_start = peak_indices[-1] + n_short - 2
        search_end = min(len(corr_mag), peak_indices[-1] + n_short + 3)
        if search_end <= search_start:
            break
        region = corr_mag[search_start:search_end]
        next_peak = search_start + np.argmax(region)
        peak_indices.append(next_peak)

    if len(peak_indices) < 2:
        return {"success": False, "reason": "couldn't find enough ZC peaks"}

    # CFO estimation: average phase differences between adjacent peaks
    phase_diffs = []
    for i in range(len(peak_indices) - 1):
        p1 = corr_short[peak_indices[i]]
        p2 = corr_short[peak_indices[i + 1]]
        phase_diffs.append(np.angle(p2 * np.conj(p1)))

    # Average the phase differences
    avg_phase_diff = np.mean(phase_diffs)
    cfo_hat_hz = avg_phase_diff / (2 * np.pi * n_short) * sample_rate

    # Coarse timing estimate (first peak)
    d_hat = peak_indices[0]

    # Stage 2: CFO correction
    n_full = np.arange(len(rx))
    cfo_hat_norm = avg_phase_diff / (2 * np.pi * n_short)
    rx_corr = rx * np.exp(-1j * 2 * np.pi * cfo_hat_norm * n_full)

    # Stage 3: Fine timing with long ZC matched filter
    corr_long = matched_filter(rx_corr, zc_long)
    corr_long_mag = np.abs(corr_long)

    # Search for long ZC peak starting after all short ZCs
    search_start = d_hat + n_short_reps * n_short - 5
    search_end = min(len(corr_long_mag), d_hat + n_short_reps * n_short + 10)

    if search_start < 0:
        search_start = 0

    fine_region = corr_long_mag[search_start:search_end]
    timing_hat = search_start + np.argmax(fine_region)

    return {
        "success": True,
        "d_hat": d_hat,
        "cfo_hat_hz": cfo_hat_hz,
        "timing_hat": timing_hat,
        "peak_indices": peak_indices,
        "n_phase_diffs": len(phase_diffs),
    }


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

    # Build preamble: N_SHORT_REPS copies of zc_short, then zc_long
    preamble_short = np.tile(zc_short, N_SHORT_REPS)
    tx = np.concatenate([preamble_short, zc_long, np.zeros(DELAY_PADDING, dtype=complex)])
    rx = channel.apply(tx)

    # Run matched-filter based detection
    result = detect_preamble(rx, zc_short, zc_long, N_SHORT, N_SHORT_REPS, SAMPLE_RATE)

    if not result["success"]:
        return {"success": False, "reason": result["reason"]}

    d_hat = result["d_hat"]
    cfo_hat_hz = result["cfo_hat_hz"]
    timing_hat = result["timing_hat"]

    true_zc_long_start = actual_delay + N_SHORT_REPS * N_SHORT

    return {
        "success": True,
        "delay_true": actual_delay,
        "delay_hat": d_hat,
        "delay_error": d_hat - actual_delay,
        "cfo_true_hz": actual_cfo,
        "cfo_hat_hz": cfo_hat_hz,
        "cfo_error_hz": cfo_hat_hz - actual_cfo,
        "timing_true": true_zc_long_start,
        "timing_hat": timing_hat,
        "timing_error": timing_hat - true_zc_long_start,
        "snr": snr,
        "seed": seed,
    }


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
    plt.savefig("sync_test_results.png", dpi=150)
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
