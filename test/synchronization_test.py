"""Tests for synchronization algorithms - frame detection and CFO estimation."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
)
from modules.synchronization import CoarseCFOSequence, ZadofChu
from modules.util import calculate_reference_power


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate():
    return 1e6


@pytest.fixture
def zc_sequence():
    """Generate a length-139 Zadoff-Chu sequence."""
    return ZadofChu().generate(u=7, N_ZC=139)


@pytest.fixture
def cfo_sequence():
    """Generate a coarse CFO estimation sequence."""
    return CoarseCFOSequence(N=16, M=4, modulation='qpsk')


# =============================================================================
# Helper Functions
# =============================================================================


def cyclic_correlate(rx_signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Compute cyclic correlation using FFT.

    Args:
        rx_signal: Received signal
        template: Template sequence to correlate against

    Returns:
        Cyclic correlation result (same length as rx_signal)
    """
    L = len(rx_signal)
    RX = np.fft.fft(rx_signal)
    TEMPLATE = np.fft.fft(template, L)
    return np.fft.ifft(RX * np.conj(TEMPLATE))


def detect_frame(rx_signal: np.ndarray, template: np.ndarray) -> tuple[int, np.ndarray]:
    """Detect frame start using cyclic correlation.

    Args:
        rx_signal: Received signal
        template: Template sequence (e.g., ZC sequence)

    Returns:
        Tuple of (detected_delay, correlation_output)
    """
    corr = cyclic_correlate(rx_signal, template)
    detected_delay = np.argmax(np.abs(corr))
    return int(detected_delay), corr


def run_detection_trial(
    zc_seq: np.ndarray,
    snr_db: float,
    true_delay: int,
    seed: int,
    sample_rate: float = 1e6,
    padding: int = 10000,
    cfo_hz: float = 0.0,
) -> tuple[int, bool]:
    """Run a single frame detection trial.

    Args:
        zc_seq: Zadoff-Chu sequence
        snr_db: SNR in dB
        true_delay: True delay in samples
        seed: Random seed for noise
        sample_rate: Sample rate in Hz
        padding: Zero padding after sequence
        cfo_hz: Carrier frequency offset in Hz

    Returns:
        Tuple of (detected_delay, is_correct)
    """
    tx_signal = np.concatenate([zc_seq, np.zeros(padding, dtype=complex)])
    ref_power = calculate_reference_power(zc_seq)

    overrides = ProfileOverrides(delay_samples=float(true_delay))
    if cfo_hz != 0.0:
        overrides = ProfileOverrides(delay_samples=float(true_delay), cfo_hz=cfo_hz)

    channel_conf = ProfileRequest(
        sample_rate=sample_rate,
        snr_db=snr_db,
        seed=seed,
        reference_power=ref_power,
        overrides=overrides,
    )
    channel = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=channel_conf)

    rx_signal = channel.apply(tx_signal)
    detected_delay, _ = detect_frame(rx_signal, zc_seq)

    # Allow 1 sample tolerance for edge cases
    is_correct = abs(detected_delay - true_delay) <= 1

    return detected_delay, is_correct


def measure_detection_probability(
    zc_seq: np.ndarray,
    snr_db: float,
    true_delay: int,
    n_trials: int = 100,
    sample_rate: float = 1e6,
    cfo_hz: float = 0.0,
) -> float:
    """Measure detection probability over multiple noise realizations.

    Args:
        zc_seq: Zadoff-Chu sequence
        snr_db: SNR in dB
        true_delay: True delay in samples
        n_trials: Number of trials to run
        sample_rate: Sample rate in Hz
        cfo_hz: Carrier frequency offset in Hz

    Returns:
        Detection probability (0.0 to 1.0)
    """
    n_correct = 0
    for seed in range(n_trials):
        _, is_correct = run_detection_trial(
            zc_seq, snr_db, true_delay, seed, sample_rate, cfo_hz=cfo_hz
        )
        if is_correct:
            n_correct += 1

    return n_correct / n_trials


# =============================================================================
# Frame Detection Tests
# =============================================================================


class TestFrameDetection:
    """Tests for ZC-based frame detection."""

    def test_detection_no_noise(self, zc_sequence, sample_rate):
        """Frame detection should work perfectly without noise."""
        true_delay = 500
        detected, is_correct = run_detection_trial(
            zc_sequence, snr_db=30.0, true_delay=true_delay, seed=0, sample_rate=sample_rate
        )
        assert is_correct, f"Expected {true_delay}, got {detected}"

    @pytest.mark.parametrize("true_delay", [0, 100, 4444, 9000])
    def test_detection_various_delays(self, zc_sequence, sample_rate, true_delay):
        """Frame detection should work at various delays."""
        detected, is_correct = run_detection_trial(
            zc_sequence, snr_db=20.0, true_delay=true_delay, seed=42, sample_rate=sample_rate
        )
        assert is_correct, f"Expected {true_delay}, got {detected}"

    @pytest.mark.parametrize("snr_db", [10.0, 5.0, 0.0, -5.0])
    def test_detection_various_snr(self, zc_sequence, sample_rate, snr_db):
        """Frame detection should work at moderate SNR values."""
        true_delay = 4444
        detected, is_correct = run_detection_trial(
            zc_sequence, snr_db=snr_db, true_delay=true_delay, seed=42, sample_rate=sample_rate
        )
        assert is_correct, f"Failed at {snr_db} dB: expected {true_delay}, got {detected}"

    def test_detection_at_threshold(self, zc_sequence, sample_rate):
        """Test detection near the theoretical limit (-10 to -12 dB)."""
        true_delay = 4444
        snr_db = -10.0
        n_trials = 50

        prob = measure_detection_probability(
            zc_sequence, snr_db, true_delay, n_trials, sample_rate
        )

        # At -10 dB with length-139 ZC, expect >50% detection
        assert prob > 0.5, f"Detection probability {prob:.1%} too low at {snr_db} dB"

    @pytest.mark.parametrize("seed", range(10))
    def test_detection_seed_variability(self, zc_sequence, sample_rate, seed):
        """Show detection variability across seeds at challenging SNR."""
        true_delay = 4444
        snr_db = -8.0

        detected, is_correct = run_detection_trial(
            zc_sequence, snr_db=snr_db, true_delay=true_delay, seed=seed, sample_rate=sample_rate
        )

        # Log result but don't fail - this shows variability
        status = "PASS" if is_correct else f"FAIL (got {detected})"
        print(f"Seed {seed}: {status}")


class TestFrameDetectionWithCFO:
    """Tests for frame detection with carrier frequency offset."""

    @pytest.fixture
    def zc_sequence_61(self):
        """Generate a length-61 Zadoff-Chu sequence."""
        return ZadofChu().generate(u=7, N_ZC=61)

    def test_detection_small_cfo(self, zc_sequence_61, sample_rate):
        """Frame detection should work with small CFO."""
        true_delay = 4444
        cfo_hz = 1000.0  # 1 kHz CFO

        detected, is_correct = run_detection_trial(
            zc_sequence_61, snr_db=10.0, true_delay=true_delay,
            seed=42, sample_rate=sample_rate, cfo_hz=cfo_hz
        )
        assert is_correct, f"Failed with {cfo_hz} Hz CFO: expected {true_delay}, got {detected}"

    @pytest.mark.parametrize("cfo_hz", [0.0, 1000.0, 2000.0, 4000.0])
    def test_detection_various_cfo(self, zc_sequence_61, sample_rate, cfo_hz):
        """Frame detection should work at moderate CFO values."""
        true_delay = 4444

        detected, is_correct = run_detection_trial(
            zc_sequence_61, snr_db=10.0, true_delay=true_delay,
            seed=42, sample_rate=sample_rate, cfo_hz=cfo_hz
        )
        assert is_correct, f"Failed with {cfo_hz} Hz CFO: expected {true_delay}, got {detected}"

    def test_detection_degrades_with_large_cfo(self, zc_sequence_61, sample_rate):
        """Detection probability should degrade with large CFO.

        For length-61 ZC at 1 MHz sample rate:
        - Phase rotation = 360 * cfo * N / fs degrees
        - At 4 kHz: 360 * 4000 * 61 / 1e6 = 87.8 degrees (near limit)
        - At 8 kHz: 360 * 8000 * 61 / 1e6 = 175.7 degrees (beyond limit)
        """
        true_delay = 4444
        n_trials = 30

        # Should work well with small CFO
        prob_small_cfo = measure_detection_probability(
            zc_sequence_61, snr_db=5.0, true_delay=true_delay,
            n_trials=n_trials, sample_rate=sample_rate, cfo_hz=1000.0
        )

        # Should degrade with large CFO
        prob_large_cfo = measure_detection_probability(
            zc_sequence_61, snr_db=5.0, true_delay=true_delay,
            n_trials=n_trials, sample_rate=sample_rate, cfo_hz=8000.0
        )

        print(f"Small CFO (1 kHz): {prob_small_cfo:.1%}")
        print(f"Large CFO (8 kHz): {prob_large_cfo:.1%}")

        # Large CFO should have noticeably worse performance
        assert prob_small_cfo > prob_large_cfo, \
            f"Expected degradation with large CFO, got {prob_small_cfo:.1%} vs {prob_large_cfo:.1%}"


# =============================================================================
# Coarse CFO Estimation Tests
# =============================================================================


class TestCoarseCFOEstimation:
    """Tests for coarse CFO estimation."""

    def test_cfo_estimation_no_noise(self, cfo_sequence, sample_rate):
        """CFO estimation should work perfectly without noise."""
        true_cfo_hz = -5000.0
        Ts = 1 / sample_rate

        preamble = cfo_sequence.preamble
        tx_signal = np.concatenate([preamble, np.zeros(100, dtype=complex)])
        ref_power = calculate_reference_power(preamble)

        channel_conf = ProfileRequest(
            sample_rate=sample_rate,
            snr_db=30.0,
            seed=42,
            reference_power=ref_power,
            overrides=ProfileOverrides(cfo_hz=true_cfo_hz, delay_samples=100.0),
        )
        channel = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=channel_conf)

        rx_signal = channel.apply(tx_signal)
        rx_preamble = rx_signal[100:100 + cfo_sequence.N * cfo_sequence.M]
        estimated_cfo = cfo_sequence.estimate_corase_cfo(rx_preamble, Ts)

        error = abs(estimated_cfo - true_cfo_hz)
        assert error < 100, f"CFO error {error:.1f} Hz too large"

    @pytest.mark.parametrize("true_cfo_hz", [-10000.0, -5000.0, 0.0, 5000.0, 10000.0])
    def test_cfo_estimation_various_offsets(self, cfo_sequence, sample_rate, true_cfo_hz):
        """CFO estimation should work for various frequency offsets."""
        Ts = 1 / sample_rate

        preamble = cfo_sequence.preamble
        tx_signal = np.concatenate([preamble, np.zeros(100, dtype=complex)])
        ref_power = calculate_reference_power(preamble)

        channel_conf = ProfileRequest(
            sample_rate=sample_rate,
            snr_db=20.0,
            seed=42,
            reference_power=ref_power,
            overrides=ProfileOverrides(cfo_hz=true_cfo_hz, delay_samples=100.0),
        )
        channel = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=channel_conf)

        rx_signal = channel.apply(tx_signal)
        rx_preamble = rx_signal[100:100 + cfo_sequence.N * cfo_sequence.M]
        estimated_cfo = cfo_sequence.estimate_corase_cfo(rx_preamble, Ts)

        error = abs(estimated_cfo - true_cfo_hz)
        assert error < 200, f"CFO error {error:.1f} Hz too large for {true_cfo_hz} Hz offset"


# =============================================================================
# Detection Probability Analysis (run with pytest -s or as script)
# =============================================================================


def plot_detection_probability_vs_snr(
    zc_length: int = 139,
    snr_range: tuple[float, float] = (-15.0, 5.0),
    snr_step: float = 1.0,
    n_trials: int = 100,
    true_delay: int = 4444,
    save_path: str | None = None,
):
    """Generate detection probability vs SNR curve.

    Args:
        zc_length: Length of ZC sequence (must be prime)
        snr_range: (min_snr, max_snr) in dB
        snr_step: SNR step size in dB
        n_trials: Number of trials per SNR point
        true_delay: True delay in samples
        save_path: Path to save figure (optional)
    """
    zc_seq = ZadofChu().generate(u=7, N_ZC=zc_length)

    snr_values = np.arange(snr_range[0], snr_range[1] + snr_step, snr_step)
    probabilities = []

    print(f"ZC sequence length: {zc_length}")
    print(f"Theoretical processing gain: {10 * np.log10(zc_length):.1f} dB")
    print(f"Running {n_trials} trials per SNR point...")
    print()

    for snr_db in snr_values:
        prob = measure_detection_probability(zc_seq, float(snr_db), true_delay, n_trials)
        probabilities.append(prob)
        print(f"SNR = {snr_db:+5.1f} dB: {prob:6.1%} detection rate")

    # Find -3dB point (50% detection)
    probabilities = np.array(probabilities)
    idx_50 = np.argmin(np.abs(probabilities - 0.5))
    snr_50 = snr_values[idx_50]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snr_values, probabilities * 100, 'b-o', linewidth=2, markersize=6)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50% threshold')
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.7, label='90% threshold')
    ax.axvline(x=snr_50, color='r', linestyle=':', alpha=0.5)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Detection Probability (%)', fontsize=12)
    ax.set_title(
        f'Frame Detection Probability vs SNR\n'
        f'ZC length={zc_length}, Processing gain={10*np.log10(zc_length):.1f} dB, '
        f'{n_trials} trials/point',
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(-5, 105)

    # Add annotation for 50% point
    ax.annotate(
        f'50% @ {snr_50:.1f} dB',
        xy=(snr_50, 50),
        xytext=(snr_50 + 3, 40),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved figure to {save_path}")

    return fig, snr_values, probabilities


def plot_detection_probability_vs_zc_length(
    zc_lengths: list[int] | None = None,
    snr_range: tuple[float, float] = (-20.0, 5.0),
    snr_step: float = 2.0,
    n_trials: int = 50,
    true_delay: int = 4444,
    save_path: str | None = None,
):
    """Compare detection probability curves for different ZC lengths.

    Args:
        zc_lengths: List of ZC sequence lengths (must be prime)
        snr_range: (min_snr, max_snr) in dB
        snr_step: SNR step size in dB
        n_trials: Number of trials per SNR point
        true_delay: True delay in samples
        save_path: Path to save figure (optional)
    """
    if zc_lengths is None:
        zc_lengths = [61, 139, 251]

    snr_values = np.arange(snr_range[0], snr_range[1] + snr_step, snr_step)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, zc_length in enumerate(zc_lengths):
        zc_seq = ZadofChu().generate(u=7, N_ZC=zc_length)
        probabilities = []

        processing_gain = 10 * np.log10(zc_length)
        print(f"\nZC length {zc_length} (processing gain: {processing_gain:.1f} dB)")

        for snr_db in snr_values:
            prob = measure_detection_probability(zc_seq, float(snr_db), true_delay, n_trials)
            probabilities.append(prob)
            print(f"  SNR = {snr_db:+5.1f} dB: {prob:6.1%}")

        label = f'N={zc_length} (PG={processing_gain:.1f} dB)'
        ax.plot(snr_values, np.array(probabilities) * 100,
                f'-o', color=colors[i % len(colors)], linewidth=2, markersize=5, label=label)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Detection Probability (%)', fontsize=12)
    ax.set_title(f'Frame Detection: ZC Sequence Length Comparison\n{n_trials} trials/point', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(-5, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved figure to {save_path}")

    return fig


def plot_detection_probability_vs_cfo(
    zc_length: int = 61,
    cfo_values_hz: list[float] | None = None,
    snr_range: tuple[float, float] = (-15.0, 10.0),
    snr_step: float = 2.0,
    n_trials: int = 50,
    true_delay: int = 4444,
    sample_rate: float = 1e6,
    save_path: str | None = None,
):
    """Compare detection probability curves for different CFO values.

    Args:
        zc_length: Length of ZC sequence (must be prime)
        cfo_values_hz: List of CFO values in Hz to test
        snr_range: (min_snr, max_snr) in dB
        snr_step: SNR step size in dB
        n_trials: Number of trials per SNR point
        true_delay: True delay in samples
        sample_rate: Sample rate in Hz
        save_path: Path to save figure (optional)
    """
    if cfo_values_hz is None:
        cfo_values_hz = [0.0, 1000.0, 5000.0, 10000.0, 20000.0]

    zc_seq = ZadofChu().generate(u=7, N_ZC=zc_length)
    snr_values = np.arange(snr_range[0], snr_range[1] + snr_step, snr_step)

    # Calculate the CFO tolerance limit
    # Phase rotation over N samples: phi = 2*pi*cfo*N/fs
    # Correlation degrades significantly when phi > pi/2 (90 degrees)
    # So rough limit: cfo_max ~ fs / (4*N)
    cfo_limit = sample_rate / (4 * zc_length)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

    print(f"ZC sequence length: {zc_length}")
    print(f"Sample rate: {sample_rate/1e6:.1f} MHz")
    print(f"Theoretical CFO limit (90° rotation): {cfo_limit:.0f} Hz")
    print(f"Running {n_trials} trials per point...")

    for i, cfo_hz in enumerate(cfo_values_hz):
        probabilities = []
        phase_rotation_deg = 360 * cfo_hz * zc_length / sample_rate

        print(f"\nCFO = {cfo_hz:.0f} Hz (phase rotation over sequence: {phase_rotation_deg:.1f}°)")

        for snr_db in snr_values:
            prob = measure_detection_probability(
                zc_seq, float(snr_db), true_delay, n_trials, sample_rate, cfo_hz
            )
            probabilities.append(prob)
            print(f"  SNR = {snr_db:+5.1f} dB: {prob:6.1%}")

        label = f'CFO={cfo_hz/1000:.1f} kHz ({phase_rotation_deg:.0f}°)'
        ax.plot(snr_values, np.array(probabilities) * 100,
                '-o', color=colors[i % len(colors)], linewidth=2, markersize=5, label=label)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Detection Probability (%)', fontsize=12)
    ax.set_title(
        f'Frame Detection with CFO (ZC length={zc_length})\n'
        f'Theoretical CFO limit: {cfo_limit/1000:.1f} kHz (90° rotation)',
        fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim(-5, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved figure to {save_path}")

    return fig


# =============================================================================
# Main (for running analysis directly)
# =============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Frame Detection Analysis")
    print("=" * 60)

    # Single ZC length analysis
    fig1, snr_vals, probs = plot_detection_probability_vs_snr(
        zc_length=139,
        snr_range=(-15.0, 5.0),
        snr_step=1.0,
        n_trials=100,
    )

    print("\n" + "=" * 60)
    print("ZC Length Comparison")
    print("=" * 60)

    # Compare different ZC lengths
    fig2 = plot_detection_probability_vs_zc_length(
        zc_lengths=[61, 139, 251],
        snr_range=(-20.0, 5.0),
        snr_step=2.0,
        n_trials=50,
    )

    print("\n" + "=" * 60)
    print("CFO Impact Analysis (ZC length=61)")
    print("=" * 60)

    # CFO impact analysis with length-61 ZC sequence
    fig3 = plot_detection_probability_vs_cfo(
        zc_length=61,
        cfo_values_hz=[0.0, 1000.0, 2000.0, 4000.0, 8000.0],
        snr_range=(-10.0, 10.0),
        snr_step=2.0,
        n_trials=50,
    )

    plt.show()
