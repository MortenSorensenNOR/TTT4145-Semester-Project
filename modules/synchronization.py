"""Synchronization algorithms for digital communication receivers.

1. Coarse CFO estimation
    - Repeated sequence
2. Frame detection and frame timing
    - Zadoff-Chu
3. Symbol timing
    - Gardner TED
4. Fine CFO estimation
    - Costas loop
"""

from dataclasses import dataclass, field

import numpy as np
from sympy import isprime

from modules.pulse_shaping import upsample_and_filter


def generate_zadoff_chu(u: int = 7, n_zc: int = 61) -> np.ndarray:
    """Generate a Zadoff-Chu sequence of length n_zc with root u."""
    if not isprime(n_zc):
        msg = f"n_zc must be prime, got {n_zc}"
        raise ValueError(msg)
    c_f = n_zc % 2
    q = 0
    n = np.arange(0, n_zc, 1)
    return np.exp(-1j * (np.pi * u * n * (n + c_f + 2 * q)) / n_zc)


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    n_short: int = 19
    n_long: int = 139
    zc_root: int = 7
    n_short_reps: int = 8
    peak_threshold: float = 0.5
    peak_margin_factor: int = 2
    long_margin_factor: int = 5


@dataclass
class SynchronizationResult:
    """Result from preamble detection and synchronization."""

    success: bool = True
    reason: str = ""
    d_hat: int = 0
    cfo_hat_hz: float = 0.0
    timing_hat: int = 0
    peak_indices: list[int] = field(default_factory=list)
    n_phase_diffs: int = 0


def build_preamble(config: SynchronizerConfig | None = None) -> np.ndarray:
    """Build a ZC preamble at symbol rate: n_short_reps x ZC_short + ZC_long."""
    if config is None:
        config = SynchronizerConfig()
    zc_short = generate_zadoff_chu(config.zc_root, config.n_short)
    zc_long = generate_zadoff_chu(config.zc_root, config.n_long)
    return np.concatenate([np.tile(zc_short, config.n_short_reps), zc_long])


class Synchronizer:
    """Matched-filter synchronization using Zadoff-Chu preambles.

    Performs coarse CFO estimation from repeated short ZC sequences,
    then fine timing from a long ZC sequence after CFO correction.

    Args:
        config: ZC sequence parameters.
        sps: Samples per symbol (1 = no pulse shaping).
        rrc_taps: RRC filter coefficients (required when sps > 1).

    """

    def __init__(
        self,
        config: SynchronizerConfig,
        sps: int = 1,
        rrc_taps: np.ndarray | None = None,
    ) -> None:
        """Initialize the synchronizer with ZC preamble sequences."""
        self.config = config
        self.sps = sps

        self.zc_short = generate_zadoff_chu(config.zc_root, config.n_short)
        self.zc_long = generate_zadoff_chu(config.zc_root, config.n_long)

        if sps > 1:
            if rrc_taps is None:
                msg = "rrc_taps required when sps > 1"
                raise ValueError(msg)
            self.rrc_taps = rrc_taps
            # Templates shaped to match the received signal after TX RRC + RX matched filter
            # (double RRC = raised cosine)
            tx_short = upsample_and_filter(self.zc_short, sps, rrc_taps)
            tx_long = upsample_and_filter(self.zc_long, sps, rrc_taps)
            self._template_short = np.convolve(tx_short, rrc_taps, mode="same")
            self._template_long = np.convolve(tx_long, rrc_taps, mode="same")
        else:
            self.rrc_taps = None
            self._template_short = self.zc_short
            self._template_long = self.zc_long

        self.preamble = build_preamble(config)

    def detect_preamble(self, rx: np.ndarray, sample_rate: float) -> SynchronizationResult:
        """Detect preamble and estimate CFO and timing.

        The returned timing_hat is always at the sample rate of the input signal
        (upsampled rate when sps > 1).
        """
        sps = self.sps
        n_short_samples = self.config.n_short * sps
        peak_margin = self.config.peak_margin_factor * sps

        corr_short = self._matched_filter(rx, self._template_short)
        corr_mag = np.abs(corr_short)
        global_max_idx = np.argmax(corr_mag)

        # find the peaks of all the repetitions of zc_short
        peak_indices = []
        current_idx = global_max_idx
        while current_idx >= n_short_samples:
            search_start = current_idx - n_short_samples - peak_margin
            search_end = current_idx - n_short_samples + peak_margin + 1
            if search_start < 0:
                break
            prev_region = corr_mag[search_start:search_end]
            if len(prev_region) > 0 and np.max(prev_region) > self.config.peak_threshold * corr_mag[global_max_idx]:
                current_idx = search_start + np.argmax(prev_region)
            else:
                break

        # current_idx should now be at the first zc_short -> find the rest going forward
        first_peak_idx = current_idx
        peak_indices = [first_peak_idx]
        for _i in range(1, self.config.n_short_reps):
            search_start = peak_indices[-1] + n_short_samples - peak_margin
            search_end = min(len(corr_mag), peak_indices[-1] + n_short_samples + peak_margin + 1)
            if search_end <= search_start:
                break
            region = corr_mag[search_start:search_end]
            next_peak = search_start + np.argmax(region)
            peak_indices.append(next_peak)

        min_peaks = 2
        if len(peak_indices) < min_peaks:
            return SynchronizationResult(success=False, reason="Couldn't find enough ZC peaks")

        # CFO estimation
        phase_diffs = []
        for i in range(len(peak_indices) - 1):
            p1 = corr_short[peak_indices[i]]
            p2 = corr_short[peak_indices[i + 1]]
            phase_diffs.append(np.angle(p2 * np.conj(p1)))

        # average the phase differences
        avg_phase_diff = np.mean(phase_diffs)
        cfo_hat = float(avg_phase_diff / (2 * np.pi * n_short_samples) * sample_rate)

        # coarse timing estimate
        d_hat = peak_indices[0]

        # cfo correction
        n_full = np.arange(len(rx))
        cfo_hat_norm = avg_phase_diff / (2 * np.pi * n_short_samples)
        rx_corr = rx * np.exp(-1j * 2 * np.pi * cfo_hat_norm * n_full)

        # fine timing with long ZC
        corr_long = self._matched_filter(rx_corr, self._template_long)
        corr_long_mag = np.abs(corr_long)

        # search for peak in zc_long
        long_margin = self.config.long_margin_factor * sps
        expected_long_start = d_hat + self.config.n_short_reps * n_short_samples
        search_start = max(expected_long_start - long_margin, 0)
        search_end = min(len(corr_long_mag), expected_long_start + 2 * long_margin)

        fine_region = corr_long_mag[search_start:search_end]
        timing_hat = search_start + np.argmax(fine_region)

        return SynchronizationResult(
            success=True,
            d_hat=int(d_hat),
            cfo_hat_hz=cfo_hat,
            timing_hat=int(timing_hat),
            peak_indices=[int(p) for p in peak_indices],
            n_phase_diffs=len(phase_diffs),
        )

    def _matched_filter(self, rx: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Cross-correlation matched filter."""
        n_fft = len(rx)
        signal_padded = np.zeros(n_fft, dtype=complex)
        signal_padded[: len(signal)] = signal

        return np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(signal_padded)))
