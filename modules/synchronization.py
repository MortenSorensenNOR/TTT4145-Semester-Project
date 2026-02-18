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

class ZadoffChu:
    """Zadoff-Chu sequence generator."""
    def generate(self, u: int = 7, N_ZC: int = 61) -> np.ndarray:
        """Generate a sequence of length `N_ZC` of root u."""
        assert isprime(N_ZC)
        c_f = N_ZC % 2
        q = 0
        n = np.arange(0, N_ZC, 1)
        x = np.exp(-1j * (np.pi * u * n * (n + c_f + 2*q)) / N_ZC)
        return x

@dataclass
class SynchronizerConfig:
    N_SHORT: int = 19
    N_LONG: int = 139
    ZC_ROOT: int = 7
    N_SHORT_REPS: int = 8

@dataclass
class SynchronizationResult:
    success: bool = True
    reason: str = ""
    d_hat: int = 0
    cfo_hat_hz: float = 0.0
    timing_hat: int = 0
    peak_indices: list[int] = field(default_factory=list)
    n_phase_diffs: int = 0

class Synchronizer:
    """Placeholder synchronization pipeline."""

    def __init__(self, config: SynchronizerConfig) -> None:
        """Initialize the synchronizer placeholder."""
        self.config = config
        self.N_SHORT = config.N_SHORT
        self.N_LONG = config.N_LONG
        self.ZC_ROOT = config.ZC_ROOT
        self.N_SHORT_REPS = config.N_SHORT_REPS

        self.zc = ZadoffChu()
        self.zc_short = self.zc.generate(self.ZC_ROOT, self.N_SHORT)
        self.zc_long = self.zc.generate(self.ZC_ROOT, self.N_LONG)
        self.preamble = self.build_preamble()

    def build_preamble(self) -> np.ndarray:
        preamble_short = np.tile(self.zc_short, self.N_SHORT_REPS)
        preamble = np.concatenate([preamble_short, self.zc_long])
        return preamble

    def detect_preamble(self, rx: np.ndarray, sample_rate: float) -> SynchronizationResult:
        """Matched filter based preamble detection with multiple short ZC repetitions."""
        corr_short = self._matched_filter(rx, self.zc_short)
        corr_mag = np.abs(corr_short)
        global_max_idx = np.argmax(corr_mag)

        # find the peaks of all the repetitions of zc_short
        peak_indices = []
        current_idx = global_max_idx
        while current_idx >= self.N_SHORT:
            search_start = current_idx - self.N_SHORT - 2
            search_end = current_idx - self.N_SHORT + 3
            if search_start < 0:
                break
            prev_region = corr_mag[search_start:search_end]
            if len(prev_region) > 0 and np.max(prev_region) > 0.5 * corr_mag[global_max_idx]:
                current_idx = search_start + np.argmax(prev_region)
            else:
                break

        # current_idx should now be at the first zc_short -> find the rest going forward
        first_peak_idx = current_idx
        peak_indices = [first_peak_idx]
        for i in range(1, self.N_SHORT_REPS):
            search_start = peak_indices[-1] + self.N_SHORT - 2
            search_end = min(len(corr_mag), peak_indices[-1] + self.N_SHORT + 3)
            if search_end <= search_start:
                break
            region = corr_mag[search_start:search_end]
            next_peak = search_start + np.argmax(region)
            peak_indices.append(next_peak)

        if len(peak_indices) < 2:
            return SynchronizationResult(
                success=False,
                reason = "Couldn't find enough ZC peaks"
            )

        # CFO estimation
        phase_diffs = []
        for i in range(len(peak_indices) - 1):
            p1 = corr_short[peak_indices[i]]
            p2 = corr_short[peak_indices[i+1]]
            phase_diffs.append(np.angle(p2 * np.conj(p1)))

        # average the phase differences
        avg_phase_diff = np.mean(phase_diffs)
        cfo_hat = float(avg_phase_diff / (2 * np.pi * self.N_SHORT) * sample_rate)

        # coarse timing estimate
        d_hat = peak_indices[0]

        # cfo correction
        n_full = np.arange(len(rx))
        cfo_hat_norm = avg_phase_diff / (2 * np.pi * self.N_SHORT)
        rx_corr = rx * np.exp(-1j * 2 * np.pi * cfo_hat_norm * n_full)

        # fine timing with long ZC
        corr_long = self._matched_filter(rx_corr, self.zc_long)
        corr_long_mag = np.abs(corr_long)

        # search for peak in zc_long
        search_start = d_hat + self.N_SHORT_REPS * self.N_SHORT - 5
        search_end = min(len(corr_long_mag), d_hat + self.N_SHORT_REPS * self.N_SHORT + 10)
        if search_start < 0:
            search_start = 0

        fine_region = corr_long_mag[search_start:search_end]
        timing_hat = search_start + np.argmax(fine_region)

        return SynchronizationResult(
            success = True,
            d_hat = d_hat,
            cfo_hat_hz = cfo_hat,
            timing_hat = timing_hat,
            peak_indices = peak_indices,
            n_phase_diffs = len(phase_diffs)
        )

    def _matched_filter(self, rx: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Cross-correlation matched filter."""
        n_fft = len(rx)
        signal_padded = np.zeros(n_fft, dtype=complex)
        signal_padded[:len(signal)] = signal

        corr = np.fft.ifft(np.fft.fft(rx) * np.conj(np.fft.fft(signal_padded)))
        return corr

