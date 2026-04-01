r"""Frame-level synchronization: preamble timing and carrier frequency offset (CFO).

Coarse timing + CFO via Schmidl-Cox autocorrelation [1][2], then fine
timing via matched-filter cross-correlation with the long preamble [4][5].

References:
[1] GNU Radio - Schmidl & Cox OFDM synch. (ofdm_sync_sc_cfb block):
    https://wiki.gnuradio.org/index.php/Schmidl_&_Cox_OFDM_synch. (trailing dot is part of URL)
[2] MATLAB - OFDM Synchronization example (same algorithm):
    https://www.mathworks.com/help/comm/ug/ofdm-synchronization.html
[3] Zadoff-Chu sequence (used in 3GPP LTE):
    https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence
[4] Sklar & Harris, "Digital Communications", 3rd ed., Pearson, 2021,
    Sec. 3.2.2-3.2.3 (matched filter / cross-correlation), Sec. 10.2.2 (peak timing).
[5] PySDR - Synchronization chapter ("Frame Synchronization"):
    https://pysdr.org/content/sync.html

"""

from dataclasses import dataclass
from math import isqrt

import numpy as np


def _is_prime(n: int) -> bool:
    """Trial division for small n."""
    return n > 1 and all(n % i for i in range(2, isqrt(n) + 1))


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    zc_root: int = 7

    short_preamble_nsym: int = 13
    short_preamble_nreps: int = 8

    long_preamble_nsym: int = 139
    long_margin_nsym: int = 5

    energy_floor: float = np.finfo(np.float64).tiny
    detection_threshold: float = 0.5


@dataclass
class CoarseResult:
    """Output of coarse timing + CFO estimation."""

    d_hats: np.ndarray
    cfo_hats: np.ndarray
    m_peaks: np.ndarray


@dataclass
class FineResult:
    """Output of fine timing via cross-correlation."""

    sample_idxs: np.ndarray
    peak_ratios: np.ndarray
    phase_estimates: np.ndarray


def generate_zadoff_chu(u: int, n_zc: int) -> np.ndarray:
    r"""Generate a Zadoff-Chu sequence of length n_zc with root u [3].

    Formula from [3]: $x_u(n) = \exp(-j\pi \cdot u \cdot n \cdot (n+1) / N_{ZC})$ (for $q=1$ and prime $N_{ZC}$)
    Also, $\gcd(u, N_{ZC}) = 1$ for any $0 < u < N_{ZC}$.
    """
    if not _is_prime(n_zc):
        msg = f"n_zc ({n_zc}) must be prime to achieve DFT property"
        raise ValueError(msg)

    if not (0 < u < n_zc):
        msg = f"u ({u}) must be in the range [1, {n_zc - 1}]"
        raise ValueError(msg)

    n = np.arange(n_zc)
    return np.exp(-1j * np.pi * u * n * (n + 1) / n_zc)


def generate_preamble(config: SynchronizerConfig) -> np.ndarray:
    """Build the full preamble: repeated short ZC followed by long ZC."""
    zc_short = generate_zadoff_chu(config.zc_root, config.short_preamble_nsym)
    zc_long = generate_zadoff_chu(config.zc_root, config.long_preamble_nsym)
    short_rep = np.tile(zc_short, config.short_preamble_nreps)
    return np.concatenate([short_rep, zc_long])


def build_long_ref(cfg: SynchronizerConfig, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Build the upsampled + filtered long preamble reference for matched filtering."""
    zc = generate_zadoff_chu(cfg.zc_root, cfg.long_preamble_nsym)
    up = np.zeros(len(zc) * sps, dtype=complex)
    up[::sps] = zc
    return np.convolve(up, rrc_taps, mode="same")


def coarse_sync(
    samples: np.ndarray,
    fs: int,
    samples_per_symbol: int,
    cfg: SynchronizerConfig,
) -> CoarseResult:
    r"""Coarse timing + CFO via the Schmidl-Cox autocorrelation metric [1] [2].

    Exploits the repeated short preamble: when the receiver slides over two
    adjacent copies, $P(d)$ peaks and $M(d)$ approaches 1.

    Formulas (variable names match the code):

        $P(d) = \sum_{m=0}^{L-1} r^*_{d+m} \cdot r_{d+m+L}$    → p_d


        $R(d) = \sum_{m=0}^{L-1} |r_{d+m+L}|^2$        → r_d

        $M(d) = |P(d)|^2 / R(d)^2$       → m_d

    where $L = n_\text{short} \cdot \text{sps}$ → sample_cnt.
    Sliding sums computed via prefix sums (O(N)).

        $\hat{\varphi} = \angle P(\hat{d})$           → phi_hat
        $\Delta f = \hat{\varphi} \cdot f_s / (2\pi \cdot L)$    → cfo_hat_hz

    Acquisition range: $\pm f_s / (2L)$.
    """
    if not np.iscomplexobj(samples):
        msg = "samples must be complex (conj is a no-op on reals)"
        raise TypeError(msg)

    if samples_per_symbol < 1 or fs < 1:
        msg = "samples_per_symbol and fs must be >= 1"
        raise ValueError(msg)

    sample_cnt = cfg.short_preamble_nsym * samples_per_symbol
    if len(samples) < 2 * sample_cnt:
        msg = f"samples too short ({len(samples)} samples): need >= 2L={2 * sample_cnt} for two adjacent windows"
        raise ValueError(msg)

    cs_p = np.concatenate(([0j], np.cumsum(np.conj(samples[:-sample_cnt]) * samples[sample_cnt:])))
    p_d = cs_p[sample_cnt:] - cs_p[:-sample_cnt]

    cs_r = np.concatenate(([0.0], np.cumsum(np.abs(samples[sample_cnt:]) ** 2)))
    r_d = cs_r[sample_cnt:] - cs_r[:-sample_cnt]

    m_d = np.abs(p_d) ** 2 / np.maximum(r_d**2, cfg.energy_floor)

    # Multi-frame detection — batch iterate-and-advance (cf. gr-ieee802-11 sync_short MIN_GAP)
    min_gap = (
        cfg.short_preamble_nsym * samples_per_symbol * cfg.short_preamble_nreps
        + cfg.long_preamble_nsym * samples_per_symbol
    )

    above = np.flatnonzero(m_d > cfg.detection_threshold)
    if above.size == 0:
        return CoarseResult(np.empty(0, np.intp), np.empty(0), np.empty(0))

    # Cluster by gaps > min_gap — each cluster is one frame's plateau
    splits = np.r_[0, np.flatnonzero(np.diff(above) > min_gap) + 1]
    d_hats = above[splits]
    m_peaks = np.maximum.reduceat(m_d[above], splits)
    plateau_sum = np.add.reduceat(p_d[above], splits)
    plateau_cnt = np.diff(np.r_[splits, above.size])
    cfo_hats = np.angle(plateau_sum / plateau_cnt) * fs / (2 * np.pi * sample_cnt)

    return CoarseResult(d_hats, cfo_hats, m_peaks)


def fine_timing(
    samples: np.ndarray,
    long_ref: np.ndarray,
    d_hats: np.ndarray,
    cfo_hats: np.ndarray,
    fs: int,
    samples_per_symbol: int,
    cfg: SynchronizerConfig,
) -> FineResult:
    """Fine timing by cross-correlation with the long preamble [4] [5]."""
    if not np.iscomplexobj(samples):
        msg = "samples must be complex (conj is a no-op on reals)"
        raise TypeError(msg)

    d_hats = np.atleast_1d(np.asarray(d_hats, dtype=np.intp))
    cfo_hats = np.atleast_1d(np.asarray(cfo_hats, dtype=np.float64))

    samples_per_rep = cfg.short_preamble_nsym * samples_per_symbol
    sample_margin = cfg.long_margin_nsym * samples_per_symbol
    window_len = 2 * sample_margin + len(long_ref)

    starts = np.clip(
        d_hats + cfg.short_preamble_nreps * samples_per_rep - sample_margin,
        0,
        len(samples) - window_len,
    )

    # (n_frames, window_len) index array via broadcasting
    indices = starts[:, None] + np.arange(window_len)
    cfo_phase = -2j * np.pi * (cfo_hats[:, None] / fs) * indices
    windows = samples[indices] * np.exp(cfo_phase)

    # Batch cross-correlation via FFT
    valid_len = window_len - len(long_ref) + 1
    pad_len = window_len + len(long_ref) - 1
    ref_f = np.conj(np.fft.fft(long_ref, n=pad_len))
    w_f = np.fft.fft(windows, n=pad_len, axis=1)

    z_complex = np.fft.ifft(w_f * ref_f, axis=1)[:, :valid_len]
    z_mag = np.abs(z_complex)

    peak_idxs = np.argmax(z_mag, axis=1)
    peak_complex = z_complex[np.arange(len(peak_idxs)), peak_idxs]
    z_mean = np.mean(z_mag, axis=1)

    sample_idxs = starts + peak_idxs
    channel_phase = np.angle(peak_complex)

    # Project the phase forward to the payload start so a constant correction
    # removes the CFO-accumulated phase there (Costas loop tracks the rest).
    payload_positions = sample_idxs + len(long_ref)
    phase_at_payload = channel_phase + 2 * np.pi * (cfo_hats / fs) * payload_positions

    return FineResult(
        sample_idxs=sample_idxs,
        peak_ratios=np.max(z_mag, axis=1) / np.where(z_mean == 0, 1, z_mean),
        phase_estimates=phase_at_payload,
    )

