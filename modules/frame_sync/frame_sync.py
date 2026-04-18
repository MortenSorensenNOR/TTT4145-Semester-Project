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

import logging
from dataclasses import dataclass
from math import isqrt

import numpy as np

logger = logging.getLogger(__name__)

try:
    from modules.frame_sync import frame_sync_ext as _ext
    logger.info("Loaded frame_sync_ext pybind11 C++ extension.")
except ImportError:
    _ext = None
    logger.warning(
        "frame_sync_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python setup.py build_ext --inplace"
    )


def _is_prime(n: int) -> bool:
    """Trial division for small n."""
    return n > 1 and all(n % i for i in range(2, isqrt(n) + 1))


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    zc_root_short: int = 7
    zc_root_long: int = 13

    short_preamble_nsym: int = 43   # prime; ≥43 for L > RRC filter length at full rate
    short_preamble_nreps: int = 4   # 4 reps for Minn/Park [+,+,-,-] sign pattern

    long_preamble_nsym: int = 89    # prime; longer ZC → higher fine-timing peak-to-mean
    long_margin_nsym: int = 20     # ≥ GUARD_SYMS_LENGTH/2 + RRC_SPAN to cover early d_hat

    energy_floor: np.float32 = np.finfo(np.float32).tiny
    detection_threshold: np.float32 = np.float32(0.5)
    # Minimum r_d as a fraction of the peak r_d in the buffer.
    # Filters low-energy regions (noise floor) where M(d)≈1 spuriously.
    # Set to 0.0 to disable (default for backward compat with simulation).
    energy_gate_fraction: np.float32 = np.float32(0.05)
    # Minimum fine-timing cross-correlation peak-to-mean ratio required to
    # accept a coarse detection as a real frame.  Real ZC preambles produce
    # ratios > 4.0; LO leakage and colored noise sit at 1.5–2.3.  Set to 0.0
    # to disable (reverts to coarse-only gating).
    fine_peak_ratio_min: np.float32 = np.float32(3.0)
    # Minn/Park sign pattern [+,+,-,-] on short repetitions.
    # Adds a CFO-independent sign-flip check after Schmidl-Cox detection to
    # suppress LO-leakage false positives: for any tone r[n]=A·e^{jω₀n},
    # P(d+L)·conj(P(d)) is always positive-real; for the [+,+,-,-] preamble
    # it is always negative-real regardless of CFO.  Requires nreps==4.
    minn_park: bool = True


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
    return np.exp(-1j * np.pi * u * n * (n + 1) / n_zc).astype(np.complex64)


def generate_preamble(config: SynchronizerConfig) -> np.ndarray:
    """Build the full preamble: repeated short ZC followed by long ZC.

    When ``config.minn_park`` is True and ``short_preamble_nreps == 4``, the
    four short repetitions carry a Minn/Park sign pattern [+1, +1, -1, -1].
    This makes the Schmidl-Cox cross-product P(d+L)·conj(P(d)) negative-real
    at the preamble start for any CFO, while any pure tone yields a positive-
    real product — enabling LO-leakage rejection at the coarse-sync stage.
    """
    zc_short = generate_zadoff_chu(config.zc_root_short, config.short_preamble_nsym)
    zc_long = generate_zadoff_chu(config.zc_root_long, config.long_preamble_nsym)
    if config.minn_park and config.short_preamble_nreps == 4:
        signs = np.array([+1.0, +1.0, -1.0, -1.0], dtype=np.complex64)
        sign_pattern = np.repeat(signs, len(zc_short))
        short_rep = np.tile(zc_short, 4) * sign_pattern
    else:
        short_rep = np.tile(zc_short, config.short_preamble_nreps)
    return np.concatenate([short_rep, zc_long])


def build_long_ref(cfg: SynchronizerConfig, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Build the long preamble reference for matched filtering on post-RRC signals.

    Applies RRC twice (TX + RX matched filter = RC) so the reference matches
    what the preamble looks like in the matched-filtered receive stream.
    Output length is unchanged: long_preamble_nsym * sps samples.
    """
    zc = generate_zadoff_chu(cfg.zc_root_long, cfg.long_preamble_nsym)
    up = np.zeros(len(zc) * sps, dtype=np.complex64)
    up[::sps] = zc
    tx_filtered = np.convolve(up, rrc_taps, mode="same")
    # Apply second RRC (RX matched filter) — mode="same" keeps the ZC peak aligned at sample 0
    return np.convolve(tx_filtered.astype(np.complex64), rrc_taps, mode="same").astype(np.complex64)


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

    if _ext is not None:
        d_hats, cfo_hats, m_peaks = _ext.coarse_sync(
            samples, fs, samples_per_symbol,
            cfg.short_preamble_nsym, cfg.short_preamble_nreps, cfg.long_preamble_nsym,
            float(cfg.energy_floor), float(cfg.detection_threshold), float(cfg.energy_gate_fraction),
            bool(cfg.minn_park),
        )
        return CoarseResult(d_hats, cfo_hats, m_peaks)

    samples = samples.astype(np.complex64)
    cs_p = np.concatenate((np.zeros(1, dtype=np.complex64), np.cumsum(np.conj(samples[:-sample_cnt]) * samples[sample_cnt:])))
    p_d = cs_p[sample_cnt:] - cs_p[:-sample_cnt]

    cs_r = np.concatenate((np.zeros(1, dtype=np.float32), np.cumsum(np.abs(samples[sample_cnt:]) ** 2)))
    r_d = cs_r[sample_cnt:] - cs_r[:-sample_cnt]

    m_d = np.abs(p_d) ** 2 / np.maximum(r_d**2, cfg.energy_floor)

    # Multi-frame detection — batch iterate-and-advance (cf. gr-ieee802-11 sync_short MIN_GAP)
    # Minn/Park [+,+,-,-] plateau spans (nreps-1) sub-clusters, each L samples
    # apart, for a total width of (nreps-1)*L samples.  min_gap must exceed
    # the plateau width to avoid splitting within a single preamble, but should
    # be as small as possible to cleanly separate consecutive packets even when
    # noise creates above-threshold bridges between them.
    min_gap = (cfg.short_preamble_nreps - 1) * sample_cnt

    # Energy gate: suppress low-power regions where thermal noise yields M(d)≈1 spuriously.
    # In hardware, the guard has r_d ~1000x smaller than the preamble but M(d)≈1 because
    # both |p_d| and r_d are equally tiny. Requiring r_d > fraction*max(r_d) filters this out.
    if cfg.energy_gate_fraction > 0 and r_d.max() > 0:
        energy_gate = r_d > (r_d.max() * cfg.energy_gate_fraction)
    else:
        energy_gate = np.ones(len(m_d), dtype=bool)

    above = np.flatnonzero((m_d > cfg.detection_threshold) & energy_gate)
    if above.size == 0:
        return CoarseResult(np.empty(0, np.intp), np.empty(0), np.empty(0))

    # Cluster by gaps > min_gap — each cluster is one frame's plateau
    splits = np.r_[0, np.flatnonzero(np.diff(above) > min_gap) + 1]
    ends   = np.r_[splits[1:], len(above)]
    m_peaks = np.maximum.reduceat(m_d[above], splits)

    # Per-cluster d_hat selection with integrated Minn/Park.
    #
    # Within each cluster, large internal gaps (> 1.5·L) mark boundaries between
    # sub-clusters: the first gap separates any spurious early trigger (filter
    # transient in the preceding packet's payload/guard) from the true preamble.
    #
    # Candidates are: cluster[0], then the position after each large gap.
    # We try them in order and accept the first one that passes both:
    #   1. Energy-ratio check: r_d[cand] >= 25% of r_d[next_cand]
    #      (spurious near-silence positions have << 25%; AGC-attenuated preambles
    #       have ~42%, so the 25% threshold cleanly separates them)
    #   2. Minn/Park sign-flip: Re(P(d+L)·conj(P(d))) < 0
    #      (holds for [+,+,-,-] ZC preamble, rejects any single-tone interferer
    #       and payload data that accidentally triggered above threshold)
    #
    # If no candidate in a cluster passes, that cluster is dropped.
    large_gap_thr    = int(1.5 * sample_cnt)
    energy_ratio_thr = 0.25
    d_hats_list  = []
    m_peaks_list = []

    for (s0, e0), mp in zip(zip(splits, ends), m_peaks):
        cluster  = above[s0:e0]
        int_gaps = np.diff(cluster)

        # Build ordered candidate list + the last position BEFORE each gap
        # (used for the energy-ratio check — that position reflects whether the
        # pre-gap region has real signal or is near-silent tail/guard)
        large_gap_idxs = np.flatnonzero(int_gaps > large_gap_thr)
        candidates     = [int(cluster[0])] + [int(cluster[gi + 1]) for gi in large_gap_idxs]
        pre_gap_pos    = [int(cluster[gi])     for gi in large_gap_idxs]

        chosen = None
        for ci, cand in enumerate(candidates):
            # Energy-ratio: skip if the signal just before the gap leading to
            # the next candidate is near-silent (< 25% of post-gap energy).
            # Uses the last above-threshold position before the gap, not the
            # start of the current sub-cluster, so long payloads don't fool it.
            if ci < len(candidates) - 1:
                before_pos = pre_gap_pos[ci]
                next_cand  = candidates[ci + 1]
                if r_d[before_pos] < r_d[next_cand] * energy_ratio_thr:
                    continue

            # Minn/Park sign-flip check (CFO-independent).
            #
            # For preamble [+ZC, +ZC, -ZC, -ZC] with any CFO φ per lag:
            #   P(d₀)   ≈ +L·e^{jφ}   (rep1→rep2, same sign)
            #   P(d₀+L) ≈ −L·e^{jφ}   (rep2→rep3, sign flip)
            # ⇒  P(d₀+L)·conj(P(d₀)) ≈ −L²   (real-negative, φ cancels)
            #
            # For any pure tone or payload data triggering above threshold:
            #   Re(P(d+L)·conj(P(d))) >= 0  → rejected
            if cfg.minn_park:
                mp_idx = min(cand + sample_cnt, len(p_d) - 1)
                if np.real(p_d[mp_idx] * np.conj(p_d[cand])) >= 0:
                    continue

            chosen = cand
            break

        if chosen is not None:
            d_hats_list.append(chosen)
            m_peaks_list.append(float(mp))

    if not d_hats_list:
        return CoarseResult(np.empty(0, np.intp), np.empty(0), np.empty(0))

    d_hats  = np.array(d_hats_list, dtype=np.intp)
    m_peaks = np.array(m_peaks_list)

    if cfg.minn_park:
        # CFO: combine P(d) and P(d+L) constructively.
        # P(d₀) − P(d₀+L) = +L·e^{jφ} − (−L·e^{jφ}) = 2L·e^{jφ}  → 2× stronger.
        mp_idx    = np.minimum(d_hats + sample_cnt, len(p_d) - 1)
        p_combined = p_d[d_hats] - p_d[mp_idx]
        cfo_hats  = np.angle(p_combined) * fs / (2 * np.pi * sample_cnt)
    else:
        plateau_sum = np.add.reduceat(p_d[above], splits)
        plateau_cnt = np.diff(np.r_[splits, above.size])
        cfo_hats = np.angle(plateau_sum / plateau_cnt) * fs / (2 * np.pi * sample_cnt)

    return CoarseResult(d_hats, cfo_hats, m_peaks)


def build_fine_ref(long_ref: np.ndarray, cfg: SynchronizerConfig, sps: int) -> np.ndarray:
    """Precompute the FFT of the long reference for use in fine_timing.

    Call once at startup and pass the result to fine_timing to avoid
    recomputing the FFT on every call. Pads to the next power of 2 so
    numpy uses its fast radix-2 path.
    """
    sample_margin = cfg.long_margin_nsym * sps
    window_len = 2 * sample_margin + len(long_ref)
    min_pad = window_len + len(long_ref) - 1
    pad_len = 1 << (min_pad - 1).bit_length()  # next power of 2
    return np.conj(np.fft.fft(long_ref, n=pad_len)).astype(np.complex64)


def fine_timing(
    samples: np.ndarray,
    long_ref: np.ndarray,
    d_hats: np.ndarray,
    cfo_hats: np.ndarray,
    fs: int,
    samples_per_symbol: int,
    cfg: SynchronizerConfig,
    ref_f: np.ndarray | None = None,
) -> FineResult:
    """Fine timing by cross-correlation with the long preamble [4] [5].

    Pass a precomputed ref_f (from build_fine_ref) to avoid recomputing
    the FFT of long_ref on every call.
    """
    if not np.iscomplexobj(samples):
        msg = "samples must be complex (conj is a no-op on reals)"
        raise TypeError(msg)

    d_hats = np.atleast_1d(np.asarray(d_hats, dtype=np.intp))
    cfo_hats = np.atleast_1d(np.asarray(cfo_hats, dtype=np.float32))

    if _ext is not None:
        sample_idxs, peak_ratios, phase_estimates = _ext.fine_timing(
            samples, long_ref, d_hats, cfo_hats,
            fs, samples_per_symbol,
            cfg.short_preamble_nsym, cfg.short_preamble_nreps, cfg.long_margin_nsym,
            ref_f,
        )
        return FineResult(
            sample_idxs=sample_idxs,
            peak_ratios=peak_ratios,
            phase_estimates=phase_estimates,
        )

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
    phase_rad = (-2 * np.pi * (cfo_hats[:, None] / fs) * indices).astype(np.float32)
    windows = samples[indices] * (np.cos(phase_rad) + 1j * np.sin(phase_rad)).astype(np.complex64)

    # Batch cross-correlation via FFT
    valid_len = window_len - len(long_ref) + 1
    if ref_f is not None:
        pad_len = len(ref_f)  # matches whatever build_fine_ref used
    else:
        min_pad = window_len + len(long_ref) - 1
        pad_len = 1 << (min_pad - 1).bit_length()  # next power of 2
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
        phase_estimates=phase_at_payload%(2*np.pi),
    )
