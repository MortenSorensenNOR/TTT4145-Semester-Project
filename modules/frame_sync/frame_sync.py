r"""Frame-level synchronization: preamble timing, CFO, and carrier phase.

Single-stage detector: full-buffer normalized cross-correlation against the
long Zadoff-Chu preamble.  CFO is recovered per detection from a half-window
phase split of the derotated preamble.

References:
[1] Zadoff-Chu sequence (used in 3GPP LTE):
    https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence
[2] Sklar & Harris, "Digital Communications", 3rd ed., Pearson, 2021,
    Sec. 3.2.2-3.2.3 (matched filter / cross-correlation), Sec. 10.2.2 (peak timing).
[3] PySDR - Synchronization chapter ("Frame Synchronization"):
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

    short_preamble_nsym: int = 43   # prime; leading guard ahead of the long ZC
    short_preamble_nreps: int = 2

    long_preamble_nsym: int = 89    # prime; longer ZC → higher xcorr peak-to-mean

    # Single-stage detector NCC threshold (full_buffer_xcorr_sync).
    # Normalized cross-correlation against the long ZC: bounded in [0,1], where
    # a clean preamble yields ~0.99.  CFO smears the peak by sinc²(cfo·N/fs),
    # so at ±5 kHz CFO (N=356, fs=4 MHz) the peak drops to ~0.49.  Payload
    # sidelobes in dense back-to-back bursts sit at <0.1.  0.3 leaves margin
    # for CFO smearing + AWGN while still rejecting sidelobes cleanly; raise
    # for stricter rejection or lower if you need wider CFO acquisition.
    single_stage_ncc_threshold: np.float32 = np.float32(0.3)


@dataclass
class FineResult:
    """Output of fine timing via cross-correlation."""

    sample_idxs: np.ndarray
    peak_ratios: np.ndarray
    phase_estimates: np.ndarray


def generate_zadoff_chu(u: int, n_zc: int) -> np.ndarray:
    r"""Generate a Zadoff-Chu sequence of length n_zc with root u [1].

    Formula from [1]: $x_u(n) = \exp(-j\pi \cdot u \cdot n \cdot (n+1) / N_{ZC})$ (for $q=1$ and prime $N_{ZC}$)
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
    """Build the full preamble: repeated short ZC followed by long ZC."""
    zc_short = generate_zadoff_chu(config.zc_root_short, config.short_preamble_nsym)
    zc_long = generate_zadoff_chu(config.zc_root_long, config.long_preamble_nsym)
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


def build_long_ref_rev(long_ref: np.ndarray) -> np.ndarray:
    """Time-reversed conjugate of long_ref. Cross-correlation of x with long_ref
    equals convolution of x with conj(long_ref[::-1]).
    """
    return np.conj(long_ref[::-1]).astype(np.complex64)


def full_buffer_xcorr_sync(
    samples: np.ndarray,
    long_ref: np.ndarray,
    long_ref_rev: np.ndarray,
    ncc_threshold: float,
    fs: int,
) -> tuple[FineResult, np.ndarray]:
    """Full-buffer normalized cross-correlation against the long ZC.

    Computes the normalized cross-correlation (NCC) of the receive buffer
    against the long-ZC reference:

        NCC(k) = |sum_m s[k+m]·conj(ref[m])|² / (||s[k:k+N]||² · ||ref||²)

    NCC is bounded in [0, 1] and is independent of buffer composition, so
    multi-frame bursts don't dilute the metric the way a global mean does.
    Picks peaks above ``ncc_threshold`` (clean preamble ≈ 0.99) and extracts
    a CFO estimate per peak from a half-window split of the derotated preamble.

    Use only when CFO is small enough that the long-ZC autocorrelation peak
    stays sharp; rule of thumb |CFO|·len(long_ref)/fs well below 1 rad of
    total rotation across the reference.
    """
    from scipy.signal import oaconvolve

    if not np.iscomplexobj(samples):
        msg = "samples must be complex"
        raise TypeError(msg)

    empty = (
        FineResult(np.empty(0, np.intp), np.empty(0, np.float32), np.empty(0, np.float32)),
        np.empty(0, np.float32),
    )

    n_ref = len(long_ref)
    if len(samples) < n_ref:
        return empty

    z = oaconvolve(samples, long_ref_rev, mode="valid").astype(np.complex64)
    n_z = len(z)
    if n_z == 0:
        return empty

    if _ext is not None and hasattr(_ext, "xcorr_ncc_post"):
        # Single C++ pass replaces a chain of numpy ops (z²+sig_pwr cumsum+NCC
        # +mask+cluster+CFO).
        result = _ext.xcorr_ncc_post(
            z, samples.astype(np.complex64, copy=False), long_ref,
            float(ncc_threshold), int(fs), 0.05,
        )
        if result.sample_idxs.size == 0:
            return empty
        return (
            FineResult(
                sample_idxs=result.sample_idxs,
                peak_ratios=result.peak_ratios,
                phase_estimates=result.phase_estimates,
            ),
            result.cfo_hats,
        )

    # Pure-numpy fallback (kept for environments without the extension).
    z_re, z_im = z.real, z.imag
    z_mag2 = z_re * z_re + z_im * z_im

    s_re, s_im = samples.real, samples.imag
    sig_pwr = s_re * s_re + s_im * s_im
    csum = np.empty(len(sig_pwr) + 1, dtype=np.float64)
    csum[0] = 0.0
    np.cumsum(sig_pwr, out=csum[1:])
    sig_energy = (csum[n_ref:n_ref + n_z] - csum[:n_z]).astype(np.float32)

    lr_re, lr_im = long_ref.real, long_ref.imag
    ref_energy = float(np.sum(lr_re * lr_re + lr_im * lr_im))
    denom = sig_energy * ref_energy
    ncc = z_mag2 / np.maximum(denom, np.float32(np.finfo(np.float32).tiny))

    max_sig_energy = float(sig_energy.max()) if n_z > 0 else 0.0
    if max_sig_energy > 0:
        ncc[sig_energy < np.float32(0.05 * max_sig_energy)] = 0.0
    else:
        return empty

    above = np.flatnonzero(ncc > ncc_threshold)
    if above.size == 0:
        return empty

    half = n_ref // 2
    long_ref_conj = np.conj(long_ref).astype(np.complex64)

    splits = np.r_[0, np.flatnonzero(np.diff(above) > n_ref) + 1, above.size]
    sample_idxs, peak_ratios, phase_estimates, cfo_hats = [], [], [], []
    for i in range(len(splits) - 1):
        cluster = above[splits[i]:splits[i + 1]]
        peak = int(cluster[np.argmax(ncc[cluster])])
        if peak + n_ref > len(samples):
            continue

        window = samples[peak:peak + n_ref] * long_ref_conj
        p = np.vdot(window[:half], window[half:half * 2])
        cfo_hat = float(np.angle(p)) * fs / (np.pi * n_ref)

        sample_idxs.append(peak)
        peak_ratios.append(float(ncc[peak]))
        phase_mid = float(np.angle(z[peak]))
        phase_at_payload = phase_mid + 2 * np.pi * cfo_hat / fs * (n_ref / 2)
        phase_estimates.append(phase_at_payload % (2 * np.pi))
        cfo_hats.append(cfo_hat)

    if not sample_idxs:
        return empty

    return (
        FineResult(
            sample_idxs=np.array(sample_idxs, dtype=np.intp),
            peak_ratios=np.array(peak_ratios, dtype=np.float32),
            phase_estimates=np.array(phase_estimates, dtype=np.float32),
        ),
        np.array(cfo_hats, dtype=np.float32),
    )
