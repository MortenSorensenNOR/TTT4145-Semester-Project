"""
Frame-level synchronization: preamble timing, CFO, and carrier phase.

Single-stage detector: full-buffer normalized cross-correlation against the
Zadoff-Chu preamble.  CFO is recovered per detection from a half-window
phase split of the derotated preamble.

References:
[1] Zadoff-Chu sequence (used in 3GPP LTE):
    https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence
[2] Sklar & Harris, "Digital Communications", 3rd ed., Pearson, 2021,
    Sec. 3.2.2-3.2.3 (matched filter / cross-correlation), Sec. 10.2.2 (peak timing).
[3] PySDR - Synchronization chapter ("Frame Synchronization"):
    https://pysdr.org/content/sync.html

"""

import numpy as np
from scipy.signal import oaconvolve
from dataclasses import dataclass

import logging
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


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    zc_root: int = 13
    preamble_nsym: int = 89
    ncc_threshold: np.float32 = np.float32(0.3)


@dataclass
class FineResult:
    """Output of fine timing via cross-correlation."""

    sample_idxs: np.ndarray
    peak_ratios: np.ndarray
    phase_estimates: np.ndarray


def generate_zadoff_chu(u: int, n_zc: int) -> np.ndarray:
    """Generate a Zadoff-Chu sequence of length n_zc with root u"""
    n = np.arange(n_zc)
    return np.exp(-1j * np.pi * u * n * (n + 1) / n_zc).astype(np.complex64)


def generate_preamble(config: SynchronizerConfig) -> np.ndarray:
    """Build the preamble at symbol rate (used by TX)."""
    return generate_zadoff_chu(config.zc_root, config.preamble_nsym)


def build_preamble_ref(cfg: SynchronizerConfig, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Build the preamble reference for matched filtering on post-RRC signals.

    Applies RRC twice (TX + RX matched filter = RC) so the reference matches
    what the preamble looks like in the matched-filtered receive stream.
    """
    zc = generate_zadoff_chu(cfg.zc_root, cfg.preamble_nsym)
    up = np.zeros(len(zc) * sps, dtype=np.complex64)
    up[::sps] = zc
    tx_filtered = np.convolve(up, rrc_taps, mode="same")
    return np.convolve(tx_filtered.astype(np.complex64), rrc_taps, mode="same").astype(np.complex64)


def build_preamble_ref_rev(preamble_ref: np.ndarray) -> np.ndarray:
    """Time-reversed conjugate of preamble_ref"""
    return np.conj(preamble_ref[::-1]).astype(np.complex64)


def full_buffer_xcorr_sync(
    samples: np.ndarray,
    preamble_ref: np.ndarray,
    preamble_ref_rev: np.ndarray,
    ncc_threshold: float,
    fs: int,
) -> tuple[FineResult, np.ndarray]:
    """Full-buffer normalized cross-correlation against the ZC preamble"""
    if not np.iscomplexobj(samples):
        msg = "samples must be complex"
        raise TypeError(msg)

    empty = (
        FineResult(np.empty(0, np.intp), np.empty(0, np.float32), np.empty(0, np.float32)),
        np.empty(0, np.float32),
    )

    n_ref = len(preamble_ref)
    if len(samples) < n_ref:
        return empty

    z = oaconvolve(samples, preamble_ref_rev, mode="valid").astype(np.complex64)
    n_z = len(z)
    if n_z == 0:
        return empty

    if _ext is not None and hasattr(_ext, "xcorr_ncc_post"):
        result = _ext.xcorr_ncc_post(
            z, samples.astype(np.complex64, copy=False), preamble_ref,
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

    # normalized cross-correlation
    sig_pwr = np.abs(samples).astype(np.float32) ** 2
    csum = np.concatenate(([0.0], np.cumsum(sig_pwr, dtype=np.float64)))
    sig_energy = (csum[n_ref:n_ref + n_z] - csum[:n_z]).astype(np.float32)
    ref_energy = float(np.sum(np.abs(preamble_ref) ** 2))
    ncc = np.abs(z).astype(np.float32) ** 2 / np.maximum(
        sig_energy * ref_energy, np.float32(np.finfo(np.float32).tiny)
    )

    # mask near-silent windows (else noise floor produces garbage NCC)
    max_sig = float(sig_energy.max())
    if max_sig <= 0:
        return empty
    ncc[sig_energy < np.float32(0.05 * max_sig)] = 0.0

    # cluster above-threshold hits; a large gap starts a new frame
    above = np.flatnonzero(ncc > ncc_threshold)
    if above.size == 0:
        return empty
    clusters = np.split(above, np.flatnonzero(np.diff(above) > n_ref) + 1)

    sample_idxs, peak_ratios, phase_estimates, cfo_hats = [], [], [], []
    half = n_ref // 2
    preamble_conj = np.conj(preamble_ref).astype(np.complex64)
    for cluster in clusters:
        peak = int(cluster[np.argmax(ncc[cluster])])
        if peak + n_ref > len(samples):
            continue

        # CFO from phase difference between halves of the derotated preamble
        window = samples[peak:peak + n_ref] * preamble_conj
        p = np.vdot(window[:half], window[half:2 * half])
        cfo_hat = float(np.angle(p)) * fs / (np.pi * n_ref)

        # carry mid-window phase forward to payload start for Costas
        phase_mid = float(np.angle(z[peak]))
        phase_payload = (phase_mid + 2 * np.pi * cfo_hat / fs * (n_ref / 2)) % (2 * np.pi)

        sample_idxs.append(peak)
        peak_ratios.append(float(ncc[peak]))
        phase_estimates.append(phase_payload)
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
