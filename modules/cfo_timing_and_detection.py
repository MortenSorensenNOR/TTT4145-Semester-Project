"""Synchronization algorithms for digital communication receivers.

References:
[1] Schmidl & Cox, "Robust Frequency and Timing Synchronization for OFDM",
    IEEE Trans. Comm., vol. 45, no. 12, Dec. 1997, pp. 1613-1621.
[2] Moose, "A Technique for Orthogonal Frequency Division Multiplexing
    Frequency Offset Correction", IEEE Trans. Comm., vol. 42, no. 10, Oct. 1994,
    pp. 2908-2914.
[3] IEEE Std 802.11a-1999, Sec. 17.3.3, Fig. 110, p.12:
    PLCP (Physical Layer Convergence Procedure) preamble
    (STF = Short Training Field, LTF = Long Training Field).
[4] Sklar & Harris, "Digital Communications", 3rd ed., Pearson, 2021,
    Sec. 3.2.2-3.2.3 (matched filter), Sec. 10.2.2 (correlation peak timing).

Steps:
1. Coarse timing  - Schmidl-Cox metric M(d)       [1] Eq. 5-8
2. Coarse CFO     - MLE from phase of P(d_hat)    [1] Eq. 37-39; [2] Eq. 21
3. Fine timing    - matched-filter with known LTF [3] Sec. 17.3.3; [4]

"""

from dataclasses import dataclass

import numpy as np
from sympy import isprime


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    n_short: int = 13  # prime ZC length; +-fs/(2·n_short·sps) [2] p.2912
    n_long: int = 139  # prime ZC length for LTF
    zc_root: int = 7  # root u; any 0 < u < n_zc for prime n_zc
    n_short_reps: int = 8  # STF reps; 802.11a uses 10 [3] Fig. 110
    threshold: float = 0.5  # M(d) detect; M→1 at high SNR [1] Eq. 19
    plateau_edge_fraction: float = 0.9  # [1] Sec. III-B-3
    energy_floor: float = np.finfo(np.float64).tiny  # /0 guard
    long_margin_factor: int = 5  # search +-margin·sps around LTF


@dataclass
class CoarseResult:
    """Output of coarse timing + CFO estimation."""

    d_hat: np.intp  # coarse timing estimate (sample index)
    cfo_hat_hz: np.floating  # coarse CFO estimate (Hz)
    fs: int  # sample rate used for CFO estimation


def generate_zadoff_chu(u: int = 7, n_zc: int = 61) -> np.ndarray:
    """Generate a Zadoff-Chu sequence of length n_zc with root u.

    Source: https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence
    n_zc must be prime so c_f=1 and gcd(u, n_zc)=1 for any 0 < u < n_zc.
    """
    if not isprime(n_zc):
        msg = f"n_zc ({n_zc}) must be prime to achieve DFT property"
        raise ValueError(msg)

    if not (0 < u < n_zc):
        msg = f"u ({u}) must be in the range [1, {n_zc - 1}]"
        raise ValueError(msg)

    n = np.arange(n_zc)
    return np.exp(-1j * np.pi * u * n * (n + 1) / n_zc)


def coarse_timing(
    rx: np.ndarray,
    config: SynchronizerConfig,
    sps: int,
) -> tuple[np.intp, np.complexfloating]:
    """Coarse timing via the Schmidl-Cox autocorrelation metric.

    P(d) = Σ_{m=0}^{L-1} r*_{d+m} · r_{d+m+L}    [1] Eq. 5, p.1615
    R(d) = Σ_{m=0}^{L-1} |r_{d+m+L}|²            [1] Eq. 7, p.1615
    M(d) = |P(d)|² / (R(d))²                      [1] Eq. 8, p.1615

    where L = n_short * sps (one ZC repetition in samples).

    The paper's "avg 90%" method ([1] Sec. III-B-3, Table II, p.1619) averages
    the left and right 90%-of-peak edges to find the plateau center. Here we
    take only the left (rising) edge since we need the preamble start, not center.

    Returns (d_hat, p_peak). p_peak carries the phase needed for CFO estimation.
    """
    length = config.n_short * sps
    if len(rx) < 2 * length:
        msg = f"rx too short ({len(rx)} samples) for L={length}"
        raise ValueError(msg)

    # P(d): sliding autocorrelation between adjacent L-sample blocks [1] Eq. 5
    conj_prod = np.conj(rx[:-length]) * rx[length:]
    p_d = np.convolve(conj_prod, np.ones(length), mode="valid")

    # R(d): sliding energy of the second block [1] Eq. 7
    r_d = np.convolve(abs(rx[length:]) ** 2, np.ones(length), mode="valid")

    # M(d) = |P(d)|² / R(d)² [1] Eq. 8
    m_d = np.abs(p_d) ** 2 / np.maximum(r_d**2, config.energy_floor)

    d_peak = np.argmax(m_d)
    if m_d[d_peak] < config.threshold:
        msg = f"M(d) peak {m_d[d_peak]:.3f} below threshold {config.threshold}"
        raise ValueError(msg)

    # Preamble start: rising edge at 90% of peak [1] Sec. III-B-3, p.1619
    d_hat = np.argmax(config.plateau_edge_fraction * m_d[d_peak] <= m_d)

    return d_hat, p_d[d_peak]


def estimate_coarse_cfo(
    p_peak: np.complexfloating,
    n_short: int,
    fs: int,
    sps: int,
) -> np.floating:
    """Coarse CFO from phase of autocorrelation at the timing estimate.

    φ = πTΔf                       [1] Eq. 37, p.1619
    φ̂ = angle(P(d))               [1] Eq. 38, p.1619
    Δf = φ̂/(πT) = φ̂·fs/(2πL)    [1] Eq. 39, p.1620

    Equivalent to Moose's MLE:     [2] Eq. 21, p.2910
    Acquisition range: +-fs/(2L)   [2] Sec. IV, p.2912
    """
    length = n_short * sps
    phi_hat = np.angle(p_peak)
    return phi_hat * fs / (2 * np.pi * length)


def fine_timing(
    rx: np.ndarray,
    coarse: CoarseResult,
    s: np.ndarray,
    config: SynchronizerConfig,
    sps: int,
) -> np.intp:
    """Fine timing by matched-filter cross-correlation with the long training sequence.

    z(T) = ∫ r(τ)s(τ) dτ                       [4] Eq. 3.59, p.127
    max over d [ ∫ r(t) s(t-d) dt ]            [4] Eq. 10.10, p.734

    Discrete form:
    z[d] = Σ_{n=0}^{N-1} r[d+n] · s*[n]       np.correlate(r, s)
    d_fine = argmax_d |z[d]|                    np.argmax(np.abs(z))

    where s is the known long ZC template and r is the CFO-corrected
    received signal.

    The LTF in 802.11a is specified for channel estimation and fine CFO
    ([3] Sec. 17.3.3, Fig. 110, p.12); using it for fine timing via
    cross-correlation is the matched-filter receiver technique [4] Sec. 3.2.2.
    """
    # Search window around expected LTF start
    length = config.n_short * sps
    margin = config.long_margin_factor * sps
    expected_start = coarse.d_hat + config.n_short_reps * length
    search_start = max(expected_start - margin, 0)
    search_end = min(len(rx), expected_start + 2 * margin + len(s))

    if search_end - search_start < len(s):
        got = search_end - search_start
        msg = f"search window too short (need {len(s)}, got {got})"
        raise ValueError(msg)

    # r: CFO-corrected received signal
    n = np.arange(search_start, search_end)
    cfo_phase = -1j * 2 * np.pi * coarse.cfo_hat_hz / coarse.fs * n
    r = rx[search_start:search_end] * np.exp(cfo_phase)

    # z[d] = Σ r[d+n] · s*[n]  (matched filter [4] Eq. 3.59)
    z = np.correlate(r, s, mode="valid")

    # d_fine = argmax |z[d]|  (correlation peak timing [4] Eq. 10.10)
    return search_start + np.argmax(np.abs(z))
