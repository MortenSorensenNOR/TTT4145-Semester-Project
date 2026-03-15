"""Synchronization algorithms for digital communication receivers.

References:
[1] GNU Radio - Schmidl & Cox OFDM synch. (ofdm_sync_sc_cfb block):
    https://wiki.gnuradio.org/index.php/Schmidl_&_Cox_OFDM_synch. (trailing dot is part of URL)
[2] MATLAB - OFDM Synchronization example (same algorithm):
    https://www.mathworks.com/help/comm/ug/ofdm-synchronization.html
[3] Zadoff-Chu sequence (used in 3GPP LTE/5G):
    https://en.wikipedia.org/wiki/Zadoff%E2%80%93Chu_sequence
[4] Sklar & Harris, "Digital Communications", 3rd ed., Pearson, 2021,
    Sec. 3.2.2-3.2.3 (matched filter), Sec. 10.2.2 (correlation peak timing).
[5] PySDR - Synchronization chapter:
    https://pysdr.org/content/sync.html
[6] MATLAB - HDL OFDM Receiver (full coarse→fine pipeline):
    https://www.mathworks.com/help/wireless-hdl/ug/hdlofdmreceiver.html

Steps (see [6] for overall pipeline):
1. Coarse timing  - Schmidl-Cox autocorrelation metric M(d)  [1] [2]
2. Coarse CFO     - MLE from phase of P(d_hat)               [1] [2]
3. Fine timing    - matched-filter with long preamble          [4] [5]

"""

from dataclasses import dataclass

import numpy as np
from sympy import isprime


@dataclass
class SynchronizerConfig:
    """Configuration for the synchronizer."""

    n_short: int = 13  # prime ZC length [3]; +-fs/(2·n_short·sps)
    n_long: int = 139  # prime ZC length for LTF [3]
    zc_root: int = 7  # root u; any 0 < u < n_zc for prime n_zc
    n_short_reps: int = 8  # short preamble repetitions [2]
    threshold: float = 0.5  # M(d) detect; M→1 at high SNR [1]
    plateau_edge_fraction: float = 0.9  # rising-edge fraction [1]
    energy_floor: float = np.finfo(np.float64).tiny  # /0 guard
    long_margin_factor: int = 5  # search +-margin·sps around long preamble


@dataclass
class CoarseResult:
    """Output of coarse timing + CFO estimation."""

    d_hat: np.intp  # coarse timing estimate (sample index)
    cfo_hat_hz: np.floating  # coarse CFO estimate (Hz)
    fs: int  # sample rate used for CFO estimation


def generate_zadoff_chu(u: int = 7, n_zc: int = 61) -> np.ndarray:
    """Generate a Zadoff-Chu sequence of length n_zc with root u [3].

    Formula from [3]:  x_u(n) = exp(-jπ·u·n·(n+1) / N_ZC)  (for odd N_ZC)
    n_zc must be prime so gcd(u, n_zc)=1 for any 0 < u < n_zc.
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
    """Coarse timing via the Schmidl-Cox autocorrelation metric [1] [2].

    The algorithm exploits the repeated structure of the short preamble.
    When the receiver slides over two adjacent copies of the same sequence,
    their correlation P(d) peaks and the timing metric M(d) approaches 1.
    See [1] for the formulas below; [2] shows the same algorithm in MATLAB.

    Formulas (variable names match the code):
        P(d)  = Σ_{m=0}^{L-1} r*_{d+m} · r_{d+m+L}   → p_d
        R(d)  = Σ_{m=0}^{L-1} |r_{d+m+L}|²            → r_d
        M(d)  = |P(d)|² / (R(d))²                      → m_d

    where L = n_short * sps (one ZC repetition in samples) → length.

    Implementation: the sliding sums P(d) and R(d) are computed via
    convolution with a rectangular window of ones:
        conv(x, ones(L), "valid")[d] = Σ_{m=0}^{L-1} x[d+m]

    Returns (d_hat, p_peak). p_peak carries the phase needed for CFO estimation.
    """
    length = config.n_short * sps  # L in the formulas above
    if len(rx) < 2 * length:
        msg = f"rx too short ({len(rx)} samples) for L={length}"
        raise ValueError(msg)

    # Step 1: P(d) — sliding autocorrelation between adjacent L-sample blocks.
    # conj(rx[m]) * rx[m+L] gives one term of the sum;
    # convolving with ones(L) computes the sliding sum over L consecutive terms.
    conj_prod = np.conj(rx[:-length]) * rx[length:]
    p_d = np.convolve(conj_prod, np.ones(length), mode="valid")

    # Step 2: R(d) — sliding energy of the second block (same sliding-sum trick).
    r_d = np.convolve(abs(rx[length:]) ** 2, np.ones(length), mode="valid")

    # Step 3: M(d) = |P(d)|² / R(d)²  (energy_floor prevents division by zero)
    m_d = np.abs(p_d) ** 2 / np.maximum(r_d**2, config.energy_floor)

    # Find the peak of M(d); reject if below detection threshold [1].
    d_peak = np.argmax(m_d)
    if m_d[d_peak] < config.threshold:
        msg = f"M(d) peak {m_d[d_peak]:.3f} below threshold {config.threshold}"
        raise ValueError(msg)

    # Preamble start: find the first sample where M(d) reaches plateau_edge_fraction of peak.
    # argmax on a boolean array returns the index of the first True value.
    d_hat = np.argmax(config.plateau_edge_fraction * m_d[d_peak] <= m_d)

    return d_hat, p_d[d_peak]


def estimate_coarse_cfo(
    p_peak: np.complexfloating,
    n_short: int,
    fs: int,
    sps: int,
) -> np.floating:
    """Coarse CFO from phase of autocorrelation at the timing estimate [1] [2].

    The phase of P(d) at the timing peak encodes the frequency offset.
    GNU Radio [1] outputs this as φ̂ scaled by symbol duration;
    we convert to Hz directly:

        φ̂  = angle(P(d))              → phi_hat
        Δf  = φ̂ · fs / (2π · L)       → return value (Hz)

    Acquisition range: +-fs/(2L).
    """
    length = n_short * sps  # L in the formula above
    phi_hat = np.angle(p_peak)
    return phi_hat * fs / (2 * np.pi * length)


def fine_timing(
    rx: np.ndarray,
    coarse: CoarseResult,
    s: np.ndarray,
    config: SynchronizerConfig,
    sps: int,
) -> np.intp:
    """Fine timing by cross-correlation with the long preamble sequence [4] [5].

    After coarse timing and CFO correction, we cross-correlate the received
    signal with the known long preamble to pinpoint the exact start.
    See [5] "Frame Synchronization" for the cross-correlation approach;
    [4] Sec. 3.2.2 for the matched-filter interpretation.

        z[d]   = Σ_{n=0}^{N-1} r[d+n] · s*[n]     → np.correlate(r, s)
        d_fine = argmax_d |z[d]|                    → np.argmax(np.abs(z))

    where s is the known long preamble sequence and r is the CFO-corrected
    received signal.  np.correlate(r, s, "valid") computes exactly this
    cross-correlation: it slides s* over r, returning the inner product at
    each lag.
    """
    # The long preamble sits right after (n_short_reps) copies of the short
    # preamble. We search in a window around the expected position.
    length = config.n_short * sps
    margin = config.long_margin_factor * sps
    expected_start = coarse.d_hat + config.n_short_reps * length
    search_start = max(expected_start - margin, 0)
    search_end = min(len(rx), expected_start + 2 * margin + len(s))

    if search_end - search_start < len(s):
        got = search_end - search_start
        msg = f"search window too short (need {len(s)}, got {got})"
        raise ValueError(msg)

    # Remove the coarse CFO before correlating: multiply by exp(-j2π·Δf·n/fs)
    n = np.arange(search_start, search_end)
    cfo_phase = -1j * 2 * np.pi * coarse.cfo_hat_hz / coarse.fs * n
    r = rx[search_start:search_end] * np.exp(cfo_phase)

    # Cross-correlate with the known preamble and pick the peak [5]
    z = np.correlate(r, s, mode="valid")
    return search_start + np.argmax(np.abs(z))
