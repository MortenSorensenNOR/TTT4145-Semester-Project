"""Gardner timing error detector with cubic interpolation.

This module provides a Gardner TED implementation for symbol timing recovery.
It corrects fractional timing offsets in the received signal without requiring
pilot symbols.

Build the C++ extension (recommended) with:
    uv run python gardner_setup.py build_ext --inplace

The pure-Python fallback is used automatically if the extension is not available.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load pybind11 extension
# ---------------------------------------------------------------------------

try:
    from modules.gardner_ted import gardner_ext as _ext
    logger.info("Loaded gardner_ext pybind11 C++ extension.")
except ImportError:
    _ext = None
    logger.warning(
        "gardner_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python gardner_setup.py build_ext --inplace"
    )

# ---------------------------------------------------------------------------
# Pure-Python fallback
# ---------------------------------------------------------------------------

def _cubic_interp(s_re, s_im, idx, mu, n):
    """Catmull-Rom cubic interpolation at fractional offset mu from idx."""
    if idx < 1 or idx + 2 >= n:
        c = max(0, min(idx, n - 1))
        return s_re[c], s_im[c]
    def _interp1(v0, v1, v2, v3, mu):
        c0 =  v1
        c1 = -0.5*v0 + 0.5*v2
        c2 =  v0 - 2.5*v1 + 2.0*v2 - 0.5*v3
        c3 = -0.5*v0 + 1.5*v1 - 1.5*v2 + 0.5*v3
        return c0 + mu*(c1 + mu*(c2 + mu*c3))
    re = _interp1(s_re[idx-1], s_re[idx], s_re[idx+1], s_re[idx+2], mu)
    im = _interp1(s_im[idx-1], s_im[idx], s_im[idx+1], s_im[idx+2], mu)
    return re, im


def _gardner_py(signal: np.ndarray, sps: int, gain: float) -> np.ndarray:
    """Pure-Python Gardner TED with Catmull-Rom cubic interpolation."""
    signal = np.asarray(signal, dtype=np.complex64)
    n      = len(signal)
    re     = signal.real.copy()
    im     = signal.imag.copy()

    out = []
    mu  = 0.0
    k   = float(sps)

    while True:
        k_int     = int(k)
        k_mid_f   = k - sps * 0.5
        k_mid_int = int(k_mid_f)
        k_prev    = k_int - sps

        if k_int + 2 >= n or k_mid_int < 1 or k_prev < 1:
            break

        cr, ci = _cubic_interp(re, im, k_int,     k - k_int,           n)
        mr, mi = _cubic_interp(re, im, k_mid_int, k_mid_f - k_mid_int, n)
        pr, pi = _cubic_interp(re, im, k_prev,    k - k_int,           n)

        # Gardner error: Re{ (curr - prev) * conj(mid) }
        e  = (cr - pr) * mr + (ci - pi) * mi

        mu = float(np.clip(mu + gain * e, -0.5, 0.5))
        k += sps + mu

        out.append(complex(cr, ci))

    return np.array(out, dtype=np.complex64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_gardner_ted(
    signal: np.ndarray,
    sps: int,
    gain: float = 0.01,
) -> np.ndarray:
    """Apply Gardner timing error detector to correct fractional timing offset.

    Parameters
    ----------
    signal : np.ndarray
        RRC matched-filtered signal at ``sps`` samples per symbol.
        Must NOT include the preamble — pass only header/payload samples.
    sps : int
        Samples per symbol (must be >= 2).
    gain : float
        Loop gain. Typical range 0.001–0.01.
        Too high → oscillation. Too low → slow convergence.

    Returns
    -------
    np.ndarray[complex64]
        Timing-corrected symbols, one per input symbol period.
    """
    signal = np.asarray(signal, dtype=np.complex64)
    print("using ext:", _ext is not None)
    print("input length:", len(signal))
    if _ext is not None:
        result = _ext.gardner_ted(signal, int(sps), float(gain))
        print("output length:", len(result))
        return result
    else:
        return _gardner_py(signal, int(sps), float(gain))

    signal = np.asarray(signal, dtype=np.complex64)

    if _ext is not None:
        return _ext.gardner_ted(signal, int(sps), float(gain))
    else:
        return _gardner_py(signal, int(sps), float(gain))
