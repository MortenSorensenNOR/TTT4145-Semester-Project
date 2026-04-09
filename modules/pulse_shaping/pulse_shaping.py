"""Root-raised-cosine pulse shaping: filter design, upsampling and downsampling."""

import logging
import numpy as np
import matplotlib.pyplot as plt

from modules.gardner_ted.gardner import apply_gardner_ted

logger = logging.getLogger(__name__)

try:
    from modules.pulse_shaping import pulse_shaping_ext as _ps_ext
    logger.info("Loaded pulse_shaping_ext pybind11 C++ extension.")
except ImportError:
    _ps_ext = None
    logger.warning(
        "pulse_shaping_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python setup.py build_ext --inplace"
    )

def rrc_filter(sps: int, alpha: np.float32, num_taps: int) -> np.ndarray:
    """Design a root-raised-cosine filter with unit energy."""
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    zero_mask = t == 0
    if alpha > 0:
        special_val = 1 / (4 * alpha)
        special_mask = np.abs(np.abs(t) - special_val) < 8 * np.finfo(np.float32).eps
        special_case = (
            alpha / np.sqrt(2)
            * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        )
    else:
        special_mask = np.zeros_like(t, dtype=bool)
        special_case = 0.0
    general_mask = ~zero_mask & ~special_mask

    num = np.sin(np.pi * t * (1 - alpha)) + 4 * alpha * t * np.cos(np.pi * t * (1 + alpha))
    den = np.pi * t * (1 - (4 * alpha * t) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        general_vals = num / den

    h = np.select(
        [zero_mask, special_mask, general_mask],
        [
            1 + alpha * (4 / np.pi - 1),
            special_case,
            general_vals,
        ],
    )

    # h_norm = (h / np.sqrt(np.sum(h**2))) 
    h_norm = h / np.max(h)
    return h_norm.astype(np.float32)


def upsample(symbols: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Zero-insert at sps rate and convolve with RRC taps."""
    if len(symbols) == 0:
        return np.ndarray([], dtype=np.complex64)
    if _ps_ext is not None:
        return _ps_ext.upsample(symbols.astype(np.complex64), sps, rrc_taps.astype(np.float32))
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols.astype(np.complex64)
    filtered = np.convolve(upsampled, rrc_taps, mode="full")
    return filtered


def upsample_no_filter(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Zero-insert at sps rate without any filtering (for hardware RRC path)."""
    if len(symbols) == 0:
        return np.ndarray([], dtype=np.complex64)
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
    upsampled[::sps] = symbols.astype(np.complex64)
    return upsampled

def downsample(signal: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Match-filter with RRC taps, strip group delay, and decimate."""

    if len(signal) == 0:
        return np.zeros(0, dtype=np.complex64)
    filtered = np.convolve(signal, rrc_taps, mode="full")
    delay = len(rrc_taps) - 1
    n_symbols = (len(signal) - (len(rrc_taps) - 1)) // sps
    return filtered[delay : delay + n_symbols * sps : sps]

def match_filter(signal: np.ndarray, rrc_taps: np.ndarray) -> np.ndarray:
    if _ps_ext is not None:
        return _ps_ext.match_filter(signal.astype(np.complex64), rrc_taps.astype(np.float32))
    filtered_full = np.convolve(signal.astype(np.complex64), rrc_taps, mode="full")
    delay = len(rrc_taps) - 1
    return filtered_full[delay:]

def decimate(signal: np.ndarray, sps: int) -> np.ndarray:
    n_symbols = len(signal) // sps
    return signal[:n_symbols * sps : sps]


if __name__ == "__main__":
    taps = rrc_filter(8, 0.25, 2 * 8 * 8 + 1)
    plt.plot(taps)
    plt.savefig("rrc.png")
