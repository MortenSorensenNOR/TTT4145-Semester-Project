"""Root-raised-cosine pulse shaping: filter design, upsampling and downsampling."""

import numpy as np
import matplotlib.pyplot as plt


def rrc_filter(sps: int, alpha: float, num_taps: int) -> np.ndarray:
    """Design a root-raised-cosine filter with unit energy."""
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    zero_mask = t == 0
    if alpha > 0:
        special_val = 1 / (4 * alpha)
        special_mask = np.abs(np.abs(t) - special_val) < 8 * np.finfo(float).eps
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

    return h / np.max(h)


def upsample(symbols: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Zero-insert at sps rate and convolve with RRC taps."""
    if len(symbols) == 0:
        return np.ndarray([], dtype=complex)
    upsampled = np.zeros(len(symbols) * sps, dtype=complex)
    upsampled[::sps] = symbols
    return np.convolve(upsampled, rrc_taps, mode="full")

def downsample(signal: np.ndarray, sps: int, rrc_taps: np.ndarray) -> np.ndarray:
    """Match-filter with RRC taps, strip group delay, and decimate."""
    if len(signal) == 0:
        return np.ndarray([], dtype=complex)
    delay = (len(rrc_taps) - 1)//2
    n_out = max(0, (len(signal) - len(rrc_taps) + 1) // sps)
    filtered = np.convolve(signal, rrc_taps, mode="full")
    return filtered[delay : delay + n_out * sps : sps]


if __name__ == "__main__":
    taps = rrc_filter(8, 0.25, 2 * 8 * 8 + 1)
    plt.plot(taps)
    plt.savefig("rrc.png")
