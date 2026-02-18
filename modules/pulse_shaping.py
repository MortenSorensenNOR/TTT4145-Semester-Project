"""Pulse shaping filters for signal modulation."""

import numpy as np


def rrc_filter(sps: int, alpha: float, num_taps: int = 101) -> np.ndarray:
    """Root raised cosine filter."""
    t = np.arange(num_taps) - (num_taps - 1) / 2
    t = t / sps  # normalize to symbol periods

    h = np.zeros(num_taps)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1 + alpha * (4 / np.pi - 1)
        elif abs(ti) == 1 / (4 * alpha + 1e-10):
            h[i] = (
                alpha
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
                )
            )
        else:
            num = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(
                np.pi * ti * (1 + alpha),
            )
            den = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = num / (den + 1e-10)

    return h / np.sqrt(np.sum(h**2))  # normalize energy


class PulseShaper:
    """Pulse shaper for symbol transmission."""

    def __init__(self, sps: int, alpha: float, taps: int) -> None:
        """Initialize pulse shaper."""
        self.sps = sps
        self.taps = taps
        self.alpha = alpha
        self.pulse_shape = rrc_filter(sps, alpha, taps)

    def shape(self, symbols: np.ndarray) -> np.ndarray:
        """Apply pulse shaping to symbols."""
        return np.convolve(symbols, self.pulse_shape, mode="same")
