"""Synchronization algorithms for digital communication receivers.

1. Coarse CFO estimation
    - Repeated sequence
2. Frame detection and frame timing
    - Zadoff-Chu
3. Symbol timing
    - Gardner TED
4. Fine CFO estimation
    - Costas loop
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime

from modules.modulation import QPSK
from modules.plotting import plot_iq

class ZadofChu:
    """Zadoff-Chu sequence generator."""
    def generate(self, u: int = 7, N_ZC: int = 61) -> np.ndarray:
        """Generate a sequence of length `N_ZC` of root u."""
        assert isprime(N_ZC)
        c_f = N_ZC % 2
        q = 0
        n = np.arange(0, N_ZC, 1)
        x = np.exp(-1j * (np.pi * u * n * (n + c_f + 2*q)) / N_ZC)
        return x


class CoarseCFOSequence:
    """Sequence generator to estimate the coarse frequency offset for the synchronization process.
    based on sending a repeated known sequence. The coarse frequency offset can be found as
        Δf = 1/(2pi * N * T_S) * ∠ (∑ s∗[n] * s[n + N])
    This gives frequency ambiguity of < 1/(2 * N * T_S). 
    For Pluto max CFO @2.4 GHz is around 60 khz per radio. Assuming 1 MHz bandwidth and alpha = 0.25, 
    we have Ts = 1.25 us, so N = 3 to encompase the possible frequency offset. This kinda sucks.
    """

    def __init__(self, N: int, M: int, modulation: str = 'qpsk') -> None:
        self.N = N
        self.M = M
        self.modulation = modulation

        self.preamble = self.generate()

    def generate(self) -> np.ndarray:
        seq = np.zeros(self.N)
        if self.modulation == 'bpsk':
            np.random.seed(42)
            seq = 2 * (np.random.randint(0, 2, self.N)) - 1
        elif self.modulation == 'qpsk':
            np.random.seed(42)
            bits = np.random.randint(0, 2, self.N * 2)
            seq = QPSK().bits2symbols(bits)

        return np.tile(seq, self.M)

    def estimate_corase_cfo(self, rx_signal, Ts):
        """Correlate block n with block n+1, average over M-1 paris"""
        acc = 0
        for m in range(self.M - 1):
            block0 = rx_signal[m*self.N : (m+1)*self.N]
            block1 = rx_signal[(m+1)*self.N : (m+2)*self.N]
            acc += np.vdot(block0, block1)

        phase = np.angle(acc)
        cfo = phase / (2 * np.pi * self.N * Ts)
        return cfo


class Syncronizer:
    """Placeholder synchronization pipeline."""

    def __init__(self) -> None:
        """Initialize the synchronizer placeholder."""
