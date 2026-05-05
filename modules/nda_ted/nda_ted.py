"""
NDA symbol timing synchroniser.

Non-Data-Aided symbol timing recovery using the algorithm from:
    M. Rice, "Digital Communications: A Discrete-Time Approach",
    Prentice Hall, 2009.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from modules.nda_ted import nda_ted_ext as _ext
    logger.info("Loaded nda_ted_ext pybind11 C++ extension.")
except ImportError:
    _ext = None
    logger.warning(
        "nda_ted_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python setup.py build_ext --inplace"
    )

def _nda_py(z: np.ndarray, Ns: int, L: int, BnTs: float, zeta: float) -> np.ndarray:
    K0 = -1.0
    Kp =  1.0
    denom = zeta + 1.0 / (4.0 * zeta)
    K1 = 4.0 * zeta / denom * BnTs / Ns / Kp / K0
    K2 = 4.0 / denom**2 * (BnTs / Ns)**2 / Kp / K0

    zz    = np.zeros(len(z), dtype=np.complex128)
    e_tau = np.zeros(len(z))
    c1_buff = np.zeros(2 * L + 1, dtype=complex)

    vi        = 0.0
    CNT_next  = 0.0
    mu_next   = 0.0
    underflow = 0
    epsilon   = 0.0
    mm        = 1

    z = np.hstack(([0], z))

    for nn in range(1, Ns * int(np.floor(len(z) / float(Ns)) - (Ns - 1))):
        CNT = CNT_next
        mu  = mu_next

        if underflow == 1:
            # Farrow cubic interpolant
            def farrow(arr, n, mu):
                v3 = np.sum(arr[n+2:n-1-1:-1] * [ 1/6., -1/2.,  1/2., -1/6.])
                v2 = np.sum(arr[n+2:n-1-1:-1] * [    0,  1/2.,    -1,  1/2.])
                v1 = np.sum(arr[n+2:n-1-1:-1] * [-1/6.,     1, -1/2., -1/3.])
                v0 = arr[n]
                return ((mu * v3 + v2) * mu + v1) * mu + v0

            z_interp = farrow(z, nn, mu)

            # NDA TED
            c1 = 0.0
            for kk in range(Ns):
                z_ted = farrow(z, nn + kk, mu)
                c1 += abs(z_ted)**2 * np.exp(-1j * 2 * np.pi / Ns * kk)
            c1 /= Ns

            c1_buff = np.hstack(([c1], c1_buff[:-1]))
            epsilon = -1.0 / (2.0 * np.pi) * np.angle(np.sum(c1_buff) / (2 * L + 1))

            zz[mm]    = z_interp
            e_tau[mm] = epsilon
            mm += 1

        vp       = K1 * epsilon
        vi      += K2 * epsilon
        v        = vp + vi
        W        = 1.0 / float(Ns) + v
        CNT_next = CNT - W

        if CNT_next < 0:
            CNT_next  = 1.0 + CNT_next
            underflow = 1
            mu_next   = CNT / W
        else:
            underflow = 0
            mu_next   = mu

    zz    = zz[1:mm]
    zz   /= np.std(zz) if np.std(zz) > 1e-10 else 1.0
    return zz.astype(np.complex64)


def apply_nda_ted(
    signal: np.ndarray,
    sps: int,
    BnTs: float = 0.01,
    zeta: float = 0.707,
    L: int = 2,
    prepend_first: bool = False,
) -> np.ndarray:
    """NDA symbol timing synchroniser with Farrow cubic interpolation.

    Parameters
    ----------
    signal : np.ndarray
        RRC matched-filtered signal at ``sps`` samples per symbol.
        Must NOT include the ZC preamble.
    sps : int
        Nominal samples per symbol.
    BnTs : float
        Normalised loop bandwidth (loop_bw * symbol_period).
        Typical range: 0.005 – 0.02.
    zeta : float
        Loop damping factor. 0.707 = maximally flat (Butterworth).
    L : int
        TED smoothing half-length. Smoothing window = 2*L+1 symbols.
    prepend_first : bool
        If True the kernel duplicates the first input sample internally to
        cancel the NDA TED 1-sample bias.  This replaces the previous
        Python-side ``np.concatenate([signal[:1], signal])`` and skips a
        full input-array copy on the hot RX path.

    Returns
    -------
    np.ndarray[complex64]
        Timing-corrected symbols at 1 sample/symbol, normalised to unit std.
    """
    signal = np.asarray(signal, dtype=np.complex64)

    if _ext is not None:
        return _ext.nda_ted(signal, int(sps), float(BnTs), float(zeta), int(L),
                            bool(prepend_first))
    if prepend_first and signal.size > 0:
        signal = np.concatenate([signal[:1], signal])
    return _nda_py(signal, int(sps), int(L), float(BnTs), float(zeta))
