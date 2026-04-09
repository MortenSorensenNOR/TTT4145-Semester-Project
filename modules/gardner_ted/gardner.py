"""
modules/gardner_ted.py
----------------------
Gardner Timing Error Detector — Python wrapper (mirrors costas_loop.py style).

The Gardner TED is a *non-data-aided*, second-order feedback timing recovery
algorithm.  It operates on an oversampled stream (>= 2 samples per symbol)
after the matched filter and requires no knowledge of the transmitted data.

Timing error equations
----------------------
BPSK:   e[m] = Re(z_mid) * ( Re(z_prev) - Re(z_curr) )
QPSK:   e[m] = Re{ conj(z_mid) * (z_prev - z_curr) }
8-PSK:  same as QPSK, normalised by |z_mid|

Strobe alignment
----------------
strobe is pre-loaded to (sps - 1) so the first fire is at sample index 0.
With mu=0 this exactly reproduces simple decimation: x[0], x[sps], x[2*sps]...

Reference
---------
F. M. Gardner, "A BPSK/QPSK Timing-Error Detector for Sampled Receivers,"
IEEE Trans. Commun., vol. COM-34, pp. 423-429, May 1986.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from modules.frame_constructor.frame_constructor import ModulationSchemes

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
        "Build it with: uv run python setup.py build_ext --inplace"
    )

# ---------------------------------------------------------------------------
# Pure-Python fallbacks
# ---------------------------------------------------------------------------

def _xget(x: np.ndarray, idx: int) -> complex:
    return complex(x[max(0, min(idx, len(x) - 1))])


def _farrow(xm1: complex, x0: complex, x1: complex, x2: complex, eta: float) -> complex:
    """4-tap cubic Farrow, eta in [0, 1)."""
    c0 =  x0
    c1 = -(1/6)*xm1 - 0.5*x0  +     x1 - (1/3)*x2
    c2 =   0.5 *xm1 -     x0  + 0.5*x1
    c3 = -(1/6)*xm1 + 0.5*x0  - 0.5*x1 + (1/6)*x2
    return c0 + eta*(c1 + eta*(c2 + eta*c3))


def _interp(x: np.ndarray, pos: float) -> complex:
    n   = int(math.floor(pos))
    eta = pos - n
    return _farrow(_xget(x,n-1), _xget(x,n), _xget(x,n+1), _xget(x,n+2), eta)


def _gardner_core_py(samples, alpha, beta, sps, mu, integrator, ted_fn):
    x     = np.asarray(samples, dtype=np.complex64)
    N     = len(x)
    sps_f = float(sps)

    out_syms  = []
    out_mu    = []

    # Pre-load so first fire is at i=0, matching simple decimation.
    strobe    = sps_f - 1.0
    prev_sym  = None

    for i in range(N):
        strobe += 1.0
        if strobe < sps_f:
            continue

        strobe -= sps_f

        on_time_pos = float(i) - strobe + mu * sps_f
        mid_pos     = on_time_pos - sps_f * 0.5

        on_time = _interp(x, on_time_pos)
        mid_sym = _interp(x, mid_pos)

        if prev_sym is not None:
            e = ted_fn(prev_sym, mid_sym, on_time)
            integrator += beta  * e
            mu         += alpha * e + integrator
            if   mu >  0.5: mu -= 1.0
            elif mu < -0.5: mu += 1.0

        prev_sym = on_time
        out_syms.append(on_time)
        out_mu.append(mu)

    return (
        np.array(out_syms, dtype=np.complex64),
        np.array(out_mu,   dtype=np.float32),
    )


def _ted_bpsk_py(prev, mid, curr):
    return mid.real * (prev.real - curr.real)

def _ted_qpsk_py(prev, mid, curr):
    d = prev - curr
    return mid.real * d.real + mid.imag * d.imag

def _ted_8psk_py(prev, mid, curr):
    d    = prev - curr
    e    = mid.real * d.real + mid.imag * d.imag
    ampl = abs(mid)
    return e / ampl if ampl > 1e-6 else e


def _make_py_fn(ted_fn):
    def _fn(samples, alpha, beta, sps, mu=0.0, integrator=0.0):
        return _gardner_core_py(samples, alpha, beta, sps, mu, integrator, ted_fn)
    return _fn

_py_bpsk = _make_py_fn(_ted_bpsk_py)
_py_qpsk = _make_py_fn(_ted_qpsk_py)
_py_8psk = _make_py_fn(_ted_8psk_py)

# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_func_map: dict = {
    ModulationSchemes.BPSK: _ext.gardner_loop_bpsk if _ext else _py_bpsk,
    ModulationSchemes.QPSK: _ext.gardner_loop_qpsk if _ext else _py_qpsk,
    ModulationSchemes.PSK8: _ext.gardner_loop_8psk if _ext else _py_8psk,
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _loop_parameters(bn: float, zeta: float) -> tuple[float, float]:
    wn    = bn / (zeta + 1.0 / (4.0 * zeta))
    denom = 1.0 + 2.0 * zeta * wn + wn ** 2
    return (4.0 * zeta * wn) / denom, (4.0 * wn ** 2) / denom


@dataclass(frozen=True)
class GardnerConfig:
    """Second-order Gardner TED loop configuration.

    Parameters
    ----------
    loop_noise_bandwidth_normalized:
        Normalised loop noise bandwidth (Bn·Ts). Typical range 0.005–0.05.
    damping_factor:
        Loop damping factor ζ. 0.707 = maximally-flat (Butterworth) response.
    initial_timing_offset:
        Seed value for mu, symbol-period units [-0.5, 0.5].
    initial_frequency_offset:
        Seed value for the PI integrator.
    alpha, beta:
        Loop-filter gains — computed automatically, do not set manually.
    """

    loop_noise_bandwidth_normalized: float = 0.01
    damping_factor: float = 0.707
    initial_timing_offset: float = 0.0
    initial_frequency_offset: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        a, b = _loop_parameters(
            self.loop_noise_bandwidth_normalized, self.damping_factor
        )
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_gardner_ted(
    samples: np.ndarray,
    config: GardnerConfig,
    modulation: ModulationSchemes,
    sps: int,
    current_timing_offset: float | None = None,
    current_frequency_offset: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a second-order Gardner TED to correct symbol timing.

    Parameters
    ----------
    samples:
        Complex baseband samples at ``sps`` samples-per-symbol (matched-filter
        output), cast to complex64 internally.
    config:
        Loop configuration.
    modulation:
        Modulation scheme — selects the TED error detector variant.
    sps:
        Samples per symbol (>= 2).
    current_timing_offset:
        Override for initial mu ∈ [-0.5, 0.5] (symbol-period units).
        Defaults to ``config.initial_timing_offset``.
    current_frequency_offset:
        Override for initial PI integrator state.
        Defaults to ``config.initial_frequency_offset``.

    Returns
    -------
    corrected_symbols : np.ndarray[complex64]
        One timing-corrected sample per symbol period.
    timing_estimates  : np.ndarray[float32]
        Fractional timing offset mu per symbol (diagnostic / state handoff).
    """
    if sps < 2:
        raise ValueError(f"apply_gardner_ted: sps must be >= 2, got {sps}")

    samples = np.asarray(samples, dtype=np.complex64)
    if len(samples) < 2 * sps:
        raise ValueError(
            f"apply_gardner_ted: need at least 2*sps={2*sps} samples, "
            f"got {len(samples)}"
        )

    func = _func_map.get(modulation)
    if func is None:
        raise NotImplementedError(
            f"No Gardner TED implemented for modulation scheme: {modulation!r}"
        )

    mu   = float(current_timing_offset
                 if current_timing_offset is not None
                 else config.initial_timing_offset)
    intg = float(current_frequency_offset
                 if current_frequency_offset is not None
                 else config.initial_frequency_offset)

    return func(
        samples,
        float(config.alpha),
        float(config.beta),
        int(sps),
        mu,
        intg,
    )

