"""Costas loop for blind carrier phase synchronization.

This module provides a Costas loop implementation for recovering carrier phase
from a modulated signal without the need for pilot symbols. It is a form of
phase-locked loop (PLL) that is "decision-directed," meaning it uses the
output of a symbol slicer to derive its phase error estimate.

Build the C++ extension (recommended) with:
    uv run python costas_setup.py build_ext --inplace

The pure-Python fallback is used automatically if the extension is not available.
"""

import logging
from dataclasses import dataclass

import numpy as np

from modules.modulation_schemes import BPSK, PSK8, QPSK

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load pybind11 extension
# ---------------------------------------------------------------------------

try:
    from modules.costas_loop import costas_ext as _ext
    logger.info("Loaded costas_ext pybind11 C++ extension.")
except ImportError:
    _ext = None
    logger.warning(
        "costas_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python costas_setup.py build_ext --inplace"
    )

# ---------------------------------------------------------------------------
# Pure-Python fallbacks (used when extension is not available)
# ---------------------------------------------------------------------------


def _bpsk_py(symbols, alpha, beta, phase, integrator):
    n = len(symbols)
    out_syms  = np.empty(n, dtype=np.complex64)
    out_phase = np.empty(n, dtype=np.float32)
    for i, s in enumerate(symbols):
        c = s * np.exp(-1j * phase)
        e = np.imag(c) * np.sign(np.real(c))
        integrator += beta  * e
        phase      += alpha * e + integrator
        phase       = (phase + np.pi) % (2 * np.pi) - np.pi
        out_syms[i] = c; out_phase[i] = phase
    return out_syms, out_phase


def _qpsk_py(symbols, alpha, beta, phase, integrator):
    n = len(symbols)
    out_syms  = np.empty(n, dtype=np.complex64)
    out_phase = np.empty(n, dtype=np.float32)
    for i, s in enumerate(symbols):
        c  = s * np.exp(-1j * phase)
        re, im = np.real(c), np.imag(c)
        e  = im * np.sign(re) - re * np.sign(im)
        integrator += beta  * e
        phase      += alpha * e + integrator
        phase       = (phase + np.pi) % (2 * np.pi) - np.pi
        out_syms[i] = c; out_phase[i] = phase
    return out_syms, out_phase


def _8psk_py(symbols, alpha, beta, phase, integrator):
    n = len(symbols)
    out_syms  = np.empty(n, dtype=np.complex64)
    out_phase = np.empty(n, dtype=np.float32)
    for i, s in enumerate(symbols):
        c  = s * np.exp(-1j * phase)
        e  = np.angle(c ** 8) / 8.0
        integrator += beta  * e
        phase      += alpha * e + integrator
        phase       = (phase + np.pi) % (2 * np.pi) - np.pi
        out_syms[i] = c; out_phase[i] = phase
    return out_syms, out_phase


# ---------------------------------------------------------------------------
# Modulator classes gain a costas_loop() method
#
# Each method tries the C++ extension first and falls back to Python silently.
# Adding a new modulation scheme only requires editing modulation.py.
# ---------------------------------------------------------------------------

def _attach_costas_methods() -> None:
    """Monkey-patch costas_loop() onto each modulator class at import time."""

    def _make_method(cpp_fn, py_fn):
        def costas_loop(self, symbols, alpha, beta, phase=0.0, integrator=0.0):
            fn = cpp_fn if _ext else py_fn
            return fn(
                np.asarray(symbols, dtype=np.complex64),
                float(alpha), float(beta),
                float(phase), float(integrator),
            )
        return costas_loop

    BPSK.costas_loop     = _make_method(getattr(_ext, "costas_loop_bpsk", None), _bpsk_py)
    QPSK.costas_loop     = _make_method(getattr(_ext, "costas_loop_qpsk", None), _qpsk_py)
    PSK8.costas_loop     = _make_method(getattr(_ext, "costas_loop_8psk", None), _8psk_py)


_attach_costas_methods()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _loop_parameters(bn: float, zeta: float) -> tuple[float, float]:
    wn    = bn / (zeta + 1.0 / (4.0 * zeta))
    denom = 1.0 + 2.0 * zeta * wn + wn ** 2
    return (4.0 * zeta * wn) / denom, (4.0 * wn ** 2) / denom


@dataclass(frozen=True)
class CostasConfig:
    """Second-order Costas loop configuration.

    Parameters
    ----------
    loop_noise_bandwidth_normalized:
        Normalised loop noise bandwidth (Bn·Ts). Typical range 0.005–0.05.
    damping_factor:
        Loop damping factor ζ. 0.707 gives a maximally-flat (Butterworth) response.
    initial_freq_offset_rad_per_symbol:
        Seed value for the integrator (frequency offset in rad/symbol).
    alpha, beta:
        Loop-filter gains — computed automatically, do not set manually.
    """

    loop_noise_bandwidth_normalized: float = 0.05
    damping_factor: float = 0.707
    initial_freq_offset_rad_per_symbol: float = 0.0
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


def apply_costas_loop(
    symbols: np.ndarray,
    config: CostasConfig,
    modulator,
    current_phase_estimate: float = 0.0,
    current_frequency_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a second-order Costas loop to correct carrier phase offset.

    Parameters
    ----------
    symbols:
        Complex baseband symbols (cast to complex64 internally).
    config:
        Loop configuration.
    modulator:
        Modulation scheme — selects the phase-error detector.
    current_phase_estimate:
        Initial phase estimate in radians.
    current_frequency_offset:
        Initial integrator state in rad/symbol.

    Returns
    -------
    corrected_symbols : np.ndarray[complex64]
    phase_estimates   : np.ndarray[float32]
    """
    return modulator.costas_loop(
        symbols,
        config.alpha,
        config.beta,
        current_phase_estimate,
        current_frequency_offset,
    )
