"""Costas loop for blind carrier phase synchronization.

This module provides a Costas loop implementation for recovering carrier phase
from a modulated signal without the need for pilot symbols. It is a form of
phase-locked loop (PLL) that is "decision-directed," meaning it uses the
output of a symbol slicer to derive its phase error estimate.
"""

import numpy as np
from dataclasses import dataclass
from modules.modulators.modulators import BPSK, PSK8, QPSK, Modulator
from modules.frame_constructor.frame_constructor import ModulationSchemes

import logging
logger = logging.getLogger(__name__)

try:
    from modules.costas_loop import costas_ext as _ext
    logger.info("Loaded costas_ext pybind11 C++ extension.")
except ImportError:
    _ext = None
    logger.warning(
        "costas_ext not found — falling back to pure-Python implementation. "
        "Build it with: uv run python setup.py build_ext --inplace"
    )


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
        out_syms[i] = c
        out_phase[i] = phase
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
        out_syms[i] = c
        out_phase[i] = phase
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
        out_syms[i] = c
        out_phase[i] = phase
    return out_syms, out_phase


def _16psk_py(symbols, alpha, beta, phase, integrator):
    n = len(symbols)
    out_syms  = np.empty(n, dtype=np.complex64)
    out_phase = np.empty(n, dtype=np.float32)
    for i, s in enumerate(symbols):
        c  = s * np.exp(-1j * phase)
        e  = np.angle(c ** 16) / 16.0
        integrator += beta  * e
        phase      += alpha * e + integrator
        phase       = (phase + np.pi) % (2 * np.pi) - np.pi
        out_syms[i] = c
        out_phase[i] = phase
    return out_syms, out_phase


_func_map = {
    ModulationSchemes.BPSK:  _ext.costas_loop_bpsk  if _ext else _bpsk_py,
    ModulationSchemes.QPSK:  _ext.costas_loop_qpsk  if _ext else _qpsk_py,
    ModulationSchemes.PSK8:  _ext.costas_loop_8psk  if _ext else _8psk_py,
    ModulationSchemes.PSK16: _ext.costas_loop_16psk if _ext else _16psk_py,
}

def _loop_parameters(bn: float, zeta: float) -> tuple[float, float]:
    wn    = bn / (zeta + 1.0 / (4.0 * zeta))
    denom = 1.0 + 2.0 * zeta * wn + wn ** 2
    return (4.0 * zeta * wn) / denom, (4.0 * wn ** 2) / denom


@dataclass(frozen=True)
class CostasConfig:
    """Second-order Costas loop configuration."""

    loop_noise_bandwidth_normalized: float = 0.01
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


def apply_costas_loop(
    symbols: np.ndarray,
    config: CostasConfig,
    modulator: ModulationSchemes,
    current_phase_estimate: np.float32 = np.float32(0.0),
    current_frequency_offset: np.float32 = np.float32(0.0),
) -> tuple[np.ndarray, np.ndarray]:
    func = _func_map.get(modulator)
    if func is None:
        raise NotImplementedError(
            f"No Costas loop implemented for {modulator.__name__}"
        )

    return func(
        np.asarray(symbols, dtype=np.complex64),
        float(config.alpha),
        float(config.beta),
        float(current_phase_estimate),
        float(current_frequency_offset),
    )
