"""Costas loop for blind carrier phase synchronization.

This module provides a Costas loop implementation for recovering carrier phase
from a modulated signal without the need for pilot symbols. It is a form of
phase-locked loop (PLL) that is "decision-directed," meaning it uses the
output of a symbol slicer to derive its phase error estimate.

This module can be accelerated by compiling the C implementation in `costas.c`.
From the project root, run the following command:
    gcc -shared -o modules/costas.so -fPIC modules/costas.c -lm
"""

import ctypes
import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from matplotlib import figure

from modules.modulation import BPSK, EightPSK, Modulator, QPSK

logger = logging.getLogger(__name__)

# --- Ctypes Foreign Function Interface ---

_costas_lib = None
_c_func_map = {}

try:
    lib_path = os.path.join(os.path.dirname(__file__), "costas.so")
    if os.path.exists(lib_path):
        _costas_lib = ctypes.CDLL(lib_path)
        logger.info("Successfully loaded costas.so C extension.")
    else:
        logger.warning("costas.so not found. Falling back to Python implementation.")
except (OSError, ImportError) as e:
    logger.warning(
        f"Failed to load costas.so C extension: {e}. Falling back to Python implementation."
    )


# --- Pure Python Implementations (Fallback) ---


def _costas_loop_bpsk_py(
    symbols, alpha, beta, phase_estimate, integrator, corrected_symbols, phase_estimates
) -> tuple[float, float]:
    for i, sym in enumerate(symbols):
        corrected_sym = sym * np.exp(-1j * phase_estimate)
        error = np.imag(corrected_sym) * np.sign(np.real(corrected_sym))
        integrator += beta * error
        proportional = alpha * error
        phase_estimate += proportional + integrator
        phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi
        corrected_symbols[i] = corrected_sym
        phase_estimates[i] = phase_estimate
    return phase_estimate, integrator


def _costas_loop_qpsk_py(
    symbols, alpha, beta, phase_estimate, integrator, corrected_symbols, phase_estimates
) -> tuple[float, float]:
    for i, sym in enumerate(symbols):
        corrected_sym = sym * np.exp(-1j * phase_estimate)
        error = np.imag(corrected_sym) * np.sign(
            np.real(corrected_sym)
        ) - np.real(corrected_sym) * np.sign(np.imag(corrected_sym))
        integrator += beta * error
        proportional = alpha * error
        phase_estimate += proportional + integrator
        phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi
        corrected_symbols[i] = corrected_sym
        phase_estimates[i] = phase_estimate
    return phase_estimate, integrator


def _costas_loop_8psk_py(
    symbols, alpha, beta, phase_estimate, integrator, corrected_symbols, phase_estimates
) -> tuple[float, float]:
    for i, sym in enumerate(symbols):
        corrected_sym = sym * np.exp(-1j * phase_estimate)
        error = np.angle(corrected_sym**8) / 8.0
        integrator += beta * error
        proportional = alpha * error
        phase_estimate += proportional + integrator
        phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi
        corrected_symbols[i] = corrected_sym
        phase_estimates[i] = phase_estimate
    return phase_estimate, integrator


# --- C Function Wrappers ---


def _create_c_wrapper(c_func: Callable) -> Callable:
    """Creates a Python wrapper for a C function to match the Python fallback signature."""

    def wrapper(
        symbols,
        alpha,
        beta,
        phase_estimate,
        integrator,
        corrected_symbols,
        phase_estimates,
    ) -> tuple[float, float]:
        n_symbols = len(symbols)
        symbols_contig = np.ascontiguousarray(symbols, dtype=np.complex64)
        phase_ptr = ctypes.c_float(phase_estimate)
        freq_ptr = ctypes.c_float(integrator)

        c_func(
            symbols_contig,
            n_symbols,
            alpha,
            beta,
            ctypes.byref(phase_ptr),
            ctypes.byref(freq_ptr),
            corrected_symbols,
            phase_estimates,
        )
        return phase_ptr.value, freq_ptr.value

    return wrapper


if _costas_lib:
    # Define the ctypes interface and create the wrappers
    for name in ["bpsk", "qpsk", "8psk"]:
        func_name = f"costas_loop_{name}"
        c_func = getattr(_costas_lib, func_name)
        c_func.restype = None
        c_func.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.complex64, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            np.ctypeslib.ndpointer(dtype=np.complex64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        _c_func_map[name] = _create_c_wrapper(c_func)

# Map modulator types to the correct function key
_mod_map = {BPSK: "bpsk", QPSK: "qpsk", EightPSK: "8psk"}

# --- Main Logic ---


def _calculate_loop_parameters(
    loop_noise_bandwidth_normalized: float = 0.05,
    damping_factor: float = 0.707,
) -> tuple[float, float]:
    wn_normalized = loop_noise_bandwidth_normalized / (
        damping_factor + 1 / (4 * damping_factor)
    )
    alpha = (4 * damping_factor * wn_normalized) / (
        1 + 2 * damping_factor * wn_normalized + wn_normalized**2
    )
    beta = (4 * wn_normalized**2) / (
        1 + 2 * damping_factor * wn_normalized + wn_normalized**2
    )
    return alpha, beta


@dataclass(frozen=True)
class CostasConfig:
    """Configuration for Costas loop phase estimation."""

    loop_noise_bandwidth_normalized: float = 0.05
    damping_factor: float = 0.707
    initial_freq_offset_rad_per_symbol: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Compute and set loop filter gains from bandwidth and damping factor."""
        a, b = _calculate_loop_parameters(
            self.loop_noise_bandwidth_normalized, self.damping_factor
        )
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)


def apply_costas_loop(
    symbols: np.ndarray,
    config: CostasConfig,
    modulator: Modulator = BPSK(),
    current_phase_estimate: float = 0.0,
    current_frequency_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a second-order Costas loop to correct carrier phase offset."""
    n = len(symbols)
    corrected_symbols = np.zeros(n, dtype=np.complex64)
    phase_estimates = np.zeros(n, dtype=np.float32)
    alpha, beta = np.float32(config.alpha), np.float32(config.beta)

    key = _mod_map.get(type(modulator))
    if not key:
        raise NotImplementedError(
            f"Costas loop not implemented for modulator: {type(modulator).__name__}"
        )

    # Choose C wrapper if available, otherwise Python fallback
    loop_func = _c_func_map.get(key) or _py_func_map.get(key)
    
    # Signatures are now identical, so the call is unified
    loop_func(
        symbols,
        alpha,
        beta,
        current_phase_estimate,
        current_frequency_offset,
        corrected_symbols,
        phase_estimates,
    )

    return corrected_symbols, phase_estimates


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test parameters
    num_symbols = 1000
    initial_phase_offset_rad = np.pi / 8
    modulator = QPSK()
    config = CostasConfig(loop_noise_bandwidth_normalized=0.01)

    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, size=num_symbols * modulator.bits_per_symbol)
    base_symbols = modulator.bits2symbols(bits)
    phase_noise = np.linspace(0, np.pi / 4, num_symbols)

    input_symbols = base_symbols * np.exp(1j * (initial_phase_offset_rad + phase_noise))

    start = time.perf_counter()
    corrected, phase_est = apply_costas_loop(
        symbols=input_symbols, config=config, modulator=modulator
    )
    duration = (time.perf_counter() - start) * 1e3
    implementation_type = "C" if _c_func_map else "Python"
    print(f"Costas loop ({implementation_type}) took: {duration:.4f} ms")


    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.degrees(phase_est),
        label=f"Estimated Phase ({implementation_type})",
        linewidth=2,
    )
    plt.plot(
        np.degrees(initial_phase_offset_rad + phase_noise),
        label="Actual Phase",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Costas Loop Phase Tracking")
    plt.xlabel("Symbol Index")
    plt.ylabel("Phase (degrees)")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("examples/data/costas_phase_trajectories.png")
    print("Saved plot to examples/data/costas_phase_trajectories.png")

_py_func_map = {
    "bpsk": _costas_loop_bpsk_py,
    "qpsk": _costas_loop_qpsk_py,
    "8psk": _costas_loop_8psk_py,
}
