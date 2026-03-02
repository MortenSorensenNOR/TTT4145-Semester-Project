"""Costas loop for blind carrier phase synchronization.

This module provides a Costas loop implementation for recovering carrier phase
from a modulated signal without the need for pilot symbols. It is a form of
phase-locked loop (PLL) that is "decision-directed," meaning it uses the
output of a symbol slicer to derive its phase error estimate.

The implementation is specifically designed for QPSK modulation but can be
extended to other M-PSK schemes.
"""

import logging
from dataclasses import dataclass
from typing import Any

from matplotlib import figure
import numpy as np
import time

from modules.modulation import BPSK, Modulator, QPSK, EightPSK

logger = logging.getLogger(__name__)


def _calculate_loop_parameters(
    loop_noise_bandwidth_normalized: np.float32 = np.float32(0.05),  # Normalized to symbol rate
    damping_factor: np.float32 = np.float32(0.707),  # zeta 1/sqrt(2)
) -> tuple[np.float32, np.float32]: # Changed return type hint

    wn_normalized = loop_noise_bandwidth_normalized / (damping_factor + np.float32(1) / (np.float32(4) * damping_factor)) # Explicitly cast to np.float32
    alpha = (np.float32(4) * damping_factor * wn_normalized) / (np.float32(1) + np.float32(2) * damping_factor * wn_normalized + wn_normalized**np.float32(2)) # Explicitly cast to np.float32
    beta = (np.float32(4) * wn_normalized**np.float32(2)) / (np.float32(1) + np.float32(2) * damping_factor * wn_normalized + wn_normalized**np.float32(2)) # Explicitly cast to np.float32

    return alpha, beta


@dataclass(frozen=True)
class CostasConfig:
    """Configuration for Costas loop phase estimation."""

    loop_noise_bandwidth_normalized: np.float32 = np.float32(0.05)  # Normalized to symbol rate # Changed type hint and default
    damping_factor: np.float32 = np.float32(0.707)  # zeta 1/sqrt(2) # Changed type hint and default
    initial_freq_offset_rad_per_symbol: np.float32 = np.float32(0.0) # Changed type hint and default
    alpha: np.float32 = np.float32(0.0) # Changed type hint and default
    beta: np.float32 = np.float32(0.0) # Changed type hint and default

    def __post_init__(self) -> None:
        """Compute and set loop filter gains from bandwidth and damping factor."""
        a, b = _calculate_loop_parameters(self.loop_noise_bandwidth_normalized, self.damping_factor)
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)


def apply_costas_loop(
    symbols: np.ndarray, config: CostasConfig, modulator: Modulator = BPSK(), current_phase_estimate: np.float32 = np.float32(0), current_frequency_offset: np.float32 = np.float32(0), # Changed type hints and defaults
) -> tuple[np.ndarray[np.complex64, Any], np.ndarray[np.float32, Any]]: # Changed return type hint
    """Apply a second-order Costas loop to correct carrier phase offset.

    This function implements a digital Proportional-Plus-Integrator (PI)
    controller to track and correct the phase of the incoming symbols. The
    controller gains (alpha, beta) are calculated from the parameters in the
    `config` object.

    The error detector implemented here is unified for BPSK and QPSK modulation.
    It may not work correctly for other modulation schemes like QAM.

    Args:
        symbols: The input array of complex-valued symbols.

        config: CostasConfig object with loop parameters.
        current_phase_estimate: Initial phase estimate in radians.
        current_frequency_offset: Initial frequency offset in radians per symbol.

    Returns:
        A tuple containing:
        - corrected_symbols: An array of phase-corrected complex-valued symbols.
        - phase_estimates: The history of the phase estimate at each symbol.

    """
    # Calculate alpha and beta from loop_noise_bandwidth_normalized and damping_factor
    # These formulas are derived for a second-order, type-II digital PLL
    # (PI loop filter)
    n = len(symbols)
    phase_estimate = np.float32(current_phase_estimate)  # Initial phase estimate
    integrator = np.float32(current_frequency_offset)  # Initialize integrator with frequency offset guess
    corrected_symbols = np.zeros(n, dtype=np.complex64)
    phase_estimates = np.zeros(n, dtype=np.float32)

    alpha = np.float32(config.alpha)
    beta = np.float32(config.beta)

    if isinstance(modulator, BPSK):
        for i, sym in enumerate(symbols):

            # 1. Correct phase of the current symbol
            corrected_sym = sym * np.exp(np.complex64(-1j) * phase_estimate)

            # 2. Calculate the phase error (unified for BPSK)
            error = np.imag(corrected_sym) * np.sign(np.real(corrected_sym))
            
            # 3. Update the loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update the phase estimate for the next symbol
            phase_estimate += proportional + integrator
            
            # 5. Wrap phase estimate to -pi to pi for consistent plotting and analysis
            phase_estimate = (phase_estimate + np.float32(np.pi)) % (np.float32(2) * np.float32(np.pi)) - np.float32(np.pi)
            
            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    if isinstance(modulator, QPSK):
        for i, sym in enumerate(symbols):

            # 1. Correct phase of the current symbol
            corrected_sym = sym * np.exp(np.complex64(-1j) * phase_estimate)

            # 2. Calculate the phase error (unified for QPSK)
            
            error = -(np.real(corrected_sym)*np.sign(np.imag(corrected_sym)))+(np.imag(corrected_sym) * np.sign(np.real(corrected_sym)))
            # 3. Update the loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update the phase estimate for the next symbol
            phase_estimate += proportional + integrator
            
            # 5. Wrap phase estimate to -pi to pi for consistent plotting and analysis
            phase_estimate = (phase_estimate + np.float32(np.pi)) % (np.float32(2) * np.float32(np.pi)) - np.float32(np.pi)
            
            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    if isinstance(modulator, EightPSK):
        for i, sym in enumerate(symbols):

            # 1. Phase correction
            corrected_sym = sym * np.exp(np.complex64(-1j) * phase_estimate)

            # 2. 8PSK phase error detector (Mth-power method)
            error = np.angle(corrected_sym ** np.float32(8)) / np.float32(8.0) # Explicitly cast to np.float32

            # 3. Loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update phase estimate
            phase_estimate += proportional + integrator

            # 5. Wrap phase
            phase_estimate = (phase_estimate + np.float32(np.pi)) % (np.float32(2) * np.float32(np.pi)) - np.float32(np.pi)

            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    return corrected_symbols, phase_estimates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Test parameters
    num_symbols = 100
    initial_phase_offset_rad = np.float32(np.pi) / np.float32(8)  # 45 degrees # Explicitly cast to np.float32

    costas_config = CostasConfig(loop_noise_bandwidth_normalized=np.float32(0.05)) # Explicitly cast to np.float32
    modulator = EightPSK()


    # Generate test symbols
    rng = np.random.default_rng()
    bits = rng.integers(0, 2, size=num_symbols * modulator.bits_per_symbol)
    base_symbols = modulator.bits2symbols(bits)
    phase_noise = np.linspace(np.float32(0), np.float32(3), len(base_symbols), dtype=np.float32) # Explicitly cast to np.float32

    # Apply a constant phase offset
    # The actual phase that the loop should converge to is initial_phase_offset_rad

    input_symbols = base_symbols * np.exp(np.complex64(1j) * (initial_phase_offset_rad + phase_noise)) # Explicitly cast to np.complex64
    start=time.perf_counter()
    # Apply Costas loop
    corrected_symbols, phase_estimates = apply_costas_loop(symbols=input_symbols, config=costas_config, modulator=modulator)
    print(f"{(time.perf_counter()-start)*np.float32(1e3)} ms") # Explicitly cast to np.float32
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(phase_estimates), label="Estimated Phase (degrees)")
    plt.plot(np.degrees(initial_phase_offset_rad + phase_noise), label="Actual phase")
    plt.axhline(
        np.degrees(initial_phase_offset_rad),
        color="r",
        linestyle="--",
        label="Actual Phase Offset (degrees)",
    )
    plt.title("Costas Loop Phase Tracking Test")
    plt.xlabel("Symbol Index")
    plt.ylabel("Phase (degrees)")
    plt.grid(visible=True)
    plt.legend()
    plt.savefig("examples/data/phase.png")

    # Verify if it converged
    CONVERGENCE_TOLERANCE_DEG = np.float32(5) # Explicitly cast to np.float32
    final_phase_error = np.degrees(initial_phase_offset_rad - phase_estimates[-1])
    if abs(final_phase_error) < CONVERGENCE_TOLERANCE_DEG:
        pass
    else:
        pass