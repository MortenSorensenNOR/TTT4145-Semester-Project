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

import numpy as np

from modules.modulation import BPSK

logger = logging.getLogger(__name__)


def _calculate_loop_parameters(
    loop_noise_bandwidth_normalized: float = 0.01,  # Normalized to symbol rate
    damping_factor: float = 0.707,  # zeta 1/sqrt(2)
    initial_freq_offset_rad_per_symbol: float = 0.0,
) -> tuple[float, float]:

    wn_normalized = loop_noise_bandwidth_normalized / (damping_factor + 1 / (4 * damping_factor))
    alpha = (4 * damping_factor * wn_normalized) / (1 + 2 * damping_factor * wn_normalized + wn_normalized**2)
    beta = (4 * wn_normalized**2) / (1 + 2 * damping_factor * wn_normalized + wn_normalized**2)

    return alpha, beta


@dataclass(frozen=True)
class CostasConfig:
    """Configuration for Costas loop phase estimation."""

    loop_noise_bandwidth_normalized: float = 0.01  # Normalized to symbol rate
    damping_factor: float = 0.707  # zeta 1/sqrt(2)
    initial_freq_offset_rad_per_symbol: float = 0.0

    alpha, beta = _calculate_loop_parameters(
        loop_noise_bandwidth_normalized, damping_factor, initial_freq_offset_rad_per_symbol,
    )


def _costas_loop_iteration(
    current_symbol: complex,
    phase_estimate: float,
    integrator: float,
    alpha: float,
    beta: float,
) -> tuple[complex, float, float]:
    """Performs a single iteration of the Costas loop.

    Args:
        current_symbol: The incoming complex-valued symbol.
        modulator: The modulator object, used for the decision-slicer.
        phase_estimate: The current phase estimate (rad).
        integrator: The current state of the loop filter's integrator.
        alpha: The proportional gain of the PI loop filter.
        beta: The integral gain of the PI loop filter.

    Returns:
        A tuple containing:
        - corrected_symbol: The phase-corrected symbol.
        - new_phase_estimate: The updated phase estimate.
        - new_integrator: The updated integrator state.

    """
    # 1. Correct phase of the current symbol
    corrected_sym = current_symbol * np.exp(-1j * phase_estimate)

    # 2. Make a hard decision (slice) on the corrected symbol

    # decision = modulator.symbol_mapping[np.argmin(np.abs(corrected_sym - modulator.symbol_mapping))]
    # logger.debug(f"Costas Iteration: decision = {decision}")

    # 3. Calculate the phase error (unified for BPSK/QPSK)
    error = np.imag(corrected_sym) * np.sign(np.real(corrected_sym))

    # 4. Update the loop filter (PI controller)
    integrator += beta * error
    proportional = alpha * error

    # 5. Update the phase estimate for the next symbol
    new_phase_estimate = phase_estimate + proportional + integrator
    # Wrap phase estimate to -pi to pi for consistent plotting and analysis
    new_phase_estimate = (new_phase_estimate + np.pi) % (2 * np.pi) - np.pi
    return corrected_sym, new_phase_estimate, integrator


def apply_costas_loop(
    symbols: np.ndarray, config: CostasConfig, current_phase_estimate: float = 0, current_frequency_offset: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a second-order Costas loop to correct carrier phase offset.

    This function implements a digital Proportional-Plus-Integrator (PI)
    controller to track and correct the phase of the incoming symbols. The
    controller gains (alpha, beta) are calculated based on desired loop noise
    bandwidth and damping factor, normalized to the symbol rate.

    The error detector implemented here is unified for BPSK and QPSK modulation.
    It may not work correctly for other modulation schemes like QAM.

    Args:
        symbols: The input array of complex-valued symbols.
        modulator: The modulator object, used for the decision-slicer.
        loop_noise_bandwidth_normalized: The desired loop noise bandwidth,
            normalized to the symbol rate (e.g., 0.01 means 1% of symbol rate).
            This parameter helps set the responsiveness and noise rejection of the loop.
        damping_factor: The damping factor (zeta) of the loop, typically around
            0.707 for optimal transient response.
        initial_freq_offset_rad_per_symbol: Initial guess for the frequency
            offset in radians per symbol. This is used to initialize the
            integrator of the PI loop filter.

    Returns:
        A tuple containing:
        - corrected_symbols: An array of phase-corrected complex-valued symbols.
        - phase_estimates: The history of the phase estimate at each symbol.

    """
    # Calculate alpha and beta from loop_noise_bandwidth_normalized and damping_factor
    # These formulas are derived for a second-order, type-II digital PLL
    # (PI loop filter)
    n = len(symbols)
    phase_estimate = current_phase_estimate  # Initial phase estimate
    integrator = current_frequency_offset  # Initialize integrator with frequency offset guess
    corrected_symbols = np.zeros(n, dtype=complex)
    phase_estimates = np.zeros(n, dtype=float)

    for i, sym in enumerate(symbols):
        corrected_sym, phase_estimate, integrator = _costas_loop_iteration(
            sym, phase_estimate, integrator, config.alpha, config.beta,
        )
        corrected_symbols[i] = corrected_sym
        phase_estimates[i] = phase_estimate

    return corrected_symbols, phase_estimates


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Test parameters
    num_symbols = 1000
    initial_phase_offset_rad = np.pi / 4  # 45 degrees

    costas_config = CostasConfig()
    modulator = BPSK()


    # Generate test symbols
    bits = np.random.randint(0, 2, size=num_symbols * modulator.bits_per_symbol)
    base_symbols = modulator.bits2symbols(bits)
    phase_noise = np.linspace(0, 3, len(base_symbols))

    # Apply a constant phase offset
    # The actual phase that the loop should converge to is initial_phase_offset_rad

    input_symbols = base_symbols * np.exp(1j * (initial_phase_offset_rad + phase_noise))

    # Apply Costas loop
    corrected_symbols, phase_estimates = apply_costas_loop(symbols=input_symbols, config=costas_config)

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
    plt.grid(True)
    plt.legend()
    plt.show()

    # Verify if it converged
    final_phase_error = np.degrees(initial_phase_offset_rad - phase_estimates[-1])
    if abs(final_phase_error) < 5:  # Arbitrary small tolerance for this demo
        pass
    else:
        pass
