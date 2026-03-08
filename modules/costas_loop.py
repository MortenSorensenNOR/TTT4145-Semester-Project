"""Costas loop for blind carrier phase synchronization.

This module provides a Costas loop implementation for recovering carrier phase
from a modulated signal without the need for pilot symbols. It is a form of
phase-locked loop (PLL) that is "decision-directed," meaning it uses the
output of a symbol slicer to derive its phase error estimate.

The implementation is specifically designed for QPSK modulation but can be
extended to other M-PSK schemes.
"""

from dataclasses import dataclass

import numpy as np

from modules.modulation import BPSK, QPSK, EightPSK, Modulator


def _calculate_loop_parameters(
    loop_noise_bandwidth_normalized: float = 0.05,  # Normalized to symbol rate
    damping_factor: float = 0.707,  # zeta 1/sqrt(2)
) -> tuple[float, float]:

    wn_normalized = loop_noise_bandwidth_normalized / (damping_factor + 1 / (4 * damping_factor))
    alpha = (4 * damping_factor * wn_normalized) / (1 + 2 * damping_factor * wn_normalized + wn_normalized**2)
    beta = (4 * wn_normalized**2) / (1 + 2 * damping_factor * wn_normalized + wn_normalized**2)

    return alpha, beta


@dataclass(frozen=True)
class CostasConfig:
    """Configuration for Costas loop phase estimation."""

    loop_noise_bandwidth_normalized: float = 0.05  # Normalized to symbol rate
    damping_factor: float = 0.707  # zeta 1/sqrt(2)
    initial_freq_offset_rad_per_symbol: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Compute and set loop filter gains from bandwidth and damping factor."""
        a, b = _calculate_loop_parameters(self.loop_noise_bandwidth_normalized, self.damping_factor)
        object.__setattr__(self, "alpha", a)
        object.__setattr__(self, "beta", b)


_DEFAULT_BPSK = BPSK()


def apply_costas_loop(
    symbols: np.ndarray,
    config: CostasConfig,
    modulator: Modulator = _DEFAULT_BPSK,
    current_phase_estimate: float = 0,
    current_frequency_offset: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
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
        modulator: Modulation scheme instance (default: BPSK).
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
    phase_estimate = current_phase_estimate  # Initial phase estimate
    integrator = current_frequency_offset  # Initialize integrator with frequency offset guess
    corrected_symbols = np.zeros(n, dtype=complex)
    phase_estimates = np.zeros(n, dtype=float)

    alpha, beta = config.alpha, config.beta

    if isinstance(modulator, BPSK):
        for i, sym in enumerate(symbols):
            # 1. Correct phase of the current symbol
            corrected_sym = sym * np.exp(-1j * phase_estimate)

            # 2. Calculate the phase error (unified for BPSK)
            error = np.imag(corrected_sym) * np.sign(np.real(corrected_sym))

            # 3. Update the loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update the phase estimate for the next symbol
            phase_estimate += proportional + integrator

            # 5. Wrap phase estimate to -pi to pi for consistent plotting and analysis
            phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi

            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    if isinstance(modulator, QPSK):
        for i, sym in enumerate(symbols):
            # 1. Correct phase of the current symbol
            corrected_sym = sym * np.exp(-1j * phase_estimate)

            # 2. Calculate the phase error (unified for QPSK)

            error = -(np.real(corrected_sym) * np.sign(np.imag(corrected_sym))) + (
                np.imag(corrected_sym) * np.sign(np.real(corrected_sym))
            )
            # 3. Update the loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update the phase estimate for the next symbol
            phase_estimate += proportional + integrator

            # 5. Wrap phase estimate to -pi to pi for consistent plotting and analysis
            phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi

            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    if isinstance(modulator, EightPSK):
        for i, sym in enumerate(symbols):
            # 1. Phase correction
            corrected_sym = sym * np.exp(-1j * phase_estimate)

            # 2. 8PSK phase error detector (Mth-power method)
            error = np.angle(corrected_sym**8) / 8.0

            # 3. Loop filter (PI controller)
            integrator += beta * error
            proportional = alpha * error

            # 4. Update phase estimate
            phase_estimate += proportional + integrator

            # 5. Wrap phase
            phase_estimate = (phase_estimate + np.pi) % (2 * np.pi) - np.pi

            corrected_symbols[i] = corrected_sym
            phase_estimates[i] = phase_estimate

    return corrected_symbols, phase_estimates
