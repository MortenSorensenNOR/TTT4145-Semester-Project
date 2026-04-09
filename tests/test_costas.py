"""Quick smoke-test and visual demo for the Costas loop extension.

Run from the project root with:
    uv run python modules/costas_loop/test_costas.py
"""

import time
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import composite

import matplotlib.pyplot as plt
import numpy as np

from modules.costas_loop.costas import CostasConfig, apply_costas_loop, _ext
from modules.modulators import BPSK, QPSK, PSK8, Modulator
from modules.frame_constructor.frame_constructor import ModulationSchemes

@composite
def random_phase_offset(draw):
    return draw(st.floats(min_value=-np.pi/3, max_value=np.pi/3))

@composite
def random_phase_drift(draw):
    return draw(st.floats(min_value=-np.pi/5, max_value=np.pi/5))

@given(random_phase_offset = random_phase_offset(), random_phase_drift = random_phase_drift())
def test_costas(random_phase_offset, random_phase_drift):
    run(BPSK(),     "BPSK", ModulationSchemes.BPSK, running_pytest=True, random_phase_offset=random_phase_offset, random_phase_drift=random_phase_drift)
    run(QPSK(),     "QPSK", ModulationSchemes.QPSK, running_pytest=True, random_phase_offset=random_phase_offset/2, random_phase_drift=random_phase_drift)
    run(PSK8(),     "8PSK", ModulationSchemes.PSK8, running_pytest=True, random_phase_offset=random_phase_offset/4, random_phase_drift=random_phase_drift)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

NUM_SYMBOLS          = 10000
INITIAL_PHASE_OFFSET = np.pi / 8
PHASE_DRIFT          = np.pi / 5   # total drift over all symbols
BN                   = 0.07        # loop noise bandwidth

impl = "C++ (pybind11)" if _ext else "Python (fallback)"
print(f"Implementation : {impl}\n")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(modulator: Modulator, label, modulation_scheme, running_pytest=False, random_phase_offset = INITIAL_PHASE_OFFSET, random_phase_drift = PHASE_DRIFT):
    rng   = np.random.default_rng(42)
    bits  = rng.integers(0, 2, size=NUM_SYMBOLS * modulator.bits_per_symbol).reshape(-1, modulation_scheme.value+1)
    syms  = modulator.bits2symbols(bits)
    drift = np.linspace(0, random_phase_drift, NUM_SYMBOLS)
    noisy = syms * np.exp(1j * (random_phase_offset + drift))

    config = CostasConfig(loop_noise_bandwidth_normalized=BN)

    t0 = time.perf_counter()
    corrected, phase_est = apply_costas_loop(noisy, config, modulation_scheme)
    ms = (time.perf_counter() - t0) * 1e3

    residual_phase_error = np.degrees(np.mean(np.abs(phase_est[-1000:] - drift[-1000:] - random_phase_offset)))

    print(f"{label:8s}  {ms:.4f} ms   residual phase error: "
          f"{residual_phase_error:.3f} deg")

    corrected_bits = modulator.symbols2bits(corrected)

    ber = np.mean(corrected_bits != bits)
    print(modulator, ber)
    
    if running_pytest:
        assert ber == 0
        #assert residual_phase_error <= 0.35

    return corrected, phase_est, drift


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Run all three modulators
    # ---------------------------------------------------------------------------

    bpsk_corr, bpsk_phase, bpsk_drift = run(BPSK(),     "BPSK", ModulationSchemes.BPSK)
    qpsk_corr, qpsk_phase, qpsk_drift = run(QPSK(),     "QPSK", ModulationSchemes.QPSK)
    psk8_corr, psk8_phase, psk8_drift = run(PSK8(),     "8PSK", ModulationSchemes.PSK8)

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Costas Loop Demo  [{impl}]", fontsize=14)

    for ax, corr, phase_est, drift, label in zip(
        axes[0],
        [bpsk_corr, qpsk_corr, psk8_corr],
        [bpsk_phase, qpsk_phase, psk8_phase],
        [bpsk_drift, qpsk_drift, psk8_drift],
        ["BPSK", "QPSK", "8PSK"],
    ):
        ax.set_title(f"{label} — phase tracking")
        ax.plot(np.degrees(phase_est),
                label="Estimated", linewidth=1.5)
        ax.plot(np.degrees(INITIAL_PHASE_OFFSET + drift),
                label="Actual", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Symbol index")
        ax.set_ylabel("Phase (degrees)")
        ax.legend(fontsize=8)
        ax.grid(True)

    for ax, corr, label in zip(
        axes[1],
        [bpsk_corr, qpsk_corr, psk8_corr],
        ["BPSK", "QPSK", "8PSK"],
    ):
        ax.set_title(f"{label} — corrected constellation")
        ax.scatter(corr.real, corr.imag, s=4, alpha=0.4)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_aspect("equal")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("tests/plots/costas_phase_trajectories.png", dpi=150)
    print("\nSaved plot to costas_phase_trajectories.png")
