import numpy as np
from hypothesis import given, strategies as st

from modules.costas_loop.costas import apply_costas_loop
from modules.modulators.modulators import BPSK, QPSK, PSK8, PSK16, Modulator
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pipeline import PipelineConfig

NUM_SYMBOLS   = 10000
COSTAS_CONFIG = PipelineConfig.COSTAS_CONFIG


def _run(modulator: Modulator, mod_scheme: ModulationSchemes,
         phase_offset: float, phase_drift: float) -> float:
    rng   = np.random.default_rng(42)
    bits  = rng.integers(0, 2, size=NUM_SYMBOLS * modulator.bits_per_symbol).reshape(-1, modulator.bits_per_symbol)
    syms  = modulator.bits2symbols(bits)
    drift = np.linspace(0, phase_drift, NUM_SYMBOLS)
    noisy = syms * np.exp(1j * (phase_offset + drift))

    corrected, _ = apply_costas_loop(noisy, COSTAS_CONFIG, mod_scheme)
    return float(np.mean(modulator.symbols2bits(corrected) != bits))


@given(
    phase_offset=st.floats(min_value=-np.pi / 3, max_value=np.pi / 3),
    phase_drift=st.floats(min_value=-np.pi / 5, max_value=np.pi / 5),
)
def test_costas(phase_offset, phase_drift):
    # Higher-order modulations have tighter decision regions, so scale the
    # offset down proportionally to keep the test in the lockable range.
    cases = [
        (BPSK(),  ModulationSchemes.BPSK,  phase_offset),
        (QPSK(),  ModulationSchemes.QPSK,  phase_offset / 2),
        (PSK8(),  ModulationSchemes.PSK8,  phase_offset / 4),
        (PSK16(), ModulationSchemes.PSK16, phase_offset / 8),
    ]
    for mod, scheme, offset in cases:
        ber = _run(mod, scheme, offset, phase_drift)
        assert ber == 0, f"{scheme.name}: BER={ber}"
