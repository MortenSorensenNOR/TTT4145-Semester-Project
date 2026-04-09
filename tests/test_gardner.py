from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from modules.gardner_ted.gardner import apply_gardner_ted, GardnerConfig
from modules.modulators import BPSK, QPSK, PSK8, Modulator
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pulse_shaping import decimate, rrc_filter
from modules.pipeline import TXPipeline, PipelineConfig
from modules.costas_loop.costas import CostasConfig, apply_costas_loop

"""
fractional_delay.py
-------------------
Apply a fractional sample delay to a complex baseband signal for testing
symbol timing recovery (e.g. the Gardner TED).

Two methods
-----------
fractional_delay_fd(x, delay_samples)
    Frequency-domain phase rotation — the recommended default.
    Exact for strictly bandlimited signals, no filter design required.
    Handles any real-valued delay (positive = late, negative = early).

fractional_delay_farrow(x, delay_samples)
    Cubic Farrow interpolation — same filter used inside the Gardner TED.
    Useful when you want the injected delay to be exactly recoverable by
    the TED's own interpolator.  Less accurate on wideband/non-Nyquist signals.

apply_timing_offset(x, offset_symbols, sps, method='fd')
    Convenience wrapper: express the delay in symbol-period units (same
    units as Gardner's mu), so you can directly compare the injected offset
    against mu after the TED converges.

Sign convention
---------------
A *positive* offset_symbols means the received signal arrives *late*
(the receiver's sampling clock is early relative to the transmitter).
After the Gardner TED converges, mu ≈ -offset_symbols.

Usage
-----
    from fractional_delay import apply_timing_offset

    # Inject 0.3 symbol-period delay at sps=8
    x_delayed = apply_timing_offset(x, offset_symbols=0.3, sps=8)

    # Run Gardner TED — mu should converge to ≈ -0.3
    symbols, mu = apply_gardner_ted(x_delayed, config, ModulationSchemes.QPSK, sps=8)
    assert abs(mu[-1] + 0.3) < 0.02
"""


import math


# ---------------------------------------------------------------------------
# Method 1: frequency-domain fractional delay (recommended)
# ---------------------------------------------------------------------------

def fractional_delay_fd(
    x: np.ndarray,
    delay_samples: float,
) -> np.ndarray:

    x = np.asarray(x, dtype=np.complex64)
    N = len(x)
    f = np.fft.fftfreq(N).astype(np.float64)
    H = np.exp(-1j * 2.0 * np.pi * f * float(delay_samples))
    return np.fft.ifft(np.fft.fft(x.astype(np.complex128)) * H).astype(np.complex64)


# ---------------------------------------------------------------------------
# Method 2: cubic Farrow interpolation (matches Gardner TED internals)
# ---------------------------------------------------------------------------

def _xget(x: np.ndarray, idx: int) -> complex:
    """Clamp-boundary sample access."""
    return complex(x[max(0, min(idx, len(x) - 1))])


def _farrow_interp(xm1: complex, x0: complex, x1: complex, x2: complex, eta: float) -> complex:
    """4-tap cubic Farrow interpolation.

    eta = 0 → x0,  eta → 1 → x1.
    Identical to the filter used inside gardner_ext.cpp.
    """
    c0 =  x0
    c1 = -(1/6)*xm1 - 0.5*x0 +       x1 - (1/3)*x2
    c2 =   0.5 *xm1 -     x0 + 0.5*x1
    c3 = -(1/6)*xm1 + 0.5*x0 - 0.5*x1 + (1/6)*x2
    return c0 + eta * (c1 + eta * (c2 + eta * c3))


def fractional_delay_farrow(
    x: np.ndarray,
    delay_samples: float,
) -> np.ndarray:

    x = np.asarray(x, dtype=np.complex64)
    N = len(x)

    # y[n] = x[n - delay]
    # Source position: p = n - delay
    # Integer part: floor(p), fractional part: eta = p - floor(p) in [0,1)
    out = np.empty(N, dtype=np.complex64)
    for n in range(N):
        p       = float(n) - float(delay_samples)
        floor_p = int(math.floor(p))
        eta     = p - floor_p
        out[n]  = _farrow_interp(
            _xget(x, floor_p - 1),
            _xget(x, floor_p),
            _xget(x, floor_p + 1),
            _xget(x, floor_p + 2),
            eta,
        )

    return out


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def apply_timing_offset(
    x: np.ndarray,
    offset_symbols: float,
    sps: int,
    method: str = "fd",
) -> np.ndarray:

    delay_samples = float(offset_symbols) * float(sps)

    if method == "fd":
        return fractional_delay_fd(x, delay_samples)
    elif method == "farrow":
        return fractional_delay_farrow(x, delay_samples)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'fd' or 'farrow'.")


MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]#, ModulationSchemes.PSK8]

def generate_syms(n_symbols, mod_scheme, modulator: Modulator):
    
    rng = np.random.default_rng(42)
    bits = rng.integers(0, 2, n_symbols*(mod_scheme.value+1)).reshape(-1, mod_scheme.value+1)

    return modulator.bits2symbols(bits), bits

# -------------------------------
# Apply fractional delay
# -------------------------------
def fractional_delay(x, delay):
    n = np.arange(len(x))
    return np.interp(n - delay, n, x.real) + 1j*np.interp(n - delay, n, x.imag)

# -------------------------------
# Test pipeline
# -------------------------------
def run_test(sps=4, gain=0.005, timing_offset=0.3, snr_db=30, mod_scheme: ModulationSchemes=ModulationSchemes.BPSK):
    n_symbols = 2000
    
    tx = TXPipeline(PipelineConfig(MOD_SCHEME=mod_scheme))

    match (mod_scheme):
        case ModulationSchemes.BPSK:
            modulator = BPSK()
        case ModulationSchemes.QPSK:
            modulator = QPSK()
        case ModulationSchemes.PSK8:
            modulator = PSK8()
    
    # Generate symbols
    syms, bits = generate_syms(n_symbols, mod_scheme, modulator)

    # Upsample
    up = np.zeros(n_symbols * sps, dtype=np.complex64)
    up[::sps] = syms

    # Pulse shape
    h = rrc_filter(sps, 0.25, 2*8*8+1)
    tx = np.convolve(up, h, mode='same')

    # Apply timing offset
    rx = apply_timing_offset(tx, timing_offset, sps)

    # Add noise
    noise = (np.random.randn(len(rx)) + 1j*np.random.randn(len(rx))) / np.sqrt(2)
    rx += noise * 10**(-snr_db/20)

    rx = rx.astype(np.complex64)

    # -------------------------------
    # Run Gardner
    # -------------------------------
    out_gardner, time_est = apply_gardner_ted(rx, GardnerConfig(gain), mod_scheme, sps)

    out_gardner, _ = apply_costas_loop(out_gardner, CostasConfig(0.07), mod_scheme)

    out_decimate = decimate(rx, sps)
    out_decimate, _ = apply_costas_loop(out_decimate, CostasConfig(0.07), mod_scheme)

    # -------------------------------
    # Plot results
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.title("Before (raw samples)")
    plt.plot(rx.real[:200], label="I")
    plt.plot(rx.imag[:200], label="Q")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title("After Gardner (constellation)")
    plt.scatter(out_decimate.real, out_decimate.imag, s=5, label="no gardner")
    plt.scatter(out_gardner.real, out_gardner.imag, s=5, label="gardner")
    plt.legend()
    plt.axhline(0)
    plt.axvline(0)

    plt.tight_layout()
    plt.savefig(f"tests/plots/gardner/{snr_db}_{gain}_{mod_scheme.value + 1}.png")

    print("Output symbols:", len(out_gardner), len(out_decimate))
    
    bits_rec_gardner = modulator.symbols2bits(out_gardner)
    bits_rec_decimate = modulator.symbols2bits(out_decimate)
    print("BER:", np.mean(bits != bits_rec_gardner), np.mean(bits != bits_rec_decimate))

# -------------------------------
# Run multiple gains
# -------------------------------
if __name__ == "__main__":
    for gain in [0.005, 0.003, 0.002, 0.0025, 0.0005, 0.001]:
        print("\nTesting gain =", gain)
        for scheme in MOD_SCHEMES:
            run_test(sps=8, gain=gain, mod_scheme=scheme, timing_offset=0.1, snr_db=20)


