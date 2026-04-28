"""Compare oaconvolve vs fftconvolve vs direct FFT for the detect xcorr."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import oaconvolve, fftconvolve
from modules.pipeline import PipelineConfig, RXPipeline

cfg = PipelineConfig()
rx = RXPipeline(cfg)
buf_dir = Path(__file__).resolve().parents[1] / "pluto" / "rx_buffs"
buf = np.load(buf_dir / "rxbuf_0000.npz")["samples"].astype(np.complex64)

# Match filter the buffer (the input to detect)
from modules.pulse_shaping.pulse_shaping import match_filter
mf = match_filter(buf, rx.rrc_taps)

ref_rev = rx.long_ref_rev


def time_it(fn, n=10):
    fn()  # warm
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


t1 = time_it(lambda: oaconvolve(mf, ref_rev, mode="valid"))
print(f"oaconvolve   : {t1:.2f} ms")

t2 = time_it(lambda: fftconvolve(mf, ref_rev, mode="valid"))
print(f"fftconvolve  : {t2:.2f} ms")

# Direct: pad both, FFT, multiply, IFFT
N = len(mf)
M = len(ref_rev)
out_len = N - M + 1


def direct_fft():
    F1 = np.fft.fft(mf)
    F2 = np.fft.fft(ref_rev, n=N)
    z = np.fft.ifft(F1 * F2)[M - 1:M - 1 + out_len]
    return z.astype(np.complex64)


t3 = time_it(direct_fft)
print(f"direct-fft   : {t3:.2f} ms")

# scipy.fft variants
from scipy.fft import fft, ifft


def scipy_direct():
    F1 = fft(mf)
    F2 = fft(ref_rev, n=N)
    z = ifft(F1 * F2)[M - 1:M - 1 + out_len]
    return z.astype(np.complex64)


t4 = time_it(scipy_direct)
print(f"scipy.fft    : {t4:.2f} ms")

# Verify
z_oa = oaconvolve(mf, ref_rev, mode="valid").astype(np.complex64)
z_dir = direct_fft()
print("max diff (oaconv vs direct):", np.max(np.abs(z_oa - z_dir)))
