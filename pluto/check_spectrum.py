"""Capture a raw RX buffer and plot the power spectral density.

Verifies that the hardware RRC filter is active: a properly filtered signal
should show a raised-cosine spectrum with bandwidth ≈ fs/SPS * (1 + alpha)
≈ 667 kHz * 1.35 ≈ 900 kHz, and sharp roll-off outside that.
A wideband (unfiltered) signal would fill the full 5.336 MHz span.

Usage (run on the radiotester, with Pluto connected):
    python pluto/check_spectrum.py [--ip 192.168.2.1] [--out spectrum.png]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import adi

from pluto.config import CENTER_FREQ, SAMPLE_RATE, configure_rx

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ip",  default="ip:localhost", help="Pluto URI (default: ip:localhost)")
parser.add_argument("--out", default="pluto/plots/spectrum.png", help="Output PNG path")
args = parser.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

sdr = adi.Pluto(args.ip)
configure_rx(sdr, freq=CENTER_FREQ, gain_mode="slow_attack")
sdr.rx_buffer_size = 2**14

# Flush then capture
for _ in range(5):
    sdr.rx()
rx = sdr.rx().astype(np.complex64)

# Welch PSD via FFT averaging
N = len(rx)
nfft = 1024
n_segs = N // nfft
psd = np.zeros(nfft)
window = np.hanning(nfft).astype(np.float32)
for k in range(n_segs):
    seg = rx[k * nfft:(k + 1) * nfft] * window
    psd += np.abs(np.fft.fftshift(np.fft.fft(seg, nfft))) ** 2
psd /= n_segs
psd_db = 10 * np.log10(psd + 1e-20)
freqs_khz = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / SAMPLE_RATE)) / 1e3

# Expected RRC bandwidth markers
from modules.pipeline import PipelineConfig
cfg = PipelineConfig()
symbol_rate_khz = SAMPLE_RATE / cfg.SPS / 1e3
bw_khz = symbol_rate_khz * (1 + cfg.RRC_ALPHA) / 2  # one-sided

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs_khz, psd_db, linewidth=0.8)
ax.axvline( bw_khz, color="r", linestyle="--", linewidth=1, label=f"Expected RRC edge ±{bw_khz:.0f} kHz")
ax.axvline(-bw_khz, color="r", linestyle="--", linewidth=1)
ax.set_xlabel("Frequency offset (kHz)")
ax.set_ylabel("Power (dB, relative)")
ax.set_title(f"RX spectrum  |  fc={CENTER_FREQ/1e9:.3f} GHz  |  fs={SAMPLE_RATE/1e6:.3f} MHz")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-SAMPLE_RATE / 2e3, SAMPLE_RATE / 2e3)

plt.tight_layout()
plt.savefig(args.out, dpi=150)
print(f"Saved spectrum to {args.out}")
print(f"Expected RRC bandwidth: ±{bw_khz:.0f} kHz  (total {2*bw_khz:.0f} kHz)")
print(f"Full capture bandwidth: ±{SAMPLE_RATE/2e3:.0f} kHz")
print("If hardware RRC is active the signal should be confined within the red markers.")
