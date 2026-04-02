import numpy as np
import adi
import matplotlib.pyplot as plt
import os

sample_rate = 1e6       # Hz
center_freq = 915e6     # Hz
num_samps = 100000      # number of samples per call to rx()
tone_freq = 100e3       # Hz, baseband tone offset from LO
tx_gain = -10           # dB
rx_gain = 0.0           # dB

# Output directory
os.makedirs("pluto/plots", exist_ok=True)

sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)

# -------------------------
# TX configuration
# -------------------------
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_lo = int(center_freq)
sdr.tx_hardwaregain_chan0 = tx_gain
sdr.tx_cyclic_buffer = True

# -------------------------
# RX configuration
# -------------------------
sdr.rx_lo = int(center_freq)
sdr.rx_rf_bandwidth = int(sample_rate)
sdr.rx_buffer_size = num_samps
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = rx_gain

# -------------------------
# Generate TX sinusoid
# -------------------------
N_tx = num_samps
t = np.arange(N_tx) / sample_rate

# Complex baseband tone
tx_samples = np.exp(1j * 2 * np.pi * tone_freq * t)

# Scale to Pluto range
tx_samples = tx_samples * (2**14 * 0.7)
tx_samples = tx_samples.astype(np.complex64)

# -------------------------
# Start TX
# -------------------------
sdr.tx(tx_samples)

# Flush stale RX buffers
for _ in range(10):
    _ = sdr.rx()

# Receive one buffer
rx_samples = sdr.rx().astype(np.complex64)

# -------------------------
# Align RX to TX
# -------------------------
# Use a shorter reference segment for correlation to keep it robust/fast
N_ref = 4096
tx_ref = tx_samples[:N_ref]
rx_ref = rx_samples[:N_ref * 2]

# Cross-correlate RX against TX reference
corr = np.correlate(rx_ref, tx_ref, mode="full")
lag = np.argmax(np.abs(corr)) - (len(tx_ref) - 1)

# Positive lag means tx_ref starts later in rx_ref
if lag >= 0:
    rx_aligned = rx_samples[lag:]
else:
    # If lag is negative, pad the front so indexing still works
    rx_aligned = np.concatenate([np.zeros(-lag, dtype=rx_samples.dtype), rx_samples])

# Match length to TX
N_common = min(len(tx_samples), len(rx_aligned))
tx_aligned = tx_samples[:N_common]
rx_aligned = rx_aligned[:N_common]

# Estimate and correct residual complex phase/amplitude offset
# Best-fit complex scalar a such that rx ≈ a * tx
a = np.vdot(tx_aligned, rx_aligned) / np.vdot(tx_aligned, tx_aligned)
rx_corrected = rx_aligned / a if a != 0 else rx_aligned

# -------------------------
# Frequency-domain analysis
# -------------------------
psd = np.abs(np.fft.fftshift(np.fft.fft(rx_samples)))**2
psd_dB = 10 * np.log10(psd + 1e-12)
f = np.linspace(-sample_rate / 2, sample_rate / 2, len(psd))

# -------------------------
# Plot 1: RX time domain
# -------------------------
plt.figure(figsize=(10, 4))
plt.plot(np.real(rx_samples[::100]), label="RX I")
plt.plot(np.imag(rx_samples[::100]), label="RX Q")
plt.xlabel("Sample index / 100")
plt.ylabel("Amplitude")
plt.title("Received Signal (Time Domain)")
plt.legend()
plt.tight_layout()
plt.savefig("pluto/plots/rx_time.png", dpi=150)

# -------------------------
# Plot 2: RX PSD
# -------------------------
plt.figure(figsize=(10, 4))
plt.plot(f / 1e6, psd_dB)
plt.xlabel("Frequency [MHz]")
plt.ylabel("PSD [dB]")
plt.title("Received Signal Spectrum")
plt.tight_layout()
plt.savefig("pluto/plots/rx_psd.png", dpi=150)

# -------------------------
# Plot 3: Aligned TX vs RX (clean subplots)
# -------------------------
N_plot = 200  # smaller window for nicer sinusoid view
n = np.arange(N_plot)

plt.figure(figsize=(10, 6))

# I component
plt.subplot(2, 1, 1)
plt.plot(n, np.real(tx_aligned[:N_plot]), label="TX I", linewidth=2)
plt.plot(n, np.real(rx_corrected[:N_plot]), "--", label="RX I (aligned)", linewidth=2)
plt.ylabel("Amplitude")
plt.title("Aligned TX and RX (I component)")
plt.legend()
plt.grid(True)

# Q component
plt.subplot(2, 1, 2)
plt.plot(n, np.imag(tx_aligned[:N_plot]), label="TX Q", linewidth=2)
plt.plot(n, np.imag(rx_corrected[:N_plot]), "--", label="RX Q (aligned)", linewidth=2)
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.title("Aligned TX and RX (Q component)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("pluto/plots/tx_rx_aligned.png", dpi=150)
