"""Integration checks for channel synchronization-related behavior."""

import numpy as np
import matplotlib.pyplot as plt

from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
    delay_ns_to_samples,
    distance_to_delay,
)

# --- Coarse CFO estimation test ---
from modules.syncronization import CoarseCFOSequence
from modules.plotting import plot_iq

# Parameters
sample_rate = 1e6
Ts = 1 / sample_rate
N = 16  # symbols per block
M = 4   # number of repetitions
true_cfo_hz = -7400.0  # 24 kHz CFO
delay_samples_cfo = 100
snr_db_cfo = 30.0

# Create preamble
cfo_seq = CoarseCFOSequence(N=N, M=M, modulation='qpsk')
preamble = cfo_seq.preamble

# Pad signal so there's room after delay
tx_signal = np.concatenate([preamble, np.zeros(delay_samples_cfo, dtype=complex)])

# Setup channel with CFO and delay
request_cfo = ProfileRequest(
    sample_rate=sample_rate,
    snr_db=snr_db_cfo,
    seed=42,
    overrides=ProfileOverrides(
        cfo_hz=true_cfo_hz,
        delay_samples=float(delay_samples_cfo),
    ),
)
channel_cfo = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=request_cfo)

# Pass through channel
rx_signal_cfo = channel_cfo.apply(tx_signal)

# Extract preamble from received signal (account for delay)
rx_preamble = rx_signal_cfo[delay_samples_cfo : delay_samples_cfo + N * M]

# Estimate CFO
estimated_cfo = cfo_seq.estimate_corase_cfo(rx_preamble, Ts)

print(f"True CFO: {true_cfo_hz:.1f} Hz")
print(f"Estimated CFO: {estimated_cfo:.1f} Hz")
print(f"Error: {abs(estimated_cfo - true_cfo_hz):.1f} Hz")

# # Plotting
# plot_iq(preamble, title="TX Preamble")
# plot_iq(rx_preamble, title="RX Preamble (with CFO)")
#
# # Correct CFO and plot
# t = np.arange(len(rx_preamble)) * Ts
# rx_corrected = rx_preamble * np.exp(-1j * 2 * np.pi * estimated_cfo * t)
# plot_iq(rx_corrected, title="RX Preamble (CFO corrected)")
# plt.show()

# --- Pulse shaped version with cyclic prefix ---
from modules.pulseshaping import rrc_filter

print("\n--- With pulse shaping + cyclic prefix ---")

# Pulse shaping params
sps = 4
alpha = 0.35
num_taps = 101

rrc = rrc_filter(sps, alpha, num_taps)

# Upsample symbols
preamble_up = np.zeros(len(preamble) * sps, dtype=complex)
preamble_up[::sps] = preamble

# TX pulse shape (mode='full' to get all transients)
preamble_shaped = np.convolve(preamble_up, rrc, mode='full')

# Add cyclic prefix: prepend last num_taps samples
cp_len = num_taps
preamble_with_cp = np.concatenate([preamble_shaped[-cp_len:], preamble_shaped])

# Pad for channel delay
tx_shaped = np.concatenate([preamble_with_cp, np.zeros(delay_samples_cfo, dtype=complex)])

# Through channel
request_shaped = ProfileRequest(
    sample_rate=sample_rate,
    snr_db=snr_db_cfo,
    seed=42,
    overrides=ProfileOverrides(
        cfo_hz=true_cfo_hz,
        delay_samples=float(delay_samples_cfo),
    ),
)
channel_shaped = ChannelModel.from_profile(profile=ChannelProfile.IDEAL, request=request_shaped)
rx_shaped = channel_shaped.apply(tx_shaped)

# CFO estimation BEFORE matched filter
# Extract preamble (skip CP, account for delay)
preamble_start = delay_samples_cfo + cp_len
preamble_len_samples = N * M * sps
rx_preamble_raw = rx_shaped[preamble_start : preamble_start + preamble_len_samples]

# Downsample to symbol rate BEFORE CFO estimation
# This restores the CFO range to ±1/(2*N*Ts_symbol) = ±31.25 kHz
rx_downsampled = rx_preamble_raw[::sps]
Ts_symbol = sps * Ts

# Now use the original estimator at symbol rate
estimated_cfo_shaped = cfo_seq.estimate_corase_cfo(rx_downsampled, Ts_symbol)

print(f"True CFO: {true_cfo_hz:.1f} Hz")
print(f"Estimated CFO: {estimated_cfo_shaped:.1f} Hz")
print(f"Error: {abs(estimated_cfo_shaped - true_cfo_hz):.1f} Hz")
