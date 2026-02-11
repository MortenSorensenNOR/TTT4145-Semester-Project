import numpy as np
import matplotlib.pyplot as plt
from modules.channel import ChannelModel, ChannelProfile, ChannelConfig, delay_ns_to_samples, distance_to_delay

# Your test signal (e.g., from modulator)
sample_rate = 1e6
distance = 1000

f = 1000
pulse_length = 0.01 # seconds
t_pulse = np.arange(0, pulse_length, step=1/sample_rate)
pulse = 1j * np.sin(2 * np.pi * f * t_pulse, dtype=complex)

wait_time = 0.0025
signal_length = pulse_length + 2 * wait_time
t = np.arange(0, signal_length, step=1/sample_rate)

signal = np.concatenate((np.zeros(int(wait_time * sample_rate), dtype=complex), pulse, np.zeros(int(wait_time * sample_rate), dtype=complex)))
print(signal.shape)

# === Option 1: Use preset profiles ===
channel = ChannelModel.from_profile(
    profile=ChannelProfile.IDEAL,  # AWGN only
    sample_rate=sample_rate,
    snr_db=15.0,
    cfo_hz=000.0,           # Add 500 Hz carrier offset
    phase_offset_rad=0.5,   # Random initial phase
    seed=42,                # For reproducibility
    delay_samples=int(delay_ns_to_samples(distance_to_delay(distance) * 1e9, sample_rate))
)
print("Samples delay: ", delay_ns_to_samples(distance_to_delay(distance) * 1e9, sample_rate))

# Apply to signal
rx_signal = channel.apply(signal)

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, np.real(rx_signal), label="RX Signal: I")
ax[0].plot(t, np.real(signal), label="Signal: I")
ax[1].plot(t, np.imag(rx_signal), label="RX Signal: Q")
ax[1].plot(t, np.imag(signal), label="Signal: Q")
plt.legend()
plt.show()
