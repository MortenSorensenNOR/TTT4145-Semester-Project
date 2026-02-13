import numpy as np
import matplotlib.pyplot as plt
from modules.plotting import plot_iq, plot_constellation, plot_spectrum, plot_constellation_confidence

from modules.channel import ChannelModel, ChannelProfile, ChannelConfig
from modules.modulation import QPSK
from modules.pulseshaping import *

# Setup
bandwidth   = 1e6
alpha       = 0.25 # roll-off factor
symbol_rate = bandwidth / (1 + alpha)
sps         = 8
sample_rate = symbol_rate * sps
overlapping_factor = 6
pulse_shaper = PulseShaper(sps, alpha, sps * overlapping_factor * 2 + 1)

channel_profile = ChannelProfile.URBAN_LOS
snr = 15.0
cfo = 100.0
phase_offset = 0.25

num_symbols = 512
bits_per_symbol = 2 # QPSK
qpsk = QPSK()

# Channel
channel = ChannelModel.from_profile(
    profile=channel_profile,
    sample_rate=sample_rate,
    snr_db=snr,
    cfo_hz=cfo,
    phase_offset_rad=phase_offset,
    seed=42,
    delay_samples=0 # does not make much difference at small distances
)

# TX
bitstream = np.random.randint(0, 2, (num_symbols, bits_per_symbol), dtype=int)
symbols = qpsk.bits2symbols(bitstream)
assert(np.array_equal(bitstream, qpsk.symbols2bits(symbols)))

upsampled = np.zeros(len(symbols) * sps, dtype=complex)
upsampled[::sps] = symbols
tx_signal = pulse_shaper.shape(upsampled)

# channel
rx_signal = channel.apply(tx_signal)

# RX
rx_filtered = pulse_shaper.shape(rx_signal)
rx_symbols = rx_filtered[::sps]
rx_bits = qpsk.symbols2bits(rx_symbols)
rx_bits_soft = qpsk.symbols2bits_soft(rx_symbols)
print(rx_bits_soft)

ber = np.mean(rx_bits != bitstream)
print(f"BER: {ber}")

plot_constellation_confidence(rx_symbols, rx_bits_soft, tx_symbols=symbols)
plt.show()
