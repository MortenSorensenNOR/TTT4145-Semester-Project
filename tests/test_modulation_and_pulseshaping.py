import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import plot_iq, plot_constellation, plot_modulation_schemes_ideal_constallations
from modules.modulators import BPSK, PSK8, QPSK
from modules.pulse_shaping import rrc_filter, upsample, downsample

if __name__ == "__main__":
    # setup
    bpsk = BPSK()
    qpsk = QPSK()
    psk8 = PSK8()

    sps, alpha, ntaps = 8, 0.25, 101
    rrc_taps = rrc_filter(sps, alpha, ntaps)

    # generate some random bits
    num_symbols = 128
    bpsk_bits = np.random.randint(0, 2, (num_symbols, bpsk.bits_per_symbol))
    qpsk_bits = np.random.randint(0, 2, (num_symbols, qpsk.bits_per_symbol))
    psk8_bits = np.random.randint(0, 2, (num_symbols, psk8.bits_per_symbol))

    # modulate
    bpsk_syms = bpsk.bits2symbols(bpsk_bits)
    qpsk_syms = qpsk.bits2symbols(qpsk_bits)
    psk8_syms = psk8.bits2symbols(psk8_bits)

    # upsample and filter
    bpsk_sig = upsample(bpsk_syms, sps, rrc_taps)
    qpsk_sig = upsample(qpsk_syms, sps, rrc_taps)
    psk8_sig = upsample(psk8_syms, sps, rrc_taps)

    plot_iq(qpsk_sig)

    # filter and downsample
    bpsk_rec = downsample(bpsk_sig, sps, rrc_taps)
    qpsk_rec = downsample(qpsk_sig, sps, rrc_taps)
    psk8_rec = downsample(psk8_sig, sps, rrc_taps)

    # demodulate
    bpsk_bits_rec = bpsk.symbols2bits(bpsk_rec)
    qpsk_bits_rec = qpsk.symbols2bits(qpsk_rec)
    psk8_bits_rec = psk8.symbols2bits(psk8_rec)

    # calcualte BER
    bpsk_ber = np.mean(bpsk_bits_rec != bpsk_bits)
    qpsk_ber = np.mean(qpsk_bits_rec != qpsk_bits)
    psk8_ber = np.mean(psk8_bits_rec != psk8_bits)

    assert bpsk_ber == 0.0
    assert qpsk_ber == 0.0
    assert psk8_ber == 0.0

    print("-----------------------")
    print(f"BER: ")
    print(f"BPSK: {bpsk_ber}")
    print(f"QPSK: {qpsk_ber}")
    print(f"PSK8: {psk8_ber}")
    print("-----------------------")
