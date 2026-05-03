import numpy as np
import matplotlib.pyplot as plt
import time

from utils.plotting import plot_iq, plot_constellation, plot_modulation_schemes_ideal_constallations
from modules.modulators.modulators import BPSK, PSK8, QPSK
from modules.pulse_shaping.pulse_shaping import rrc_filter, upsample, downsample

if __name__ == "__main__":
    # setup
    bpsk = BPSK()
    qpsk = QPSK()
    start=time.perf_counter()
    psk8 = PSK8()
    print(time.perf_counter()-start)

    sps, alpha, ntaps = 8, 0.25, 101
    rrc_taps = rrc_filter(sps, alpha, ntaps)

    # generate some random bits
    num_symbols = 120000
    bpsk_bits = np.random.randint(0, 2, (num_symbols, bpsk.bits_per_symbol))
    qpsk_bits = np.random.randint(0, 2, (num_symbols//2, qpsk.bits_per_symbol))
    psk8_bits = np.random.randint(0, 2, (num_symbols//3, psk8.bits_per_symbol))

    # modulate
    start = time.perf_counter()
    bpsk_syms = bpsk.bits2symbols(bpsk_bits)
    stop_bpsk = time.perf_counter()
    qpsk_syms = qpsk.bits2symbols(qpsk_bits)
    stop_qpsk = time.perf_counter()
    psk8_syms = psk8.bits2symbols(psk8_bits)
    stop_psk8 = time.perf_counter()

    print("bpsk.bits2symbols:", stop_bpsk-start)
    print("qpsk.bits2symbols:", stop_qpsk-stop_bpsk)
    print("psk8.bits2symbols:", stop_psk8-stop_qpsk)

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
    start = time.perf_counter()
    bpsk_bits_rec = bpsk.symbols2bits(bpsk_rec)
    stop_bpsk = time.perf_counter()
    qpsk_bits_rec = qpsk.symbols2bits(qpsk_rec)
    stop_qpsk = time.perf_counter()
    psk8_bits_rec = psk8.symbols2bits(psk8_rec)
    stop_psk8 = time.perf_counter()

    print("bpsk.symbols2bits:", stop_bpsk-start)
    print("qpsk.symbols2bits:", stop_qpsk-stop_bpsk)
    print("psk8.symbols2bits:", stop_psk8-stop_qpsk)


    # calcualte BER
    bpsk_ber = np.mean(bpsk_bits_rec != bpsk_bits)
    qpsk_ber = np.mean(qpsk_bits_rec != qpsk_bits)
    psk8_ber = np.mean(psk8_bits_rec != psk8_bits)

    print(bpsk_bits.shape, bpsk_bits_rec.shape)
    print(qpsk_bits.shape, qpsk_bits_rec.shape)
    print(psk8_bits.shape, psk8_bits_rec.shape)
    # assert bpsk_ber == 0.0
    #assert qpsk_ber == 0.0
    #assert psk8_ber == 0.0

    print("-----------------------")
    print(f"BER: ")
    print(f"BPSK: {bpsk_ber}")
    print(f"QPSK: {qpsk_ber}")
    print(f"PSK8: {psk8_ber}")
    print("-----------------------")
