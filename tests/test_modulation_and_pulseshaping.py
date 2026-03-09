import numpy as np
import matplotlib.pyplot as plt

from utils.plotting import plot_iq, plot_constellation
from modules.modulation_schemes import BPSK, PSK8, QPSK
from modules.pulse_shaping import RRCConfig, rrc_filter, upsample, downsample

if __name__ == "__main__":
    # setup
    bpsk = BPSK()
    qpsk = QPSK()
    psk8 = PSK8()

    rrc_config = RRCConfig(8, 0.25, 101)
    rrc_taps = rrc_filter(rrc_config)

    # generate some random bits
    num_symbols = 32
    bits_bpsk = np.random.randint(0, 2, (num_symbols, bpsk.bits_per_symbol))
    bits_qpsk = np.random.randint(0, 2, (num_symbols, qpsk.bits_per_symbol))
    bits_psk8 = np.random.randint(0, 2, (num_symbols, psk8.bits_per_symbol))

    # modulate
    bpsk_syms = bpsk.bits2symbols(bits_bpsk)
    qpsk_syms = qpsk.bits2symbols(bits_qpsk)
    psk8_syms = psk8.bits2symbols(bits_psk8)

    plot_constellation(psk8_syms)
    plt.show()

    # upsample
