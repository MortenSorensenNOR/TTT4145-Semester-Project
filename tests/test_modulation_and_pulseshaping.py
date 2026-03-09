import matplotlib.pyplot as plt
import numpy as np

from modules.modulation_schemes import BPSK, QPSK
from modules.pulse_shaping import RRCConfig, rrc_filter
from utils.plotting import plot_constellation

if __name__ == "__main__":
    # setup
    bpsk = BPSK()
    qpsk = QPSK()

    rrc_config = RRCConfig(8, 0.25, 101)
    rrc_taps = rrc_filter(rrc_config)

    # generate some random bits
    num_symbols = 32
    bits_bpsk = np.random.randint(0, 2, (num_symbols, bpsk.bits_per_symbol))
    bits_qpsk = np.random.randint(0, 2, (num_symbols, qpsk.bits_per_symbol))

    # modulate
    bpsk_syms = bpsk.bits2symbols(bits_bpsk)
    qpsk_syms = qpsk.bits2symbols(bits_qpsk)

    plot_constellation(bpsk_syms)
    plt.show()

    # upsample
