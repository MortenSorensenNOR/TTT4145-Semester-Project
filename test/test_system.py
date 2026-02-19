"""System-level checks for modulation, shaping, and channel flow."""

import matplotlib.pyplot as plt
import numpy as np

from modules.channel import (
    ChannelConfig,
    ChannelModel,
)
from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.modulation import BPSK, QPSK
from modules.plotting import plot_iq
from modules.pulse_shaping import rrc_filter

BANDWIDTH_HZ = 1e6
ROLL_OFF = 0.25
SAMPLES_PER_SYMBOL = 8
OVERLAP_FACTOR = 6
NUM_SYMBOLS = 512
BITS_PER_SYMBOL = 2
SNR_DB = 1.0
SEED = 42

PAYLOAD_LENGTH = 324
SRC = 1
DST = 2
MOD_SCHEME = ModulationSchemes.QPSK
CODING_RATE = CodeRates.HALF_RATE


def test() -> None:
    """Verify end-to-end chain output dimensions and finite BER."""
    symbol_rate = BANDWIDTH_HZ / (1 + ROLL_OFF)
    sample_rate = symbol_rate * SAMPLES_PER_SYMBOL
    pulse_taps = SAMPLES_PER_SYMBOL * OVERLAP_FACTOR * 2 + 1

    # modulation
    pulse = rrc_filter(SAMPLES_PER_SYMBOL, ROLL_OFF, pulse_taps)
    bpsk = BPSK()
    qpsk = QPSK()
    rng = np.random.default_rng(SEED)

    # channel
    config = ChannelConfig(sample_rate=sample_rate, snr_db=SNR_DB, seed=SEED)
    channel = ChannelModel(config)

    # generate payload
    header = FrameHeader(length=PAYLOAD_LENGTH, src=SRC, dst=DST, mod_scheme=MOD_SCHEME, coding_rate=CODING_RATE)
    payload = rng.integers(0, 2, size=(PAYLOAD_LENGTH), dtype=int)

    frame_constructor = FrameConstructor()
    header_encoded, payload_encoded = frame_constructor.encode(header, payload)

    # modulate header and payload
    header_modulated = bpsk.bits2symbols(header_encoded)
    payload_modulated = qpsk.bits2symbols(payload_encoded)
    frame_modulated = np.concatenate([header_modulated, payload_modulated])

    # rest of tx
    upsampled = np.zeros(len(frame_modulated) * SAMPLES_PER_SYMBOL, dtype=complex)
    upsampled[::SAMPLES_PER_SYMBOL] = frame_modulated
    tx_signal = np.convolve(upsampled, pulse, mode="same")

    rx_signal = channel.apply(tx_signal)
    rx_filtered = np.convolve(rx_signal, pulse, mode="same")
    rx_symbols = rx_filtered[::SAMPLES_PER_SYMBOL]

    rx_header_sym = rx_symbols[: len(header_modulated)]
    rx_payload_sym = rx_symbols[len(header_modulated) :]

    rx_header_bits = bpsk.symbols2bits(rx_header_sym)
    rx_payload_bits = qpsk.symbols2bits_soft(rx_payload_sym)

    # decode header and payload
    rx_header = frame_constructor.decode_header(rx_header_bits)
    rx_payload = frame_constructor.decode_payload(rx_header, rx_payload_bits.flatten(), soft=True)
    ber = float(np.mean(rx_payload != payload))

    np.testing.assert_equal(rx_header.length, PAYLOAD_LENGTH)
    np.testing.assert_equal(rx_payload_bits.shape[1], BITS_PER_SYMBOL)
    np.testing.assert_array_less(-1e-12, ber)
    np.testing.assert_array_less(ber, 1.0 + 1e-12)

    plot_iq(tx_signal)
    plot_iq(rx_signal)
    plt.show(block=False)


if __name__ == "__main__":
    test()
