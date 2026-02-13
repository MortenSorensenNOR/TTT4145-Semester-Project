"""System-level checks for modulation, shaping, and channel flow."""

import numpy as np

from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
)
from modules.modulation import QPSK
from modules.pulseshaping import PulseShaper

BANDWIDTH_HZ = 1e6
ROLL_OFF = 0.25
SAMPLES_PER_SYMBOL = 8
OVERLAP_FACTOR = 6
NUM_SYMBOLS = 512
BITS_PER_SYMBOL = 2
SNR_DB = 15.0
SEED = 42


def test_end_to_end_signal_chain_shapes_and_outputs() -> None:
    """Verify end-to-end chain output dimensions and finite BER."""
    symbol_rate = BANDWIDTH_HZ / (1 + ROLL_OFF)
    sample_rate = symbol_rate * SAMPLES_PER_SYMBOL
    pulse_taps = SAMPLES_PER_SYMBOL * OVERLAP_FACTOR * 2 + 1

    pulse_shaper = PulseShaper(SAMPLES_PER_SYMBOL, ROLL_OFF, pulse_taps)
    qpsk = QPSK()
    rng = np.random.default_rng(SEED)

    request = ProfileRequest(
        sample_rate=sample_rate,
        snr_db=SNR_DB,
        seed=SEED,
        overrides=ProfileOverrides(
            cfo_hz=0.0,
            phase_offset_rad=0.0,
            delay_samples=0.0,
        ),
    )
    channel = ChannelModel.from_profile(
        profile=ChannelProfile.IDEAL,
        request=request,
    )

    bitstream = rng.integers(
        0,
        BITS_PER_SYMBOL,
        size=(NUM_SYMBOLS, BITS_PER_SYMBOL),
        dtype=int,
    )
    symbols = qpsk.bits2symbols(bitstream)

    upsampled = np.zeros(len(symbols) * SAMPLES_PER_SYMBOL, dtype=complex)
    upsampled[::SAMPLES_PER_SYMBOL] = symbols
    tx_signal = pulse_shaper.shape(upsampled)

    rx_signal = channel.apply(tx_signal)
    rx_filtered = pulse_shaper.shape(rx_signal)
    rx_symbols = rx_filtered[::SAMPLES_PER_SYMBOL]

    rx_bits_hard = qpsk.symbols2bits(rx_symbols)
    rx_bits_soft = qpsk.symbols2bits_soft(rx_symbols)
    ber = float(np.mean(rx_bits_hard != bitstream))

    np.testing.assert_equal(rx_bits_soft.shape[1], BITS_PER_SYMBOL)
    np.testing.assert_equal(len(rx_symbols), NUM_SYMBOLS)
    np.testing.assert_array_less(-1e-12, ber)
    np.testing.assert_array_less(ber, 1.0 + 1e-12)
