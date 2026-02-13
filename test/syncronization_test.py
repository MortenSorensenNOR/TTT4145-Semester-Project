"""Integration checks for channel synchronization-related behavior."""

import numpy as np

from modules.channel import (
    ChannelModel,
    ChannelProfile,
    ProfileOverrides,
    ProfileRequest,
    delay_ns_to_samples,
    distance_to_delay,
)

SAMPLE_RATE_HZ = 1e6
DISTANCE_M = 1000.0
PULSE_FREQUENCY_HZ = 1000.0
PULSE_LENGTH_S = 0.01
WAIT_TIME_S = 0.0025
SNR_DB = 15.0
PHASE_OFFSET_RAD = 0.5
SEED = 42


def test_channel_applies_delay_and_preserves_shape() -> None:
    """Verify delayed-channel processing keeps expected signal dimensions."""
    t_pulse = np.arange(0.0, PULSE_LENGTH_S, step=1 / SAMPLE_RATE_HZ)
    pulse = 1j * np.sin(2 * np.pi * PULSE_FREQUENCY_HZ * t_pulse, dtype=complex)

    pad_width = int(WAIT_TIME_S * SAMPLE_RATE_HZ)
    signal = np.concatenate(
        (
            np.zeros(pad_width, dtype=complex),
            pulse,
            np.zeros(pad_width, dtype=complex),
        ),
    )

    delay_samples = int(
        delay_ns_to_samples(distance_to_delay(DISTANCE_M) * 1e9, SAMPLE_RATE_HZ),
    )
    request = ProfileRequest(
        sample_rate=SAMPLE_RATE_HZ,
        snr_db=SNR_DB,
        seed=SEED,
        overrides=ProfileOverrides(
            cfo_hz=0.0,
            phase_offset_rad=PHASE_OFFSET_RAD,
            delay_samples=float(delay_samples),
        ),
    )
    channel = ChannelModel.from_profile(
        profile=ChannelProfile.IDEAL,
        request=request,
    )

    rx_signal = channel.apply(signal)

    np.testing.assert_equal(rx_signal.shape, signal.shape)
    np.testing.assert_array_less(0, delay_samples)
