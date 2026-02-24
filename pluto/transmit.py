"""Transmit a framed digital signal using the PlutoSDR.

Pipeline: payload bits -> frame encode -> modulate -> preamble + frame -> pulse shape -> SDR
"""

import logging

import numpy as np

from modules.channel_coding import CodeRates, ldpc_get_supported_payload_lengths
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.pilots import insert_pilots
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import build_preamble
from modules.util import text_to_bits
from pluto.config import (
    CODING_RATE,
    DEFAULT_TX_GAIN,
    MOD_SCHEME,
    PILOT_CONFIG,
    PIPELINE,
    RRC_ALPHA,
    RRC_NUM_TAPS,
    SPS,
    SYNC_CONFIG,
    get_modulator,
)

GUARD_SAMPLES = 500

_fc = FrameConstructor()
_h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)


def max_payload_bits(coding_rate: CodeRates = CODING_RATE) -> int:
    """Maximum number of payload bits for the given coding rate."""
    if not PIPELINE.channel_coding:
        return 2**10 - 1  # header length field limit
    max_k = int(max(ldpc_get_supported_payload_lengths(coding_rate)))
    return max_k - FrameConstructor.PAYLOAD_CRC_BITS


def build_tx_signal_from_bits(
    payload_bits: np.ndarray,
    mod_scheme: ModulationSchemes = MOD_SCHEME,
    coding_rate: CodeRates = CODING_RATE,
) -> np.ndarray:
    """Run the full TX pipeline from raw payload bits and return baseband samples."""
    max_bits = max_payload_bits(coding_rate)
    if len(payload_bits) > max_bits:
        msg = f"Payload too long: {len(payload_bits)} bits exceeds max {max_bits} for {coding_rate.name}"
        raise ValueError(msg)

    header = FrameHeader(
        length=len(payload_bits),
        src=0,
        dst=0,
        frame_type=0,
        mod_scheme=mod_scheme,
        coding_rate=coding_rate,
        sequence_number=0,
    )
    header_encoded, payload_encoded = _fc.encode(
        header,
        payload_bits,
        channel_coding=PIPELINE.channel_coding,
        interleaving=PIPELINE.interleaving,
    )

    header_symbols = get_modulator(ModulationSchemes.BPSK).bits2symbols(header_encoded)
    payload_symbols = get_modulator(header.mod_scheme).bits2symbols(payload_encoded)
    if PIPELINE.pilots:
        payload_symbols = insert_pilots(payload_symbols, PILOT_CONFIG)

    preamble = build_preamble(SYNC_CONFIG)
    frame = np.concatenate([preamble, header_symbols, payload_symbols])

    tx_signal = upsample_and_filter(frame, SPS, _h_rrc) if PIPELINE.pulse_shaping else frame
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    return np.concatenate([zeros, tx_signal, zeros])


def build_tx_signal(
    message: str,
    mod_scheme: ModulationSchemes = MOD_SCHEME,
    coding_rate: CodeRates = CODING_RATE,
) -> np.ndarray:
    """Run the full TX pipeline from a text message and return baseband samples."""
    return build_tx_signal_from_bits(text_to_bits(message), mod_scheme, coding_rate)


if __name__ == "__main__":
    import argparse

    from pluto import create_pluto
    from pluto.config import CENTER_FREQ, DAC_SCALE, SAMPLE_RATE

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Transmit a message over PlutoSDR")
    parser.add_argument("message", nargs="?", default="Hello, PlutoSDR!", help="Text message to transmit")
    parser.add_argument("--tx-gain", type=float, default=DEFAULT_TX_GAIN, help="TX gain in dB (default: %(default)s)")
    args = parser.parse_args()

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr = create_pluto()
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.tx_hardwaregain_chan0 = args.tx_gain

    # ── Build TX signal ───────────────────────────────────────────────
    tx_signal = build_tx_signal(args.message, MOD_SCHEME, CODING_RATE)
    samples = tx_signal * DAC_SCALE

    logger.info("TX: %r  (%d samples @ %d sps, gain=%.0f dB)", args.message, len(samples), SPS, args.tx_gain)

    # ── Transmit ──────────────────────────────────────────────────────
    try:
        while True:
            sdr.tx(samples)
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
