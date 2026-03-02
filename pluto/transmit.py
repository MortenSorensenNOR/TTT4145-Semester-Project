"""Transmit a framed digital signal using the PlutoSDR.

Pipeline: payload bits -> frame encode -> modulate -> preamble + frame -> pulse shape -> SDR
"""

import logging

import numpy as np
import time

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
    NODE_DST,
    NODE_SRC,
    PILOT_CONFIG,
    PIPELINE,
    RRC_ALPHA,
    RRC_NUM_TAPS,
    SPS,
    SYNC_CONFIG,
    get_modulator,
)

GUARD_SAMPLES = 500  # zeros before/after TX to avoid DAC transients
UNCODED_MAX_PAYLOAD_BITS = 2**10 - 1  # conservative limit when channel coding is off

PACKETS_PER_SECOND = 10


def max_payload_bits(coding_rate: CodeRates = CODING_RATE) -> int:
    """Maximum number of payload bits for the given coding rate."""
    if not PIPELINE.channel_coding:
        return UNCODED_MAX_PAYLOAD_BITS
    max_k = int(max(ldpc_get_supported_payload_lengths(coding_rate)))
    return max_k - FrameConstructor.PAYLOAD_CRC_BITS


def build_tx_signal_from_bits(
    payload_bits: np.ndarray,
    frame_constructor: FrameConstructor,
    mod_scheme: ModulationSchemes = MOD_SCHEME,
    coding_rate: CodeRates = CODING_RATE,
    src: int = NODE_SRC,
    dst: int = NODE_DST,
    frame_type: int = 0,
    sequence_number: int = 0,
) -> np.ndarray:
    """Run the full TX pipeline from raw payload bits and return baseband samples."""
    max_bits = max_payload_bits(coding_rate)
    if len(payload_bits) > max_bits:
        msg = f"Payload too long: {len(payload_bits)} bits exceeds max {max_bits} for {coding_rate.name}"
        raise ValueError(msg)

    header = FrameHeader(
        length=len(payload_bits),
        src=src,
        dst=dst,
        frame_type=frame_type,
        mod_scheme=mod_scheme,
        coding_rate=coding_rate,
        sequence_number=sequence_number,
    )
    header_encoded, payload_encoded = frame_constructor.encode(
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

    if PIPELINE.pulse_shaping:
        h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
        tx_signal = upsample_and_filter(frame, SPS, h_rrc)
    else:
        tx_signal = frame
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    return np.concatenate([zeros, tx_signal, zeros])


def build_tx_signal(
    message: str,
    frame_constructor: FrameConstructor,
    mod_scheme: ModulationSchemes = MOD_SCHEME,
    coding_rate: CodeRates = CODING_RATE,
    src: int = NODE_SRC,
    dst: int = NODE_DST,
    frame_type: int = 0,
    sequence_number: int = 0,
) -> np.ndarray:
    """Run the full TX pipeline from a text message and return baseband samples."""
    return build_tx_signal_from_bits(
        text_to_bits(message),
        frame_constructor,
        mod_scheme,
        coding_rate,
        src,
        dst,
        frame_type,
        sequence_number,
    )


if __name__ == "__main__":
    import argparse

    from pluto import create_pluto
    from pluto.config import DAC_SCALE, configure_tx

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Transmit a message over PlutoSDR")
    parser.add_argument("message", nargs="?", default="Hello, PlutoSDR!", help="Text message to transmit")
    parser.add_argument("--tx-gain", type=float, default=DEFAULT_TX_GAIN, help="TX gain in dB (default: %(default)s)")
    parser.add_argument("--pluto-ip", default="192.168.2.1", help="PlutoSDR IP address (default: %(default)s)")
    args = parser.parse_args()

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr = create_pluto(f"ip:{args.pluto_ip}")
    configure_tx(sdr, gain=args.tx_gain)

    # ── Build TX signal ───────────────────────────────────────────────
    frame_constructor = FrameConstructor()
    tx_signal = build_tx_signal(args.message, frame_constructor, MOD_SCHEME, CODING_RATE)
    samples = tx_signal * DAC_SCALE

    logger.info("TX: %r  (%d samples @ %d sps, gain=%.0f dB)", args.message, len(samples), SPS, args.tx_gain)

    # ── Transmit ──────────────────────────────────────────────────────
    try:
        while True:
            for i in range(PACKETS_PER_SECOND):
                start = time.perf_counter()
                sdr.tx(samples)
                time.sleep((1/PACKETS_PER_SECOND) - time.perf_counter()-start)

    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
