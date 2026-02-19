"""Transmit a framed digital signal using the PlutoSDR.

Pipeline: text -> bits -> frame encode -> modulate -> preamble + frame -> pulse shape -> SDR
"""

import logging

import numpy as np

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.modulation import BPSK
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import build_preamble
from modules.util import text_to_bits
from pluto.config import RRC_ALPHA, RRC_NUM_TAPS, SPS, get_modulator

TX_GAIN = -30
GUARD_SAMPLES = 500
MOD_SCHEME = ModulationSchemes.QPSK
CODING_RATE = CodeRates.HALF_RATE


def build_tx_signal(
    message: str,
    mod_scheme: ModulationSchemes = MOD_SCHEME,
    coding_rate: CodeRates = CODING_RATE,
) -> np.ndarray:
    """Run the full TX pipeline and return baseband samples (pre-DAC scaling).

    Args:
        message: Text string to transmit.
        mod_scheme: Modulation scheme for the payload.
        coding_rate: LDPC coding rate.

    Returns:
        Complex baseband samples ready for DAC scaling and transmission.

    """
    payload_bits = text_to_bits(message)

    header = FrameHeader(
        length=len(payload_bits),
        src=0,
        dst=0,
        mod_scheme=mod_scheme,
        coding_rate=coding_rate,
    )
    fc = FrameConstructor()
    header_encoded, payload_encoded = fc.encode(header, payload_bits)

    header_symbols = BPSK().bits2symbols(header_encoded)  # header always BPSK
    payload_symbols = get_modulator(header.mod_scheme).bits2symbols(payload_encoded)

    preamble = build_preamble()
    frame = np.concatenate([preamble, header_symbols, payload_symbols])

    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    tx_signal = upsample_and_filter(frame, SPS, h_rrc)
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    return np.concatenate([zeros, tx_signal, zeros])


if __name__ == "__main__":
    import sys
    from typing import Any, cast

    from pluto import create_pluto
    from pluto.config import CENTER_FREQ, DAC_SCALE, SAMPLE_RATE

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr: Any = cast("Any", create_pluto())
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(CENTER_FREQ)
    sdr.tx_hardwaregain_chan0 = TX_GAIN

    # ── Build TX signal ───────────────────────────────────────────────
    message = sys.argv[1] if len(sys.argv) > 1 else "Hello, PlutoSDR!"
    tx_signal = build_tx_signal(message, MOD_SCHEME, CODING_RATE)
    samples = tx_signal * DAC_SCALE

    logger.info("TX: %r  (%d samples @ %d sps)", message, len(samples), SPS)

    # ── Transmit ──────────────────────────────────────────────────────
    try:
        while True:
            sdr.tx(samples)
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
