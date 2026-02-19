"""Receive and decode a framed digital signal from the PlutoSDR.

Pipeline: SDR -> matched filter -> sync -> CFO correct -> phase correct
       -> downsample -> normalize -> header decode -> payload demod -> channel decode -> text
"""

import logging
from dataclasses import dataclass

import numpy as np

from modules.channel_coding import CodeRates, LDPCConfig, ldpc_get_supported_payload_lengths
from modules.frame_constructor import FrameConstructor, FrameHeader
from modules.modulation import BPSK
from modules.pulse_shaping import rrc_filter
from modules.synchronization import Synchronizer, SynchronizerConfig
from modules.util import bits_to_text
from pluto.config import RRC_ALPHA, RRC_NUM_TAPS, SPS, get_modulator

logger = logging.getLogger(__name__)

RX_GAIN = 70.0


@dataclass
class FrameResult:
    """Result from a successful frame decode."""

    text: str
    header: FrameHeader
    cfo_hz: float
    consumed_samples: int


@dataclass
class DecodeConfig:
    """Configuration for frame decoding."""

    sample_rate: float
    sps: int
    bpsk: BPSK
    golay_ratio: int


def try_decode_frame(
    filtered: np.ndarray,
    abs_offset: int,
    sync: Synchronizer,
    fc: FrameConstructor,
    config: DecodeConfig,
) -> FrameResult | None:
    """Try to detect and decode a single frame from filtered samples.

    Args:
        filtered: RRC-matched-filtered samples.
        abs_offset: Absolute sample index of the start of `filtered`.
        sync: Synchronizer instance.
        fc: FrameConstructor instance.
        config: Decode configuration parameters.

    Returns:
        FrameResult on success, None on failure.

    """
    header_n_symbols = fc.frame_header_constructor.header_length * config.golay_ratio

    # ── Synchronization (at upsampled rate) ───────────────────────
    result = sync.detect_preamble(filtered, config.sample_rate)
    if not result.success:
        logger.debug("No preamble detected")
        return None

    # ── CFO correction (absolute sample index for phase continuity)
    n_vec = abs_offset + np.arange(len(filtered))
    rx_corr = filtered * np.exp(-1j * 2 * np.pi * result.cfo_hat_hz / config.sample_rate * n_vec)

    # ── Downsample to symbol rate ─────────────────────────────────
    data_start = result.timing_hat + sync.config.n_long * config.sps
    symbols = rx_corr[data_start :: config.sps]

    # ── Residual phase correction (data-aided via long ZC) ────────
    zc_long_ref = sync.zc_long
    zc_start = result.timing_hat
    zc_rx = rx_corr[zc_start :: config.sps][: len(zc_long_ref)]
    if len(zc_rx) == len(zc_long_ref):
        phase_hat = np.angle(np.sum(zc_rx * np.conj(zc_long_ref)))
        symbols = symbols * np.exp(-1j * phase_hat)

    # ── Decode header ────────────────────────────────────────────
    header = _decode_header_with_validation(symbols, header_n_symbols, fc, config)
    if header is None:
        return None

    # ── Validate and prepare payload parameters ──────────────────
    payload_params = _prepare_payload_parameters(header, fc)
    if payload_params is None:
        return None
    n_payload_symbols = payload_params

    # ── Check symbol buffer and extract payload ──────────────────
    if len(symbols) < header_n_symbols + n_payload_symbols:
        logger.debug(
            "Not enough symbols for payload (%d < %d)",
            len(symbols) - header_n_symbols,
            n_payload_symbols,
        )
        return None

    # ── Soft-demodulate payload → LLRs ────────────────────────────
    payload_symbols = symbols[header_n_symbols : header_n_symbols + n_payload_symbols]
    modulator = get_modulator(header.mod_scheme)
    sigma_sq = modulator.estimate_noise_variance(payload_symbols)
    payload_llrs = modulator.symbols2bits_soft(payload_symbols, sigma_sq=sigma_sq).flatten()

    # ── Channel decode (LDPC + CRC) ───────────────────────────────
    try:
        payload_bits = fc.decode_payload(header, payload_llrs, soft=True)
    except ValueError as exc:
        logger.debug("Payload decode failed: %s", exc)
        return None

    # ── Compute consumed samples ──────────────────────────────────
    consumed = result.timing_hat + sync.config.n_long * config.sps + (header_n_symbols + n_payload_symbols) * config.sps

    text = bits_to_text(payload_bits)
    return FrameResult(
        text=text,
        header=header,
        cfo_hz=result.cfo_hat_hz,
        consumed_samples=consumed,
    )


def _decode_header_with_validation(
    symbols: np.ndarray,
    header_n_symbols: int,
    fc: FrameConstructor,
    config: DecodeConfig,
) -> FrameHeader | None:
    """Decode header and validate with normalization."""
    if len(symbols) < header_n_symbols:
        logger.debug("Not enough symbols for header (%d < %d)", len(symbols), header_n_symbols)
        return None

    # ── Amplitude normalization (from known-power header) ─────────
    header_power = np.mean(np.abs(symbols[:header_n_symbols]) ** 2)
    normalized_symbols = symbols
    if header_power > 0:
        normalized_symbols = symbols / np.sqrt(header_power)
    header_hard = config.bpsk.symbols2bits(normalized_symbols[:header_n_symbols])

    try:
        return fc.decode_header(header_hard)
    except ValueError:
        logger.debug("Header CRC failed")
        return None


def _prepare_payload_parameters(
    header: FrameHeader,
    fc: FrameConstructor,
) -> int | None:
    """Prepare and validate payload parameters.

    Returns number of payload symbols needed, or None on error.
    """
    try:
        modulator = get_modulator(header.mod_scheme)
        supported_k = ldpc_get_supported_payload_lengths(header.coding_rate)
        k = int(min(kk for kk in supported_k if kk >= header.length + fc.PAYLOAD_CRC_BITS))
        n_coded = LDPCConfig(k=k, code_rate=header.coding_rate).n
        return n_coded // modulator.bits_per_symbol
    except ValueError:
        logger.debug("Invalid payload parameters from header")
        return None


if __name__ == "__main__":
    from typing import Any, cast

    from pluto import create_pluto
    from pluto.config import CENTER_FREQ, SAMPLE_RATE

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr: Any = cast("Any", create_pluto())
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = 2**16

    # ── Reusable objects ──────────────────────────────────────────────
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(SynchronizerConfig(), sps=SPS, rrc_taps=h_rrc)
    fc = FrameConstructor()
    bpsk = BPSK()
    golay_ratio = fc.golay.block_length // fc.golay.message_length
    decode_config = DecodeConfig(
        sample_rate=SAMPLE_RATE,
        sps=SPS,
        bpsk=bpsk,
        golay_ratio=golay_ratio,
    )

    # Maximum frame length in samples for overlap retention
    max_preamble_symbols = sync.config.n_short * sync.config.n_short_reps + sync.config.n_long
    max_header_symbols = fc.frame_header_constructor.header_length * golay_ratio
    max_k = int(max(ldpc_get_supported_payload_lengths()))
    max_payload_coded = LDPCConfig(k=max_k, code_rate=CodeRates.HALF_RATE).n
    MAX_FRAME_SAMPLES = (max_preamble_symbols + max_header_symbols + max_payload_coded) * SPS

    # Flush a few buffers so the SDR settles
    for _ in range(5):
        sdr.rx()

    logger.info("RX: listening on %.0f MHz ...", CENTER_FREQ / 1e6)

    # Overlap-save state for incremental matched filtering
    filter_overlap = len(h_rrc) - 1
    filter_state = np.zeros(filter_overlap, dtype=complex)

    # Pre-allocated receive buffer
    BUF_CAPACITY = MAX_FRAME_SAMPLES * 4
    rx_buf = np.zeros(BUF_CAPACITY, dtype=complex)
    buf_len = 0
    abs_offset = 0

    try:
        while True:
            rx = sdr.rx()

            # ── Incremental matched filter (overlap-save) ─────────────
            chunk = np.concatenate([filter_state, rx])
            new_filtered = np.convolve(chunk, h_rrc, mode="valid")
            filter_state = rx[-filter_overlap:]

            # Append to pre-allocated buffer
            n_new = len(new_filtered)
            if buf_len + n_new > BUF_CAPACITY:
                # Shift out old data to make room
                keep = MAX_FRAME_SAMPLES
                rx_buf[:keep] = rx_buf[buf_len - keep : buf_len]
                abs_offset += buf_len - keep
                buf_len = keep
            rx_buf[buf_len : buf_len + n_new] = new_filtered
            buf_len += n_new

            # ── Multi-frame detection loop ────────────────────────────
            while True:
                frame_result = try_decode_frame(
                    rx_buf[:buf_len],
                    abs_offset,
                    sync,
                    fc,
                    decode_config,
                )

                if frame_result is not None:
                    logger.info(
                        "RX: %r  (mod=%s, rate=%s, CFO=%+.0f Hz, bits=%d)",
                        frame_result.text,
                        frame_result.header.mod_scheme.name,
                        frame_result.header.coding_rate.name,
                        frame_result.cfo_hz,
                        frame_result.header.length,
                    )
                    # Trim consumed samples from front
                    consumed = frame_result.consumed_samples
                    remaining = buf_len - consumed
                    rx_buf[:remaining] = rx_buf[consumed:buf_len]
                    buf_len = remaining
                    abs_offset += consumed
                else:
                    break

            # Keep overlap for frames that may straddle buffer boundaries
            if buf_len > MAX_FRAME_SAMPLES:
                trim = buf_len - MAX_FRAME_SAMPLES
                rx_buf[:MAX_FRAME_SAMPLES] = rx_buf[trim:buf_len]
                buf_len = MAX_FRAME_SAMPLES
                abs_offset += trim
    except KeyboardInterrupt:
        pass
