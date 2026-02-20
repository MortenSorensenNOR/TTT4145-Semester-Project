"""Receive and decode a framed digital signal from the PlutoSDR.

Pipeline: SDR -> matched filter -> sync -> CFO correct -> phase correct
       -> downsample -> normalize -> header decode -> equalize -> phase track
       -> soft demod -> channel decode -> text
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from modules.channel_coding import CodeRates, LDPCConfig, ldpc_get_supported_payload_lengths
from modules.equalization import equalize_payload
from modules.frame_constructor import FrameConstructor, FrameHeader
from modules.modulation import BPSK, estimate_noise_variance
from modules.pilots import PilotConfig, data_indices, n_total_symbols, pilot_aided_phase_track, pilot_indices
from modules.pulse_shaping import rrc_filter
from modules.synchronization import SynchronizationResult, Synchronizer
from modules.util import bits_to_bytes, bits_to_text
from pluto import SDRReceiver
from pluto.config import PILOT_CONFIG, PIPELINE, PipelineConfig, RRC_ALPHA, RRC_NUM_TAPS, SAMPLE_RATE, SPS, SYNC_CONFIG, get_modulator

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Result from a successful frame decode."""

    payload_bits: np.ndarray
    header: FrameHeader
    cfo_hz: float
    consumed_samples: int

    @property
    def text(self) -> str:
        """Decode payload bits as UTF-8 text."""
        return bits_to_text(self.payload_bits)

    @property
    def payload_bytes(self) -> bytes:
        """Decode payload bits as raw bytes."""
        return bits_to_bytes(self.payload_bits)


class FrameDecoder:
    """Detect and decode a single frame from filtered samples."""

    def __init__(
        self,
        sync: Synchronizer,
        fc: FrameConstructor,
        sample_rate: float,
        sps: int,
        pilot_config: PilotConfig | None = None,
        pipeline: PipelineConfig | None = None,
    ) -> None:
        """Initialize with fixed radio parameters."""
        self.sync = sync
        self.fc = fc
        self.sample_rate = sample_rate
        self.sps = sps
        self.pilot_config = pilot_config or PilotConfig()
        self.pipeline = pipeline or PipelineConfig()
        self.bpsk = BPSK()
        self.header_n_symbols = fc.header_encoded_n_bits

    def try_decode(
        self,
        filtered: np.ndarray,
        abs_offset: int,
    ) -> FrameResult | None:
        """Try to detect and decode a single frame from filtered samples."""
        # ── Synchronization (at upsampled rate) ───────────────────────
        result = self.sync.detect_preamble(filtered, self.sample_rate)
        if not result.success:
            logger.debug("No preamble detected")
            return None

        symbols = self._extract_symbols(filtered, result, abs_offset)
        if symbols is None or len(symbols) < self.header_n_symbols:
            logger.debug("Not enough symbols for header")
            return None

        # ── Decode header ────────────────────────────────────────────
        header_hard = self.bpsk.symbols2bits(symbols[: self.header_n_symbols]).flatten()
        try:
            header = self.fc.decode_header(header_hard)
        except ValueError:
            logger.debug("Header CRC failed")
            return None

        # ── Validate payload length ──────────────────────────────────
        n_payload_symbols = self._payload_n_symbols(header)
        if n_payload_symbols is None or len(symbols) < self.header_n_symbols + n_payload_symbols:
            logger.debug("Not enough symbols for payload")
            return None

        return self._decode_payload(symbols, header, n_payload_symbols, result)

    def _extract_symbols(
        self,
        filtered: np.ndarray,
        result: SynchronizationResult,
        abs_offset: int,
    ) -> np.ndarray | None:
        """Apply CFO/phase correction, downsample, and normalize to symbol rate."""
        # ── CFO correction (only from long ZC onward) ─────────────────
        needed_start = result.long_zc_start
        needed = filtered[needed_start:]
        if self.pipeline.cfo_correction:
            n_vec = (abs_offset + needed_start) + np.arange(len(needed))
            rx_corr = needed * np.exp(-1j * 2 * np.pi * result.cfo_hat_hz / self.sample_rate * n_vec)
        else:
            rx_corr = needed

        # ── Residual phase correction (data-aided via long ZC) ────────
        zc_long_ref = self.sync.zc_long
        zc_rx = rx_corr[:: self.sps][: len(zc_long_ref)]
        if len(zc_rx) == len(zc_long_ref):
            phase_hat = np.angle(np.sum(zc_rx * np.conj(zc_long_ref)))
            rx_corr = rx_corr * np.exp(-1j * phase_hat)

        # ── Downsample to symbol rate ─────────────────────────────────
        data_offset = self.sync.config.n_long * self.sps
        symbols = rx_corr[data_offset :: self.sps]

        # ── Amplitude normalization (from known-power BPSK header) ────
        if len(symbols) >= self.header_n_symbols:
            header_power = np.mean(np.abs(symbols[: self.header_n_symbols]) ** 2)
            if header_power > 0:
                symbols = symbols / np.sqrt(header_power)

        return symbols

    def _decode_payload(
        self,
        symbols: np.ndarray,
        header: FrameHeader,
        n_payload_symbols: int,
        sync_result: SynchronizationResult,
    ) -> FrameResult | None:
        """Demodulate and channel-decode the payload from normalized symbols.

        Pipeline: extract -> equalize -> phase track -> re-estimate noise -> soft demod -> decode
        """
        modulator = get_modulator(header.mod_scheme)
        n_coded = self.fc.payload_coded_n_bits(header, channel_coding=self.pipeline.channel_coding)
        n_data = n_coded // modulator.bits_per_symbol

        # ── Extract payload symbols ───────────────────────────────────
        payload_symbols = symbols[self.header_n_symbols : self.header_n_symbols + n_payload_symbols]

        # ── Initial noise estimate from BPSK header (for MMSE regularization) ──
        sigma_sq = self.bpsk.estimate_noise_variance(symbols[: self.header_n_symbols])

        if self.pipeline.pilots:
            # ── Pre-compute pilot/data indices (shared by equalization and phase tracking) ──
            p_idx = pilot_indices(n_data, self.pilot_config)
            d_idx = data_indices(n_data, self.pilot_config)

            # ── Equalize first (before any amplitude modification) ────
            payload_symbols = equalize_payload(payload_symbols, n_data, self.pilot_config, sigma_sq, p_idx=p_idx)

            # ── Pilot-aided phase tracking (returns data-only symbols) ─
            payload_symbols = pilot_aided_phase_track(payload_symbols, n_data, self.pilot_config, p_idx=p_idx, d_idx=d_idx)

        # ── Re-estimate noise variance from equalized payload symbols ─
        sigma_sq = estimate_noise_variance(payload_symbols, modulator.symbol_mapping)

        # ── Soft-demodulate payload -> LLRs ───────────────────────────
        payload_llrs = modulator.symbols2bits_soft(payload_symbols, sigma_sq=sigma_sq).flatten()

        # ── Channel decode (LDPC + CRC) ───────────────────────────────
        try:
            payload_bits = self.fc.decode_payload(
                header,
                payload_llrs,
                soft=True,
                channel_coding=self.pipeline.channel_coding,
                interleaving=self.pipeline.interleaving,
            )
        except ValueError as exc:
            logger.debug("Payload decode failed: %s", exc)
            return None

        # ── Compute consumed samples ──────────────────────────────────
        consumed = (
            sync_result.long_zc_start
            + self.sync.config.n_long * self.sps
            + (self.header_n_symbols + n_payload_symbols) * self.sps
        )

        return FrameResult(
            payload_bits=payload_bits,
            header=header,
            cfo_hz=sync_result.cfo_hat_hz,
            consumed_samples=consumed,
        )

    @property
    def max_frame_samples(self) -> int:
        """Maximum possible frame length in samples (for buffer sizing)."""
        cfg = self.sync.config
        max_preamble = cfg.n_short * cfg.n_short_reps + cfg.n_long
        max_header = self.fc.header_encoded_n_bits

        if self.pipeline.channel_coding:
            max_k = int(max(ldpc_get_supported_payload_lengths(CodeRates.HALF_RATE)))
            max_data_symbols = LDPCConfig(k=max_k, code_rate=CodeRates.HALF_RATE).n
        else:
            # header length field is 10 bits -> max 1023 payload bits + 16 CRC, pad to 12
            raw = (2**10 - 1) + FrameConstructor.PAYLOAD_CRC_BITS
            max_data_symbols = raw + (-raw % 12)

        if self.pipeline.pilots:
            max_payload = n_total_symbols(max_data_symbols, self.pilot_config)
        else:
            max_payload = max_data_symbols
        return (max_preamble + max_header + max_payload) * self.sps

    def _payload_n_symbols(self, header: FrameHeader) -> int | None:
        """Compute number of payload symbols (including pilots) from header, or None on error."""
        try:
            modulator = get_modulator(header.mod_scheme)
            n_coded = self.fc.payload_coded_n_bits(header, channel_coding=self.pipeline.channel_coding)
            n_data = n_coded // modulator.bits_per_symbol
            if self.pipeline.pilots:
                return n_total_symbols(n_data, self.pilot_config)
            return n_data
        except ValueError:
            logger.debug("Invalid payload parameters from header")
            return None


def run_receiver(
    sdr: SDRReceiver,
    decoder: FrameDecoder,
    h_rrc: np.ndarray | None,
    rx_buffer_size: int,
    on_frame: Callable[[FrameResult], None] | None = None,
    pipeline: PipelineConfig | None = None,
) -> None:
    """Continuously receive, matched-filter, and decode frames from the SDR.

    Runs an overlap-save matched filter feeding a multi-frame detection
    loop until interrupted by KeyboardInterrupt or SDR failure.
    """
    pipeline = pipeline or PIPELINE
    if on_frame is None:

        def _log_frame(result: FrameResult) -> None:
            logger.info(
                "RX: %r  (mod=%s, rate=%s, CFO=%+.0f Hz, bits=%d)",
                result.text,
                result.header.mod_scheme.name,
                result.header.coding_rate.name,
                result.cfo_hz,
                result.header.length,
            )

        on_frame = _log_frame

    max_frame_samples = decoder.max_frame_samples
    use_filter = pipeline.pulse_shaping and h_rrc is not None

    # Overlap-save state for incremental matched filtering
    if use_filter:
        filter_overlap = len(h_rrc) - 1
        filter_state = np.zeros(filter_overlap, dtype=complex)

    # Pre-allocated receive buffer
    buf_capacity = max_frame_samples + rx_buffer_size
    rx_buf = np.zeros(buf_capacity, dtype=complex)
    buf_len = 0
    abs_offset = 0

    try:
        while True:
            try:
                rx = sdr.rx()
            except Exception:
                logger.exception("RX: SDR read failed")
                break

            if use_filter:
                # ── Incremental matched filter (overlap-save) ─────────
                chunk = np.concatenate([filter_state, rx])
                new_filtered = np.convolve(chunk, h_rrc, mode="valid")
                filter_state = rx[-filter_overlap:]
            else:
                new_filtered = rx

            # Append to pre-allocated buffer
            n_new = len(new_filtered)
            if buf_len + n_new > buf_capacity:
                keep = max_frame_samples
                rx_buf[:keep] = rx_buf[buf_len - keep : buf_len]
                abs_offset += buf_len - keep
                buf_len = keep
            rx_buf[buf_len : buf_len + n_new] = new_filtered
            buf_len += n_new

            # ── Multi-frame detection loop ────────────────────────────
            while True:
                frame_result = decoder.try_decode(rx_buf[:buf_len], abs_offset)
                if frame_result is not None:
                    on_frame(frame_result)
                    consumed = frame_result.consumed_samples
                    remaining = buf_len - consumed
                    rx_buf[:remaining] = rx_buf[consumed:buf_len]
                    buf_len = remaining
                    abs_offset += consumed
                else:
                    break

            # Keep overlap for frames that may straddle buffer boundaries
            if buf_len > max_frame_samples:
                trim = buf_len - max_frame_samples
                rx_buf[:max_frame_samples] = rx_buf[trim:buf_len]
                buf_len = max_frame_samples
                abs_offset += trim
    except KeyboardInterrupt:
        pass


def create_decoder(pipeline: PipelineConfig | None = None) -> tuple[FrameDecoder, np.ndarray | None]:
    """Build FrameDecoder + h_rrc respecting pipeline toggles."""
    pipeline = pipeline or PIPELINE
    effective_sps = SPS if pipeline.pulse_shaping else 1
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS) if pipeline.pulse_shaping else None
    sync = Synchronizer(SYNC_CONFIG, sps=effective_sps, rrc_taps=h_rrc)
    fc = FrameConstructor()
    decoder = FrameDecoder(sync, fc, sample_rate=SAMPLE_RATE, sps=effective_sps, pilot_config=PILOT_CONFIG, pipeline=pipeline)
    return decoder, h_rrc


if __name__ == "__main__":
    from pluto import create_pluto
    from pluto.config import CENTER_FREQ, RX_BUFFER_SIZE

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr = create_pluto()
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE

    # ── Reusable objects ──────────────────────────────────────────────
    decoder, h_rrc = create_decoder()

    sdr.rx()  # flush stale DMA buffer

    logger.info("RX: listening on %.0f MHz ...", CENTER_FREQ / 1e6)
    run_receiver(sdr, decoder, h_rrc, RX_BUFFER_SIZE)
