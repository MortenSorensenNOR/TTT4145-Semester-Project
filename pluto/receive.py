"""Receive and decode a framed digital signal from the PlutoSDR.

Pipeline: SDR -> matched filter -> sync -> CFO correct -> phase correct
       -> downsample -> normalize -> header decode -> equalize -> phase track
       -> soft demod -> channel decode -> text
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from modules.channel_coding import CodeRates, LDPCConfig, ldpc_get_supported_payload_lengths
from modules.costas_loop import apply_costas_loop
from modules.equalization import equalize_payload
from modules.frame_constructor import FrameConstructor, FrameHeader
from modules.modulation import BPSK, estimate_noise_variance
from modules.pilots import (
    PilotConfig,
    data_indices,
    n_total_symbols,
    pilot_aided_phase_track,
    pilot_indices,
)
from modules.pulse_shaping import rrc_filter
from modules.synchronization import SynchronizationResult, Synchronizer
from modules.util import bits_to_bytes, bits_to_text
from pluto import SDRReceiver
from pluto.config import (
    COSTAS_CONFIG,
    PIPELINE,
    RRC_ALPHA,
    RRC_NUM_TAPS,
    SAMPLE_RATE,
    SPS,
    SYNC_CONFIG,
    PipelineConfig,
    get_modulator,
)

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
        pipeline: PipelineConfig | None = None,
    ) -> None:
        """Initialize with fixed radio parameters."""
        self.sync = sync
        self.fc = fc
        self.sample_rate = sample_rate
        self.sps = sps
        self.pipeline = pipeline or PipelineConfig()
        self.pilot_config = self.pipeline.pilot_config
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
        header_symbols = symbols[: self.header_n_symbols]

        # ── Apply costas phase correction (only implemented for BPSK at the moment)──
        if self.pipeline.costas_loop:
            header_symbols, _ = apply_costas_loop(symbols=header_symbols, config=COSTAS_CONFIG)

        header_hard = self.bpsk.symbols2bits(header_symbols).flatten()
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

        # ── Apply costas phase correction ─────────────────────────────
        # Apply here when implemented for QPSK as well and then remove from header above

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
            pilot_cfg = cast("PilotConfig", self.pilot_config)
            p_idx = pilot_indices(n_data, pilot_cfg)
            d_idx = data_indices(n_data, pilot_cfg)

            # ── Equalize first (before any amplitude modification) ────
            payload_symbols = equalize_payload(payload_symbols, n_data, pilot_cfg, sigma_sq, p_idx=p_idx)

            # ── Pilot-aided phase tracking (returns data-only symbols) ─
            payload_symbols = pilot_aided_phase_track(
                payload_symbols,
                n_data,
                pilot_cfg,
                p_idx=p_idx,
                d_idx=d_idx,
            )

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

        max_payload = n_total_symbols(max_data_symbols, cast("PilotConfig", self.pilot_config)) if self.pipeline.pilots else max_data_symbols
        return (max_preamble + max_header + max_payload) * self.sps

    def _payload_n_symbols(self, header: FrameHeader) -> int | None:
        """Compute number of payload symbols (including pilots) from header, or None on error."""
        try:
            modulator = get_modulator(header.mod_scheme)
            n_coded = self.fc.payload_coded_n_bits(header, channel_coding=self.pipeline.channel_coding)
            n_data = n_coded // modulator.bits_per_symbol
        except ValueError:
            logger.debug("Invalid payload parameters from header")
            return None
        else:
            if self.pipeline.pilots:
                return n_total_symbols(n_data, cast("PilotConfig", self.pilot_config))
            return n_data


@dataclass
class _MatchedFilter:
    """Overlap-save matched filter state for incremental RRC filtering."""

    h_rrc: np.ndarray
    overlap: int
    state: np.ndarray

    @classmethod
    def create(cls, h_rrc: np.ndarray) -> _MatchedFilter:
        """Build a new matched filter from RRC taps."""
        overlap = len(h_rrc) - 1
        return cls(h_rrc=h_rrc, overlap=overlap, state=np.zeros(overlap, dtype=complex))

    def apply(self, rx: np.ndarray) -> np.ndarray:
        """Apply the overlap-save matched filter and update state."""
        chunk = np.concatenate([self.state, rx])
        filtered = np.convolve(chunk, self.h_rrc, mode="valid")
        self.state = rx[-self.overlap :]
        return filtered


@dataclass
class _RxBuffer:
    """Pre-allocated receive buffer with overlap management."""

    buf: np.ndarray
    capacity: int
    length: int
    abs_offset: int
    max_frame_samples: int

    @classmethod
    def create(cls, max_frame_samples: int, rx_buffer_size: int) -> _RxBuffer:
        """Build a new receive buffer."""
        capacity = max_frame_samples + rx_buffer_size
        return cls(
            buf=np.zeros(capacity, dtype=complex),
            capacity=capacity,
            length=0,
            abs_offset=0,
            max_frame_samples=max_frame_samples,
        )

    def append(self, new_filtered: np.ndarray) -> None:
        """Append filtered samples, compacting if the buffer is full."""
        n_new = len(new_filtered)
        if self.length + n_new > self.capacity:
            keep = self.max_frame_samples
            self.buf[:keep] = self.buf[self.length - keep : self.length]
            self.abs_offset += self.length - keep
            self.length = keep
        self.buf[self.length : self.length + n_new] = new_filtered
        self.length += n_new

    def consume(self, n: int) -> None:
        """Remove the first *n* samples from the buffer."""
        consumed = min(n, self.length)
        remaining = self.length - consumed
        if remaining > 0:
            self.buf[:remaining] = self.buf[consumed : self.length]
        self.length = remaining
        self.abs_offset += consumed

    def trim_overlap(self) -> None:
        """Trim the buffer to keep at most one max-frame of overlap."""
        if self.length > self.max_frame_samples:
            trim = self.length - self.max_frame_samples
            self.buf[: self.max_frame_samples] = self.buf[trim : self.length]
            self.length = self.max_frame_samples
            self.abs_offset += trim


def _default_frame_callback(result: FrameResult) -> None:
    """Log a successfully decoded frame."""
    logger.info(
        "RX: %r  (mod=%s, rate=%s, CFO=%+.0f Hz, bits=%d)",
        result.text,
        result.header.mod_scheme.name,
        result.header.coding_rate.name,
        result.cfo_hz,
        result.header.length,
    )


def run_receiver(
    sdr: SDRReceiver,
    decoder: FrameDecoder,
    h_rrc: np.ndarray | None,
    rx_buffer_size: int,
    on_frame: Callable[[FrameResult], None] | None = None,
) -> None:
    """Continuously receive, matched-filter, and decode frames from the SDR.

    Runs an overlap-save matched filter feeding a multi-frame detection
    loop until interrupted by KeyboardInterrupt or SDR failure.
    """
    on_frame = on_frame or _default_frame_callback
    use_filter = decoder.pipeline.pulse_shaping and h_rrc is not None
    mf = _MatchedFilter.create(cast("np.ndarray", h_rrc)) if use_filter else None
    rxbuf = _RxBuffer.create(decoder.max_frame_samples, rx_buffer_size)

    try:
        while True:
            try:
                rx = sdr.rx()
            except Exception:
                logger.exception("RX: SDR read failed")
                break

            new_filtered = mf.apply(rx) if mf is not None else rx
            rxbuf.append(new_filtered)

            # Multi-frame detection loop
            while True:
                frame_result = decoder.try_decode(rxbuf.buf[: rxbuf.length], rxbuf.abs_offset)
                if frame_result is None:
                    break
                on_frame(frame_result)
                rxbuf.consume(frame_result.consumed_samples)

            rxbuf.trim_overlap()
    except KeyboardInterrupt:
        pass


def create_decoder(pipeline: PipelineConfig | None = None) -> tuple[FrameDecoder, np.ndarray | None]:
    """Build FrameDecoder + h_rrc respecting pipeline toggles."""
    pipeline = pipeline or PIPELINE
    effective_sps = SPS if pipeline.pulse_shaping else 1
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS) if pipeline.pulse_shaping else None
    sync = Synchronizer(SYNC_CONFIG, sps=effective_sps, rrc_taps=h_rrc)
    fc = FrameConstructor()
    decoder = FrameDecoder(
        sync,
        fc,
        sample_rate=SAMPLE_RATE,
        sps=effective_sps,
        pipeline=pipeline,
    )
    return decoder, h_rrc


if __name__ == "__main__":
    import argparse

    from pluto import create_pluto
    from pluto.config import CENTER_FREQ, RX_BUFFER_SIZE

    parser = argparse.ArgumentParser(description="Receive frames from PlutoSDR")
    parser.add_argument(
        "--cfo-offset",
        type=int,
        default=0,
        help="CFO offset in Hz to add to RX LO (compensate for TX oscillator drift)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # ── SDR setup ─────────────────────────────────────────────────────
    rx_freq = CENTER_FREQ + args.cfo_offset
    sdr = create_pluto()
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_lo = int(rx_freq)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE

    # ── Reusable objects ──────────────────────────────────────────────
    decoder, h_rrc = create_decoder()

    sdr.rx()  # flush stale DMA buffer

    logger.info("RX: listening on %.0f Hz (offset %+d Hz)...", rx_freq, args.cfo_offset)
    run_receiver(sdr, decoder, h_rrc, RX_BUFFER_SIZE)
