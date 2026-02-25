"""Frame decoding: detect and decode a single frame from filtered samples.

Pipeline: sync -> CFO correct -> phase correct -> downsample -> normalize
       -> header decode -> equalize -> phase track -> soft demod -> channel decode
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.modulation import Modulator

import numpy as np

from modules.channel_coding import CodeRates, LDPCConfig, ldpc_get_supported_payload_lengths
from modules.costas_loop import apply_costas_loop
from modules.equalization import equalize_payload
from modules.frame_constructor import FrameConstructor, FrameHeader
from modules.modulation import BPSK
from modules.pilots import (
    PilotConfig,
    data_indices,
    n_total_symbols,
    pilot_aided_phase_track,
    pilot_indices,
)
from modules.pulse_shaping import rrc_filter
from modules.synchronization import Synchronizer
from modules.util import bits_to_bytes, bits_to_text
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

_HEADER_BPSK = BPSK()


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
        frame_constructor: FrameConstructor,
        sample_rate: float,
        pipeline: PipelineConfig,
    ) -> None:
        """Initialize with fixed radio parameters."""
        self.sync = sync
        self.frame_constructor = frame_constructor
        self.sample_rate = sample_rate
        self.pipeline = pipeline
        self.sps = sync.sps
        self.header_n_symbols = frame_constructor.header_encoded_n_bits // _HEADER_BPSK.bits_per_symbol
        self.pilot_config: PilotConfig = pipeline.pilot_config
        self.max_frame_samples = self._compute_max_frame_samples()

    @property
    def rrc_taps(self) -> np.ndarray | None:
        """RRC filter taps, or None when pulse shaping is disabled."""
        return self.sync.rrc_taps

    # type: ignore[union-attr]
    def try_decode(
        self,
        rx_filtered: np.ndarray,
        global_sample_offset: int,
    ) -> FrameResult | None:
        """Try to detect and decode a single frame from filtered samples."""
        detection = self.sync.detect_preamble(rx_filtered, self.sample_rate)
        if not detection.success:
            logger.debug("No preamble detected")
            return None

        samples_from_zc = rx_filtered[detection.long_zc_start :]
        global_start = global_sample_offset + detection.long_zc_start
        symbols = self._recover_symbols(samples_from_zc, detection.cfo_hat_hz, global_start)
        if len(symbols) < self.header_n_symbols:
            logger.debug("Not enough symbols for header")
            return None

        header, header_final_phase = self._demodulate_header(symbols)
        if header is None:
            return None

        try:
            modulator = get_modulator(header.mod_scheme)
        except ValueError:
            logger.debug("Unsupported modulation scheme: %s", header.mod_scheme)
            return None

        try:
            n_coded = self.frame_constructor.payload_coded_n_bits(header, channel_coding=self.pipeline.channel_coding)
        except ValueError:
            logger.debug("Invalid payload length in header: %d bits", header.length)
            return None

        n_data_symbols = n_coded // modulator.bits_per_symbol
        n_total_payload = n_total_symbols(n_data_symbols, self.pilot_config) if self.pipeline.pilots else n_data_symbols

        if len(symbols) < self.header_n_symbols + n_total_payload:
            logger.debug("Not enough symbols for payload")
            return None

        payload_bits = self._decode_payload(
            symbols,
            header,
            modulator,
            n_data_symbols,
            n_total_payload,
            header_final_phase,
        )
        if payload_bits is None:
            return None

        samples_before_zc = detection.long_zc_start
        zc_samples = self.sync.config.n_long * self.sps
        header_and_payload_samples = (self.header_n_symbols + n_total_payload) * self.sps
        consumed = samples_before_zc + zc_samples + header_and_payload_samples

        return FrameResult(
            payload_bits=payload_bits,
            header=header,
            cfo_hz=detection.cfo_hat_hz,
            consumed_samples=consumed,
        )

    def _recover_symbols(
        self,
        samples_from_zc: np.ndarray,
        cfo_hz: float,
        global_sample_start: int,
    ) -> np.ndarray:
        """CFO correct, phase correct, downsample, and normalize to symbol rate."""
        cfo_corrected = self._correct_cfo(samples_from_zc, cfo_hz, global_sample_start)
        phase_corrected = self._correct_residual_phase(cfo_corrected)

        # Skip the long ZC preamble and downsample from sample rate to symbol rate
        long_zc_samples = self.sync.config.n_long * self.sps
        symbols = phase_corrected[long_zc_samples :: self.sps]

        return self._normalize_amplitude(symbols)

    def _correct_cfo(
        self,
        samples: np.ndarray,
        cfo_hz: float,
        global_sample_start: int,
    ) -> np.ndarray:
        """Apply carrier frequency offset correction."""
        if not self.pipeline.cfo_correction:
            return samples
        sample_indices = global_sample_start + np.arange(len(samples))
        phase_increment = 2 * np.pi * cfo_hz / self.sample_rate
        return samples * np.exp(-1j * phase_increment * sample_indices)

    def _correct_residual_phase(self, cfo_corrected: np.ndarray) -> np.ndarray:
        """Remove residual phase offset estimated from the long ZC sequence.

        Downsamples internally to compare against the symbol-rate ZC reference,
        then applies the correction at the full sample rate.

        Assumes cfo_corrected starts at the first sample of the long ZC so
        that cfo_corrected[::sps] lands on symbol centres.
        """
        zc_long_ref = self.sync.zc_long
        zc_rx = cfo_corrected[:: self.sps][: len(zc_long_ref)]
        if len(zc_rx) != len(zc_long_ref):
            logger.debug("ZC length mismatch (%d vs %d), skipping phase correction", len(zc_rx), len(zc_long_ref))
            return cfo_corrected
        phase_hat = np.angle(np.sum(zc_rx * np.conj(zc_long_ref)))
        return cfo_corrected * np.exp(-1j * phase_hat)

    def _normalize_amplitude(self, symbols: np.ndarray) -> np.ndarray:
        """Normalize symbol amplitude using the known-power BPSK header."""
        if len(symbols) < self.header_n_symbols:
            logger.warning("Too few symbols (%d) to estimate header power, skipping normalization", len(symbols))
            return symbols
        header_power = np.mean(np.abs(symbols[: self.header_n_symbols]) ** 2)
        if header_power == 0:
            logger.warning("Header power is zero, skipping normalization")
            return symbols
        return symbols / np.sqrt(header_power)

    def _demodulate_header(self, symbols: np.ndarray) -> tuple[FrameHeader | None, float]:
        """Hard-demodulate the BPSK header, applying Costas loop if enabled.

        Returns the decoded header and the final Costas phase estimate (0.0 if
        the Costas loop is disabled).
        """
        header_symbols = symbols[: self.header_n_symbols]
        final_phase = 0.0
        if self.pipeline.costas_loop:
            header_symbols, phase_estimates = apply_costas_loop(symbols=header_symbols, config=COSTAS_CONFIG)
            final_phase = float(phase_estimates[-1])
        header_bits = _HEADER_BPSK.symbols2bits(header_symbols).flatten()
        try:
            return self.frame_constructor.decode_header(header_bits), final_phase
        except ValueError:
            logger.debug("Header CRC failed")
            return None, final_phase

    def _decode_payload(
        self,
        symbols: np.ndarray,
        header: FrameHeader,
        modulator: Modulator,
        n_data_symbols: int,
        n_total_payload_symbols: int,
        header_final_phase: float = 0.0,
    ) -> np.ndarray | None:
        """Demodulate and channel-decode the payload, returning decoded bits or None."""
        payload_symbols = symbols[self.header_n_symbols : self.header_n_symbols + n_total_payload_symbols]

        if self.pipeline.pilots:
            header_symbols = symbols[: self.header_n_symbols]
            payload_symbols = self._apply_pilot_corrections(payload_symbols, header_symbols, n_data_symbols)
        elif self.pipeline.costas_loop:
            payload_symbols, _ = apply_costas_loop(
                symbols=payload_symbols,
                config=COSTAS_CONFIG,
                current_phase_estimate=header_final_phase,
            )

        payload_sigma_sq = modulator.estimate_noise_variance(payload_symbols)
        payload_llrs = modulator.symbols2bits_soft(payload_symbols, sigma_sq=payload_sigma_sq).flatten()

        try:
            return self.frame_constructor.decode_payload(
                header,
                payload_llrs,
                soft=True,
                channel_coding=self.pipeline.channel_coding,
                interleaving=self.pipeline.interleaving,
            )
        except ValueError as exc:
            logger.debug("Payload decode failed: %s", exc)
            return None

    def _apply_pilot_corrections(
        self,
        payload_symbols: np.ndarray,
        header_symbols: np.ndarray,
        n_data_symbols: int,
    ) -> np.ndarray:
        """Equalize and phase-track payload using pilot symbols and header noise estimate."""
        header_sigma_sq = _HEADER_BPSK.estimate_noise_variance(header_symbols)
        p_idx = pilot_indices(n_data_symbols, self.pilot_config)
        d_idx = data_indices(n_data_symbols, self.pilot_config)

        payload_symbols = equalize_payload(
            payload_symbols,
            n_data_symbols,
            self.pilot_config,
            header_sigma_sq,
            p_idx=p_idx,
        )
        return pilot_aided_phase_track(
            payload_symbols,
            n_data_symbols,
            self.pilot_config,
            p_idx=p_idx,
            d_idx=d_idx,
        )

    def _compute_max_frame_samples(self) -> int:
        """Worst-case frame length in samples.

        The receive buffer must be large enough to hold at least one complete
        frame regardless of modulation scheme, code rate, or pilot configuration.
        This computes that upper bound so the receive buffer works correctly.
        """
        cfg = self.sync.config
        max_preamble_symbols = cfg.n_short * cfg.n_short_reps + cfg.n_long
        max_header_symbols = self.header_n_symbols

        if self.pipeline.channel_coding:
            all_coded_rates = [r for r in CodeRates if r != CodeRates.NONE]
            max_coded_bits = max(
                LDPCConfig(k=int(max(ldpc_get_supported_payload_lengths(rate))), code_rate=rate).n
                for rate in all_coded_rates
            )
        else:
            max_payload_bits = 2**self.frame_constructor.header_config.payload_length_bits - 1
            raw_bits = max_payload_bits + FrameConstructor.PAYLOAD_CRC_BITS
            pad = FrameConstructor.PAYLOAD_PAD_MULTIPLE
            max_coded_bits = raw_bits + (-raw_bits % pad)

        max_data_symbols = max_coded_bits // 1  # BPSK (1 bit/symbol) is the worst case

        if self.pipeline.pilots:
            max_payload_symbols = n_total_symbols(max_data_symbols, self.pilot_config)
        else:
            max_payload_symbols = max_data_symbols

        max_frame_symbols = max_preamble_symbols + max_header_symbols + max_payload_symbols
        return max_frame_symbols * self.sps


def create_decoder(
    pipeline: PipelineConfig | None = None,
) -> FrameDecoder:
    """Build a FrameDecoder respecting pipeline toggles."""
    pipeline = pipeline or PIPELINE
    effective_sps = SPS if pipeline.pulse_shaping else 1
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS) if pipeline.pulse_shaping else None
    sync = Synchronizer(SYNC_CONFIG, sps=effective_sps, rrc_taps=h_rrc)
    frame_constructor = FrameConstructor()
    return FrameDecoder(
        sync,
        frame_constructor,
        sample_rate=SAMPLE_RATE,
        pipeline=pipeline,
    )
