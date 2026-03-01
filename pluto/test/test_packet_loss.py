"""Packet loss diagnostic tool.

Sends numbered packets at a configurable rate from TX and tracks
what arrives at RX, classifying losses into:
  - Missed detections (frame never triggered preamble detection)
  - Header decode failures (preamble detected but header CRC failed)
  - Payload decode failures (header OK but payload CRC failed)
  - Successful decodes

Usage (two terminals via setup_namespaces.sh):
    # Terminal 1: transmitter — send 100 packets, 200ms apart
    sudo ./setup_namespaces.sh exec A uv run python -m pluto.test.test_packet_loss \
        tx --pluto-ip 192.168.2.1 --interval 0.2 --count 100

    # Terminal 2: receiver — listen and report stats
    sudo ./setup_namespaces.sh exec B uv run python -m pluto.test.test_packet_loss \
        rx --pluto-ip 192.168.3.1 --cfo-offset 15503

    # Or with the helper:
    sudo ./setup_namespaces.sh exec A uv run python -m pluto.test.test_packet_loss tx
    sudo ./setup_namespaces.sh exec B uv run python -m pluto.test.test_packet_loss rx --cfo-offset 15503
"""

from __future__ import annotations

import argparse
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.util import bytes_to_bits, text_to_bits
from pluto import create_pluto
from pluto.config import (
    CENTER_FREQ,
    CODING_RATE,
    DAC_SCALE,
    DEFAULT_TX_GAIN,
    MOD_SCHEME,
    NODE_DST,
    NODE_SRC,
    PIPELINE,
    SAMPLE_RATE,
    configure_rx,
    configure_tx,
    get_modulator,
)
from pluto.decode import FrameDecoder, FrameResult, _HEADER_BPSK, create_decoder
from pluto.receive import _MatchedFilter, _RxBuffer
from pluto.transmit import build_tx_signal_from_bits, max_payload_bits

logger = logging.getLogger(__name__)

# ── Payload format ────────────────────────────────────────────────────────
# We embed a 32-bit sequence number in the payload so we can track packets
# beyond the 4-bit header sequence_number (which wraps at 16).
# Format: [4 bytes big-endian seq] [padding to fill reasonable size]
PAYLOAD_SEQ_BYTES = 4
PAYLOAD_TOTAL_BYTES = 32  # small but enough to test the full pipeline
PAYLOAD_TOTAL_BITS = PAYLOAD_TOTAL_BYTES * 8


def _encode_seq_payload(seq: int) -> np.ndarray:
    """Encode a sequence number into a fixed-size payload bit array."""
    payload = struct.pack(">I", seq) + b"\x00" * (PAYLOAD_TOTAL_BYTES - PAYLOAD_SEQ_BYTES)
    return bytes_to_bits(payload)


def _decode_seq_payload(payload_bits: np.ndarray) -> int | None:
    """Extract the sequence number from a payload, or None if too short."""
    if len(payload_bits) < PAYLOAD_SEQ_BYTES * 8:
        return None
    seq_bits = payload_bits[: PAYLOAD_SEQ_BYTES * 8]
    seq_bytes = bytes(
        int("".join(str(int(b)) for b in seq_bits[i : i + 8]), 2) for i in range(0, PAYLOAD_SEQ_BYTES * 8, 8)
    )
    return struct.unpack(">I", seq_bytes)[0]


# ── Failure classification ────────────────────────────────────────────────


class DecodeStage(Enum):
    """Where in the decode pipeline a frame attempt ended."""

    NO_PREAMBLE = auto()
    INSUFFICIENT_SYMBOLS = auto()
    HEADER_CRC_FAIL = auto()
    INVALID_HEADER = auto()
    INSUFFICIENT_PAYLOAD = auto()
    PAYLOAD_CRC_FAIL = auto()
    SUCCESS = auto()


@dataclass
class DecodeAttempt:
    """Result of a single decode attempt with stage information."""

    stage: DecodeStage
    result: FrameResult | None = None
    cfo_hz: float | None = None


class InstrumentedDecoder:
    """Wraps FrameDecoder to report *why* decoding failed, not just None."""

    def __init__(self, decoder: FrameDecoder) -> None:
        self.decoder = decoder
        self.sync = decoder.sync
        self.pipeline = decoder.pipeline
        self.frame_constructor = decoder.frame_constructor
        self.header_n_symbols = decoder.header_n_symbols
        self.pilot_config = decoder.pilot_config
        self.sps = decoder.sps
        self.sample_rate = decoder.sample_rate

    @property
    def rrc_taps(self) -> np.ndarray | None:
        return self.decoder.rrc_taps

    def try_decode(
        self,
        rx_filtered: np.ndarray,
        global_sample_offset: int,
    ) -> DecodeAttempt:
        """Like FrameDecoder.try_decode but returns stage info on failure."""
        detection = self.sync.detect_preamble(rx_filtered, self.sample_rate)
        if not detection.success:
            return DecodeAttempt(DecodeStage.NO_PREAMBLE)

        samples_from_zc = rx_filtered[detection.long_zc_start :]
        global_start = global_sample_offset + detection.long_zc_start
        symbols = self.decoder._recover_symbols(samples_from_zc, detection.cfo_hat_hz, global_start)

        if len(symbols) < self.header_n_symbols:
            return DecodeAttempt(DecodeStage.INSUFFICIENT_SYMBOLS, cfo_hz=detection.cfo_hat_hz)

        header, header_final_phase = self.decoder._demodulate_header(symbols)
        if header is None:
            return DecodeAttempt(DecodeStage.HEADER_CRC_FAIL, cfo_hz=detection.cfo_hat_hz)

        try:
            modulator = get_modulator(header.mod_scheme)
            n_coded = self.frame_constructor.payload_coded_n_bits(
                header, channel_coding=self.pipeline.channel_coding
            )
        except ValueError:
            return DecodeAttempt(DecodeStage.INVALID_HEADER, cfo_hz=detection.cfo_hat_hz)

        from modules.pilots import n_total_symbols

        n_data_symbols = n_coded // modulator.bits_per_symbol
        n_total_payload = (
            n_total_symbols(n_data_symbols, self.pilot_config) if self.pipeline.pilots else n_data_symbols
        )

        if len(symbols) < self.header_n_symbols + n_total_payload:
            return DecodeAttempt(DecodeStage.INSUFFICIENT_PAYLOAD, cfo_hz=detection.cfo_hat_hz)

        payload_bits = self.decoder._decode_payload(
            symbols, header, modulator, n_data_symbols, n_total_payload, header_final_phase
        )

        # Compute consumed samples for buffer advancement
        samples_before_zc = detection.long_zc_start
        zc_samples = self.sync.config.n_long * self.sps
        header_and_payload_samples = (self.header_n_symbols + n_total_payload) * self.sps
        consumed = samples_before_zc + zc_samples + header_and_payload_samples

        if payload_bits is None:
            # Build a partial result so we can still consume the right number of samples
            partial = FrameResult(
                payload_bits=np.array([], dtype=int),
                header=header,
                cfo_hz=detection.cfo_hat_hz,
                consumed_samples=consumed,
            )
            return DecodeAttempt(DecodeStage.PAYLOAD_CRC_FAIL, result=partial, cfo_hz=detection.cfo_hat_hz)

        result = FrameResult(
            payload_bits=payload_bits,
            header=header,
            cfo_hz=detection.cfo_hat_hz,
            consumed_samples=consumed,
        )
        return DecodeAttempt(DecodeStage.SUCCESS, result=result, cfo_hz=detection.cfo_hat_hz)


# ── RX stats tracking ─────────────────────────────────────────────────────


@dataclass
class RxStats:
    """Accumulate and display receive statistics."""

    start_time: float = field(default_factory=time.time)
    success: int = 0
    header_fail: int = 0
    payload_fail: int = 0
    no_preamble_chunks: int = 0  # RX chunks with no preamble detected
    insufficient_symbols: int = 0
    invalid_header: int = 0
    insufficient_payload: int = 0
    received_seqs: set[int] = field(default_factory=set)
    max_seq_seen: int = -1
    cfo_values: list[float] = field(default_factory=list)

    def record(self, attempt: DecodeAttempt) -> None:
        stage = attempt.stage
        if attempt.cfo_hz is not None:
            self.cfo_values.append(attempt.cfo_hz)

        if stage == DecodeStage.SUCCESS:
            self.success += 1
            if attempt.result is not None:
                seq = _decode_seq_payload(attempt.result.payload_bits)
                if seq is not None:
                    self.received_seqs.add(seq)
                    self.max_seq_seen = max(self.max_seq_seen, seq)
        elif stage == DecodeStage.HEADER_CRC_FAIL:
            self.header_fail += 1
        elif stage == DecodeStage.PAYLOAD_CRC_FAIL:
            self.payload_fail += 1
        elif stage == DecodeStage.NO_PREAMBLE:
            self.no_preamble_chunks += 1
        elif stage == DecodeStage.INSUFFICIENT_SYMBOLS:
            self.insufficient_symbols += 1
        elif stage == DecodeStage.INVALID_HEADER:
            self.invalid_header += 1
        elif stage == DecodeStage.INSUFFICIENT_PAYLOAD:
            self.insufficient_payload += 1

    @property
    def detected(self) -> int:
        """Total frames where preamble was found."""
        return self.success + self.header_fail + self.payload_fail + self.invalid_header + self.insufficient_payload

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def report(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  Packet Loss Report  ({self.elapsed:.1f}s elapsed)",
            "=" * 60,
            "",
            f"  Preamble detected:         {self.detected}",
            f"    → Header CRC fail:       {self.header_fail}",
            f"    → Invalid header fields:  {self.invalid_header}",
            f"    → Insufficient payload:   {self.insufficient_payload}",
            f"    → Payload CRC fail:       {self.payload_fail}",
            f"    → Successfully decoded:   {self.success}",
            "",
        ]

        if self.max_seq_seen >= 0:
            expected = self.max_seq_seen + 1
            missed = expected - len(self.received_seqs)
            not_detected = missed - (self.header_fail + self.payload_fail + self.invalid_header)
            not_detected = max(0, not_detected)  # floor at 0

            lines += [
                f"  Sequence numbers seen:      0–{self.max_seq_seen} ({expected} expected)",
                f"    → Received unique:        {len(self.received_seqs)}",
                f"    → Missing (total):        {missed}",
                f"    → Est. not detected:      {not_detected}",
                "",
            ]

            if expected > 0:
                lines += [
                    f"  Packet success rate:        {len(self.received_seqs)/expected*100:.1f}%",
                    f"  Detection rate:             {self.detected/expected*100:.1f}%  (of expected)",
                    f"  Header decode rate:         {(self.detected - self.header_fail)/max(1,self.detected)*100:.1f}%  (of detected)",
                    f"  Payload decode rate:        {self.success/max(1,self.detected - self.header_fail)*100:.1f}%  (of header OK)",
                    "",
                ]

        if self.cfo_values:
            arr = np.array(self.cfo_values)
            lines += [
                f"  CFO:  mean={np.mean(arr):+.0f} Hz  std={np.std(arr):.0f} Hz  "
                f"range=[{np.min(arr):+.0f}, {np.max(arr):+.0f}] Hz",
                "",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)


# ── TX mode ───────────────────────────────────────────────────────────────


def run_tx(pluto_ip: str, tx_gain: float, interval: float, count: int) -> None:
    """Transmit numbered test packets at a fixed interval."""
    sdr = create_pluto(f"ip:{pluto_ip}")
    configure_tx(sdr, gain=tx_gain, cyclic=True)

    frame_constructor = FrameConstructor()
    max_bits = max_payload_bits(CODING_RATE)

    # Pre-build all signals for consistent timing
    logger.info("TX: pre-building %d packets...", count)
    signals = []
    for seq in range(count):
        payload_bits = _encode_seq_payload(seq)
        if len(payload_bits) > max_bits:
            logger.error("TX: payload %d bits > max %d bits", len(payload_bits), max_bits)
            return
        tx_signal = build_tx_signal_from_bits(
            payload_bits,
            frame_constructor,
            MOD_SCHEME,
            CODING_RATE,
            sequence_number=seq % 16,  # 4-bit header field
        )
        samples = tx_signal * DAC_SCALE
        signals.append(samples)

    # All signals must be the same length for PlutoSDR
    max_len = max(len(s) for s in signals)
    sdr.tx_buffer_size = max_len
    silence = np.zeros(max_len, dtype=complex)

    # How long one packet takes to transmit at sample rate
    tx_duration = max_len / SAMPLE_RATE
    # Transmit for 3x the buffer duration to ensure it goes out
    tx_time = max(0.05, tx_duration * 3)

    logger.info(
        "TX: sending %d packets at %.0fms intervals (gain=%.0f dB, %.1fms TX window)",
        count, interval * 1000, tx_gain, tx_time * 1000,
    )
    try:
        for seq in range(count):
            padded = np.zeros(max_len, dtype=complex)
            padded[: len(signals[seq])] = signals[seq]
            sdr.tx(padded)  # starts cycling this packet
            logger.info("TX: seq=%d/%d", seq, count - 1)
            time.sleep(interval)
            sdr.tx_destroy_buffer()
    except KeyboardInterrupt:
        logger.info("TX: interrupted at seq=%d", seq)
    finally:
        sdr.tx_destroy_buffer()
        logger.info("TX: done")

# ── RX mode ───────────────────────────────────────────────────────────────


def run_rx(pluto_ip: str, cfo_offset: int, duration: float | None) -> None:
    """Receive and classify packets, printing a report on exit."""
    sdr = create_pluto(f"ip:{pluto_ip}")
    rx_freq = CENTER_FREQ + cfo_offset
    configure_rx(sdr, freq=rx_freq)

    base_decoder = create_decoder(PIPELINE)
    decoder = InstrumentedDecoder(base_decoder)
    matched_filter = _MatchedFilter(decoder.rrc_taps)
    rx_buffer = _RxBuffer()

    stats = RxStats()

    logger.info("RX: listening on %.0f Hz (CFO offset %+d Hz)...", rx_freq, cfo_offset)
    logger.info("RX: Ctrl-C to stop and print report")

    sdr.rx()  # flush stale DMA

    deadline = time.time() + duration if duration else None

    try:
        while True:
            if deadline and time.time() > deadline:
                break

            try:
                rx = sdr.rx()
            except OSError:
                logger.exception("RX: SDR read failed")
                break

            rx_buffer.append(matched_filter(rx))

            while True:
                attempt = decoder.try_decode(rx_buffer.samples, rx_buffer.sample_offset)
                stats.record(attempt)

                if attempt.stage == DecodeStage.NO_PREAMBLE:
                    break

                # Consume samples for any detection (successful or not)
                if attempt.result is not None:
                    rx_buffer.consume(attempt.result.consumed_samples)
                    if attempt.stage == DecodeStage.SUCCESS:
                        seq = _decode_seq_payload(attempt.result.payload_bits)
                        logger.info(
                            "RX: seq=%s  CFO=%+.0f Hz  (%s)",
                            seq,
                            attempt.result.cfo_hz,
                            attempt.result.header.mod_scheme.name,
                        )
                else:
                    # Header failed — skip past the detected preamble region
                    # Use a conservative skip to avoid getting stuck
                    skip = decoder.sync.config.n_long * decoder.sps * 2
                    rx_buffer.consume(min(skip, len(rx_buffer.samples) // 2))

    except KeyboardInterrupt:
        pass

    print(stats.report())


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Packet loss diagnostic tool")
    sub = parser.add_subparsers(dest="mode", required=True)

    tx_parser = sub.add_parser("tx", help="Transmit numbered test packets")
    tx_parser.add_argument("--pluto-ip", default="192.168.2.1")
    tx_parser.add_argument("--tx-gain", type=float, default=-50)
    tx_parser.add_argument("--interval", type=float, default=0.2, help="Seconds between packets (default: 0.2)")
    tx_parser.add_argument("--count", type=int, default=100, help="Number of packets to send (default: 100)")

    rx_parser = sub.add_parser("rx", help="Receive and classify packets")
    rx_parser.add_argument("--pluto-ip", default="192.168.2.1")
    rx_parser.add_argument("--cfo-offset", type=int, default=0, help="RX CFO offset in Hz")
    rx_parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: run until Ctrl-C)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.mode == "tx":
        run_tx(args.pluto_ip, args.tx_gain, args.interval, args.count)
    else:
        run_rx(args.pluto_ip, args.cfo_offset, args.duration)


if __name__ == "__main__":
    main()
