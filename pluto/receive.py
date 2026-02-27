"""SDR receive loop: matched filter, buffering, and continuous frame decoding."""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

import numpy as np

from pluto import create_pluto
from pluto.config import CENTER_FREQ, configure_rx
from pluto.decode import create_decoder

if TYPE_CHECKING:
    from collections.abc import Callable

    from pluto import SDRReceiver
    from pluto.decode import FrameDecoder, FrameResult

logger = logging.getLogger(__name__)


class _MatchedFilter:
    """Overlap-save matched filter.  Pass-through when no taps are provided."""

    def __init__(self, h_rrc: np.ndarray | None) -> None:
        self._h = h_rrc
        if h_rrc is not None:
            self._overlap = len(h_rrc) - 1
            self._state = np.zeros(self._overlap, dtype=complex)

    def __call__(self, rx: np.ndarray) -> np.ndarray:
        if self._h is None:
            return rx
        # Overlap-save: prepend previous tail so the convolution is
        # continuous across calls, then save the new tail for next time.
        chunk = np.concatenate([self._state, rx])
        self._state = chunk[-self._overlap :]
        return np.convolve(chunk, self._h, mode="valid")


class _RxBuffer:
    """Sliding window over a sample stream.

    Tracks an absolute sample offset into the stream.  Consume is O(1);
    the compaction is folded into the next append.
    """

    def __init__(self) -> None:
        self._buffer = np.empty(0, dtype=complex)
        self._start = 0
        self._sample_offset = 0

    @property
    def sample_offset(self) -> int:
        """Absolute sample offset into the stream."""
        return self._sample_offset

    @property
    def samples(self) -> np.ndarray:
        """Zero-copy view of the current buffered samples."""
        return self._buffer[self._start :]

    def append(self, samples: np.ndarray) -> None:
        """Append new samples, compacting any discarded prefix in one pass."""
        if self._start > 0:
            self._buffer = np.concatenate([self._buffer[self._start :], samples])
            self._start = 0
        else:
            self._buffer = np.concatenate([self._buffer, samples])

    def consume(self, n: int) -> None:
        """Discard the first *n* samples and advance the stream offset (O(1))."""
        self._start += n
        self._sample_offset += n


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
    on_frame: Callable[[FrameResult], None] | None = None,
) -> None:
    """Continuously receive, matched-filter, and decode frames from the SDR.

    Flushes the stale DMA buffer, then runs an overlap-save matched filter
    feeding a frame-detection loop until KeyboardInterrupt or SDR failure.
    """
    on_frame = on_frame or _default_frame_callback
    matched_filter = _MatchedFilter(decoder.rrc_taps)
    rx_buffer = _RxBuffer()
    sdr.rx()  # PlutoSDR returns stale DMA data on first read; discard it

    try:
        while True:
            try:
                rx = sdr.rx()
            except OSError:
                logger.exception("RX: SDR read failed — stopping receiver")
                break

            rx_buffer.append(matched_filter(rx))

            while True:
                frame = decoder.try_decode(rx_buffer.samples, rx_buffer.sample_offset)
                if frame is None:
                    break
                on_frame(frame)
                rx_buffer.consume(frame.consumed_samples)

    except KeyboardInterrupt:
        logger.info("RX: stopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive frames from PlutoSDR")
    parser.add_argument(
        "--cfo-offset",
        type=int,
        default=0,
        help="CFO offset in Hz to add to RX LO (compensate for TX oscillator drift)",
    )
    parser.add_argument("--pluto-ip", default="192.168.2.1", help="PlutoSDR IP address (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rx_freq = CENTER_FREQ + args.cfo_offset
    sdr = create_pluto(f"ip:{args.pluto_ip}")
    configure_rx(sdr, freq=rx_freq)

    decoder = create_decoder()

    logger.info("RX: listening on %.0f Hz (offset %+d Hz)...", rx_freq, args.cfo_offset)
    run_receiver(sdr, decoder)
