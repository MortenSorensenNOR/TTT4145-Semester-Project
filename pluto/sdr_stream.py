"""TX and RX stream helpers for PlutoSDR.

RxStream — continuous hardware buffer drain
    Decouples buffer collection from processing so the DMA ring never stalls.
    Supports a lossless mode (blocks instead of dropping) for capture tests.

    Typical usage::

        stream = RxStream(sdr)
        stream.start(flush=10)   # drain stale DMA, let AGC settle
        while True:
            samples = stream.get()   # complex64, normalised
            packets = rx_pipe.receive(samples)
        stream.stop()

TxStream — continuous queue-based transmit
    Maintains a background packer thread that continuously pushes fixed-size
    buffers to the SDR.  Callers enqueue packet samples via send(); the packer
    greedily packs as many queued packets as possible into each buffer (FIFO).
    When the queue is empty, silence (zeros) is transmitted to keep the TX
    stream alive — symmetric with RxStream's continuous drain.

    Typical usage::

        stream = TxStream(sdr, sample_rate=pipe_cfg.SAMPLE_RATE, buf_size=32768)
        stream.start()
        for pkt in packets:
            samples = tx_pipe.transmit(pkt)
            samples /= np.max(np.abs(samples))   # normalise to [-1, 1]
            stream.send(samples)
        stream.stop()
"""

import queue
import threading
import logging
logger = logging.getLogger(__name__)

import numpy as np
import adi
from pluto.config import DAC_SCALE


_SCALE = np.float32(2.0 / DAC_SCALE)

class RxStream:
    """Continuously drains PlutoSDR hardware buffers in a background thread.

    get() returns the next buffer as normalised complex64, with typically
    1-3 ms of wait vs ~18 ms for a bare sdr.rx() call.

    Two modes:
      - lossless=False (default): drops the oldest buffer when the queue is
        full so the producer never stalls.  Best for real-time processing
        where latency matters more than completeness.
      - lossless=True: blocks when the queue is full.  Use a large maxsize
        to avoid stalling the hardware reader.  Best for capture scenarios
        where every buffer must be processed (e.g. loopback tests).
    """

    def __init__(self, sdr: adi.Pluto, maxsize: int = 2, lossless: bool = False):
        """
        Args:
            sdr:      configured adi.Pluto instance (rx_buffer_size already set)
            maxsize:  number of prefetched buffers to keep in flight
            lossless: block when the queue is full instead of dropping the
                      oldest buffer; use a large maxsize to avoid stalling
        """
        self._sdr = sdr
        self._lossless = lossless
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._overruns = 0

    def start(self, flush: int = 0) -> None:
        """Start the background producer thread.

        Args:
            flush: number of hardware buffers to read synchronously (and
                   discard) before the background thread is started — drains
                   stale DMA data and lets the AGC settle.
        """

        for _ in range(flush):
            self._rx()
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        """Stop the background producer thread and join it."""
        self._stop.set()
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join()

    def get(self, timeout: float = 2.0) -> np.ndarray:
        """Return the next buffer as normalised complex64 samples."""
        return self._q.get(timeout=timeout)

    def flush(self, n: int = 10) -> None:
        """Discard n buffers (requires the thread to already be started)."""
        for _ in range(n):
            self.get()

    @property
    def overruns(self) -> int:
        """Number of buffers dropped because the consumer was too slow."""
        return self._overruns

    # ------------------------------------------------------------------

    def _producer(self) -> None:
        while not self._stop.is_set():
            buf = self._rx()
            if self._lossless:
                # Block until the consumer drains the queue, but still honour stop().
                while not self._stop.is_set():
                    try:
                        self._q.put(buf, timeout=0.1)
                        break
                    except queue.Full:
                        pass
            else:
                try:
                    self._q.put_nowait(buf)
                except queue.Full:
                    # Consumer is behind — drop oldest, keep newest
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                    self._q.put_nowait(buf)
                    self._overruns += 1

    def _rx(self) -> np.ndarray:
        """Read one buffer directly from libiio, bypassing chan.read().

        buf.read() returns the raw interleaved bytes [I0,Q0,I1,Q1,...] in a
        single call, saving ~5 ms vs _rx_buffered_data() which calls
        chan.read() twice for separate deinterleaved channel arrays.
        """
        if not self._sdr._rxbuf:
            self._sdr._rx_init_channels()
        self._sdr._rxbuf.refill()
        raw = np.frombuffer(self._sdr._rxbuf.read(), dtype=np.int16)
        arr = np.empty((len(raw) // 2, 2), dtype=np.float32)
        arr[:, 0] = raw[0::2]   # I
        arr[:, 1] = raw[1::2]   # Q
        out = arr.view(np.complex64).reshape(-1)
        out *= _SCALE
        return out


class TxStream:
    """Continuous non-cyclic transmit stream for PlutoSDR.

    Maintains a background thread that continuously pushes fixed-size buffers
    to the SDR, keeping the TX stream alive at all times.  Callers enqueue
    packet samples via :meth:`send`; the packer thread greedily packs as many
    queued packets as possible into each buffer (FIFO order).  When the queue
    is empty the buffer is zero-filled (silence), so TX never goes idle.

    If a packet does not fit entirely in the current buffer the remainder is
    carried over to the next buffer — no data is lost or reordered.

    Typical usage::

        stream = TxStream(sdr, sample_rate=pipe_cfg.SAMPLE_RATE, buf_size=32768)
        stream.start()
        for pkt in packets:
            samples = tx_pipe.transmit(pkt)
            samples /= np.max(np.abs(samples))
            stream.send(samples)
        stream.stop()
    """

    def __init__(self, sdr: adi.Pluto, sample_rate: int, buf_size: int, *,
                 maxsize: int = 64):
        """
        Args:
            sdr:         configured adi.Pluto instance (tx_cyclic_buffer=False)
            sample_rate: SDR sample rate in Hz — used to compute air-time sleep
            buf_size:    fixed TX buffer length in samples (every sdr.tx() push
                         is exactly this many samples)
            maxsize:     maximum number of packet sample arrays to queue before
                         send() blocks
        """
        self._sdr = sdr
        self._sample_rate = sample_rate
        self._buf_size = buf_size
        self._air_time = buf_size / sample_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._packer, daemon=True)
        self._bufs_sent = 0
        self._bufs_with_payload = 0

    def start(self) -> None:
        """Launch the background packer/sender thread."""
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        """Signal the packer thread to finish and wait for it to join."""
        self._stop.set()
        self._thread.join(timeout=self._air_time + 1.0)

    def send(self, samples: np.ndarray) -> None:
        """Enqueue one packet's samples for transmission.

        The samples should already be DAC-scaled complex64.  Blocks if the
        internal queue is full (back-pressure to the producer).
        """
        while not self._stop.is_set():
            try:
                self._q.put(samples, timeout=0.1)
                return
            except queue.Full:
                pass

    @property
    def pending(self) -> int:
        """Approximate number of packets waiting in the queue."""
        return self._q.qsize()

    @property
    def bufs_sent(self) -> int:
        """Total TX buffers pushed to the SDR so far."""
        return self._bufs_sent

    @property
    def bufs_with_payload(self) -> int:
        """TX buffers that carried real packet samples (vs all-silence)."""
        return self._bufs_with_payload

    # ------------------------------------------------------------------

    def _packer(self) -> None:
        pending: np.ndarray | None = None

        while not self._stop.is_set():
            buf = np.zeros(self._buf_size, dtype=np.complex64)
            write_pos = 0

            # Place held-over packet from previous buffer at the start.
            if pending is not None:
                buf[:len(pending)] = pending
                write_pos = len(pending)
                pending = None

            # Greedily pack queued packets
            while write_pos < self._buf_size:
                try:
                    pkt_samples = self._q.get(timeout=0.001)
                except queue.Empty:
                    break

                space = self._buf_size - write_pos
                if len(pkt_samples) <= space:
                    buf[write_pos:write_pos + len(pkt_samples)] = pkt_samples
                    write_pos += len(pkt_samples)
                else:
                    # Doesn't fit — hold the whole packet for the next
                    # buffer and pad the rest of this one with silence.
                    pending = pkt_samples
                    break

            self._sdr.tx(buf)
            self._bufs_sent += 1
            if write_pos > 0:
                self._bufs_with_payload += 1
                logger.info(
                    "TX buf #%d: %d/%d samples carry payload",
                    self._bufs_sent, write_pos, self._buf_size,
                )
