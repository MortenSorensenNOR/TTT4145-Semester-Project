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

TxStream — chunked non-cyclic transmit
    Accepts one packet's samples at a time and sends them in fixed-size
    chunks.  Because the USB push completes in less than one chunk's air
    time, the hardware receives the next chunk before finishing the current
    one, giving gapless transmission without a single giant buffer.

    Typical usage::

        stream = TxStream(sdr, sample_rate=pipe_cfg.SAMPLE_RATE, chunk_packets=8)
        for pkt in packets:
            samples = tx_pipe.transmit(pkt)
            samples /= np.max(np.abs(samples))   # normalise to [-1, 1]
            stream.send(samples)
        stream.close()   # flush remainder + wait for last chunk to finish
"""

import queue
import threading
import time

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
    """Chunked non-cyclic transmit for PlutoSDR.

    Buffers ``chunk_packets`` packets' worth of samples and pushes them to the
    hardware in one DMA call, eliminating per-packet USB overhead while
    modelling a real windowed/ARQ transmitter (send window → wait for ACKs →
    next window).

    Each chunk is timed so the next push does not start until the hardware has
    finished transmitting the current one, preventing DMA buffer corruption.

    Typical usage::

        stream = TxStream(sdr, sample_rate=pipe_cfg.SAMPLE_RATE, chunk_packets=8)
        for pkt in packets:
            samples = tx_pipe.transmit(pkt)
            samples /= np.max(np.abs(samples))   # normalise to [-1, 1]
            stream.send(samples)
        stream.close()   # flush remainder + wait for last chunk to finish
    """

    def __init__(self, sdr: adi.Pluto, sample_rate: int, chunk_packets: int = 8):
        """
        Args:
            sdr:           configured adi.Pluto instance (tx_cyclic_buffer=False)
            sample_rate:   SDR sample rate in Hz — used to compute air time
            chunk_packets: number of packets to accumulate before each push
        """
        self._sdr           = sdr
        self._sample_rate   = sample_rate
        self._chunk_packets = chunk_packets
        self._pending: list[np.ndarray] = []
        self._n_pending     = 0
        self._chunk_len: int | None = None  # fixed on first flush; used to pad later batches

    def send(self, samples: np.ndarray) -> None:
        """Buffer one packet's normalised samples (range [-1, 1]).

        Scales to the DAC range internally.  Flushes to hardware automatically
        once ``chunk_packets`` packets have been accumulated.

        Args:
            samples: complex64 (or float-compatible) array, peak amplitude ≤ 1.
        """
        self._pending.append((samples * DAC_SCALE).astype(np.complex64))
        self._n_pending += 1
        if self._n_pending >= self._chunk_packets:
            self._flush()

    def close(self) -> None:
        """Flush any remaining buffered samples and wait for transmission to end."""
        self._flush()

    # ------------------------------------------------------------------

    def _flush(self) -> None:
        if not self._pending:
            return
        chunk = np.concatenate(self._pending)

        # pyadi-iio creates the TX DMA buffer on the first sdr.tx() call and
        # rejects any subsequent push with a different length.  Lock in the
        # full-batch length on the first flush, then zero-pad partial batches
        # (e.g. the final window when packets % chunk_packets != 0).
        if self._chunk_len is None:
            self._chunk_len = len(chunk)
        elif len(chunk) < self._chunk_len:
            chunk = np.concatenate([chunk, np.zeros(self._chunk_len - len(chunk), dtype=np.complex64)])

        # On PlutoSDR over USB, sdr.tx() returns after the DMA push completes
        # and the hardware begins transmitting from the start of that point.
        # Sleep for the full air time so the next push does not start before
        # the hardware has finished reading the current buffer.
        air_time = self._chunk_len / self._sample_rate
        self._sdr.tx(chunk)
        time.sleep(air_time)

        self._pending.clear()
        self._n_pending = 0
