"""Threaded SDR RX stream for PlutoSDR.

Decouples hardware buffer collection from processing so the DMA ring never
stalls while the pipeline is running.  The producer thread calls buf.refill()
+ buf.read() continuously, bypassing iiod's chan.read() deinterleave overhead
and keeping a buffer ready for the consumer at all times.

Typical usage::

    sdr = adi.Pluto("ip:192.168.2.1")
    configure_rx(sdr)

    stream = RxStream(sdr)
    stream.start()
    stream.flush()          # let AGC settle

    while True:
        samples = stream.get()   # complex64, normalised — replaces sdr.rx()
        packets = rx_pipe.receive(samples)
        ...

    stream.stop()
"""

import queue
import threading

import numpy as np
import adi

from pluto.config import DAC_SCALE

_SCALE = np.float32(2.0 / DAC_SCALE)


class RxStream:
    """Continuously drains PlutoSDR hardware buffers in a background thread.

    get() returns the next buffer as normalised complex64, with typically
    1-3 ms of wait vs ~18 ms for a bare sdr.rx() call.
    """

    def __init__(self, sdr: adi.Pluto, maxsize: int = 2):
        """
        Args:
            sdr:     configured adi.Pluto instance (rx_buffer_size already set)
            maxsize: number of prefetched buffers to keep in flight
        """
        self._sdr = sdr
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._overruns = 0

    def start(self) -> None:
        """Start the background producer thread."""
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
        """Discard n buffers so the AGC can settle before processing begins."""
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
