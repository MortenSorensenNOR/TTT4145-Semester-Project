"""Process-pool wrappers around TXPipeline / RXPipeline.

Used by pluto/tun_link.py when the user passes --workers N. Each worker
process owns its own TXPipeline / RXPipeline instance; jobs are scattered
across workers so the OS scheduler can park them on P-cores rather than
having one Python thread bouncing between P/E cores on hybrid Intel
hardware.

TX path
-------
TXWorkerPool wraps a small process pool; tun_link's TX thread calls
``submit(pkt)`` per outgoing packet and gets back a Future yielding the
DAC-scaled complex64 samples. Packets are pickled to the worker (small —
just bits + a few ints), samples come back via pickle (large — but the
work that produced them is the LDPC encode + RRC convolution that we
actually wanted to parallelize).

RX path
-------
RXWorkerPool uses a shared-memory ring for the post-match-filter buffer to
avoid copying ~MB-sized arrays through a multiprocessing pipe per detection.

Flow per RX buffer:
  1. parent runs match_filter + detect (cheap, single-threaded)
  2. parent calls ``submit_buffer(filtered, detections)`` — picks a free
     ring slot, memcpys the filtered buffer into the slot, fires off one
     Future per detection
  3. parent collects futures in detection order
  4. when the last future for a slot resolves, the slot is auto-released
     for re-use

Workers receive (slot_name, start, end, cfo, phase) tuples — tiny pickled
payload, then read directly out of the shared-memory mapping.

Both pools default to the ``spawn`` start method to avoid fork-with-threads
hazards (libiio, the SDR-side iio threads, etc.). Spawn pays a one-time
~1-2 s startup cost per worker for re-importing the modules; this is fine
for a long-running tun_link process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from modules.pipeline import DetectionResult, Packet, PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-side globals — initialized once per worker via pool initializer.
# ---------------------------------------------------------------------------

# TX worker state.
_TX_PIPE = None              # TXPipeline instance (per-worker)

# RX worker state.
_RX_PIPE = None              # RXPipeline instance (per-worker)
_RX_SHM: dict = {}           # shm_name -> (SharedMemory, np.ndarray) — one entry per ring slot
_RX_INCLUDE_SYMBOLS = False  # whether to ship rx_symbols back (large, usually unused)


# ---------------------------------------------------------------------------
# RXResult — compact, picklable representation of a decoded frame.
# ---------------------------------------------------------------------------

@dataclass
class RXResult:
    """Payload returned by an RX worker for one detection.

    status:
      "ok"           — decode succeeded; header_* and payload_bytes valid
      "tail_cutoff"  — frame extends past the buffer; caller should retry
                       on the next iteration (preserves original break
                       semantics in RXPipeline.receive)
      "decode_error" — header CRC failed or payload CRC/LDPC failed; counted
                       as a real loss
    """
    status: str
    src_mac: int = -1
    dst_mac: int = -1
    type: int = -1
    seq_num: int = -1
    length: int = -1
    valid: bool = False
    payload_bytes: bytes = b""           # IP packet bytes, ready for tun.write
    rx_symbols: np.ndarray | None = None  # only populated if include_rx_symbols=True
    err: str = ""


# ---------------------------------------------------------------------------
# TX worker functions.
# ---------------------------------------------------------------------------

def _tx_init(pipe_cfg: "PipelineConfig") -> None:
    """Worker initializer: build a TXPipeline, warm caches with a small probe."""
    global _TX_PIPE
    # Heavy imports happen here under spawn so the parent stays light.
    from modules.pipeline import Packet, TXPipeline
    _TX_PIPE = TXPipeline(pipe_cfg)
    # Warm LDPC encoder cache + RRC paths so the first user-driven transmit
    # on this worker isn't unusually slow.
    probe = Packet(src_mac=0, dst_mac=0, type=0, seq_num=0,
                   length=64, payload=np.zeros(64 * 8, dtype=np.uint8))
    _TX_PIPE.transmit(probe)


def _tx_build(payload_bytes: bytes, src_mac: int, dst_mac: int, ftype: int,
              seq_num: int, dac_scale: float) -> np.ndarray:
    """Worker entrypoint: build one packet, return DAC-scaled complex64 samples."""
    from modules.pipeline import Packet
    bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    pkt = Packet(src_mac=src_mac, dst_mac=dst_mac, type=ftype, seq_num=seq_num,
                 length=len(payload_bytes), payload=bits)
    samples = _TX_PIPE.transmit(pkt)
    peak = float(np.max(np.abs(samples)))
    if peak > 0:
        samples = samples / peak
    return (samples * dac_scale).astype(np.complex64)


# ---------------------------------------------------------------------------
# RX worker functions.
# ---------------------------------------------------------------------------

def _rx_init(pipe_cfg: "PipelineConfig", shm_names: list[str], slot_samples: int,
             include_rx_symbols: bool) -> None:
    """Worker initializer: build RXPipeline + attach to every shared-memory slot.

    The attached shared-memory blocks live for the lifetime of the worker; their
    backing pages are reused across decode jobs (each job just picks the slot
    by name and reads its slice).
    """
    global _RX_PIPE, _RX_SHM, _RX_INCLUDE_SYMBOLS
    from modules.pipeline import RXPipeline
    _RX_PIPE = RXPipeline(pipe_cfg)
    _RX_SHM = {}
    for name in shm_names:
        shm = SharedMemory(name=name, create=False)
        arr = np.ndarray((slot_samples,), dtype=np.complex64, buffer=shm.buf)
        _RX_SHM[name] = (shm, arr)
    _RX_INCLUDE_SYMBOLS = include_rx_symbols


def _rx_decode(slot_name: str, start: int, end: int,
               cfo_estimate: float, phase_estimate: float) -> RXResult:
    """Worker entrypoint: decode one detection from the named slot."""
    arr = _RX_SHM[slot_name][1]
    rx_syms = arr[start:end]
    try:
        pkt = _RX_PIPE.decode(rx_syms,
                              np.float32(cfo_estimate), np.float32(phase_estimate))
    except IndexError as e:
        return RXResult(status="tail_cutoff", err=f"{type(e).__name__}: {e}")
    except Exception as e:
        return RXResult(status="decode_error", err=f"{type(e).__name__}: {e}")

    # Pack payload bits to bytes here (in the worker) so we ship ~1.5 KB
    # instead of ~96 KB (1500*8 int64) through the result pipe.
    if pkt.length > 0 and pkt.payload is not None and pkt.payload.size >= pkt.length * 8:
        payload_bytes = bytes(np.packbits(pkt.payload[:pkt.length * 8].astype(np.uint8)).tobytes())
    else:
        payload_bytes = b""

    return RXResult(
        status="ok",
        src_mac=int(pkt.src_mac),
        dst_mac=int(pkt.dst_mac),
        type=int(pkt.type),
        seq_num=int(pkt.seq_num),
        length=int(pkt.length),
        valid=bool(pkt.valid),
        payload_bytes=payload_bytes,
        rx_symbols=(pkt.rx_symbols if _RX_INCLUDE_SYMBOLS else None),
    )


# ---------------------------------------------------------------------------
# Pool wrappers.
# ---------------------------------------------------------------------------

def _resolve_context(start_method: str | None) -> mp.context.BaseContext:
    """Pick a multiprocessing start method; default to ``spawn`` for safety."""
    if start_method is None:
        # Spawn avoids the fork-with-threads pitfalls (libiio, AGC threads).
        # The cold-start cost is paid once at pool init; tun_link is long-running.
        start_method = os.environ.get("RADIO_MP_START_METHOD", "spawn")
    return mp.get_context(start_method)


class TXWorkerPool:
    """Multi-process TX builder.

    Each worker owns a TXPipeline and turns ``submit(pkt)`` calls into
    DAC-scaled complex64 sample arrays. Order is preserved by the caller
    (collect futures in submission order).
    """

    def __init__(self, pipe_cfg: "PipelineConfig", n_workers: int,
                 start_method: str | None = None):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self._ctx = _resolve_context(start_method)
        self._pool = self._ctx.Pool(
            processes=n_workers,
            initializer=_tx_init,
            initargs=(pipe_cfg,),
        )
        self._n_workers = n_workers
        logger.info(f"TXWorkerPool: {n_workers} workers ({self._ctx._name})")

    @property
    def n_workers(self) -> int:
        return self._n_workers

    def submit(self, payload_bytes: bytes, src_mac: int, dst_mac: int,
               ftype: int, seq_num: int, dac_scale: float):
        """Submit one packet for building. Returns a multiprocessing AsyncResult.

        Use ``.get(timeout)`` on the returned object to retrieve the samples.
        """
        return self._pool.apply_async(
            _tx_build,
            args=(payload_bytes, src_mac, dst_mac, ftype, seq_num, dac_scale),
        )

    def shutdown(self) -> None:
        try:
            self._pool.close()
            self._pool.join()
        except Exception:
            self._pool.terminate()


# ---------------------------------------------------------------------------
# RX pool: shared-memory ring + worker pool.
# ---------------------------------------------------------------------------

class _RXBufferRing:
    """Ring of shared-memory slots used to hand RX buffers to workers.

    Each slot holds at most ``slot_samples`` complex64 samples. The producer
    cycles through slots; before reusing a slot it blocks until that slot's
    in-flight job count returns to zero (i.e. all decodes that referenced
    it have completed and released).
    """

    def __init__(self, n_slots: int, slot_samples: int):
        self.n = n_slots
        self.slot_samples = slot_samples
        nbytes = slot_samples * np.dtype(np.complex64).itemsize
        self.shms: list[SharedMemory] = []
        self.arrays: list[np.ndarray] = []
        self.names: list[str] = []
        for _ in range(n_slots):
            shm = SharedMemory(create=True, size=nbytes)
            arr = np.ndarray((slot_samples,), dtype=np.complex64, buffer=shm.buf)
            self.shms.append(shm)
            self.arrays.append(arr)
            self.names.append(shm.name)
        self._inflight = [0] * n_slots
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._next_idx = 0

    def acquire(self, timeout: float | None = None) -> int:
        """Block (up to ``timeout``) until next slot has zero in-flight jobs.

        Returns the slot index. Raises TimeoutError if the slot is still busy
        after ``timeout`` seconds.
        """
        with self._cond:
            deadline = None
            if timeout is not None:
                import time as _time
                deadline = _time.monotonic() + timeout
            while self._inflight[self._next_idx] > 0:
                if deadline is None:
                    self._cond.wait()
                else:
                    import time as _time
                    remaining = deadline - _time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("RXBufferRing: no free slot")
                    self._cond.wait(timeout=remaining)
            idx = self._next_idx
            self._next_idx = (self._next_idx + 1) % self.n
            return idx

    def write(self, idx: int, data: np.ndarray) -> int:
        """Copy ``data`` into the slot. Returns the number of samples written."""
        n = len(data)
        if n > self.slot_samples:
            raise ValueError(f"buffer length {n} > slot capacity {self.slot_samples}")
        if data.dtype != np.complex64:
            data = data.astype(np.complex64)
        self.arrays[idx][:n] = data
        return n

    def increment(self, idx: int) -> None:
        with self._cond:
            self._inflight[idx] += 1

    def decrement(self, idx: int) -> None:
        with self._cond:
            self._inflight[idx] -= 1
            if self._inflight[idx] == 0:
                self._cond.notify_all()

    def close(self) -> None:
        for shm in self.shms:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass


@dataclass
class _SubmittedBuffer:
    """Bookkeeping for one submit_buffer() call."""
    slot_idx: int
    futures: list  # of multiprocessing AsyncResult, in detection order
    abs_offsets: list[int]   # detection.payload_start translated to absolute buffer offsets
    detections: list = field(default_factory=list)  # parallel to futures, for diagnostics


class RXWorkerPool:
    """Multi-process RX decoder backed by a shared-memory ring.

    Use::

        pool = RXWorkerPool(pipe_cfg, n_workers=4, slot_samples=2*rx_buf, n_slots=4)
        # for each post-match-filter buffer:
        sub = pool.submit_buffer(filtered_buffer, detections, search_from)
        for fut, abs_offset in zip(sub.futures, sub.abs_offsets):
            result: RXResult = fut.get(timeout=5.0)
            ...
        # slot is auto-released once all of sub.futures have completed.
    """

    def __init__(self, pipe_cfg: "PipelineConfig", n_workers: int,
                 slot_samples: int, n_slots: int = 4,
                 include_rx_symbols: bool = False,
                 start_method: str | None = None):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        if n_slots < 1:
            raise ValueError("n_slots must be >= 1")
        self._ring = _RXBufferRing(n_slots=n_slots, slot_samples=slot_samples)
        self._ctx = _resolve_context(start_method)
        self._pool = self._ctx.Pool(
            processes=n_workers,
            initializer=_rx_init,
            initargs=(pipe_cfg, list(self._ring.names), slot_samples, include_rx_symbols),
        )
        self._n_workers = n_workers
        self._slot_samples = slot_samples
        self._n_slots = n_slots
        logger.info(
            f"RXWorkerPool: {n_workers} workers × {n_slots} slots "
            f"× {slot_samples} samples ({self._ctx._name})"
        )

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def slot_samples(self) -> int:
        return self._slot_samples

    def submit_buffer(self, filtered_buffer: np.ndarray,
                      detections: list["DetectionResult"],
                      search_from_abs: int) -> _SubmittedBuffer:
        """Hand a post-match-filter buffer + its detections to the pool.

        ``search_from_abs`` is the absolute offset in the parent's source
        buffer where ``filtered_buffer`` starts; we use it to translate each
        detection.payload_start into an absolute sample index for the caller.

        If ``detections`` is empty the slot is acquired and immediately
        released so the caller still gets a valid empty result.
        """
        slot_idx = self._ring.acquire()
        slot_len = self._ring.write(slot_idx, filtered_buffer)
        slot_name = self._ring.names[slot_idx]

        if not detections:
            # No work to do, return the slot immediately.
            self._ring.increment(slot_idx)
            self._ring.decrement(slot_idx)
            return _SubmittedBuffer(slot_idx=slot_idx, futures=[],
                                    abs_offsets=[], detections=[])

        # Pin the slot for the duration of every job we're about to submit.
        # Each job's done-callback decrements the in-flight count; when it
        # hits zero the slot is reusable.
        futures: list = []
        abs_offsets: list[int] = []
        remaining = [len(detections)]
        cb_lock = threading.Lock()

        def make_release_cb():
            def cb(_result):
                # AsyncResult.get_callback fires whether the call succeeded
                # or raised. Decrement once per submitted job; release the
                # slot after the last one.
                with cb_lock:
                    remaining[0] -= 1
                    last = (remaining[0] == 0)
                if last:
                    self._ring.decrement(slot_idx)
            return cb

        # Prebump the in-flight count once to cover all submissions; we'll
        # decrement once when the last completion callback fires.
        self._ring.increment(slot_idx)

        for det in detections:
            cb = make_release_cb()
            ar = self._pool.apply_async(
                _rx_decode,
                args=(slot_name, int(det.payload_start), int(slot_len),
                      float(det.cfo_estimate), float(det.phase_estimate)),
                callback=cb,
                error_callback=cb,
            )
            futures.append(ar)
            abs_offsets.append(search_from_abs + int(det.payload_start))

        return _SubmittedBuffer(slot_idx=slot_idx, futures=futures,
                                abs_offsets=abs_offsets,
                                detections=list(detections))

    def shutdown(self) -> None:
        try:
            self._pool.close()
            self._pool.join()
        except Exception:
            self._pool.terminate()
        self._ring.close()
