"""Go-Back-N ARQ module — full-duplex, bidirectional, three-thread design.

Frame types (2-bit field in FrameHeader):
  0b00  DATA  — carries payload, sequence_number = sender's TX seq
  0b01  ACK   — no payload, sequence_number = cumulative ACK (last in-order seq received)

Sequence space: 4 bits → 0..15, window size must be < SEQ_SPACE (16).

Byte / bit boundary
-------------------
The TUN device speaks raw bytes (IP packets).  The DSP pipeline speaks numpy
bit-arrays (Packet.payload).  The ARQ layer converts at both ends:

  TUN read()  → bytes  → bits (np.unpackbits)  → Packet.payload sent over radio
  Packet.payload received  → bytes (np.packbits)  → TUN write()

pluto_rx returns a *list* of Packet because one radio buffer may contain
multiple decoded frames (see RXPipeline.receive).
"""

import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from modules.pipeline import Packet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_TYPE_DATA: int = 0
FRAME_TYPE_ACK: int = 1

SEQ_SPACE: int = 16  # 4-bit sequence numbers: 0 .. 15


# ---------------------------------------------------------------------------
# Modular sequence arithmetic
# ---------------------------------------------------------------------------

def seq_add(a: int, n: int) -> int:
    """(a + n) mod SEQ_SPACE — handles negative n correctly."""
    return (a + n) % SEQ_SPACE


def seq_lt(a: int, b: int) -> bool:
    """True iff a strictly precedes b in the circular mod-16 sequence space."""
    return 0 < (b - a) % SEQ_SPACE < SEQ_SPACE // 2


def seq_leq(a: int, b: int) -> bool:
    """True iff a == b or a strictly precedes b."""
    return a == b or seq_lt(a, b)


def seq_diff(a: int, b: int) -> int:
    """Forward distance from a to b in mod-16 space (0 when a == b)."""
    return (b - a) % SEQ_SPACE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ARQConfig:
    window_size: int = 7               # max unacked in-flight frames (< SEQ_SPACE)
    retransmit_timeout: float = 0.1    # seconds before Go-Back-N retransmit
    send_queue_maxsize: int = 64       # TUN→TX queue depth; excess is dropped
    src: int = 0                       # this node's logical address
    dst: int = 1                       # peer's logical address


# ---------------------------------------------------------------------------
# ARQNode
# ---------------------------------------------------------------------------

class ARQNode:
    """Full-duplex Go-Back-N ARQ node.

    Spawns three daemon threads:
      * arq-tun : reads payloads from ``tun_device``, pushes into send queue
      * arq-tx  : Go-Back-N sender; retransmits entire window on timeout
      * arq-rx  : receives frames; delivers DATA to TUN, signals ACKs to TX

    Parameters
    ----------
    tun_device:
        Object with:
          - ``read() -> bytes | None``  (blocking or short-timeout poll)
          - ``write(data: bytes) -> None``
        See :class:`modules.tun.TunDevice` for the Linux implementation.
    pluto_tx:
        ``Callable[[Packet], None]`` — hand a frame to the DSP/radio TX chain.
    pluto_rx:
        ``Callable[[], list[Packet]]`` — blocking receive from DSP/radio RX chain.
        Returns a list because one radio buffer may contain multiple frames.
    config:
        Optional :class:`ARQConfig`; defaults are used when ``None``.
    """

    def __init__(
        self,
        tun_device,
        pluto_tx: Callable[[Packet], None],
        pluto_rx: Callable[[], "list[Packet]"],
        config: ARQConfig | None = None,
    ) -> None:
        self.tun = tun_device
        self.pluto_tx = pluto_tx
        self.pluto_rx = pluto_rx
        self.config = config or ARQConfig()

        # TUN reader → TX pipeline (raw IP bytes from TUN device)
        self._send_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=self.config.send_queue_maxsize
        )

        # RX → TX shared ACK state (minimal critical section)
        self._last_acked_seq: int = -1
        self._ack_lock = threading.Lock()
        self._ack_event = threading.Event()

        self._stop_event = threading.Event()
        self._threads = [
            threading.Thread(target=self._run_tun_reader, daemon=True, name="arq-tun"),
            threading.Thread(target=self._run_tx,         daemon=True, name="arq-tx"),
            threading.Thread(target=self._run_rx,         daemon=True, name="arq-rx"),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch all three ARQ threads."""
        self._stop_event.clear()
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        """Signal threads to stop and block until they exit."""
        self._stop_event.set()
        self._ack_event.set()  # unblock TX if it is waiting for an ACK
        for t in self._threads:
            t.join()

    # ------------------------------------------------------------------
    # Thread: TUN reader
    # ------------------------------------------------------------------

    def _run_tun_reader(self) -> None:
        while not self._stop_event.is_set():
            payload = self.tun.read()
            if payload is None:
                continue
            try:
                self._send_queue.put(payload, timeout=0.05)
            except queue.Full:
                pass  # drop; radio is slower than TUN — backpressure

    # ------------------------------------------------------------------
    # Thread: TX (Go-Back-N sender)
    # ------------------------------------------------------------------

    def _run_tx(self) -> None:
        send_base: int = 0
        next_seq: int = 0
        window: dict[int, bytes] = {}  # seq → raw IP bytes
        to_send: list[int] = []        # seqs pending (re)transmission

        while not self._stop_event.is_set():
            # ── 1. Admit new payloads into window ─────────────────────────
            while seq_diff(send_base, next_seq) < self.config.window_size:
                try:
                    payload = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                window[next_seq] = payload
                to_send.append(next_seq)
                next_seq = seq_add(next_seq, 1)

            # ── 2. If window is empty, block briefly waiting for data ──────
            if not window:
                try:
                    payload = self._send_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                window[next_seq] = payload
                to_send.append(next_seq)
                next_seq = seq_add(next_seq, 1)

            # ── 3. Transmit pending seqs ───────────────────────────────────
            for seq in to_send:
                self._send_data_frame(seq, window[seq])
            to_send.clear()

            # ── 4. Wait for ACK or retransmit timer ───────────────────────
            got_ack = self._ack_event.wait(timeout=self.config.retransmit_timeout)
            self._ack_event.clear()

            if self._stop_event.is_set():
                break

            if got_ack:
                with self._ack_lock:
                    acked = self._last_acked_seq
                # Slide window: remove everything up to and including acked
                while window and seq_leq(send_base, acked):
                    del window[send_base]
                    send_base = seq_add(send_base, 1)
            else:
                # Timeout → retransmit entire window (Go-Back-N)
                to_send = _window_seqs(send_base, next_seq)

    def _send_data_frame(self, seq: int, payload: bytes) -> None:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        packet = Packet(
            src_mac=self.config.src,
            dst_mac=self.config.dst,
            type=FRAME_TYPE_DATA,
            seq_num=seq,
            length=len(payload),
            payload=bits,
            valid=True,
        )
        self.pluto_tx(packet)

    def _send_ack_frame(self, seq: int) -> None:
        packet = Packet(
            src_mac=self.config.src,
            dst_mac=self.config.dst,
            type=FRAME_TYPE_ACK,
            seq_num=seq,
            length=0,
            payload=np.array([], dtype=int),
            valid=True,
        )
        self.pluto_tx(packet)

    # ------------------------------------------------------------------
    # Thread: RX
    # ------------------------------------------------------------------

    def _run_rx(self) -> None:
        expected_seq: int = 0
        last_acked: int = -1  # -1 = no in-order frame received yet

        while not self._stop_event.is_set():
            packets = self.pluto_rx()  # one radio buffer may yield multiple frames

            for packet in packets:
                if not packet.valid:
                    continue

                if packet.type == FRAME_TYPE_DATA:
                    if packet.seq_num == expected_seq:
                        data_bytes = np.packbits(
                            packet.payload[: packet.length * 8].astype(np.uint8)
                        ).tobytes()
                        self.tun.write(data_bytes)
                        last_acked = expected_seq
                        expected_seq = seq_add(expected_seq, 1)
                        self._send_ack_frame(last_acked)
                    elif last_acked >= 0:
                        # Out-of-order: repeat last good ACK to nudge sender
                        self._send_ack_frame(last_acked)

                elif packet.type == FRAME_TYPE_ACK:
                    with self._ack_lock:
                        self._last_acked_seq = packet.seq_num
                    self._ack_event.set()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window_seqs(send_base: int, next_seq: int) -> list[int]:
    """Ordered list of seqs in the half-open window [send_base, next_seq)."""
    seqs: list[int] = []
    seq = send_base
    while seq != next_seq:
        seqs.append(seq)
        seq = seq_add(seq, 1)
    return seqs
