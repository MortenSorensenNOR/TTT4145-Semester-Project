"""Selective-Repeat ARQ module — full-duplex, bidirectional, three-thread design.

Protocol:

  * Sender keeps a window of up to N unacked frames in flight.
  * Receiver buffers out-of-order frames and delivers contiguous runs to TUN.
  * Every ACK carries:
      - ``seq_num``  = cumulative ACK (last in-order seq delivered)
      - payload      = 1-byte SACK bitmap where bit *i* means seq
                       ``(cumulative + 2 + i)`` is sitting in the receiver's
                       out-of-order buffer (bit 0 covers ``cumulative + 2``
                       because ``cumulative + 1`` is what the receiver is
                       waiting for next — if that were received we'd advance
                       the cumulative ACK instead of SACKing it).
  * On retransmit-timer expiry the sender re-sends only the seqs in
    ``[send_base, next_seq)`` that are neither cumulatively nor SACK-acked
    — avoiding Go-Back-N's "one loss wastes the whole window" amplification.

Two bytes of bitmap cover any ``window_size`` ≤ 16 (== SEQ_SPACE/2).
An ACK with ``length == 0`` (cumulative-only) is treated as bitmap = 0 so
the sender falls back to Go-Back-N behaviour.

Frame types (2-bit ``frame_type`` field, see ``modules.pipeline.PacketType``):
  PacketType.DATA  — carries payload, seq_num = sender's TX seq
  PacketType.ACK   — carries SACK bitmap (2 bytes, big-endian), seq_num = cumulative ACK

Sequence space: 5 bits → 0..31; window size must be < SEQ_SPACE/2 (= 16) so
that circular-distance comparisons stay unambiguous.

Addressing
----------
``ARQConfig.src`` and ``.dst`` populate each outgoing frame's src_mac/dst_mac.
The RX thread drops any decoded frame whose ``dst_mac`` does not match our
own ``src`` — this is what lets the same physical radio run both ends of a
bridge (self-reception / cross-talk is ignored).

Byte / bit boundary
-------------------
The TUN device speaks raw bytes (IP packets).  The DSP pipeline speaks numpy
bit-arrays (Packet.payload).  The ARQ layer converts at both ends:

  TUN read()  → bytes  → bits (np.unpackbits)  → Packet.payload sent over radio
  Packet.payload received  → bytes (np.packbits)  → TUN write()

pluto_rx returns a *list* of Packet because one radio buffer may contain
multiple decoded frames (see RXPipeline.receive).
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.ldpc.channel_coding import CodeRates
from modules.pipeline import Packet, PacketType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame-type aliases (kept as ints so existing tests and the DSP pipeline —
# which packs the 2-bit frame_type field as an int — stay unchanged).
# ---------------------------------------------------------------------------

FRAME_TYPE_DATA: int = int(PacketType.DATA)
FRAME_TYPE_ACK:  int = int(PacketType.ACK)

SEQ_SPACE: int = 32  # 5-bit sequence numbers: 0 .. 31


# ---------------------------------------------------------------------------
# Modular sequence arithmetic
# ---------------------------------------------------------------------------

def seq_add(a: int, n: int) -> int:
    """(a + n) mod SEQ_SPACE — handles negative n correctly."""
    return (a + n) % SEQ_SPACE


def seq_lt(a: int, b: int) -> bool:
    """True iff a strictly precedes b in the circular mod-32 sequence space."""
    return 0 < (b - a) % SEQ_SPACE < SEQ_SPACE // 2


def seq_leq(a: int, b: int) -> bool:
    """True iff a == b or a strictly precedes b."""
    return a == b or seq_lt(a, b)


def seq_diff(a: int, b: int) -> int:
    """Forward distance from a to b in mod-32 space (0 when a == b)."""
    return (b - a) % SEQ_SPACE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ARQConfig:
    window_size: int = 15              # max unacked in-flight frames (< SEQ_SPACE/2)
    retransmit_timeout: float = 0.1    # seconds before Go-Back-N retransmit
    send_queue_maxsize: int = 64       # TUN→TX queue depth; excess is dropped
    src: int = 0                       # this node's logical address (1 bit)
    dst: int = 1                       # peer's logical address    (1 bit)


@dataclass
class ARQStats:
    """Counters for benchmark / debugging. All plain ints — no locking needed
    for single-writer-per-counter access in the TX/RX threads."""
    tun_in:          int = 0  # payloads read from TUN
    tun_out:         int = 0  # payloads written to TUN
    tun_dropped:     int = 0  # TUN reads dropped (send queue full)
    data_tx:         int = 0  # DATA frames handed to the radio (incl. retransmits)
    data_retransmit: int = 0  # DATA frames that were retransmits (subset of data_tx)
    ack_tx:          int = 0  # ACK frames sent by receiver
    data_rx_ok:      int = 0  # valid DATA frames decoded by radio (in-order + buffered)
    data_rx_buffered: int = 0 # out-of-order DATA frames stashed awaiting gap-fill
    data_rx_dup:     int = 0  # DATA already seen (stale duplicate)
    data_rx_foreign: int = 0  # frames whose dst_mac != our src (ignored)
    ack_rx:          int = 0  # ACK frames received
    sack_rx:         int = 0  # ACK frames whose SACK bitmap confirmed ≥1 seq
    timeouts:        int = 0  # retransmit-timer expirations


# ---------------------------------------------------------------------------
# ARQNode
# ---------------------------------------------------------------------------

class ARQNode:
    """Full-duplex Go-Back-N ARQ node.

    Spawns three daemon threads:
      * arq-tun : reads payloads from ``tun_device``, pushes into send queue
      * arq-tx  : Go-Back-N sender; retransmits the unacked window on timeout
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
        self.stats = ARQStats()

        # TUN reader → TX pipeline (raw IP bytes from TUN device)
        self._send_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=self.config.send_queue_maxsize
        )

        # RX → TX shared ACK state (minimal critical section).
        # _last_ack_cumul is the cumulative seq from the most recent ACK;
        # _last_ack_bitmap is that ACK's SACK bitmap (bit i → seq
        # (cumul + 2 + i) is buffered at the peer). Both are overwritten
        # on every ACK — the latest snapshot is always the most useful.
        self._last_ack_cumul:  int = -1
        self._last_ack_bitmap: int = 0
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
            self.stats.tun_in += 1
            try:
                self._send_queue.put(payload, timeout=0.05)
            except queue.Full:
                self.stats.tun_dropped += 1  # radio is slower than TUN — backpressure

    # ------------------------------------------------------------------
    # Thread: TX (Go-Back-N sender)
    # ------------------------------------------------------------------

    def _run_tx(self) -> None:
        send_base: int = 0
        next_seq:  int = 0
        window:    dict[int, bytes] = {}  # seq → raw IP bytes
        to_send:   list[int] = []         # seqs pending (re)transmission
        in_flight: set[int]  = set()      # seqs already on the wire once
        sacked:    set[int]  = set()      # seqs confirmed by SACK bits, not yet cumulatively acked

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
                if seq in in_flight:
                    self.stats.data_retransmit += 1
                else:
                    in_flight.add(seq)
                self._send_data_frame(seq, window[seq])
            to_send.clear()

            # ── 4. Wait for ACK or retransmit timer ───────────────────────
            got_ack = self._ack_event.wait(timeout=self.config.retransmit_timeout)
            self._ack_event.clear()

            if self._stop_event.is_set():
                break

            if got_ack:
                with self._ack_lock:
                    cumul  = self._last_ack_cumul
                    bitmap = self._last_ack_bitmap

                # 4a. Slide send_base over the cumulative ack.
                while window and seq_leq(send_base, cumul):
                    del window[send_base]
                    in_flight.discard(send_base)
                    sacked.discard(send_base)
                    send_base = seq_add(send_base, 1)

                # 4b. Mark individually-acked (SACK'd) seqs. We still hold
                #     them in `window` (the receiver might need them again
                #     if the subsequent in-order frame is lost and the
                #     receiver flushes its buffer for some reason), but
                #     skip them on the next retransmit.
                if bitmap:
                    confirmed_new = False
                    for i in range(self.config.window_size - 1):
                        if bitmap & (1 << i):
                            seq = seq_add(cumul, i + 2)
                            if seq in window and seq not in sacked:
                                sacked.add(seq)
                                confirmed_new = True
                    if confirmed_new:
                        self.stats.sack_rx += 1
            else:
                # Timeout → retransmit only seqs the peer has not yet
                # confirmed. SACK'd ones stay in the window but off the wire.
                self.stats.timeouts += 1
                to_send = [s for s in _window_seqs(send_base, next_seq)
                           if s not in sacked]

    def _send_data_frame(self, seq: int, payload: bytes) -> None:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        packet = Packet(
            src_mac=self.config.src,
            dst_mac=self.config.dst,
            type=FRAME_TYPE_DATA,
            seq_num=seq,
            length=len(payload),
            payload=bits,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.NONE,
            valid=True,
        )
        self.stats.data_tx += 1
        self.pluto_tx(packet)

    def _send_ack_frame(self, cumul: int, bitmap: int = 0) -> None:
        """Send an ACK carrying the cumulative ack + a 2-byte SACK bitmap (big-endian)."""
        bitmap &= 0xFFFF
        bytes_be = np.array([(bitmap >> 8) & 0xFF, bitmap & 0xFF], dtype=np.uint8)
        bits = np.unpackbits(bytes_be)
        packet = Packet(
            src_mac=self.config.src,
            dst_mac=self.config.dst,
            type=FRAME_TYPE_ACK,
            seq_num=cumul,
            length=2,
            payload=bits,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.NONE,
            valid=True,
        )
        self.stats.ack_tx += 1
        self.pluto_tx(packet)

    @staticmethod
    def _build_sack_bitmap(cumul: int, buffered: dict[int, bytes], window_size: int) -> int:
        """Bitmap where bit i = seq ``(cumul + 2 + i)`` is in ``buffered``.

        Bit 0 covers ``cumul + 2`` because ``cumul + 1`` is the next in-order
        seq — if that were in the buffer we'd deliver it and advance the
        cumulative ACK instead.
        """
        bits = 0
        for i in range(window_size - 1):
            seq = seq_add(cumul, i + 2)
            if seq in buffered:
                bits |= (1 << i)
        return bits

    # ------------------------------------------------------------------
    # Thread: RX
    # ------------------------------------------------------------------

    def _run_rx(self) -> None:
        expected_seq: int = 0
        last_acked: int = -1                   # -1 = no in-order frame received yet
        buffered: dict[int, bytes] = {}        # out-of-order frames awaiting gap-fill
        window_size: int = self.config.window_size

        while not self._stop_event.is_set():
            packets = self.pluto_rx()  # one radio buffer may yield multiple frames

            for packet in packets:
                if not packet.valid:
                    continue

                # Address filter: drop frames not meant for us. Without this
                # a radio's own TX leakage into its RX path (or a third node
                # on the same air interface) would poison the sequence state.
                # dst_mac == -1 is the unset sentinel from Packet's default and
                # bypasses the filter (used by tests / loopback fixtures).
                if packet.dst_mac >= 0 and packet.dst_mac != self.config.src:
                    self.stats.data_rx_foreign += 1
                    continue

                if packet.type == FRAME_TYPE_DATA:
                    dist = seq_diff(expected_seq, packet.seq_num)

                    if dist == 0:
                        # In-order: deliver, then drain any contiguous run of
                        # buffered frames whose gap has just been filled.
                        self.stats.data_rx_ok += 1
                        data_bytes = np.packbits(
                            packet.payload[: packet.length * 8].astype(np.uint8)
                        ).tobytes()
                        self.tun.write(data_bytes)
                        self.stats.tun_out += 1
                        last_acked = expected_seq
                        expected_seq = seq_add(expected_seq, 1)

                        while expected_seq in buffered:
                            self.tun.write(buffered.pop(expected_seq))
                            self.stats.tun_out += 1
                            last_acked = expected_seq
                            expected_seq = seq_add(expected_seq, 1)

                        bitmap = self._build_sack_bitmap(last_acked, buffered, window_size)
                        self._send_ack_frame(last_acked, bitmap)

                    elif dist < window_size:
                        # Future seq within the acceptance window — buffer
                        # it (unless a dup of something we already hold) and
                        # report it via SACK so the sender can skip it.
                        if packet.seq_num not in buffered:
                            self.stats.data_rx_ok += 1
                            self.stats.data_rx_buffered += 1
                            buffered[packet.seq_num] = np.packbits(
                                packet.payload[: packet.length * 8].astype(np.uint8)
                            ).tobytes()
                        else:
                            self.stats.data_rx_dup += 1

                        if last_acked >= 0:
                            bitmap = self._build_sack_bitmap(last_acked, buffered, window_size)
                            self._send_ack_frame(last_acked, bitmap)

                    else:
                        # Stale duplicate of already-delivered seq (outside
                        # the acceptance window in the forward direction).
                        # Re-ACK with current bitmap so the sender slides.
                        self.stats.data_rx_dup += 1
                        if last_acked >= 0:
                            bitmap = self._build_sack_bitmap(last_acked, buffered, window_size)
                            self._send_ack_frame(last_acked, bitmap)

                elif packet.type == FRAME_TYPE_ACK:
                    self.stats.ack_rx += 1
                    # length=0 ACKs carry no bitmap — treat as 0.
                    if packet.length >= 2 and packet.payload.size >= 16:
                        b = np.packbits(packet.payload[:16].astype(np.uint8))
                        ack_bitmap = (int(b[0]) << 8) | int(b[1])
                    else:
                        ack_bitmap = 0
                    with self._ack_lock:
                        self._last_ack_cumul  = packet.seq_num
                        self._last_ack_bitmap = ack_bitmap
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
