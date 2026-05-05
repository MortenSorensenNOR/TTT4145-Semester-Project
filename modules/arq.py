"""Selective-Repeat ARQ over the radio link (see report §II).

Each ACK reuses the standard frame header with type=ACK, BPSK, uncoded
2 B payload, and carries:

  seq_num  = cumulative ACK c — the highest seq delivered in order, so
             c + 1 is the next expected (missing) seq.
  payload  = 16-bit big-endian SACK bitmap covering the window past the
             gap: bit i marks seq c + 2 + i as buffered out-of-order at
             the receiver. The bitmap skips c + 1 because, by definition
             of the cumulative ACK, that seq cannot already be buffered.
             A zero-length ACK is treated as bitmap = 0, falling back to
             Go-Back-N.

On the retransmit timer the sender re-sends only the seqs in
[send_base, next_seq) that are neither cumulatively nor SACK-acked.

Frame types (PacketType): DATA carries a TUN payload; ACK carries the
SACK bitmap; NAK bypasses ARQ entirely (forwarded straight to TUN);
CTRL is reserved.

ARQConfig.src / .dst populate src_mac/dst_mac on every outgoing frame;
the RX thread drops frames whose dst_mac doesn't match our own src, so
the same radio can run both ends of a bridge without self-reception loops.

Sequence space is 5 bits (0..31); window_size must be < SEQ_SPACE/2 so
circular-distance comparisons stay unambiguous.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.ldpc.channel_coding import CodeRates
from modules.pipeline import Packet, PacketType

logger = logging.getLogger(__name__)


FRAME_TYPE_DATA: int = int(PacketType.DATA)
FRAME_TYPE_ACK:  int = int(PacketType.ACK)

SEQ_SPACE: int = 32  # 5-bit sequence numbers: 0 .. 31


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


@dataclass
class ARQConfig:
    window_size: int = 15              # max unacked in-flight frames (< SEQ_SPACE/2)
    retransmit_timeout: float = 0.1    # seconds before Go-Back-N retransmit
    send_queue_maxsize: int = 64       # TUN -> TX queue depth; excess is dropped
    src: int = 0                       # this node's logical address (1 bit)
    dst: int = 1                       # peer's logical address    (1 bit)


@dataclass
class ARQStats:
    """Counters for benchmark / debugging. All plain ints — no locking needed
    for single-writer-per-counter access in the TX/RX threads."""
    tun_in:           int = 0  # payloads read from TUN
    tun_out:          int = 0  # payloads written to TUN
    tun_dropped:      int = 0  # TUN reads dropped (send queue full)
    data_tx:          int = 0  # DATA frames handed to the radio (incl. retransmits)
    data_retransmit:  int = 0  # DATA frames that were retransmits (subset of data_tx)
    ack_tx:           int = 0  # ACK frames sent by receiver
    data_rx_ok:       int = 0  # valid DATA frames decoded by radio (in-order + buffered)
    data_rx_buffered: int = 0 # out-of-order DATA frames stashed awaiting gap-fill
    data_rx_dup:      int = 0  # DATA already seen (stale duplicate)
    data_rx_foreign:  int = 0  # frames whose dst_mac != our src (ignored)
    ack_rx:           int = 0  # ACK frames received
    sack_rx:          int = 0  # ACK frames whose SACK bitmap confirmed ≥1 seq
    timeouts:         int = 0  # retransmit-timer expirations


class ARQNode:
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

        # TUN reader -> TX pipeline, ip bytes from tun
        self._send_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=self.config.send_queue_maxsize
        )

        # RX -> TX shared ACK state
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

    def start(self) -> None:
        self._stop_event.clear()
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._ack_event.set()
        for t in self._threads:
            t.join()

    def _run_tun_reader(self) -> None:
        while not self._stop_event.is_set():
            payload = self.tun.read()
            if payload is None:
                continue
            self.stats.tun_in += 1
            try:
                self._send_queue.put(payload, timeout=0.05)
            except queue.Full:
                self.stats.tun_dropped += 1

    def _run_tx(self) -> None:
        send_base: int = 0
        next_seq:  int = 0
        window:    dict[int, bytes] = {}
        to_send:   list[int] = []    # pending, not sent
        in_flight: set[int]  = set() # pending, sent
        sacked:    set[int]  = set()

        while not self._stop_event.is_set():
            while seq_diff(send_base, next_seq) < self.config.window_size:
                try:
                    payload = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                window[next_seq] = payload
                to_send.append(next_seq)
                next_seq = seq_add(next_seq, 1)

            if not window:
                try:
                    payload = self._send_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                window[next_seq] = payload
                to_send.append(next_seq)
                next_seq = seq_add(next_seq, 1)

            for seq in to_send:
                if seq in in_flight:
                    self.stats.data_retransmit += 1
                else:
                    in_flight.add(seq)
                self._send_data_frame(seq, window[seq])
            to_send.clear()

            got_ack = self._ack_event.wait(timeout=self.config.retransmit_timeout)
            self._ack_event.clear()

            if self._stop_event.is_set():
                break

            if got_ack:
                with self._ack_lock:
                    cumul  = self._last_ack_cumul
                    bitmap = self._last_ack_bitmap

                while window and seq_leq(send_base, cumul):
                    del window[send_base]
                    in_flight.discard(send_base)
                    sacked.discard(send_base)
                    send_base = seq_add(send_base, 1)

                # Mark SACK'd seqs: kept in window (receiver may flush its
                # buffer and need them again) but skipped on retransmit.
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
                # Timeout -> retransmit only seqs the peer has not yet
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

    # Thread: RX
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


def _window_seqs(send_base: int, next_seq: int) -> list[int]:
    """Ordered list of seqs in the half-open window [send_base, next_seq)."""
    seqs: list[int] = []
    seq = send_base
    while seq != next_seq:
        seqs.append(seq)
        seq = seq_add(seq, 1)
    return seqs
