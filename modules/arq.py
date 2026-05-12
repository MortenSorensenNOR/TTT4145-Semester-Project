"""Selective-Repeat ARQ over the radio link (see report §II).

DATA frames carry the TUN payload with a sender seq_num. Each ACK reuses the
standard frame header with type=ACK, BPSK, uncoded 8 B payload:

  seq_num = cumulative ACK c (highest seq delivered in order)
  payload = 64-bit big-endian SACK bitmap; bit i marks seq c+2+i as buffered
            out-of-order at the receiver. Bit 0 covers c+2 because c+1 is
            the next in-order seq — if it were buffered we'd advance the
            cumulative ACK instead. A zero-length ACK is treated as
            bitmap = 0 (Go-Back-N fallback). 64 bits is enough to cover
            window_size - 1 slots for any legal window (< SEQ_SPACE/2 = 64).

On retransmit timeout the sender re-sends only seqs in [send_base, next_seq)
that are neither cumulatively nor SACK-acked.

Sequence space is 7 bits (0..127), matching FrameHeaderConfig.sequence_number_bits.
window_size must be < SEQ_SPACE/2 so circular-distance comparisons stay
unambiguous.

ARQConfig.src / .dst populate src_mac/dst_mac on every outgoing frame; the
RX thread drops frames whose dst_mac doesn't match our src so the same
radio can run both ends of a bridge without self-reception loops.
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

# Must match FrameHeaderConfig.sequence_number_bits (7 bits).
SEQ_SPACE: int = 128

# SACK bitmap width in bytes. 8 B = 64 bits covers window_size - 1 for any
# legal window (must be < SEQ_SPACE/2 = 64).
SACK_BYTES: int = 8
SACK_BITS: int = SACK_BYTES * 8
SACK_MASK: int = (1 << SACK_BITS) - 1


def seq_add(a: int, n: int) -> int:
    return (a + n) % SEQ_SPACE


def seq_lt(a: int, b: int) -> bool:
    return 0 < (b - a) % SEQ_SPACE < SEQ_SPACE // 2


def seq_leq(a: int, b: int) -> bool:
    return a == b or seq_lt(a, b)


def seq_diff(a: int, b: int) -> int:
    return (b - a) % SEQ_SPACE


@dataclass
class ARQConfig:
    window_size: int = 63
    retransmit_timeout: float = 0.1
    send_queue_maxsize: int = 64
    src: int = 0
    dst: int = 1


@dataclass
class ARQStats:
    tun_in:           int = 0
    tun_out:          int = 0
    tun_dropped:      int = 0
    data_tx:          int = 0
    data_retransmit:  int = 0
    ack_tx:           int = 0
    data_rx_ok:       int = 0
    data_rx_buffered: int = 0
    data_rx_dup:      int = 0
    data_rx_foreign:  int = 0
    ack_rx:           int = 0
    sack_rx:          int = 0
    timeouts:         int = 0


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

        self._send_queue: queue.Queue[bytes] = queue.Queue(
            maxsize=self.config.send_queue_maxsize
        )

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
        # Unblock TX in case it is waiting on _ack_event.
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
        to_send:   list[int] = []
        in_flight: set[int]  = set()
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

                # SACK'd seqs stay in window (peer may flush its buffer and need them again) but skip retransmit.
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
        bitmap &= SACK_MASK
        bytes_be = np.array(
            [(bitmap >> (8 * (SACK_BYTES - 1 - i))) & 0xFF for i in range(SACK_BYTES)],
            dtype=np.uint8,
        )
        bits = np.unpackbits(bytes_be)
        packet = Packet(
            src_mac=self.config.src,
            dst_mac=self.config.dst,
            type=FRAME_TYPE_ACK,
            seq_num=cumul,
            length=SACK_BYTES,
            payload=bits,
            mod_scheme=ModulationSchemes.BPSK,
            coding_rate=CodeRates.NONE,
            valid=True,
        )
        self.stats.ack_tx += 1
        self.pluto_tx(packet)

    @staticmethod
    def _build_sack_bitmap(cumul: int, buffered: dict[int, bytes], window_size: int) -> int:
        bits = 0
        for i in range(window_size - 1):
            seq = seq_add(cumul, i + 2)
            if seq in buffered:
                bits |= (1 << i)
        return bits

    def _run_rx(self) -> None:
        expected_seq: int = 0
        last_acked: int = -1
        buffered: dict[int, bytes] = {}
        window_size: int = self.config.window_size

        while not self._stop_event.is_set():
            packets = self.pluto_rx()

            for packet in packets:
                if not packet.valid:
                    continue

                # dst_mac == -1 is the unset Packet default — bypass for tests / loopback fixtures.
                if packet.dst_mac >= 0 and packet.dst_mac != self.config.src:
                    self.stats.data_rx_foreign += 1
                    continue

                if packet.type == FRAME_TYPE_DATA:
                    dist = seq_diff(expected_seq, packet.seq_num)

                    if dist == 0:
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
                        # Stale duplicate of an already-delivered seq; re-ACK so the sender slides.
                        self.stats.data_rx_dup += 1
                        if last_acked >= 0:
                            bitmap = self._build_sack_bitmap(last_acked, buffered, window_size)
                            self._send_ack_frame(last_acked, bitmap)

                elif packet.type == FRAME_TYPE_ACK:
                    self.stats.ack_rx += 1
                    if packet.length >= SACK_BYTES and packet.payload.size >= SACK_BITS:
                        b = np.packbits(packet.payload[:SACK_BITS].astype(np.uint8))
                        ack_bitmap = 0
                        for byte in b:
                            ack_bitmap = (ack_bitmap << 8) | int(byte)
                    else:
                        ack_bitmap = 0
                    with self._ack_lock:
                        self._last_ack_cumul  = packet.seq_num
                        self._last_ack_bitmap = ack_bitmap
                    self._ack_event.set()


def _window_seqs(send_base: int, next_seq: int) -> list[int]:
    seqs: list[int] = []
    seq = send_base
    while seq != next_seq:
        seqs.append(seq)
        seq = seq_add(seq, 1)
    return seqs
