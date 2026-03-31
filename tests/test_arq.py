"""Tests for the Go-Back-N ARQ module."""

import queue
import threading
import time

import numpy as np
import pytest

from modules.arq import (
    ARQConfig,
    ARQNode,
    FRAME_TYPE_ACK,
    FRAME_TYPE_DATA,
    SEQ_SPACE,
    _window_seqs,
    seq_add,
    seq_diff,
    seq_leq,
    seq_lt,
)
from modules.pipeline import Packet


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

class MockTun:
    """Simulates a TUN device using in-memory queues (bytes interface)."""

    def __init__(self) -> None:
        self._read_q: queue.Queue[bytes] = queue.Queue()
        self.written: list[bytes] = []
        self.write_event = threading.Event()

    def read(self) -> bytes | None:
        try:
            return self._read_q.get(timeout=0.02)
        except queue.Empty:
            return None

    def write(self, data: bytes) -> None:
        self.written.append(data)
        self.write_event.set()

    def feed(self, data: bytes) -> None:
        """Inject a raw IP packet to be returned by read()."""
        self._read_q.put(data)


class MockRadio:
    """Simulates the pluto_tx / pluto_rx pair using in-memory queues."""

    def __init__(self) -> None:
        self.sent: list[Packet] = []
        self._rx_q: queue.Queue[Packet] = queue.Queue()
        self.sent_event = threading.Event()

    def tx(self, packet: Packet) -> None:
        self.sent.append(packet)
        self.sent_event.set()

    def rx(self) -> list[Packet]:
        """Return one packet as a single-element list, or [] if none available."""
        try:
            return [self._rx_q.get(timeout=0.02)]
        except queue.Empty:
            return []

    def inject(self, packet: Packet) -> None:
        """Inject a packet to be returned by the next rx() call."""
        self._rx_q.put(packet)

    def wait_for_n_sent(self, n: int, timeout: float = 2.0) -> list[Packet]:
        """Block until at least n packets appear in sent, then return them."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.sent) >= n:
                return list(self.sent[:n])
            self.sent_event.wait(timeout=0.05)
            self.sent_event.clear()
        return list(self.sent)


def _ip_bytes(val: int) -> bytes:
    """Make a distinct 4-byte test payload."""
    return bytes([val & 0xFF, (val >> 8) & 0xFF, 0xAB, 0xCD])


def _bits_for(data: bytes) -> np.ndarray:
    """Convert bytes to a numpy bit array (as the DSP pipeline produces)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def make_data_packet(seq: int, data: bytes) -> Packet:
    """Create a DATA Packet as the RX pipeline would produce it."""
    return Packet(
        type=FRAME_TYPE_DATA,
        seq_num=seq,
        length=len(data),
        payload=_bits_for(data),
        valid=True,
    )


def make_ack(seq: int) -> Packet:
    return Packet(type=FRAME_TYPE_ACK, seq_num=seq, valid=True)


# ---------------------------------------------------------------------------
# Tests: sequence arithmetic
# ---------------------------------------------------------------------------

class TestSeqArith:
    def test_seq_add_normal(self):
        assert seq_add(3, 4) == 7

    def test_seq_add_wraparound(self):
        assert seq_add(14, 3) == 1

    def test_seq_add_negative(self):
        assert seq_add(0, -1) == 15

    def test_seq_lt_normal(self):
        assert seq_lt(2, 7)
        assert not seq_lt(7, 2)

    def test_seq_lt_wraparound(self):
        # 14 comes before 1 in the circular space
        assert seq_lt(14, 1)
        assert not seq_lt(1, 14)

    def test_seq_lt_equal(self):
        assert not seq_lt(5, 5)

    def test_seq_leq_equal(self):
        assert seq_leq(5, 5)

    def test_seq_leq_lt(self):
        assert seq_leq(3, 5)
        assert seq_leq(14, 0)

    def test_seq_diff_normal(self):
        assert seq_diff(2, 5) == 3

    def test_seq_diff_zero(self):
        assert seq_diff(7, 7) == 0

    def test_seq_diff_wraparound(self):
        assert seq_diff(14, 2) == 4  # 14→15→0→1→2

    def test_window_seqs_normal(self):
        assert _window_seqs(2, 5) == [2, 3, 4]

    def test_window_seqs_empty(self):
        assert _window_seqs(3, 3) == []

    def test_window_seqs_wraparound(self):
        assert _window_seqs(14, 2) == [14, 15, 0, 1]

    @pytest.mark.parametrize("a", range(SEQ_SPACE))
    def test_seq_add_full_cycle(self, a):
        assert seq_add(a, SEQ_SPACE) == a


# ---------------------------------------------------------------------------
# Tests: ARQNode TX behaviour
# ---------------------------------------------------------------------------

class TestARQNodeTX:
    def test_sends_data_frames_for_tun_input(self):
        """DATA frames with correct seq numbers are sent when TUN feeds payloads."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(window_size=4, retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)

        # Pre-seed send queue to avoid TUN reader / TX thread race
        for i in range(3):
            node._send_queue.put(_ip_bytes(i))

        node.start()

        pkts = radio.wait_for_n_sent(3)
        node.stop()

        data_pkts = [p for p in pkts if p.type == FRAME_TYPE_DATA]
        assert len(data_pkts) == 3
        assert [p.seq_num for p in data_pkts] == [0, 1, 2]

    def test_window_slides_on_ack(self):
        """After ACK, window slides and new payloads get fresh seq numbers."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(window_size=2, retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)

        # Pre-seed first two payloads to avoid TUN reader / TX thread race
        node._send_queue.put(_ip_bytes(0))
        node._send_queue.put(_ip_bytes(1))

        node.start()
        radio.wait_for_n_sent(2)

        # ACK both and queue a third payload; TX should slide window and send it
        node._send_queue.put(_ip_bytes(2))
        radio.inject(make_ack(0))
        radio.inject(make_ack(1))

        pkts = radio.wait_for_n_sent(3)
        node.stop()

        data_pkts = [p for p in pkts if p.type == FRAME_TYPE_DATA]
        seq_nums = [p.seq_num for p in data_pkts]
        assert 0 in seq_nums
        assert 1 in seq_nums
        assert 2 in seq_nums

    def test_retransmit_on_timeout(self):
        """TX retransmits the full window after retransmit_timeout with no ACK."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(window_size=4, retransmit_timeout=0.05)
        node = ARQNode(tun, radio.tx, radio.rx, config)

        node._send_queue.put(_ip_bytes(1))
        node._send_queue.put(_ip_bytes(2))

        node.start()

        # Wait long enough for at least one retransmit cycle (2 original + 2 retransmit)
        radio.wait_for_n_sent(4, timeout=2.0)
        node.stop()

        data_pkts = [p for p in radio.sent if p.type == FRAME_TYPE_DATA]
        assert len(data_pkts) >= 4  # at least one retransmit of both frames

        # Same seq numbers should appear multiple times
        seq_counts: dict[int, int] = {}
        for p in data_pkts:
            seq_counts[p.seq_num] = seq_counts.get(p.seq_num, 0) + 1
        assert seq_counts[0] >= 2
        assert seq_counts[1] >= 2


# ---------------------------------------------------------------------------
# Tests: ARQNode RX behaviour
# ---------------------------------------------------------------------------

class TestARQNodeRX:
    def test_in_order_frames_delivered_to_tun(self):
        """In-order DATA frames are written to TUN in sequence."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)
        node.start()

        payloads = [_ip_bytes(i) for i in range(3)]
        for seq, data in enumerate(payloads):
            radio.inject(make_data_packet(seq, data))

        # Wait for 3 TUN writes
        deadline = time.monotonic() + 2.0
        while len(tun.written) < 3 and time.monotonic() < deadline:
            tun.write_event.wait(timeout=0.1)
            tun.write_event.clear()

        node.stop()

        assert len(tun.written) == 3
        for i, received in enumerate(tun.written):
            assert received == payloads[i]

    def test_acks_sent_for_in_order_frames(self):
        """An ACK is sent for each correctly received DATA frame."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)
        node.start()

        for seq in range(3):
            radio.inject(make_data_packet(seq, _ip_bytes(seq)))

        ack_pkts = []
        deadline = time.monotonic() + 2.0
        while len(ack_pkts) < 3 and time.monotonic() < deadline:
            radio.wait_for_n_sent(len(ack_pkts) + 1, timeout=0.2)
            ack_pkts = [p for p in radio.sent if p.type == FRAME_TYPE_ACK]

        node.stop()

        assert len(ack_pkts) == 3
        assert [p.seq_num for p in ack_pkts] == [0, 1, 2]

    def test_out_of_order_frame_discarded(self):
        """Out-of-order DATA frame is not delivered to TUN."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)
        node.start()

        radio.inject(make_data_packet(0, _ip_bytes(0)))
        time.sleep(0.05)
        radio.inject(make_data_packet(2, _ip_bytes(2)))  # out of order
        time.sleep(0.1)

        node.stop()

        assert len(tun.written) == 1
        assert tun.written[0] == _ip_bytes(0)

    def test_out_of_order_triggers_repeat_ack(self):
        """Out-of-order frame causes a repeat ACK for the last in-order seq."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)
        node.start()

        radio.inject(make_data_packet(0, _ip_bytes(0)))
        time.sleep(0.05)
        radio.inject(make_data_packet(2, _ip_bytes(2)))  # out of order

        radio.wait_for_n_sent(2, timeout=1.0)
        node.stop()

        ack_pkts = [p for p in radio.sent if p.type == FRAME_TYPE_ACK]
        # First ACK for seq=0, second (repeat) ACK also for seq=0
        assert len(ack_pkts) >= 2
        assert all(p.seq_num == 0 for p in ack_pkts)


# ---------------------------------------------------------------------------
# Tests: sequence number wraparound
# ---------------------------------------------------------------------------

class TestWraparound:
    def test_window_advancement_wraps(self):
        """Window correctly slides past the seq=15→0 boundary."""
        tun = MockTun()
        radio = MockRadio()
        config = ARQConfig(window_size=4, retransmit_timeout=2.0)
        node = ARQNode(tun, radio.tx, radio.rx, config)
        node.start()

        # Send 18 frames to force wraparound (seq 0..15 then 0,1)
        for i in range(18):
            tun.feed(_ip_bytes(i))

        # Feed ACKs so window keeps sliding
        def ack_feeder():
            for seq in range(18):
                time.sleep(0.03)
                radio.inject(make_ack(seq % SEQ_SPACE))

        feeder = threading.Thread(target=ack_feeder, daemon=True)
        feeder.start()

        radio.wait_for_n_sent(18, timeout=5.0)
        node.stop()

        data_pkts = [p for p in radio.sent if p.type == FRAME_TYPE_DATA]
        sent_seqs = [p.seq_num for p in data_pkts]

        # Must have seen all seq numbers 0..15 plus at least seq 0 and 1 again
        assert len(data_pkts) >= 18
        assert set(sent_seqs[:16]) == set(range(16))
