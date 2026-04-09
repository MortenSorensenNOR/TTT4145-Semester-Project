"""Realistic threaded loopback test.

Spins up two threads — TX and RX — that run independently, mimicking a real
intermittent transmitter and a continuously listening receiver.

TX thread:
  - Builds a fresh packet per transmission with an incrementing sequence number
  - Waits a configurable inter-packet gap between transmissions
  - Non-cyclic: each packet is sent once, not looped

RX thread:
  - Continuously drains the SDR RX buffer
  - Searches each capture for decodable frames
  - Tracks sequence numbers and reports packet drop rate at the end

Usage:
    python pluto/loopback_threaded.py [options]

Options:
    --gain       TX hardware gain in dB   (default: -30)
    --payload    Payload size in bytes    (default: 10)
    --packets    Number of packets to TX  (default: 20)
    --interval   Inter-packet gap in ms   (default: 200)
    --ip         PlutoSDR IP address      (default: 192.168.2.1)
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from pluto.config import CENTER_FREQ, DAC_SCALE, configure_rx, configure_tx

_SCALE = np.float32(2.0 / DAC_SCALE)


def _read_buf(sdr: adi.Pluto) -> np.ndarray:
    """Read one hardware buffer directly, blocking for the hardware fill time."""
    if not sdr._rxbuf:
        sdr._rx_init_channels()
    sdr._rxbuf.refill()
    raw = np.frombuffer(sdr._rxbuf.read(), dtype=np.int16)
    arr = np.empty((len(raw) // 2, 2), dtype=np.float32)
    arr[:, 0] = raw[0::2]
    arr[:, 1] = raw[1::2]
    out = arr.view(np.complex64).reshape(-1)
    out *= _SCALE
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",     type=float, default=-30,          help="TX gain in dB (default: -30)")
parser.add_argument("--payload",  type=int,   default=10,           help="Payload bytes (default: 10)")
parser.add_argument("--packets",  type=int,   default=20,           help="Number of packets to transmit (default: 20)")
parser.add_argument("--interval", type=float, default=200,          help="Inter-packet gap in ms (default: 200)")
parser.add_argument("--ip",       type=str,   default="192.168.2.1", help="PlutoSDR IP (default: 192.168.2.1)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

pipe_cfg = PipelineConfig(hardware_rrc=False)
tx_pipe  = TXPipeline(pipe_cfg)
rx_pipe  = RXPipeline(pipe_cfg)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# SDR setup — shared device (PlutoSDR supports simultaneous TX + RX)
# ---------------------------------------------------------------------------

PLUTO_IP = "ip:" + args.ip
sdr = adi.Pluto(PLUTO_IP)

configure_tx(sdr, freq=CENTER_FREQ, gain=args.gain, cyclic=False)
configure_rx(sdr, freq=CENTER_FREQ, gain_mode="fast_attack")

# Size the RX buffer to hold ~1 full frame.  The RX loop keeps the previous
# buffer and concatenates it with the current one (2-buffer sliding window) so
# frames straddling a buffer boundary are still decoded.
# We need to transmit one dummy packet first to know the frame length.
_probe_bits = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt  = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0,
                     length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len = len(_probe_samples)
rx_buf_size = 2 * int(2 ** np.ceil(np.log2(frame_len)))
sdr.rx_buffer_size = rx_buf_size

print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
print(f"Payload   : {args.payload} bytes  ({args.payload * 8} bits)")
print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"Gap       : {args.interval} ms")
print(f"Packets   : {args.packets}")
print(f"TX gain   : {args.gain} dB\n")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

rx_results: list[dict] = []   # [{seq_num, valid, bit_errors}]
rx_lock  = threading.Lock()
tx_done  = threading.Event()
rx_ready = threading.Event()   # set once RX has flushed stale DMA buffers
_decoded_seqs: set[int] = set()   # unique valid seq nums decoded so far
_all_decoded  = threading.Event() # set when len(_decoded_seqs) >= args.packets

# ---------------------------------------------------------------------------
# TX thread
# ---------------------------------------------------------------------------

def tx_thread():
    rx_ready.wait()   # don't transmit until RX has drained stale buffers

    # Pre-build all packet sample arrays, then concatenate into one buffer
    # so only a single sdr.tx() call is needed, eliminating per-packet USB overhead.
    chunks = []
    for seq in range(args.packets):
        sequence_bits = np.unpackbits(np.array([seq], dtype=np.uint8))
        random_bits = rng.integers(0, 2, args.payload * 8 - 8, dtype=np.uint8)
        payload_bits = np.concatenate([sequence_bits, random_bits])
        pkt = Packet(
            src_mac=0,
            dst_mac=1,
            type=0,
            seq_num=0,
            length=args.payload,
            payload=payload_bits,
        )
        samples = tx_pipe.transmit(pkt)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak
        chunks.append((samples * DAC_SCALE).astype(np.complex64))

    all_samples = np.concatenate(chunks)
    air_time_s = len(all_samples) / pipe_cfg.SAMPLE_RATE

    t0 = time.perf_counter()
    sdr.tx(all_samples)
    # Sleep only for the remaining air time — the hardware was already clocking
    # out samples during the USB push, so we subtract the push duration.
    remaining = air_time_s - (time.perf_counter() - t0)
    if remaining > 0:
        time.sleep(remaining)
    t1 = time.perf_counter()

    print(f"Took: {t1 - t0} seconds. Throughput: {args.packets * args.payload / (t1 - t0)} B/s")

    tx_done.set()
    print("  [TX] done")

# ---------------------------------------------------------------------------
# RX thread
# ---------------------------------------------------------------------------

# Queue between the hardware-reader thread and the processing thread.
# Sized large enough that even a ~400 ms processing spike won't overflow
# (~128 × 3.4 ms ≈ 435 ms at the default buffer size).
_buf_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=128)
_hw_reader_done = threading.Event()


def _hw_reader_thread():
    """Read hardware buffers as fast as the SDR produces them.

    This thread has no processing load so it always keeps up with the
    hardware DMA rate.  It runs until TX is finished and the grace period
    has elapsed, then signals _hw_reader_done so the processor knows to
    drain the queue and exit.
    """
    n_total = 0
    n_after_tx = 0
    t_tx_done_wall = None
    while not _all_decoded.is_set() and (not tx_done.is_set() or n_after_tx < _BUFS_AFTER_TX):
        buf = _read_buf(sdr)
        _buf_queue.put(buf)  # blocks only if queue is full (≈ 435 ms of data)
        n_total += 1
        if tx_done.is_set():
            if t_tx_done_wall is None:
                t_tx_done_wall = time.perf_counter()
            n_after_tx += 1
    wall_grace = (time.perf_counter() - t_tx_done_wall) * 1000 if t_tx_done_wall else 0
    print(f"  [HWR] done: {n_total} bufs total, {n_after_tx} after tx_done "
          f"({n_after_tx * rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.0f} ms buf-time, "
          f"{wall_grace:.0f} ms wall-time after tx_done)")
    _hw_reader_done.set()


def rx_thread():
    # Drain the libiio kernel DMA ring (typically 4 buffers deep) plus a few
    # extra to absorb USB scheduling jitter before signalling TX to start.
    for _ in range(16):
        _read_buf(sdr)

    rx_ready.set()   # tell TX thread it is safe to start transmitting

    # Start the dedicated hardware reader now that stale buffers are gone.
    threading.Thread(target=_hw_reader_thread, name="HWReader", daemon=True).start()

    prev_buf = None
    # Tracks how far into the current concatenated window we have already
    # decoded.  On the next iteration this becomes the search_from offset so
    # we don't re-detect frames that are entirely within the previous buffer.
    search_from = 0

    # t0 = time.perf_counter()
    while not _hw_reader_done.is_set() or not _buf_queue.empty():
        try:
            curr_buf = _buf_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        
        # t1 = time.perf_counter()
        # print(f"Time for new buffer: {(t1 - t0) * 1000.0} ms")
        # t0 = t1

        raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
        prev_len = len(prev_buf) if prev_buf is not None else 0
        prev_buf = curr_buf

        packets = rx_pipe.receive(raw, search_from=search_from)

        if packets:
            # Advance past the furthest decoded payload start so the next
            # window [curr|next] doesn't re-detect the same frames.
            last_ps = max(pkt.sample_start for pkt in packets)
            search_from = max(0, last_ps - prev_len)
        else:
            search_from = 0

        if not packets:
            continue

        for pkt in packets:
            sequence_number_bits = pkt.payload[:8]
            sequence_number = np.packbits(sequence_number_bits)[0]
            entry = {"seq_num": sequence_number if pkt.valid else -1, "valid": pkt.valid, "time": time.perf_counter()}
            if pkt.valid:
                print(f"  [RX] decoded seq={sequence_number}, valid={pkt.valid}")
            else:
                print(f"  [RX] frame found but header CRC failed")
            with rx_lock:
                rx_results.append(entry)
                if pkt.valid:
                    _decoded_seqs.add(sequence_number)
                    if len(_decoded_seqs) >= args.packets:
                        _all_decoded.set()
        
        # t0 = time.perf_counter()

    print("  [RX] done")


# How many data-buffers HWR must read after tx_done before it may stop.
#
# The PlutoSDR buffers the entire TX waveform before starting playback
# (sdr.tx() returns ≈ when the hardware starts transmitting).  So at
# tx_done, the hardware is just beginning to play back the signal and will
# continue for another full air_time.  We therefore need to keep reading
# for at least ceil(air_time / buf_duration) more buffers — measured in
# DATA buffers, not wall-clock time, because RX read rate slows under GIL
# pressure when the decoder is active.
_air_time_ms = int(frame_len * args.packets / pipe_cfg.SAMPLE_RATE * 1000)
_buf_ms       = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000)
_BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 8  # data-bufs needed post-tx_done

print(f"Post-TX   : {_BUFS_AFTER_TX} bufs needed after tx_done  "
      f"(air={_air_time_ms} ms / buf={_buf_ms} ms + 8 margin)\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

t_tx = threading.Thread(target=tx_thread, name="TX", daemon=True)
t_rx = threading.Thread(target=rx_thread, name="RX", daemon=True)

t_rx.start()
t_tx.start()

t_tx.join()
t_rx.join(timeout=args.packets * args.interval / 1000.0 + _air_time_ms * 4 / 1000.0 + 15.0)

sdr.tx_destroy_buffer()
del sdr  # destroy IIO context explicitly while Python state is clean

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print()
print("=" * 50)
print("RESULTS")
print("=" * 50)

n_tx   = args.packets
n_rx   = len(rx_results)
valid  = [r for r in rx_results if r["valid"]]
n_valid = len(valid)

# Deduplicate by seq_num (a frame might be decoded twice if it straddles two RX windows)
seen_seqs = set()
unique_valid = []
for r in valid:
    if r["seq_num"] not in seen_seqs:
        seen_seqs.add(r["seq_num"])
        unique_valid.append(r)

n_unique = len(unique_valid)
n_dropped = n_tx - n_unique
drop_rate = n_dropped / n_tx if n_tx > 0 else float("nan")

received_seqs  = sorted(seen_seqs)
expected_seqs  = set(range(n_tx))
missing_seqs   = sorted(expected_seqs - seen_seqs)

if len(unique_valid) >= 2:
    rx_duration = unique_valid[-1]["time"] - unique_valid[0]["time"]
    rx_throughput = n_unique * args.payload / rx_duration if rx_duration > 0 else float("nan")
else:
    rx_duration = float("nan")
    rx_throughput = float("nan")

print(f"Transmitted : {n_tx} packets")
print(f"RX captures : {n_rx}  (raw, incl. duplicates)")
print(f"Decoded OK  : {n_unique} unique valid packets")
print(f"Dropped     : {n_dropped} packets")
print(f"Drop rate   : {drop_rate * 100:.1f}%")
print(f"RX duration : {rx_duration:.3f} s")
print(f"RX throughput: {rx_throughput*8 / 1_000_000:.2f} mbps")
if missing_seqs:
    print(f"Missing seq : {missing_seqs}")
else:
    print("Missing seq : none")
print("=" * 50)

sys.exit(0 if n_dropped == 0 else 1)
