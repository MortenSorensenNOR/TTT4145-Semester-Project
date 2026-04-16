"""Realistic threaded loopback test.

Spins up two threads — TX and RX — that run independently, mimicking a real
intermittent transmitter and a continuously listening receiver.

TX thread:
  - Builds a fresh packet per transmission with an incrementing sequence number
  - Waits a configurable inter-packet gap between transmissions
  - Non-cyclic: each packet is sent once, not looped

RX thread:
  - Drains the SDR RX buffer via RxStream (lossless, large queue)
  - Searches each capture for decodable frames
  - Tracks sequence numbers and reports packet drop rate at the end

Usage:
    python pluto/loopback_threaded.py [options]

Options:
    --gain        TX hardware gain in dB        (default: -30)
    --payload     Payload size in bytes         (default: 10)
    --packets     Number of packets to TX       (default: 20)
    --batch-size  Packets per TX window         (default: 8)
    --interval    Inter-burst gap in ms         (default: 200)
    --ip          PlutoSDR IP address           (default: 192.168.2.1)
    --mode        Operation mode: 'tx', 'rx', or 'both' (default: 'both')
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
from pluto.sdr_stream import RxStream

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",     type=float, default=-30,           help="TX gain in dB (default: -30)")
parser.add_argument("--payload",  type=int,   default=10,            help="Payload bytes (default: 10)")
parser.add_argument("--packets",  type=int,   default=20,            help="Number of packets per TX burst (default: 20)")
parser.add_argument("--interval", type=float, default=200,           help="Inter-burst gap in ms (default: 200)")
parser.add_argument("--ip",       type=str,   default="192.168.2.1", help="PlutoSDR IP (default: 192.168.2.1)")
parser.add_argument("--batch-size",type=int,   default=8,             help="Packets per TX batch/window (default: 8)")
parser.add_argument("--mode",     type=str,   default="both",        help="Mode: 'tx', 'rx', or 'both' (default: both)")
args = parser.parse_args()

if args.mode not in ("tx", "rx", "both"):
    print(f"ERROR: --mode must be 'tx', 'rx', or 'both', got '{args.mode}'")
    sys.exit(1)

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

sdr = adi.Pluto("ip:" + args.ip)

if args.mode in ("tx", "both"):
    configure_tx(sdr, freq=CENTER_FREQ, gain=args.gain, cyclic=False)

if args.mode in ("rx", "both"):
    configure_rx(sdr, freq=CENTER_FREQ, gain_mode="fast_attack")

# ---------------------------------------------------------------------------
# RX buffer sizing (needed for RX and both modes, and for the TX probe)
# ---------------------------------------------------------------------------

_probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len      = len(_probe_samples)
rx_buf_size    = 2 * int(2 ** np.ceil(np.log2(frame_len)))

if args.mode in ("rx", "both"):
    sdr.rx_buffer_size = rx_buf_size

print(f"Mode      : {args.mode}")
print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
print(f"Payload   : {args.payload} bytes  ({args.payload * 8} bits)")
print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
if args.mode in ("rx", "both"):
    print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"Gap       : {args.interval} ms")
print(f"Packets   : {args.packets} per burst")
print(f"TX gain   : {args.gain} dB\n")

# ---------------------------------------------------------------------------
# TX helpers
# ---------------------------------------------------------------------------

def _build_batch(seq_start: int, count: int) -> np.ndarray:
    """Build `count` packets starting at seq_start; return DAC-scaled samples."""
    chunks = []
    for i in range(count):
        seq       = (seq_start + i) % (2**32)
        seq_bytes = np.array([(seq >> 24) & 0xFF, (seq >> 16) & 0xFF,
                               (seq >>  8) & 0xFF,  seq        & 0xFF], dtype=np.uint8)
        sequence_bits = np.unpackbits(seq_bytes)
        random_bits   = rng.integers(0, 2, args.payload * 8 - 32, dtype=np.uint8)
        payload_bits  = np.concatenate([sequence_bits, random_bits])
        pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=payload_bits)
        samples = tx_pipe.transmit(pkt)
        peak    = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak
        chunks.append((samples * DAC_SCALE).astype(np.complex64))
    return np.concatenate(chunks)

# ---------------------------------------------------------------------------
# TX mode — loop forever: send burst in batches, wait, repeat
# ---------------------------------------------------------------------------

def run_tx():
    burst_num  = 0
    global_seq = 0

    while True:
        burst_num   += 1
        burst_start  = global_seq
        n            = args.packets
        batch_size   = args.batch_size
        offsets      = list(range(0, n, batch_size))
        chunk_len    = None
        air_time     = None

        # Lookahead: build first batch before entering the send loop.
        current = _build_batch(burst_start, min(batch_size, n))

        print(f"  [TX] burst {burst_num}: {n} packets "
              f"(seq {burst_start}–{burst_start + n - 1}), "
              f"{len(offsets)} batch(es) of {batch_size}")

        t0 = time.perf_counter()
        for i, offset in enumerate(offsets):
            # Lock buffer length to first batch; zero-pad shorter tail batches.
            if chunk_len is None:
                chunk_len = len(current)
                air_time  = chunk_len / pipe_cfg.SAMPLE_RATE
            if len(current) < chunk_len:
                current = np.concatenate([current,
                    np.zeros(chunk_len - len(current), dtype=np.complex64)])

            t_tx_start = time.perf_counter()
            sdr.tx(current)

            # Overlap: build next batch while hardware is on air.
            if i + 1 < len(offsets):
                next_offset = offsets[i + 1]
                current = _build_batch(burst_start + next_offset,
                                       min(batch_size, n - next_offset))

            elapsed   = time.perf_counter() - t_tx_start
            remaining = air_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

        global_seq += n
        t1 = time.perf_counter()

        print(f"  [TX] burst {burst_num} done: {t1 - t0:.3f} s  "
              f"({n * args.payload / (t1 - t0):.0f} B/s)")

        time.sleep(args.interval / 1000.0)

# ---------------------------------------------------------------------------
# RX mode — run forever, print every decoded packet
# ---------------------------------------------------------------------------

def run_rx():
    # Lossless stream: large queue so the hardware reader never stalls while
    # the decoder is busy.
    stream = RxStream(sdr, maxsize=128, lossless=True)
    stream.start(flush=16)

    print("  [RX] listening …")

    n_total     = 0
    n_valid     = 0
    n_dropped   = 0
    last_seq    = None
    prev_buf    = None
    search_from = 0

    try:
        while True:
            try:
                curr_buf = stream.get(timeout=0.05)
            except queue.Empty:
                continue

            raw      = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
            prev_len = len(prev_buf) if prev_buf is not None else 0
            prev_buf = curr_buf

            packets = rx_pipe.receive(raw, search_from=search_from)

            if packets:
                last_ps     = max(pkt.sample_start for pkt in packets)
                search_from = max(0, last_ps - prev_len)
            else:
                search_from = 0

            for pkt in packets:
                n_total += 1
                if pkt.valid:
                    seq_bits = pkt.payload[:32]
                    b        = np.packbits(seq_bits)
                    seq      = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])
                    n_valid += 1

                    gap = 0
                    if last_seq is not None:
                        gap = (seq - last_seq - 1) % (2**32)
                        if gap > 0:
                            n_dropped += gap
                    last_seq = seq

                    gap_str = f"  *** GAP: {gap} dropped ***" if gap > 0 else ""
                    print(f"  [RX] #{n_total}  seq={seq:10d}  valid=True   "
                          f"(ok={n_valid}, dropped≈{n_dropped}){gap_str}")
                else:
                    print(f"  [RX] #{n_total}  header CRC failed  "
                          f"(ok={n_valid}, dropped≈{n_dropped})")

    except KeyboardInterrupt:
        print(f"\n  [RX] interrupted — decoded {n_valid} valid / {n_total} total frames, ~{n_dropped} dropped by seq gap")
    finally:
        stream.stop()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if args.mode == "tx":
    try:
        run_tx()
    except KeyboardInterrupt:
        print("\n[TX] interrupted")

elif args.mode == "rx":
    run_rx()

else:  # "both" — original threaded loopback behaviour
    # ---------------------------------------------------------------------------
    # Shared state
    # ---------------------------------------------------------------------------

    rx_results: list[dict] = []
    rx_lock      = threading.Lock()
    tx_done      = threading.Event()
    rx_ready     = threading.Event()
    _decoded_seqs: set[int] = set()
    _all_decoded = threading.Event()

    _air_time_ms   = int(frame_len * args.packets / pipe_cfg.SAMPLE_RATE * 1000)
    _buf_ms        = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000)
    _BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 8

    print(f"Post-TX   : {_BUFS_AFTER_TX} bufs needed after tx_done  "
          f"(air={_air_time_ms} ms / buf={_buf_ms} ms + 8 margin)\n")

    # -----------------------------------------------------------------------
    # TX thread
    # -----------------------------------------------------------------------

    def tx_thread():
        rx_ready.wait()

        chunks = []
        for seq in range(args.packets):
            seq_bytes     = np.array([(seq >> 24) & 0xFF, (seq >> 16) & 0xFF,
                                      (seq >>  8) & 0xFF,  seq        & 0xFF], dtype=np.uint8)
            sequence_bits = np.unpackbits(seq_bytes)
            random_bits   = rng.integers(0, 2, args.payload * 8 - 32, dtype=np.uint8)
            payload_bits  = np.concatenate([sequence_bits, random_bits])
            pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=payload_bits)
            samples = tx_pipe.transmit(pkt)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak
            chunks.append((samples * DAC_SCALE).astype(np.complex64))

        all_samples = np.concatenate(chunks)
        air_time_s  = len(all_samples) / pipe_cfg.SAMPLE_RATE

        t0 = time.perf_counter()
        sdr.tx(all_samples)
        remaining = air_time_s - (time.perf_counter() - t0)
        if remaining > 0:
            time.sleep(remaining)
        t1 = time.perf_counter()

        print(f"Took: {t1 - t0} seconds. Throughput: {args.packets * args.payload / (t1 - t0)} B/s")
        tx_done.set()
        print("  [TX] done")

    # -----------------------------------------------------------------------
    # RX thread
    # -----------------------------------------------------------------------

    def rx_thread():
        stream = RxStream(sdr, maxsize=128, lossless=True)
        stream.start(flush=16)
        rx_ready.set()

        prev_buf    = None
        search_from = 0
        n_after_tx  = 0

        while not _all_decoded.is_set():
            if tx_done.is_set() and n_after_tx >= _BUFS_AFTER_TX:
                break
            try:
                curr_buf = stream.get(timeout=0.05)
            except queue.Empty:
                continue

            if tx_done.is_set():
                n_after_tx += 1

            raw      = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
            prev_len = len(prev_buf) if prev_buf is not None else 0
            prev_buf = curr_buf

            packets = rx_pipe.receive(raw, search_from=search_from)

            if packets:
                last_ps     = max(pkt.sample_start for pkt in packets)
                search_from = max(0, last_ps - prev_len)
            else:
                search_from = 0

            for pkt in packets:
                if pkt.valid:
                    seq_bits = pkt.payload[:32]
                    b        = np.packbits(seq_bits)
                    seq      = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])
                else:
                    seq = -1
                entry    = {"seq_num": seq, "valid": pkt.valid, "time": time.perf_counter()}
                if pkt.valid:
                    print(f"  [RX] decoded seq={seq:10d}, valid={pkt.valid}")
                else:
                    print(f"  [RX] frame found but header CRC failed")
                with rx_lock:
                    rx_results.append(entry)
                    if pkt.valid:
                        _decoded_seqs.add(seq)
                        if len(_decoded_seqs) >= args.packets:
                            _all_decoded.set()

        stream.stop()
        print("  [RX] done")

    # -----------------------------------------------------------------------
    # Launch threads
    # -----------------------------------------------------------------------

    t_tx = threading.Thread(target=tx_thread, name="TX", daemon=True)
    t_rx = threading.Thread(target=rx_thread, name="RX", daemon=True)

    t_rx.start()
    t_tx.start()

    t_tx.join()
    t_rx.join(timeout=args.packets * args.interval / 1000.0 + _air_time_ms * 4 / 1000.0 + 15.0)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    print()
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)

    n_tx  = args.packets
    n_rx  = len(rx_results)
    valid = [r for r in rx_results if r["valid"]]

    seen_seqs    = set()
    unique_valid = []
    for r in valid:
        if r["seq_num"] not in seen_seqs:
            seen_seqs.add(r["seq_num"])
            unique_valid.append(r)

    n_unique  = len(unique_valid)
    n_dropped = n_tx - n_unique
    drop_rate = n_dropped / n_tx if n_tx > 0 else float("nan")

    missing_seqs = sorted(set(range(n_tx)) - seen_seqs)

    if len(unique_valid) >= 2:
        rx_duration   = unique_valid[-1]["time"] - unique_valid[0]["time"]
        rx_throughput = n_unique * args.payload / rx_duration if rx_duration > 0 else float("nan")
    else:
        rx_duration   = float("nan")
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

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

if args.mode in ("tx", "both"):
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
del sdr

sys.exit(0)

