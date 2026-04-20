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

# Force line-buffered stdout so prints are visible even if the process is killed.
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from pluto.config import PIPELINE, DAC_SCALE, configure_rx, configure_tx
from pluto.sdr_stream import RxStream

import logging
logging.basicConfig(level=logging.INFO)

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
parser.add_argument("--cfo-offset", type=int, default=15200, help="CFO offset of RX relative to TX")
parser.add_argument("--freq", type=float, default=PIPELINE.CENTER_FREQ, help="Center frequency")
parser.add_argument("--constellation", action="store_true", help="Show live PSK8 constellation plot (RX mode only)")
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

frequency = args.freq
if args.mode in ("tx", "both"):
    configure_tx(sdr, freq=frequency + args.cfo_offset, gain=args.gain, cyclic=False)

if args.mode in ("rx", "both"):
    configure_rx(sdr, freq=frequency, gain_mode="slow_attack")

# ---------------------------------------------------------------------------
# RX buffer sizing (needed for RX and both modes, and for the TX probe)
# ---------------------------------------------------------------------------

_probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len      = len(_probe_samples)
rx_buf_size    = 16 * int(2 ** np.ceil(np.log2(frame_len)))

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
    current    = None  # pre-built first batch handed over from previous burst

    # Ensure packets-per-burst is a multiple of batch_size.  If not, the last
    # batch would be short and zero-padded to match the fixed DMA buffer length.
    # That silence causes a false preamble detection at the silence boundary,
    # which advances search_from past real packets in the next buffer → ~50% drops.
    n_per_burst = args.packets
    if n_per_burst % args.batch_size != 0:
        n_per_burst = int(np.ceil(n_per_burst / args.batch_size)) * args.batch_size
        print(f"  [TX] NOTE: --packets {args.packets} not divisible by "
              f"--batch-size {args.batch_size}; adjusted to {n_per_burst} "
              f"to avoid silence gaps")

    while True:
        burst_num   += 1
        burst_start  = global_seq
        n            = n_per_burst
        batch_size   = args.batch_size
        offsets      = list(range(0, n, batch_size))
        chunk_len    = None
        air_time     = None

        # Lookahead: build first batch before entering the send loop,
        # unless it was already pre-built during the previous burst's sleep.
        if current is None:
            current = _build_batch(burst_start, min(batch_size, n))

        print(f"  [TX] burst {burst_num}: {n} packets "
              f"(seq {burst_start}–{burst_start + n - 1}), "
              f"{len(offsets)} batch(es) of {batch_size}")

        t0 = time.perf_counter()
        next_burst_first = None  # pre-built first batch of the *next* burst
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
            else:
                # Last batch of this burst — pre-build the next burst's first
                # batch so sdr.tx() fires immediately after the sleep with no
                # extra build gap between bursts.
                next_burst_first = _build_batch(burst_start + n,
                                                min(batch_size, n))

            elapsed   = time.perf_counter() - t_tx_start
            remaining = air_time - elapsed
            if remaining > 0:
                time.sleep(remaining)

        global_seq += n
        t1 = time.perf_counter()
        # Hand the pre-built batch to the next iteration so the while-loop
        # body skips the build step and goes straight to sdr.tx().
        if next_burst_first is not None:
            current = next_burst_first

        print(f"  [TX] burst {burst_num} done: {t1 - t0:.3f} s  "
              f"({n * args.payload / (t1 - t0):.0f} B/s)")

        time.sleep(args.interval / 1000.0)

# ---------------------------------------------------------------------------
# RX mode — run forever, print every decoded packet
# ---------------------------------------------------------------------------

def _setup_constellation_plot():
    """Set up a live PSK8 constellation figure. Returns (fig, ax, scatter_handle)."""
    import matplotlib
    matplotlib.use("TkAgg" if "DISPLAY" in __import__("os").environ else "Agg")
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("PSK8 constellation — accumulating…")
    ax.grid(True, alpha=0.25)

    # Decision boundary lines at every π/4
    for k in range(8):
        angle = k * np.pi / 4 + np.pi / 8
        ax.plot([0, 1.55 * np.cos(angle)], [0, 1.55 * np.sin(angle)],
                "k--", alpha=0.15, linewidth=0.8)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.08, linewidth=1)

    # Ideal 8-PSK points
    ideal = rx_pipe.psk8.symbol_mapping
    ax.scatter(ideal.real, ideal.imag, s=200, marker="*", c="red",
               zorder=6, label="Ideal", linewidths=0)

    # Received symbols — start empty
    scat = ax.scatter([], [], s=6, alpha=0.35, c="steelblue",
                      label="Received", linewidths=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.pause(0.01)
    return fig, ax, scat


def run_rx():
    # Lossless stream: large queue so the hardware reader never stalls while
    # the decoder is busy.
    stream = RxStream(sdr, maxsize=128, lossless=True)
    stream.start(flush=16)

    print("  [RX] listening …")

    # --- optional constellation plot ---
    _fig = _ax = _scat = None
    _sym_buf: list[np.ndarray] = []   # post-Costas symbols accumulated across packets
    _pkt_count = 0                    # valid packets since last plot refresh
    _PLOT_EVERY = 20                  # update plot every N valid packets
    if args.constellation:
        import matplotlib.pyplot as plt
        _fig, _ax, _scat = _setup_constellation_plot()

    n_total     = 0
    n_valid     = 0
    n_dropped   = 0
    last_seq    = None
    prev_buf    = None
    search_from = 0

    # Processing time stats (printed every 500 buffers)
    _proc_times: list[float] = []
    _buf_count  = 0

    try:
        while True:
            try:
                curr_buf = stream.get(timeout=0.05)
            except queue.Empty:
                if args.constellation and _fig is not None:
                    import matplotlib.pyplot as plt
                    plt.pause(0.001)
                continue

            prev_len = len(prev_buf) if prev_buf is not None else 0
            raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

            _t0 = time.perf_counter()
            packets, max_det = rx_pipe.receive(raw, search_from=search_from)
            _proc_ms = (time.perf_counter() - _t0) * 1e3
            _proc_times.append(_proc_ms)
            _buf_count += 1
            if _buf_count % 500 == 0:
                buf_dur_ms = rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3
                print(f"  [RX perf] proc={np.mean(_proc_times):.2f} ms avg / "
                      f"{max(_proc_times):.2f} ms max  (buf={buf_dur_ms:.2f} ms, "
                      f"queue≈{stream._q.qsize()}/{stream._q.maxsize})")
                _proc_times.clear()

            prev_buf = curr_buf
            if packets:
                last_ps = max(pkt.sample_start for pkt in packets)
                search_from = max(0, max(last_ps, max_det) - prev_len)
            else:
                search_from = max(0, max_det - prev_len)

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

                    # Collect symbols for constellation plot
                    if args.constellation and pkt.rx_symbols is not None:
                        _sym_buf.append(pkt.rx_symbols)
                        _pkt_count += 1
                        if _pkt_count >= _PLOT_EVERY:
                            import matplotlib.pyplot as plt
                            all_syms = np.concatenate(_sym_buf)
                            _scat.set_offsets(np.column_stack([all_syms.real, all_syms.imag]))
                            _ax.set_title(
                                f"PSK8 constellation — last {_pkt_count} pkts "
                                f"({len(all_syms)} symbols)"
                            )
                            _fig.canvas.flush_events()
                            plt.pause(0.001)
                            _sym_buf.clear()
                            _pkt_count = 0
                else:
                    print(f"  [RX] #{n_total}  header CRC failed  "
                          f"(ok={n_valid}, dropped≈{n_dropped})")

    except KeyboardInterrupt:
        print(f"\n  [RX] interrupted — decoded {n_valid} valid / {n_total} total frames, ~{n_dropped} dropped by seq gap")
    finally:
        stream.stop()
        if args.constellation and _fig is not None:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()  # keep window open after run ends

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

    # Round up to the next full batch — the last batch is zero-padded to the
    # full batch length, so actual air time includes that padding.
    _padded_packets = int(np.ceil(args.packets / args.batch_size)) * args.batch_size
    _air_time_ms   = int(frame_len * _padded_packets / pipe_cfg.SAMPLE_RATE * 1000)
    _buf_ms        = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000)
    _BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 8

    print(f"Post-TX   : {_BUFS_AFTER_TX} bufs needed after tx_done  "
          f"(air={_air_time_ms} ms / buf={_buf_ms} ms + 8 margin, "
          f"padded {args.packets}→{_padded_packets} pkts)\n")

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

            prev_len = len(prev_buf) if prev_buf is not None else 0
            raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

            packets, max_det = rx_pipe.receive(raw, search_from=search_from)

            prev_buf = curr_buf
            if packets:
                last_ps = max(pkt.sample_start for pkt in packets)
                search_from = max(0, max(last_ps, max_det) - prev_len)
            else:
                search_from = max(0, max_det - prev_len)

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

