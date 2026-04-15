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
    --gain         TX hardware gain in dB            (default: -30)
    --payload      Payload size in bytes             (default: 10)
    --packets      Number of packets to TX           (default: 20)
    --interval     Inter-packet gap in ms            (default: 200)
    --ip           PlutoSDR IP address               (default: 192.168.2.1)
    --hardware-rrc Use FPGA RRC filter (custom bitstream required)
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
from pluto.config import DAC_SCALE, configure_rx, configure_tx
from pluto.sdr_stream import RxStream

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",         type=float, default=-30,           help="TX gain in dB (default: -30)")
parser.add_argument("--payload",      type=int,   default=10,            help="Payload bytes (default: 10)")
parser.add_argument("--packets",      type=int,   default=20,            help="Number of packets to transmit (default: 20)")
parser.add_argument("--interval",     type=float, default=200,           help="Inter-packet gap in ms (default: 200)")
parser.add_argument("--ip",           type=str,   default="192.168.2.1", help="PlutoSDR IP (default: 192.168.2.1)")
# parser.add_argument("--hardware-rrc", type=bool,  default=False,         help="Skip software RRC; use FPGA hardware RRC filter (requires custom bitstream)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

pipe_cfg = PipelineConfig(hardware_rrc=False)
tx_pipe  = TXPipeline(pipe_cfg)
rx_pipe  = RXPipeline(pipe_cfg)

rng = np.random.default_rng(0)

# Size the RX buffer to hold ~1 full frame.  The RX loop keeps the previous
# buffer and concatenates it with the current one (2-buffer sliding window) so
# frames straddling a buffer boundary are still decoded.
_probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len      = len(_probe_samples)
rx_buf_size    = 2 * int(2 ** np.ceil(np.log2(frame_len)))

# ---------------------------------------------------------------------------
# SDR setup — shared device (PlutoSDR supports simultaneous TX + RX)
# ---------------------------------------------------------------------------

sdr = adi.Pluto("ip:" + args.ip)
configure_tx(sdr, freq=pipe_cfg.CENTER_FREQ, gain=args.gain, cyclic=False, sample_rate=pipe_cfg.SAMPLE_RATE)
configure_rx(sdr, freq=pipe_cfg.CENTER_FREQ, gain_mode="fast_attack", sample_rate=pipe_cfg.SAMPLE_RATE, buffer_size=rx_buf_size)


# Lossless stream: large queue so the hardware reader never stalls while the
# decoder is busy.  128 × ~3.4 ms ≈ 435 ms of buffering at the default size.
stream = RxStream(sdr, maxsize=128, lossless=True)

print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}, hw_rrc={pipe_cfg.hardware_rrc}")
print(f"Payload   : {args.payload} bytes  ({args.payload * 8} bits)")
print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"Gap       : {args.interval} ms")
print(f"Packets   : {args.packets}")
print(f"TX gain   : {args.gain} dB\n")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

rx_results: list[dict] = []
rx_lock       = threading.Lock()
tx_done       = threading.Event()
rx_ready      = threading.Event()   # set once RX has flushed stale DMA buffers
_decoded_seqs: set[int] = set()
_all_decoded  = threading.Event()   # set when all expected packets are decoded

# How many data-buffers to read after tx_done before stopping.  The PlutoSDR
# buffers the entire TX waveform before starting playback, so at tx_done the
# hardware is just beginning to transmit — we must keep reading for at least
# ceil(air_time / buf_duration) more buffers.
_air_time_ms   = int(frame_len * args.packets / pipe_cfg.SAMPLE_RATE * 1000)
_buf_ms        = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000)
_BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 8

print(f"Post-TX   : {_BUFS_AFTER_TX} bufs needed after tx_done  "
      f"(air={_air_time_ms} ms / buf={_buf_ms} ms + 8 margin)\n")

# ---------------------------------------------------------------------------
# TX thread
# ---------------------------------------------------------------------------

def tx_thread():
    rx_ready.wait()   # don't transmit until RX has drained stale buffers

    # Pre-build all packet sample arrays and concatenate into one buffer so
    # only a single sdr.tx() call is needed, eliminating per-packet USB overhead.
    chunks = []
    for seq in range(args.packets):
        sequence_bits = np.unpackbits(np.array([seq], dtype=np.uint8))
        random_bits   = rng.integers(0, 2, args.payload * 8 - 8, dtype=np.uint8)
        payload_bits  = np.concatenate([sequence_bits, random_bits])
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=payload_bits)
        samples = tx_pipe.transmit(pkt)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak
        chunks.append((samples * DAC_SCALE).astype(np.complex64))

    all_samples = np.concatenate(chunks)
    air_time_s  = len(all_samples) / pipe_cfg.SAMPLE_RATE

    t0 = time.perf_counter()
    sdr.tx(all_samples)
    # Sleep only for remaining air time — hardware was already transmitting
    # during the USB push, so we subtract the push duration.
    remaining = air_time_s - (time.perf_counter() - t0)
    if remaining > 0:
        time.sleep(remaining)
    t1 = time.perf_counter()

    print(f"Took: {t1 - t0} seconds. Throughput: {args.packets * args.payload / (t1 - t0) * 8 / 1_000.0} kb/s")
    tx_done.set()
    print("  [TX] done")

# ---------------------------------------------------------------------------
# RX thread
# ---------------------------------------------------------------------------

def rx_thread():
    # Flush 16 stale DMA buffers synchronously before signalling TX to start.
    stream.start(flush=8)
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
            seq_bits = pkt.payload[:8]
            seq      = np.packbits(seq_bits)[0]
            entry    = {"seq_num": seq if pkt.valid else -1, "valid": pkt.valid, "time": time.perf_counter()}
            if pkt.valid:
                print(f"  [RX] decoded seq={seq}, valid={pkt.valid}")
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
del sdr

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

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

sys.exit(0 if n_dropped == 0 else 1)
