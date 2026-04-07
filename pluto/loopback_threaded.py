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
parser.add_argument("--gain",     type=float, default=-30,          help="TX gain in dB (default: -30)")
parser.add_argument("--payload",  type=int,   default=10,           help="Payload bytes (default: 10)")
parser.add_argument("--packets",  type=int,   default=20,           help="Number of packets to transmit (default: 20)")
parser.add_argument("--interval", type=float, default=200,          help="Inter-packet gap in ms (default: 200)")
parser.add_argument("--ip",       type=str,   default="192.168.2.1", help="PlutoSDR IP (default: 192.168.2.1)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

pipe_cfg = PipelineConfig(hardware_rrc=True)
tx_pipe  = TXPipeline(pipe_cfg)
rx_pipe  = RXPipeline(pipe_cfg)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# SDR setup — shared device (PlutoSDR supports simultaneous TX + RX)
# ---------------------------------------------------------------------------

PLUTO_IP = "ip:" + args.ip
sdr = adi.Pluto(PLUTO_IP)

configure_tx(sdr, freq=CENTER_FREQ, gain=args.gain, cyclic=False)
configure_rx(sdr, freq=CENTER_FREQ, gain_mode="slow_attack")

# Size the RX buffer to hold ~1 full frame.  The RX loop keeps the previous
# buffer and concatenates it with the current one (2-buffer sliding window) so
# frames straddling a buffer boundary are still decoded.
# We need to transmit one dummy packet first to know the frame length.
_probe_bits = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt  = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0,
                     length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len = len(_probe_samples)
rx_buf_size = int(2 ** np.ceil(np.log2(frame_len)))
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
rx_lock = threading.Lock()
tx_done = threading.Event()

# ---------------------------------------------------------------------------
# TX thread
# ---------------------------------------------------------------------------

def tx_thread():
    interval_s = args.interval / 1000.0
    for seq in range(args.packets):
        payload_bits = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
        pkt = Packet(
            src_mac=0,
            dst_mac=1,
            type=0,
            seq_num=seq,
            length=args.payload,
            payload=payload_bits,
        )
        samples = tx_pipe.transmit(pkt)
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak
        samples = (samples * DAC_SCALE).astype(np.complex64)

        sdr.tx(samples)
        time.sleep(frame_len / pipe_cfg.SAMPLE_RATE + 0.002)  # let frame clock out of DAC
        sdr.tx_destroy_buffer()  # flush kernel DMA ring before next packet
        print(f"  [TX] sent seq={seq}")
        time.sleep(interval_s)

    tx_done.set()
    print("  [TX] done")

# ---------------------------------------------------------------------------
# RX thread
# ---------------------------------------------------------------------------

stream = RxStream(sdr)
stream.start()

def rx_thread():
    stream.flush(5)
    # Drain the DMA ring until get() actually blocks — that means we are
    # at real-time and all pre-buffered stale samples have been discarded.
    while True:
        t = time.perf_counter()
        stream.get()
        if time.perf_counter() - t > 0.001:
            break

    prev_buf = None
    while not tx_done.is_set() or not _rx_queue_drained():
        t0 = time.perf_counter()
        curr_buf = stream.get()
        t1 = time.perf_counter()

        raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
        prev_buf = curr_buf

        packets = rx_pipe.receive(raw)
        t2 = time.perf_counter()
        print(f"  [RX] get={t1-t0:.3f}s  receive={t2-t1:.3f}s  pkts={len(packets)}", flush=True)

        if not packets:
            continue

        for pkt in packets:
            entry = {"seq_num": pkt.seq_num if pkt.valid else -1, "valid": pkt.valid}
            if pkt.valid:
                print(f"  [RX] decoded seq={pkt.seq_num}, valid={pkt.valid}")
            else:
                print(f"  [RX] frame found but header CRC failed")
            with rx_lock:
                rx_results.append(entry)

    print("  [RX] done")


def _rx_queue_drained() -> bool:
    """Keep draining for a short window after TX finishes so we don't miss
    packets that are still in flight through the SDR pipeline."""
    return getattr(_rx_queue_drained, "_deadline_passed", False)


# Give the RX thread a grace-period window after TX finishes
_GRACE_MS = 500

def _start_grace_timer():
    def _mark():
        time.sleep(_GRACE_MS / 1000.0)
        _rx_queue_drained._deadline_passed = True  # type: ignore[attr-defined]
    t = threading.Thread(target=_mark, daemon=True)
    tx_done_watcher = threading.Thread(
        target=lambda: (tx_done.wait(), _mark()),
        daemon=True,
    )
    tx_done_watcher.start()

_start_grace_timer()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

t_tx = threading.Thread(target=tx_thread, name="TX", daemon=True)
t_rx = threading.Thread(target=rx_thread, name="RX", daemon=True)

t_rx.start()
t_tx.start()

t_tx.join()
t_rx.join(timeout=args.packets * args.interval / 1000.0 + 5.0)

stream.stop()
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

print(f"Transmitted : {n_tx} packets")
print(f"RX captures : {n_rx}  (raw, incl. duplicates)")
print(f"Decoded OK  : {n_unique} unique valid packets")
print(f"Dropped     : {n_dropped} packets")
print(f"Drop rate   : {drop_rate * 100:.1f}%")
if missing_seqs:
    print(f"Missing seq : {missing_seqs}")
else:
    print("Missing seq : none")
print("=" * 50)

sys.exit(0 if n_dropped == 0 else 1)
