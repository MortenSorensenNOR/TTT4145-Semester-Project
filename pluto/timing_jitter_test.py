"""Timing jitter hypothesis test.

Tests whether fine-sync on the decimated buffer misses the optimal sub-symbol
sampling phase in loopback, causing intermittent decode failures.

For each frame detection the pipeline normally uses payload_start = (sym_idx + N) * SPS,
which is always a multiple of SPS (phase=0).  This script also tries offsets 1, 2, 3
and reports which offset(s) actually decode — if offset 0 never works but offset 2
always does, the hypothesis is confirmed.

Runs the SDR loopback N_RUNS times, each preceded by a random sleep to randomise
the transmit/receive timing alignment, then summarises which sub-sample offsets
succeeded across all detections.

Usage:
    uv run python pluto/timing_jitter_test.py [--gain G] [--runs N] [--payload B]
"""

import argparse
import queue
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet, DetectionResult
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pulse_shaping.pulse_shaping import match_filter, decimate
from pluto.config import DAC_SCALE, configure_rx, configure_tx
from pluto.sdr_stream import RxStream

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",    type=float, default=-20,          help="TX gain dB (default: -20)")
parser.add_argument("--runs",    type=int,   default=10,           help="Number of loopback runs (default: 10)")
parser.add_argument("--payload", type=int,   default=100,          help="Payload bytes (default: 100)")
parser.add_argument("--ip",      type=str,   default="192.168.2.1",help="PlutoSDR IP")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

pipe_cfg = PipelineConfig(hardware_rrc=False)
tx_pipe  = TXPipeline(pipe_cfg)
rx_pipe  = RXPipeline(pipe_cfg)
sps      = pipe_cfg.SPS

rng = np.random.default_rng(42)

# Pre-build a single probe packet to measure frame length
_probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
_probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len      = len(_probe_samples)
rx_buf_size    = 2 * int(2 ** np.ceil(np.log2(frame_len)))

print(f"Frame len : {frame_len} samples  ({frame_len/pipe_cfg.SAMPLE_RATE*1e3:.1f} ms)")
print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size/pipe_cfg.SAMPLE_RATE*1e3:.1f} ms)")
print(f"Runs      : {args.runs}")
print(f"Gain      : {args.gain} dB")
print()

# ---------------------------------------------------------------------------
# Offset-probing decoder
# ---------------------------------------------------------------------------

def try_all_offsets(filtered_buffer: np.ndarray, det: DetectionResult) -> dict:
    """Try all SPS sub-sample offsets for a detection and return which ones decode.

    Returns dict: offset -> {'valid': bool, 'err': str or None}
    """
    results = {}
    cfo_rad_per_symbol = 2 * np.pi * det.cfo_estimate / pipe_cfg.SAMPLE_RATE * sps
    for offset in range(sps):
        # Shift payload_start by `offset` samples (stays within same symbol-rate grid
        # but probes the adjacent sub-symbol phases)
        shifted_start = det.payload_start + offset
        buf = filtered_buffer[shifted_start:]
        try:
            pkt = rx_pipe.decode(buf, det.cfo_estimate, det.phase_estimate)
            results[offset] = {'valid': pkt.valid, 'err': None}
        except Exception as e:
            results[offset] = {'valid': False, 'err': str(e)[:60]}
    return results


def receive_with_offset_probe(raw: np.ndarray, search_from: int = 0):
    """Like rx_pipe.receive() but also tries all 4 offsets for each detection."""
    search_buf      = raw[search_from:]
    filtered_buffer = match_filter(search_buf, rx_pipe.rrc_taps)
    detections      = rx_pipe.detect(filtered_buffer)
    if not detections:
        return [], []

    normal_packets = []
    offset_results = []

    for det in detections:
        buf = filtered_buffer[det.payload_start:]
        try:
            pkt = rx_pipe.decode(buf, det.cfo_estimate, det.phase_estimate)
            pkt.sample_start = search_from + det.payload_start
            normal_packets.append(pkt)
        except Exception as e:
            pkt = None

        offsets = try_all_offsets(filtered_buffer, det)
        offset_results.append({
            'det': det,
            'normal_ok': pkt is not None and pkt.valid,
            'offsets': offsets,
        })

    return normal_packets, offset_results

# ---------------------------------------------------------------------------
# SDR setup
# ---------------------------------------------------------------------------

sdr = adi.Pluto("ip:" + args.ip)
configure_tx(sdr, freq=pipe_cfg.CENTER_FREQ, gain=args.gain, cyclic=False, sample_rate=pipe_cfg.SAMPLE_RATE)
configure_rx(sdr, freq=pipe_cfg.CENTER_FREQ, gain_mode="slow_attack", sample_rate=pipe_cfg.SAMPLE_RATE, buffer_size=rx_buf_size)

_air_time_ms   = int(3 * frame_len / pipe_cfg.SAMPLE_RATE * 1000) + 1
_buf_ms        = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000) + 1
_BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 6

# ---------------------------------------------------------------------------
# Per-run capture
# ---------------------------------------------------------------------------

def run_one(run_idx: int, jitter_s: float) -> list[dict]:
    """Transmit one packet and collect RX buffers; probe all offsets on each detection."""
    print(f"  Run {run_idx:2d}: sleeping {jitter_s*1000:.0f} ms before TX ... ", end="", flush=True)
    time.sleep(jitter_s)

    # Pre-compute BEFORE starting the RX stream — pure-Python pipeline is slow
    # (50-200ms), and if computation runs after stream.start() the queue fills
    # with stale pre-TX buffers that we'd then read instead of the signal.
    chunks = []
    for idx in range(3):
        payload_bits = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
        pkt          = Packet(src_mac=0, dst_mac=1, type=0, seq_num=run_idx * 3 + idx, length=args.payload, payload=payload_bits)
        s            = tx_pipe.transmit(pkt)
        peak         = np.max(np.abs(s))
        if peak > 0:
            s = s / peak
        chunks.append((s * DAC_SCALE).astype(np.complex64))
    tx_samples = np.concatenate(chunks)

    # Start fresh RX stream, flush stale DMA buffers — then TX immediately
    stream = RxStream(sdr, maxsize=64, lossless=True)
    stream.start(flush=4)

    # Transmit
    sdr.tx(tx_samples)

    # Collect buffers
    collected_bufs = []
    n_after = 0
    while n_after < _BUFS_AFTER_TX:
        try:
            buf = stream.get(timeout=0.1)
            collected_bufs.append(buf)
            n_after += 1
        except queue.Empty:
            break

    stream.stop()
    sdr.tx_destroy_buffer()

    # Report signal energy per buffer
    amps = [float(np.max(np.abs(b))) for b in collected_bufs]
    noise_floor = np.median(amps)
    signal_present = any(a > noise_floor * 3 for a in amps)
    print(f"buf amps: {[f'{a:.3f}' for a in amps[:8]]}  signal={'YES' if signal_present else 'NOISE ONLY'}")

    # Probe with sliding window
    all_offset_results = []
    prev_buf   = None
    for buf in collected_bufs:
        raw      = np.concatenate([prev_buf, buf]) if prev_buf is not None else buf
        prev_buf = buf
        _, offset_results = receive_with_offset_probe(raw)
        all_offset_results.extend(offset_results)

    # Summary for this run — only report unique detections (deduplicate by payload_start)
    seen_ps = set()
    unique_results = []
    for r in all_offset_results:
        ps = r['det'].payload_start
        if ps not in seen_ps:
            seen_ps.add(ps)
            unique_results.append(r)

    if unique_results:
        for r in unique_results:
            conf = r['det'].confidence
            offsets = r['offsets']
            ok_offsets   = [o for o, v in offsets.items() if v['valid']]
            fail_offsets = [o for o, v in offsets.items() if not v['valid']]
            normal_flag  = "OK" if r['normal_ok'] else "FAIL"
            err_samples  = {o: v['err'] for o, v in offsets.items() if v['err']}
            print(f"  conf={conf:.3f} normal={normal_flag} ok={ok_offsets} | errs={err_samples}")
    else:
        print("  no detections")

    all_offset_results = unique_results

    return all_offset_results

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

all_runs = []
for i in range(args.runs):
    jitter = random.uniform(0.02, 0.35)   # 20–350 ms jitter between runs
    results = run_one(i, jitter)
    all_runs.extend(results)

del sdr

# ---------------------------------------------------------------------------
# Summary across all runs
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("SUMMARY ACROSS ALL RUNS")
print("=" * 60)

total_detections = len(all_runs)
if total_detections == 0:
    print("No detections at all.")
    sys.exit(1)

offset_ok_counts   = {o: 0 for o in range(sps)}
offset_fail_counts = {o: 0 for o in range(sps)}
normal_ok = 0
normal_fail = 0

for r in all_runs:
    for o in range(sps):
        if r['offsets'][o]['valid']:
            offset_ok_counts[o] += 1
        else:
            offset_fail_counts[o] += 1
    if r['normal_ok']:
        normal_ok += 1
    else:
        normal_fail += 1

print(f"Total detections : {total_detections}")
print(f"Normal decode    : {normal_ok} OK / {normal_fail} FAIL  "
      f"({100*normal_ok/total_detections:.0f}%)")
print()
print("Decode success rate by sub-sample offset:")
for o in range(sps):
    ok  = offset_ok_counts[o]
    pct = 100 * ok / total_detections
    bar = "#" * int(pct / 2)
    print(f"  offset +{o}: {ok:3d}/{total_detections}  ({pct:5.1f}%)  {bar}")

print()
if normal_fail > 0 and any(offset_ok_counts[o] > normal_ok for o in range(1, sps)):
    best_offset = max(range(sps), key=lambda o: offset_ok_counts[o])
    print(f"HYPOTHESIS CONFIRMED: offset +{best_offset} decodes more than offset +0.")
    print(f"  offset +0 (current): {offset_ok_counts[0]}/{total_detections} = {100*offset_ok_counts[0]/total_detections:.0f}%")
    print(f"  offset +{best_offset} (best):   {offset_ok_counts[best_offset]}/{total_detections} = {100*offset_ok_counts[best_offset]/total_detections:.0f}%")
    print(f"  → fine timing on decimated buffer is losing sub-symbol phase information.")
else:
    print("No clear evidence that a different offset improves decoding.")
    print("Timing jitter hypothesis not supported by this data.")
