"""Pipeline stage benchmarking — run this on the Pluto to find bottlenecks.

Usage (on Pluto or locally):
    python pluto/benchmark.py [--payload BYTES] [--hardware-rrc]

Each stage is timed in isolation with realistic data sizes.  An end-to-end
receive() benchmark is also included, using a 2-buffer sliding window that
matches the actual loopback_threaded.py conditions.

Results are printed as mean ± std over N_REPS repetitions, sorted slowest first.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Make sure modules/ is importable when running from repo root or from pluto/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from modules.frame_sync.frame_sync import (
    SynchronizerConfig,
    build_long_ref,
    build_fine_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.pulse_shaping.pulse_shaping import decimate, match_filter, rrc_filter, upsample
from modules.modulators import BPSK, QPSK, PSK8
from modules.frame_constructor.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.costas_loop.costas import CostasConfig, apply_costas_loop

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--payload",      type=int,  default=4095, help="Payload bytes (default: 1500)")
parser.add_argument("--hardware-rrc", action="store_true",     help="Skip software RRC (matches hardware_rrc=True loopback)")
args = parser.parse_args()

PAYLOAD_BYTES = args.payload
HARDWARE_RRC  = args.hardware_rrc

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N_REPS = 100   # repetitions per stage (increase for more stable results)

# ---------------------------------------------------------------------------
# Build shared fixtures once (not included in timing)
# ---------------------------------------------------------------------------

cfg      = PipelineConfig(hardware_rrc=HARDWARE_RRC)
tx_pipe  = TXPipeline(cfg)
rx_pipe  = RXPipeline(cfg)

num_taps  = 2 * cfg.SPS * cfg.SPAN + 1
rrc_taps  = rrc_filter(cfg.SPS, cfg.RRC_ALPHA, num_taps)
sync_cfg  = cfg.SYNC_CONFIG
costas_cfg = cfg.COSTAS_CONFIG

# Build a full TX frame so RX stages have a realistic input
fc = FrameConstructor()
header = FrameHeader(
    length=PAYLOAD_BYTES,
    src=0, dst=1, frame_type=0,
    mod_scheme=cfg.MOD_SCHEME,
    sequence_number=0,
)
payload_input_bits = np.random.randint(0, 2, PAYLOAD_BYTES * 8, dtype=np.uint8)
header_bits, payload_bits = fc.encode(header, payload_input_bits)

bpsk = BPSK()
qpsk = QPSK()

guard_syms    = np.zeros(cfg.GUARD_SYMS_LENGTH, dtype=np.complex64)
preamble_syms = generate_preamble(sync_cfg)
header_syms   = bpsk.bits2symbols(header_bits)
payload_syms  = qpsk.bits2symbols(payload_bits)
tx_syms = np.concatenate([guard_syms, preamble_syms, header_syms, payload_syms, guard_syms])

long_ref     = build_long_ref(sync_cfg, cfg.SPS, rrc_taps)
ref_f        = build_fine_ref(long_ref, sync_cfg, cfg.SPS)
long_ref_dec = decimate(long_ref, cfg.SPS)
ref_f_dec    = build_fine_ref(long_ref_dec, sync_cfg, 1)

# Full upsampled TX signal (used as RX input)
tx_signal    = upsample(tx_syms, cfg.SPS, rrc_taps)
rx_buffer    = tx_signal.copy()

# Match-filter; both coarse and fine sync run post-RRC
filtered_buffer  = match_filter(rx_buffer, rrc_taps)
decimated_buffer = decimate(filtered_buffer, cfg.SPS)
fs_sym = cfg.SAMPLE_RATE // cfg.SPS

# Pre-compute coarse/fine results so downstream benchmarks have valid inputs
coarse         = coarse_sync(decimated_buffer, fs_sym, 1, sync_cfg)
d_hats_samples = coarse.d_hats * cfg.SPS
fine           = fine_timing(filtered_buffer, long_ref, d_hats_samples, coarse.cfo_hats,
                             cfg.SAMPLE_RATE, cfg.SPS, sync_cfg, ref_f)

# Symbol buffers for Costas loop
header_end_sym  = 2 * fc.header_config.header_total_size
header_raw_syms = decimate(filtered_buffer[:header_end_sym * cfg.SPS], cfg.SPS)

payload_syms_len = (PAYLOAD_BYTES * 8) // 2  # QPSK = 2 bits/symbol
payload_raw_syms = decimate(
    filtered_buffer[header_end_sym * cfg.SPS: (header_end_sym + payload_syms_len) * cfg.SPS],
    cfg.SPS,
)

# ---------------------------------------------------------------------------
# 2-buffer sliding window for end-to-end receive() benchmark
# (replicates the loopback_threaded.py RX conditions)
# ---------------------------------------------------------------------------

# Build a multi-frame TX signal to fill 2 × rx_buf_size (≥ 2 frames)
frame_len   = len(tx_signal)
rx_buf_size = 2 * int(2 ** np.ceil(np.log2(frame_len)))
frames_needed = int(np.ceil(2 * rx_buf_size / frame_len)) + 1

rng = np.random.default_rng(0)
multi_frame_chunks = []
for i in range(frames_needed):
    pbits = rng.integers(0, 2, PAYLOAD_BYTES * 8, dtype=np.uint8)
    pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=i,
                 length=PAYLOAD_BYTES, payload=pbits)
    multi_frame_chunks.append(tx_pipe.transmit(pkt))

multi_frame_signal = np.concatenate(multi_frame_chunks)

# Slice out exactly a 2-buffer window that starts mid-signal so it contains
# at least one complete frame (avoid leading-edge ambiguity).
_start  = frame_len // 2
window  = multi_frame_signal[_start: _start + 2 * rx_buf_size]

print(f"\nBenchmarking pipeline stages ({N_REPS} reps, payload={PAYLOAD_BYTES} bytes, hardware_rrc={HARDWARE_RRC})")
print(f"  TX symbols: {len(tx_syms)}  |  TX samples: {len(tx_signal)}  ({frame_len / cfg.SAMPLE_RATE * 1e3:.1f} ms/frame)")
print(f"  RX buf: {rx_buf_size} samples ({rx_buf_size / cfg.SAMPLE_RATE * 1e3:.1f} ms)  |  2-buf window: {len(window)} samples")
print(f"  Header syms: {header_end_sym}  |  Payload syms: {payload_syms_len}")
print(f"  Frames in window: ~{frames_needed} built, window covers ~{2*rx_buf_size/frame_len:.1f} frames\n")

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(name: str, fn) -> tuple[str, float, float]:
    """Run fn() N_REPS times, return (name, mean_ms, std_ms)."""
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3
    return name, float(np.mean(arr)), float(np.std(arr))

# ---------------------------------------------------------------------------
# Benchmark each stage
# ---------------------------------------------------------------------------

results = []

if not HARDWARE_RRC:
    # 1. RRC upsample (TX pulse shaping) — skipped when hardware does it
    results.append(bench(
        "TX: RRC upsample",
        lambda: upsample(tx_syms, cfg.SPS, rrc_taps),
    ))

    # 2. Match filter (RX, full buffer) — skipped when hardware does it
    results.append(bench(
        "RX: match_filter (full buffer)",
        lambda: match_filter(rx_buffer, rrc_taps),
    ))

# 3. Coarse sync — Schmidl-Cox (decimated buffer, symbol rate)
results.append(bench(
    "RX: coarse_sync (Schmidl-Cox, decimated)",
    lambda: coarse_sync(decimated_buffer, fs_sym, 1, sync_cfg),
))

# 4. Fine timing — FFT cross-correlation
results.append(bench(
    "RX: fine_timing (FFT xcorr)",
    lambda: fine_timing(filtered_buffer, long_ref, d_hats_samples, coarse.cfo_hats,
                        cfg.SAMPLE_RATE, cfg.SPS, sync_cfg, ref_f),
))

# 5. Costas loop — header (BPSK)
results.append(bench(
    "RX: costas_loop header (BPSK)",
    lambda: apply_costas_loop(header_raw_syms[:header_end_sym], costas_cfg,
                              ModulationSchemes.BPSK, 0.0, 0.0),
))

# 6. Costas loop — payload (QPSK)
results.append(bench(
    "RX: costas_loop payload (QPSK)",
    lambda: apply_costas_loop(payload_raw_syms[:payload_syms_len], costas_cfg,
                              ModulationSchemes.QPSK, 0.0, 0.0),
))

# 7. Decimate (symbol timing)
results.append(bench(
    "RX: decimate",
    lambda: decimate(filtered_buffer[:payload_syms_len * cfg.SPS], cfg.SPS),
))

# 8. QPSK demodulation
results.append(bench(
    "RX: QPSK symbols2bits",
    lambda: qpsk.symbols2bits(payload_raw_syms[:payload_syms_len]),
))

# 9. BPSK demodulation
results.append(bench(
    "RX: BPSK symbols2bits (header)",
    lambda: bpsk.symbols2bits(header_raw_syms[:header_end_sym]),
))

# 10. Frame header decode (Golay + CRC)
results.append(bench(
    "RX: frame_constructor.decode_header",
    lambda: fc.decode_header(bpsk.symbols2bits(header_raw_syms[:header_end_sym])),
))

# 11. End-to-end receive() on a 2-buffer window (matches loopback conditions)
results.append(bench(
    "RX: receive() end-to-end (2-buf window)",
    lambda: rx_pipe.receive(window),
))

# 12. match_filter on the FULL 2-buffer window (realistic receive size)
results.append(bench(
    "RX: match_filter (full 2-buf window)",
    lambda: match_filter(window, rrc_taps),
))

# 13. detect() on a pre-filtered full window
_filtered_window = match_filter(window, rrc_taps)
results.append(bench(
    "RX: detect() on full window (post match_filter)",
    lambda: rx_pipe.detect(_filtered_window),
))

# ---------------------------------------------------------------------------
# Print results sorted by mean time (slowest first)
# ---------------------------------------------------------------------------

results.sort(key=lambda r: r[1], reverse=True)
total_ms = sum(r[1] for r in results)

col = 44
print(f"{'Stage':<{col}} {'Mean (ms)':>10}  {'Std (ms)':>10}  {'% total':>8}")
print("-" * (col + 36))
for name, mean, std in results:
    pct = 100 * mean / total_ms if total_ms > 0 else 0
    print(f"{name:<{col}} {mean:>10.3f}  {std:>10.3f}  {pct:>7.1f}%")
print("-" * (col + 36))
print(f"{'TOTAL (summed)':<{col}} {total_ms:>10.3f}")
print()
