"""Pipeline stage benchmarking — run this on the Pluto to find bottlenecks.

Usage (on Pluto or locally):
    python pluto/benchmark.py

Each stage is timed in isolation with realistic data sizes matching the default
PipelineConfig. Results are printed as mean ± std over N_REPS repetitions,
sorted from slowest to fastest.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Make sure modules/ is importable when running from repo root or from pluto/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.pipeline import PipelineConfig
from modules.frame_sync import (
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.pulse_shaping import decimate, match_filter, rrc_filter, upsample
from modules.modulators import BPSK, QPSK, PSK8
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.costas_loop.costas import CostasConfig, apply_costas_loop

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N_REPS = 10          # repetitions per stage (increase for more stable results)
PAYLOAD_BYTES = 100  # realistic payload size

# ---------------------------------------------------------------------------
# Build shared fixtures once (not included in timing)
# ---------------------------------------------------------------------------

cfg = PipelineConfig()
num_taps = 2 * cfg.SPS * cfg.SPAN + 1
rrc_taps = rrc_filter(cfg.SPS, cfg.RRC_ALPHA, num_taps)
sync_cfg = cfg.SYNC_CONFIG
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

guard_syms   = np.zeros(500, dtype=np.complex64)
preamble_syms = generate_preamble(sync_cfg)
header_syms  = bpsk.bits2symbols(header_bits)
payload_syms = qpsk.bits2symbols(payload_bits)
tx_syms = np.concatenate([guard_syms, preamble_syms, header_syms, payload_syms, guard_syms])

long_ref = build_long_ref(sync_cfg, cfg.SPS, rrc_taps)

# Full upsampled TX signal (used as RX input)
tx_signal = upsample(tx_syms, cfg.SPS, rrc_taps)
rx_buffer = tx_signal.copy()

# Match-filter first; both coarse and fine sync run post-RRC
filtered_buffer = match_filter(rx_buffer, rrc_taps)

# Pre-compute coarse result so fine_timing has a valid input
coarse = coarse_sync(filtered_buffer, cfg.SAMPLE_RATE, cfg.SPS, sync_cfg)
fine   = fine_timing(filtered_buffer, long_ref, coarse.d_hats, coarse.cfo_hats,
                     cfg.SAMPLE_RATE, cfg.SPS, sync_cfg)

# Symbol buffers for Costas loop
header_end_sym = 2 * fc.header_config.header_total_size
header_raw_syms = decimate(filtered_buffer[:header_end_sym * cfg.SPS], cfg.SPS)

payload_syms_len = (PAYLOAD_BYTES * 8) // 2  # QPSK = 2 bits/symbol
payload_raw_syms = decimate(
    filtered_buffer[header_end_sym * cfg.SPS: (header_end_sym + payload_syms_len) * cfg.SPS],
    cfg.SPS,
)

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
    arr = np.array(times) * 1e3  # convert to ms
    return name, float(np.mean(arr)), float(np.std(arr))

# ---------------------------------------------------------------------------
# Benchmark each stage
# ---------------------------------------------------------------------------

print(f"\nBenchmarking pipeline stages ({N_REPS} reps, payload={PAYLOAD_BYTES} bytes)")
print(f"  TX symbols: {len(tx_syms)}  |  TX samples: {len(tx_signal)}")
print(f"  Header syms: {header_end_sym}  |  Payload syms: {payload_syms_len}\n")

results = []

# 1. RRC upsample (TX pulse shaping)
results.append(bench(
    "TX: RRC upsample",
    lambda: upsample(tx_syms, cfg.SPS, rrc_taps),
))

# 2. Match filter (RX, full buffer)
results.append(bench(
    "RX: match_filter (full buffer)",
    lambda: match_filter(rx_buffer, rrc_taps),
))

# 3. Coarse sync — Schmidl-Cox (full buffer scan, post-RRC)
results.append(bench(
    "RX: coarse_sync (Schmidl-Cox)",
    lambda: coarse_sync(filtered_buffer, cfg.SAMPLE_RATE, cfg.SPS, sync_cfg),
))

# 4. Fine timing — FFT cross-correlation (post-RRC)
results.append(bench(
    "RX: fine_timing (FFT xcorr)",
    lambda: fine_timing(filtered_buffer, long_ref, coarse.d_hats, coarse.cfo_hats,
                        cfg.SAMPLE_RATE, cfg.SPS, sync_cfg),
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

# 7. Decimate (symbol timing, no interpolation)
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

# ---------------------------------------------------------------------------
# Print results sorted by mean time (slowest first)
# ---------------------------------------------------------------------------

results.sort(key=lambda r: r[1], reverse=True)
total_ms = sum(r[1] for r in results)

col = 40
print(f"{'Stage':<{col}} {'Mean (ms)':>10}  {'Std (ms)':>10}  {'% total':>8}")
print("-" * (col + 36))
for name, mean, std in results:
    pct = 100 * mean / total_ms if total_ms > 0 else 0
    print(f"{name:<{col}} {mean:>10.3f}  {std:>10.3f}  {pct:>7.1f}%")
print("-" * (col + 36))
print(f"{'TOTAL (summed)':<{col}} {total_ms:>10.3f}")
print()
