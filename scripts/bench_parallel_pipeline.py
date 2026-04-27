"""Quick perf comparison: inline pipelines vs TX/RX worker pools.

Builds a batch of MTU-sized packets, mashes them into a fake RX buffer, and
times decode. Repeats with --workers=0 (inline) vs --workers=N to show how
much throughput scales with parallel decoders.

Run with:
    .venv/bin/python scripts/bench_parallel_pipeline.py
    .venv/bin/python scripts/bench_parallel_pipeline.py --workers 4 --packets 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.parallel_pipeline import RXWorkerPool, TXWorkerPool
from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline
from modules.pulse_shaping.pulse_shaping import match_filter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--packets",  type=int, default=16)
    p.add_argument("--payload",  type=int, default=1500)
    p.add_argument("--mp-start", type=str, default="spawn",
                   choices=("spawn", "fork", "forkserver"))
    args = p.parse_args()

    pipe_cfg = PipelineConfig()
    pipe_cfg.SYNC_CONFIG.fine_peak_ratio_min = np.float32(7.0)
    rx_pipe = RXPipeline(pipe_cfg)
    tx_pipe = TXPipeline(pipe_cfg)

    rng = np.random.default_rng(0xCAFE)
    payloads = [bytes(rng.integers(0, 256, args.payload, dtype=np.uint8))
                for _ in range(args.packets)]

    # --- Build samples once via the inline pipeline (so the bench focuses
    # on decode cost) -------------------------------------------------------
    print(f"Building {args.packets} × {args.payload}-byte packets inline …")
    t0 = time.perf_counter()
    chunks = []
    for payload in payloads:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0,
                     length=args.payload, payload=bits)
        samples = tx_pipe.transmit(pkt)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        chunks.append(samples.astype(np.complex64))
    inline_build_s = time.perf_counter() - t0
    print(f"  inline build: {inline_build_s*1000:.1f} ms total "
          f"({inline_build_s/args.packets*1000:.1f} ms/pkt)")

    silence = np.zeros(2048, dtype=np.complex64)
    raw = np.concatenate([silence] + [c for chunk in chunks for c in (chunk, silence)])

    # --- Inline decode -----------------------------------------------------
    print("Inline decode …")
    t0 = time.perf_counter()
    packets, _ = rx_pipe.receive(raw)
    inline_decode_s = time.perf_counter() - t0
    inline_ok = sum(1 for pkt in packets if pkt.valid)
    print(f"  inline: decoded {inline_ok}/{args.packets} in "
          f"{inline_decode_s*1000:.1f} ms")

    # --- TX pool perf ------------------------------------------------------
    print(f"\nTX pool (workers={args.workers}, start={args.mp_start}) …")
    t0 = time.perf_counter()
    tx_pool = TXWorkerPool(pipe_cfg, n_workers=args.workers,
                           start_method=args.mp_start)
    print(f"  pool ready: {(time.perf_counter()-t0)*1000:.1f} ms")

    # Burn a warm-up round to absorb worker module/cache-warming costs.
    warm = [tx_pool.submit(payloads[0], 0, 1, 0, 0, 1.0)
            for _ in range(args.workers)]
    [a.get(timeout=30.0) for a in warm]

    t0 = time.perf_counter()
    pending = [tx_pool.submit(p, 0, 1, 0, i, 1.0)
               for i, p in enumerate(payloads)]
    _ = [a.get(timeout=30.0) for a in pending]
    pool_build_s = time.perf_counter() - t0
    print(f"  pool build: {pool_build_s*1000:.1f} ms total  "
          f"(speedup vs inline: {inline_build_s/pool_build_s:.2f}x)")
    tx_pool.shutdown()

    # --- RX pool perf ------------------------------------------------------
    print(f"\nRX pool (workers={args.workers}, start={args.mp_start}) …")
    t0 = time.perf_counter()
    slot_samples = len(raw) + 4096
    rx_pool = RXWorkerPool(pipe_cfg, n_workers=args.workers,
                           slot_samples=slot_samples,
                           n_slots=2, start_method=args.mp_start)
    print(f"  pool ready: {(time.perf_counter()-t0)*1000:.1f} ms")

    # Warm-up: do one filter+detect+decode pass so each worker hits its
    # decode cache + LDPC structures before timing.
    filtered = match_filter(raw, rx_pipe.rrc_taps)
    warmup_dets = rx_pipe.detect(filtered)
    sub = rx_pool.submit_buffer(filtered, warmup_dets, search_from_abs=0)
    [ar.get(timeout=30.0) for ar in sub.futures]

    t0 = time.perf_counter()
    detections = rx_pipe.detect(filtered)
    sub = rx_pool.submit_buffer(filtered, detections, search_from_abs=0)
    results = [ar.get(timeout=30.0) for ar in sub.futures]
    pool_decode_s = time.perf_counter() - t0
    pool_ok = sum(1 for r in results if r.status == "ok" and r.valid)
    print(f"  pool decode: decoded {pool_ok}/{args.packets} in "
          f"{pool_decode_s*1000:.1f} ms  "
          f"(speedup vs inline: {inline_decode_s/pool_decode_s:.2f}x)")

    rx_pool.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
