"""Quick smoke-test for modules.parallel_pipeline (no SDR needed).

Builds a few packets via the TX pool, concatenates the resulting samples into
a single fake "RX buffer", runs match_filter + detect on the parent thread,
and decodes them back via the RX pool. Verifies the decoded payloads round-
trip cleanly.

Run with:
    .venv/bin/python scripts/smoke_parallel_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.parallel_pipeline import RXWorkerPool, TXWorkerPool
from modules.pipeline import PipelineConfig, RXPipeline
from modules.pulse_shaping.pulse_shaping import match_filter


N_PACKETS = 6
PAYLOAD_SIZE = 200      # bytes — small so the test is fast
WORKERS      = 2
START        = "spawn"


def main() -> int:
    pipe_cfg = PipelineConfig()
    pipe_cfg.SYNC_CONFIG.fine_peak_ratio_min = np.float32(7.0)
    rx_pipe = RXPipeline(pipe_cfg)

    rng = np.random.default_rng(0xBEEF)
    payloads: list[bytes] = []
    for _ in range(N_PACKETS):
        payloads.append(bytes(rng.integers(0, 256, PAYLOAD_SIZE, dtype=np.uint8)))

    print(f"Spinning up TX pool ({WORKERS} workers, start={START}) …")
    t0 = time.perf_counter()
    tx_pool = TXWorkerPool(pipe_cfg, n_workers=WORKERS, start_method=START)
    print(f"  ready in {time.perf_counter() - t0:.2f}s")

    print(f"Building {N_PACKETS} packets via TX pool …")
    t0 = time.perf_counter()
    pending = [tx_pool.submit(p, 0, 1, 0, i, 16384.0)
               for i, p in enumerate(payloads)]
    sample_chunks = [ar.get(timeout=15.0) for ar in pending]
    print(f"  done in {time.perf_counter() - t0:.2f}s; "
          f"frame_len={len(sample_chunks[0])} samples")

    # Insert silent gaps between packets so frame_sync sees clean packet
    # boundaries (mirrors how TxStream lays them out on air).
    silence = np.zeros(2048, dtype=np.complex64)
    raw = np.concatenate([silence] + [c for chunk in sample_chunks
                                      for c in (chunk, silence)])
    # Normalize back into the same scale the receiver expects.
    raw = raw.astype(np.complex64) / 16384.0

    print(f"Spinning up RX pool ({WORKERS} workers, start={START}) …")
    t0 = time.perf_counter()
    slot_samples = len(raw) + 4096
    rx_pool = RXWorkerPool(pipe_cfg, n_workers=WORKERS,
                           slot_samples=slot_samples,
                           n_slots=2, start_method=START)
    print(f"  ready in {time.perf_counter() - t0:.2f}s")

    print("Running match_filter + detect on parent …")
    t0 = time.perf_counter()
    filtered   = match_filter(raw, rx_pipe.rrc_taps)
    detections = rx_pipe.detect(filtered)
    print(f"  detected {len(detections)} packets in {time.perf_counter() - t0:.2f}s")

    if len(detections) != N_PACKETS:
        print(f"WARN: expected {N_PACKETS} detections, got {len(detections)}")

    print("Submitting decodes to RX pool …")
    t0 = time.perf_counter()
    sub = rx_pool.submit_buffer(filtered, detections, search_from_abs=0)
    results = [ar.get(timeout=15.0) for ar in sub.futures]
    print(f"  decoded {len(results)} in {time.perf_counter() - t0:.2f}s")

    # Verify each successful decode matches one of our payloads.
    ok = bad = 0
    decoded_payloads: set[bytes] = set()
    for result in results:
        if result.status != "ok":
            print(f"  result status={result.status!r}: {result.err}")
            bad += 1
            continue
        if not result.valid:
            print("  result.valid=False — header CRC failed")
            bad += 1
            continue
        decoded_payloads.add(result.payload_bytes[:result.length])
        ok += 1

    matched = sum(1 for p in payloads if p in decoded_payloads)
    print(f"Round-trip: {matched}/{N_PACKETS} payloads matched, ok={ok}, bad={bad}")

    tx_pool.shutdown()
    rx_pool.shutdown()

    return 0 if matched == N_PACKETS else 1


if __name__ == "__main__":
    sys.exit(main())
