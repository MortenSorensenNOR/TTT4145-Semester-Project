"""Replay rx_buffs through the RX pipeline and report decode counts + timing.

Used as a smoke test to verify real-world buffer compatibility after
optimization changes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.pipeline import PipelineConfig, RXPipeline


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    cfg = PipelineConfig()
    rx = RXPipeline(cfg)

    rx_buf_dir = Path(__file__).resolve().parents[1] / "pluto" / "rx_buffs"
    files = sorted(rx_buf_dir.glob("rxbuf_*.npz"))
    if not files:
        print("No rx_buffs found.")
        return 1

    print(f"Loaded {len(files)} rx_buffs from {rx_buf_dir}")

    buffers = []
    for f in files:
        d = np.load(f)
        # buffers may have varying keys — try common ones
        if "samples" in d:
            buf = d["samples"]
        else:
            buf = d[d.files[0]]
        buffers.append(buf.astype(np.complex64))

    # Warm up
    total_packets = 0
    valid_packets = 0
    for buf in buffers:
        pkts, _ = rx.receive(buf)
        total_packets += len(pkts)
        valid_packets += sum(1 for p in pkts if p.valid)

    print(f"Warm pass: {valid_packets}/{total_packets} valid packets across {len(buffers)} buffers")

    # Timed runs
    t0 = time.perf_counter()
    total_pkts = 0
    valid_pkts = 0
    for _ in range(args.iters):
        for buf in buffers:
            pkts, _ = rx.receive(buf)
            total_pkts += len(pkts)
            valid_pkts += sum(1 for p in pkts if p.valid)
    elapsed = time.perf_counter() - t0
    avg_per_buf_ms = elapsed / (args.iters * len(buffers)) * 1000.0

    print(f"\n{args.iters} iters × {len(buffers)} buffers in {elapsed*1000:.1f} ms")
    print(f"  avg {avg_per_buf_ms:.2f} ms / buffer")
    print(f"  total packets: {total_pkts} ({valid_pkts} valid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
