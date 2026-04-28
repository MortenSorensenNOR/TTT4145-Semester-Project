"""Profile TX pipeline: where does the per-packet time go?"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.pipeline import Packet, PipelineConfig, TXPipeline


def main() -> int:
    cfg = PipelineConfig()
    tx = TXPipeline(cfg)

    rng = np.random.default_rng(0xCAFE)
    payload = bytes(rng.integers(0, 256, 1500, dtype=np.uint8))
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0,
                 length=1500, payload=bits)

    # Warm up
    for _ in range(20):
        tx.transmit(pkt)

    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        tx.transmit(pkt)
    elapsed = (time.perf_counter() - t0) / n_iter * 1000
    print(f"TX: {elapsed:.3f} ms / packet")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        tx.transmit(pkt)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(25)
    print(s.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
