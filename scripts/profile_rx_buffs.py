"""Profile RX on real-world rx_buffs."""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from modules.pipeline import PipelineConfig, RXPipeline


def main() -> int:
    cfg = PipelineConfig()
    rx = RXPipeline(cfg)

    rx_buf_dir = Path(__file__).resolve().parents[1] / "pluto" / "rx_buffs"
    files = sorted(rx_buf_dir.glob("rxbuf_*.npz"))
    buffers = [np.load(f)["samples"].astype(np.complex64) for f in files]

    # warm
    for buf in buffers[:2]:
        rx.receive(buf)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        for buf in buffers:
            rx.receive(buf)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(40)
    print(s.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
