"""Cross-evaluate sync parameter configs across PSK8 and PSK16 capture sets.

Each config is run on every buffer (no train/val split — we want generalisation
across all captured data, not held-out search), and EVM + decode counts are
reported per (config, dataset) pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sweep_sync_params import _build_pipeline, load_buffers, score_buffers
from modules.pipeline import PipelineConfig, RXPipeline


# (name, bn, ζ_costas, BnTs, ζ_nda, L)
CONFIGS = [
    ("baseline (current PipelineConfig)",
        # Read from PipelineConfig() so this stays in sync with the source.
        None, None, None, None, None),
    ("PSK8  best, free ζ",   0.01507, 1.691, 0.004991, 2.833, 27),
    ("PSK16 best, free ζ",   0.01133, 1.638, 0.005883, 2.877, 24),
    ("PSK8  best, ζ=1/√2",   0.01593, 0.7071, 0.006489, 0.7071, 19),
    ("PSK16 best, ζ=1/√2",   0.01776, 0.7071, 0.005575, 0.7071, 16),
]


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "pluto" / "rx_buffs"
    bufs = {
        "psk8":  load_buffers(root / "psk8"),
        "psk16": load_buffers(root / "psk16"),
    }
    print(f"PSK8: {len(bufs['psk8'])} buffers   PSK16: {len(bufs['psk16'])} buffers\n")

    print(f"{'config':<32}  {'set':<6}  {'valid':>6}  {'detected':>8}  "
          f"{'matched':>7}  {'expected':>8}  {'EVM%':>6}")
    print("-" * 85)
    for name, bn, zc, bnts, zn, L in CONFIGS:
        if bn is None:
            rx_factory = lambda: RXPipeline(PipelineConfig())
        else:
            rx_factory = lambda bn=bn, zc=zc, bnts=bnts, zn=zn, L=L: \
                _build_pipeline(bn, zc, bnts, zn, L)
        for ds in ("psk8", "psk16"):
            rx = rx_factory()
            s = score_buffers(rx, bufs[ds])
            print(f"{name:<32}  {ds:<6}  {s.valid:>6d}  {s.detected:>8d}  "
                  f"{s.matched:>7d}  {s.expected:>8d}  {s.evm_pct:>6.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
