"""Profile RX/TX pipeline using cProfile + manual timings.

Runs end-to-end TX and RX (decode) and reports where time is spent.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline
from modules.pulse_shaping.pulse_shaping import match_filter
from modules.frame_sync.frame_sync import coarse_sync, fine_timing
from modules.gardner_ted.gardner import apply_gardner_ted
from modules.costas_loop.costas import apply_costas_loop
from modules.frame_constructor.frame_constructor import ModulationSchemes


def make_rx_buffer(tx_pipe: TXPipeline, payloads: list[bytes], silence_len: int = 256):
    chunks = []
    for i, payload in enumerate(payloads):
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=i,
                     length=len(payload), payload=bits)
        s = tx_pipe.transmit(pkt)
        peak = float(np.max(np.abs(s)))
        if peak > 0:
            s = s / peak
        chunks.append(s.astype(np.complex64))
    silence = np.zeros(silence_len, dtype=np.complex64)
    return np.concatenate([silence] + [c for chunk in chunks for c in (chunk, silence)])


def time_it(fn, n_iter: int):
    # warm up
    fn()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) / n_iter


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--packets", type=int, default=4)
    p.add_argument("--payload", type=int, default=1500)
    p.add_argument("--iters",   type=int, default=10)
    args = p.parse_args()

    cfg = PipelineConfig()
    tx = TXPipeline(cfg)
    rx = RXPipeline(cfg)

    rng = np.random.default_rng(0xCAFE)
    payloads = [bytes(rng.integers(0, 256, args.payload, dtype=np.uint8))
                for _ in range(args.packets)]

    # === TX timing ===
    bits_lists = [np.unpackbits(np.frombuffer(p, dtype=np.uint8)) for p in payloads]
    pkts = [Packet(src_mac=0, dst_mac=1, type=0, seq_num=i,
                   length=len(payloads[i]), payload=bits_lists[i])
            for i in range(args.packets)]

    def tx_one():
        for pkt in pkts:
            tx.transmit(pkt)

    tx_per_call_ms = time_it(tx_one, args.iters) * 1000.0 / args.packets
    print(f"\nTX: {tx_per_call_ms:.3f} ms / packet ({args.payload} bytes)")

    # === RX timing on built-in buffer ===
    raw = make_rx_buffer(tx, payloads)
    print(f"RX buffer size: {len(raw)} samples ({len(raw)/cfg.SAMPLE_RATE*1000:.1f} ms air time)")

    # warm
    pkts_decoded, _ = rx.receive(raw)
    valid = sum(1 for p in pkts_decoded if p.valid)
    print(f"  decoded {valid}/{args.packets} packets")

    def rx_one():
        rx.receive(raw)

    rx_total_ms = time_it(rx_one, args.iters) * 1000.0
    rx_per_pkt_ms = rx_total_ms / args.packets
    print(f"RX: {rx_total_ms:.3f} ms / call ({rx_per_pkt_ms:.3f} ms / packet)")

    # === Detailed breakdown ===
    print("\n--- RX breakdown ---")
    # Match filter
    def mf():
        match_filter(raw, rx.rrc_taps)
    mf_ms = time_it(mf, args.iters) * 1000.0
    print(f"match_filter: {mf_ms:.3f} ms")

    fb = match_filter(raw, rx.rrc_taps)

    # Detection
    def det():
        rx.detect(fb)
    det_ms = time_it(det, args.iters) * 1000.0
    print(f"detect:       {det_ms:.3f} ms")

    detections = rx.detect(fb)
    print(f"  detections found: {len(detections)}")

    # Decode per detection
    def dec_all():
        for d in detections:
            try:
                rx.decode(fb[d.payload_start:], d.cfo_estimate, d.phase_estimate)
            except Exception:
                pass
    dec_ms = time_it(dec_all, args.iters) * 1000.0
    dec_per_pkt_ms = dec_ms / max(1, len(detections))
    print(f"decode (all): {dec_ms:.3f} ms ({dec_per_pkt_ms:.3f} ms/pkt)")

    # Gardner & Costas
    if detections:
        d0 = detections[0]
        sample_buf = fb[d0.payload_start: d0.payload_start + cfg.SPS * (1500 * 8 // 3 + 1000)]
        cfo_rad = np.float32(2 * np.pi * float(d0.cfo_estimate) / cfg.SAMPLE_RATE * cfg.SPS)

        def gardner_only():
            sps = cfg.SPS
            guard = sps * sps
            buf = sample_buf
            gin = np.concatenate([buf[:1], buf])
            apply_gardner_ted(gin, sps, BnTs=cfg.GARDNER_BN_TS,
                              zeta=cfg.GARDNER_ZETA, L=cfg.GARDNER_L)
        g_ms = time_it(gardner_only, args.iters) * 1000.0
        print(f"  gardner (per pkt body): {g_ms:.3f} ms  (input={len(sample_buf)} samples)")

        # Pre-Gardner -> get rx_syms then Costas
        gin = np.concatenate([sample_buf[:1], sample_buf])
        rx_syms = apply_gardner_ted(gin, cfg.SPS, BnTs=cfg.GARDNER_BN_TS,
                                    zeta=cfg.GARDNER_ZETA, L=cfg.GARDNER_L)
        def costas_only():
            apply_costas_loop(rx_syms, cfg.COSTAS_CONFIG, ModulationSchemes.PSK8,
                              current_phase_estimate=np.float32(0.0),
                              current_frequency_offset=cfo_rad)
        c_ms = time_it(costas_only, args.iters) * 1000.0
        print(f"  costas  (per pkt body): {c_ms:.3f} ms  (input={len(rx_syms)} symbols)")

    # === cProfile session ===
    print("\n--- cProfile (RX only) ---")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(args.iters):
        rx.receive(raw)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumulative").print_stats(30)
    print(s.getvalue())

    return 0


if __name__ == "__main__":
    sys.exit(main())
