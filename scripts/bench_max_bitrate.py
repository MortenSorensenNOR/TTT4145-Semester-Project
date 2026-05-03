"""Benchmark the maximum sustainable sample rate of the TX/RX pipeline.

Answers: assuming the Pluto can keep up, how high could the sample rate go
before the PC starts running behind, and what does that translate to in
useful payload bits/s?

Method:
  - TX side: time TXPipeline.transmit() over many max-sized packets.  Yields
    samples/s the encoder can emit on a single thread.
  - RX side: build a back-to-back buffer of N frames (no inter-packet gap
    beyond the existing guard symbols), time RXPipeline.receive() on it.
    Yields samples/s the decoder can drain on a single thread.
  - Bottleneck = min(TX, RX).  Useful bitrate = bottleneck × payload_bits /
    samples_per_frame.

Run (from repo root):
    uv run python scripts/bench_max_bitrate.py
    uv run python scripts/bench_max_bitrate.py --mod psk16 --code 5/6
    uv run python scripts/bench_max_bitrate.py --length 1500 --frames 16
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from modules.pipeline import (
    PipelineConfig,
    Packet,
    TXPipeline,
    RXPipeline,
)
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.ldpc.channel_coding import CodeRates


MOD_NAMES = {
    "bpsk":  ModulationSchemes.BPSK,
    "qpsk":  ModulationSchemes.QPSK,
    "psk8":  ModulationSchemes.PSK8,
    "psk16": ModulationSchemes.PSK16,
}
MOD_BPS = {
    ModulationSchemes.BPSK:  1,
    ModulationSchemes.QPSK:  2,
    ModulationSchemes.PSK8:  3,
    ModulationSchemes.PSK16: 4,
}

CODE_NAMES = {
    "none": CodeRates.NONE,
    "2/3":  CodeRates.TWO_THIRDS_RATE,
    "3/4":  CodeRates.THREE_QUARTER_RATE,
    "5/6":  CodeRates.FIVE_SIXTH_RATE,
}


def fmt_hz(hz: float) -> str:
    if hz >= 1e9: return f"{hz/1e9:8.3f} GHz"
    if hz >= 1e6: return f"{hz/1e6:8.3f} MHz"
    if hz >= 1e3: return f"{hz/1e3:8.3f} kHz"
    return f"{hz:8.1f}  Hz"


def fmt_bps(bps: float) -> str:
    if bps >= 1e9: return f"{bps/1e9:8.3f} Gbit/s"
    if bps >= 1e6: return f"{bps/1e6:8.3f} Mbit/s"
    if bps >= 1e3: return f"{bps/1e3:8.3f} kbit/s"
    return f"{bps:8.1f}  bit/s"


def time_calls(fn, *args, n: int, warmup: int) -> float:
    """Return the average wall-time per call (seconds)."""
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    return (time.perf_counter() - t0) / n


def make_packet(length: int, seq_num: int, rng: np.random.Generator) -> Packet:
    bits = rng.integers(0, 2, length * 8, dtype=np.uint8).reshape(-1, 1)
    return Packet(
        src_mac=0, dst_mac=1, type=0,
        seq_num=seq_num, length=length, payload=bits,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Max sustainable bitrate benchmark for the TX/RX pipeline.")
    ap.add_argument("--mod", choices=list(MOD_NAMES), default="psk8")
    ap.add_argument("--code", choices=list(CODE_NAMES), default="5/6")
    ap.add_argument("--length", type=int, default=2047,
                    help="payload length in bytes (max 2047 — the 11-bit header field max)")
    ap.add_argument("--sps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=8,
                    help="back-to-back frames per RX buffer (default 8)")
    ap.add_argument("--tx-iters", type=int, default=200)
    ap.add_argument("--rx-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0xBEEF)
    args = ap.parse_args()

    if args.length > 2047:
        ap.error("--length must be <= 2047 (11-bit payload_length field)")

    # The pipeline logs at INFO during decode; quiet it so the benchmark
    # output isn't drowned out.
    logging.basicConfig(level=logging.WARNING)

    cfg = PipelineConfig(
        SPS=args.sps,
        MOD_SCHEME=MOD_NAMES[args.mod],
        CODING_RATE=CODE_NAMES[args.code],
    )
    tx = TXPipeline(cfg)
    rx = RXPipeline(cfg)
    rng = np.random.default_rng(args.seed)

    bps = MOD_BPS[cfg.MOD_SCHEME]

    # ---- TX timing ----
    pkt = make_packet(args.length, seq_num=0, rng=rng)
    samples = tx.transmit(pkt)
    samples_per_frame = samples.size

    print("=== Pipeline ===")
    print(f"  modulation         : {cfg.MOD_SCHEME.name} ({bps} bits/sym)")
    print(f"  code rate          : {cfg.CODING_RATE.name}")
    print(f"  SPS                : {cfg.SPS}")
    print(f"  payload            : {args.length} bytes ({args.length*8} useful bits)")
    print(f"  samples per frame  : {samples_per_frame}")
    print(f"  symbols per frame  : {samples_per_frame // cfg.SPS}")
    print()

    tx_avg = time_calls(tx.transmit, pkt, n=args.tx_iters, warmup=5)
    tx_sps = samples_per_frame / tx_avg

    # ---- RX timing ----
    # Concatenate N back-to-back frames.  Each transmit() output already starts
    # and ends with cfg.GUARD_SYMS_LENGTH guard symbols, so concatenation gives
    # a stream with 2× guard between adjacent frames — i.e. as tight as the
    # pipeline emits on its own.
    pkts = [make_packet(args.length, seq_num=i & 0xF, rng=rng)
            for i in range(args.frames)]
    parts = [tx.transmit(p) for p in pkts]
    rx_buffer = np.concatenate(parts).astype(np.complex64)
    # Tail pad so the last frame's RRC ringing doesn't get clipped by the
    # match-filter trailing edge.
    rx_buffer = np.concatenate([
        rx_buffer,
        np.zeros(cfg.SPS * (2 * 8 + 16), dtype=np.complex64),
    ])
    rx_samples = rx_buffer.size

    # Sanity check: every packet should decode in a noiseless loopback.
    decoded, _ = rx.receive(rx_buffer)
    n_ok = sum(1 for p in decoded if p.valid)
    if n_ok != args.frames:
        print(f"!! WARNING: only {n_ok}/{args.frames} frames decoded — "
              f"benchmark may not reflect a healthy pipeline path")
        print()

    rx_avg = time_calls(rx.receive, rx_buffer, n=args.rx_iters, warmup=2)
    rx_sps = rx_samples / rx_avg

    # ---- Useful-bits accounting ----
    useful_bits_per_frame = args.length * 8
    bits_per_sample = useful_bits_per_frame / samples_per_frame  # raw payload / on-air

    # ---- Report ----
    print("=== TX (single-threaded TXPipeline.transmit) ===")
    print(f"  encode time        : {tx_avg*1e3:8.3f} ms / packet")
    print(f"  TX sample rate     : {fmt_hz(tx_sps)}  (max fs the encoder can feed)")
    print(f"  TX symbol rate     : {fmt_hz(tx_sps / cfg.SPS)}")
    print(f"  TX useful bitrate  : {fmt_bps(tx_sps * bits_per_sample)}")
    print()

    per_frame_rx = rx_avg / args.frames
    print(f"=== RX (single-threaded RXPipeline.receive, {args.frames} frames/buffer, {rx_samples} samples) ===")
    print(f"  decode time        : {rx_avg*1e3:8.3f} ms / buffer  ({per_frame_rx*1e3:.3f} ms / packet)")
    print(f"  RX sample rate     : {fmt_hz(rx_sps)}  (max fs the decoder can drain)")
    print(f"  RX symbol rate     : {fmt_hz(rx_sps / cfg.SPS)}")
    print(f"  RX useful bitrate  : {fmt_bps(rx_sps * bits_per_sample)}")
    print()

    bottleneck_sps = min(tx_sps, rx_sps)
    bottleneck_side = "TX" if tx_sps < rx_sps else "RX"
    headroom = max(tx_sps, rx_sps) / bottleneck_sps
    eff_pct = bits_per_sample * 100

    print("=== System (TX + RX combined, single-thread each) ===")
    print(f"  bottleneck         : {bottleneck_side} (other side has {headroom:.2f}× headroom)")
    print(f"  max Pluto fs       : {fmt_hz(bottleneck_sps)}")
    print(f"  max symbol rate    : {fmt_hz(bottleneck_sps / cfg.SPS)}")
    print(f"  frame efficiency   : {eff_pct:.2f} % ({useful_bits_per_frame} useful "
          f"bits / {samples_per_frame} samples per frame)")
    print(f"  USEFUL BITRATE     : {fmt_bps(bottleneck_sps * bits_per_sample)}")
    print()
    print("Notes:")
    print("  * Numbers are single-threaded.  parallel_pipeline.{TX,RX}WorkerPool")
    print("    can scale either side roughly linearly with worker count up to")
    print("    available P-cores.")
    print("  * 'frame efficiency' is the fraction of on-air samples that carry")
    print("    user payload (preamble, header, guard, LDPC parity, RRC upsampling")
    print("    all eat into the rest).  Increase it by raising bits/symbol")
    print("    (--mod psk16), the code rate (--code 5/6), or by lowering --sps.")
    print("  * Pluto/AD9363 nominal fs ceiling is 61.44 MHz; the practical USB-2")
    print("    Pluto ceiling is closer to 5–10 MHz.  Anything above that here")
    print("    is PC-side headroom, not link headroom.")


if __name__ == "__main__":
    main()
