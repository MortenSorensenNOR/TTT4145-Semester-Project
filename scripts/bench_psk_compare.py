"""Head-to-head: PSK8 vs PSK16 on the same payload, tracked end-to-end and
per-stage so we can answer "why is PSK16 so slow?".
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pulse_shaping.pulse_shaping import match_filter
from modules.gardner_ted.gardner import apply_gardner_ted
from modules.costas_loop.costas import apply_costas_loop


def time_it(fn, n=50):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1000


def make_buffer(mod: ModulationSchemes, n_packets: int, payload: int):
    cfg = PipelineConfig(MOD_SCHEME=mod)
    tx = TXPipeline(cfg)
    rng = np.random.default_rng(0xCAFE)
    payloads = [bytes(rng.integers(0, 256, payload, dtype=np.uint8))
                for _ in range(n_packets)]
    chunks = []
    for i, p in enumerate(payloads):
        bits = np.unpackbits(np.frombuffer(p, dtype=np.uint8))
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=i,
                     length=payload, payload=bits)
        s = tx.transmit(pkt)
        peak = float(np.max(np.abs(s)))
        if peak > 0:
            s = s / peak
        chunks.append(s.astype(np.complex64))
    silence = np.zeros(256, dtype=np.complex64)
    raw = np.concatenate([silence] + [c for chunk in chunks for c in (chunk, silence)])
    return cfg, raw


def bench(mod: ModulationSchemes, n_packets: int = 8, payload: int = 1500):
    cfg, raw = make_buffer(mod, n_packets, payload)
    rx = RXPipeline(cfg)

    # Sanity
    pkts, _ = rx.receive(raw)
    valid = sum(1 for p in pkts if p.valid)
    print(f"\n=== {mod.name}, {n_packets}×{payload}B, buf={len(raw)} samples ===")
    print(f"  decoded {valid}/{n_packets}  (air = {len(raw)/cfg.SAMPLE_RATE*1000:.2f} ms)")

    rx_total = time_it(lambda: rx.receive(raw))
    print(f"  RX total : {rx_total:.3f} ms / call → {rx_total/n_packets:.3f} ms / pkt")

    # Match-filter once for fair per-stage measurement
    fb = match_filter(raw, rx.rrc_taps)
    detections = rx.detect(fb)

    if not detections:
        print("  no detections")
        return

    # Stage timings on the first detection
    d0 = detections[0]
    cfo_rad = np.float32(2 * np.pi * float(d0.cfo_estimate) / cfg.SAMPLE_RATE * cfg.SPS)

    # Compute payload extent for this mod
    bps = mod.value + 1
    pre_ldpc_n_bits = payload * 8 + rx.frame_constructor.PAYLOAD_CRC_BITS
    from modules.pipeline import _on_air_payload_n_bits
    from modules.ldpc.channel_coding import CodeRates
    n_cw, k, n_air_bits = _on_air_payload_n_bits(pre_ldpc_n_bits, cfg.CODING_RATE)
    n_payload_syms = -(-n_air_bits // bps)

    payload_start = d0.payload_start + rx.frame_constructor.header_encoded_n_bits
    sample_buf = fb[payload_start * cfg.SPS:
                    (payload_start + n_payload_syms) * cfg.SPS + cfg.SPS * cfg.SPS]

    t_g = time_it(lambda: apply_gardner_ted(
        sample_buf, cfg.SPS,
        BnTs=cfg.GARDNER_BN_TS, zeta=cfg.GARDNER_ZETA, L=cfg.GARDNER_L,
        prepend_first=True,
    ))
    print(f"  gardner   : {t_g:.3f} ms ({len(sample_buf)} samples in, "
          f"~{n_payload_syms} symbols out)")

    rx_syms = apply_gardner_ted(
        sample_buf, cfg.SPS,
        BnTs=cfg.GARDNER_BN_TS, zeta=cfg.GARDNER_ZETA, L=cfg.GARDNER_L,
        prepend_first=True,
    )[:n_payload_syms]

    t_c = time_it(lambda: apply_costas_loop(
        rx_syms, cfg.COSTAS_CONFIG, mod,
        current_phase_estimate=np.float32(0.0),
        current_frequency_offset=cfo_rad,
    ))
    print(f"  costas    : {t_c:.3f} ms ({len(rx_syms)} symbols)")

    # LLR + LDPC by replaying the rest of the decode path through Modulator
    syms_corr, _ = apply_costas_loop(
        rx_syms, cfg.COSTAS_CONFIG, mod,
        current_phase_estimate=np.float32(0.0),
        current_frequency_offset=cfo_rad,
    )

    match mod:
        case ModulationSchemes.PSK8:  modulator = rx.psk8
        case ModulationSchemes.PSK16: modulator = rx.psk16
        case ModulationSchemes.QPSK:  modulator = rx.qpsk
        case ModulationSchemes.BPSK:  modulator = rx.bpsk

    t_l = time_it(lambda: modulator.symbols2llrs(syms_corr))
    print(f"  llr       : {t_l:.3f} ms")

    from modules.ldpc.ldpc import LDPCConfig, ldpc_decode_batch
    llrs = modulator.symbols2llrs(syms_corr).ravel()[:n_air_bits]
    cfg_ldpc = LDPCConfig(k=k, code_rate=cfg.CODING_RATE)
    t_d = time_it(lambda: ldpc_decode_batch(
        llrs.reshape(n_cw, n_air_bits // n_cw), cfg_ldpc,
        max_iterations=cfg.LDPC_MAX_ITER,
    ))
    print(f"  ldpc x{n_cw}  : {t_d:.3f} ms")

    print(f"  per-pkt body est : "
          f"gardner+costas+llr+ldpc = "
          f"{t_g+t_c+t_l+t_d:.3f} ms")


def main() -> int:
    bench(ModulationSchemes.BPSK,  n_packets=8, payload=1500)
    bench(ModulationSchemes.QPSK,  n_packets=8, payload=1500)
    bench(ModulationSchemes.PSK8,  n_packets=8, payload=1500)
    bench(ModulationSchemes.PSK16, n_packets=8, payload=1500)
    return 0


if __name__ == "__main__":
    sys.exit(main())
