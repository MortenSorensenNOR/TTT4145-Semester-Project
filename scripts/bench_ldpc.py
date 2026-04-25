"""Benchmark LDPC encode/decode time for a 1500-byte packet @ 5/6 rate.

Reports per-codeword and per-packet timings to estimate how much budget
LDPC would consume if added to the pipeline.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.ldpc.channel_coding import CodeRates
from modules.ldpc.ldpc import LDPCConfig, ldpc_decode, ldpc_encode

PAYLOAD_BYTES = 1500
PAYLOAD_BITS = PAYLOAD_BYTES * 8  # 12000

# 5/6 rate options (k, n):
#   n=648,  k=540   → 23 codewords (12420 bits, 420 padding)
#   n=1296, k=1080  → 12 codewords (12960 bits, 960 padding)
#   n=1944, k=1620  →  8 codewords (12960 bits, 960 padding)
CONFIGS = [
    LDPCConfig(k=540, code_rate=CodeRates.FIVE_SIXTH_RATE),
    LDPCConfig(k=1080, code_rate=CodeRates.FIVE_SIXTH_RATE),
    LDPCConfig(k=1620, code_rate=CodeRates.FIVE_SIXTH_RATE),
]

# Run a few SNRs so we can see how the early-exit behaves.
# At high SNR BP converges in ~1-2 iterations; near threshold it hits max.
SNR_DBS = [6.0, 4.0, 2.0]
MAX_ITER = 50
N_TRIALS = 10


def bpsk_channel(codeword: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Map bits 0/1 → +1/-1, add AWGN, return LLR (positive = bit 0)."""
    tx = 1.0 - 2.0 * codeword.astype(np.float32)
    sigma = np.sqrt(0.5 / 10 ** (snr_db / 10))
    noise = rng.standard_normal(len(tx)).astype(np.float32) * sigma
    rx = tx + noise
    return (2.0 * rx / sigma**2).astype(np.float32)


def bench_one(config: LDPCConfig, n_codewords: int) -> None:
    rng = np.random.default_rng(42)
    print(f"\n=== n={config.n}, k={config.k}, R={config.code_rate.value_float:.4f}, "
          f"codewords/packet={n_codewords} ===")

    # Warmup (caches H, G, decode structures).
    msg = rng.integers(0, 2, config.k, dtype=np.uint8)
    cw = ldpc_encode(msg, config)
    llr = bpsk_channel(cw, 8.0, rng)
    _ = ldpc_decode(llr, config, max_iterations=MAX_ITER)

    # ---- encode timing ----
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        for _ in range(n_codewords):
            msg = rng.integers(0, 2, config.k, dtype=np.uint8)
            ldpc_encode(msg, config)
    enc_total = (time.perf_counter() - t0) / N_TRIALS
    enc_per_cw = enc_total / n_codewords
    print(f"  encode : {enc_per_cw * 1e3:6.2f} ms/cw  →  {enc_total * 1e3:6.1f} ms/packet")

    # ---- decode timing at several SNRs ----
    for snr in SNR_DBS:
        # Build a fresh batch of noisy LLRs so timing reflects realistic
        # convergence behavior, not a single repeated codeword.
        msgs = [rng.integers(0, 2, config.k, dtype=np.uint8) for _ in range(n_codewords)]
        cws = [ldpc_encode(m, config) for m in msgs]
        llrs = [bpsk_channel(c, snr, rng) for c in cws]

        # Track success rate to make sure we're not in a degenerate regime.
        n_correct = 0

        t0 = time.perf_counter()
        for _ in range(N_TRIALS):
            for m, llr in zip(msgs, llrs):
                decoded = ldpc_decode(llr, config, max_iterations=MAX_ITER)
                if np.array_equal(decoded, m):
                    n_correct += 1
        dec_total = (time.perf_counter() - t0) / N_TRIALS
        dec_per_cw = dec_total / n_codewords
        ber_cw = 1.0 - n_correct / (N_TRIALS * n_codewords)
        print(f"  decode @ {snr:+.1f} dB : {dec_per_cw * 1e3:6.2f} ms/cw  →  "
              f"{dec_total * 1e3:7.1f} ms/packet   (cw error rate: {ber_cw:5.2%})")


def main() -> None:
    print(f"Payload: {PAYLOAD_BYTES} bytes ({PAYLOAD_BITS} bits)  @  5/6 rate")
    print(f"Trials per config: {N_TRIALS}, max BP iterations: {MAX_ITER}")
    print(f"Numba JIT available: ", end="")
    try:
        import numba  # noqa: F401
        print("YES")
    except ImportError:
        print("NO  (decoder runs in pure Python — expect ~10-50x slowdown)")

    # Reference air time for the data alone.
    sample_rate = 4_000_000
    sps = 4
    sym_rate = sample_rate / sps  # 1 Msym/s
    # QPSK = 2 bits/sym, PSK8 = 3 bits/sym.
    for mod_name, bps in [("QPSK", 2), ("PSK8", 3)]:
        coded_bits = PAYLOAD_BITS / (5 / 6)  # bits on the air after FEC
        air_ms = coded_bits / bps / sym_rate * 1e3
        print(f"  reference air time @ {mod_name}, R=5/6: {air_ms:.2f} ms/packet")

    for cfg in CONFIGS:
        n_cw = int(np.ceil(PAYLOAD_BITS / cfg.k))
        bench_one(cfg, n_cw)


if __name__ == "__main__":
    main()
