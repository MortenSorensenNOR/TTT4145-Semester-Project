"""Profile encode + decode at n=1944, R=5/6 to localize hot paths."""
import cProfile, pstats, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.ldpc.channel_coding import CodeRates
from modules.ldpc.ldpc import LDPCConfig, ldpc_encode, ldpc_decode

cfg = LDPCConfig(k=1620, code_rate=CodeRates.FIVE_SIXTH_RATE)
rng = np.random.default_rng(0)

# warm caches
m = rng.integers(0, 2, cfg.k, dtype=np.uint8)
cw = ldpc_encode(m, cfg)
sigma = np.sqrt(0.5 / 10**(4/10))
tx = 1.0 - 2.0 * cw.astype(np.float32)
llr = (2 * (tx + rng.standard_normal(cfg.n).astype(np.float32) * sigma) / sigma**2).astype(np.float32)
ldpc_decode(llr, cfg, max_iterations=50)

def bench_encode():
    for _ in range(50):
        msg = rng.integers(0, 2, cfg.k, dtype=np.uint8)
        ldpc_encode(msg, cfg)

def bench_decode():
    for _ in range(50):
        ldpc_decode(llr, cfg, max_iterations=50)

print("=== ENCODE profile (50 calls, n=1944, k=1620) ===")
pr = cProfile.Profile()
pr.enable(); bench_encode(); pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(15)

print("\n=== DECODE profile (50 calls, n=1944, k=1620, near threshold) ===")
pr = cProfile.Profile()
pr.enable(); bench_decode(); pr.disable()
pstats.Stats(pr).sort_stats('cumulative').print_stats(15)
