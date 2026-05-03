"""Measure LDPC coding gain over AWGN for THREE_QUARTER_RATE and FIVE_SIXTH_RATE
versus uncoded (NONE), using the project's PSK8 + LLR demapper + LDPC decoder.

Pipeline per Eb/N0 point:
  info_bits → [LDPC encode] → PSK8 → AWGN → max-log LLR → [LDPC decode] → BER

Energy normalisation (PSK8 is unit-norm so Es = 1):
  Eb/N0 (lin) = Es / (bits_per_sym · code_rate · N0)
  ⇒ N0 = 1 / (Eb/N0_lin · bits_per_sym · code_rate)

Coding gain at a target BER y* is reported as the dB shift between curves:
  gain(rate)_dB = Eb/N0(NONE; BER=y*) − Eb/N0(rate; BER=y*)

Usage:
  uv run python scripts/ldpc_coding_gain.py
  uv run python scripts/ldpc_coding_gain.py --ebn0 2:12:0.5 --target-ber 1e-4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from modules.ldpc.channel_coding import CodeRates
from modules.ldpc.ldpc import LDPCConfig, ldpc_encode_batch, ldpc_decode_batch
from modules.modulators.modulators import BPSK, QPSK, PSK8, PSK16

_MOD_CLASSES = {"bpsk": BPSK, "qpsk": QPSK, "psk8": PSK8, "psk16": PSK16}


def _ber_for_rate(
    rate: CodeRates,
    ebn0_db_grid: np.ndarray,
    *,
    n_block: int,
    cw_per_batch: int,
    target_errs: int,
    max_bits_per_pt: int,
    early_zero_bits: int,
    max_iter: int,
    mod_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    mod = _MOD_CLASSES[mod_name]()
    bps = mod.bits_per_symbol

    if rate == CodeRates.NONE:
        info_per_block = n_block
        cfg = None
        rate_lin = 1.0
    else:
        num, denom = rate.rate_fraction
        info_per_block = (n_block * num) // denom
        cfg = LDPCConfig(k=info_per_block, code_rate=rate)
        rate_lin = num / denom

    out = np.full_like(ebn0_db_grid, np.nan, dtype=np.float64)

    for i, ebn0_db in enumerate(ebn0_db_grid):
        ebn0_lin = 10 ** (ebn0_db / 10)
        # Es = 1, info-bits/symbol = bps * rate_lin
        N0 = 1.0 / (ebn0_lin * bps * rate_lin)
        sigma = float(np.sqrt(N0 / 2.0))

        n_errs = 0
        n_bits = 0
        t0 = time.perf_counter()
        while n_errs < target_errs and n_bits < max_bits_per_pt:
            info = rng.integers(0, 2, size=cw_per_batch * info_per_block, dtype=np.uint8)
            if cfg is None:
                coded = info
            else:
                coded = ldpc_encode_batch(info.reshape(cw_per_batch, info_per_block), cfg).ravel()

            # Pad coded bits up to a multiple of bits_per_symbol
            pad = (-coded.size) % bps
            if pad:
                coded = np.concatenate([coded, np.zeros(pad, dtype=np.uint8)])

            syms = mod.bits2symbols(coded)
            noise = (sigma * (rng.standard_normal(syms.size) + 1j * rng.standard_normal(syms.size))).astype(np.complex64)
            rx = syms + noise

            llrs = mod.symbols2llrs(rx).ravel()
            n_coded_bits = cw_per_batch * (info_per_block if cfg is None else cfg.n)
            llrs = llrs[:n_coded_bits]

            if cfg is None:
                bits_hat = (llrs < 0).astype(np.uint8)
                errs = int(np.count_nonzero(bits_hat != info))
            else:
                decoded = ldpc_decode_batch(
                    llrs.reshape(cw_per_batch, cfg.n), cfg,
                    max_iterations=max_iter,
                ).ravel()
                errs = int(np.count_nonzero(decoded != info))

            n_errs += errs
            n_bits += info.size

            # Once we're well past the target with zero errors, this point is
            # safely below the noise floor — stop instead of grinding the cap.
            if n_errs == 0 and n_bits >= early_zero_bits:
                break

        elapsed = time.perf_counter() - t0
        ber = n_errs / n_bits if n_bits else float("nan")
        out[i] = ber
        rate_name = rate.name
        print(f"  {rate_name:<22s}  Eb/N0={ebn0_db:5.2f} dB  "
              f"BER={ber:.3e}  ({n_errs} errs / {n_bits} bits, {elapsed:.1f}s)")

    return out


def _interp_x_at_y(x: np.ndarray, y: np.ndarray, y_target: float) -> float | None:
    """Linear-interpolate x at y=y_target along log10(y).

    Treats BER=0 (below detection floor) as 'well below target' so a positive
    sample followed by zero counts as a valid bracket.
    """
    if x.size < 2 or not np.any(np.isfinite(y)):
        return None
    log_t = np.log10(y_target)
    # Floor: any zero-BER point is treated as "much smaller than target" so
    # it can still anchor a downward crossing.
    floor = log_t - 6.0
    log_y = np.where((y > 0) & np.isfinite(y), np.log10(np.clip(y, 1e-300, None)), floor)
    for i in range(x.size - 1):
        a, b = log_y[i], log_y[i + 1]
        if (a - log_t) * (b - log_t) <= 0 and a != b:
            t = (log_t - a) / (b - a)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def _parse_grid(spec: str) -> np.ndarray:
    a, b, s = (float(x) for x in spec.split(":"))
    return np.arange(a, b + s / 2, s, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ebn0", default="2:12:0.5",
                    help="Eb/N0 sweep as 'start:stop:step' in dB (default 2:12:0.5)")
    ap.add_argument("--target-ber", type=float, default=1e-4,
                    help="Report Eb/N0 needed at this BER for each rate (default 1e-4)")
    ap.add_argument("--n", type=int, default=1944, choices=(648, 1296, 1944),
                    help="LDPC codeword length (default 1944 — fastest, biggest gain)")
    ap.add_argument("--batch", type=int, default=64, help="Codewords per batch (default 64)")
    ap.add_argument("--errs", type=int, default=200, help="Target errors per point (default 200)")
    ap.add_argument("--max-bits", type=int, default=4_000_000,
                    help="Max info bits per point (default 4M)")
    ap.add_argument("--early-zero-bits", type=int, default=1_500_000,
                    help="If we've seen this many bits with 0 errors, stop "
                         "processing this point early (default 1.5M)")
    ap.add_argument("--max-iter", type=int, default=50, help="Max BP iterations (default 50)")
    ap.add_argument("--mod", type=str, default="psk8",
                    choices=tuple(_MOD_CLASSES.keys()),
                    help="Modulation: bpsk, qpsk, psk8 (default), psk16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/ldpc_coding_gain.csv")
    ap.add_argument("--plot", default=None,
                    help="Optional path to save BER-vs-Eb/N0 plot (PNG)")
    args = ap.parse_args()

    ebn0 = _parse_grid(args.ebn0)
    rates = (CodeRates.NONE, CodeRates.THREE_QUARTER_RATE, CodeRates.FIVE_SIXTH_RATE)
    rng = np.random.default_rng(args.seed)

    bers: dict[CodeRates, np.ndarray] = {}
    for r in rates:
        print(f"\n[rate {r.name}]  k={(args.n * r.rate_fraction[0]) // r.rate_fraction[1]}, n={args.n}")
        bers[r] = _ber_for_rate(
            r, ebn0,
            n_block=args.n,
            cw_per_batch=args.batch,
            target_errs=args.errs,
            max_bits_per_pt=args.max_bits,
            early_zero_bits=args.early_zero_bits,
            max_iter=args.max_iter,
            mod_name=args.mod,
            rng=rng,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ebn0_db"] + [r.name for r in rates])
        for i, e in enumerate(ebn0):
            w.writerow([f"{e:g}"] + [f"{bers[r][i]:.6e}" for r in rates])
    print(f"\nBER table written to {out_path}")

    print(f"\nEb/N0 (dB) required to reach BER = {args.target_ber:g}:")
    ebn0_at_target: dict[CodeRates, float | None] = {}
    for r in rates:
        v = _interp_x_at_y(ebn0, bers[r], args.target_ber)
        ebn0_at_target[r] = v
        print(f"  {r.name:<22s}: {('%.2f dB' % v) if v is not None else 'out of sweep range'}")

    base = ebn0_at_target[CodeRates.NONE]
    if base is not None:
        print(f"\nCoding gain @ BER={args.target_ber:g}  (positive ⇒ coded does better):")
        for r in (CodeRates.THREE_QUARTER_RATE, CodeRates.FIVE_SIXTH_RATE):
            v = ebn0_at_target[r]
            if v is None:
                print(f"  {r.name:<22s}: n/a (curve doesn't cross target in sweep)")
            else:
                print(f"  {r.name:<22s}: {base - v:+.2f} dB")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for r in rates:
                ax.semilogy(ebn0, bers[r], marker="o", label=r.name)
            ax.axhline(args.target_ber, color="k", ls="--", lw=0.8, alpha=0.5,
                       label=f"target={args.target_ber:g}")
            ax.set_xlabel("Eb/N0 (dB)")
            ax.set_ylabel("BER (info bits)")
            ax.set_title(f"{args.mod.upper()} + LDPC (n={args.n}) over AWGN")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.plot, dpi=130)
            print(f"plot saved to {args.plot}")
        except Exception as e:
            print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()
