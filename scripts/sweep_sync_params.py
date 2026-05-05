"""Joint Optuna search for Costas + NDA-TED parameters.

Tunes five parameters together against captured rx_buffs:
    - COSTAS_CONFIG.loop_noise_bandwidth_normalized   (bn)
    - COSTAS_CONFIG.damping_factor                    (zeta_costas)
    - PipelineConfig.NDA_BN_TS                        (bnts)
    - PipelineConfig.NDA_ZETA                         (zeta_nda)
    - PipelineConfig.NDA_L                            (L)

Each trial scores a random mini-batch of training buffers; the best trials
are re-evaluated on a held-out validation set.  Buffers come from
``pluto/one_way_threaded.py --save-rx-buf``; the captured ``seq_nums`` are
treated as a lower bound — score is total valid packets, since better
parameters can decode frames the original config missed.

Run:
    uv run python scripts/sweep_sync_params.py psk8
    uv run python scripts/sweep_sync_params.py psk16 --n-trials 400
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.costas_loop.costas import CostasConfig
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.modulators.modulators import BPSK, PSK8, PSK16, QPSK
from modules.pipeline import PipelineConfig, RXPipeline


_MOD_INSTANCE = {
    ModulationSchemes.BPSK:  BPSK(),
    ModulationSchemes.QPSK:  QPSK(),
    ModulationSchemes.PSK8:  PSK8(),
    ModulationSchemes.PSK16: PSK16(),
}


def _evm_sq_sum(rx_symbols: np.ndarray, mod_scheme: ModulationSchemes) -> tuple[float, int]:
    """Return (sum_sq_error, n_symbols) of rx_symbols vs nearest ideal point."""
    if rx_symbols is None or rx_symbols.size == 0:
        return 0.0, 0
    ideal = _MOD_INSTANCE[mod_scheme].symbol_mapping
    d = np.abs(rx_symbols[:, None] - ideal[None, :])
    nearest = ideal[np.argmin(d, axis=1)]
    err = rx_symbols - nearest
    return float(np.sum(np.abs(err) ** 2)), int(rx_symbols.size)


@dataclass
class Buffer:
    samples: np.ndarray
    search_from: int
    expected_seqs: set[int]
    path: str


def load_buffers(buf_dir: Path) -> list[Buffer]:
    files = sorted(glob.glob(str(buf_dir / "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz buffers in {buf_dir}")
    bufs: list[Buffer] = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        bufs.append(Buffer(
            samples=d["samples"].astype(np.complex64),
            search_from=int(d["search_from"]),
            expected_seqs=set(int(s) for s in d["seq_nums"]) if "seq_nums" in d.files else set(),
            path=f,
        ))
    return bufs


def _seq_from_payload(payload: np.ndarray) -> int | None:
    if payload is None or payload.size < 32:
        return None
    sb = np.packbits(payload[:32].astype(np.uint8))
    return ((int(sb[0]) << 24) | (int(sb[1]) << 16)
            | (int(sb[2]) << 8)  |  int(sb[3]))


def _build_pipeline(bn: float, zeta_costas: float, bnts: float,
                    zeta_nda: float, L: int) -> RXPipeline:
    cfg = PipelineConfig()
    cfg.COSTAS_CONFIG = CostasConfig(
        loop_noise_bandwidth_normalized=bn,
        damping_factor=zeta_costas,
    )
    cfg.NDA_BN_TS = bnts
    cfg.NDA_ZETA  = zeta_nda
    cfg.NDA_L     = int(L)
    return RXPipeline(cfg)


@dataclass
class ScoreResult:
    valid: int
    detected: int
    matched: int
    expected: int
    sq_err: float
    n_syms: int

    @property
    def evm_pct(self) -> float:
        if self.n_syms == 0:
            return float("nan")
        return float(np.sqrt(self.sq_err / self.n_syms) * 100.0)


def score_buffers(rx: RXPipeline, buffers: list[Buffer]) -> ScoreResult:
    valid = detected = matched = expected = n_syms = 0
    sq_err = 0.0
    for b in buffers:
        expected += len(b.expected_seqs)
        try:
            packets, _ = rx.receive(b.samples, search_from=b.search_from)
        except Exception:
            continue
        decoded_seqs: set[int] = set()
        for p in packets:
            detected += 1
            if not p.valid:
                continue
            valid += 1
            seq = _seq_from_payload(p.payload)
            if seq is not None:
                decoded_seqs.add(seq)
            if p.rx_symbols is not None and p.mod_scheme is not None:
                e, n = _evm_sq_sum(p.rx_symbols, p.mod_scheme)
                sq_err += e
                n_syms += n
        matched += len(decoded_seqs & b.expected_seqs)
    return ScoreResult(valid=valid, detected=detected, matched=matched,
                       expected=expected, sq_err=sq_err, n_syms=n_syms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["psk8", "psk16", "both"])
    ap.add_argument("--n-trials", type=int, default=300,
                    help="number of Optuna trials (default 300)")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="fraction of buffers used as the training pool")
    ap.add_argument("--batch-size", type=int, default=24,
                    help="random buffers drawn per trial from the training pool")
    ap.add_argument("--top-k", type=int, default=10,
                    help="top trials re-scored on the full validation set")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="parallel Optuna workers (threads). >1 only safe if "
                         "the C++ extensions release the GIL")
    ap.add_argument("--bn-lo",   type=float, default=1e-4)
    ap.add_argument("--bn-hi",   type=float, default=5e-1)
    ap.add_argument("--zeta-costas-lo", type=float, default=0.3)
    ap.add_argument("--zeta-costas-hi", type=float, default=2.0)
    ap.add_argument("--bnts-lo", type=float, default=1e-6)
    ap.add_argument("--bnts-hi", type=float, default=5e-2)
    ap.add_argument("--zeta-nda-lo", type=float, default=0.3)
    ap.add_argument("--zeta-nda-hi", type=float, default=2.0)
    ap.add_argument("--fix-zeta", type=float, default=None,
                    help="if set, pin both Costas and NDA damping factors to "
                         "this value and skip suggesting them (e.g. 0.7071 "
                         "for the Butterworth/critically-damped baseline). "
                         "Use to test whether high-ζ optima are spurious.")
    ap.add_argument("--L-lo",    type=int,   default=1)
    ap.add_argument("--L-hi",    type=int,   default=24)
    ap.add_argument("--out", type=str, default=None,
                    help="optional JSON file to dump full results")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parents[1] / "pluto" / "rx_buffs"
    if args.dataset == "both":
        buffers = load_buffers(base_dir / "psk8") + load_buffers(base_dir / "psk16")
        print(f"Loaded {len(buffers)} buffers from psk8 + psk16")
    else:
        buffers = load_buffers(base_dir / args.dataset)
        print(f"Loaded {len(buffers)} buffers from {base_dir / args.dataset}")

    n = len(buffers)
    perm = rng.permutation(n)
    n_train = int(round(n * args.train_frac))
    train_pool = [buffers[i] for i in perm[:n_train]]
    val_pool   = [buffers[i] for i in perm[n_train:]]
    print(f"Train pool: {len(train_pool)}   Val pool: {len(val_pool)}")

    # Baseline (current PipelineConfig defaults) on the val set, for context.
    base_rx = RXPipeline(PipelineConfig())
    base = score_buffers(base_rx, val_pool)
    print(f"\nBaseline on val: valid={base.valid}/{base.detected} detected, "
          f"matched_seqs={base.matched}/{base.expected}, EVM={base.evm_pct:.2f}%")

    # ---- Optuna study --------------------------------------------------------
    def objective(trial: optuna.Trial) -> float:
        bn   = trial.suggest_float("bn",   args.bn_lo,   args.bn_hi,   log=True)
        bnts = trial.suggest_float("bnts", args.bnts_lo, args.bnts_hi, log=True)
        L    = trial.suggest_int  ("L",    args.L_lo,    args.L_hi)
        if args.fix_zeta is not None:
            zeta_costas = zeta_nda = float(args.fix_zeta)
            # Persist them on the trial so the printed table still shows the
            # value used, and the val re-eval can read it back.
            trial.set_user_attr("zeta_costas", zeta_costas)
            trial.set_user_attr("zeta_nda",    zeta_nda)
        else:
            zeta_costas = trial.suggest_float("zeta_costas", args.zeta_costas_lo, args.zeta_costas_hi)
            zeta_nda    = trial.suggest_float("zeta_nda",    args.zeta_nda_lo,    args.zeta_nda_hi)

        # Mini-batch: use a deterministic-per-trial subset so the same trial
        # number is reproducible. Different trials see different subsets so
        # the TPE sampler is regularised against batch-specific overfitting.
        sub_rng = np.random.default_rng(args.seed + 1 + trial.number)
        size = min(args.batch_size, len(train_pool))
        idx = sub_rng.choice(len(train_pool), size=size, replace=False)
        batch = [train_pool[i] for i in idx]

        rx = _build_pipeline(bn, zeta_costas, bnts, zeta_nda, L)
        s = score_buffers(rx, batch)
        # Objective: minimise EVM (Optuna maximises, so negate).  A config
        # that fails to decode anything has no symbols to compute EVM on —
        # treat as the worst possible score so the search avoids that region.
        if s.n_syms == 0:
            return -1e6
        return -s.evm_pct

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True)
    # Direction is "maximize" since the objective returns -EVM%.
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=f"sync-{args.dataset}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nRunning Optuna: n_trials={args.n_trials}  batch_size={args.batch_size}  n_jobs={args.n_jobs}")
    t0 = time.perf_counter()
    show_pbar = sys.stdout.isatty()
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs,
                   show_progress_bar=show_pbar)
    elapsed = time.perf_counter() - t0
    print(f"Search done in {elapsed:.1f}s")

    # ---- Promote top-K to full validation -----------------------------------
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: -t.value)
    top = completed[: args.top_k]

    def _full_params(trial: optuna.trial.FrozenTrial) -> dict:
        p = dict(trial.params)
        if "zeta_costas" not in p:
            p["zeta_costas"] = trial.user_attrs.get("zeta_costas", float(args.fix_zeta))
        if "zeta_nda" not in p:
            p["zeta_nda"] = trial.user_attrs.get("zeta_nda", float(args.fix_zeta))
        return p

    print(f"\nTop {len(top)} by mini-batch score (train pool):")
    hdr = f"{'rank':>4}  {'bn':>8}  {'ζ_cos':>5}  {'BnTs':>9}  {'ζ_nda':>5}  {'L':>3}  {'EVM%':>7}"
    print(hdr)
    for i, t in enumerate(top, 1):
        p = _full_params(t)
        print(f"{i:>4}  {p['bn']:>8.5f}  {p['zeta_costas']:>5.2f}  "
              f"{p['bnts']:>9.6f}  {p['zeta_nda']:>5.2f}  {p['L']:>3d}  {-t.value:>7.3f}")

    print(f"\nRe-evaluating top-{len(top)} on validation pool ({len(val_pool)} buffers)...")
    val_scores: list[tuple[dict, ScoreResult]] = []
    for t in top:
        p = _full_params(t)
        rx = _build_pipeline(p["bn"], p["zeta_costas"], p["bnts"], p["zeta_nda"], p["L"])
        s = score_buffers(rx, val_pool)
        val_scores.append((p, s))

    # primary: EVM asc, tiebreak: valid desc
    val_scores.sort(key=lambda x: (x[1].evm_pct, -x[1].valid))

    print(f"\nValidation ranking (sorted by EVM asc):")
    print(f"{'rank':>4}  {'bn':>8}  {'ζ_cos':>5}  {'BnTs':>9}  {'ζ_nda':>5}  {'L':>3}  "
          f"{'valid':>6}  {'detected':>8}  {'matched':>7}  {'expected':>8}  {'EVM%':>6}")
    for i, (p, s) in enumerate(val_scores, 1):
        print(f"{i:>4}  {p['bn']:>8.5f}  {p['zeta_costas']:>5.2f}  {p['bnts']:>9.6f}  "
              f"{p['zeta_nda']:>5.2f}  {p['L']:>3d}  "
              f"{s.valid:>6d}  {s.detected:>8d}  {s.matched:>7d}  {s.expected:>8d}  "
              f"{s.evm_pct:>6.2f}")

    best_p, best_s = val_scores[0]
    print(f"\n=== BEST for {args.dataset} ===")
    print(f"  loop_noise_bandwidth_normalized = {best_p['bn']:.5f}")
    print(f"  damping_factor (Costas)         = {best_p['zeta_costas']:.3f}")
    print(f"  NDA_BN_TS                       = {best_p['bnts']:.6f}")
    print(f"  NDA_ZETA                        = {best_p['zeta_nda']:.3f}")
    print(f"  NDA_L                           = {best_p['L']}")
    print(f"  val: valid={best_s.valid}  detected={best_s.detected}  "
          f"matched_seqs={best_s.matched}/{best_s.expected}  EVM={best_s.evm_pct:.2f}%")
    print(f"  baseline: valid={base.valid}  matched_seqs={base.matched}/{base.expected}  "
          f"EVM={base.evm_pct:.2f}%")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "dataset": args.dataset,
                "n_trials": args.n_trials,
                "train_size": len(train_pool),
                "val_size": len(val_pool),
                "baseline_val": {
                    "valid": base.valid, "detected": base.detected,
                    "matched": base.matched, "expected": base.expected,
                    "evm_pct": base.evm_pct,
                },
                "best": {
                    "params": best_p,
                    "val_valid": best_s.valid, "val_detected": best_s.detected,
                    "val_matched": best_s.matched, "val_expected": best_s.expected,
                    "val_evm_pct": best_s.evm_pct,
                },
                "top": [
                    {"params": p, "val_valid": s.valid, "val_detected": s.detected,
                     "val_matched": s.matched, "val_expected": s.expected,
                     "val_evm_pct": s.evm_pct}
                    for (p, s) in val_scores
                ],
            }, f, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
