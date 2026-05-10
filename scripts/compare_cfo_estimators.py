"""Compare current single-lag-N/2 CFO estimator vs Luise-Reggiannini multi-lag.

Sweeps SNR across multiple channel scenarios with Monte-Carlo realisations.
Both estimators operate on the same modulation-removed preamble window
(samples[peak:peak+n_ref] * conj(preamble_ref)); only the combiner differs:
    - half-window: single autocorrelation at lag = N/2 (current code path)
    - L&R       : sum of autocorrelations over lags 1..L, single angle

Run:
    uv run python scripts/compare_cfo_estimators.py
    uv run python scripts/compare_cfo_estimators.py --realizations 200 --L 128 --plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.channel import ChannelConfig, ChannelModel
from modules.frame_sync.frame_sync import (
    SynchronizerConfig,
    build_preamble_ref,
    build_preamble_ref_rev,
    full_buffer_xcorr_sync,
    generate_preamble,
)
from modules.pulse_shaping.pulse_shaping import match_filter, rrc_filter, upsample


SPS = 4
SAMPLE_RATE = 6_000_000.0
RRC_ALPHA = 0.25
RRC_SPAN = 8


def cfo_halfwindow(window: np.ndarray, fs: float) -> float:
    """Single-lag-at-N/2 autocorrelation CFO estimator (current code path)."""
    n = len(window)
    half = n // 2
    p = np.vdot(window[:half], window[half:2 * half])
    return float(np.angle(p)) * fs / (np.pi * n)


def cfo_lr(window: np.ndarray, fs: float, L: int) -> float:
    """Luise-Reggiannini multi-lag CFO estimator.

    Computes R(m) = (1/(N-m)) sum_n x*[n] x[n+m] for m=1..L via FFT
    (zero-padded to 2N to avoid circular wrap), sums them, and takes
    one arctan.  Cost: O(N log N).

    The (N-m) normalisation removes the triangular bias from finite-window
    autocorrelation; without it, large L produces a systematic CFO offset.
    """
    n = len(window)
    X = np.fft.fft(window, 2 * n)
    psd = np.abs(X) ** 2
    R_all = np.fft.ifft(psd)[:n]
    weights = (n - np.arange(n)).astype(np.float64)
    R_norm = R_all / weights
    p = np.sum(R_norm[1:L + 1])
    return float(np.angle(p)) * fs / (np.pi * (L + 1))


@dataclass
class Scenario:
    name: str
    channel_kwargs: dict = field(default_factory=dict)


def make_scenarios() -> list[Scenario]:
    return [
        Scenario("awgn"),
        Scenario("multipath_2tap", {
            "enable_multipath": True,
            "multipath_delays_samples": (np.float32(0.0), np.float32(3.0)),
            "multipath_gains_db":       (np.float32(0.0), np.float32(-6.0)),
        }),
        Scenario("rayleigh_3tap", {
            "enable_multipath": True,
            "multipath_delays_samples": (np.float32(0.0), np.float32(2.0), np.float32(5.0)),
            "multipath_gains_db":       (np.float32(0.0), np.float32(-3.0), np.float32(-8.0)),
            "doppler_hz": np.float32(10.0),
            "fading_type": "rayleigh",
        }),
        Scenario("rician_k10", {
            "enable_multipath": True,
            "multipath_delays_samples": (np.float32(0.0), np.float32(2.0)),
            "multipath_gains_db":       (np.float32(0.0), np.float32(-6.0)),
            "doppler_hz":   np.float32(5.0),
            "fading_type":  "rician",
            "rician_k_db":  np.float32(10.0),
        }),
        Scenario("phase_noise", {
            "enable_phase_noise": True,
            "phase_noise_psd_dbchz": np.float32(-90.0),
        }),
    ]


def build_tx_signal(sync_cfg: SynchronizerConfig, rrc_taps: np.ndarray,
                    n_payload_syms: int, rng: np.random.Generator) -> np.ndarray:
    preamble = generate_preamble(sync_cfg)
    # QPSK payload — keeps signal power ~1
    bits = rng.integers(0, 2, (n_payload_syms, 2))
    syms_pay = ((2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)) / np.sqrt(2)
    syms = np.concatenate([preamble, syms_pay.astype(np.complex64)])
    tx = upsample(syms, SPS, rrc_taps)
    return np.concatenate([
        np.zeros(200, dtype=np.complex64),
        tx.astype(np.complex64),
        np.zeros(200, dtype=np.complex64),
    ])


def run_trial(
    sync_cfg: SynchronizerConfig,
    rrc_taps: np.ndarray,
    preamble_ref: np.ndarray,
    preamble_ref_rev: np.ndarray,
    preamble_conj: np.ndarray,
    cfo_hz: float,
    snr_db: float,
    scenario: Scenario,
    L: int,
    rng: np.random.Generator,
) -> tuple[float, float] | None:
    tx = build_tx_signal(sync_cfg, rrc_taps, n_payload_syms=200, rng=rng)

    cfg = ChannelConfig(
        sample_rate=np.float32(SAMPLE_RATE),
        snr_db=np.float32(snr_db),
        cfo_hz=np.float32(cfo_hz),
        seed=int(rng.integers(0, 2**31 - 1)),
        **scenario.channel_kwargs,
    )
    chan = ChannelModel(cfg)
    rx = chan.apply(tx)

    filtered = match_filter(rx.astype(np.complex64), rrc_taps)
    result, _ = full_buffer_xcorr_sync(
        filtered, preamble_ref, preamble_ref_rev,
        ncc_threshold=0.05, fs=int(SAMPLE_RATE),
    )
    if result.sample_idxs.size == 0:
        return None

    peak = int(result.sample_idxs[np.argmax(result.peak_ratios)])
    n_ref = len(preamble_ref)
    if peak + n_ref > len(filtered):
        return None

    window = (filtered[peak:peak + n_ref] * preamble_conj).astype(np.complex64)
    err_hw = cfo_halfwindow(window, SAMPLE_RATE) - cfo_hz
    err_lr = cfo_lr(window, SAMPLE_RATE, L) - cfo_hz
    return err_hw, err_lr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snrs", type=float, nargs="+",
                    default=[-5, 0, 5, 10, 15, 20, 25])
    ap.add_argument("--cfo-hz", type=float, default=2000.0,
                    help="True CFO to inject. Must be < fs/n_ref ≈ "
                         "fs/(preamble_nsym*sps) for unambiguous half-window lag.")
    ap.add_argument("--realizations", type=int, default=100)
    ap.add_argument("--L", type=int, default=178,
                    help="Multi-lag depth for L&R. Default = preamble_n_ref/2.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true",
                    help="Save cfo_compare.png with per-scenario RMSE-vs-SNR curves.")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional .npz path to dump raw results.")
    args = ap.parse_args()

    sync_cfg = SynchronizerConfig()
    num_taps = 2 * SPS * RRC_SPAN + 1
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, num_taps)
    preamble_ref = build_preamble_ref(sync_cfg, SPS, rrc_taps)
    preamble_ref_rev = build_preamble_ref_rev(preamble_ref)
    preamble_conj = np.conj(preamble_ref).astype(np.complex64)
    n_ref = len(preamble_ref)

    print(f"# fs={SAMPLE_RATE/1e6:.2f} MHz  sps={SPS}  alpha={RRC_ALPHA}")
    print(f"# preamble_nsym={sync_cfg.preamble_nsym}  n_ref={n_ref}  half-lag={n_ref//2}")
    print(f"# L&R lags = 1..{args.L}")
    print(f"# CFO_true = {args.cfo_hz:.1f} Hz  "
          f"max unambig (half-window) ≈ {SAMPLE_RATE/n_ref:.1f} Hz  "
          f"max unambig (L&R) ≈ {SAMPLE_RATE/(2*args.L):.1f} Hz")
    print(f"# realizations = {args.realizations}")
    print()

    scenarios = make_scenarios()
    rng = np.random.default_rng(args.seed)

    results: dict[tuple[str, float], dict] = {}

    hdr = (f"{'scenario':>16}  {'SNR':>5}  {'valid':>5}  "
           f"{'RMSE_hw':>9}  {'RMSE_LR':>9}  "
           f"{'std_hw':>8}  {'std_LR':>8}  "
           f"{'bias_hw':>8}  {'bias_LR':>8}  {'gain_dB':>7}")
    print(hdr)
    print("-" * len(hdr))

    for scenario in scenarios:
        for snr in args.snrs:
            errs_hw, errs_lr = [], []
            for _ in range(args.realizations):
                r = run_trial(sync_cfg, rrc_taps,
                              preamble_ref, preamble_ref_rev, preamble_conj,
                              args.cfo_hz, snr, scenario, args.L, rng)
                if r is None:
                    continue
                errs_hw.append(r[0])
                errs_lr.append(r[1])

            if not errs_hw:
                print(f"{scenario.name:>16}  {snr:>5.1f}  {0:>5d}  "
                      f"{'-':>9}  {'-':>9}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>8}  {'-':>7}")
                continue

            ehw = np.asarray(errs_hw)
            elr = np.asarray(errs_lr)
            rmse_hw, rmse_lr = float(np.sqrt(np.mean(ehw**2))), float(np.sqrt(np.mean(elr**2)))
            std_hw, std_lr = float(np.std(ehw)), float(np.std(elr))
            bias_hw, bias_lr = float(np.mean(ehw)), float(np.mean(elr))
            gain_db = 20 * np.log10(rmse_hw / rmse_lr) if rmse_lr > 0 else float("inf")

            results[(scenario.name, snr)] = {
                "rmse_hw": rmse_hw, "rmse_lr": rmse_lr,
                "std_hw": std_hw,   "std_lr": std_lr,
                "bias_hw": bias_hw, "bias_lr": bias_lr,
                "n_valid": len(errs_hw),
            }

            print(f"{scenario.name:>16}  {snr:>5.1f}  {len(errs_hw):>5d}  "
                  f"{rmse_hw:>9.2f}  {rmse_lr:>9.2f}  "
                  f"{std_hw:>8.2f}  {std_lr:>8.2f}  "
                  f"{bias_hw:>8.2f}  {bias_lr:>8.2f}  "
                  f"{gain_db:>+7.2f}")

    if args.out:
        np.savez(args.out, **{
            f"{k[0]}_snr{k[1]:.0f}": np.array([v["rmse_hw"], v["rmse_lr"],
                                               v["std_hw"], v["std_lr"],
                                               v["bias_hw"], v["bias_lr"],
                                               v["n_valid"]])
            for k, v in results.items()
        })
        print(f"\nWrote {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt
        n_sc = len(scenarios)
        fig, axes = plt.subplots(1, n_sc, figsize=(4 * n_sc, 4), sharey=True)
        if n_sc == 1:
            axes = [axes]
        for ax, scenario in zip(axes, scenarios):
            snrs_p, hw_p, lr_p = [], [], []
            for snr in args.snrs:
                k = (scenario.name, snr)
                if k in results:
                    snrs_p.append(snr)
                    hw_p.append(results[k]["rmse_hw"])
                    lr_p.append(results[k]["rmse_lr"])
            ax.semilogy(snrs_p, hw_p, "o-", label="half-window (N/2)")
            ax.semilogy(snrs_p, lr_p, "s-", label=f"L&R (L={args.L})")
            ax.set_xlabel("SNR (dB)")
            ax.set_title(scenario.name)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("CFO RMSE (Hz)")
        plt.tight_layout()
        plt.savefig("cfo_compare.png", dpi=120)
        print("\nSaved cfo_compare.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
