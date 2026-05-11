"""Diagnose why Pedestrian-A multipath causes ~1-2% PER in tests.

Hypothesis: the test enables Doppler=1.2 Hz fading, which activates Clarke's
sum-of-sinusoids model on every tap (channel.py:298-299), giving the strongest
0 dB tap pure Rayleigh statistics. The receiver has no diversity / no
equalizer, so when |h_0| drops by ~10 dB during a packet, the packet dies.

This script does N_TRIALS, and for each trial records:
  - mean |h_0|^2 averaged over the packet duration (fade depth)
  - whether the packet decoded correctly

Then we plot decode-success vs fade depth.
"""

import numpy as np

from modules.channel import (
    ChannelConfig,
    ChannelModel,
    apply_awgn,
    apply_cfo_and_phase,
    apply_multipath,
    generate_fading_gains,
)
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline

N_TRIALS = 300
SNR_DB   = 30.0  # high SNR: AWGN alone is irrelevant; fade is the culprit
CFO_HZ   = 1000.0
PHASE    = 0.5
XPD_DB   = 20.0  # cross-pol discrimination; 0 to disable

PEDESTRIAN_A_DELAYS   = (np.float32(0.0), np.float32(0.587),
                        np.float32(1.014), np.float32(2.188))
PEDESTRIAN_A_GAINS_DB = (np.float32(0.0), np.float32(-9.7),
                        np.float32(-19.2), np.float32(-22.8))
DOPPLER_HZ = np.float32(1.2)


def build_tx(seed):
    config = PipelineConfig(MOD_SCHEME=ModulationSchemes.QPSK)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, 8 * 8).reshape(-1, 1)
    pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=8, payload=bits)
    signal = TXPipeline(config).transmit(pkt)
    return pkt, config, signal


def run_one(trial_seed):
    tx_pkt, config, signal = build_tx(trial_seed)
    cfg = ChannelConfig(
        sample_rate=config.SAMPLE_RATE, snr_db=SNR_DB,
        cfo_hz=CFO_HZ, initial_phase_rad=PHASE,
        enable_multipath=True,
        multipath_delays_samples=PEDESTRIAN_A_DELAYS,
        multipath_gains_db=PEDESTRIAN_A_GAINS_DB,
        cross_pol_discrimination_db=np.float32(XPD_DB),
        doppler_hz=DOPPLER_HZ,
        fading_type="rayleigh",
        seed=trial_seed,
    )

    rng = np.random.default_rng(trial_seed)
    n_samples = len(signal)
    fading_gains, _ = generate_fading_gains(cfg, n_samples, None, rng)

    h0_mean_power = float(np.mean(np.abs(fading_gains[0]) ** 2))
    h1_mean_power = float(np.mean(np.abs(fading_gains[1]) ** 2))

    rx = ChannelModel(cfg).apply(signal)
    rx_packets, _ = RXPipeline(config).receive(rx)
    rx_by_seq = {p.seq_num: p for p in rx_packets}
    rx_pkt = rx_by_seq.get(0)
    ok = bool(rx_pkt is not None
              and rx_pkt.valid
              and rx_pkt.length == tx_pkt.length
              and np.array_equal(rx_pkt.payload, tx_pkt.payload))

    return h0_mean_power, h1_mean_power, ok


def main():
    results = []
    for i in range(N_TRIALS):
        seed = 1 * 10_007 + i  # mirror test_pipeline_no_hypothesis base_seed=1, trial i
        h0, h1, ok = run_one(seed)
        results.append((seed, h0, h1, ok))

    n_fail = sum(1 for *_, ok in results if not ok)
    print(f"PER = {n_fail}/{N_TRIALS} = {n_fail / N_TRIALS:.2%}")
    print()

    failures = sorted([r for r in results if not r[3]], key=lambda r: r[1])
    successes = sorted([r for r in results if r[3]],  key=lambda r: r[1])

    print(f"Failures sorted by |h0|^2 (dominant tap mean power):")
    print(f"  {'seed':>8}  {'|h0|^2':>10}  {'|h0|^2 dB':>10}  {'|h1|^2':>10}  ok")
    for seed, h0, h1, ok in failures:
        print(f"  {seed:>8}  {h0:>10.4f}  {10*np.log10(h0):>+10.1f}  {h1:>10.4f}  {ok}")

    print()
    print(f"Successful trials (|h0|^2 distribution):")
    s_h0 = np.array([r[1] for r in successes])
    print(f"  min={s_h0.min():.3f}  p5={np.percentile(s_h0,5):.3f}  "
          f"p50={np.percentile(s_h0,50):.3f}  p95={np.percentile(s_h0,95):.3f}  "
          f"max={s_h0.max():.3f}")

    if failures:
        f_h0 = np.array([r[1] for r in failures])
        print(f"Failed trials (|h0|^2 distribution):")
        print(f"  min={f_h0.min():.3f}  p5={np.percentile(f_h0,5):.3f}  "
              f"p50={np.percentile(f_h0,50):.3f}  p95={np.percentile(f_h0,95):.3f}  "
              f"max={f_h0.max():.3f}")
        print()
        print(f"Median dominant-tap power: success={10*np.log10(np.median(s_h0)):+.1f} dB, "
              f"failure={10*np.log10(np.median(f_h0)):+.1f} dB")


if __name__ == "__main__":
    main()
