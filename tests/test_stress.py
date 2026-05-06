"""Stress tests: large payloads (1–1500 bytes), variable burst sizes, bursty-with-gaps.

Tune N_TRIALS / N_CASES to control runtime.
  N_TRIALS=50,  N_CASES=100  → ~5 min
  N_TRIALS=100, N_CASES=100  → ~10 min
  N_TRIALS=200, N_CASES=100  → ~20 min
"""

import numpy as np
import pytest

from modules.channel import ChannelConfig, ChannelModel
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline

N_TRIALS = 200
N_CASES = 100

MOD_SCHEMES        = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]
MOD_SCHEMES_ROBUST = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]


def _specs(n_packets: int, max_length: int, seed: int, mods=MOD_SCHEMES):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, max_length + 1, n_packets)
    picked = [mods[int(i)] for i in rng.integers(0, len(mods), n_packets)]
    return [(i, int(lengths[i]), picked[i]) for i in range(n_packets)]


def _make_signal(specs, seed):
    rng = np.random.default_rng(seed + 999_999)
    tx_packets, parts = [], []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((pkt, config))
        parts.append(TXPipeline(config).transmit(pkt))
    return tx_packets, np.concatenate(parts)


def _make_bursty_signal(groups, silence_samples, seed):
    rng = np.random.default_rng(seed + 888_888)
    tx_packets, parts = [], []
    seq_num = 0
    for group_idx, group in enumerate(groups):
        for _, length, mod in group:
            config = PipelineConfig(MOD_SCHEME=mod)
            bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
            pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
            tx_packets.append((pkt, config))
            parts.append(TXPipeline(config).transmit(pkt))
            seq_num += 1
        gap = silence_samples[group_idx] if group_idx < len(silence_samples) else 0
        if gap > 0:
            parts.append(np.zeros(gap, dtype=np.complex64))
    return tx_packets, np.concatenate(parts)


def _assert_all_received(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"seq_num={seq_num} not received"
        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid, f"seq_num={seq_num} marked invalid"
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload), f"seq_num={seq_num} payload mismatch"


def _run_trials(tx_packets, signal, config, snr_db, n_trials, base_seed):
    """Run n_trials noise/CFO/phase realizations; report all failures at once."""
    failures = []
    for i in range(n_trials):
        rng = np.random.default_rng(base_seed * 10_007 + i)
        cfo_hz = float(rng.uniform(-5000, 5000))
        phase = float(rng.uniform(0, 2 * np.pi))
        ch = ChannelModel(ChannelConfig(
            sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
            cfo_hz=cfo_hz, initial_phase_rad=phase,
            seed=int(rng.integers(0, 2**31)),
        ))
        rx_packets, _ = RXPipeline(config).receive(ch.apply(signal))
        try:
            _assert_all_received(tx_packets, rx_packets)
        except AssertionError as e:
            failures.append((i, cfo_hz, phase, str(e)))

    if failures:
        lines = [f"  trial={t}, cfo={c:.0f}Hz, phase={p:.2f}: {e}" for t, c, p, e in failures[:5]]
        extra = f"  ... and {len(failures)-5} more" if len(failures) > 5 else ""
        pytest.fail(f"{len(failures)}/{n_trials} trials failed:\n" + "\n".join(lines) + extra)


# --- parameter generation (fixed seed → reproducible) ---

_rng_gen = np.random.default_rng(0xDEADBEEF)

_LARGE_IDEAL_CASES = [(seed, _specs(int(_rng_gen.integers(1, 4)), 1500, seed)) for seed in range(N_CASES)]
_BURST_IDEAL_CASES = [(seed, _specs(int(_rng_gen.integers(1, 9)), 1500, seed, MOD_SCHEMES_ROBUST)) for seed in range(N_CASES)]

_LARGE_CHANNEL_CASES = [
    (seed, snr, _specs(int(_rng_gen.integers(1, 4)), 1500, seed, MOD_SCHEMES_ROBUST))
    for snr in [15, 18, 20, 25, 30]
    for seed in range(N_CASES)
]
_BURST_CHANNEL_CASES = [
    (seed, snr, _specs(int(_rng_gen.integers(1, 6)), 1500, seed, MOD_SCHEMES_ROBUST))
    for snr in [18, 20, 25, 30]
    for seed in range(N_CASES // 2)
]

_BURSTY_CASES = []
for _seed in range(N_CASES // 2):
    _r = np.random.default_rng(_seed + 1234)
    _n_groups = int(_r.integers(2, 5))
    _groups = [
        [(0, int(_r.integers(10, 1501)), MOD_SCHEMES_ROBUST[int(_r.integers(0, 2))])
         for _ in range(int(_r.integers(1, 4)))]
        for _ in range(_n_groups)
    ]
    _silences = [int(_r.integers(1000, 10001)) for _ in range(_n_groups - 1)]
    _BURSTY_CASES.append((_seed, _groups, _silences))


# --- tests ---

@pytest.mark.parametrize("seed,specs", _LARGE_IDEAL_CASES)
def test_large_packet_ideal(seed, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


@pytest.mark.parametrize("seed,specs", _BURST_IDEAL_CASES)
def test_burst_ideal(seed, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


@pytest.mark.parametrize("seed,snr_db,specs", _LARGE_CHANNEL_CASES)
def test_large_packet_channel(seed, snr_db, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS, seed)


@pytest.mark.parametrize("seed,snr_db,specs", _BURST_CHANNEL_CASES)
def test_burst_channel(seed, snr_db, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS, seed)


@pytest.mark.parametrize("seed,groups,silences", _BURSTY_CASES)
def test_bursty_with_gaps(seed, groups, silences):
    tx_packets, signal = _make_bursty_signal(groups, silences, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


@pytest.mark.parametrize("seed,groups,silences", _BURSTY_CASES)
@pytest.mark.parametrize("snr_db", [20, 25, 30])
def test_bursty_with_gaps_channel(snr_db, seed, groups, silences):
    tx_packets, signal = _make_bursty_signal(groups, silences, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS // 2, seed + snr_db)
