"""Stress tests: large payloads (1–1500 bytes), variable burst sizes, bursty-with-gaps.

Each channel test runs N_TRIALS noise realizations internally so the suite
actually takes time proportional to the number of (packet, channel) combinations
tested — not just the number of pytest items.

Tune N_TRIALS and N_CASES at the top to control runtime.

Rough timing guide (measured on dev machine):
  N_TRIALS=50,  N_CASES=100  → ~5 min
  N_TRIALS=100, N_CASES=100  → ~10 min
  N_TRIALS=200, N_CASES=100  → ~20 min

Run:
    nohup .venv/bin/python -m pytest tests/test_stress.py -v --tb=short > test_results.log 2>&1 &
    tail -f test_results.log
"""

import numpy as np
import pytest

from modules.channel import ChannelConfig, ChannelModel
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline

# ---------------------------------------------------------------------------
# Knobs — tune these to control total runtime
# ---------------------------------------------------------------------------
N_TRIALS = 200     # noise realizations per (packet-layout, SNR) combination
N_CASES = 100      # distinct packet-layout seeds per test group

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]
MOD_SCHEMES_ROBUST = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _specs(n_packets: int, max_length: int, seed: int) -> list[tuple[int, int, ModulationSchemes]]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, max_length + 1, n_packets)
    mods = [MOD_SCHEMES[int(i)] for i in rng.integers(0, len(MOD_SCHEMES), n_packets)]
    return [(i, int(lengths[i]), mods[i]) for i in range(n_packets)]


def _specs_robust(n_packets: int, max_length: int, seed: int) -> list[tuple[int, int, ModulationSchemes]]:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, max_length + 1, n_packets)
    mods = [MOD_SCHEMES_ROBUST[int(i)] for i in rng.integers(0, len(MOD_SCHEMES_ROBUST), n_packets)]
    return [(i, int(lengths[i]), mods[i]) for i in range(n_packets)]


def _make_signal(specs: list, seed: int) -> tuple[list, np.ndarray]:
    rng = np.random.default_rng(seed + 999_999)
    tx_packets, parts = [], []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((pkt, config))
        parts.append(TXPipeline(config).transmit(pkt))
    return tx_packets, np.concatenate(parts)


def _make_bursty_signal(
    groups: list[list[tuple[int, int, ModulationSchemes]]],
    silence_samples: list[int],
    seed: int,
) -> tuple[list, np.ndarray]:
    """Groups of packets separated by silence gaps (zero samples)."""
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


def _assert_all_received(tx_packets: list, rx_packets: list) -> None:
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"seq_num={seq_num} not received"
        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid, f"seq_num={seq_num} marked invalid"
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload), f"seq_num={seq_num} payload mismatch"


def _run_trials(tx_packets, signal, config, snr_db, n_trials, base_seed):
    """Apply N_TRIALS different noise+CFO+phase realizations and assert all pass.

    Reports all failures at once rather than stopping at the first.
    """
    failures = []
    for i in range(n_trials):
        rng = np.random.default_rng(base_seed * 10_007 + i)
        cfo_hz = float(rng.uniform(-5000, 5000))
        phase = float(rng.uniform(0, 2 * np.pi))
        ch = ChannelModel(ChannelConfig(
            sample_rate=config.SAMPLE_RATE,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            initial_phase_rad=phase,
            seed=int(rng.integers(0, 2**31)),
        ))
        noisy = ch.apply(signal)
        rx_packets, _ = RXPipeline(config).receive(noisy)
        try:
            _assert_all_received(tx_packets, rx_packets)
        except AssertionError as e:
            failures.append((i, cfo_hz, phase, str(e)))

    if failures:
        lines = [f"  trial={t}, cfo={c:.0f}Hz, phase={p:.2f}: {e}" for t, c, p, e in failures[:5]]
        extra = f"  ... and {len(failures)-5} more" if len(failures) > 5 else ""
        pytest.fail(f"{len(failures)}/{n_trials} trials failed:\n" + "\n".join(lines) + extra)


# ---------------------------------------------------------------------------
# Parameter generation (fixed seeds → reproducible)
# ---------------------------------------------------------------------------

_rng_gen = np.random.default_rng(0xDEADBEEF)

_LARGE_IDEAL_CASES = [
    (seed, _specs(int(_rng_gen.integers(1, 4)), 1500, seed))
    for seed in range(N_CASES)
]

_BURST_IDEAL_CASES = [
    (seed, _specs_robust(int(_rng_gen.integers(1, 9)), 1500, seed))
    for seed in range(N_CASES)
]

_CHANNEL_SNRS_NORMAL = [15, 18, 20, 25, 30]
_CHANNEL_SNRS_HARD   = [10, 12]

_LARGE_CHANNEL_CASES = [
    (seed, snr, _specs_robust(int(_rng_gen.integers(1, 4)), 1500, seed))
    for snr in _CHANNEL_SNRS_NORMAL
    for seed in range(N_CASES)
]

_BURST_CHANNEL_CASES = [
    (seed, snr, _specs_robust(int(_rng_gen.integers(1, 6)), 1500, seed))
    for snr in [18, 20, 25, 30]
    for seed in range(N_CASES // 2)
]

_BURSTY_CASES = []
for _seed in range(N_CASES // 2):
    _r = np.random.default_rng(_seed + 1234)
    _n_groups = int(_r.integers(2, 5))
    _groups = [
        [
            (0, int(_r.integers(10, 1501)), MOD_SCHEMES_ROBUST[int(_r.integers(0, 2))])
            for _ in range(int(_r.integers(1, 4)))
        ]
        for _ in range(_n_groups)
    ]
    _silences = [int(_r.integers(1000, 10001)) for _ in range(_n_groups - 1)]
    _BURSTY_CASES.append((_seed, _groups, _silences))


# ---------------------------------------------------------------------------
# Test: large packets, ideal channel (no noise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,specs", _LARGE_IDEAL_CASES)
def test_large_packet_ideal(seed, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


# ---------------------------------------------------------------------------
# Test: variable burst (1–8 packets), ideal channel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,specs", _BURST_IDEAL_CASES)
def test_burst_ideal(seed, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


# ---------------------------------------------------------------------------
# Test: large packets × N_TRIALS noise/CFO/phase realizations per case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,snr_db,specs", _LARGE_CHANNEL_CASES)
def test_large_packet_channel(seed, snr_db, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS, seed)


# ---------------------------------------------------------------------------
# Test: burst × N_TRIALS noise/CFO/phase realizations per case
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,snr_db,specs", _BURST_CHANNEL_CASES)
def test_burst_channel(seed, snr_db, specs):
    tx_packets, signal = _make_signal(specs, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS, seed)


# ---------------------------------------------------------------------------
# Test: bursty-with-gaps — groups of packets separated by silence, ideal channel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,groups,silences", _BURSTY_CASES)
def test_bursty_with_gaps(seed, groups, silences):
    tx_packets, signal = _make_bursty_signal(groups, silences, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    _assert_all_received(tx_packets, rx_packets)


# ---------------------------------------------------------------------------
# Test: bursty-with-gaps × N_TRIALS channel realizations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed,groups,silences", _BURSTY_CASES)
@pytest.mark.parametrize("snr_db", [20, 25, 30])
def test_bursty_with_gaps_channel(snr_db, seed, groups, silences):
    tx_packets, signal = _make_bursty_signal(groups, silences, seed)
    _, config = tx_packets[0]
    _run_trials(tx_packets, signal, config, snr_db, N_TRIALS // 2, seed + snr_db)

if __name__ == "__main__":
    import numpy as np
    from modules.pipeline import *
    from modules.channel import *
    from utils.plotting import *

    seed = 30
    snr_db = 20
    trial = 37

    # reconstruct groups/silences for seed=30
    _r = np.random.default_rng(seed + 1234)
    n_groups = int(_r.integers(2, 5))
    groups = [
        [
            (0, int(_r.integers(10, 1501)), MOD_SCHEMES_ROBUST[int(_r.integers(0, 2))])
            for _ in range(int(_r.integers(1, 4)))
        ]
        for _ in range(n_groups)
    ]
    silences = [int(_r.integers(1000, 10001)) for _ in range(n_groups - 1)]

    tx_packets, signal = _make_bursty_signal(groups, silences, seed)
    _, config = tx_packets[0]

    rng = np.random.default_rng((seed + snr_db) * 10_007 + trial)
    cfo_hz = float(rng.uniform(-5000, 5000))
    phase  = float(rng.uniform(0, 2 * np.pi))
    ch_seed = int(rng.integers(0, 2**31))

    print(f"Groups: {n_groups}, total packets: {len(tx_packets)}")
    print(f"CFO={cfo_hz:.0f}Hz  phase={phase:.3f}  ch_seed={ch_seed}")

    ch = ChannelModel(ChannelConfig(
      sample_rate=config.SAMPLE_RATE,
      snr_db=snr_db,
      cfo_hz=cfo_hz,
      initial_phase_rad=phase,
      seed=ch_seed,
      ))
    rx_sig = ch.apply(signal)
    rx_packets, _ = RXPipeline(config).receive(rx_sig)

    for packet, _ in tx_packets:
        print(packet.length)
    # plot_iq(rx_sig)
    # plt.show()

    rx_seqs = {p.seq_num for p in rx_packets}
    tx_seqs = {p.seq_num for p, _ in tx_packets}
    missed = tx_seqs - rx_seqs
    print(f"TX seq_nums: {sorted(tx_seqs)}")
    print(f"RX seq_nums: {sorted(rx_seqs)}")
    print(f"Missed: {missed}")
