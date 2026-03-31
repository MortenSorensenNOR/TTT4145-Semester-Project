import pytest
import numpy as np
import matplotlib.pyplot as plt
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.pipeline import *
from modules.frame_constructor import ModulationSchemes
from modules.channel import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]

@composite
def packet_specs(draw):
    n = draw(st.integers(1, 4))
    lengths = draw(st.lists(st.integers(6, 10), min_size=n, max_size=n))
    mods = draw(st.lists(st.sampled_from(MOD_SCHEMES), min_size=n, max_size=n))
    return list(zip(range(n), lengths, mods))

def make_packets_and_signal(specs):
    tx_packets = []
    signal_parts = []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        tx = TXPipeline(config)
        bits = np.random.default_rng().integers(0, 2, length * 8).reshape(-1, mod.value + 1)
        packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((packet, config))
        signal_parts.append(tx.transmit(packet))
    return tx_packets, np.concat(signal_parts)

def assert_packets(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    for rx_pkt in rx_packets:
        assert rx_pkt.seq_num in tx_by_seq
        tx_pkt = tx_by_seq[rx_pkt.seq_num]
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)

# --- Tests ---

@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(specs=packet_specs())
def test_ideal(specs):
    tx_packets, signal = make_packets_and_signal(specs)
    _, config = tx_packets[0]
    rx_packets = RXPipeline(config).receive(signal)
    assert_packets(tx_packets, rx_packets)
    assert len(rx_packets) == len(tx_packets)

@pytest.mark.parametrize("snr_db", [15, 20, 25, 30])
@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(
    specs=packet_specs(),
    cfo_hz=st.floats(-5000, 5000),
    phase=st.floats(0, np.pi),
)
def test_channel(snr_db, specs, cfo_hz, phase):
    tx_packets, signal = make_packets_and_signal(specs)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
    ))
    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_packets(tx_packets, rx_packets)
    assert len(rx_packets) == len(tx_packets)

@pytest.mark.xfail(strict=False, reason="sub-15 dB SNR may not decode reliably")
@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(
    specs=packet_specs(),
    cfo_hz=st.floats(-5000, 5000),
    phase=st.floats(0, np.pi),
    snr_db=st.floats(5, 14),
)
def test_hard_channel(specs, cfo_hz, phase, snr_db):
    tx_packets, signal = make_packets_and_signal(specs)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
    ))
    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_packets(tx_packets, rx_packets)
    assert len(rx_packets) == len(tx_packets)


# --- BER Report ---

SNR_SWEEP = [5, 8, 10, 12, 15, 18, 20, 25, 30]
REPORT_CFO = 4321
REPORT_LENGTH = 32
N_TRIALS = 10

def _trial(mod, snr_db, cfo_hz):
    rng = np.random.default_rng()
    config = PipelineConfig(MOD_SCHEME=mod)
    bits = rng.integers(0, 2, REPORT_LENGTH * 8).reshape(-1, mod.value + 1)
    packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=REPORT_LENGTH, payload=bits)
    signal = TXPipeline(config).transmit(packet)

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=rng.uniform(0, np.pi),
    ))
    rx_packets = RXPipeline(config).receive(channel.apply(signal))

    match = next((p for p in rx_packets if p.seq_num == 0), None)
    if match is None:
        return 0.5, False
    ber = float(np.mean(match.payload != packet.payload))
    return ber, True

def ber_report():
    results = {}
    for mod in MOD_SCHEMES:
        bers, pers = [], []
        for snr_db in SNR_SWEEP:
            trial_bers, decoded = zip(*[_trial(mod, snr_db, REPORT_CFO) for _ in range(N_TRIALS)])
            bers.append(np.mean(trial_bers))
            pers.append(1 - np.mean(decoded))
        results[mod] = (bers, pers)

    print(f"\n{'SNR':>6} | {'BPSK BER':>10} {'BPSK PER':>10} | {'QPSK BER':>10} {'QPSK PER':>10}")
    print("-" * 60)
    for i, snr in enumerate(SNR_SWEEP):
        row = f"{snr:>6}"
        for mod in MOD_SCHEMES:
            bers, pers = results[mod]
            row += f" | {bers[i]:>10.4f} {pers[i]:>10.4f}"
        print(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for mod in MOD_SCHEMES:
        bers, pers = results[mod]
        label = mod.name
        ax1.semilogy(SNR_SWEEP, [max(b, 1e-5) for b in bers], marker='o', label=label)
        ax2.plot(SNR_SWEEP, pers, marker='o', label=label)

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("BER")
    ax1.set_title(f"BER vs SNR (CFO={REPORT_CFO} Hz, N={N_TRIALS})")
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("PER")
    ax2.set_title(f"PER vs SNR (CFO={REPORT_CFO} Hz, N={N_TRIALS})")
    ax2.legend()
    ax2.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("tests/plots/pipeline/ber_report.png", dpi=150)
    print("\nPlot saved to tests/plots/pipeline/ber_report.png")


if __name__ == "__main__":
    ber_report()
