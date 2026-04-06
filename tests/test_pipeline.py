import pytest
import numpy as np
import matplotlib.pyplot as plt
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.pipeline import *
from modules.frame_constructor import ModulationSchemes
from modules.channel import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]

@composite
def packet_specs(draw):
    n = draw(st.integers(1, 4))
    lengths = draw(st.lists(st.integers(6, 10), min_size=n, max_size=n))
    mods = draw(st.lists(st.sampled_from(MOD_SCHEMES), min_size=n, max_size=n))
    return list(zip(range(n), lengths, mods))

def make_packets_and_signal(specs, seed=None):
    rng = np.random.default_rng(seed)
    tx_packets = []
    signal_parts = []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        tx = TXPipeline(config)
        bits = rng.integers(0, 2, (length * 8)).reshape(-1, 1)
        packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((packet, config))
        signal_parts.append(tx.transmit(packet))
    return tx_packets, np.concatenate(signal_parts)

def assert_packets(tx_packets, rx_packets):
    """Strict: no false detections, every TX packet received with correct content."""
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    for rx_pkt in rx_packets:
        assert rx_pkt.seq_num in tx_by_seq
        assert rx_pkt.valid
        tx_pkt = tx_by_seq[rx_pkt.seq_num]
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)

def assert_all_received(tx_packets, rx_packets):
    """Channel-aware: all TX packets received with correct content; false detections tolerated."""
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"packet seq_num={seq_num} not received"
        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)

# --- Detection rate tests ---

FA_TRIALS = 50
FA_BUFFER_LENGTH = 2**15
FA_THRESHOLD = 0.05  # max fraction of buffers allowed to have a false alarm

def test_false_alarm_on_noise():
    rx = RXPipeline(PipelineConfig())
    rng = np.random.default_rng(0)
    false_alarms = sum(
        len(rx.receive((rng.standard_normal(FA_BUFFER_LENGTH) + 1j * rng.standard_normal(FA_BUFFER_LENGTH)) / np.sqrt(2)))
        for _ in range(FA_TRIALS)
    )
    fa_rate = false_alarms / FA_TRIALS
    print(f"\nFA rate (noise only): {fa_rate:.3f} detections/buffer ({FA_TRIALS} trials)")
    assert fa_rate < FA_THRESHOLD

def test_overdetection_on_signal():
    rng = np.random.default_rng(0)
    specs = [(i, 32, ModulationSchemes.QPSK) for i in range(4)]
    tx_packets, signal = make_packets_and_signal(specs, seed=0)
    _, config = tx_packets[0]
    rx = RXPipeline(config)

    overdetections = 0
    for _ in range(FA_TRIALS):
        noise_signal = signal + (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))) / np.sqrt(2) * 0.1
        rx_packets = rx.receive(noise_signal)
        overdetections += max(0, len(rx_packets) - len(tx_packets))

    od_rate = overdetections / FA_TRIALS
    print(f"\nOverdetection rate: {od_rate:.3f} extra detections/buffer ({FA_TRIALS} trials)")
    assert od_rate < FA_THRESHOLD

# --- Integration tests ---

@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(specs=packet_specs(), seed=st.integers(0, 2**31))
def test_ideal(specs, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets = RXPipeline(config).receive(signal)
    assert_packets(tx_packets, rx_packets)
    assert len(rx_packets) == len(tx_packets)

@pytest.mark.parametrize("snr_db", [15, 16, 17.5, 18.5, 20, 25, 30])
@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(
    specs=packet_specs(),
    cfo_hz=st.floats(-5000, 5000, allow_nan=False, allow_infinity=False),
    phase=st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False),
    seed=st.integers(0, 2**31),
)
def test_channel(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        seed=seed,
    ))

    if snr_db < 19:
        for i in specs:
            if i[2] == ModulationSchemes.PSK8:
                pytest.xfail("8PSK requires ~18dB SNR; below this threshold decoding is unreliable")

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)

@pytest.mark.xfail(strict=False, reason="sub-15 dB SNR may not decode reliably")
@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(
    specs=packet_specs(),
    cfo_hz=st.floats(-5000, 5000, allow_nan=False, allow_infinity=False),
    phase=st.floats(0, 2 * np.pi, allow_nan=False, allow_infinity=False),
    snr_db=st.floats(5, 14, allow_nan=False, allow_infinity=False),
    seed=st.integers(0, 2**31),
)
def test_hard_channel(specs, cfo_hz, phase, snr_db, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        seed=seed,
    ))
    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)


# --- BER Report ---

SNR_SWEEP = [5, 8, 10, 12, 15, 18, 20, 25, 30]
REPORT_LENGTH = 32
N_TRIALS = 10

def _trial(mod, snr_db, cfo_hz):
    rng = np.random.default_rng()
    config = PipelineConfig(MOD_SCHEME=mod)
    bits = rng.integers(0, 2, REPORT_LENGTH * 8).reshape(-1, 1)
    packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=REPORT_LENGTH, payload=bits)
    signal = TXPipeline(config).transmit(packet)

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=rng.uniform(0, 2 * np.pi),
    ))
    rx_packets = RXPipeline(config).receive(channel.apply(signal))

    match = next((p for p in rx_packets if p.seq_num == 0), None)
    if match is None or match.payload.shape != packet.payload.shape:
        return None, False
    ber = float(np.mean(match.payload != packet.payload))
    return ber, True

def ber_report():
    CFO_CONDITIONS = [("no CFO", 0), ("CFO 2.5kHz", 2500), ("CFO 5kHz", 5000)]
    # Smallest BER detectable given trial count and packet size
    BER_FLOOR = 0.5 / (N_TRIALS * REPORT_LENGTH * 8)

    # Run all trials once and store per CFO condition
    all_results = {}
    for cfo_label, cfo_hz in CFO_CONDITIONS:
        results = {}
        for mod in MOD_SCHEMES:
            snr_bers, snr_pers = [], []
            for snr_db in SNR_SWEEP:
                trials = [_trial(mod, snr_db, cfo_hz) for _ in range(N_TRIALS)]
                decoded_bers = [b for b, ok in trials if ok]
                per = 1 - len(decoded_bers) / N_TRIALS
                mean_ber = np.mean(decoded_bers) if decoded_bers else float("nan")
                snr_bers.append(mean_ber)
                snr_pers.append(per)
            results[mod] = (snr_bers, snr_pers)
        all_results[cfo_label] = (cfo_hz, results)

    # Print table
    for cfo_label, (_, results) in all_results.items():
        print(f"\n=== {cfo_label} ===")
        print(f"{'SNR':>6} | {'BPSK BER':>10} {'BPSK PER':>10} | {'QPSK BER':>10} {'QPSK PER':>10} | {'8PSK BER':>10} {'8PSK PER':>10}")
        print("-" * 80)
        for i, snr in enumerate(SNR_SWEEP):
            row = f"{snr:>6}"
            for mod in MOD_SCHEMES:
                bers, pers = results[mod]
                ber_str = f"{bers[i]:.2e}" if not np.isnan(bers[i]) else "   N/A"
                row += f" | {ber_str:>10} {pers[i]:>10.4f}"
            print(row)

    # Plot
    fig, axes = plt.subplots(len(CFO_CONDITIONS), 2, figsize=(12, 4 * len(CFO_CONDITIONS)), sharex=True)
    for row_idx, (cfo_label, (_, results)) in enumerate(all_results.items()):
        ax_ber, ax_per = axes[row_idx]

        for mod in MOD_SCHEMES:
            bers, pers = results[mod]
            # Split into points where BER was measured vs where no packet decoded
            xs_measured, ys_measured = [], []
            xs_zero, ys_zero = [], []
            for snr, ber in zip(SNR_SWEEP, bers):
                if np.isnan(ber):
                    continue
                if ber == 0.0:
                    xs_zero.append(snr)
                    ys_zero.append(BER_FLOOR)
                else:
                    xs_measured.append(snr)
                    ys_measured.append(ber)

            color = None
            if xs_measured:
                line, = ax_ber.semilogy(xs_measured, ys_measured, marker='o', label=mod.name)
                color = line.get_color()
            if xs_zero:
                ax_ber.semilogy(xs_zero, ys_zero, marker='v', linestyle='none',
                                color=color, label=f"{mod.name} (0 errors)" if not xs_measured else None)

            ax_per.plot(SNR_SWEEP, pers, marker='o', label=mod.name)

        ax_ber.axhline(BER_FLOOR, color='grey', linestyle=':', linewidth=0.8, label=f"floor (1/{N_TRIALS*REPORT_LENGTH*8} bits)")
        ax_ber.set_title(f"BER — {cfo_label}")
        ax_ber.set_ylabel("BER")
        ax_ber.legend(fontsize=8)
        ax_ber.grid(True, which="both", ls="--", alpha=0.5)

        ax_per.set_title(f"PER — {cfo_label}")
        ax_per.set_ylabel("PER")
        ax_per.legend()
        ax_per.grid(True, ls="--", alpha=0.5)

    for ax in axes[-1]:
        ax.set_xlabel("SNR (dB)")

    plt.tight_layout()
    plt.savefig("tests/plots/pipeline/ber_report.png", dpi=150)
    print("\nPlot saved to tests/plots/pipeline/ber_report.png")

def replay_scenario(specs, cfo_hz, phase, snr_db, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        seed=seed,
    ))

    rx_signal_sim = channel.apply(signal)
    rx_signal = rx_signal_sim# np.load("pluto/plots/rx_raw_0.npy")

    rx_packets = RXPipeline(config).receive(rx_signal)
    assert_all_received(tx_packets, rx_packets)

if __name__ == "__main__":
    # ber_report()
    replay_scenario(
        snr_db=20, # fails at 18
        specs=[(0, 6, ModulationSchemes.BPSK),
        (1, 6, ModulationSchemes.PSK8),
        (2, 6, ModulationSchemes.BPSK),
        (3, 8, ModulationSchemes.BPSK)],
        cfo_hz=0.0,
        phase=0.0,
        seed=175,
    )
