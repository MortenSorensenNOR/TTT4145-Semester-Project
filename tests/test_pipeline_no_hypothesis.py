import pytest
import numpy as np
import matplotlib.pyplot as plt

from modules.pipeline import *
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.channel import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]


def make_packets_and_signal(specs, seed=None):
    rng = np.random.default_rng(seed)
    tx_packets = []
    signal_parts = []

    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        tx = TXPipeline(config)

        bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
        packet = Packet(
            src_mac=0,
            dst_mac=1,
            type=0,
            seq_num=seq_num,
            length=length,
            payload=bits,
        )

        tx_packets.append((packet, config))
        signal_parts.append(tx.transmit(packet))

    return tx_packets, np.concatenate(signal_parts)


def assert_packets(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}

    assert len(rx_packets) == len(tx_packets), (
        f"Expected {len(tx_packets)} packets, got {len(rx_packets)}"
    )

    for rx_pkt in rx_packets:
        assert rx_pkt.seq_num in tx_by_seq, f"Unexpected packet seq_num={rx_pkt.seq_num}"
        assert rx_pkt.valid, f"Packet seq_num={rx_pkt.seq_num} is marked invalid"

        tx_pkt = tx_by_seq[rx_pkt.seq_num]
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)


def assert_all_received(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}

    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"packet seq_num={seq_num} not received"

        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid, f"packet seq_num={seq_num} is invalid"
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)


def diagnose_and_assert(tx_packets, rx_packets):
    """Like assert_all_received, but prints a failure breakdown before asserting.

    Categorises failures into three buckets:
      - not_detected : sync missed entirely (counts against PER)
      - invalid      : detected but header/CRC marked packet invalid (counts against PER)
      - payload_err  : detected and valid header, but payload bits wrong (counts against BER)
    """
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}

    not_detected, invalid, payload_err = [], [], []

    for seq_num, tx_pkt in tx_by_seq.items():
        if seq_num not in rx_by_seq:
            not_detected.append(seq_num)
            continue
        rx_pkt = rx_by_seq[seq_num]
        if not rx_pkt.valid:
            invalid.append(seq_num)
            continue
        if rx_pkt.length != tx_pkt.length or not np.array_equal(rx_pkt.payload, tx_pkt.payload):
            if rx_pkt.payload.shape == tx_pkt.payload.shape:
                ber = float(np.mean(rx_pkt.payload != tx_pkt.payload))
            else:
                ber = 1.0
            payload_err.append((seq_num, ber))

    if not_detected or invalid or payload_err:
        total = len(tx_by_seq)
        parts = [f"{total} tx"]
        if not_detected:
            parts.append(f"not_detected={len(not_detected)}")
        if invalid:
            parts.append(f"invalid={len(invalid)}")
        if payload_err:
            mean_ber = float(np.mean([b for _, b in payload_err]))
            parts.append(f"payload_err={len(payload_err)} (BER={mean_ber:.2e})")
        print(f"\n  MULTIPATH DIAG: {', '.join(parts)}")

    assert_all_received(tx_packets, rx_packets)


# ----------------------------
# Detection tests
# ----------------------------

FA_TRIALS = 50
FA_BUFFER_LENGTH = 2**15
FA_THRESHOLD = 0.05


def test_false_alarm_on_noise():
    rx = RXPipeline(PipelineConfig())
    rng = np.random.default_rng(0)

    false_alarms = 0
    for _ in range(FA_TRIALS):
        noise = (
            rng.standard_normal(FA_BUFFER_LENGTH)
            + 1j * rng.standard_normal(FA_BUFFER_LENGTH)
        ) / np.sqrt(2)

        rx_packets = rx.receive(noise)
        false_alarms += len(rx_packets)

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
        noise = (
            rng.standard_normal(len(signal))
            + 1j * rng.standard_normal(len(signal))
        ) / np.sqrt(2) * 0.1

        noisy_signal = signal + noise
        rx_packets = rx.receive(noisy_signal)
        overdetections += max(0, len(rx_packets) - len(tx_packets))

    od_rate = overdetections / FA_TRIALS
    print(f"\nOverdetection rate: {od_rate:.3f} extra detections/buffer ({FA_TRIALS} trials)")
    assert od_rate < FA_THRESHOLD


# ----------------------------
# Ideal channel tests
# ----------------------------

@pytest.mark.parametrize(
    "specs, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1),
        ([(0, 6, ModulationSchemes.BPSK), (1, 10, ModulationSchemes.BPSK)], 2),
        ([(0, 6, ModulationSchemes.QPSK), (1, 8, ModulationSchemes.QPSK)], 3),
        (
            [
                (0, 6, ModulationSchemes.BPSK),
                (1, 8, ModulationSchemes.QPSK),
                (2, 10, ModulationSchemes.BPSK),
            ],
            4,
        ),
    ],
)
def test_ideal(specs, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    rx_packets = RXPipeline(config).receive(signal)
    assert_packets(tx_packets, rx_packets)


# ----------------------------
# Channel tests
# ----------------------------

@pytest.mark.parametrize("snr_db", [15, 20, 25, 30])
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1000.0, 0.5, 1),
        (
            [(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],
            -2500.0,
            1.0,
            2,
        ),
        (
            [
                (0, 10, ModulationSchemes.QPSK),
                (1, 6, ModulationSchemes.BPSK),
                (2, 8, ModulationSchemes.QPSK),
            ],
            5000.0,
            2.0,
            3,
        ),
    ],
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

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)


# ITU-R M.1225 channel profiles at 5.336 MHz sample rate
# Pedestrian speed (~3 km/h at 433 MHz) → Doppler ≈ 1.2 Hz
# Vehicular speed (~120 km/h at 433 MHz) → Doppler ≈ 48 Hz
PEDESTRIAN_DOPPLER_HZ = np.float32(1.2)
VEHICULAR_DOPPLER_HZ = np.float32(48.0)

# Pedestrian A — tap delays (ns): 0, 110, 190, 410  →  ~0–2 samples spread
PEDESTRIAN_A_DELAYS = (np.float32(0.0), np.float32(0.587), np.float32(1.014), np.float32(2.188))
PEDESTRIAN_A_GAINS_DB = (np.float32(0.0), np.float32(-9.7), np.float32(-19.2), np.float32(-22.8))

# Pedestrian B — tap delays (ns): 0, 200, 800, 1200, 2300, 3700  →  ~0–20 samples spread
PEDESTRIAN_B_DELAYS = (
    np.float32(0.0), np.float32(1.067), np.float32(4.269),
    np.float32(6.403), np.float32(12.273), np.float32(19.743),
)
PEDESTRIAN_B_GAINS_DB = (
    np.float32(0.0), np.float32(-0.9), np.float32(-4.9),
    np.float32(-8.0), np.float32(-7.8), np.float32(-23.9),
)

# Vehicular A — tap delays (ns): 0, 310, 710, 1090, 1730, 2510  →  ~0–13 samples spread
VEHICULAR_A_DELAYS = (
    np.float32(0.0), np.float32(1.654), np.float32(3.789),
    np.float32(5.816), np.float32(9.227), np.float32(13.393),
)
VEHICULAR_A_GAINS_DB = (
    np.float32(0.0), np.float32(-1.0), np.float32(-9.0),
    np.float32(-10.0), np.float32(-15.0), np.float32(-20.0),
)

# Vehicular B — tap delays (ns): 0, 300, 8900, 12900, 17100, 20000  →  ~0–107 samples spread
VEHICULAR_B_DELAYS = (
    np.float32(0.0), np.float32(1.601), np.float32(47.490),
    np.float32(68.834), np.float32(91.246), np.float32(106.720),
)
VEHICULAR_B_GAINS_DB = (
    np.float32(-2.5), np.float32(0.0), np.float32(-12.8),
    np.float32(-10.0), np.float32(-25.2), np.float32(-16.0),
)


@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1000.0, 0.5, 1),
        (
            [(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],
            -2500.0,
            1.0,
            2,
        ),
    ],
)
def test_channel_multipath(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=PEDESTRIAN_A_DELAYS,
        multipath_gains_db=PEDESTRIAN_A_GAINS_DB,
        doppler_hz=PEDESTRIAN_DOPPLER_HZ,
        fading_type="rayleigh",
        seed=seed,
    ))

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Pedestrian B: ~20-sample delay spread stresses receiver without equalizer")
@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1000.0, 0.5, 1),
        (
            [(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],
            -2500.0,
            1.0,
            2,
        ),
    ],
)
def test_channel_pedestrian_b(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=PEDESTRIAN_B_DELAYS,
        multipath_gains_db=PEDESTRIAN_B_GAINS_DB,
        doppler_hz=PEDESTRIAN_DOPPLER_HZ,
        fading_type="rayleigh",
        seed=seed,
    ))

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Vehicular A: fast fading (~48 Hz Doppler) within packet duration")
@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1000.0, 0.5, 1),
        (
            [(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],
            -2500.0,
            1.0,
            2,
        ),
    ],
)
def test_channel_vehicular_a(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=VEHICULAR_A_DELAYS,
        multipath_gains_db=VEHICULAR_A_GAINS_DB,
        doppler_hz=VEHICULAR_DOPPLER_HZ,
        fading_type="rayleigh",
        seed=seed,
    ))

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Vehicular B: ~107-sample delay spread and fast fading — no equalizer")
@pytest.mark.parametrize("snr_db", [25, 30])
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 1000.0, 0.5, 1),
    ],
)
def test_channel_vehicular_b(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]

    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=VEHICULAR_B_DELAYS,
        multipath_gains_db=VEHICULAR_B_GAINS_DB,
        doppler_hz=VEHICULAR_DOPPLER_HZ,
        fading_type="rayleigh",
        seed=seed,
    ))

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="sub-15 dB SNR may not decode reliably")
@pytest.mark.parametrize(
    "specs, cfo_hz, phase, snr_db, seed",
    [
        ([(0, 6, ModulationSchemes.BPSK)], 0.0, 0.0, 12.0, 0),
        ([(0, 8, ModulationSchemes.QPSK)], 2500.0, 1.0, 10.0, 1),
        (
            [(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],
            -5000.0,
            2.0,
            8.0,
            2,
        ),
    ],
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


# ----------------------------
# BER report
# ----------------------------

SNR_SWEEP = [5, 8, 10, 12, 15, 18, 20, 25, 30]
REPORT_LENGTH = 32
N_TRIALS = 10


def _trial(mod, snr_db, cfo_hz):
    rng = np.random.default_rng()
    config = PipelineConfig(MOD_SCHEME=mod)

    bits = rng.integers(0, 2, REPORT_LENGTH * 8).reshape(-1, 1)
    packet = Packet(
        src_mac=0,
        dst_mac=1,
        type=0,
        seq_num=0,
        length=REPORT_LENGTH,
        payload=bits,
    )

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


@pytest.mark.skip(reason="report generator, not part of normal unit test run")
def test_generate_ber_report():
    ber_report()


def ber_report():
    cfo_conditions = [("no CFO", 0), ("CFO 2.5kHz", 2500), ("CFO 5kHz", 5000)]
    ber_floor = 0.5 / (N_TRIALS * REPORT_LENGTH * 8)

    all_results = {}

    for cfo_label, cfo_hz in cfo_conditions:
        results = {}

        for mod in MOD_SCHEMES:
            snr_bers = []
            snr_pers = []

            for snr_db in SNR_SWEEP:
                trials = [_trial(mod, snr_db, cfo_hz) for _ in range(N_TRIALS)]
                decoded_bers = [b for b, ok in trials if ok]

                per = 1 - len(decoded_bers) / N_TRIALS
                mean_ber = np.mean(decoded_bers) if decoded_bers else float("nan")

                snr_bers.append(mean_ber)
                snr_pers.append(per)

            results[mod] = (snr_bers, snr_pers)

        all_results[cfo_label] = results

    for cfo_label, results in all_results.items():
        print(f"\n=== {cfo_label} ===")
        print(f"{'SNR':>6} | {'BPSK BER':>10} {'BPSK PER':>10} | {'QPSK BER':>10} {'QPSK PER':>10}")
        print("-" * 62)

        for i, snr in enumerate(SNR_SWEEP):
            row = f"{snr:>6}"
            for mod in MOD_SCHEMES:
                bers, pers = results[mod]
                ber_str = f"{bers[i]:.2e}" if not np.isnan(bers[i]) else "   N/A"
                row += f" | {ber_str:>10} {pers[i]:>10.4f}"
            print(row)

    fig, axes = plt.subplots(len(cfo_conditions), 2, figsize=(12, 4 * len(cfo_conditions)), sharex=True)

    for row_idx, (cfo_label, results) in enumerate(all_results.items()):
        ax_ber, ax_per = axes[row_idx]

        for mod in MOD_SCHEMES:
            bers, pers = results[mod]

            xs_measured, ys_measured = [], []
            xs_zero, ys_zero = [], []

            for snr, ber in zip(SNR_SWEEP, bers):
                if np.isnan(ber):
                    continue
                if ber == 0.0:
                    xs_zero.append(snr)
                    ys_zero.append(ber_floor)
                else:
                    xs_measured.append(snr)
                    ys_measured.append(ber)

            color = None
            if xs_measured:
                line, = ax_ber.semilogy(xs_measured, ys_measured, marker="o", label=mod.name)
                color = line.get_color()

            if xs_zero:
                ax_ber.semilogy(
                    xs_zero,
                    ys_zero,
                    marker="v",
                    linestyle="none",
                    color=color,
                    label=f"{mod.name} (0 errors)" if not xs_measured else None,
                )

            ax_per.plot(SNR_SWEEP, pers, marker="o", label=mod.name)

        ax_ber.axhline(
            ber_floor,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            label=f"floor (1/{N_TRIALS * REPORT_LENGTH * 8} bits)",
        )
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

    rx_packets = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)


if __name__ == "__main__":
    replay_scenario(
        snr_db=15,
        specs=[(0, 6, ModulationSchemes.BPSK)],
        cfo_hz=0.0,
        phase=2.0,
        seed=0,
    )
