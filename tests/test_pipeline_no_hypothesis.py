import numpy as np
import pytest

from modules.pipeline import *
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.channel import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK]


def make_packets_and_signal(specs, seed=None):
    rng = np.random.default_rng(seed)
    tx_packets, signal_parts = [], []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((pkt, config))
        signal_parts.append(TXPipeline(config).transmit(pkt))
    return tx_packets, np.concatenate(signal_parts)


def assert_packets(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    assert len(rx_packets) == len(tx_packets)
    for rx_pkt in rx_packets:
        assert rx_pkt.seq_num in tx_by_seq
        assert rx_pkt.valid
        tx_pkt = tx_by_seq[rx_pkt.seq_num]
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)


def assert_all_received(tx_packets, rx_packets):
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"packet seq_num={seq_num} not received"
        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)


def diagnose_and_assert(tx_packets, rx_packets):
    """Like assert_all_received, but prints a per-bucket failure breakdown first."""
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
            ber = float(np.mean(rx_pkt.payload != tx_pkt.payload)) if rx_pkt.payload.shape == tx_pkt.payload.shape else 1.0
            payload_err.append((seq_num, ber))

    if not_detected or invalid or payload_err:
        parts = [f"{len(tx_by_seq)} tx"]
        if not_detected: parts.append(f"not_detected={len(not_detected)}")
        if invalid:      parts.append(f"invalid={len(invalid)}")
        if payload_err:
            mean_ber = float(np.mean([b for _, b in payload_err]))
            parts.append(f"payload_err={len(payload_err)} (BER={mean_ber:.2e})")
        print(f"\n  MULTIPATH DIAG: {', '.join(parts)}")

    assert_all_received(tx_packets, rx_packets)


# --- detection ---

FA_TRIALS = 50
FA_BUFFER_LENGTH = 2**15
FA_THRESHOLD = 0.05


def test_false_alarm_on_noise():
    rx = RXPipeline(PipelineConfig())
    rng = np.random.default_rng(0)
    false_alarms = 0
    for _ in range(FA_TRIALS):
        noise = (rng.standard_normal(FA_BUFFER_LENGTH) + 1j * rng.standard_normal(FA_BUFFER_LENGTH)) / np.sqrt(2)
        rx_packets, _ = rx.receive(noise)
        false_alarms += len(rx_packets)
    assert false_alarms / FA_TRIALS < FA_THRESHOLD


def test_overdetection_on_signal():
    rng = np.random.default_rng(0)
    specs = [(i, 32, ModulationSchemes.QPSK) for i in range(4)]
    tx_packets, signal = make_packets_and_signal(specs, seed=0)
    _, config = tx_packets[0]
    rx = RXPipeline(config)

    overdetections = 0
    for _ in range(FA_TRIALS):
        noise = (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))) / np.sqrt(2) * 0.1
        rx_packets, _ = rx.receive(signal + noise)
        overdetections += max(0, len(rx_packets) - len(tx_packets))
    assert overdetections / FA_TRIALS < FA_THRESHOLD


# --- ideal channel ---

IDEAL_CASES = [
    ([(0, 6, ModulationSchemes.BPSK)], 0),
    ([(0, 8, ModulationSchemes.QPSK)], 1),
    ([(0, 6, ModulationSchemes.BPSK), (1, 10, ModulationSchemes.BPSK)], 2),
    ([(0, 6, ModulationSchemes.QPSK), (1, 8, ModulationSchemes.QPSK)], 3),
    ([(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK), (2, 10, ModulationSchemes.BPSK)], 4),
]


@pytest.mark.parametrize("specs,seed", IDEAL_CASES)
def test_ideal(specs, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
    assert_packets(tx_packets, rx_packets)


# --- AWGN channel ---

CHANNEL_CASES = [
    ([(0, 6, ModulationSchemes.BPSK)],                                          0.0,    0.0, 0),
    ([(0, 8, ModulationSchemes.QPSK)],                                          1000.0, 0.5, 1),
    ([(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)],         -2500.0, 1.0, 2),
    ([(0, 10, ModulationSchemes.QPSK), (1, 6, ModulationSchemes.BPSK),
      (2, 8, ModulationSchemes.QPSK)],                                          5000.0, 2.0, 3),
]


@pytest.mark.parametrize("snr_db", [15, 20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", CHANNEL_CASES)
def test_channel(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase, seed=seed,
    ))
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)


# --- ITU-R M.1225 multipath profiles (sample rate 5.336 MHz) ---

PEDESTRIAN_DOPPLER_HZ = np.float32(1.2)
VEHICULAR_DOPPLER_HZ  = np.float32(48.0)

PEDESTRIAN_A_DELAYS   = (0.0, 0.587, 1.014, 2.188)
PEDESTRIAN_A_GAINS_DB = (0.0, -9.7, -19.2, -22.8)

PEDESTRIAN_B_DELAYS   = (0.0, 1.067, 4.269, 6.403, 12.273, 19.743)
PEDESTRIAN_B_GAINS_DB = (0.0, -0.9, -4.9, -8.0, -7.8, -23.9)

VEHICULAR_A_DELAYS    = (0.0, 1.654, 3.789, 5.816, 9.227, 13.393)
VEHICULAR_A_GAINS_DB  = (0.0, -1.0, -9.0, -10.0, -15.0, -20.0)

VEHICULAR_B_DELAYS    = (0.0, 1.601, 47.490, 68.834, 91.246, 106.720)
VEHICULAR_B_GAINS_DB  = (-2.5, 0.0, -12.8, -10.0, -25.2, -16.0)

MULTIPATH_CASES = [
    ([(0, 6, ModulationSchemes.BPSK)],                                  0.0, 0.0, 0),
    ([(0, 8, ModulationSchemes.QPSK)],                                  1000.0, 0.5, 1),
    ([(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)], -2500.0, 1.0, 2),
]


def _multipath_channel(config, snr_db, cfo_hz, phase, seed, delays, gains_db, doppler_hz):
    return ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=tuple(np.float32(d) for d in delays),
        multipath_gains_db=tuple(np.float32(g) for g in gains_db),
        doppler_hz=doppler_hz, fading_type="rayleigh", seed=seed,
    ))


@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES)
def test_channel_multipath(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = _multipath_channel(config, snr_db, cfo_hz, phase, seed,
                                 PEDESTRIAN_A_DELAYS, PEDESTRIAN_A_GAINS_DB, PEDESTRIAN_DOPPLER_HZ)
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Pedestrian B: ~20-sample delay spread stresses receiver without equalizer")
@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES)
def test_channel_pedestrian_b(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = _multipath_channel(config, snr_db, cfo_hz, phase, seed,
                                 PEDESTRIAN_B_DELAYS, PEDESTRIAN_B_GAINS_DB, PEDESTRIAN_DOPPLER_HZ)
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Vehicular A: fast fading (~48 Hz Doppler) within packet duration")
@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES)
def test_channel_vehicular_a(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = _multipath_channel(config, snr_db, cfo_hz, phase, seed,
                                 VEHICULAR_A_DELAYS, VEHICULAR_A_GAINS_DB, VEHICULAR_DOPPLER_HZ)
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="Vehicular B: ~107-sample delay spread and fast fading — no equalizer")
@pytest.mark.parametrize("snr_db", [25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES[:2])
def test_channel_vehicular_b(snr_db, specs, cfo_hz, phase, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = _multipath_channel(config, snr_db, cfo_hz, phase, seed,
                                 VEHICULAR_B_DELAYS, VEHICULAR_B_GAINS_DB, VEHICULAR_DOPPLER_HZ)
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    diagnose_and_assert(tx_packets, rx_packets)


@pytest.mark.xfail(strict=False, reason="sub-15 dB SNR may not decode reliably")
@pytest.mark.parametrize("specs,cfo_hz,phase,snr_db,seed", [
    ([(0, 6, ModulationSchemes.BPSK)],                                  0.0,    0.0, 12.0, 0),
    ([(0, 8, ModulationSchemes.QPSK)],                                  2500.0, 1.0, 10.0, 1),
    ([(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)], -5000.0, 2.0,  8.0, 2),
])
def test_hard_channel(specs, cfo_hz, phase, snr_db, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    channel = ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase, seed=seed,
    ))
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)
