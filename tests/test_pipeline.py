import pytest
import numpy as np
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.pipeline import *
from modules.frame_constructor.frame_constructor import ModulationSchemes
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
    tx_packets, signal_parts = [], []
    for seq_num, length, mod in specs:
        config = PipelineConfig(MOD_SCHEME=mod)
        bits = rng.integers(0, 2, length * 8).reshape(-1, 1)
        packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=seq_num, length=length, payload=bits)
        tx_packets.append((packet, config))
        signal_parts.append(TXPipeline(config).transmit(packet))
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
    """Channel-aware: every TX packet decoded; false detections tolerated."""
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    for seq_num, tx_pkt in tx_by_seq.items():
        assert seq_num in rx_by_seq, f"packet seq_num={seq_num} not received"
        rx_pkt = rx_by_seq[seq_num]
        assert rx_pkt.valid
        assert rx_pkt.length == tx_pkt.length
        assert np.array_equal(rx_pkt.payload, tx_pkt.payload)


# --- detection rate ---

FA_TRIALS = 50
FA_BUFFER_LENGTH = 2**15
FA_THRESHOLD = 0.05  # max fraction of buffers allowed to have a false alarm


def test_false_alarm_on_noise():
    rx = RXPipeline(PipelineConfig())
    rng = np.random.default_rng(0)
    false_alarms = sum(
        len(rx.receive((rng.standard_normal(FA_BUFFER_LENGTH) + 1j * rng.standard_normal(FA_BUFFER_LENGTH)) / np.sqrt(2))[0])
        for _ in range(FA_TRIALS)
    )
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


# --- integration ---

@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow])
@given(specs=packet_specs(), seed=st.integers(0, 2**31))
def test_ideal(specs, seed):
    tx_packets, signal = make_packets_and_signal(specs, seed)
    _, config = tx_packets[0]
    rx_packets, _ = RXPipeline(config).receive(signal)
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
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase, seed=seed,
    ))

    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
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
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase, seed=seed,
    ))
    rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
    assert_all_received(tx_packets, rx_packets)
