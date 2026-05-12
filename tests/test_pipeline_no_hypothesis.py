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


def _packet_failures(tx_packets, rx_packets):
    """Count TX packets not successfully decoded at the receiver."""
    tx_by_seq = {p.seq_num: p for p, _ in tx_packets}
    rx_by_seq = {p.seq_num: p for p in rx_packets if p.seq_num in tx_by_seq}
    failed = 0
    for seq_num, tx_pkt in tx_by_seq.items():
        rx_pkt = rx_by_seq.get(seq_num)
        if (rx_pkt is None
                or not rx_pkt.valid
                or rx_pkt.length != tx_pkt.length
                or not np.array_equal(rx_pkt.payload, tx_pkt.payload)):
            failed += 1
    return failed


N_CHANNEL_TRIALS = 500
PER_THRESHOLD = 0.01


def _run_channel_trials(specs, build_channel, base_seed, n_trials=N_CHANNEL_TRIALS):
    """Run n_trials realizations with different seeds; return (total_pkts, total_failed)."""
    total_pkts = 0
    total_failed = 0
    for trial in range(n_trials):
        trial_seed = base_seed * 10_007 + trial
        tx_packets, signal = make_packets_and_signal(specs, trial_seed)
        _, config = tx_packets[0]
        channel = build_channel(config, trial_seed)
        rx_packets, _ = RXPipeline(config).receive(channel.apply(signal))
        total_pkts += len(tx_packets)
        total_failed += _packet_failures(tx_packets, rx_packets)
    return total_pkts, total_failed


def _assert_per(total_pkts, total_failed):
    per = total_failed / total_pkts
    assert per <= PER_THRESHOLD, (
        f"PER {per:.2%} ({total_failed}/{total_pkts}) > {PER_THRESHOLD:.2%}"
    )


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
    ([(0, 8, ModulationSchemes.PSK8)], 5),
    ([(0, 8, ModulationSchemes.PSK16)], 6),
    ([(0, 6, ModulationSchemes.PSK8), (1, 8, ModulationSchemes.PSK16)], 7),
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


@pytest.mark.parametrize("snr_db", [11, 15, 20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", CHANNEL_CASES)
def test_channel(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return ChannelModel(ChannelConfig(
            sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
            cfo_hz=cfo_hz, initial_phase_rad=phase, seed=trial_seed,
        ))
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


PSK8_CASES = [
    ([(0, 8, ModulationSchemes.PSK8)],                                          0.0,    0.0, 4),
]

PSK16_CASES = [
    ([(0, 8, ModulationSchemes.PSK16)],                                         1000.0, 0.5, 5),
    ([(0, 6, ModulationSchemes.PSK8), (1, 8, ModulationSchemes.PSK16)],        -2500.0, 1.0, 6),
]


@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", PSK8_CASES + PSK16_CASES)
def test_channel_higher_order(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return ChannelModel(ChannelConfig(
            sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
            cfo_hz=cfo_hz, initial_phase_rad=phase, seed=trial_seed,
        ))
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


# --- ITU-R M.1225 multipath profiles (sample rate 5.336 MHz) ---

PEDESTRIAN_DOPPLER_HZ = np.float32(1.2)
VEHICULAR_DOPPLER_HZ  = np.float32(48.0)

# Helix antennas on both ends → matched-handedness CP link. Single-bounce
# reflections invert handedness and are suppressed by the antenna XPD; the
# aimed direct path is LOS-dominant so its fading is Rician (not Rayleigh).
# Log-normal LOS shadowing models slow blockages (humans, vehicle bodies,
# antenna mis-aim) on top of the fast Rician fade.
#
# Pedestrian: slow motion, antennas mostly stay aimed → strong LOS, good XPD,
#   mild shadowing.
# Vehicular: faster motion + beam misalignment + ground reflections from the
#   vehicle body weaken both LOS dominance and pol selectivity, and create
#   substantial shadowing.
PEDESTRIAN_XPD_DB        = np.float32(28.3)
PEDESTRIAN_RICIAN_K_DB   = np.float32(10.0)
PEDESTRIAN_SHADOW_STD_DB = np.float32(4.0)
VEHICULAR_XPD_DB         = np.float32(23.3)
VEHICULAR_RICIAN_K_DB    = np.float32(5.0)
VEHICULAR_SHADOW_STD_DB  = np.float32(8.0)

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


def _build_multipath_channel(config, trial_seed, snr_db, cfo_hz, phase,
                              delays, gains_db, doppler_hz,
                              xpd_db, rician_k_db, shadow_std_db):
    return ChannelModel(ChannelConfig(
        sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
        cfo_hz=cfo_hz, initial_phase_rad=phase,
        enable_multipath=True,
        multipath_delays_samples=tuple(np.float32(d) for d in delays),
        multipath_gains_db=tuple(np.float32(g) for g in gains_db),
        cross_pol_discrimination_db=xpd_db,
        los_shadow_std_db=shadow_std_db,
        doppler_hz=doppler_hz, fading_type="rician",
        rician_k_db=rician_k_db, seed=trial_seed,
    ))


PEDESTRIAN_PARAMS = [
    *[(snr, *c) for snr in [20, 25, 30] for c in MULTIPATH_CASES + PSK8_CASES],
    *[(snr, *c) for snr in [25, 30]     for c in PSK16_CASES],
]


@pytest.mark.parametrize("snr_db,specs,cfo_hz,phase,seed", PEDESTRIAN_PARAMS)
def test_channel_pedestrian_a(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return _build_multipath_channel(config, trial_seed, snr_db, cfo_hz, phase,
                                        PEDESTRIAN_A_DELAYS, PEDESTRIAN_A_GAINS_DB, PEDESTRIAN_DOPPLER_HZ,
                                        PEDESTRIAN_XPD_DB, PEDESTRIAN_RICIAN_K_DB,
                                        PEDESTRIAN_SHADOW_STD_DB)
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


@pytest.mark.parametrize("snr_db,specs,cfo_hz,phase,seed", PEDESTRIAN_PARAMS)
def test_channel_pedestrian_b(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return _build_multipath_channel(config, trial_seed, snr_db, cfo_hz, phase,
                                        PEDESTRIAN_B_DELAYS, PEDESTRIAN_B_GAINS_DB, PEDESTRIAN_DOPPLER_HZ,
                                        PEDESTRIAN_XPD_DB, PEDESTRIAN_RICIAN_K_DB,
                                        PEDESTRIAN_SHADOW_STD_DB)
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


@pytest.mark.xfail(strict=False, reason="Vehicular A: log-normal shadowing on LOS drives ~5-10% PER without equalizer")
@pytest.mark.parametrize("snr_db", [20, 25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES)
def test_channel_vehicular_a(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return _build_multipath_channel(config, trial_seed, snr_db, cfo_hz, phase,
                                        VEHICULAR_A_DELAYS, VEHICULAR_A_GAINS_DB, VEHICULAR_DOPPLER_HZ,
                                        VEHICULAR_XPD_DB, VEHICULAR_RICIAN_K_DB,
                                        VEHICULAR_SHADOW_STD_DB)
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


@pytest.mark.xfail(strict=False, reason="Vehicular B: deep shadowing + 107-sample delay spread breaks decode")
@pytest.mark.parametrize("snr_db", [25, 30])
@pytest.mark.parametrize("specs,cfo_hz,phase,seed", MULTIPATH_CASES[:2])
def test_channel_vehicular_b(snr_db, specs, cfo_hz, phase, seed):
    def build(config, trial_seed):
        return _build_multipath_channel(config, trial_seed, snr_db, cfo_hz, phase,
                                        VEHICULAR_B_DELAYS, VEHICULAR_B_GAINS_DB, VEHICULAR_DOPPLER_HZ,
                                        VEHICULAR_XPD_DB, VEHICULAR_RICIAN_K_DB,
                                        VEHICULAR_SHADOW_STD_DB)
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)


@pytest.mark.parametrize("specs,cfo_hz,phase,snr_db,seed", [
    ([(0, 6, ModulationSchemes.BPSK)],                                  0.0,    0.0, 12.0, 0),
    ([(0, 8, ModulationSchemes.QPSK)],                                  2500.0, 1.0, 10.0, 1),
    ([(0, 6, ModulationSchemes.BPSK), (1, 8, ModulationSchemes.QPSK)], -5000.0, 2.0,  8.0, 2),
])
def test_hard_channel(specs, cfo_hz, phase, snr_db, seed):
    def build(config, trial_seed):
        return ChannelModel(ChannelConfig(
            sample_rate=config.SAMPLE_RATE, snr_db=snr_db,
            cfo_hz=cfo_hz, initial_phase_rad=phase, seed=trial_seed,
        ))
    total_pkts, total_failed = _run_channel_trials(specs, build, base_seed=seed)
    _assert_per(total_pkts, total_failed)
