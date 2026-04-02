"""Frame synchronization pipeline test.

References
----------
[1] ITU-R M.1225 (1997), Annex 2, Indoor Office Test Environment.
[2] IEEE 802.11 iterate-and-advance pattern (MATLAB WLAN Toolbox
    searchOffset, gr-ieee802-11 MIN_GAP).
[3] AD9361 Reference Manual UG-570 (Analog Devices), "Automatic
    Gain Control" and "Data Interface" sections.

"""

from dataclasses import dataclass

import numpy as np
import pytest

from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.frame_sync import (
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.channel import ChannelConfig, ChannelModel
from modules.modulators import BPSK, QPSK
from modules.pulse_shaping import match_filter, rrc_filter, upsample
from pluto.config import RRC_ALPHA, SAMPLE_RATE, SPAN, SPS

SYNC_CFG = SynchronizerConfig()
NUM_TAPS = 2 * SPS * SPAN + 1
COARSE_CFO_RANGE_HZ = SAMPLE_RATE / (2 * SYNC_CFG.short_preamble_nsym * SPS)

N_PAYLOAD_SYMBOLS = 200
SAMPLE_OFFSET = 200
GUARD_SAMPLES = 500
N_TRIALS = 30
N_FP_TRIALS = 10

MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECTION_RATE = 0.9
MAX_CFO_ERROR_HZ = 200
MIN_ALIGNED_RATE = 0.95
MIN_PEAK_RATIO = 5.0
MAX_PHASE_ERR_NOISY_DEG = 15.0
OTA_MAX_CFO_ERROR_HZ = 400  # relaxed for combined phase noise + SCO stress test

CHANNEL_TAPS_DB = np.array([0.0, -3.0])
_MP_GAINS = 10 ** (CHANNEL_TAPS_DB / 20)
_MP_GAINS = _MP_GAINS / np.linalg.norm(_MP_GAINS)
MULTIPATH_GAINS_DB = tuple(float(g) for g in 20 * np.log10(_MP_GAINS))
MULTIPATH_DELAYS = (0.0, 1.0)
PHASE_NOISE_PSD_DBCHZ = -48.2  # ~0.003 rad/sqrt(sample) at 5.336 MHz
SAMPLE_RATE_OFFSET_PPM = 50
AGC_STEP_DB = 6
AGC_SETTLE_SAMPLES = int(0.1e-3 * SAMPLE_RATE)
ADC_CLIP_FRACTION = 0.7

CFO_VALUES_HZ = [
    pytest.param(0, id="0kHz"),
    pytest.param(100, id="0.1kHz"),
    pytest.param(500, id="0.5kHz"),
    pytest.param(1_000, id="1kHz"),
    pytest.param(5_000, id="5kHz"),
    pytest.param(10_000, id="10kHz"),
    pytest.param(15_000, id="15kHz"),
    pytest.param(20_000, id="20kHz"),
    pytest.param(25_000, id="25kHz"),
]
SNR_VALUES_DB = [
    pytest.param(20, id="20dB"),
    pytest.param(15, id="15dB"),
    pytest.param(13, id="13dB"),
    pytest.param(10, id="10dB"),
]


@dataclass
class SyncFixture:
    """Pre-computed channel objects reused across test scenarios."""

    rrc_taps: np.ndarray
    long_ref: np.ndarray
    group_delay: int
    snr_db: float
    sig_power: float
    noise_scale: float  # for false-positive noise-only test


@pytest.fixture(params=SNR_VALUES_DB, scope="module")
def channel(request: pytest.FixtureRequest) -> SyncFixture:
    """Build channel fixture for the parametrized SNR."""
    snr_db = request.param
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    return SyncFixture(
        rrc_taps=rrc_taps,
        long_ref=build_long_ref(SYNC_CFG, SPS, rrc_taps),
        group_delay=(NUM_TAPS - 1) // 2,
        snr_db=snr_db,
        sig_power=sig_power,
        noise_scale=np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
    )


def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _make_channel_model(ch: SyncFixture, cfo_hz: float, seed: int, **kwargs) -> ChannelModel:
    """Create a ChannelModel with multipath, CFO, and AWGN from the fixture."""
    return ChannelModel(ChannelConfig(
        sample_rate=SAMPLE_RATE,
        snr_db=ch.snr_db,
        reference_power=ch.sig_power,
        enable_multipath=True,
        multipath_delays_samples=MULTIPATH_DELAYS,
        multipath_gains_db=MULTIPATH_GAINS_DB,
        cfo_hz=float(cfo_hz),
        seed=seed,
        **kwargs,
    ))


def _pad(tx: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), tx])


def _run_sync(rx, long_ref, rrc_taps):
    """Run full sync pipeline on the matched-filtered signal.

    short_preamble_nsym=37 gives L=296 > N-1=128, so filtered noise has no
    autocorrelation at lag L and false-positive rejection is reliable.
    Both coarse and fine timing operate in the filtered domain, so sample_idxs
    are directly usable for indexing the filtered buffer.
    """
    rx = match_filter(rx, rrc_taps)
    coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
    if coarse.m_peaks.size == 0 or coarse.m_peaks[0] < MIN_DETECTION_CONFIDENCE:
        return coarse, None
    fine = fine_timing(rx, long_ref, coarse.d_hats[:1], coarse.cfo_hats[:1], SAMPLE_RATE, SPS, SYNC_CFG)
    return coarse, fine


@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_sync_pipeline(cfo_hz: int, channel: SyncFixture) -> None:
    """Full sync pipeline: detection, CFO, timing, phase, false alarm, multi-frame, OTA.

    1. Detection rate >= 90 % over N_TRIALS.
    2. CFO accuracy: median error < 200 Hz.
    3. Sub-symbol timing on correct sample grid >= 95 %.
    4. Peak ratio quality for detected frames.
    5. False-positive rejection on noise-only buffers.
    6. Multi-frame iterate-and-advance [4].
    7. Partial preamble detection.
    8. Combined OTA: Golay header + phase noise + AGC + SRO + clipping.
    9. Phase estimate recovery (noiseless and noisy).
    """
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    # TX and RX RRC group delays cancel in the matched-filtered stream
    expected_fine = SAMPLE_OFFSET + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS

    # --- 1-4. Detection rate, CFO, timing, peak quality ---
    detects, aligned = 0, 0
    cfo_errs: list[float] = []
    ratios: list[float] = []

    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx = _make_channel_model(channel, cfo_hz, seed).apply(_pad(_make_frame(rng, channel.rrc_taps)))
        coarse, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
        if fine is None:
            continue
        detects += 1
        cfo_errs.append(abs(float(coarse.cfo_hats[0]) - cfo_hz))
        aligned += int((int(fine.sample_idxs[0]) - expected_fine) % SPS == 0)
        ratios.append(float(fine.peak_ratios[0]))

    assert detects / N_TRIALS >= MIN_DETECTION_RATE, f"detect rate {detects / N_TRIALS:.0%}"
    assert float(np.median(cfo_errs)) < MAX_CFO_ERROR_HZ, f"median CFO error {np.median(cfo_errs):.0f} Hz"
    assert aligned / detects >= MIN_ALIGNED_RATE, f"aligned {aligned / detects:.0%}"
    assert float(np.median(ratios)) > MIN_PEAK_RATIO, f"median peak_ratio {np.median(ratios):.1f}"

    # --- 5. False positive rejection ---
    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.long_ref)
    for s in range(N_FP_TRIALS):
        rng = np.random.default_rng(10_000 + s)
        noise = channel.noise_scale * (rng.standard_normal(noise_len) + 1j * rng.standard_normal(noise_len))
        fp = coarse_sync(match_filter(noise.astype(np.complex64), channel.rrc_taps), SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp.m_peaks.size == 0 or fp.m_peaks[0] < MIN_DETECTION_CONFIDENCE

    # --- 6. Multi-frame ---
    rng = np.random.default_rng(7777)
    f1, f2 = _make_frame(rng, channel.rrc_taps), _make_frame(rng, channel.rrc_taps)
    amp = np.sqrt(np.mean(np.abs(f1) ** 2) / 2)
    guard = amp * (rng.standard_normal(GUARD_SAMPLES) + 1j * rng.standard_normal(GUARD_SAMPLES))
    multi_tx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), f1, guard, f2, guard])
    buf = _make_channel_model(channel, cfo_hz, 7777).apply(multi_tx)
    buf_filtered = match_filter(buf, channel.rrc_taps)
    cm = coarse_sync(buf_filtered, SAMPLE_RATE, SPS, SYNC_CFG)
    assert cm.m_peaks.size >= 2, f"expected >=2 frames, got {cm.m_peaks.size}"
    for i in range(2):
        assert cm.m_peaks[i] >= MIN_DETECTION_CONFIDENCE, f"frame {i+1} not detected"
        assert abs(float(cm.cfo_hats[i]) - cfo_hz) < MAX_CFO_ERROR_HZ, f"frame {i+1} CFO error"
    fm = fine_timing(buf_filtered, channel.long_ref, cm.d_hats[:1], cm.cfo_hats[:1], SAMPLE_RATE, SPS, SYNC_CFG)
    assert fm.peak_ratios[0] > MIN_PEAK_RATIO

    # --- 7. Partial preamble ---
    rng = np.random.default_rng(6666)
    rx = _make_channel_model(channel, cfo_hz, 6666).apply(_pad(_make_frame(rng, channel.rrc_taps)))
    half = SYNC_CFG.short_preamble_nsym * (SYNC_CFG.short_preamble_nreps // 2) * SPS
    rx_filtered = match_filter(rx, channel.rrc_taps)
    cp = coarse_sync(rx_filtered[SAMPLE_OFFSET + half:], SAMPLE_RATE, SPS, SYNC_CFG)
    assert cp.m_peaks.size > 0 and cp.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "partial preamble not detected"
    assert abs(float(cp.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "partial preamble CFO error"

    # --- 8. OTA combined ---
    rng = np.random.default_rng(9999)
    fc = FrameConstructor()
    hdr = FrameHeader(length=100, src=0, dst=1, frame_type=0, mod_scheme=ModulationSchemes.QPSK, sequence_number=0)
    hdr_bits, _ = fc.encode(hdr, rng.integers(0, 2, 100 * 8))
    tx = upsample(
        np.concatenate([generate_preamble(SYNC_CFG), BPSK().bits2symbols(hdr_bits),
                        rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)]),
        SPS, channel.rrc_taps,
    )
    rx = _make_channel_model(
        channel, cfo_hz, 9999,
        enable_phase_noise=True,
        phase_noise_psd_dbchz=PHASE_NOISE_PSD_DBCHZ,
        sco_ppm=SAMPLE_RATE_OFFSET_PPM,
    ).apply(_pad(tx))
    agc = np.ones(len(rx))
    agc[:AGC_SETTLE_SAMPLES] = 10 ** (-AGC_STEP_DB / 20)
    rx *= agc
    clip = ADC_CLIP_FRACTION * max(np.max(np.abs(np.real(rx))), np.max(np.abs(np.imag(rx))))
    rx = np.clip(np.real(rx), -clip, clip) + 1j * np.clip(np.imag(rx), -clip, clip)
    co, fo = _run_sync(rx, channel.long_ref, channel.rrc_taps)
    assert fo is not None, "OTA combined: not detected"
    assert abs(float(co.cfo_hats[0]) - cfo_hz) < OTA_MAX_CFO_ERROR_HZ, "OTA combined: CFO error"
    assert abs(int(fo.sample_idxs[0]) - expected_fine) % SPS <= 1, "OTA combined: off grid"
    assert fo.peak_ratios[0] > MIN_PEAK_RATIO

    # --- 9. Phase estimate must predict phase at payload position (not preamble) ---
    # Without this, the Costas loop starts ~90° off and locks on the wrong phase.
    # Uses the full channel model to test across CFO × initial phase combinations.
    for test_cfo_hz in [0, 500, 2000]:
        for test_phase_rad in [0, np.pi / 2, -np.pi * 3 / 4]:
            rng = np.random.default_rng(42)
            rx = _pad(_make_frame(rng, channel.rrc_taps))
            ch_model = ChannelModel(ChannelConfig(
                sample_rate=SAMPLE_RATE,
                snr_db=30.0,
                cfo_hz=test_cfo_hz,
                initial_phase_rad=test_phase_rad,
                seed=42,
            ))
            rx = ch_model.apply(rx)

            coarse, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
            assert fine is not None, f"CFO={test_cfo_hz} phase={np.degrees(test_phase_rad):.0f}°: not detected"

            # sample_idxs is in the filtered domain; add group_delay to get physical time
            payload_pos = fine.sample_idxs[0] + len(channel.long_ref) + channel.group_delay
            expected_phase = test_phase_rad + 2 * np.pi * test_cfo_hz * payload_pos / SAMPLE_RATE
            err = abs(np.angle(np.exp(1j * (fine.phase_estimates[0] - expected_phase))))
            assert err < np.radians(MAX_PHASE_ERR_NOISY_DEG), (
                f"CFO={test_cfo_hz} phase={np.degrees(test_phase_rad):.0f}°: "
                f"error {np.degrees(err):.1f}° at payload position"
            )
