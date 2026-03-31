"""Frame synchronization pipeline test.

References
----------
[1] AD9361 Datasheet Rev. G (Analog Devices), Table 1.
[2] Analog Devices, Analog Dialogue, "Understanding Image Rejection
    and Its Impact on Desired Signals".
[3] ITU-R M.1225 (1997), Annex 2, Indoor Office Test Environment.
[4] IEEE 802.11 iterate-and-advance pattern (MATLAB WLAN Toolbox
    searchOffset, gr-ieee802-11 MIN_GAP).
[5] AD9361 Reference Manual UG-570 (Analog Devices), "Automatic
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
from modules.modulators import BPSK, QPSK
from modules.pulse_shaping import rrc_filter, upsample
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
MIN_MULTI_FRAMES = 2

CHANNEL_TAPS_DB = np.array([0.0, -3.0])
IQ_GAIN_ERROR_PCT = 0.2
IQ_PHASE_ERROR_DEG = 0.2
DC_OFFSET_DBC = -50
PHASE_NOISE_STD = 0.003
SAMPLE_RATE_OFFSET_PPM = 50
AGC_STEP_DB = 6
AGC_SETTLE_SAMPLES = int(0.1e-3 * SAMPLE_RATE)
ADC_CLIP_FRACTION = 0.7

CFO_VALUES_HZ = [
    pytest.param(0, id="0kHz"),
    pytest.param(10_000, id="10kHz"),
    pytest.param(25_000, id="25kHz"),
    pytest.param(50_000, id="50kHz"),
    pytest.param(60_000, id="60kHz"),
    pytest.param(80_000, id="80kHz"),
    pytest.param(120_000, id="120kHz"),
]
SNR_VALUES_DB = [
    pytest.param(20, id="20dB"),
    pytest.param(15, id="15dB"),
    pytest.param(10, id="10dB"),
]



@dataclass
class SyncFixture:
    """Pre-computed channel objects reused across test scenarios."""

    rrc_taps: np.ndarray
    long_ref: np.ndarray
    group_delay: int
    taps: np.ndarray
    noise_scale: float
    dc_offset: float
    iq_gain: float
    iq_phase: float


@pytest.fixture(params=SNR_VALUES_DB, scope="module")
def channel(request: pytest.FixtureRequest) -> SyncFixture:
    """Build channel fixture for the parametrized SNR."""
    snr_db = request.param
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    taps = 10 ** (CHANNEL_TAPS_DB / 20)
    taps /= np.linalg.norm(taps)
    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    return SyncFixture(
        rrc_taps=rrc_taps,
        long_ref=build_long_ref(SYNC_CFG, SPS, rrc_taps),
        group_delay=(NUM_TAPS - 1) // 2,
        taps=taps,
        noise_scale=np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
        dc_offset=np.sqrt(sig_power) * 10 ** (DC_OFFSET_DBC / 20),
        iq_gain=1 + IQ_GAIN_ERROR_PCT / 100,
        iq_phase=np.radians(IQ_PHASE_ERROR_DEG),
    )



def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _apply_channel(rx: np.ndarray, cfo_hz: float, channel: SyncFixture, rng: np.random.Generator) -> np.ndarray:
    """Multipath [3] -> CFO -> DC [1] -> IQ imbalance [1][2] -> AWGN."""
    delayed = np.empty_like(rx)
    delayed[0] = 0
    delayed[1:] = rx[:-1]
    rx = channel.taps[0] * rx + channel.taps[1] * delayed
    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += channel.dc_offset
    sig_i, sig_q = np.real(rx), np.imag(rx)
    rx = sig_i + 1j * channel.iq_gain * (np.sin(channel.iq_phase) * sig_i + np.cos(channel.iq_phase) * sig_q)
    rx += channel.noise_scale * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx




@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_sync_pipeline(cfo_hz: int, channel: SyncFixture) -> None:
    """Sync pipeline under realistic PlutoSDR channel conditions.

    1. Detection rate >= 90 % over N_TRIALS realisations.
    2. CFO accuracy: median error < 200 Hz.
    3. Sub-symbol timing: fine_timing on correct sample grid >= 95 %.
    4. False-positive rejection: noise-only buffers must not trigger.
    5. Multi-frame iterate-and-advance [4]: detect -> consume -> re-detect.
    6. Partial preamble: buffer starting mid-preamble still detects.
    7. Combined OTA: real frame (Golay BPSK header) with all RX impairments.
    """
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds +/-{COARSE_CFO_RANGE_HZ / 1e3:.1f} kHz")

    expected_fine_idx = (
        SAMPLE_OFFSET + channel.group_delay
        + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS
    )

    detect_count, n_aligned = 0, 0
    cfo_errors: list[float] = []

    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        tx = _make_frame(rng, channel.rrc_taps)
        rx = _apply_channel(np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx]), cfo_hz, channel, rng)

        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        if coarse.m_peaks.size == 0 or coarse.m_peaks[0] < MIN_DETECTION_CONFIDENCE:
            continue
        detect_count += 1
        cfo_errors.append(abs(float(coarse.cfo_hats[0]) - cfo_hz))
        fine = fine_timing(
            rx, channel.long_ref, coarse.d_hats[:1], coarse.cfo_hats[:1],
            SAMPLE_RATE, SPS, SYNC_CFG,
        )
        n_aligned += int((int(fine.sample_idxs[0]) - expected_fine_idx) % SPS == 0)

    assert detect_count / N_TRIALS >= MIN_DETECTION_RATE, f"detect rate {detect_count / N_TRIALS:.0%}"
    assert float(np.median(cfo_errors)) < MAX_CFO_ERROR_HZ, f"median CFO error {np.median(cfo_errors):.0f} Hz"
    assert n_aligned / detect_count >= MIN_ALIGNED_RATE, f"sub-symbol aligned {n_aligned / detect_count:.0%}"

    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.long_ref)
    for fp_seed in range(N_FP_TRIALS):
        rng_fp = np.random.default_rng(10_000 + fp_seed)
        noise_only = channel.noise_scale * (
            rng_fp.standard_normal(noise_len) + 1j * rng_fp.standard_normal(noise_len)
        )
        fp_result = coarse_sync(noise_only, SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp_result.m_peaks.size == 0 or fp_result.m_peaks[0] < MIN_DETECTION_CONFIDENCE, (
            f"false positive m_peak={fp_result.m_peaks[0]:.3f}"
        )

    rng_multi = np.random.default_rng(7777)
    frame1 = _make_frame(rng_multi, channel.rrc_taps)
    frame2 = _make_frame(rng_multi, channel.rrc_taps)
    guard_amp = np.sqrt(np.mean(np.abs(frame1) ** 2) / 2)
    guard = guard_amp * (rng_multi.standard_normal(GUARD_SAMPLES) + 1j * rng_multi.standard_normal(GUARD_SAMPLES))

    buf = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), frame1, guard, frame2, guard])
    buf = _apply_channel(buf, cfo_hz, channel, rng_multi)

    coarse_multi = coarse_sync(buf, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_multi.m_peaks.size >= MIN_MULTI_FRAMES, f"expected >=2 frames, got {coarse_multi.m_peaks.size}"
    assert coarse_multi.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "frame 1 not detected"
    assert abs(float(coarse_multi.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 1 CFO error"
    fine1 = fine_timing(
        buf, channel.long_ref, coarse_multi.d_hats[:1], coarse_multi.cfo_hats[:1],
        SAMPLE_RATE, SPS, SYNC_CFG,
    )
    assert fine1.sample_idxs[0] > 0
    assert coarse_multi.m_peaks[1] >= MIN_DETECTION_CONFIDENCE, (
        f"frame 2 not detected (m_peak={coarse_multi.m_peaks[1]:.3f})"
    )
    assert abs(float(coarse_multi.cfo_hats[1]) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 2 CFO error"

    rng_partial = np.random.default_rng(6666)
    tx_partial = _make_frame(rng_partial, channel.rrc_taps)
    rx_partial = _apply_channel(
        np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx_partial]),
        cfo_hz, channel, rng_partial,
    )
    half_preamble_len = SYNC_CFG.short_preamble_nsym * (SYNC_CFG.short_preamble_nreps // 2) * SPS
    partial = rx_partial[SAMPLE_OFFSET + half_preamble_len :]
    coarse_partial = coarse_sync(partial, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_partial.m_peaks.size > 0, "partial preamble: not detected"
    assert coarse_partial.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "partial preamble: not detected"
    assert abs(float(coarse_partial.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "partial preamble: CFO error"

    rng_full = np.random.default_rng(9999)
    frame_ctor = FrameConstructor()
    header = FrameHeader(length=100, src=0, dst=1, frame_type=0, mod_scheme=ModulationSchemes.QPSK, sequence_number=0)
    header_bits, _ = frame_ctor.encode(header, rng_full.integers(0, 2, 100 * 8))
    header_syms = BPSK().bits2symbols(header_bits)
    payload = rng_full.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    tx_full = upsample(np.concatenate([generate_preamble(SYNC_CFG), header_syms, payload]), SPS, channel.rrc_taps)

    rx_full = _apply_channel(
        np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx_full]),
        cfo_hz, channel, rng_full,
    )
    rx_full = rx_full * np.exp(1j * np.cumsum(rng_full.standard_normal(len(rx_full)) * PHASE_NOISE_STD))
    agc_gain = np.ones(len(rx_full))
    agc_gain[:AGC_SETTLE_SAMPLES] = 10 ** (-AGC_STEP_DB / 20)
    rx_full = rx_full * agc_gain
    t_shifted = np.arange(len(rx_full)) * (1 + SAMPLE_RATE_OFFSET_PPM * 1e-6)
    t_nominal = np.arange(len(rx_full), dtype=float)
    rx_full = (
        np.interp(t_shifted, t_nominal, np.real(rx_full))
        + 1j * np.interp(t_shifted, t_nominal, np.imag(rx_full))
    )
    clip = ADC_CLIP_FRACTION * max(np.max(np.abs(np.real(rx_full))), np.max(np.abs(np.imag(rx_full))))
    rx_full = np.clip(np.real(rx_full), -clip, clip) + 1j * np.clip(np.imag(rx_full), -clip, clip)
    coarse_full = coarse_sync(rx_full, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_full.m_peaks.size > 0, "OTA combined: not detected"
    assert coarse_full.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "OTA combined: not detected"
    assert abs(float(coarse_full.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "OTA combined: CFO error"
    fine_full = fine_timing(
        rx_full, channel.long_ref, coarse_full.d_hats[:1], coarse_full.cfo_hats[:1],
        SAMPLE_RATE, SPS, SYNC_CFG,
    )
    assert (int(fine_full.sample_idxs[0]) - expected_fine_idx) % SPS == 0, "OTA combined: fine timing off grid"
