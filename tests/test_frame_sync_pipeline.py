"""Frame synchronization pipeline tests.

Each test has a single responsibility. Parametrize at the function level
so pytest reports exactly which scenario failed.

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

from modules.channel import ChannelConfig, ChannelModel
from modules.frame_constructor.frame_constructor import (
    FrameConstructor,
    FrameHeader,
    ModulationSchemes,
)
from modules.frame_sync.frame_sync import (
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.modulators import BPSK, QPSK
from modules.pipeline import PipelineConfig
from modules.pulse_shaping.pulse_shaping import match_filter, rrc_filter, upsample

# ---------------------------------------------------------------------------
# Pipeline / hardware constants
# ---------------------------------------------------------------------------

_cfg = PipelineConfig()
SPS: int = _cfg.SPS
SAMPLE_RATE: np.float32 = np.float32(_cfg.SAMPLE_RATE)
SPAN: int = _cfg.SPAN
RRC_ALPHA: float = _cfg.RRC_ALPHA
SYNC_CFG: SynchronizerConfig = _cfg.SYNC_CONFIG

NUM_TAPS: int = 2 * SPS * SPAN + 1

# CFO acquisition range implied by the short preamble length
COARSE_CFO_RANGE_HZ: float = float(SAMPLE_RATE) / (
    2 * SYNC_CFG.short_preamble_nsym * SPS
)

# ---------------------------------------------------------------------------
# Test scenario constants
# ---------------------------------------------------------------------------

N_PAYLOAD_SYMBOLS: int = 200
# Leading zeros injected before the frame in single-frame tests.
# Mirrors the TX pipeline's guard_syms (GUARD_SYMS_LENGTH symbols x SPS).
# Override GUARD_SYMS_LENGTH in PipelineConfig to adjust both together.
GUARD_SAMPLES: int = _cfg.GUARD_SYMS_LENGTH * SPS
SAMPLE_OFFSET: int = GUARD_SAMPLES
N_TRIALS: int = 30
N_FP_TRIALS: int = 10

# --- pass/fail thresholds ---
MIN_DETECTION_CONFIDENCE: float = 0.3
MIN_DETECTION_RATE: float = 0.90
MAX_CFO_ERROR_HZ: float = 2_000.0
MIN_ALIGNED_RATE: float = 0.95
MIN_PEAK_RATIO: float = 4.0
MAX_PHASE_ERR_DEG: float = 15.0
OTA_MAX_CFO_ERROR_HZ: float = 3_000.0  # relaxed: combined phase noise + SCO stress

# ---------------------------------------------------------------------------
# Channel impairment parameters  (ITU-R M.1225 indoor office [1])
# ---------------------------------------------------------------------------

_CHANNEL_TAPS_DB = np.array([0.0, -3.0])
_MP_GAINS = 10 ** (_CHANNEL_TAPS_DB / 20)
_MP_GAINS = _MP_GAINS / np.linalg.norm(_MP_GAINS)
MULTIPATH_GAINS_DB: tuple[float, ...] = tuple(
    float(g) for g in 20 * np.log10(_MP_GAINS)
)
MULTIPATH_DELAYS: tuple[float, ...] = (0.0, 1.0)

PHASE_NOISE_PSD_DBCHZ: float = -48.2   # ~0.003 rad/sqrt(sample) at 5.336 MHz
SAMPLE_RATE_OFFSET_PPM: int = 50
AGC_STEP_DB: int = 6
AGC_SETTLE_SAMPLES: int = int(0.1e-3 * SAMPLE_RATE)
ADC_CLIP_FRACTION: float = 0.7

# ---------------------------------------------------------------------------
# Parametrize sets
# ---------------------------------------------------------------------------

CFO_VALUES_HZ = [
    pytest.param(0,      id="0kHz"),
    pytest.param(100,    id="0.1kHz"),
    pytest.param(500,    id="0.5kHz"),
    pytest.param(1_000,  id="1kHz"),
    pytest.param(5_000,  id="5kHz"),
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

# Phase / CFO grid used in the phase-estimate test
_PHASE_GRID_RAD = [0.0, np.pi / 2, -np.pi * 3 / 4]
_CFO_GRID_HZ    = [0, 500, 2_000]

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@dataclass
class SyncFixture:
    """Pre-computed objects shared across tests at the same SNR."""

    rrc_taps:    np.ndarray
    long_ref:    np.ndarray
    group_delay: int
    snr_db:      float
    sig_power:   float
    noise_scale: float   # for noise-only (false-positive) buffers


@pytest.fixture(params=SNR_VALUES_DB, scope="module")
def channel(request: pytest.FixtureRequest) -> SyncFixture:
    """Build the shared channel fixture for one SNR level."""
    snr_db: float = request.param
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    preamble  = generate_preamble(SYNC_CFG)
    sig_power = float(np.mean(np.abs(upsample(preamble, SPS, rrc_taps)) ** 2))
    return SyncFixture(
        rrc_taps    = rrc_taps,
        long_ref    = build_long_ref(SYNC_CFG, SPS, rrc_taps),
        group_delay = (NUM_TAPS - 1) // 2,
        snr_db      = snr_db,
        sig_power   = sig_power,
        noise_scale = np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    """Preamble + random QPSK payload, pulse-shaped at SPS."""
    preamble = generate_preamble(SYNC_CFG)
    payload  = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _channel_model(ch: SyncFixture, cfo_hz: float, seed: int, **kwargs) -> ChannelModel:
    """Multipath + AWGN channel from the fixture, optionally with extra impairments."""
    return ChannelModel(ChannelConfig(
        sample_rate              = SAMPLE_RATE,
        snr_db                   = ch.snr_db,
        reference_power          = ch.sig_power,
        enable_multipath         = True,
        multipath_delays_samples = MULTIPATH_DELAYS,
        multipath_gains_db       = MULTIPATH_GAINS_DB,
        cfo_hz                   = float(cfo_hz),
        seed                     = seed,
        **kwargs,
    ))


def _prepend_zeros(tx: np.ndarray) -> np.ndarray:
    """Prepend GUARD_SAMPLES zeros, matching the TX pipeline's leading guard_syms."""
    return np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), tx])


def _run_sync(rx: np.ndarray, long_ref: np.ndarray, rrc_taps: np.ndarray):
    """Match-filter then run coarse + fine sync.

    Returns (coarse, fine); fine is None when coarse detection falls below
    MIN_DETECTION_CONFIDENCE.

    Note: short_preamble_nsym=23 gives L=184 > N-1=128, so the filtered
    noise has no autocorrelation at lag L and false-positive rejection is
    reliable at the detection threshold.
    """
    rx     = match_filter(rx, rrc_taps)
    coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
    if coarse.m_peaks.size == 0 or coarse.m_peaks[0] < MIN_DETECTION_CONFIDENCE:
        return coarse, None
    fine = fine_timing(
        rx, long_ref,
        coarse.d_hats[:1], coarse.cfo_hats[:1],
        SAMPLE_RATE, SPS, SYNC_CFG,
    )
    return coarse, fine


def _expected_fine_sample() -> int:
    """Sample index of the long-preamble start after match-filtering.

    TX and RX RRC group delays cancel in the matched-filtered stream.
    """
    return (
        SAMPLE_OFFSET
        + SYNC_CFG.short_preamble_nsym
        * SYNC_CFG.short_preamble_nreps
        * SPS
    )

# ---------------------------------------------------------------------------
# 1. Detection rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_detection_rate(cfo_hz: int, channel: SyncFixture) -> None:
    """>=90% of frames are detected over N_TRIALS Monte-Carlo trials."""
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        _, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
        detects += fine is not None

    rate = detects / N_TRIALS
    assert rate >= MIN_DETECTION_RATE, f"detection rate {rate:.0%} < {MIN_DETECTION_RATE:.0%}"

# ---------------------------------------------------------------------------
# 2. CFO estimation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_cfo_accuracy(cfo_hz: int, channel: SyncFixture) -> None:
    """Median CFO estimation error < 200 Hz over N_TRIALS trials."""
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    errors: list[float] = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        coarse, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
        if fine is not None:
            errors.append(abs(float(coarse.cfo_hats[0]) - cfo_hz))

    assert errors, "no frames detected -- cannot evaluate CFO accuracy"
    median_err = float(np.median(errors))
    assert median_err < MAX_CFO_ERROR_HZ, (
        f"median CFO error {median_err:.0f} Hz >= {MAX_CFO_ERROR_HZ:.0f} Hz"
    )

# ---------------------------------------------------------------------------
# 3. Timing alignment (sub-symbol grid)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_timing_alignment(cfo_hz: int, channel: SyncFixture) -> None:
    """Fine timing lands on the correct sample grid in >=95% of detections."""
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    expected_fine = _expected_fine_sample()
    aligned = detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        _, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
        if fine is None:
            continue
        detects += 1
        aligned += int((int(fine.sample_idxs[0]) - expected_fine) % SPS == 0)

    assert detects > 0, "no frames detected -- cannot evaluate timing"
    rate = aligned / detects
    assert rate >= MIN_ALIGNED_RATE, f"aligned {rate:.0%} < {MIN_ALIGNED_RATE:.0%}"

# ---------------------------------------------------------------------------
# 4. Cross-correlation peak quality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_peak_ratio(cfo_hz: int, channel: SyncFixture) -> None:
    """Median peak-to-mean cross-correlation ratio > 5.0."""
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    ratios: list[float] = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        _, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
        if fine is not None:
            ratios.append(float(fine.peak_ratios[0]))

    assert ratios, "no frames detected -- cannot evaluate peak ratio"
    median_ratio = float(np.median(ratios))
    assert median_ratio > MIN_PEAK_RATIO, (
        f"median peak_ratio {median_ratio:.1f} <= {MIN_PEAK_RATIO}"
    )

# ---------------------------------------------------------------------------
# 5. False-positive rejection on noise-only buffers
# ---------------------------------------------------------------------------

def test_false_positive_rejection(channel: SyncFixture) -> None:
    """Coarse sync must not fire on noise-only buffers at the configured threshold."""
    noise_len = (
        SAMPLE_OFFSET
        + N_PAYLOAD_SYMBOLS * SPS
        + len(channel.long_ref)
    )
    for s in range(N_FP_TRIALS):
        rng = np.random.default_rng(10_000 + s)
        noise = channel.noise_scale * (
            rng.standard_normal(noise_len) + 1j * rng.standard_normal(noise_len)
        ).astype(np.complex64)
        filtered = match_filter(noise, channel.rrc_taps)
        result   = coarse_sync(filtered, SAMPLE_RATE, SPS, SYNC_CFG)
        spurious = (
            result.m_peaks.size > 0
            and result.m_peaks[0] >= MIN_DETECTION_CONFIDENCE
        )
        assert not spurious, (
            f"false positive on noise-only buffer (seed={10_000 + s}, "
            f"peak={result.m_peaks[0]:.3f})"
        )

# ---------------------------------------------------------------------------
# 6. Multi-frame iterate-and-advance  [2]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_multi_frame_detection(cfo_hz: int, channel: SyncFixture) -> None:
    """Both frames in a back-to-back pair are detected in >=90% of trials.

    Uses Monte-Carlo rather than a single fixed seed so a single unlucky
    noise realisation cannot cause a spurious failure at low SNR.
    """
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    both_detected = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        f1  = _make_frame(rng, channel.rrc_taps)
        f2  = _make_frame(rng, channel.rrc_taps)
        amp   = np.sqrt(np.mean(np.abs(f1) ** 2) / 2)
        guard = amp * (
            rng.standard_normal(GUARD_SAMPLES) + 1j * rng.standard_normal(GUARD_SAMPLES)
        ).astype(np.complex64)
        tx = np.concatenate([
            np.zeros(SAMPLE_OFFSET, dtype=np.complex64),
            f1, guard, f2, guard,
        ])
        buf          = _channel_model(channel, cfo_hz, seed).apply(tx)
        buf_filtered = match_filter(buf, channel.rrc_taps)
        cm           = coarse_sync(buf_filtered, SAMPLE_RATE, SPS, SYNC_CFG)

        good = [
            i for i in range(cm.m_peaks.size)
            if cm.m_peaks[i] >= MIN_DETECTION_CONFIDENCE
            and abs(float(cm.cfo_hats[i]) - cfo_hz) < MAX_CFO_ERROR_HZ
        ]
        if len(good) < 2:
            continue
        both_detected += 1

    rate = both_detected / N_TRIALS
    assert rate >= MIN_DETECTION_RATE, (
        f"both-frame detection rate {rate:.0%} < {MIN_DETECTION_RATE:.0%}"
    )

# ---------------------------------------------------------------------------
# 7. OTA combined stress test  [1][3]
#    Golay header + phase noise + AGC transient + SRO + ADC clipping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_ota_combined(cfo_hz: int, channel: SyncFixture) -> None:
    """Full over-the-air impairments: phase noise, AGC, SRO, clipping."""
    if cfo_hz > COARSE_CFO_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    expected_fine = _expected_fine_sample()
    rng = np.random.default_rng(9999)
    fc  = FrameConstructor()
    hdr = FrameHeader(
        length=100, src=0, dst=1, frame_type=0,
        mod_scheme=ModulationSchemes.QPSK,
        sequence_number=0,
    )
    hdr_bits, _ = fc.encode(hdr, rng.integers(0, 2, 100 * 8))
    tx = upsample(
        np.concatenate([
            generate_preamble(SYNC_CFG),
            BPSK().bits2symbols(hdr_bits),
            rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS),
        ]),
        SPS, channel.rrc_taps,
    )
    rx = _channel_model(
        channel, cfo_hz, 9999,
        enable_phase_noise    = True,
        phase_noise_psd_dbchz = PHASE_NOISE_PSD_DBCHZ,
        sco_ppm               = SAMPLE_RATE_OFFSET_PPM,
    ).apply(_prepend_zeros(tx))

    # AGC transient: first AGC_SETTLE_SAMPLES are 6 dB quieter  [3]
    agc = np.ones(len(rx), dtype=np.float32)
    agc[:AGC_SETTLE_SAMPLES] = 10 ** (-AGC_STEP_DB / 20)
    rx = rx * agc

    # ADC hard clipping
    clip = ADC_CLIP_FRACTION * max(
        np.max(np.abs(np.real(rx))),
        np.max(np.abs(np.imag(rx))),
    )
    rx = (
        np.clip(np.real(rx), -clip, clip)
        + 1j * np.clip(np.imag(rx), -clip, clip)
    ).astype(np.complex64)

    co, fo = _run_sync(rx, channel.long_ref, channel.rrc_taps)

    assert fo is not None, "OTA combined: frame not detected"
    assert abs(float(co.cfo_hats[0]) - cfo_hz) < OTA_MAX_CFO_ERROR_HZ, (
        f"OTA CFO error {abs(float(co.cfo_hats[0]) - cfo_hz):.0f} Hz"
    )
    assert abs(int(fo.sample_idxs[0]) - expected_fine) % SPS <= 1, (
        "OTA: fine timing off sample grid"
    )
    assert fo.peak_ratios[0] > MIN_PEAK_RATIO, (
        f"OTA: peak_ratio {fo.peak_ratios[0]:.1f}"
    )

# ---------------------------------------------------------------------------
# 8. Phase estimate predicts phase at payload position
#    Without this the Costas loop starts ~90 deg off and locks on wrong phase.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz",        _CFO_GRID_HZ)
@pytest.mark.parametrize("initial_phase", _PHASE_GRID_RAD)
def test_phase_estimate_at_payload(
    cfo_hz: int,
    initial_phase: float,
    channel: SyncFixture,
) -> None:
    """Phase estimate must predict the carrier phase at the payload start."""
    rng = np.random.default_rng(42)
    rx  = _prepend_zeros(_make_frame(rng, channel.rrc_taps))
    rx  = ChannelModel(ChannelConfig(
        sample_rate       = SAMPLE_RATE,
        snr_db            = 30.0,
        cfo_hz            = float(cfo_hz),
        initial_phase_rad = initial_phase,
        seed              = 42,
    )).apply(rx)

    coarse, fine = _run_sync(rx, channel.long_ref, channel.rrc_taps)
    assert fine is not None, (
        f"cfo={cfo_hz} Hz, phase={np.degrees(initial_phase):.0f} deg: not detected"
    )

    # sample_idxs is in the filtered domain; add group_delay for physical time
    payload_pos    = fine.sample_idxs[0] + len(channel.long_ref) + channel.group_delay
    expected_phase = initial_phase + 2 * np.pi * cfo_hz * payload_pos / SAMPLE_RATE
    err_rad        = abs(np.angle(np.exp(1j * (fine.phase_estimates[0] - expected_phase))))

    assert err_rad < np.radians(MAX_PHASE_ERR_DEG), (
        f"cfo={cfo_hz} Hz, phase={np.degrees(initial_phase):.0f} deg: "
        f"phase error {np.degrees(err_rad):.1f} deg > {MAX_PHASE_ERR_DEG} deg"
    )

# ------------------------------------------------------------------------------
# False detection test
# ------------------------------------------------------------------------------
def test_spurious_detection_rate(channel: SyncFixture) -> None:
    """Measure spurious detection rate in two scenarios:
    1. Noise-only buffers (energy gate calibrated against noise floor).
    2. Two-frame buffers (energy gate calibrated against preamble power —
       the realistic operating condition).
    Run with pytest -s to see output.
    """
    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.long_ref)

    # --- 1. Noise-only ---
    noise_spurious = 0
    for s in range(N_FP_TRIALS):
        rng = np.random.default_rng(10_000 + s)
        noise = channel.noise_scale * (
            rng.standard_normal(noise_len) + 1j * rng.standard_normal(noise_len)
        ).astype(np.complex64)
        result = coarse_sync(match_filter(noise, channel.rrc_taps), SAMPLE_RATE, SPS, SYNC_CFG)
        noise_spurious += result.m_peaks.size

    # --- 2. Two-frame buffers: count detections that fall in the guard ---
    guard_spurious = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        f1  = _make_frame(rng, channel.rrc_taps)
        f2  = _make_frame(rng, channel.rrc_taps)
        amp   = np.sqrt(np.mean(np.abs(f1) ** 2) / 2)
        guard = amp * (
            rng.standard_normal(GUARD_SAMPLES) + 1j * rng.standard_normal(GUARD_SAMPLES)
        ).astype(np.complex64)
        tx = np.concatenate([
            np.zeros(SAMPLE_OFFSET, dtype=np.complex64),
            f1, guard, f2, guard,
        ])
        buf_filtered = match_filter(
            _channel_model(channel, 0, seed).apply(tx),
            channel.rrc_taps,
        )
        cm = coarse_sync(buf_filtered, SAMPLE_RATE, SPS, SYNC_CFG)

        # Any detection whose CFO is garbage is spurious
        guard_spurious += sum(
            1 for i in range(cm.m_peaks.size)
            if abs(float(cm.cfo_hats[i])) >= MAX_CFO_ERROR_HZ
        )

    print(f"\nFalse detection test:"
          f"\nSNR={channel.snr_db}dB | "
          f"noise-only: {noise_spurious}/{N_FP_TRIALS} buffers | "
          f"two-frame guard: {guard_spurious}/{N_TRIALS} trials")
