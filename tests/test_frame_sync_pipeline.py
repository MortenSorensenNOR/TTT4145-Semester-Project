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
    build_long_ref_rev,
    full_buffer_xcorr_sync,
    generate_preamble,
)
from modules.modulators.modulators import BPSK, QPSK
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

# CFO acquisition range for the single-stage long-ZC NCC detector.
# Unambiguous CFO range from the half-window phase-difference estimator is
# fs / (2 * N) where N = long_preamble_nsym * SPS.  The NCC peak also rolls
# off as sinc²(cfo·N/fs), so this same bound is also where detection itself
# fails — they coincide.
SINGLE_STAGE_CFO_RANGE_HZ: float = float(SAMPLE_RATE) / (
    2 * SYNC_CFG.long_preamble_nsym * SPS
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
# NCC gate: matches the single_stage_ncc_threshold used by the production
# RXPipeline.  Below this the half-window CFO estimator is unreliable too.
MIN_NCC: float = float(SYNC_CFG.single_stage_ncc_threshold)
MIN_DETECTION_RATE: float = 0.90
MAX_CFO_ERROR_HZ: float = 2_000.0
MIN_ALIGNED_RATE: float = 0.95
# Median NCC for a clean detection is ~0.99 at zero CFO; sinc² rolloff drops
# it toward MIN_NCC at the acquisition edge.  0.5 leaves margin for AWGN +
# multipath at the lowest tested SNR while still being well above the gate.
MIN_PEAK_NCC: float = 0.5
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
    pytest.param(2_000,  id="2kHz"),
    pytest.param(3_000,  id="3kHz"),
    pytest.param(5_000,  id="5kHz"),
    pytest.param(8_000,  id="8kHz"),
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

    rrc_taps:     np.ndarray
    long_ref:     np.ndarray
    long_ref_rev: np.ndarray
    group_delay:  int
    snr_db:       float
    sig_power:    float
    noise_scale:  float   # for noise-only (false-positive) buffers


@pytest.fixture(params=SNR_VALUES_DB, scope="module")
def channel(request: pytest.FixtureRequest) -> SyncFixture:
    """Build the shared channel fixture for one SNR level."""
    snr_db: float = request.param
    rrc_taps  = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    long_ref  = build_long_ref(SYNC_CFG, SPS, rrc_taps)
    preamble  = generate_preamble(SYNC_CFG)
    sig_power = float(np.mean(np.abs(upsample(preamble, SPS, rrc_taps)) ** 2))
    return SyncFixture(
        rrc_taps     = rrc_taps,
        long_ref     = long_ref,
        long_ref_rev = build_long_ref_rev(long_ref),
        group_delay  = (NUM_TAPS - 1) // 2,
        snr_db       = snr_db,
        sig_power    = sig_power,
        noise_scale  = np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
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


def _run_sync(rx: np.ndarray, ch: SyncFixture):
    """Match-filter then run the single-stage NCC detector.

    Mirrors RXPipeline.detect(): full-buffer cross-correlation against the
    long-ZC reference, NCC threshold gate, half-window CFO estimate.

    Returns (fine, cfo_hats); both None when no detection fires.
    """
    rx_filtered  = match_filter(rx, ch.rrc_taps)
    fine, cfos   = full_buffer_xcorr_sync(
        rx_filtered, ch.long_ref, ch.long_ref_rev,
        MIN_NCC, SAMPLE_RATE,
    )
    if fine.sample_idxs.size == 0:
        return None, None
    return fine, cfos


def _expected_fine_sample() -> int:
    """Sample index of the long-preamble start after match-filtering.

    TX and RX RRC group delays cancel in the matched-filtered stream, so the
    long preamble starts at SAMPLE_OFFSET + (short_total_syms * SPS).
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
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        fine, _ = _run_sync(rx, channel)
        detects += fine is not None

    rate = detects / N_TRIALS
    assert rate >= MIN_DETECTION_RATE, f"detection rate {rate:.0%} < {MIN_DETECTION_RATE:.0%}"

# ---------------------------------------------------------------------------
# 2. CFO estimation accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_cfo_accuracy(cfo_hz: int, channel: SyncFixture) -> None:
    """Median CFO estimation error < 2 kHz over N_TRIALS trials."""
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    errors: list[float] = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        fine, cfos = _run_sync(rx, channel)
        if fine is not None:
            errors.append(abs(float(cfos[0]) - cfo_hz))

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
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    expected_fine = _expected_fine_sample()
    aligned = detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        fine, _ = _run_sync(rx, channel)
        if fine is None:
            continue
        detects += 1
        aligned += int((int(fine.sample_idxs[0]) - expected_fine) % SPS == 0)

    assert detects > 0, "no frames detected -- cannot evaluate timing"
    rate = aligned / detects
    assert rate >= MIN_ALIGNED_RATE, f"aligned {rate:.0%} < {MIN_ALIGNED_RATE:.0%}"

# ---------------------------------------------------------------------------
# 4. Cross-correlation peak quality (NCC)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_peak_ncc(cfo_hz: int, channel: SyncFixture) -> None:
    """Median normalized cross-correlation > MIN_PEAK_NCC at the detected peak."""
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")

    nccs: list[float] = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx  = _channel_model(channel, cfo_hz, seed).apply(
            _prepend_zeros(_make_frame(rng, channel.rrc_taps))
        )
        fine, _ = _run_sync(rx, channel)
        if fine is not None:
            nccs.append(float(fine.peak_ratios[0]))

    assert nccs, "no frames detected -- cannot evaluate NCC"
    median_ncc = float(np.median(nccs))
    assert median_ncc > MIN_PEAK_NCC, (
        f"median NCC {median_ncc:.2f} <= {MIN_PEAK_NCC:.2f}"
    )

# ---------------------------------------------------------------------------
# 5. False-positive rejection on noise-only buffers
# ---------------------------------------------------------------------------

def test_false_positive_rejection(channel: SyncFixture) -> None:
    """Single-stage sync must not fire on noise-only buffers at the NCC gate."""
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
        fine, _  = full_buffer_xcorr_sync(
            filtered, channel.long_ref, channel.long_ref_rev,
            MIN_NCC, SAMPLE_RATE,
        )
        assert fine.sample_idxs.size == 0, (
            f"false positive on noise-only buffer (seed={10_000 + s}, "
            f"NCC={float(fine.peak_ratios.max()) if fine.peak_ratios.size else 0.0:.3f})"
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
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
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
        fine, cfos   = full_buffer_xcorr_sync(
            buf_filtered, channel.long_ref, channel.long_ref_rev,
            MIN_NCC, SAMPLE_RATE,
        )

        good = [
            i for i in range(fine.sample_idxs.size)
            if abs(float(cfos[i]) - cfo_hz) < MAX_CFO_ERROR_HZ
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
#    header + phase noise + AGC transient + SRO + ADC clipping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_ota_combined(cfo_hz: int, channel: SyncFixture) -> None:
    """Full over-the-air impairments: phase noise, AGC, SRO, clipping."""
    if cfo_hz > SINGLE_STAGE_CFO_RANGE_HZ * 0.85:
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

    fine, cfos = _run_sync(rx, channel)

    assert fine is not None, "OTA combined: frame not detected"
    assert abs(float(cfos[0]) - cfo_hz) < OTA_MAX_CFO_ERROR_HZ, (
        f"OTA CFO error {abs(float(cfos[0]) - cfo_hz):.0f} Hz"
    )
    assert abs(int(fine.sample_idxs[0]) - expected_fine) % SPS <= 1, (
        "OTA: fine timing off sample grid"
    )
    assert fine.peak_ratios[0] > MIN_NCC, (
        f"OTA: NCC {fine.peak_ratios[0]:.2f} <= gate {MIN_NCC:.2f}"
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

    fine, _ = _run_sync(rx, channel)
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
    1. Noise-only buffers.
    2. Two-frame buffers (count any detection whose CFO is garbage — these
       must be sidelobes / payload bits accidentally correlating with the ZC).
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
        fine, _ = full_buffer_xcorr_sync(
            match_filter(noise, channel.rrc_taps),
            channel.long_ref, channel.long_ref_rev,
            MIN_NCC, SAMPLE_RATE,
        )
        noise_spurious += fine.sample_idxs.size

    # --- 2. Two-frame buffers: count detections with garbage CFO ---
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
        fine, cfos = full_buffer_xcorr_sync(
            buf_filtered, channel.long_ref, channel.long_ref_rev,
            MIN_NCC, SAMPLE_RATE,
        )

        guard_spurious += sum(
            1 for i in range(fine.sample_idxs.size)
            if abs(float(cfos[i])) >= MAX_CFO_ERROR_HZ
        )

    print(f"\nFalse detection test:"
          f"\nSNR={channel.snr_db}dB | "
          f"noise-only: {noise_spurious}/{N_FP_TRIALS} buffers | "
          f"two-frame guard: {guard_spurious}/{N_TRIALS} trials")

# ---------------------------------------------------------------------------
# 9. False-positive rejection on colored noise (LO-leakage model)
#
# A pure tone correlated against a long Zadoff-Chu reference gives
# |sum exp(jωm)·conj(zc[m])|² ≈ ||zc||² / N_zc (flat-spectrum ZC), so the
# normalized cross-correlation collapses to ~1/N_ref ≈ 0.003 — well below
# the NCC gate.  Test asserts no detections fire on tone+noise even with
# a +15 dB LO excess.
# ---------------------------------------------------------------------------

_LO_EXCESS_DB: float = 15.0

# Typical Pluto LO leakage frequencies relative to the tuned centre: DC,
# small crystal-error offset, and a moderate offset.
_LO_OFFSETS_HZ = [
    pytest.param(0,       id="DC"),
    pytest.param(20_000,  id="20kHz"),
    pytest.param(50_000,  id="50kHz"),
    pytest.param(100_000, id="100kHz"),
]


@pytest.mark.parametrize("lo_hz", _LO_OFFSETS_HZ)
def test_false_positive_colored_noise(lo_hz: int, channel: SyncFixture) -> None:
    """Single-stage NCC must not fire on tone+noise (LO leakage model)."""
    noise_len = (
        SAMPLE_OFFSET
        + N_PAYLOAD_SYMBOLS * SPS
        + len(channel.long_ref)
    )
    lo_amp = channel.noise_scale * 10 ** (_LO_EXCESS_DB / 20)
    t = np.arange(noise_len, dtype=np.float32) / float(SAMPLE_RATE)

    for seed in range(N_FP_TRIALS):
        rng = np.random.default_rng(40_000 + seed)
        base = channel.noise_scale * (
            rng.standard_normal(noise_len) + 1j * rng.standard_normal(noise_len)
        ).astype(np.complex64)
        lo_tone = (lo_amp * np.exp(2j * np.pi * lo_hz * t)).astype(np.complex64)
        colored = (base + lo_tone).astype(np.complex64)

        filtered = match_filter(colored, channel.rrc_taps)
        fine, _  = full_buffer_xcorr_sync(
            filtered, channel.long_ref, channel.long_ref_rev,
            MIN_NCC, SAMPLE_RATE,
        )
        assert fine.sample_idxs.size == 0, (
            f"lo_hz={lo_hz} Hz, snr={channel.snr_db}dB, seed={seed}: "
            f"NCC peaks {fine.peak_ratios} fired on colored noise "
            f"(LO +{_LO_EXCESS_DB}dB)"
        )

# ---------------------------------------------------------------------------
# 10. Full-pipeline false-positive rejection on hardware-matched noise
#
# The real PlutoSDR noise floor (TX muted, measured from hardware) has two
# components:
#   (a) broadband AWGN from the ADC thermal floor
#   (b) a ~5 dB DC-band elevation from LO leakage into the ADC
#
# We model this synthetically as white noise + low-pass-filtered noise
# (rectangular LPF, cutoff ~50 kHz / 4 MHz ≈ 1/80 of Nyquist) rather than
# shipping a 17 MB hardware capture in the repo.  The test exercises the full
# RXPipeline.receive() path (decimated coarse, full-rate fine, peak_ratio gate,
# header decode) so a regression at any layer is caught.
# ---------------------------------------------------------------------------

# DC-band power excess above the broadband floor, matching the ~5 dB measured
# on the Pluto.  Low-pass filter half-width in samples (≈50 kHz at 4 MHz).
_DC_EXCESS_DB: float = 5.0
_LPF_HALF_WIDTH: int = 40   # rectangular filter → cutoff at fs/(2*LPF_HALF_WIDTH) ≈ 50 kHz
_HW_NOISE_TRIALS: int = 4   # short: this runs through the full decoder


def _make_hw_noise(rng: np.random.Generator, n: int, noise_scale: float) -> np.ndarray:
    """Broadband noise + low-frequency (LO-leakage) component.

    Low-frequency component is a rectangular-LPF-filtered white noise stream
    scaled to be _DC_EXCESS_DB above the broadband floor, matching the
    measured Pluto hardware noise profile.
    """
    base = noise_scale * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)
    # LPF via convolution with a normalised rectangular window
    kernel   = np.ones(_LPF_HALF_WIDTH * 2 + 1, dtype=np.float32) / (_LPF_HALF_WIDTH * 2 + 1)
    lo_raw   = noise_scale * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    ).astype(np.complex64)
    lo_filt  = np.convolve(lo_raw, kernel, mode="same").astype(np.complex64)
    lo_scale = 10 ** (_DC_EXCESS_DB / 20)
    return (base + lo_scale * lo_filt).astype(np.complex64)


def test_false_positive_hw_matched_noise(channel: SyncFixture) -> None:
    """Full pipeline must produce zero frames on hardware-matched colored noise.

    Runs RXPipeline.receive() on a buffer whose length and noise power match
    a ~10-packet RX window (the scenario that produced ~800 spurious
    detections before the fine_peak_ratio gate was added).
    """
    from modules.pipeline import RXPipeline  # avoid heavy import at collection time

    rx_pipe  = RXPipeline(PipelineConfig())
    # ~10 frames worth of samples — mirrors the hardware capture length
    noise_len = 10 * (_cfg.GUARD_SYMS_LENGTH * SPS + len(channel.long_ref) + N_PAYLOAD_SYMBOLS * SPS)
    noise_len = int(2 ** np.ceil(np.log2(noise_len)))  # round to power-of-2

    for seed in range(_HW_NOISE_TRIALS):
        rng     = np.random.default_rng(50_000 + seed)
        buf     = _make_hw_noise(rng, noise_len, channel.noise_scale)
        packets, _ = rx_pipe.receive(buf)
        assert len(packets) == 0, (
            f"snr={channel.snr_db}dB, seed={seed}: "
            f"{len(packets)} false detection(s) on hardware-matched colored noise"
        )
