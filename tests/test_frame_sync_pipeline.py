"""Frame synchronization pipeline tests."""

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
    build_preamble_ref,
    build_preamble_ref_rev,
    full_buffer_xcorr_sync,
    generate_preamble,
)
from modules.modulators.modulators import BPSK, QPSK
from modules.pipeline import PipelineConfig, RXPipeline
from modules.pulse_shaping.pulse_shaping import match_filter, rrc_filter, upsample

# --- pipeline / hardware constants ---

_cfg = PipelineConfig()
SPS         = _cfg.SPS
SAMPLE_RATE = np.float32(_cfg.SAMPLE_RATE)
SPAN        = _cfg.SPAN
RRC_ALPHA   = _cfg.RRC_ALPHA
SYNC_CFG    = _cfg.SYNC_CONFIG
NUM_TAPS    = 2 * SPS * SPAN + 1

# Unambiguous CFO range from the half-window phase-difference estimator is
# fs / (2 * preamble_nsym * SPS); the NCC peak rolls off as sinc² with the
# same width, so this is also where detection itself fails.
CFO_RANGE_HZ = float(SAMPLE_RATE) / (2 * SYNC_CFG.preamble_nsym * SPS)

# --- test scenario constants ---

N_PAYLOAD_SYMBOLS = 200
GUARD_SAMPLES     = _cfg.GUARD_SYMS_LENGTH * SPS
SAMPLE_OFFSET     = GUARD_SAMPLES
N_TRIALS          = 30
N_FP_TRIALS       = 10

MIN_NCC              = float(SYNC_CFG.ncc_threshold)
MIN_DETECTION_RATE   = 0.90
MAX_CFO_ERROR_HZ     = 2_000.0
MIN_ALIGNED_RATE     = 0.95
MIN_PEAK_NCC         = 0.5
MAX_PHASE_ERR_DEG    = 15.0
OTA_MAX_CFO_ERROR_HZ = 3_000.0

# --- channel impairments (ITU-R M.1225 indoor office) ---

_CHANNEL_TAPS_DB = np.array([0.0, -3.0])
_MP_GAINS = 10 ** (_CHANNEL_TAPS_DB / 20)
_MP_GAINS = _MP_GAINS / np.linalg.norm(_MP_GAINS)
MULTIPATH_GAINS_DB = tuple(float(g) for g in 20 * np.log10(_MP_GAINS))
MULTIPATH_DELAYS   = (0.0, 1.0)

PHASE_NOISE_PSD_DBCHZ  = -48.2
SAMPLE_RATE_OFFSET_PPM = 50
AGC_STEP_DB            = 6
AGC_SETTLE_SAMPLES     = int(0.1e-3 * SAMPLE_RATE)
ADC_CLIP_FRACTION      = 0.7

# --- parametrize sets ---

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
_PHASE_GRID_RAD = [0.0, np.pi / 2, -np.pi * 3 / 4]
_CFO_GRID_HZ    = [0, 500, 2_000]


@dataclass
class SyncFixture:
    rrc_taps:         np.ndarray
    preamble_ref:     np.ndarray
    preamble_ref_rev: np.ndarray
    group_delay:      int
    snr_db:           float
    sig_power:        float
    noise_scale:      float


@pytest.fixture(params=SNR_VALUES_DB, scope="module")
def channel(request):
    snr_db = request.param
    rrc_taps     = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    preamble_ref = build_preamble_ref(SYNC_CFG, SPS, rrc_taps)
    preamble     = generate_preamble(SYNC_CFG)
    sig_power    = float(np.mean(np.abs(upsample(preamble, SPS, rrc_taps)) ** 2))
    return SyncFixture(
        rrc_taps         = rrc_taps,
        preamble_ref     = preamble_ref,
        preamble_ref_rev = build_preamble_ref_rev(preamble_ref),
        group_delay      = (NUM_TAPS - 1) // 2,
        snr_db           = snr_db,
        sig_power        = sig_power,
        noise_scale      = np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
    )

# --- helpers ---

def _make_frame(rng, rrc_taps):
    preamble = generate_preamble(SYNC_CFG)
    payload  = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _channel_model(ch, cfo_hz, seed, **kwargs):
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


def _prepend_zeros(tx):
    return np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), tx])


def _run_sync(rx, ch):
    filtered = match_filter(rx, ch.rrc_taps)
    fine, cfos = full_buffer_xcorr_sync(
        filtered, ch.preamble_ref, ch.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
    )
    if fine.sample_idxs.size == 0:
        return None, None
    return fine, cfos


def _xfail_if_above_range(cfo_hz):
    if cfo_hz > CFO_RANGE_HZ * 0.85:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range")


def _make_noise(rng, n, scale):
    return scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)

# --- 1. detection rate ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_detection_rate(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx = _channel_model(channel, cfo_hz, seed).apply(_prepend_zeros(_make_frame(rng, channel.rrc_taps)))
        fine, _ = _run_sync(rx, channel)
        detects += fine is not None
    rate = detects / N_TRIALS
    assert rate >= MIN_DETECTION_RATE, f"detection rate {rate:.0%} < {MIN_DETECTION_RATE:.0%}"

# --- 2. CFO accuracy ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_cfo_accuracy(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    errors = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx = _channel_model(channel, cfo_hz, seed).apply(_prepend_zeros(_make_frame(rng, channel.rrc_taps)))
        fine, cfos = _run_sync(rx, channel)
        if fine is not None:
            errors.append(abs(float(cfos[0]) - cfo_hz))
    assert errors, "no frames detected -- cannot evaluate CFO accuracy"
    median_err = float(np.median(errors))
    assert median_err < MAX_CFO_ERROR_HZ, f"median CFO error {median_err:.0f} Hz >= {MAX_CFO_ERROR_HZ:.0f} Hz"

# --- 3. timing alignment (sub-symbol grid) ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_timing_alignment(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    aligned = detects = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx = _channel_model(channel, cfo_hz, seed).apply(_prepend_zeros(_make_frame(rng, channel.rrc_taps)))
        fine, _ = _run_sync(rx, channel)
        if fine is None:
            continue
        detects += 1
        aligned += int((int(fine.sample_idxs[0]) - SAMPLE_OFFSET) % SPS == 0)
    assert detects > 0, "no frames detected -- cannot evaluate timing"
    rate = aligned / detects
    assert rate >= MIN_ALIGNED_RATE, f"aligned {rate:.0%} < {MIN_ALIGNED_RATE:.0%}"

# --- 4. NCC peak quality ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_peak_ncc(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    nccs = []
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        rx = _channel_model(channel, cfo_hz, seed).apply(_prepend_zeros(_make_frame(rng, channel.rrc_taps)))
        fine, _ = _run_sync(rx, channel)
        if fine is not None:
            nccs.append(float(fine.peak_ratios[0]))
    assert nccs, "no frames detected -- cannot evaluate NCC"
    median_ncc = float(np.median(nccs))
    assert median_ncc > MIN_PEAK_NCC, f"median NCC {median_ncc:.2f} <= {MIN_PEAK_NCC:.2f}"

# --- 5. false-positive rejection on noise-only buffers ---

def test_false_positive_rejection(channel):
    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.preamble_ref)
    for s in range(N_FP_TRIALS):
        rng = np.random.default_rng(10_000 + s)
        noise = _make_noise(rng, noise_len, channel.noise_scale)
        filtered = match_filter(noise, channel.rrc_taps)
        fine, _ = full_buffer_xcorr_sync(
            filtered, channel.preamble_ref, channel.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
        )
        assert fine.sample_idxs.size == 0, (
            f"false positive (seed={10_000+s}, "
            f"NCC={float(fine.peak_ratios.max()) if fine.peak_ratios.size else 0.0:.3f})"
        )

# --- 6. multi-frame detection ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_multi_frame_detection(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    both_detected = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        f1 = _make_frame(rng, channel.rrc_taps)
        f2 = _make_frame(rng, channel.rrc_taps)
        amp = np.sqrt(np.mean(np.abs(f1) ** 2) / 2)
        guard = _make_noise(rng, GUARD_SAMPLES, amp)
        tx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), f1, guard, f2, guard])
        buf = _channel_model(channel, cfo_hz, seed).apply(tx)
        fine, cfos = full_buffer_xcorr_sync(
            match_filter(buf, channel.rrc_taps),
            channel.preamble_ref, channel.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
        )
        good = sum(1 for i in range(fine.sample_idxs.size) if abs(float(cfos[i]) - cfo_hz) < MAX_CFO_ERROR_HZ)
        if good >= 2:
            both_detected += 1
    rate = both_detected / N_TRIALS
    assert rate >= MIN_DETECTION_RATE, f"both-frame detection rate {rate:.0%} < {MIN_DETECTION_RATE:.0%}"

# --- 7. OTA combined stress: phase noise + AGC + SRO + clipping ---

@pytest.mark.parametrize("cfo_hz", CFO_VALUES_HZ)
def test_ota_combined(cfo_hz, channel):
    _xfail_if_above_range(cfo_hz)
    rng = np.random.default_rng(9999)
    hdr = FrameHeader(length=100, src=0, dst=1, frame_type=0,
                      mod_scheme=ModulationSchemes.QPSK, sequence_number=0)
    hdr_bits, _ = FrameConstructor().encode(hdr, rng.integers(0, 2, 100 * 8))
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

    # AGC transient: first AGC_SETTLE_SAMPLES are 6 dB quieter
    agc = np.ones(len(rx), dtype=np.float32)
    agc[:AGC_SETTLE_SAMPLES] = 10 ** (-AGC_STEP_DB / 20)
    rx = rx * agc

    # ADC hard clip
    clip = ADC_CLIP_FRACTION * max(np.max(np.abs(np.real(rx))), np.max(np.abs(np.imag(rx))))
    rx = (np.clip(np.real(rx), -clip, clip) + 1j * np.clip(np.imag(rx), -clip, clip)).astype(np.complex64)

    fine, cfos = _run_sync(rx, channel)
    assert fine is not None, "OTA combined: frame not detected"
    assert abs(float(cfos[0]) - cfo_hz) < OTA_MAX_CFO_ERROR_HZ
    assert abs(int(fine.sample_idxs[0]) - SAMPLE_OFFSET) % SPS <= 1, "OTA: fine timing off sample grid"
    assert fine.peak_ratios[0] > MIN_NCC

# --- 8. phase estimate at payload start ---

@pytest.mark.parametrize("cfo_hz",        _CFO_GRID_HZ)
@pytest.mark.parametrize("initial_phase", _PHASE_GRID_RAD)
def test_phase_estimate_at_payload(cfo_hz, initial_phase, channel):
    rng = np.random.default_rng(42)
    rx = _prepend_zeros(_make_frame(rng, channel.rrc_taps))
    rx = ChannelModel(ChannelConfig(
        sample_rate=SAMPLE_RATE, snr_db=30.0,
        cfo_hz=float(cfo_hz), initial_phase_rad=initial_phase, seed=42,
    )).apply(rx)

    fine, _ = _run_sync(rx, channel)
    assert fine is not None, f"cfo={cfo_hz} Hz, phase={np.degrees(initial_phase):.0f} deg: not detected"

    payload_pos    = fine.sample_idxs[0] + len(channel.preamble_ref) + channel.group_delay
    expected_phase = initial_phase + 2 * np.pi * cfo_hz * payload_pos / SAMPLE_RATE
    err_rad        = abs(np.angle(np.exp(1j * (fine.phase_estimates[0] - expected_phase))))

    assert err_rad < np.radians(MAX_PHASE_ERR_DEG), (
        f"cfo={cfo_hz} Hz, phase={np.degrees(initial_phase):.0f} deg: "
        f"phase error {np.degrees(err_rad):.1f} deg > {MAX_PHASE_ERR_DEG} deg"
    )

# --- 9. spurious detection rate (informational) ---

def test_spurious_detection_rate(channel):
    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.preamble_ref)

    noise_spurious = 0
    for s in range(N_FP_TRIALS):
        rng = np.random.default_rng(10_000 + s)
        noise = _make_noise(rng, noise_len, channel.noise_scale)
        fine, _ = full_buffer_xcorr_sync(
            match_filter(noise, channel.rrc_taps),
            channel.preamble_ref, channel.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
        )
        noise_spurious += fine.sample_idxs.size

    guard_spurious = 0
    for seed in range(N_TRIALS):
        rng = np.random.default_rng(seed)
        f1 = _make_frame(rng, channel.rrc_taps)
        f2 = _make_frame(rng, channel.rrc_taps)
        amp = np.sqrt(np.mean(np.abs(f1) ** 2) / 2)
        guard = _make_noise(rng, GUARD_SAMPLES, amp)
        tx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), f1, guard, f2, guard])
        fine, cfos = full_buffer_xcorr_sync(
            match_filter(_channel_model(channel, 0, seed).apply(tx), channel.rrc_taps),
            channel.preamble_ref, channel.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
        )
        guard_spurious += sum(1 for i in range(fine.sample_idxs.size) if abs(float(cfos[i])) >= MAX_CFO_ERROR_HZ)

    print(f"\nFalse detection: SNR={channel.snr_db}dB | "
          f"noise-only: {noise_spurious}/{N_FP_TRIALS} buffers | "
          f"two-frame guard: {guard_spurious}/{N_TRIALS} trials")

# --- 10. false-positive on colored noise (LO-leakage model) ---

# A pure tone correlated against a Zadoff-Chu reference yields NCC ~1/N_ref
# (~0.003), well below the gate, so single-stage NCC is structurally robust
# to LO leakage even with a +15 dB excess.
_LO_EXCESS_DB = 15.0
_LO_OFFSETS_HZ = [
    pytest.param(0,       id="DC"),
    pytest.param(20_000,  id="20kHz"),
    pytest.param(50_000,  id="50kHz"),
    pytest.param(100_000, id="100kHz"),
]


@pytest.mark.parametrize("lo_hz", _LO_OFFSETS_HZ)
def test_false_positive_colored_noise(lo_hz, channel):
    noise_len = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(channel.preamble_ref)
    lo_amp = channel.noise_scale * 10 ** (_LO_EXCESS_DB / 20)
    t = np.arange(noise_len, dtype=np.float32) / float(SAMPLE_RATE)

    for seed in range(N_FP_TRIALS):
        rng = np.random.default_rng(40_000 + seed)
        base = _make_noise(rng, noise_len, channel.noise_scale)
        lo_tone = (lo_amp * np.exp(2j * np.pi * lo_hz * t)).astype(np.complex64)
        colored = (base + lo_tone).astype(np.complex64)

        fine, _ = full_buffer_xcorr_sync(
            match_filter(colored, channel.rrc_taps),
            channel.preamble_ref, channel.preamble_ref_rev, MIN_NCC, SAMPLE_RATE,
        )
        assert fine.sample_idxs.size == 0, (
            f"lo_hz={lo_hz} Hz, snr={channel.snr_db}dB, seed={seed}: "
            f"NCC peaks {fine.peak_ratios} fired (LO +{_LO_EXCESS_DB}dB)"
        )

# --- 11. full-pipeline FP on hardware-matched colored noise ---

# Pluto noise floor ≈ broadband AWGN + ~5 dB DC-band elevation from LO leakage.
# Modeled as white noise + LPF-filtered noise (cutoff ~50 kHz / 4 MHz).
_DC_EXCESS_DB    = 5.0
_LPF_HALF_WIDTH  = 40
_HW_NOISE_TRIALS = 4


def _make_hw_noise(rng, n, noise_scale):
    base    = _make_noise(rng, n, noise_scale)
    kernel  = np.ones(_LPF_HALF_WIDTH * 2 + 1, dtype=np.float32) / (_LPF_HALF_WIDTH * 2 + 1)
    lo_raw  = _make_noise(rng, n, noise_scale)
    lo_filt = np.convolve(lo_raw, kernel, mode="same").astype(np.complex64)
    lo_scale = 10 ** (_DC_EXCESS_DB / 20)
    return (base + lo_scale * lo_filt).astype(np.complex64)


def test_false_positive_hw_matched_noise(channel):
    rx_pipe = RXPipeline(PipelineConfig())
    noise_len = 10 * (_cfg.GUARD_SYMS_LENGTH * SPS + len(channel.preamble_ref) + N_PAYLOAD_SYMBOLS * SPS)
    noise_len = int(2 ** np.ceil(np.log2(noise_len)))

    for seed in range(_HW_NOISE_TRIALS):
        rng = np.random.default_rng(50_000 + seed)
        buf = _make_hw_noise(rng, noise_len, channel.noise_scale)
        packets, _ = rx_pipe.receive(buf)
        assert len(packets) == 0, (
            f"snr={channel.snr_db}dB, seed={seed}: "
            f"{len(packets)} false detection(s) on hardware-matched colored noise"
        )
