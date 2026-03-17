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
ACQ_RANGE_HZ = SAMPLE_RATE / (2 * SYNC_CFG.short_preamble_nsym * SPS)

N_PAYLOAD_SYMBOLS = 200
SAMPLE_OFFSET = 200
GUARD_SAMPLES = 500
N_SEEDS = 30
N_FP_SEEDS = 10

MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECT_RATE = 0.9
MAX_CFO_ERROR_HZ = 200
MIN_SUBSYM_RATE = 0.95

MULTIPATH_TAPS_DB = np.array([0.0, -3.0])
AD9361_IQ_GAIN_ERROR_PCT = 0.2
AD9361_IQ_PHASE_ERROR_DEG = 0.2
AD9361_DC_OFFSET_DBC = -50
TCXO_PHASE_NOISE_STD = 0.003
TCXO_SRO_PPM = 50
AD9361_AGC_STEP_DB = 6
AD9361_AGC_SETTLE_SAMPLES = int(0.1e-3 * SAMPLE_RATE)
AD9361_ADC_CLIP_FRACTION = 0.7

PLUTOSDR_CFO_HZ = [
    pytest.param(0, id="0kHz"),
    pytest.param(10_000, id="10kHz"),
    pytest.param(25_000, id="25kHz"),
    pytest.param(50_000, id="50kHz"),
    pytest.param(60_000, id="60kHz"),
    pytest.param(80_000, id="80kHz"),
    pytest.param(120_000, id="120kHz"),
]
PLUTOSDR_SNR_DB = [
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
    ch: np.ndarray
    noise_scale: float
    dc: float
    iq_g: float
    iq_phi: float


@pytest.fixture(params=PLUTOSDR_SNR_DB, scope="module")
def sf(request: pytest.FixtureRequest) -> SyncFixture:
    """Build channel fixture for the parametrized SNR."""
    snr_db = request.param
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    ch = 10 ** (MULTIPATH_TAPS_DB / 20)
    ch /= np.linalg.norm(ch)
    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    return SyncFixture(
        rrc_taps=rrc_taps,
        long_ref=build_long_ref(SYNC_CFG, SPS, rrc_taps),
        group_delay=(NUM_TAPS - 1) // 2,
        ch=ch,
        noise_scale=np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
        dc=np.sqrt(sig_power) * 10 ** (AD9361_DC_OFFSET_DBC / 20),
        iq_g=1 + AD9361_IQ_GAIN_ERROR_PCT / 100,
        iq_phi=np.radians(AD9361_IQ_PHASE_ERROR_DEG),
    )



def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _apply_channel(rx: np.ndarray, cfo_hz: float, f: SyncFixture, rng: np.random.Generator) -> np.ndarray:
    """Multipath [3] -> CFO -> DC [1] -> IQ imbalance [1][2] -> AWGN."""
    delayed = np.empty_like(rx)
    delayed[0] = 0
    delayed[1:] = rx[:-1]
    rx = f.ch[0] * rx + f.ch[1] * delayed
    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += f.dc
    ri, rq = np.real(rx), np.imag(rx)
    rx = ri + 1j * f.iq_g * (np.sin(f.iq_phi) * ri + np.cos(f.iq_phi) * rq)
    rx += f.noise_scale * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx




@pytest.mark.parametrize("cfo_hz", PLUTOSDR_CFO_HZ)
def test_sync_pipeline(cfo_hz: int, sf: SyncFixture) -> None:
    """Sync pipeline under realistic PlutoSDR channel conditions.

    1. Detection rate >= 90 % over N_SEEDS realisations.
    2. CFO accuracy: median error < 200 Hz.
    3. Sub-symbol timing: fine_timing on correct sample grid >= 95 %.
    4. False-positive rejection: noise-only buffers must not trigger.
    5. Multi-frame iterate-and-advance [4]: detect -> consume -> re-detect.
    6. Partial preamble: buffer starting mid-preamble still detects.
    7. Combined OTA: real frame (Golay BPSK header) with all RX impairments.
    """
    if cfo_hz > ACQ_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds +/-{ACQ_RANGE_HZ / 1e3:.1f} kHz")

    expected_ft = SAMPLE_OFFSET + sf.group_delay + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS

    detect_count, subsym_zero = 0, 0
    cfo_errors: list[float] = []

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        tx = _make_frame(rng, sf.rrc_taps)
        rx = _apply_channel(np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx]), cfo_hz, sf, rng)

        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        if coarse.m_peaks.size == 0 or coarse.m_peaks[0] < MIN_DETECTION_CONFIDENCE:
            continue
        detect_count += 1
        cfo_errors.append(abs(float(coarse.cfo_hats[0]) - cfo_hz))
        fine = fine_timing(rx, sf.long_ref, int(coarse.d_hats[0]), float(coarse.cfo_hats[0]), SAMPLE_RATE, SPS, SYNC_CFG)
        subsym_zero += int((int(fine.sample_idxs[0]) - expected_ft) % SPS == 0)

    assert detect_count / N_SEEDS >= MIN_DETECT_RATE, f"detect rate {detect_count / N_SEEDS:.0%}"
    assert float(np.median(cfo_errors)) < MAX_CFO_ERROR_HZ, f"median CFO error {np.median(cfo_errors):.0f} Hz"
    assert subsym_zero / detect_count >= MIN_SUBSYM_RATE, f"sub-symbol aligned {subsym_zero / detect_count:.0%}"

    n_fp = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(sf.long_ref)
    for fp_seed in range(N_FP_SEEDS):
        rng_fp = np.random.default_rng(10_000 + fp_seed)
        noise_only = sf.noise_scale * (rng_fp.standard_normal(n_fp) + 1j * rng_fp.standard_normal(n_fp))
        fp = coarse_sync(noise_only, SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp.m_peaks.size == 0 or fp.m_peaks[0] < MIN_DETECTION_CONFIDENCE, f"false positive m_peak={fp.m_peaks[0]:.3f}"

    rng_mf = np.random.default_rng(7777)
    frame1 = _make_frame(rng_mf, sf.rrc_taps)
    frame2 = _make_frame(rng_mf, sf.rrc_taps)
    guard_amp = np.sqrt(np.mean(np.abs(frame1) ** 2) / 2)
    guard = guard_amp * (rng_mf.standard_normal(GUARD_SAMPLES) + 1j * rng_mf.standard_normal(GUARD_SAMPLES))

    buf = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), frame1, guard, frame2, guard])
    buf = _apply_channel(buf, cfo_hz, sf, rng_mf)

    coarse_mf = coarse_sync(buf, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_mf.m_peaks.size >= 2, f"expected >=2 frames, got {coarse_mf.m_peaks.size}"
    assert coarse_mf.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "frame 1 not detected"
    assert abs(float(coarse_mf.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 1 CFO error"
    fine1 = fine_timing(buf, sf.long_ref, int(coarse_mf.d_hats[0]), float(coarse_mf.cfo_hats[0]), SAMPLE_RATE, SPS, SYNC_CFG)
    assert fine1.sample_idxs[0] > 0
    assert coarse_mf.m_peaks[1] >= MIN_DETECTION_CONFIDENCE, f"frame 2 not detected (m_peak={coarse_mf.m_peaks[1]:.3f})"
    assert abs(float(coarse_mf.cfo_hats[1]) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 2 CFO error"

    rng_pp = np.random.default_rng(6666)
    tx_pp = _make_frame(rng_pp, sf.rrc_taps)
    rx_pp = _apply_channel(np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx_pp]), cfo_hz, sf, rng_pp)
    half_short = SYNC_CFG.short_preamble_nsym * (SYNC_CFG.short_preamble_nreps // 2) * SPS
    partial = rx_pp[SAMPLE_OFFSET + half_short :]
    coarse_pp = coarse_sync(partial, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_pp.m_peaks.size > 0 and coarse_pp.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "partial preamble: not detected"
    assert abs(float(coarse_pp.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "partial preamble: CFO error"

    rng_ota = np.random.default_rng(9999)
    fc = FrameConstructor()
    hdr = FrameHeader(length=100, src=0, dst=1, frame_type=0, mod_scheme=ModulationSchemes.QPSK, sequence_number=0)
    hdr_enc, _ = fc.encode(hdr, rng_ota.integers(0, 2, 100 * 8))
    hdr_syms = BPSK().bits2symbols(hdr_enc)
    payload = rng_ota.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    tx_ota = upsample(np.concatenate([generate_preamble(SYNC_CFG), hdr_syms, payload]), SPS, sf.rrc_taps)

    rx_ota = _apply_channel(np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx_ota]), cfo_hz, sf, rng_ota)
    rx_ota = rx_ota * np.exp(1j * np.cumsum(rng_ota.standard_normal(len(rx_ota)) * TCXO_PHASE_NOISE_STD))
    agc_gain = np.ones(len(rx_ota))
    agc_gain[:AD9361_AGC_SETTLE_SAMPLES] = 10 ** (-AD9361_AGC_STEP_DB / 20)
    rx_ota = rx_ota * agc_gain
    t_sro = np.arange(len(rx_ota)) * (1 + TCXO_SRO_PPM * 1e-6)
    t_orig = np.arange(len(rx_ota), dtype=float)
    rx_ota = np.interp(t_sro, t_orig, np.real(rx_ota)) + 1j * np.interp(t_sro, t_orig, np.imag(rx_ota))
    clip = AD9361_ADC_CLIP_FRACTION * max(np.max(np.abs(np.real(rx_ota))), np.max(np.abs(np.imag(rx_ota))))
    rx_ota = np.clip(np.real(rx_ota), -clip, clip) + 1j * np.clip(np.imag(rx_ota), -clip, clip)
    coarse_ota = coarse_sync(rx_ota, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_ota.m_peaks.size > 0 and coarse_ota.m_peaks[0] >= MIN_DETECTION_CONFIDENCE, "OTA combined: not detected"
    assert abs(float(coarse_ota.cfo_hats[0]) - cfo_hz) < MAX_CFO_ERROR_HZ, "OTA combined: CFO error"
    fine_ota = fine_timing(rx_ota, sf.long_ref, int(coarse_ota.d_hats[0]), float(coarse_ota.cfo_hats[0]), SAMPLE_RATE, SPS, SYNC_CFG)
    assert (int(fine_ota.sample_idxs[0]) - expected_ft) % SPS == 0, "OTA combined: fine timing off grid"
