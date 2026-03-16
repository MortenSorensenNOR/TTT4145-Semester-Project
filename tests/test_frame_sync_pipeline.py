"""Frame synchronization pipeline test -- PlutoSDR 480p IP bridge.

Verifies the full sync chain (coarse timing, CFO estimation, fine timing,
multi-frame iterate-and-advance, false-positive rejection) under realistic
PlutoSDR channel conditions including AD9361 RX impairments, indoor
multipath, and multi-frame stream buffers.

No CFO calibration assumed. PlutoSDR AD9361 TCXO: +/-25 ppm.
At 2.4 GHz, two uncalibrated devices can differ by up to +/-120 kHz.
Acquisition range with sps=8, nsym=13: +/-25.7 kHz.
Cases beyond the acquisition range are marked xfail.

References
----------
[1] AD9361 Datasheet Rev. G (Analog Devices), Table 1.
[2] Analog Devices, Analog Dialogue, "Understanding Image Rejection
    and Its Impact on Desired Signals".
[3] ITU-R M.1225 (1997), Annex 2, Indoor Office Test Environment.
[4] IEEE 802.11 iterate-and-advance pattern (MATLAB WLAN Toolbox
    searchOffset, gr-ieee802-11 MIN_GAP).

"""

from dataclasses import dataclass

import numpy as np
import pytest

from modules.frame_sync import (
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.modulators import QPSK
from modules.pulse_shaping import rrc_filter, upsample

# ── Radio constants ───────────────────────────────────────────────────────
SAMPLE_RATE = 5_336_000
SPS = 8
SPAN = 8
RRC_ALPHA = 0.35
N_PAYLOAD_SYMBOLS = 200
SAMPLE_OFFSET = 200
GUARD_SAMPLES = 500
N_SEEDS = 30
N_FP_SEEDS = 10
SYNC_CFG = SynchronizerConfig()
ACQ_RANGE_HZ = SAMPLE_RATE / (2 * SYNC_CFG.short_preamble_nsym * SPS)

# ── Pass/fail thresholds ─────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECT_RATE = 0.9
MAX_CFO_ERROR_HZ = 200
MIN_SUBSYM_RATE = 0.95

# ── AD9361 RX impairments [1][2] ─────────────────────────────────────────
AD9361_IQ_GAIN_ERROR_PCT = 0.2
AD9361_IQ_PHASE_ERROR_DEG = 0.2
AD9361_DC_OFFSET_DBC = -50

# ── Indoor multipath [3] ─────────────────────────────────────────────────
MULTIPATH_TAPS_DB = np.array([0.0, -3.0])

# ── Parametrize axes ─────────────────────────────────────────────────────
PLUTOSDR_CFO_HZ = [
    pytest.param(0, id="0kHz_matched"),
    pytest.param(10_000, id="10kHz"),
    pytest.param(25_000, id="25kHz_single_dev"),
    pytest.param(50_000, id="50kHz"),
    pytest.param(60_000, id="60kHz"),
    pytest.param(80_000, id="80kHz"),
    pytest.param(120_000, id="120kHz_worst_case"),
]
PLUTOSDR_SNR_DB = [
    pytest.param(20, id="close_range"),
    pytest.param(15, id="short_range"),
    pytest.param(10, id="mid_range"),
]


# ── Shared fixtures ──────────────────────────────────────────────────────
@dataclass
class SyncFixture:
    """Pre-computed objects reused across test sections."""

    rrc_taps: np.ndarray
    long_ref: np.ndarray
    group_delay: int
    ch: np.ndarray
    noise_scale: float
    dc: float
    iq_g: float
    iq_phi: float


@pytest.fixture
def sync(snr_db: int) -> SyncFixture:
    num_taps = 2 * SPS * SPAN + 1
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, num_taps)
    ch = 10 ** (MULTIPATH_TAPS_DB / 20)
    ch /= np.linalg.norm(ch)
    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    return SyncFixture(
        rrc_taps=rrc_taps,
        long_ref=build_long_ref(SYNC_CFG, SPS, rrc_taps),
        group_delay=(num_taps - 1) // 2,
        ch=ch,
        noise_scale=np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
        dc=np.sqrt(sig_power) * 10 ** (AD9361_DC_OFFSET_DBC / 20),
        iq_g=1 + AD9361_IQ_GAIN_ERROR_PCT / 100,
        iq_phi=np.radians(AD9361_IQ_PHASE_ERROR_DEG),
    )


# ── Helpers ───────────────────────────────────────────────────────────────
def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _apply_channel(rx: np.ndarray, cfo_hz: float, sf: SyncFixture, rng: np.random.Generator) -> np.ndarray:
    """Multipath [3] -> CFO -> DC [1] -> IQ imbalance [1] -> AWGN."""
    delayed = np.empty_like(rx)
    delayed[0] = 0
    delayed[1:] = rx[:-1]
    rx = sf.ch[0] * rx + sf.ch[1] * delayed
    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += sf.dc
    ri, rq = rx.real, rx.imag
    rx = ri + 1j * sf.iq_g * (np.sin(sf.iq_phi) * ri + np.cos(sf.iq_phi) * rq)
    rx += sf.noise_scale * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx


# ── Test ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cfo_hz", PLUTOSDR_CFO_HZ)
@pytest.mark.parametrize("snr_db", PLUTOSDR_SNR_DB)
def test_sync_pipeline(cfo_hz: int, snr_db: int, sync: SyncFixture) -> None:
    """Sync pipeline under realistic PlutoSDR channel conditions.

    1. Detection rate >= 90 % over N_SEEDS realisations.
    2. CFO accuracy: median error < 200 Hz.
    3. Sub-symbol timing: fine_timing on correct sample grid >= 95 %.
    4. False-positive rejection: noise-only buffers must not trigger.
    5. Multi-frame iterate-and-advance [4]: detect -> consume -> re-detect.
    """
    if cfo_hz > ACQ_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds +/-{ACQ_RANGE_HZ / 1e3:.1f} kHz")

    sf = sync
    expected_ft = SAMPLE_OFFSET + sf.group_delay + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS

    # -- 1-3. Statistical single-frame detection ------------------------------
    detect_count = 0
    cfo_errors: list[float] = []
    subsym_zero = 0

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        tx = _make_frame(rng, sf.rrc_taps)
        rx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx])
        rx = _apply_channel(rx, cfo_hz, sf, rng)

        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        if coarse.m_peak < MIN_DETECTION_CONFIDENCE:
            continue
        detect_count += 1
        cfo_errors.append(abs(float(coarse.cfo_hat) - cfo_hz))

        ft = fine_timing(rx, sf.long_ref, coarse, SAMPLE_RATE, SPS, SYNC_CFG)
        subsym_zero += int((int(ft) - expected_ft) % SPS == 0)

    assert detect_count / N_SEEDS >= MIN_DETECT_RATE, f"detect rate {detect_count / N_SEEDS:.0%}"
    assert float(np.median(cfo_errors)) < MAX_CFO_ERROR_HZ, f"median CFO error {np.median(cfo_errors):.0f} Hz"
    assert subsym_zero / detect_count >= MIN_SUBSYM_RATE, f"sub-symbol aligned {subsym_zero / detect_count:.0%}"

    # -- 4. False-positive rejection ------------------------------------------
    n_fp = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(sf.long_ref)
    for fp_seed in range(N_FP_SEEDS):
        rng_fp = np.random.default_rng(10_000 + fp_seed)
        noise_only = sf.noise_scale * (rng_fp.standard_normal(n_fp) + 1j * rng_fp.standard_normal(n_fp))
        fp = coarse_sync(noise_only, SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp.m_peak < MIN_DETECTION_CONFIDENCE, f"false positive m_peak={fp.m_peak:.3f}"

    # -- 5. Multi-frame iterate-and-advance [4] -------------------------------
    # CFO + AWGN only (DC offset on the zero-gap creates constant-amplitude
    # regions that the Schmidl-Cox metric mistakes for a perfect preamble).
    rng_mf = np.random.default_rng(7777)
    frame1 = _make_frame(rng_mf, sf.rrc_taps)
    frame2 = _make_frame(rng_mf, sf.rrc_taps)
    gap = np.zeros(GUARD_SAMPLES, dtype=complex)

    buf = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), frame1, gap, frame2, gap])
    buf *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(buf)))
    buf += 0.05 * (rng_mf.standard_normal(len(buf)) + 1j * rng_mf.standard_normal(len(buf)))

    coarse1 = coarse_sync(buf, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse1.m_peak >= MIN_DETECTION_CONFIDENCE, "frame 1 not detected"
    assert abs(float(coarse1.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 1 CFO error"

    ft1 = fine_timing(buf, sf.long_ref, coarse1, SAMPLE_RATE, SPS, SYNC_CFG)
    assert ft1 > 0

    remainder = buf[SAMPLE_OFFSET + len(frame1) + len(gap) :]
    coarse2 = coarse_sync(remainder, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse2.m_peak >= MIN_DETECTION_CONFIDENCE, f"frame 2 not detected (m_peak={coarse2.m_peak:.3f})"
    assert abs(float(coarse2.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 2 CFO error"
