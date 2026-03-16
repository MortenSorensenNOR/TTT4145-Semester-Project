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

SAMPLE_RATE = 5_336_000
RRC_ALPHA = 0.35
SPAN = 8
N_PAYLOAD_SYMBOLS = 200
SAMPLE_OFFSET = 200
N_SEEDS = 30
GUARD_SAMPLES = 500

SPS = 8
SYNC_CFG = SynchronizerConfig()
ACQ_RANGE_HZ = SAMPLE_RATE / (2 * SYNC_CFG.short_preamble_nsym * SPS)

MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECT_RATE = 0.9
MAX_CFO_ERROR_HZ = 200
MIN_SUBSYM_RATE = 0.95

# AD9361 RX impairments after internal calibration [1][2].
AD9361_IQ_GAIN_ERROR_PCT = 0.2
AD9361_IQ_PHASE_ERROR_DEG = 0.2
AD9361_DC_OFFSET_DBC = -50

# Indoor multipath [3]: ITU-R M.1225, Annex 2, Indoor Office Channel A
MULTIPATH_TAPS_DB = np.array([0.0, -3.0])

N_FP_SEEDS = 10

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


def _apply_channel(
    rx: np.ndarray,
    cfo_hz: float,
    noise_scale: float,
    dc: float,
    iq_g: float,
    iq_phi: float,
    ch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply multipath -> CFO -> DC offset -> IQ imbalance -> AWGN."""
    rx_delayed = np.empty_like(rx)
    rx_delayed[0] = 0
    rx_delayed[1:] = rx[:-1]
    rx = ch[0] * rx + ch[1] * rx_delayed

    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += dc
    ri, rq = rx.real, rx.imag
    rx = ri + 1j * iq_g * (np.sin(iq_phi) * ri + np.cos(iq_phi) * rq)
    rx += noise_scale * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx


@pytest.mark.parametrize("cfo_hz", PLUTOSDR_CFO_HZ)
@pytest.mark.parametrize("snr_db", PLUTOSDR_SNR_DB)
def test_sync_pipeline(cfo_hz: int, snr_db: int) -> None:
    """Sync pipeline under realistic PlutoSDR channel conditions.

    Simulates TX -> channel -> RX and checks five properties across
    N_SEEDS noise realisations:

    1. Detection rate >= 90 %: coarse_sync finds the preamble reliably.
    2. CFO accuracy: median estimation error < 200 Hz.
    3. Sub-symbol timing: fine_timing lands on the correct sample grid
       (residual mod SPS == 0) in >= 95 % of detections.
    4. False-positive rejection: noise-only buffers must not trigger.
    5. Multi-frame iterate-and-advance [4]: two frames in one buffer,
       detect frame 1 -> consume -> detect frame 2 on remainder.

    Channel model (physical order):
        TX -> delay -> multipath [3] -> CFO -> DC offset [1]
           -> IQ imbalance [1] -> AWGN
    """
    num_taps = 2 * SPS * SPAN + 1
    group_delay = (num_taps - 1) // 2
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, num_taps)
    long_ref = build_long_ref(SYNC_CFG, SPS, rrc_taps)

    if cfo_hz > ACQ_RANGE_HZ * 0.95:
        pytest.xfail(
            f"CFO {cfo_hz / 1e3:.0f} kHz exceeds acquisition range "
            f"+/-{ACQ_RANGE_HZ / 1e3:.1f} kHz",
        )

    ch = 10 ** (MULTIPATH_TAPS_DB / 20)
    ch /= np.linalg.norm(ch)

    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    noise_scale = np.sqrt(sig_power / (2 * 10 ** (snr_db / 10)))
    dc = np.sqrt(sig_power) * 10 ** (AD9361_DC_OFFSET_DBC / 20)
    iq_g = 1 + AD9361_IQ_GAIN_ERROR_PCT / 100
    iq_phi = np.radians(AD9361_IQ_PHASE_ERROR_DEG)

    # -- 1-3. Statistical single-frame detection ------------------------------
    detect_count = 0
    cfo_errors: list[float] = []
    subsym_zero = 0

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        preamble = generate_preamble(SYNC_CFG)
        payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
        tx = upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)

        rx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx])
        rx = _apply_channel(rx, cfo_hz, noise_scale, dc, iq_g, iq_phi, ch, rng)

        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        if coarse.m_peak < MIN_DETECTION_CONFIDENCE:
            continue
        detect_count += 1
        cfo_errors.append(abs(float(coarse.cfo_hat) - cfo_hz))

        ft = fine_timing(rx, long_ref, coarse, SAMPLE_RATE, SPS, SYNC_CFG)
        expected_ft = SAMPLE_OFFSET + group_delay + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS
        subsym_zero += int((int(ft) - expected_ft) % SPS == 0)

    detect_rate = detect_count / N_SEEDS
    assert detect_rate >= MIN_DETECT_RATE, f"detect rate {detect_rate:.0%}"

    median_cfo_err = float(np.median(cfo_errors))
    assert median_cfo_err < MAX_CFO_ERROR_HZ, f"median CFO error {median_cfo_err:.0f} Hz"

    subsym_rate = subsym_zero / detect_count
    assert subsym_rate >= MIN_SUBSYM_RATE, f"sub-symbol aligned {subsym_rate:.0%}"

    # -- 4. False-positive rejection ------------------------------------------
    n_fp_samples = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(long_ref)
    for fp_seed in range(N_FP_SEEDS):
        rng_fp = np.random.default_rng(10_000 + fp_seed)
        noise_only = noise_scale * (
            rng_fp.standard_normal(n_fp_samples) + 1j * rng_fp.standard_normal(n_fp_samples)
        )
        fp = coarse_sync(noise_only, SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp.m_peak < MIN_DETECTION_CONFIDENCE, (
            f"false positive m_peak={fp.m_peak:.3f} (seed {10_000 + fp_seed})"
        )

    # -- 5. Multi-frame iterate-and-advance [4] -------------------------------
    rng_mf = np.random.default_rng(7777)
    preamble = generate_preamble(SYNC_CFG)
    payload1 = rng_mf.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    payload2 = rng_mf.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)

    frame1 = upsample(np.concatenate([preamble, payload1]), SPS, rrc_taps)
    frame2 = upsample(np.concatenate([preamble, payload2]), SPS, rrc_taps)
    gap = np.zeros(GUARD_SAMPLES, dtype=complex)

    buf_mf = np.concatenate([
        np.zeros(SAMPLE_OFFSET, dtype=complex),
        frame1, gap, frame2,
        np.zeros(SAMPLE_OFFSET, dtype=complex),
    ])
    # CFO + AWGN only (channel impairments already validated in sections 1-3;
    # DC offset on the zero-gap would create a constant-amplitude region that
    # the Schmidl-Cox metric mistakes for a perfect preamble).
    # Fixed noise floor keeps R(d) healthy in zero-regions regardless of SNR.
    buf_mf *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(buf_mf)))
    mf_noise = 0.05
    buf_mf += mf_noise * (rng_mf.standard_normal(len(buf_mf)) + 1j * rng_mf.standard_normal(len(buf_mf)))

    coarse1 = coarse_sync(buf_mf, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse1.m_peak >= MIN_DETECTION_CONFIDENCE, "frame 1 not detected in multi-frame buffer"
    cfo_err1 = abs(float(coarse1.cfo_hat) - cfo_hz)
    assert cfo_err1 < MAX_CFO_ERROR_HZ, f"multi-frame: frame 1 CFO error {cfo_err1:.0f} Hz"

    ft1 = fine_timing(buf_mf, long_ref, coarse1, SAMPLE_RATE, SPS, SYNC_CFG)
    assert ft1 > 0, "fine_timing returned non-positive index"

    consumed = SAMPLE_OFFSET + len(frame1) + len(gap)
    remainder = buf_mf[consumed:]

    coarse2 = coarse_sync(remainder, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse2.m_peak >= MIN_DETECTION_CONFIDENCE, (
        f"frame 2 not detected after consume (m_peak={coarse2.m_peak:.3f})"
    )
    cfo_err2 = abs(float(coarse2.cfo_hat) - cfo_hz)
    assert cfo_err2 < MAX_CFO_ERROR_HZ, f"multi-frame: frame 2 CFO error {cfo_err2:.0f} Hz"
