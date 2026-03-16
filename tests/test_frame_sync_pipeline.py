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

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.frame_sync import (
    CoarseResult,
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.modulators import BPSK, QPSK
from modules.pulse_shaping import rrc_filter, upsample

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

MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECT_RATE = 0.9
MAX_CFO_ERROR_HZ = 200
MIN_SUBSYM_RATE = 0.95

AD9361_IQ_GAIN_ERROR_PCT = 0.2
AD9361_IQ_PHASE_ERROR_DEG = 0.2
AD9361_DC_OFFSET_DBC = -50

MULTIPATH_TAPS_DB = np.array([0.0, -3.0])

TCXO_PHASE_NOISE_STD = 0.003
TCXO_SRO_PPM = 50

AD9361_AGC_STEP_DB = 6
AD9361_AGC_SETTLE_SAMPLES = int(0.1e-3 * SAMPLE_RATE)
AD9361_ADC_CLIP_FRACTION = 0.7

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


def _build_fixture(snr_db: int) -> SyncFixture:
    """Build channel fixture for a given SNR."""
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


@pytest.fixture(params=PLUTOSDR_SNR_DB)
def sync(request: pytest.FixtureRequest) -> SyncFixture:
    """Build channel fixture for the parametrized SNR."""
    return _build_fixture(request.param)


def _make_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, payload]), SPS, rrc_taps)


def _apply_channel(rx: np.ndarray, cfo_hz: float, sf: SyncFixture, rng: np.random.Generator) -> np.ndarray:
    """Multipath [3] -> CFO -> DC [1] -> IQ imbalance [1][2] -> AWGN."""
    delayed = np.empty_like(rx)
    delayed[0] = 0
    delayed[1:] = rx[:-1]
    rx = sf.ch[0] * rx + sf.ch[1] * delayed
    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += sf.dc
    ri, rq = np.real(rx), np.imag(rx)
    rx = ri + 1j * sf.iq_g * (np.sin(sf.iq_phi) * ri + np.cos(sf.iq_phi) * rq)
    rx += sf.noise_scale * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx


def _make_rx(seed: int, cfo_hz: float, sf: SyncFixture) -> tuple[np.ndarray, np.random.Generator]:
    """Build a single-frame RX buffer with full channel impairments."""
    rng = np.random.default_rng(seed)
    tx = _make_frame(rng, sf.rrc_taps)
    rx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx])
    return _apply_channel(rx, cfo_hz, sf, rng), rng


def _make_real_frame(rng: np.random.Generator, rrc_taps: np.ndarray) -> np.ndarray:
    """Build a frame with Golay-encoded BPSK header and QPSK payload."""
    fc = FrameConstructor()
    header = FrameHeader(
        length=100,
        src=0,
        dst=1,
        frame_type=0,
        mod_scheme=ModulationSchemes.QPSK,
        sequence_number=0,
    )
    header_encoded, _ = fc.encode(header, rng.integers(0, 2, 100 * 8))
    header_syms = BPSK().bits2symbols(header_encoded)
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    return upsample(np.concatenate([preamble, header_syms, payload]), SPS, rrc_taps)


def _apply_rx_impairments(
    rx: np.ndarray,
    rng: np.random.Generator,
    *,
    phase_noise: bool = True,
    agc: bool = True,
    sro: bool = True,
    adc_clip: bool = True,
) -> np.ndarray:
    """LO phase noise [1] -> AGC transient [5] -> SRO [1] -> ADC saturation [5]."""
    if phase_noise:
        phase_walk = np.cumsum(rng.standard_normal(len(rx)) * TCXO_PHASE_NOISE_STD)
        rx = rx * np.exp(1j * phase_walk)
    if agc:
        agc_gain = np.ones(len(rx))
        agc_gain[:AD9361_AGC_SETTLE_SAMPLES] = 10 ** (-AD9361_AGC_STEP_DB / 20)
        rx = rx * agc_gain
    if sro:
        t_sro = np.arange(len(rx)) * (1 + TCXO_SRO_PPM * 1e-6)
        t_orig = np.arange(len(rx), dtype=float)
        rx = np.interp(t_sro, t_orig, np.real(rx)) + 1j * np.interp(t_sro, t_orig, np.imag(rx))
    if adc_clip:
        clip_level = AD9361_ADC_CLIP_FRACTION * max(
            np.max(np.abs(np.real(rx))),
            np.max(np.abs(np.imag(rx))),
        )
        rx = np.clip(np.real(rx), -clip_level, clip_level) + 1j * np.clip(np.imag(rx), -clip_level, clip_level)
    return rx


@pytest.mark.parametrize("cfo_hz", PLUTOSDR_CFO_HZ)
def test_sync_pipeline(cfo_hz: int, sync: SyncFixture) -> None:
    """Sync pipeline under realistic PlutoSDR channel conditions.

    1. Detection rate >= 90 % over N_SEEDS realisations.
    2. CFO accuracy: median error < 200 Hz.
    3. Sub-symbol timing: fine_timing on correct sample grid >= 95 %.
    4. False-positive rejection: noise-only buffers must not trigger.
    5. Multi-frame iterate-and-advance [4]: detect -> consume -> re-detect.
    6. Partial preamble: buffer starting mid-preamble still detects.
    7. Combined OTA: real frame (Golay BPSK header) with all RX impairments
       (phase noise + AGC transient + SRO + ADC clipping) applied simultaneously.
    """
    if cfo_hz > ACQ_RANGE_HZ * 0.95:
        pytest.xfail(f"CFO {cfo_hz / 1e3:.0f} kHz exceeds +/-{ACQ_RANGE_HZ / 1e3:.1f} kHz")

    sf = sync
    expected_ft = SAMPLE_OFFSET + sf.group_delay + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS

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

        fine = fine_timing(rx, sf.long_ref, coarse, SAMPLE_RATE, SPS, SYNC_CFG)
        subsym_zero += int((int(fine.sample_idx) - expected_ft) % SPS == 0)

    assert detect_count / N_SEEDS >= MIN_DETECT_RATE, f"detect rate {detect_count / N_SEEDS:.0%}"
    assert float(np.median(cfo_errors)) < MAX_CFO_ERROR_HZ, f"median CFO error {np.median(cfo_errors):.0f} Hz"
    assert subsym_zero / detect_count >= MIN_SUBSYM_RATE, f"sub-symbol aligned {subsym_zero / detect_count:.0%}"

    n_fp = SAMPLE_OFFSET + N_PAYLOAD_SYMBOLS * SPS + len(sf.long_ref)
    for fp_seed in range(N_FP_SEEDS):
        rng_fp = np.random.default_rng(10_000 + fp_seed)
        noise_only = sf.noise_scale * (rng_fp.standard_normal(n_fp) + 1j * rng_fp.standard_normal(n_fp))
        fp = coarse_sync(noise_only, SAMPLE_RATE, SPS, SYNC_CFG)
        assert fp.m_peak < MIN_DETECTION_CONFIDENCE, f"false positive m_peak={fp.m_peak:.3f}"

    rng_mf = np.random.default_rng(7777)
    frame1 = _make_frame(rng_mf, sf.rrc_taps)
    frame2 = _make_frame(rng_mf, sf.rrc_taps)
    guard_amp = np.sqrt(np.mean(np.abs(frame1) ** 2) / 2)
    guard = guard_amp * (rng_mf.standard_normal(GUARD_SAMPLES) + 1j * rng_mf.standard_normal(GUARD_SAMPLES))

    buf = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), frame1, guard, frame2, guard])
    buf = _apply_channel(buf, cfo_hz, sf, rng_mf)

    coarse1 = coarse_sync(buf, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse1.m_peak >= MIN_DETECTION_CONFIDENCE, "frame 1 not detected"
    assert abs(float(coarse1.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 1 CFO error"

    fine1 = fine_timing(buf, sf.long_ref, coarse1, SAMPLE_RATE, SPS, SYNC_CFG)
    assert fine1.sample_idx > 0

    remainder = buf[SAMPLE_OFFSET + len(frame1) + GUARD_SAMPLES :]
    coarse2 = coarse_sync(remainder, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse2.m_peak >= MIN_DETECTION_CONFIDENCE, f"frame 2 not detected (m_peak={coarse2.m_peak:.3f})"
    assert abs(float(coarse2.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "frame 2 CFO error"

    rx_pp, _ = _make_rx(6666, cfo_hz, sf)
    half_short = SYNC_CFG.short_preamble_nsym * (SYNC_CFG.short_preamble_nreps // 2) * SPS
    partial = rx_pp[SAMPLE_OFFSET + half_short :]
    coarse_pp = coarse_sync(partial, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_pp.m_peak >= MIN_DETECTION_CONFIDENCE, "partial preamble: not detected"
    assert abs(float(coarse_pp.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "partial preamble: CFO error"

    rng_ota = np.random.default_rng(9999)
    rx_ota = np.concatenate(
        [np.zeros(SAMPLE_OFFSET, dtype=complex), _make_real_frame(rng_ota, sf.rrc_taps)],
    )
    rx_ota = _apply_channel(rx_ota, cfo_hz, sf, rng_ota)
    rx_ota = _apply_rx_impairments(rx_ota, rng_ota)
    coarse_ota = coarse_sync(rx_ota, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse_ota.m_peak >= MIN_DETECTION_CONFIDENCE, "OTA combined: not detected"
    assert abs(float(coarse_ota.cfo_hat) - cfo_hz) < MAX_CFO_ERROR_HZ, "OTA combined: CFO error"
    fine_ota = fine_timing(rx_ota, sf.long_ref, coarse_ota, SAMPLE_RATE, SPS, SYNC_CFG)
    assert (int(fine_ota.sample_idx) - expected_ft) % SPS == 0, "OTA combined: fine timing off grid"


def _timing_metric(samples: np.ndarray) -> np.ndarray:
    """Compute Schmidl-Cox M(d) [1] (same formula as coarse_sync)."""
    sample_cnt = SYNC_CFG.short_preamble_nsym * SPS
    cs_p = np.concatenate(([0j], np.cumsum(np.conj(samples[:-sample_cnt]) * samples[sample_cnt:])))
    p_d = cs_p[sample_cnt:] - cs_p[:-sample_cnt]
    cs_r = np.concatenate(([0.0], np.cumsum(np.abs(samples[sample_cnt:]) ** 2)))
    r_d = cs_r[sample_cnt:] - cs_r[:-sample_cnt]
    return np.abs(p_d) ** 2 / np.maximum(r_d**2, SYNC_CFG.energy_floor)


def _fine_corr(
    samples: np.ndarray, long_ref: np.ndarray, coarse: CoarseResult,
) -> tuple[np.ndarray, int]:
    """Compute fine-timing cross-correlation (same formula as fine_timing)."""
    samples_per_rep = SYNC_CFG.short_preamble_nsym * SPS
    start = int(coarse.d_hat) + SYNC_CFG.short_preamble_nreps * samples_per_rep
    margin = SYNC_CFG.long_margin_nsym * SPS
    s0 = max(start - margin, 0)
    s1 = min(len(samples), start + 2 * margin + len(long_ref))
    n = np.arange(s0, s1)
    r = samples[s0:s1] * np.exp(-2j * np.pi * (coarse.cfo_hat / SAMPLE_RATE) * n)
    z = np.correlate(r, long_ref, mode="valid")
    return np.abs(z), s0


@dataclass
class _SweepResult:
    """Per-SNR sweep data for plotting."""

    p_detect: list[float]
    p_correct: list[float]
    mpeak_base: list[float]
    mpeak_ota: list[float]


def _sweep_snr(
    snr_db: int, cfo_sweep_hz: np.ndarray, expected_ft: int,
) -> _SweepResult:
    """Sweep CFO for one SNR level, collecting detection and sync statistics."""
    sf = _build_fixture(snr_db)
    res = _SweepResult([], [], [], [])

    for cfo_hz in cfo_sweep_hz:
        raw_det, correct_sync = 0, 0
        peaks_b: list[float] = []
        peaks_o: list[float] = []

        for seed in range(N_SEEDS):
            rx_b, _ = _make_rx(seed, float(cfo_hz), sf)
            c_b = coarse_sync(rx_b, SAMPLE_RATE, SPS, SYNC_CFG)
            peaks_b.append(float(c_b.m_peak))

            detected = c_b.m_peak >= MIN_DETECTION_CONFIDENCE
            if detected:
                raw_det += 1
                cfo_ok = abs(float(c_b.cfo_hat) - float(cfo_hz)) < MAX_CFO_ERROR_HZ
                try:
                    fine = fine_timing(rx_b, sf.long_ref, c_b, SAMPLE_RATE, SPS, SYNC_CFG)
                    timing_ok = (int(fine.sample_idx) - expected_ft) % SPS == 0
                except ValueError:
                    timing_ok = False
                if cfo_ok and timing_ok:
                    correct_sync += 1

            rng_imp = np.random.default_rng(seed + 100_000)
            rx_o = _apply_rx_impairments(rx_b.copy(), rng_imp)
            c_o = coarse_sync(rx_o, SAMPLE_RATE, SPS, SYNC_CFG)
            peaks_o.append(float(c_o.m_peak))

        res.p_detect.append(raw_det / N_SEEDS)
        res.p_correct.append(correct_sync / N_SEEDS)
        res.mpeak_base.append(float(np.median(peaks_b)))
        res.mpeak_ota.append(float(np.median(peaks_o)))

    return res


def plot_sync_diagnostics(out_dir: str = "plots") -> None:
    """Generate synchronization performance plots.

    1. Schmidl-Cox timing metric M(d) at each SNR [1].
    2. Fine-timing cross-correlation: within-range vs. aliased CFO.
    3. Raw detection rate vs. correct sync rate (detect + CFO + timing).
    4. Detection margin (median m_peak), baseline vs. all impairments.
    """
    snr_sweep_db = [10, 15, 20]
    cfo_sweep_hz = np.arange(0, 130_001, 5_000)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    sf_ref = _build_fixture(snr_sweep_db[1])
    expected_ft = (
        SAMPLE_OFFSET + sf_ref.group_delay
        + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS
    )

    ax = axes[0, 0]
    for i, snr_db in enumerate(snr_sweep_db):
        sf = _build_fixture(snr_db)
        rx, _ = _make_rx(42, 0.0, sf)
        m_d = _timing_metric(rx)
        t_us = np.arange(len(m_d)) / SAMPLE_RATE * 1e6
        ax.plot(t_us, m_d, linewidth=0.5, color=colors[i], label=f"{snr_db} dB")
    ax.axhline(
        MIN_DETECTION_CONFIDENCE, color="r", linestyle="--", linewidth=1,
        label=f"threshold = {MIN_DETECTION_CONFIDENCE}",
    )
    ax.axvspan(
        SAMPLE_OFFSET / SAMPLE_RATE * 1e6, expected_ft / SAMPLE_RATE * 1e6,
        alpha=0.1, color="green", label="short preamble",
    )
    ax.set(xlabel="Time (\u03bcs)", ylabel="M(d)")
    ax.set_title("Schmidl\u2013Cox timing metric M(d)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)

    ax = axes[0, 1]
    cfo_demo_hz = [0, 20_000, 30_000]
    z_data = []
    z_peak = 0.0
    for cfo_hz in cfo_demo_hz:
        rx, _ = _make_rx(42, float(cfo_hz), sf_ref)
        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        z_abs, s0 = _fine_corr(rx, sf_ref.long_ref, coarse)
        rel_sym = (np.arange(len(z_abs)) + s0 - expected_ft) / SPS
        z_data.append((rel_sym, z_abs, cfo_hz))
        z_peak = max(z_peak, float(np.max(z_abs)))
    for rel_sym, z_abs, cfo_hz in z_data:
        within = cfo_hz < ACQ_RANGE_HZ
        ax.plot(
            rel_sym, z_abs / z_peak, linewidth=0.8,
            linestyle="-" if within else ":",
            label=f"{cfo_hz / 1e3:.0f} kHz" + ("" if within else " (aliased)"),
        )
    ax.set(xlabel="Offset from expected (symbols)", ylabel="Normalized |z(n)|")
    ax.set_title("Fine timing cross-correlation (SNR = 15 dB)")
    ax.legend(fontsize=7)
    ax.grid(visible=True, alpha=0.3)

    cfo_khz = cfo_sweep_hz / 1e3
    for i, snr_db in enumerate(snr_sweep_db):
        res = _sweep_snr(snr_db, cfo_sweep_hz, expected_ft)
        label = f"{snr_db} dB"
        axes[1, 0].plot(
            cfo_khz, res.p_detect, marker=".", markersize=3,
            color=colors[i], alpha=0.35, label=f"{label} raw detect",
        )
        axes[1, 0].plot(
            cfo_khz, res.p_correct, marker="s", markersize=3,
            color=colors[i], label=f"{label} correct sync",
        )
        axes[1, 1].plot(
            cfo_khz, res.mpeak_base, marker=".", markersize=3,
            color=colors[i], label=f"{label} baseline",
        )
        axes[1, 1].plot(
            cfo_khz, res.mpeak_ota, marker="x", markersize=3,
            linestyle="--", color=colors[i], label=f"{label} + impairments",
        )

    axes[1, 0].axhline(
        MIN_DETECT_RATE, color="r", linestyle="--", linewidth=0.5,
        alpha=0.5, label=f"req. {MIN_DETECT_RATE:.0%}",
    )
    axes[1, 0].set(ylabel="Rate")
    axes[1, 0].set_title("Raw detection vs. correct sync rate")
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].axhline(
        MIN_DETECTION_CONFIDENCE, color="r", linestyle="--", linewidth=1,
        alpha=0.5, label=f"threshold = {MIN_DETECTION_CONFIDENCE}",
    )
    axes[1, 1].set(ylabel="Median m_peak")
    axes[1, 1].set_title("Detection margin: baseline vs. impaired")

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.axvline(
            ACQ_RANGE_HZ / 1e3, color="k", linestyle=":", linewidth=1,
            label=f"acq. \u00b1{ACQ_RANGE_HZ / 1e3:.0f} kHz",
        )
        ax.set_xlabel("CFO (kHz)")
        ax.grid(visible=True, alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = out / "frame_sync_diagnostics.png"
    fig.savefig(path, dpi=150)
    sys.stdout.write(f"Saved \u2192 {path}\n")
