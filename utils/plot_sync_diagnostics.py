"""Synchronization performance diagnostic plots.

Standalone script — not a test. Run directly:
    python -m utils.plot_sync_diagnostics

Generates four plots:
1. Schmidl-Cox timing metric M(d) at each SNR.
2. Fine-timing cross-correlation: within-range vs. aliased CFO.
3. Raw detection rate vs. correct sync rate (detect + CFO + timing).
4. Detection margin (median m_peak), baseline vs. all impairments.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.frame_sync import (
    CoarseResult,
    SynchronizerConfig,
    build_long_ref,
    coarse_sync,
    fine_timing,
    generate_preamble,
)
from modules.modulators import QPSK
from modules.pulse_shaping import rrc_filter, upsample
from pluto.config import RRC_ALPHA, SAMPLE_RATE, SPAN, SPS

SYNC_CFG = SynchronizerConfig()
NUM_TAPS = 2 * SPS * SPAN + 1
ACQ_RANGE_HZ = SAMPLE_RATE / (2 * SYNC_CFG.short_preamble_nsym * SPS)

N_PAYLOAD_SYMBOLS = 200
SAMPLE_OFFSET = 200
N_SEEDS = 30

MIN_DETECTION_CONFIDENCE = 0.3
MIN_DETECT_RATE = 0.9
MAX_CFO_ERROR_HZ = 200

MULTIPATH_TAPS_DB = np.array([0.0, -3.0])
AD9361_IQ_GAIN_ERROR_PCT = 0.2
AD9361_IQ_PHASE_ERROR_DEG = 0.2
AD9361_DC_OFFSET_DBC = -50
TCXO_PHASE_NOISE_STD = 0.003
TCXO_SRO_PPM = 50
AD9361_AGC_STEP_DB = 6
AD9361_AGC_SETTLE_SAMPLES = int(0.1e-3 * SAMPLE_RATE)
AD9361_ADC_CLIP_FRACTION = 0.7


def _build_fixture(snr_db: int) -> dict:
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, NUM_TAPS)
    ch = 10 ** (MULTIPATH_TAPS_DB / 20)
    ch /= np.linalg.norm(ch)
    sig_power = np.mean(np.abs(upsample(generate_preamble(SYNC_CFG), SPS, rrc_taps)) ** 2)
    return {
        "rrc_taps": rrc_taps,
        "long_ref": build_long_ref(SYNC_CFG, SPS, rrc_taps),
        "group_delay": (NUM_TAPS - 1) // 2,
        "ch": ch,
        "noise_scale": np.sqrt(sig_power / (2 * 10 ** (snr_db / 10))),
        "dc": np.sqrt(sig_power) * 10 ** (AD9361_DC_OFFSET_DBC / 20),
        "iq_g": 1 + AD9361_IQ_GAIN_ERROR_PCT / 100,
        "iq_phi": np.radians(AD9361_IQ_PHASE_ERROR_DEG),
    }


def _make_rx(seed: int, cfo_hz: float, sf: dict) -> np.ndarray:
    rng = np.random.default_rng(seed)
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    tx = upsample(np.concatenate([preamble, payload]), SPS, sf["rrc_taps"])
    rx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=np.complex64), tx])
    # multipath -> CFO -> DC -> IQ -> AWGN
    delayed = np.empty_like(rx); delayed[0] = 0; delayed[1:] = rx[:-1]
    rx = sf["ch"][0] * rx + sf["ch"][1] * delayed
    rx *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(rx)))
    rx += sf["dc"]
    ri, rq = np.real(rx), np.imag(rx)
    rx = ri + 1j * sf["iq_g"] * (np.sin(sf["iq_phi"]) * ri + np.cos(sf["iq_phi"]) * rq)
    rx += sf["noise_scale"] * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    return rx


def _apply_rx_impairments(rx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rx = rx * np.exp(1j * np.cumsum(rng.standard_normal(len(rx)) * TCXO_PHASE_NOISE_STD))
    agc = np.ones(len(rx)); agc[:AD9361_AGC_SETTLE_SAMPLES] = 10 ** (-AD9361_AGC_STEP_DB / 20)
    rx = rx * agc
    t_sro = np.arange(len(rx)) * (1 + TCXO_SRO_PPM * 1e-6)
    t_orig = np.arange(len(rx), dtype=float)
    rx = np.interp(t_sro, t_orig, np.real(rx)) + 1j * np.interp(t_sro, t_orig, np.imag(rx))
    clip = AD9361_ADC_CLIP_FRACTION * max(np.max(np.abs(np.real(rx))), np.max(np.abs(np.imag(rx))))
    return np.clip(np.real(rx), -clip, clip) + 1j * np.clip(np.imag(rx), -clip, clip)


def _timing_metric(samples: np.ndarray) -> np.ndarray:
    """Schmidl-Cox M(d) (same formula as coarse_sync)."""
    L = SYNC_CFG.short_preamble_nsym * SPS
    cs_p = np.concatenate(([0j], np.cumsum(np.conj(samples[:-L]) * samples[L:])))
    p_d = cs_p[L:] - cs_p[:-L]
    cs_r = np.concatenate(([0.0], np.cumsum(np.abs(samples[L:]) ** 2)))
    r_d = cs_r[L:] - cs_r[:-L]
    return np.abs(p_d) ** 2 / np.maximum(r_d**2, SYNC_CFG.energy_floor)


def _fine_corr(samples: np.ndarray, long_ref: np.ndarray, coarse: CoarseResult) -> tuple[np.ndarray, int]:
    """Fine-timing cross-correlation (same formula as fine_timing)."""
    samples_per_rep = SYNC_CFG.short_preamble_nsym * SPS
    start = int(coarse.d_hat) + SYNC_CFG.short_preamble_nreps * samples_per_rep
    margin = SYNC_CFG.long_margin_nsym * SPS
    s0 = max(start - margin, 0)
    s1 = min(len(samples), start + 2 * margin + len(long_ref))
    n = np.arange(s0, s1)
    r = samples[s0:s1] * np.exp(-2j * np.pi * (coarse.cfo_hat / SAMPLE_RATE) * n)
    return np.abs(np.correlate(r, long_ref, mode="valid")), s0


def _sweep_snr(snr_db: int, cfo_sweep_hz: np.ndarray, expected_ft: int) -> dict:
    sf = _build_fixture(snr_db)
    p_detect, p_correct, mpeak_base, mpeak_ota = [], [], [], []

    for cfo_hz in cfo_sweep_hz:
        raw_det, correct_sync = 0, 0
        peaks_b, peaks_o = [], []

        for seed in range(N_SEEDS):
            rx_b = _make_rx(seed, float(cfo_hz), sf)
            c_b = coarse_sync(rx_b, SAMPLE_RATE, SPS, SYNC_CFG)
            peaks_b.append(float(c_b.m_peak))

            if c_b.m_peak >= MIN_DETECTION_CONFIDENCE:
                raw_det += 1
                cfo_ok = abs(float(c_b.cfo_hat) - float(cfo_hz)) < MAX_CFO_ERROR_HZ
                try:
                    fine = fine_timing(rx_b, sf["long_ref"], c_b, SAMPLE_RATE, SPS, SYNC_CFG)
                    timing_ok = (int(fine.sample_idx) - expected_ft) % SPS == 0
                except ValueError:
                    timing_ok = False
                if cfo_ok and timing_ok:
                    correct_sync += 1

            rng_imp = np.random.default_rng(seed + 100_000)
            c_o = coarse_sync(_apply_rx_impairments(rx_b.copy(), rng_imp), SAMPLE_RATE, SPS, SYNC_CFG)
            peaks_o.append(float(c_o.m_peak))

        p_detect.append(raw_det / N_SEEDS)
        p_correct.append(correct_sync / N_SEEDS)
        mpeak_base.append(float(np.median(peaks_b)))
        mpeak_ota.append(float(np.median(peaks_o)))

    return {"p_detect": p_detect, "p_correct": p_correct, "mpeak_base": mpeak_base, "mpeak_ota": mpeak_ota}


def plot_sync_diagnostics(out_dir: str = "plots") -> None:
    snr_sweep_db = [10, 15, 20]
    cfo_sweep_hz = np.arange(0, 130_001, 5_000)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    sf_ref = _build_fixture(snr_sweep_db[1])
    expected_ft = (
        SAMPLE_OFFSET + sf_ref["group_delay"]
        + SYNC_CFG.short_preamble_nsym * SYNC_CFG.short_preamble_nreps * SPS
    )

    # Top-left: Schmidl-Cox timing metric
    ax = axes[0, 0]
    for i, snr_db in enumerate(snr_sweep_db):
        sf = _build_fixture(snr_db)
        m_d = _timing_metric(_make_rx(42, 0.0, sf))
        ax.plot(np.arange(len(m_d)) / SAMPLE_RATE * 1e6, m_d, linewidth=0.5, color=colors[i], label=f"{snr_db} dB")
    ax.axhline(MIN_DETECTION_CONFIDENCE, color="r", linestyle="--", linewidth=1, label=f"threshold = {MIN_DETECTION_CONFIDENCE}")
    ax.axvspan(SAMPLE_OFFSET / SAMPLE_RATE * 1e6, expected_ft / SAMPLE_RATE * 1e6, alpha=0.1, color="green", label="short preamble")
    ax.set(xlabel="Time (\u03bcs)", ylabel="M(d)")
    ax.set_title("Schmidl\u2013Cox timing metric M(d)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)

    # Top-right: fine-timing cross-correlation
    ax = axes[0, 1]
    z_data, z_peak = [], 0.0
    for cfo_hz in [0, 20_000, 30_000]:
        rx = _make_rx(42, float(cfo_hz), sf_ref)
        coarse = coarse_sync(rx, SAMPLE_RATE, SPS, SYNC_CFG)
        z_abs, s0 = _fine_corr(rx, sf_ref["long_ref"], coarse)
        rel_sym = (np.arange(len(z_abs)) + s0 - expected_ft) / SPS
        z_data.append((rel_sym, z_abs, cfo_hz))
        z_peak = max(z_peak, float(np.max(z_abs)))
    for rel_sym, z_abs, cfo_hz in z_data:
        within = cfo_hz < ACQ_RANGE_HZ
        ax.plot(rel_sym, z_abs / z_peak, linewidth=0.8, linestyle="-" if within else ":",
                label=f"{cfo_hz / 1e3:.0f} kHz" + ("" if within else " (aliased)"))
    ax.set(xlabel="Offset from expected (symbols)", ylabel="Normalized |z(n)|")
    ax.set_title("Fine timing cross-correlation (SNR = 15 dB)")
    ax.legend(fontsize=7)
    ax.grid(visible=True, alpha=0.3)

    # Bottom: sweep detection + margin plots
    cfo_khz = cfo_sweep_hz / 1e3
    for i, snr_db in enumerate(snr_sweep_db):
        res = _sweep_snr(snr_db, cfo_sweep_hz, expected_ft)
        label = f"{snr_db} dB"
        axes[1, 0].plot(cfo_khz, res["p_detect"], marker=".", markersize=3, color=colors[i], alpha=0.35, label=f"{label} raw detect")
        axes[1, 0].plot(cfo_khz, res["p_correct"], marker="s", markersize=3, color=colors[i], label=f"{label} correct sync")
        axes[1, 1].plot(cfo_khz, res["mpeak_base"], marker=".", markersize=3, color=colors[i], label=f"{label} baseline")
        axes[1, 1].plot(cfo_khz, res["mpeak_ota"], marker="x", markersize=3, linestyle="--", color=colors[i], label=f"{label} + impairments")

    axes[1, 0].axhline(MIN_DETECT_RATE, color="r", linestyle="--", linewidth=0.5, alpha=0.5, label=f"req. {MIN_DETECT_RATE:.0%}")
    axes[1, 0].set(ylabel="Rate")
    axes[1, 0].set_title("Raw detection vs. correct sync rate")
    axes[1, 0].set_ylim(-0.05, 1.05)

    axes[1, 1].axhline(MIN_DETECTION_CONFIDENCE, color="r", linestyle="--", linewidth=1, alpha=0.5, label=f"threshold = {MIN_DETECTION_CONFIDENCE}")
    axes[1, 1].set(ylabel="Median m_peak")
    axes[1, 1].set_title("Detection margin: baseline vs. impaired")

    for ax in [axes[1, 0], axes[1, 1]]:
        ax.axvline(ACQ_RANGE_HZ / 1e3, color="k", linestyle=":", linewidth=1, label=f"acq. \u00b1{ACQ_RANGE_HZ / 1e3:.0f} kHz")
        ax.set_xlabel("CFO (kHz)")
        ax.grid(visible=True, alpha=0.3)
    for ax in axes.flat:
        ax.legend(fontsize=7)

    fig.tight_layout()
    path = out / "frame_sync_diagnostics.png"
    fig.savefig(path, dpi=150)
    sys.stdout.write(f"Saved \u2192 {path}\n")


if __name__ == "__main__":
    plot_sync_diagnostics()
