"""Real-time spectrum analyzer and IQ visualizer for the PlutoSDR."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes

from pluto import create_pluto
from pluto.config import CENTER_FREQ, RX_GAIN, SAMPLE_RATE


def compute_psd(
    samples: np.ndarray,
    window: np.ndarray,
    fft_size: int,
    p_fullscale_dbm: float,
    rx_gain: float,
) -> np.ndarray:
    """Compute the power spectral density of the given samples."""
    frame = samples[:fft_size] * window
    spectrum = np.fft.fftshift(np.fft.fft(frame, n=fft_size))
    power_norm = (np.abs(spectrum) / fft_size) ** 2
    return 10 * np.log10(power_norm + 1e-20) + p_fullscale_dbm - rx_gain


def bandpass_around_tone(samples: np.ndarray, offset_hz: float, sample_rate: float, bw: float = 20e3) -> np.ndarray:
    """Signed bandpass - only matches the correct +/- frequency bin."""
    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1 / sample_rate)
    mask = np.abs(freqs - offset_hz) > bw / 2
    spectrum[mask] = 0
    return np.fft.ifft(spectrum)


def style(ax: Axes) -> None:
    """Apply dark theme styling to a matplotlib axis."""
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.grid(visible=True, color="#2a2d35", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


if __name__ == "__main__":
    # ─── Config ───────────────────────────────────────────────────────────────────
    NUM_SAMPS = 10000
    FFT_SIZE = 1024 * 4
    AVERAGING = 10
    UPDATE_MS = 50

    TONE_FREQ = 100e3
    ATTENUATOR_DB = 30.0
    P_FULLSCALE_DBM = 10 * np.log10(5)

    # ─── Connect ──────────────────────────────────────────────────────────────────
    sdr_obj = cast("Any", create_pluto())
    sdr: Any = sdr_obj
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN
    sdr.rx_lo = int(CENTER_FREQ)
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = NUM_SAMPS

    # ─── Helpers ──────────────────────────────────────────────────────────────────
    freqs_mhz = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / SAMPLE_RATE)) + CENTER_FREQ) / 1e6
    tone_mhz = (CENTER_FREQ + TONE_FREQ) / 1e6

    window = np.hanning(FFT_SIZE)
    psd_buffer = np.full((AVERAGING, FFT_SIZE), -120.0)

    # ─── Figure ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"PlutoSDR RX  |  {CENTER_FREQ / 1e6:.1f} MHz  |  Tone @ {tone_mhz:.3f} MHz  |  "
        f"TX gain -50 dBm  |  {ATTENUATOR_DB:.0f} dB attenuator  |  RX gain {RX_GAIN:.0f} dB",
        color="white",
        fontsize=11,
        fontweight="bold",
        y=0.99,
    )

    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1.5], hspace=0.45, wspace=0.35)
    ax_spec = fig.add_subplot(gs[0, :])
    ax_wf = fig.add_subplot(gs[1, :])
    ax_iq = fig.add_subplot(gs[2, 0])
    ax_time = fig.add_subplot(gs[2, 1])

    for ax in [ax_spec, ax_wf, ax_iq, ax_time]:
        style(ax)

    # ── Spectrum ───────────────────────────────────────────────────────────────────
    ax_spec.set_xlim(freqs_mhz[0], freqs_mhz[-1])
    ax_spec.set_ylim(-120, 0)
    ax_spec.set_ylabel("Power (dBm)", color="white")
    ax_spec.set_xlabel("Frequency (MHz)", color="white")

    (line_live,) = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, -120.0), color="#00d4ff", lw=0.8, alpha=0.4, label="Live")
    (line_avg,) = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, -120.0), color="#ff6b35", lw=1.4, label=f"Avg x{AVERAGING}")
    (line_peak,) = ax_spec.plot(
        freqs_mhz,
        np.full(FFT_SIZE, -120.0),
        color="#a8ff3e",
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="Peak hold",
    )

    ax_spec.axvline(tone_mhz, color="#ffdd00", lw=1.0, ls="--", alpha=0.8, label=f"Expected {tone_mhz:.3f} MHz")
    ax_spec.axvline(CENTER_FREQ / 1e6, color="white", lw=0.5, ls=":", alpha=0.3)
    ax_spec.legend(loc="upper right", framealpha=0.25, facecolor="#1a1d24", labelcolor="white", fontsize=8)

    stats_text = ax_spec.text(
        0.01,
        0.97,
        "",
        transform=ax_spec.transAxes,
        color="#cccccc",
        fontsize=8,
        va="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#1a1d24", "alpha": 0.7},
    )

    # ── Waterfall ─────────────────────────────────────────────────────────────────
    WATERFALL_ROWS = 80
    waterfall_data = np.full((WATERFALL_ROWS, FFT_SIZE), -120.0)

    wf_img = ax_wf.imshow(
        waterfall_data,
        aspect="auto",
        origin="upper",
        extent=(freqs_mhz[0], freqs_mhz[-1], WATERFALL_ROWS, 0),
        vmin=-120,
        vmax=0,
        cmap="inferno",
        interpolation="nearest",
    )
    ax_wf.set_ylabel("Time (frames)", color="white")
    ax_wf.set_xlabel("Frequency (MHz)", color="white")
    ax_wf.axvline(tone_mhz, color="#ffdd00", lw=0.8, ls="--", alpha=0.6)
    cbar = fig.colorbar(wf_img, ax=ax_wf, orientation="vertical", pad=0.01)
    cbar.set_label("dBm", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # ── IQ Constellation ──────────────────────────────────────────────────────────
    ax_iq.set_xlim(-1.5, 1.5)
    ax_iq.set_ylim(-1.5, 1.5)
    ax_iq.set_xlabel("I", color="white")
    ax_iq.set_ylabel("Q", color="white")
    ax_iq.set_title("IQ Constellation (mixed to DC)", color="#aaaaaa", fontsize=9)
    ax_iq.axhline(0, color="#444", lw=0.6)
    ax_iq.axvline(0, color="#444", lw=0.6)
    ax_iq.set_aspect("equal")
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_iq.plot(np.cos(theta), np.sin(theta), color="#333355", lw=0.8, ls="--")
    iq_scatter = ax_iq.scatter([], [], c=[], cmap="plasma", s=4, vmin=0, vmax=1, alpha=0.6)

    # ── Time domain ───────────────────────────────────────────────────────────────
    SHOW_SAMPLES = 500
    t_axis_us = np.arange(SHOW_SAMPLES) / SAMPLE_RATE * 1e6

    ax_time.set_xlim(0, t_axis_us[-1])
    ax_time.set_ylim(-1.5, 1.5)
    ax_time.set_xlabel("Time (µs)", color="white")
    ax_time.set_ylabel("Amplitude (norm.)", color="white")
    ax_time.set_title("Time Domain (mixed to DC)", color="#aaaaaa", fontsize=9)

    (line_i,) = ax_time.plot(t_axis_us, np.zeros(SHOW_SAMPLES), color="#00d4ff", lw=1.0, label="I")
    (line_q,) = ax_time.plot(t_axis_us, np.zeros(SHOW_SAMPLES), color="#ff4dff", lw=1.0, label="Q")
    ax_time.legend(loc="upper right", framealpha=0.2, facecolor="#1a1d24", labelcolor="white", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # ─── Animation ────────────────────────────────────────────────────────────────
    LP_CUTOFF_HZ = 10e3

    _state: dict[str, object] = {
        "peak_hold": np.full(FFT_SIZE, -120.0),
        "buf_idx": 0,
        "waterfall_data": waterfall_data,
    }
    t_mix_full = np.arange(NUM_SAMPS) / SAMPLE_RATE  # pre-allocate mix vector

    def update(_frame: int) -> tuple[object, ...]:
        """Update all plot elements with fresh SDR data."""
        raw = cast("np.ndarray", sdr.rx())

        # ── Spectrum + waterfall ──────────────────────────────────────────────────
        psd = compute_psd(raw, window, FFT_SIZE, P_FULLSCALE_DBM, RX_GAIN)
        buf_idx = cast("int", _state["buf_idx"])
        psd_buffer[buf_idx % AVERAGING] = psd
        _state["buf_idx"] = buf_idx + 1
        psd_avg = np.mean(psd_buffer, axis=0)
        peak_hold = cast("np.ndarray", _state["peak_hold"])
        _state["peak_hold"] = np.maximum(peak_hold * 0.995, psd)

        wf = cast("np.ndarray", _state["waterfall_data"])
        wf = np.roll(wf, 1, axis=0)
        wf[0] = psd_avg
        _state["waterfall_data"] = wf

        line_live.set_ydata(psd)
        line_avg.set_ydata(psd_avg)
        line_peak.set_ydata(_state["peak_hold"])
        wf_img.set_data(_state["waterfall_data"])

        # ── Actual peak ───────────────────────────────────────────────────────────
        peak_idx = np.argmax(psd_avg)
        peak_freq = freqs_mhz[peak_idx]
        tone_pwr = psd_avg[peak_idx]
        noise_floor = np.percentile(psd_avg, 10)
        snr = tone_pwr - noise_floor
        detected_offset_hz = (peak_freq - CENTER_FREQ / 1e6) * 1e6

        stats_text.set_text(
            f"Peak: {peak_freq:.4f} MHz  |  {tone_pwr:.1f} dBm  |  Noise: {noise_floor:.1f} dBm  |  SNR: {snr:.1f} dB",
        )

        # ── Bandpass around detected peak (signed frequency) ─────────────────────
        filtered = bandpass_around_tone(cast("np.ndarray", raw), detected_offset_hz, SAMPLE_RATE, bw=20e3)

        # ── Mix down to DC so constellation doesn't spin ──────────────────────────
        filtered = filtered * np.exp(-2j * np.pi * detected_offset_hz * t_mix_full)

        # ── Low-pass at 10 kHz to clean up after mix ──────────────────────────────
        spectrum = np.fft.fft(filtered)
        lp_freqs = np.fft.fftfreq(len(filtered), d=1 / SAMPLE_RATE)
        spectrum[np.abs(lp_freqs) > LP_CUTOFF_HZ] = 0
        filtered = np.fft.ifft(spectrum)

        scale = np.max(np.abs(filtered)) + 1e-9
        filtered_norm = filtered / scale

        # ── IQ constellation ─────────────────────────────────────────────────────
        iq_pts = filtered_norm[-1000:]
        colors = np.linspace(0, 1, len(iq_pts))
        iq_scatter.set_offsets(np.column_stack([iq_pts.real, iq_pts.imag]))
        iq_scatter.set_array(colors)

        # ── Time domain ───────────────────────────────────────────────────────────
        seg = filtered_norm[:SHOW_SAMPLES]
        line_i.set_ydata(seg.real)
        line_q.set_ydata(seg.imag)

        return (line_live, line_avg, line_peak, wf_img, stats_text, iq_scatter, line_i, line_q)

    ani = animation.FuncAnimation(fig, cast("Any", update), interval=UPDATE_MS, blit=True, cache_frame_data=False)
    plt.show()
