"""Real-time spectrum analyzer and IQ visualizer for the PlutoSDR."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes

from pluto import create_pluto
from pluto.config import CENTER_FREQ, RX_GAIN, SAMPLE_RATE


def compute_psd(
    samples: np.ndarray[np.complex64, Any], # Changed type hint
    window: np.ndarray[np.float32, Any], # Changed type hint
    fft_size: int,
    p_fullscale_dbm: np.float32, # Changed type hint
    rx_gain: np.float32, # Changed type hint
) -> np.ndarray[np.float32, Any]: # Added return type hint
    """Compute the power spectral density of the given samples."""
    frame = samples[:fft_size] * window
    spectrum = np.fft.fftshift(np.fft.fft(frame, n=fft_size))
    power_norm = (np.abs(spectrum) / fft_size) ** np.float32(2) # Explicitly cast to np.float32
    return np.float32(10) * np.log10(power_norm + np.float32(1e-20)) + p_fullscale_dbm - rx_gain # Explicitly cast to np.float32


def bandpass_around_tone(samples: np.ndarray[np.complex64, Any], offset_hz: np.float32, sample_rate: np.float32, bw: np.float32 = np.float32(20e3)) -> np.ndarray[np.complex64, Any]: # Changed type hints and default
    """Signed bandpass - only matches the correct +/- frequency bin."""
    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=np.float32(1) / sample_rate)
    mask = np.abs(freqs - offset_hz) > bw / np.float32(2) # Explicitly cast to np.float32
    spectrum[mask] = np.float32(0) # Explicitly cast to np.float32
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

    TONE_FREQ = np.float32(100e3) # Changed to np.float32
    ATTENUATOR_DB = np.float32(30.0) # Changed to np.float32
    P_FULLSCALE_DBM = np.float32(10) * np.log10(np.float32(5)) # Changed to np.float32

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
    freqs_mhz = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=np.float32(1) / SAMPLE_RATE)) + CENTER_FREQ) / np.float32(1e6) # Explicitly cast to np.float32
    tone_mhz = (CENTER_FREQ + TONE_FREQ) / np.float32(1e6) # Explicitly cast to np.float32

    window = np.hanning(FFT_SIZE).astype(np.float32) # Ensure np.float32
    psd_buffer = np.full((AVERAGING, FFT_SIZE), np.float32(-120.0)) # Explicitly cast to np.float32

    # ─── Figure ───────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 11))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(
        f"PlutoSDR RX  |  {CENTER_FREQ / np.float32(1e6):.1f} MHz  |  Tone @ {tone_mhz:.3f} MHz  |  " # Explicitly cast to np.float32
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
    ax_spec.set_ylim(np.float32(-120), np.float32(0)) # Explicitly cast to np.float32
    ax_spec.set_ylabel("Power (dBm)", color="white")
    ax_spec.set_xlabel("Frequency (MHz)", color="white")

    (line_live,) = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, np.float32(-120.0)), color="#00d4ff", lw=np.float32(0.8), alpha=np.float32(0.4), label="Live") # Explicitly cast to np.float32
    (line_avg,) = ax_spec.plot(freqs_mhz, np.full(FFT_SIZE, np.float32(-120.0)), color="#ff6b35", lw=np.float32(1.4), label=f"Avg x{AVERAGING}") # Explicitly cast to np.float32
    (line_peak,) = ax_spec.plot(
        freqs_mhz,
        np.full(FFT_SIZE, np.float32(-120.0)), # Explicitly cast to np.float32
        color="#a8ff3e",
        lw=np.float32(0.8), # Explicitly cast to np.float32
        ls="--",
        alpha=np.float32(0.7), # Explicitly cast to np.float32
        label="Peak hold",
    )

    ax_spec.axvline(tone_mhz, color="#ffdd00", lw=np.float32(1.0), ls="--", alpha=np.float32(0.8), label=f"Expected {tone_mhz:.3f} MHz") # Explicitly cast to np.float32
    ax_spec.axvline(CENTER_FREQ / np.float32(1e6), color="white", lw=np.float32(0.5), ls=":", alpha=np.float32(0.3)) # Explicitly cast to np.float32
    ax_spec.legend(loc="upper right", framealpha=np.float32(0.25), facecolor="#1a1d24", labelcolor="white", fontsize=8) # Explicitly cast to np.float32

    stats_text = ax_spec.text(
        np.float32(0.01), # Explicitly cast to np.float32
        np.float32(0.97), # Explicitly cast to np.float32
        "",
        transform=ax_spec.transAxes,
        color="#cccccc",
        fontsize=8,
        va="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#1a1d24", "alpha": np.float32(0.7)}, # Explicitly cast to np.float32
    )

    # ── Waterfall ─────────────────────────────────────────────────────────────────
    WATERFALL_ROWS = 80
    waterfall_data = np.full((WATERFALL_ROWS, FFT_SIZE), np.float32(-120.0)) # Explicitly cast to np.float32

    wf_img = ax_wf.imshow(
        waterfall_data,
        aspect="auto",
        origin="upper",
        extent=(freqs_mhz[0], freqs_mhz[-1], WATERFALL_ROWS, np.float32(0)), # Explicitly cast to np.float32
        vmin=np.float32(-120), # Explicitly cast to np.float32
        vmax=np.float32(0), # Explicitly cast to np.float32
        cmap="inferno",
        interpolation="nearest",
    )
    ax_wf.set_ylabel("Time (frames)", color="white")
    ax_wf.set_xlabel("Frequency (MHz)", color="white")
    ax_wf.axvline(tone_mhz, color="#ffdd00", lw=np.float32(0.8), ls="--", alpha=np.float32(0.6)) # Explicitly cast to np.float32
    cbar = fig.colorbar(wf_img, ax=ax_wf, orientation="vertical", pad=np.float32(0.01)) # Explicitly cast to np.float32
    cbar.set_label("dBm", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # ── IQ Constellation ──────────────────────────────────────────────────────────
    ax_iq.set_xlim(np.float32(-1.5), np.float32(1.5)) # Explicitly cast to np.float32
    ax_iq.set_ylim(np.float32(-1.5), np.float32(1.5)) # Explicitly cast to np.float32
    ax_iq.set_xlabel("I", color="white")
    ax_iq.set_ylabel("Q", color="white")
    ax_iq.set_title("IQ Constellation (mixed to DC)", color="#aaaaaa", fontsize=9)
    ax_iq.axhline(np.float32(0), color="#444", lw=np.float32(0.6)) # Explicitly cast to np.float32
    ax_iq.axvline(np.float32(0), color="#444", lw=np.float32(0.6)) # Explicitly cast to np.float32
    ax_iq.set_aspect("equal")
    theta = np.linspace(np.float32(0), np.float32(2) * np.float32(np.pi), np.int32(200), dtype=np.float32) # Explicitly cast to np.float32
    ax_iq.plot(np.cos(theta), np.sin(theta), color="#333355", lw=np.float32(0.8), ls="--") # Explicitly cast to np.float32
    iq_scatter = ax_iq.scatter([], [], c=[], cmap="plasma", s=np.float32(4), vmin=np.float32(0), vmax=np.float32(1), alpha=np.float32(0.6)) # Explicitly cast to np.float32

    # ── Time domain ───────────────────────────────────────────────────────────────
    SHOW_SAMPLES = 500
    t_axis_us = np.arange(SHOW_SAMPLES).astype(np.float32) / SAMPLE_RATE * np.float32(1e6) # Explicitly cast to np.float32

    ax_time.set_xlim(np.float32(0), t_axis_us[-np.int32(1)]) # Explicitly cast to np.float32
    ax_time.set_ylim(np.float32(-1.5), np.float32(1.5)) # Explicitly cast to np.float32
    ax_time.set_xlabel("Time (µs)", color="white")
    ax_time.set_ylabel("Amplitude (norm.)", color="white")
    ax_time.set_title("Time Domain (mixed to DC)", color="#aaaaaa", fontsize=9)

    (line_i,) = ax_time.plot(t_axis_us, np.zeros(SHOW_SAMPLES, dtype=np.float32), color="#00d4ff", lw=np.float32(1.0), label="I") # Explicitly cast to np.float32
    (line_q,) = ax_time.plot(t_axis_us, np.zeros(SHOW_SAMPLES, dtype=np.float32), color="#ff4dff", lw=np.float32(1.0), label="Q") # Explicitly cast to np.float32
    ax_time.legend(loc="upper right", framealpha=np.float32(0.2), facecolor="#1a1d24", labelcolor="white", fontsize=8) # Explicitly cast to np.float32

    plt.tight_layout(rect=(np.float32(0), np.float32(0), np.float32(1), np.float32(0.97))) # Explicitly cast to np.float32

    # ─── Animation ────────────────────────────────────────────────────────────────
    LP_CUTOFF_HZ = np.float32(10e3) # Changed to np.float32

    _state: dict[str, object] = {
        "peak_hold": np.full(FFT_SIZE, np.float32(-120.0)), # Explicitly cast to np.float32
        "buf_idx": 0,
        "waterfall_data": waterfall_data,
    }
    t_mix_full = np.arange(NUM_SAMPS).astype(np.float32) / SAMPLE_RATE  # pre-allocate mix vector # Explicitly cast to np.float32

    def update(_frame: int) -> tuple[object, ...]:
        """Update all plot elements with fresh SDR data."""
        raw = cast("np.ndarray[np.complex64, Any]", sdr.rx().astype(np.complex64)) # Ensure np.complex64

        # ── Spectrum + waterfall ──────────────────────────────────────────────────
        psd = compute_psd(raw, window, FFT_SIZE, P_FULLSCALE_DBM, RX_GAIN)
        buf_idx = cast("int", _state["buf_idx"])
        psd_buffer[buf_idx % AVERAGING] = psd
        _state["buf_idx"] = buf_idx + 1
        psd_avg = np.mean(psd_buffer, axis=0)
        peak_hold = cast("np.ndarray", _state["peak_hold"])
        _state["peak_hold"] = np.maximum(peak_hold * np.float32(0.995), psd) # Explicitly cast to np.float32

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
        noise_floor = np.percentile(psd_avg, np.float32(10)) # Explicitly cast to np.float32
        snr = tone_pwr - noise_floor
        detected_offset_hz = (peak_freq - CENTER_FREQ / np.float32(1e6)) * np.float32(1e6) # Explicitly cast to np.float32

        stats_text.set_text(
            f"Peak: {peak_freq:.4f} MHz  |  {tone_pwr:.1f} dBm  |  Noise: {noise_floor:.1f} dBm  |  SNR: {snr:.1f} dB",
        )

        # ── Bandpass around detected peak (signed frequency) ─────────────────────
        filtered = bandpass_around_tone(raw, detected_offset_hz, SAMPLE_RATE, bw=np.float32(20e3)) # Explicitly cast to np.float32

        # ── Mix down to DC so constellation doesn't spin ──────────────────────────
        filtered = filtered * np.exp(np.complex64(-2j) * np.float32(np.pi) * detected_offset_hz * t_mix_full) # Explicitly cast to np.float32

        # ── Low-pass at 10 kHz to clean up after mix ──────────────────────────────
        spectrum = np.fft.fft(filtered)
        lp_freqs = np.fft.fftfreq(len(filtered), d=np.float32(1) / SAMPLE_RATE) # Explicitly cast to np.float32
        spectrum[np.abs(lp_freqs) > LP_CUTOFF_HZ] = np.float32(0) # Explicitly cast to np.float32
        filtered = np.fft.ifft(spectrum)

        scale = np.max(np.abs(filtered)) + np.float32(1e-9) # Explicitly cast to np.float32
        filtered_norm = filtered / scale

        # ── IQ constellation ─────────────────────────────────────────────────────
        iq_pts = filtered_norm[-np.int32(1000):] # Explicitly cast to np.int32
        colors = np.linspace(np.float32(0), np.float32(1), len(iq_pts)) # Explicitly cast to np.float32
        iq_scatter.set_offsets(np.column_stack([iq_pts.real, iq_pts.imag]))
        iq_scatter.set_array(colors)

        # ── Time domain ───────────────────────────────────────────────────────────
        seg = filtered_norm[:SHOW_SAMPLES]
        line_i.set_ydata(seg.real)
        line_q.set_ydata(seg.imag)

        return (line_live, line_avg, line_peak, wf_img, stats_text, iq_scatter, line_i, line_q)

    ani = animation.FuncAnimation(fig, cast("Any", update), interval=UPDATE_MS, blit=True, cache_frame_data=False)
    plt.show()