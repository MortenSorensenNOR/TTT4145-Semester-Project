import adi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# ─── Config ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 1e6
CENTER_FREQ = 2400e6
NUM_SAMPS = 10000
RX_GAIN = 70.0
FFT_SIZE = 1024
AVERAGING = 10
UPDATE_MS = 50

TONE_FREQ = 100e3
ATTENUATOR_DB = 30.0
P_FULLSCALE_DBM = 10 * np.log10(5)

# ─── Connect ──────────────────────────────────────────────────────────────────
print("Connecting to PlutoSDR RX...")
sdr = adi.Pluto("ip:192.168.2.1")
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = RX_GAIN
sdr.rx_lo = int(CENTER_FREQ)
sdr.sample_rate = int(SAMPLE_RATE)
sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
sdr.rx_buffer_size = NUM_SAMPS
print("Connected!\n")

# ─── Helpers ──────────────────────────────────────────────────────────────────
freqs_mhz = (
    np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / SAMPLE_RATE)) + CENTER_FREQ
) / 1e6
tone_mhz = (CENTER_FREQ + TONE_FREQ) / 1e6

window = np.hanning(FFT_SIZE)
psd_buffer = np.full((AVERAGING, FFT_SIZE), -120.0)
buf_idx = 0


def compute_psd(samples):
    frame = samples[:FFT_SIZE] * window
    spectrum = np.fft.fftshift(np.fft.fft(frame, n=FFT_SIZE))
    power_norm = (np.abs(spectrum) / FFT_SIZE) ** 2
    return 10 * np.log10(power_norm + 1e-20) + P_FULLSCALE_DBM - RX_GAIN


def bandpass_around_tone(samples, tone_freq, bw=20e3):
    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), d=1 / SAMPLE_RATE)
    mask = np.abs(np.abs(freqs) - tone_freq) > bw / 2
    spectrum[mask] = 0
    return np.fft.ifft(spectrum)


# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 11))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    f"PlutoSDR RX  |  {CENTER_FREQ / 1e6:.1f} MHz  |  Tone @ {tone_mhz:.3f} MHz  |  "
    f"TX gain −50 dBm  |  {ATTENUATOR_DB:.0f} dB attenuator  |  RX gain {RX_GAIN:.0f} dB",
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


def style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.grid(True, color="#2a2d35", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")


for ax in [ax_spec, ax_wf, ax_iq, ax_time]:
    style(ax)

# ── Spectrum ───────────────────────────────────────────────────────────────────
ax_spec.set_xlim(freqs_mhz[0], freqs_mhz[-1])
ax_spec.set_ylim(-120, 0)  # ← changed
ax_spec.set_ylabel("Power (dBm)", color="white")
ax_spec.set_xlabel("Frequency (MHz)", color="white")

(line_live,) = ax_spec.plot(
    freqs_mhz,
    np.full(FFT_SIZE, -120.0),
    color="#00d4ff",
    lw=0.8,
    alpha=0.4,
    label="Live",
)
(line_avg,) = ax_spec.plot(
    freqs_mhz,
    np.full(FFT_SIZE, -120.0),
    color="#ff6b35",
    lw=1.4,
    label=f"Avg x{AVERAGING}",
)
(line_peak,) = ax_spec.plot(
    freqs_mhz,
    np.full(FFT_SIZE, -120.0),
    color="#a8ff3e",
    lw=0.8,
    ls="--",
    alpha=0.7,
    label="Peak hold",
)

ax_spec.axvline(
    tone_mhz,
    color="#ffdd00",
    lw=1.0,
    ls="--",
    alpha=0.8,
    label=f"Expected {tone_mhz:.3f} MHz",
)
ax_spec.axvline(CENTER_FREQ / 1e6, color="white", lw=0.5, ls=":", alpha=0.3)

ax_spec.legend(
    loc="upper right",
    framealpha=0.25,
    facecolor="#1a1d24",
    labelcolor="white",
    fontsize=8,
)

stats_text = ax_spec.text(
    0.01,
    0.97,
    "",
    transform=ax_spec.transAxes,
    color="#cccccc",
    fontsize=8,
    va="top",
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d24", alpha=0.7),
)

# ── Waterfall ─────────────────────────────────────────────────────────────────
WATERFALL_ROWS = 80
waterfall_data = np.full((WATERFALL_ROWS, FFT_SIZE), -120.0)

wf_img = ax_wf.imshow(
    waterfall_data,
    aspect="auto",
    origin="upper",
    extent=[freqs_mhz[0], freqs_mhz[-1], WATERFALL_ROWS, 0],
    vmin=-120,
    vmax=0,  # ← changed
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
ax_iq.set_title("IQ Constellation (filtered tone)", color="#aaaaaa", fontsize=9)
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
ax_time.set_title("Time Domain (filtered tone)", color="#aaaaaa", fontsize=9)

(line_i,) = ax_time.plot(
    t_axis_us, np.zeros(SHOW_SAMPLES), color="#00d4ff", lw=1.0, label="I"
)
(line_q,) = ax_time.plot(
    t_axis_us, np.zeros(SHOW_SAMPLES), color="#ff4dff", lw=1.0, label="Q"
)
ax_time.legend(
    loc="upper right",
    framealpha=0.2,
    facecolor="#1a1d24",
    labelcolor="white",
    fontsize=8,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

# ─── Animation ────────────────────────────────────────────────────────────────
peak_hold = np.full(FFT_SIZE, -120.0)


def update(_frame):
    global buf_idx, peak_hold, waterfall_data

    raw = sdr.rx()

    # ── Spectrum + waterfall ──────────────────────────────────────────────────
    psd = compute_psd(raw)
    psd_buffer[buf_idx % AVERAGING] = psd
    buf_idx += 1
    psd_avg = np.mean(psd_buffer, axis=0)
    peak_hold = np.maximum(peak_hold * 0.995, psd)

    waterfall_data = np.roll(waterfall_data, 1, axis=0)
    waterfall_data[0] = psd_avg

    line_live.set_ydata(psd)
    line_avg.set_ydata(psd_avg)
    line_peak.set_ydata(peak_hold)
    wf_img.set_data(waterfall_data)

    # ── Actual peak across full spectrum ─────────────────────────────────────
    peak_idx = np.argmax(psd_avg)  # ← actual peak, not near tone
    peak_freq = freqs_mhz[peak_idx]  # ← actual frequency
    tone_pwr = psd_avg[peak_idx]
    noise_floor = np.percentile(psd_avg, 10)
    snr = tone_pwr - noise_floor

    stats_text.set_text(
        f"Peak: {peak_freq:.4f} MHz  |  {tone_pwr:.1f} dBm  |  "
        f"Noise: {noise_floor:.1f} dBm  |  SNR: {snr:.1f} dB"
    )

    # ── Filtered tone -> IQ + time ────────────────────────────────────────────
    filtered = bandpass_around_tone(raw, TONE_FREQ, bw=20e3)
    scale = np.max(np.abs(filtered)) + 1e-9
    filtered_norm = filtered / scale

    iq_pts = filtered_norm[-1000:]
    colors = np.linspace(0, 1, len(iq_pts))
    iq_scatter.set_offsets(np.column_stack([iq_pts.real, iq_pts.imag]))
    iq_scatter.set_array(colors)

    seg = filtered_norm[:SHOW_SAMPLES]
    line_i.set_ydata(seg.real)
    line_q.set_ydata(seg.imag)

    return (
        line_live,
        line_avg,
        line_peak,
        wf_img,
        stats_text,
        iq_scatter,
        line_i,
        line_q,
    )


ani = animation.FuncAnimation(
    fig, update, interval=UPDATE_MS, blit=True, cache_frame_data=False
)
print("Live plot running – close the window to stop.")
plt.show()

del sdr
print("Done.")
