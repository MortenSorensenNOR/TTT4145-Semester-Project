import numpy as np
import adi

center_freq = 915e6     # Hz
num_samps   = 100000
tx_gain     = -10
rx_gain     = 0.0
num_flush   = 10

# Tone is placed at 10% of sample rate — well within the passband
TONE_RATIO  = 0.10

# Sample rates to sweep (Hz)
SAMPLE_RATES = [
    5_336_000,
]

sdr = adi.Pluto("ip:192.168.2.1")

results = []

for fs in SAMPLE_RATES:
    tone_freq = fs * TONE_RATIO

    # ---- Configure ----
    sdr.sample_rate          = int(fs)
    sdr.tx_rf_bandwidth      = int(fs)
    sdr.rx_rf_bandwidth      = int(fs)
    sdr.tx_lo                = int(center_freq)
    sdr.rx_lo                = int(center_freq)
    sdr.tx_hardwaregain_chan0 = tx_gain
    sdr.rx_hardwaregain_chan0 = rx_gain
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_buffer_size       = num_samps
    sdr.tx_cyclic_buffer     = True

    # ---- Generate tone ----
    t = np.arange(num_samps) / fs
    tx_samples = np.exp(1j * 2 * np.pi * tone_freq * t)
    tx_samples = (tx_samples * 2**14 * 0.7).astype(np.complex64)

    sdr.tx(tx_samples)

    # ---- Flush stale buffers ----
    for _ in range(num_flush):
        sdr.rx()

    rx_samples = sdr.rx().astype(np.complex64)

    sdr.tx_destroy_buffer()

    # ---- FFT peak detection ----
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(rx_samples))) ** 2
    freqs    = np.fft.fftshift(np.fft.fftfreq(len(rx_samples), d=1/fs))
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]

    # Tolerance: 2 FFT bins
    bin_width = fs / num_samps
    error_hz  = abs(peak_freq - tone_freq)
    passed    = error_hz <= 2 * bin_width

    results.append((fs, tone_freq, peak_freq, error_hz, passed))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] fs={fs/1e6:.2f} MHz  tone={tone_freq/1e3:.1f} kHz  "
          f"peak={peak_freq/1e3:.1f} kHz  err={error_hz:.1f} Hz")

# ---- Summary ----
print()
passed_count = sum(r[4] for r in results)
print(f"Result: {passed_count}/{len(results)} passed")
