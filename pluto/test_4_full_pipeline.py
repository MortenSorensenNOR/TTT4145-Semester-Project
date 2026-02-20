"""Full pipeline test for PlutoSDR loopback.

End-to-end test with framing, channel coding, and data recovery.
Measures BER vs SNR for different modulation schemes.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "examples"

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.modulation import BPSK
from modules.pilots import insert_pilots, PilotConfig, data_indices
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import Synchronizer, SynchronizerConfig, build_preamble
from pluto.config import get_modulator
from pluto.loopback import setup_pluto, transmit_and_receive, SPS, SAMPLE_RATE, CENTER_FREQ
from pluto.receive import FrameDecoder
from pluto.config import PipelineConfig

RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
GUARD_SAMPLES = 500


def add_awgn(signal: np.ndarray, snr_db: float, verbose: bool = False, return_noise: bool = False):
    """Add AWGN noise to achieve target SNR (in dB).

    Measures signal power from the strongest portion (where the frame is),
    not the entire buffer which includes quiet regions.
    """
    # Find the active region by looking at where power is significant
    window = 1000
    if len(signal) > window:
        # Sliding window to find max power region
        power_profile = np.convolve(np.abs(signal)**2, np.ones(window)/window, mode='valid')
        peak_idx = np.argmax(power_profile)
        # Measure power in region around peak
        start = max(0, peak_idx)
        end = min(len(signal), peak_idx + window * 4)
        sig_power = np.mean(np.abs(signal[start:end]) ** 2)
        active_region = (start, end)
    else:
        sig_power = np.mean(np.abs(signal) ** 2)
        active_region = (0, len(signal))

    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    noisy_signal = signal + noise

    if verbose:
        # Verify actual SNR in active region
        actual_noise_power = np.mean(np.abs(noise[active_region[0]:active_region[1]]) ** 2)
        actual_snr = 10 * np.log10(sig_power / actual_noise_power)
        print(f"    [NOISE] sig_pwr={sig_power:.2e}, noise_pwr={actual_noise_power:.2e}, "
              f"target_SNR={snr_db:.1f}dB, actual_SNR={actual_snr:.1f}dB, "
              f"active_region={active_region[0]}-{active_region[1]}")

    if return_noise:
        return noisy_signal, noise, active_region
    return noisy_signal


def build_test_frame(
    payload_bits: np.ndarray,
    fc: FrameConstructor,
    sync_config: SynchronizerConfig,
    pilot_config: PilotConfig,
    pipeline: PipelineConfig,
    mod_scheme: ModulationSchemes = ModulationSchemes.QPSK,
    coding_rate: CodeRates = CodeRates.HALF_RATE,
) -> tuple[np.ndarray, FrameHeader]:
    """Build a complete TX frame with preamble, header, and encoded payload."""
    header = FrameHeader(
        length=len(payload_bits),
        src=0,
        dst=0,
        frame_type=0,
        mod_scheme=mod_scheme,
        coding_rate=coding_rate,
        sequence_number=0,
    )

    header_encoded, payload_encoded = fc.encode(
        header,
        payload_bits,
        channel_coding=pipeline.channel_coding,
        interleaving=pipeline.interleaving,
    )

    bpsk = BPSK()
    header_symbols = bpsk.bits2symbols(header_encoded)
    payload_symbols = get_modulator(mod_scheme).bits2symbols(payload_encoded)

    if pipeline.pilots:
        payload_symbols = insert_pilots(payload_symbols, pilot_config)

    preamble = build_preamble(sync_config)
    frame = np.concatenate([preamble, header_symbols, payload_symbols])

    return frame, header


def run_ber_test(
    mod_scheme: ModulationSchemes,
    snr_db: float,
    n_payload_bits: int,
    sdr,
    h_rrc: np.ndarray,
    sync: Synchronizer,
    fc: FrameConstructor,
    pilot_config: PilotConfig,
    pipeline: PipelineConfig,
    cfo_hz: int = 0,
    verbose: bool = False,
    timing: bool = False,
) -> dict:
    """Run a single BER test at given SNR."""
    timings = {}

    t0 = time.perf_counter()
    tx_bits = np.random.randint(0, 2, n_payload_bits)

    sync_config = sync.config
    frame_symbols, header = build_test_frame(
        tx_bits, fc, sync_config, pilot_config, pipeline, mod_scheme, CodeRates.HALF_RATE
    )

    tx_signal = upsample_and_filter(frame_symbols, SPS, h_rrc)
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    tx_signal = np.concatenate([zeros, tx_signal, zeros])
    timings["tx_build"] = time.perf_counter() - t0

    # Transmit and receive through Pluto
    t0 = time.perf_counter()
    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=5)
    timings["pluto_txrx"] = time.perf_counter() - t0

    # Add AWGN noise
    t0 = time.perf_counter()
    rx_noisy = add_awgn(rx_raw, snr_db, verbose=verbose)
    timings["add_noise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rx_filtered = np.convolve(rx_noisy, h_rrc, mode="same")
    timings["matched_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoder = FrameDecoder(sync, fc, SAMPLE_RATE, SPS, pilot_config, pipeline)
    result = decoder.try_decode(rx_filtered, abs_offset=0)
    timings["decode_total"] = time.perf_counter() - t0

    if timing:
        print(f"    [TIMING] tx_build={timings['tx_build']*1000:.1f}ms, "
              f"pluto={timings['pluto_txrx']*1000:.1f}ms, "
              f"filter={timings['matched_filter']*1000:.1f}ms, "
              f"decode={timings['decode_total']*1000:.1f}ms")

    if result is None:
        return {"success": False, "ber": 0.5, "errors": n_payload_bits // 2, "timings": timings}

    rx_bits = result.payload_bits
    if len(rx_bits) != len(tx_bits):
        return {"success": False, "ber": 0.5, "errors": n_payload_bits // 2, "timings": timings}

    errors = np.sum(tx_bits != rx_bits)
    ber = errors / len(tx_bits)

    return {
        "success": True,
        "ber": ber,
        "errors": errors,
        "cfo_est": result.cfo_hz,
        "rx_filtered": rx_filtered,
        "sync_result": decoder.sync.detect_preamble(rx_filtered, SAMPLE_RATE),
        "timings": timings,
    }


def plot_signal_vs_noise(signal: np.ndarray, noise: np.ndarray, active_region: tuple, snr_db: float):
    """Plot signal and noise to visually verify noise levels."""
    start, end = active_region
    # Show a portion around the active region
    plot_start = max(0, start - 500)
    plot_end = min(len(signal), end + 500)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    t = np.arange(plot_start, plot_end)

    # Clean signal (before noise)
    clean = signal[plot_start:plot_end] - noise[plot_start:plot_end]
    ax = axes[0]
    ax.plot(t, np.real(clean), 'b-', linewidth=0.5, alpha=0.7, label='I')
    ax.plot(t, np.imag(clean), 'r-', linewidth=0.5, alpha=0.7, label='Q')
    ax.axvline(start, color='g', linestyle='--', alpha=0.5, label='Active region')
    ax.axvline(end, color='g', linestyle='--', alpha=0.5)
    ax.set_title(f"Clean Signal (before noise)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Noise only
    ax = axes[1]
    ax.plot(t, np.real(noise[plot_start:plot_end]), 'b-', linewidth=0.5, alpha=0.7, label='I')
    ax.plot(t, np.imag(noise[plot_start:plot_end]), 'r-', linewidth=0.5, alpha=0.7, label='Q')
    ax.axvline(start, color='g', linestyle='--', alpha=0.5)
    ax.axvline(end, color='g', linestyle='--', alpha=0.5)
    noise_std = np.std(noise[start:end])
    ax.set_title(f"Noise (σ={noise_std:.4f})")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Noisy signal
    ax = axes[2]
    ax.plot(t, np.real(signal[plot_start:plot_end]), 'b-', linewidth=0.5, alpha=0.7, label='I')
    ax.plot(t, np.imag(signal[plot_start:plot_end]), 'r-', linewidth=0.5, alpha=0.7, label='Q')
    ax.axvline(start, color='g', linestyle='--', alpha=0.5)
    ax.axvline(end, color='g', linestyle='--', alpha=0.5)
    ax.set_title(f"Noisy Signal (SNR={snr_db:.1f} dB)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Signal vs Noise Comparison (SNR={snr_db:.1f} dB)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_constellation_and_eye(rx_filtered: np.ndarray, sync_result, mod_scheme: ModulationSchemes, snr_db: float, title: str):
    """Plot constellation and eye diagram for a single test."""
    pilot_config = PilotConfig()

    if not sync_result.success:
        return None

    # CFO correction
    cfo_hz = sync_result.cfo_hat_hz
    n_vec = np.arange(len(rx_filtered))
    rx_corrected = rx_filtered * np.exp(-1j * 2 * np.pi * cfo_hz / SAMPLE_RATE * n_vec)

    # Extract signal from after long ZC preamble
    data_start = sync_result.long_zc_start + 139 * SPS
    rx_data = rx_corrected[data_start:]

    # Phase correction using BPSK header
    n_header_symbols = 72
    header_symbols = rx_data[::SPS][:n_header_symbols]
    phase_est = np.angle(np.sum(header_symbols * np.sign(np.real(header_symbols))))
    rx_data = rx_data * np.exp(-1j * phase_est)

    # Normalize
    power = np.mean(np.abs(rx_data[:1000]) ** 2)
    if power > 0:
        rx_data = rx_data / np.sqrt(power)

    # Skip header
    payload_start = n_header_symbols * SPS
    rx_payload = rx_data[payload_start:]

    # Filter out pilots
    payload_symbols_all = rx_payload[::SPS]
    n_payload_symbols = len(payload_symbols_all)
    n_data_est = int(n_payload_symbols * pilot_config.spacing / (pilot_config.spacing + 1))
    if n_data_est > 0:
        d_idx = data_indices(n_data_est, pilot_config)
        d_idx = d_idx[d_idx < len(payload_symbols_all)]
        payload_data_only = payload_symbols_all[d_idx]
    else:
        payload_data_only = payload_symbols_all

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Eye diagram - I
    ax = axes[0]
    eye_len = 2 * SPS
    t_eye = np.linspace(-1, 1, eye_len)
    n_traces = min(200, len(rx_payload) // SPS - 2)
    for i in range(n_traces):
        start = i * SPS
        if start + eye_len <= len(rx_payload):
            ax.plot(t_eye, np.real(rx_payload[start:start + eye_len]), 'b-', alpha=0.05, linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f"Eye Diagram (I)")
    ax.set_xlabel("Symbol Period")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)

    # Eye diagram - Q
    ax = axes[1]
    for i in range(n_traces):
        start = i * SPS
        if start + eye_len <= len(rx_payload):
            ax.plot(t_eye, np.imag(rx_payload[start:start + eye_len]), 'b-', alpha=0.05, linewidth=0.5)
    ax.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_title(f"Eye Diagram (Q)")
    ax.set_xlabel("Symbol Period")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3)

    # Constellation
    ax = axes[2]
    symbols = payload_data_only[:min(n_traces, len(payload_data_only))]
    modulator = get_modulator(mod_scheme)
    constellation = modulator.symbol_mapping / np.sqrt(np.mean(np.abs(modulator.symbol_mapping) ** 2))
    ax.scatter(np.real(symbols), np.imag(symbols), alpha=0.3, s=5, label="RX")
    ax.scatter(np.real(constellation), np.imag(constellation), c="red", s=100, marker="x", label="Ideal")
    ax.set_title(f"Constellation - {mod_scheme.name}")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(f"{title} (SNR={snr_db} dB)", fontsize=12)
    plt.tight_layout()
    return fig


def main():
    print("Full Pipeline BER vs SNR Test")
    print("=" * 60)

    # Setup
    pipeline = PipelineConfig()
    sync_config = SynchronizerConfig()
    pilot_config = PilotConfig()
    fc = FrameConstructor()
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(sync_config, sps=SPS, rrc_taps=h_rrc)

    # Random CFO
    max_cfo = 3000
    cfo_hz = np.random.randint(-max_cfo, max_cfo + 1)
    print(f"Using CFO offset: {cfo_hz:+d} Hz")

    sdr = setup_pluto()
    if cfo_hz != 0:
        sdr.rx_lo = int(CENTER_FREQ - cfo_hz)

    # Test parameters - fine granularity, 0.2 dB steps
    snr_values = list(np.arange(-6, 4.1, 0.5))
    mod_schemes = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.QAM16]
    n_payload_bits = 200
    n_trials = 3  # Average over multiple trials per SNR point

    # Collect BER results
    ber_results = {mod: [] for mod in mod_schemes}
    plot_data = {}  # Store one result per modulation for plotting

    for mod_scheme in mod_schemes:
        print(f"\n{'='*60}")
        print(f"Testing {mod_scheme.name}")
        print("=" * 60)

        for snr_db in snr_values:
            bers = []
            last_result = None
            failures = 0

            for trial in range(n_trials):
                # Print timing/noise info only on first trial of first SNR per modulation
                first_snr = (snr_db == snr_values[0])
                result = run_ber_test(
                    mod_scheme, snr_db, n_payload_bits,
                    sdr, h_rrc, sync, fc, pilot_config, pipeline, cfo_hz,
                    verbose=(trial == 0 and first_snr),
                    timing=(trial == 0 and first_snr),
                )
                if result["success"]:
                    bers.append(result["ber"])
                    last_result = result
                else:
                    failures += 1

            # Only count successful decodes for BER average
            if bers:
                avg_ber = np.mean(bers)
            else:
                avg_ber = 0.5  # All failed
            ber_results[mod_scheme].append(avg_ber)
            fail_str = f" ({failures} sync fails)" if failures > 0 else ""
            print(f"  SNR={snr_db:+5.1f} dB: BER={avg_ber:.4f}{fail_str}")

            # Store result for mid-range SNR for plotting
            if snr_db == 4 and last_result and last_result["success"]:
                plot_data[mod_scheme] = (last_result, snr_db)

    # Plot BER vs SNR
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {'BPSK': 'o', 'QPSK': 's', 'QAM16': '^'}
    colors = {'BPSK': 'blue', 'QPSK': 'green', 'QAM16': 'red'}

    for mod_scheme in mod_schemes:
        bers = ber_results[mod_scheme]
        # Replace zeros with small value for log plot
        bers_plot = [max(b, 1e-5) for b in bers]
        ax.semilogy(snr_values, bers_plot,
                   marker=markers[mod_scheme.name],
                   color=colors[mod_scheme.name],
                   label=mod_scheme.name,
                   linewidth=2, markersize=8)

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title(f"BER vs SNR - Full Pipeline with LDPC (CFO={cfo_hz:+d} Hz)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-5, 1)

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Generate noise diagnostic plot at low SNR
    print("\nGenerating noise diagnostic plot...")
    diag_snr = -4.0
    tx_bits = np.random.randint(0, 2, n_payload_bits)
    frame_symbols, _ = build_test_frame(
        tx_bits, fc, sync_config, pilot_config, pipeline,
        ModulationSchemes.QPSK, CodeRates.HALF_RATE
    )
    tx_signal = upsample_and_filter(frame_symbols, SPS, h_rrc)
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    tx_signal = np.concatenate([zeros, tx_signal, zeros])
    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=5)
    rx_noisy, noise, active_region = add_awgn(rx_raw, diag_snr, verbose=True, return_noise=True)

    fig_noise = plot_signal_vs_noise(rx_noisy, noise, active_region, diag_snr)
    noise_path = os.path.join(PLOT_DIR, f"noise_diagnostic_snr{diag_snr:.0f}dB.png")
    fig_noise.savefig(noise_path, dpi=150)
    print(f"Saved: {noise_path}")

    plt.tight_layout()
    ber_path = os.path.join(PLOT_DIR, "ber_vs_snr.png")
    plt.savefig(ber_path, dpi=150)
    print(f"\nSaved: {ber_path}")

    # Plot constellation/eye for each modulation at mid-range SNR
    for mod_scheme, (result, snr_db) in plot_data.items():
        fig = plot_constellation_and_eye(
            result["rx_filtered"],
            result["sync_result"],
            mod_scheme,
            snr_db,
            f"{mod_scheme.name}"
        )
        if fig:
            filename = f"pipeline_{mod_scheme.name}_snr{int(snr_db):+d}dB.png"
            filepath = os.path.join(PLOT_DIR, filename)
            fig.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")

    plt.show()


if __name__ == "__main__":
    main()
