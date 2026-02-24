"""Full pipeline test for PlutoSDR loopback.

End-to-end test with framing, channel coding, and data recovery.
Measures BER vs SNR for different modulation schemes.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import adi
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameConstructor, FrameHeader, ModulationSchemes
from modules.modulation import BPSK
from modules.pilots import PilotConfig, data_indices, insert_pilots
from modules.pulse_shaping import rrc_filter, upsample_and_filter
from modules.synchronization import (
    SynchronizationResult,
    Synchronizer,
    SynchronizerConfig,
    build_preamble,
)
from pluto.config import CENTER_FREQ, SAMPLE_RATE, SPS, PipelineConfig, get_modulator
from pluto.decode import FrameDecoder
from pluto.loopback import setup_pluto, transmit_and_receive

PLOT_DIR = "examples/data"
RRC_ALPHA = 0.35
RRC_NUM_TAPS = 101
GUARD_SAMPLES = 500
PLOT_SNR_THRESHOLD = 4


@dataclass
class FrameBuildParams:
    """Parameters needed to build a test frame."""

    frame_constructor: FrameConstructor
    sync_config: SynchronizerConfig
    pilot_config: PilotConfig
    pipeline: PipelineConfig


@dataclass
class TestCase:
    """Parameters for a single BER test run."""

    mod_scheme: ModulationSchemes
    snr_db: float
    n_payload_bits: int


@dataclass
class RadioContext:
    """Radio-related objects needed for BER testing."""

    sdr: adi.Pluto
    h_rrc: np.ndarray
    sync: Synchronizer
    frame_constructor: FrameConstructor
    pilot_config: PilotConfig
    pipeline: PipelineConfig


def add_awgn(
    signal: np.ndarray,
    snr_db: float,
    *,
    return_noise: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Add AWGN noise to achieve target SNR (in dB).

    Measures signal power from the strongest portion (where the frame is),
    not the entire buffer which includes quiet regions.
    """
    rng = np.random.default_rng()

    # Find the active region by looking at where power is significant
    window = 1000
    if len(signal) > window:
        # Sliding window to find max power region
        power_profile = np.convolve(np.abs(signal) ** 2, np.ones(window) / window, mode="valid")
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
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal)))
    noisy_signal = signal + noise

    if return_noise:
        return noisy_signal, noise, cast("tuple[int, int]", active_region)
    return noisy_signal


def build_test_frame(
    payload_bits: np.ndarray,
    params: FrameBuildParams,
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

    header_encoded, payload_encoded = params.frame_constructor.encode(
        header,
        payload_bits,
        channel_coding=params.pipeline.channel_coding,
        interleaving=params.pipeline.interleaving,
    )

    bpsk = BPSK()
    header_symbols = bpsk.bits2symbols(header_encoded)
    payload_symbols = get_modulator(mod_scheme).bits2symbols(payload_encoded)

    if params.pipeline.pilots:
        payload_symbols = insert_pilots(payload_symbols, params.pilot_config)

    preamble = build_preamble(params.sync_config)
    frame = np.concatenate([preamble, header_symbols, payload_symbols])

    return frame, header


def run_ber_test(
    test_case: TestCase,
    radio: RadioContext,
    *,
    timing: bool = False,
) -> dict:
    """Run a single BER test at given SNR."""
    rng = np.random.default_rng()
    timings = {}
    mod_scheme = test_case.mod_scheme
    snr_db = test_case.snr_db
    n_payload_bits = test_case.n_payload_bits

    t0 = time.perf_counter()
    tx_bits = rng.integers(0, 2, n_payload_bits)

    sync_config = radio.sync.config
    build_params = FrameBuildParams(
        frame_constructor=radio.frame_constructor,
        sync_config=sync_config,
        pilot_config=radio.pilot_config,
        pipeline=radio.pipeline,
    )
    frame_symbols, _header = build_test_frame(
        tx_bits, build_params, mod_scheme, CodeRates.HALF_RATE,
    )

    tx_signal = upsample_and_filter(frame_symbols, SPS, radio.h_rrc)
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    tx_signal = np.concatenate([zeros, tx_signal, zeros])
    timings["tx_build"] = time.perf_counter() - t0

    # Transmit and receive through Pluto
    t0 = time.perf_counter()
    rx_raw = transmit_and_receive(radio.sdr, tx_signal, rx_delay_ms=50, n_captures=5)
    timings["pluto_txrx"] = time.perf_counter() - t0

    # Add AWGN noise
    t0 = time.perf_counter()
    rx_noisy_result = add_awgn(rx_raw, snr_db)
    rx_noisy = cast("np.ndarray", rx_noisy_result)
    timings["add_noise"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rx_filtered = np.convolve(rx_noisy, radio.h_rrc, mode="same")
    timings["matched_filter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    radio.pipeline.pilot_config = radio.pilot_config
    decoder = FrameDecoder(radio.sync, radio.frame_constructor, SAMPLE_RATE, radio.pipeline)
    result = decoder.try_decode(rx_filtered, global_sample_offset=0)
    timings["decode_total"] = time.perf_counter() - t0

    if timing:
        pass

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


def plot_signal_vs_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    active_region: tuple[int, int],
    snr_db: float,
) -> "Figure":
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
    ax.plot(t, np.real(clean), "b-", linewidth=0.5, alpha=0.7, label="I")
    ax.plot(t, np.imag(clean), "r-", linewidth=0.5, alpha=0.7, label="Q")
    ax.axvline(start, color="g", linestyle="--", alpha=0.5, label="Active region")
    ax.axvline(end, color="g", linestyle="--", alpha=0.5)
    ax.set_title("Clean Signal (before noise)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    # Noise only
    ax = axes[1]
    ax.plot(t, np.real(noise[plot_start:plot_end]), "b-", linewidth=0.5, alpha=0.7, label="I")
    ax.plot(t, np.imag(noise[plot_start:plot_end]), "r-", linewidth=0.5, alpha=0.7, label="Q")
    ax.axvline(start, color="g", linestyle="--", alpha=0.5)
    ax.axvline(end, color="g", linestyle="--", alpha=0.5)
    noise_std = np.std(noise[start:end])
    ax.set_title(f"Noise (sigma={noise_std:.4f})")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    # Noisy signal
    ax = axes[2]
    ax.plot(t, np.real(signal[plot_start:plot_end]), "b-", linewidth=0.5, alpha=0.7, label="I")
    ax.plot(t, np.imag(signal[plot_start:plot_end]), "r-", linewidth=0.5, alpha=0.7, label="Q")
    ax.axvline(start, color="g", linestyle="--", alpha=0.5)
    ax.axvline(end, color="g", linestyle="--", alpha=0.5)
    ax.set_title(f"Noisy Signal (SNR={snr_db:.1f} dB)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    plt.suptitle(f"Signal vs Noise Comparison (SNR={snr_db:.1f} dB)", fontsize=14)
    plt.tight_layout()
    return fig


def _extract_payload_symbols(
    rx_filtered: np.ndarray,
    sync_result: SynchronizationResult,
    pilot_config: PilotConfig,
) -> np.ndarray:
    """Extract and preprocess payload symbols from the received signal."""
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
        return payload_symbols_all[d_idx]
    return payload_symbols_all


def _prepare_rx_payload(
    rx_filtered: np.ndarray,
    sync_result: SynchronizationResult,
) -> np.ndarray:
    """Prepare the raw (non-downsampled) RX payload for eye diagrams."""
    cfo_hz = sync_result.cfo_hat_hz
    n_vec = np.arange(len(rx_filtered))
    rx_corrected = rx_filtered * np.exp(-1j * 2 * np.pi * cfo_hz / SAMPLE_RATE * n_vec)
    data_start = sync_result.long_zc_start + 139 * SPS
    rx_data = rx_corrected[data_start:]
    n_header_symbols = 72
    header_symbols = rx_data[::SPS][:n_header_symbols]
    phase_est = np.angle(np.sum(header_symbols * np.sign(np.real(header_symbols))))
    rx_data = rx_data * np.exp(-1j * phase_est)
    power = np.mean(np.abs(rx_data[:1000]) ** 2)
    if power > 0:
        rx_data = rx_data / np.sqrt(power)
    payload_start = n_header_symbols * SPS
    return rx_data[payload_start:]


def _plot_eye_diagrams(
    axes: np.ndarray,
    rx_payload: np.ndarray,
    n_traces: int,
) -> None:
    """Plot I and Q eye diagrams on the first two axes."""
    eye_len = 2 * SPS
    t_eye = np.linspace(-1, 1, eye_len)

    # Eye diagram - I
    ax = axes[0]
    for i in range(n_traces):
        start = i * SPS
        if start + eye_len <= len(rx_payload):
            ax.plot(t_eye, np.real(rx_payload[start : start + eye_len]), "b-", alpha=0.05, linewidth=0.5)
    ax.axvline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_title("Eye Diagram (I)")
    ax.set_xlabel("Symbol Period")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-1, 1)
    ax.grid(visible=True, alpha=0.3)

    # Eye diagram - Q
    ax = axes[1]
    for i in range(n_traces):
        start = i * SPS
        if start + eye_len <= len(rx_payload):
            ax.plot(t_eye, np.imag(rx_payload[start : start + eye_len]), "b-", alpha=0.05, linewidth=0.5)
    ax.axvline(0, color="r", linestyle="--", alpha=0.5)
    ax.set_title("Eye Diagram (Q)")
    ax.set_xlabel("Symbol Period")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(-1, 1)
    ax.grid(visible=True, alpha=0.3)


def plot_constellation_and_eye(
    rx_filtered: np.ndarray,
    sync_result: SynchronizationResult,
    mod_scheme: ModulationSchemes,
    snr_db: float,
    title: str,
) -> "Figure | None":
    """Plot constellation and eye diagram for a single test."""
    pilot_config = PilotConfig()

    if not sync_result.success:
        return None

    payload_data_only = _extract_payload_symbols(rx_filtered, sync_result, pilot_config)
    rx_payload = _prepare_rx_payload(rx_filtered, sync_result)
    n_traces = min(200, len(rx_payload) // SPS - 2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    _plot_eye_diagrams(axes[:2], rx_payload, n_traces)

    # Constellation
    ax = axes[2]
    symbols = payload_data_only[: min(n_traces, len(payload_data_only))]
    modulator = get_modulator(mod_scheme)
    constellation = modulator.symbol_mapping / np.sqrt(np.mean(np.abs(modulator.symbol_mapping) ** 2))
    ax.scatter(np.real(symbols), np.imag(symbols), alpha=0.3, s=5, label="RX")
    ax.scatter(np.real(constellation), np.imag(constellation), c="red", s=100, marker="x", label="Ideal")
    ax.set_title(f"Constellation - {mod_scheme.name}")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(visible=True, alpha=0.3)
    ax.legend()

    plt.suptitle(f"{title} (SNR={snr_db} dB)", fontsize=12)
    plt.tight_layout()
    return fig


def _collect_ber_results(
    mod_schemes: list[ModulationSchemes],
    snr_values: list[float],
    n_payload_bits: int,
    n_trials: int,
    radio: RadioContext,
) -> tuple[dict[ModulationSchemes, list[float]], dict]:
    """Collect BER results for all modulation schemes and SNR values."""
    ber_results: dict[ModulationSchemes, list[float]] = {mod: [] for mod in mod_schemes}
    plot_data: dict = {}

    for mod_scheme in mod_schemes:
        for snr_db in snr_values:
            bers: list[float] = []
            last_result = None

            for trial in range(n_trials):
                first_snr = snr_db == snr_values[0]
                tc = TestCase(mod_scheme, snr_db, n_payload_bits)
                result = run_ber_test(
                    tc,
                    radio,
                    timing=(trial == 0 and first_snr),
                )
                if result["success"]:
                    bers.append(result["ber"])
                    last_result = result
                else:
                    pass

            avg_ber = np.mean(bers) if bers else 0.5
            ber_results[mod_scheme].append(float(avg_ber))

            # Store result for mid-range SNR for plotting
            if snr_db == PLOT_SNR_THRESHOLD and last_result and last_result["success"]:
                plot_data[mod_scheme] = (last_result, snr_db)

    return ber_results, plot_data


def _plot_ber_curve(
    snr_values: list[float],
    ber_results: dict[ModulationSchemes, list[float]],
    mod_schemes: list[ModulationSchemes],
    cfo_hz: int,
) -> None:
    """Plot and save the BER vs SNR curve."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    markers = {"BPSK": "o", "QPSK": "s", "QAM16": "^"}
    colors = {"BPSK": "blue", "QPSK": "green", "QAM16": "red"}

    for mod_scheme in mod_schemes:
        bers = ber_results[mod_scheme]
        bers_plot = [max(b, 1e-5) for b in bers]
        ax.semilogy(
            snr_values,
            bers_plot,
            marker=markers[mod_scheme.name],
            color=colors[mod_scheme.name],
            label=mod_scheme.name,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title(f"BER vs SNR - Full Pipeline with LDPC (CFO={cfo_hz:+d} Hz)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(visible=True, which="both", alpha=0.3)
    ax.set_ylim(1e-5, 1)

    plt.tight_layout()
    ber_path = Path(PLOT_DIR) / "ber_vs_snr.png"
    plt.savefig(ber_path, dpi=150)


def main() -> None:
    """Run full pipeline BER vs SNR test across modulation schemes over PlutoSDR loopback."""
    rng = np.random.default_rng()

    # Setup
    pipeline = PipelineConfig()
    sync_config = SynchronizerConfig()
    pilot_config = PilotConfig()
    frame_constructor = FrameConstructor()
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(sync_config, sps=SPS, rrc_taps=h_rrc)

    # Random CFO
    max_cfo = 3000
    cfo_hz = int(rng.integers(-max_cfo, max_cfo + 1))

    sdr = setup_pluto()
    if cfo_hz != 0:
        sdr.rx_lo = int(CENTER_FREQ - cfo_hz)

    radio = RadioContext(
        sdr=sdr, h_rrc=h_rrc, sync=sync, frame_constructor=frame_constructor, pilot_config=pilot_config, pipeline=pipeline,
    )

    # Test parameters - fine granularity, 0.5 dB steps
    snr_values = [float(x) for x in np.arange(-6, 4.1, 0.5)]
    mod_schemes = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.QAM16]
    n_payload_bits = 200
    n_trials = 3

    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

    ber_results, plot_data = _collect_ber_results(mod_schemes, snr_values, n_payload_bits, n_trials, radio)
    _plot_ber_curve(snr_values, ber_results, mod_schemes, cfo_hz)

    # Generate noise diagnostic plot at low SNR
    diag_snr = -4.0
    tx_bits = rng.integers(0, 2, n_payload_bits)
    build_params = FrameBuildParams(
        frame_constructor=frame_constructor, sync_config=sync_config, pilot_config=pilot_config, pipeline=pipeline,
    )
    frame_symbols, _ = build_test_frame(
        tx_bits, build_params, ModulationSchemes.QPSK, CodeRates.HALF_RATE,
    )
    tx_signal = upsample_and_filter(frame_symbols, SPS, h_rrc)
    zeros = np.zeros(GUARD_SAMPLES, dtype=complex)
    tx_signal = np.concatenate([zeros, tx_signal, zeros])
    rx_raw = transmit_and_receive(sdr, tx_signal, rx_delay_ms=50, n_captures=5)
    rx_noisy, noise, active_region = add_awgn(rx_raw, diag_snr, return_noise=True)

    fig_noise = plot_signal_vs_noise(rx_noisy, noise, active_region, diag_snr)
    noise_path = Path(PLOT_DIR) / f"noise_diagnostic_snr{diag_snr:.0f}dB.png"
    fig_noise.savefig(noise_path, dpi=150)

    # Plot constellation/eye for each modulation at mid-range SNR
    for mod_scheme, (result, snr_db) in plot_data.items():
        fig = plot_constellation_and_eye(
            result["rx_filtered"], result["sync_result"], mod_scheme, snr_db, f"{mod_scheme.name}",
        )
        if fig:
            filename = f"pipeline_{mod_scheme.name}_snr{int(snr_db):+d}dB.png"
            filepath = Path(PLOT_DIR) / filename
            fig.savefig(filepath, dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
