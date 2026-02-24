"""BER vs Eb/N0 simulation comparing LDPC-coded vs uncoded QPSK.

Demonstrates the "waterfall" effect of LDPC coding where BER drops
rapidly after a certain Eb/N0 threshold.

Uses Eb/N0 (energy per information bit over noise power spectral density)
for accurate performance comparison. This properly accounts for the coding
rate, making it possible to fairly compare codes with different rates.

Also compares standard min-sum vs normalized min-sum decoding.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc, erfcinv
from tqdm import tqdm

from modules.channel import ChannelConfig, ChannelModel
from modules.channel_coding import CodeRates, LDPCConfig, ldpc_decode, ldpc_encode
from modules.modulation import QPSK
from modules.util import ebn0_to_snr


def simulate_uncoded_qpsk(
    ebn0_db_range: np.ndarray,
    n_bits: int,
    n_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate uncoded QPSK with hard decision decoding.

    For uncoded QPSK, Eb/N0 = Es/N0 / bits_per_symbol = Es/N0 / 2.
    """
    qpsk = QPSK()
    ber = np.zeros(len(ebn0_db_range))

    for i, ebn0_db in enumerate(tqdm(ebn0_db_range, desc="Uncoded QPSK    ")):
        total_errors = 0
        total_bits = 0

        # For uncoded: code_rate = 1.0 (no coding)
        snr_db = ebn0_to_snr(ebn0_db, code_rate=1.0, bits_per_symbol=2)

        for trial in range(n_trials):
            # Generate random bits (must be even for QPSK)
            bits = rng.integers(0, 2, size=n_bits)

            # Modulate
            symbols = qpsk.bits2symbols(bits)

            # Channel
            channel = ChannelModel(
                ChannelConfig(
                    snr_db=snr_db,
                    seed=trial * 1000 + i,
                ),
            )
            rx_symbols = channel.apply(symbols)

            # Hard decision demodulation
            rx_bits = qpsk.symbols2bits(rx_symbols).flatten()

            # Count errors
            total_errors += np.sum(bits != rx_bits)
            total_bits += n_bits

        ber[i] = total_errors / total_bits if total_bits > 0 else 0

    return ber


def simulate_ldpc_coded_qpsk(
    ebn0_db_range: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
    alpha: float = 0.75,
    desc: str = "LDPC + QPSK",
) -> np.ndarray:
    """Simulate LDPC-coded QPSK with soft decision decoding."""
    code_rate = CodeRates.HALF_RATE
    rate_float = code_rate.value_float
    k = int(648 * rate_float)  # message length for n=648
    config = LDPCConfig(k=k, code_rate=code_rate)
    qpsk = QPSK()

    ber = np.zeros(len(ebn0_db_range))

    for i, ebn0_db in enumerate(tqdm(ebn0_db_range, desc=f"{desc:16}")):
        total_errors = 0
        total_bits = 0

        # Convert Eb/N0 to SNR per symbol (Es/N0)
        snr_db = ebn0_to_snr(ebn0_db, rate_float, bits_per_symbol=2)

        for trial in range(n_trials):
            # Generate random message bits
            message = rng.integers(0, 2, size=config.k)

            # LDPC encode
            codeword = ldpc_encode(message, config)

            # QPSK modulate
            symbols = qpsk.bits2symbols(codeword)

            # Channel
            channel = ChannelModel(
                ChannelConfig(
                    snr_db=snr_db,
                    seed=trial * 1000 + i + 500000,
                ),
            )
            rx_symbols = channel.apply(symbols)

            # Soft decision demodulation (LLRs)
            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)

            # LDPC decode
            decoded = ldpc_decode(llrs.flatten(), config, max_iterations=50, alpha=alpha)

            # Count errors (on message bits, not codeword)
            total_errors += np.sum(message != decoded)
            total_bits += config.k

        ber[i] = total_errors / total_bits if total_bits > 0 else 0

    return ber


def theoretical_qpsk_ber(ebn0_db: np.ndarray) -> np.ndarray:
    """Theoretical BER for uncoded QPSK over AWGN channel.

    The theoretical BER for QPSK is BER = 0.5 * erfc(sqrt(Eb/N0)),
    where Eb/N0 is the energy per bit over noise power spectral density.

    Note: For QPSK, Eb/N0 = Es/N0 / 2, where Es/N0 is the SNR per symbol.
    """
    ebn0_linear = 10 ** (ebn0_db / 10)
    return 0.5 * erfc(np.sqrt(ebn0_linear))


def measure_coding_gain_at_target_ber(
    rng: np.random.Generator,
    target_ber: float = 1e-6,
    n_trials: int = 1000,
) -> None:
    """Measure coding gain at a specific target BER using high trial count.

    Uses theoretical QPSK for uncoded (exact) and simulation for LDPC.
    All measurements are in Eb/N0 for proper comparison.
    """
    # Theoretical Eb/N0 for uncoded QPSK at target BER
    # BER = 0.5 * erfc(sqrt(Eb/N0)) => Eb/N0 = erfcinv(2*BER)^2
    ebn0_uncoded_theory = erfcinv(2 * target_ber) ** 2
    ebn0_uncoded_db = 10 * np.log10(ebn0_uncoded_theory)

    # For LDPC, simulate around expected waterfall region
    config = LDPCConfig(k=324, code_rate=CodeRates.HALF_RATE)
    qpsk = QPSK()
    code_rate = CodeRates.HALF_RATE.value_float

    # Fine Eb/N0 sweep in waterfall region
    ebn0_test = np.arange(1.5, 3.5, 0.1)

    results = []

    for ebn0_db in tqdm(ebn0_test, desc="LDPC measurement"):
        total_errors = 0
        total_bits = 0

        # Convert Eb/N0 to SNR per symbol
        snr_db = ebn0_to_snr(float(ebn0_db), code_rate, bits_per_symbol=2)

        for trial in range(n_trials):
            message = rng.integers(0, 2, size=config.k)
            codeword = ldpc_encode(message, config)
            symbols = qpsk.bits2symbols(codeword)

            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=trial))
            rx_symbols = channel.apply(symbols)

            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc_decode(llrs.flatten(), config, max_iterations=50, alpha=0.75)

            total_errors += np.sum(message != decoded)
            total_bits += config.k

        ber = total_errors / total_bits if total_bits > 0 else 0
        results.append((ebn0_db, ber))

    # Find Eb/N0 at target BER by interpolation
    ebn0_arr = np.array([r[0] for r in results])
    ber_arr = np.array([r[1] for r in results])
    ebn0_ldpc_db = _find_ebn0_at_ber(ebn0_arr, ber_arr, target_ber)

    if not np.isnan(ebn0_ldpc_db):
        _coding_gain = ebn0_uncoded_db - ebn0_ldpc_db


def _find_ebn0_at_ber(
    ebn0_range: np.ndarray,
    ber: np.ndarray,
    target_ber: float,
) -> float:
    """Interpolate to find Eb/N0 at target BER."""
    valid = ber > 0
    if not np.any(valid):
        return np.nan
    log_ber = np.log10(ber[valid])
    ebn0_valid = ebn0_range[valid]
    if np.min(log_ber) > np.log10(target_ber):
        return np.nan
    idx = np.searchsorted(-log_ber, -np.log10(target_ber))
    if idx == 0 or idx >= len(ebn0_valid):
        return ebn0_valid[min(idx, len(ebn0_valid) - 1)]
    x1, x2 = ebn0_valid[idx - 1], ebn0_valid[idx]
    y1, y2 = log_ber[idx - 1], log_ber[idx]
    return x1 + (x2 - x1) * (np.log10(target_ber) - y1) / (y2 - y1)


def _run_simulations(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run all BER simulations and return results."""
    ebn0_db_range = np.arange(-2, 12, 1.0)
    n_trials = 100
    n_bits_uncoded = 648

    ber_uncoded = simulate_uncoded_qpsk(ebn0_db_range, n_bits_uncoded, n_trials, rng)
    ber_std_minsum = simulate_ldpc_coded_qpsk(ebn0_db_range, n_trials, rng, alpha=1.0, desc="LDPC (alpha=1.0)")
    ber_norm_minsum = simulate_ldpc_coded_qpsk(ebn0_db_range, n_trials, rng, alpha=0.75, desc="LDPC (alpha=0.75)")
    ber_theory = theoretical_qpsk_ber(ebn0_db_range)

    measure_coding_gain_at_target_ber(rng, target_ber=1e-6, n_trials=1000)

    return ebn0_db_range, ber_uncoded, ber_std_minsum, ber_norm_minsum, ber_theory


def _plot_results(
    ebn0_db_range: np.ndarray,
    ber_uncoded: np.ndarray,
    ber_std_minsum: np.ndarray,
    ber_norm_minsum: np.ndarray,
    ber_theory: np.ndarray,
) -> None:
    """Plot BER vs Eb/N0 curves with annotations."""
    target_ber = 1e-3
    ebn0_uncoded_1e3 = _find_ebn0_at_ber(ebn0_db_range, ber_uncoded, target_ber)
    ebn0_std_1e3 = _find_ebn0_at_ber(ebn0_db_range, ber_std_minsum, target_ber)
    ebn0_norm_1e3 = _find_ebn0_at_ber(ebn0_db_range, ber_norm_minsum, target_ber)

    plt.figure(figsize=(10, 7))
    plt.semilogy(ebn0_db_range, ber_uncoded, "bo-", label="Uncoded QPSK (hard decision)", markersize=4, linewidth=1.5)
    plt.semilogy(
        ebn0_db_range,
        ber_std_minsum,
        "g^-",
        label="LDPC R=1/2 + standard min-sum (alpha=1.0)",
        markersize=4,
        linewidth=1.5,
    )
    plt.semilogy(
        ebn0_db_range,
        ber_norm_minsum,
        "rs-",
        label="LDPC R=1/2 + normalized min-sum (alpha=0.75)",
        markersize=4,
        linewidth=1.5,
    )
    plt.semilogy(ebn0_db_range, ber_theory, "k--", label="Theoretical QPSK", linewidth=1.5, alpha=0.7)
    plt.axhline(y=target_ber, color="gray", linestyle=":", alpha=0.5)

    if not np.isnan(ebn0_uncoded_1e3) and not np.isnan(ebn0_norm_1e3):
        coding_gain_norm = ebn0_uncoded_1e3 - ebn0_norm_1e3
        plt.annotate(
            f"Coding gain (normalized): {coding_gain_norm:.1f} dB",
            xy=(ebn0_norm_1e3, target_ber),
            xytext=(ebn0_norm_1e3 + 1, 3e-3),
            fontsize=9,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8},
        )

    min_gain_threshold = 0.1
    if not np.isnan(ebn0_std_1e3) and not np.isnan(ebn0_norm_1e3):
        norm_gain = ebn0_std_1e3 - ebn0_norm_1e3
        if norm_gain > min_gain_threshold:
            plt.annotate(
                f"Normalization gain: ~{norm_gain:.1f} dB",
                xy=((ebn0_std_1e3 + ebn0_norm_1e3) / 2, target_ber),
                xytext=((ebn0_std_1e3 + ebn0_norm_1e3) / 2, 1e-4),
                fontsize=9,
                ha="center",
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
            )

    plt.xlabel("Eb/N0 (dB)", fontsize=12)
    plt.ylabel("Bit Error Rate (BER)", fontsize=12)
    plt.title(
        "BER vs Eb/N0: Standard vs Normalized Min-Sum LDPC Decoding\n(IEEE 802.11 LDPC, Rate 1/2, n=648)",
        fontsize=13,
    )
    plt.grid(visible=True, which="both", alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    plt.xlim([ebn0_db_range[0], ebn0_db_range[-1]])
    plt.ylim([1e-5, 1])
    plt.annotate(
        "Waterfall\nregion",
        xy=(1.5, 2e-2),
        fontsize=9,
        ha="center",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    plt.tight_layout()
    plt.savefig("examples/data/ber_vs_ebn0.png", dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    """Run BER vs Eb/N0 simulation and plot results."""
    rng = np.random.default_rng(42)
    ebn0_db_range, ber_uncoded, ber_std_minsum, ber_norm_minsum, ber_theory = _run_simulations(rng)
    _plot_results(ebn0_db_range, ber_uncoded, ber_std_minsum, ber_norm_minsum, ber_theory)


if __name__ == "__main__":
    main()
