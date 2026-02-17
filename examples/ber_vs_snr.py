#!/usr/bin/env python3
"""BER vs Eb/N0 simulation comparing LDPC-coded vs uncoded QPSK.

Demonstrates the "waterfall" effect of LDPC coding where BER drops
rapidly after a certain Eb/N0 threshold.

Uses Eb/N0 (energy per information bit over noise power spectral density)
for accurate performance comparison. This properly accounts for the coding
rate, making it possible to fairly compare codes with different rates.

Also compares standard min-sum vs normalized min-sum decoding.

Usage:
    uv run python examples/ber_vs_snr.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.channel_coding import LDPC, LDPCConfig, CodeRates
from modules.modulation import QPSK
from modules.channel import ChannelModel, ChannelConfig
from modules.util import ebn0_to_snr


def simulate_uncoded_qpsk(ebn0_db_range: np.ndarray, n_bits: int, n_trials: int) -> np.ndarray:
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
            bits = np.random.randint(0, 2, n_bits)

            # Modulate
            symbols = qpsk.bits2symbols(bits)

            # Channel
            channel = ChannelModel(ChannelConfig(
                snr_db=snr_db,
                seed=trial * 1000 + i,
            ))
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
    alpha: float = 0.75,
    desc: str = "LDPC + QPSK",
    code_rate: CodeRates = CodeRates.HALF_RATE,
) -> np.ndarray:
    """Simulate LDPC-coded QPSK with soft decision decoding.

    Args:
        ebn0_db_range: Array of Eb/N0 values to simulate (dB)
        n_trials: Number of frames per Eb/N0 point
        alpha: Min-sum normalization factor (0.75=normalized, 1.0=standard)
        desc: Description for progress bar
        code_rate: LDPC code rate to use

    The simulation uses Eb/N0 which properly accounts for the coding rate,
    allowing fair comparison between different code rates.
    """
    # LDPC config based on code rate
    rate_float = code_rate.value_float
    k = int(648 * rate_float)  # message length for n=648
    config = LDPCConfig(k=k, code_rate=code_rate)
    ldpc = LDPC()
    qpsk = QPSK()

    ber = np.zeros(len(ebn0_db_range))

    for i, ebn0_db in enumerate(tqdm(ebn0_db_range, desc=f"{desc:16}")):
        total_errors = 0
        total_bits = 0

        # Convert Eb/N0 to SNR per symbol (Es/N0)
        snr_db = ebn0_to_snr(ebn0_db, rate_float, bits_per_symbol=2)

        for trial in range(n_trials):
            # Generate random message bits
            message = np.random.randint(0, 2, config.k)

            # LDPC encode
            codeword = ldpc.encode(message, config)

            # QPSK modulate
            symbols = qpsk.bits2symbols(codeword)

            # Channel
            channel = ChannelModel(ChannelConfig(
                snr_db=snr_db,
                seed=trial * 1000 + i + 500000,
            ))
            rx_symbols = channel.apply(symbols)

            # Soft decision demodulation (LLRs)
            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)

            # LDPC decode
            decoded = ldpc.decode(llrs.flatten(), config, max_iterations=50, alpha=alpha)

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
    from scipy.special import erfc
    ebn0_linear = 10 ** (ebn0_db / 10)
    return 0.5 * erfc(np.sqrt(ebn0_linear))


def measure_coding_gain_at_target_ber(target_ber: float = 1e-6, n_trials: int = 1000):
    """Measure coding gain at a specific target BER using high trial count.

    Uses theoretical QPSK for uncoded (exact) and simulation for LDPC.
    All measurements are in Eb/N0 for proper comparison.
    """
    from scipy.special import erfcinv

    print(f"\n{'='*60}")
    print(f"MEASURING CODING GAIN AT BER = {target_ber:.0e}")
    print(f"{'='*60}")

    # Theoretical Eb/N0 for uncoded QPSK at target BER
    # BER = 0.5 * erfc(sqrt(Eb/N0)) => Eb/N0 = erfcinv(2*BER)^2
    ebn0_uncoded_theory = erfcinv(2 * target_ber) ** 2
    ebn0_uncoded_db = 10 * np.log10(ebn0_uncoded_theory)
    print(f"\nUncoded QPSK @ BER={target_ber:.0e}: Eb/N0 = {ebn0_uncoded_db:.2f} dB (theoretical)")

    # For LDPC, simulate around expected waterfall region
    config = LDPCConfig(k=324, code_rate=CodeRates.HALF_RATE)
    ldpc = LDPC()
    qpsk = QPSK()
    code_rate = CodeRates.HALF_RATE.value_float

    # Fine Eb/N0 sweep in waterfall region
    ebn0_test = np.arange(1.5, 3.5, 0.1)

    print(f"\nSimulating LDPC rate 1/2 (α=0.75) with {n_trials} trials per Eb/N0 point...")
    print(f"Total bits per Eb/N0 point: {n_trials * config.k:,}")
    print()

    results = []

    for ebn0_db in tqdm(ebn0_test, desc="LDPC measurement"):
        total_errors = 0
        total_bits = 0

        # Convert Eb/N0 to SNR per symbol
        snr_db = ebn0_to_snr(ebn0_db, code_rate, bits_per_symbol=2)

        for trial in range(n_trials):
            message = np.random.randint(0, 2, config.k)
            codeword = ldpc.encode(message, config)
            symbols = qpsk.bits2symbols(codeword)

            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=trial))
            rx_symbols = channel.apply(symbols)

            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc.decode(llrs.flatten(), config, max_iterations=50, alpha=0.75)

            total_errors += np.sum(message != decoded)
            total_bits += config.k

        ber = total_errors / total_bits if total_bits > 0 else 0
        results.append((ebn0_db, ber, total_errors, total_bits))

        if ber > 0:
            print(f"  Eb/N0={ebn0_db:.1f} dB: BER={ber:.2e} ({total_errors} errors / {total_bits:,} bits)")
        else:
            print(f"  Eb/N0={ebn0_db:.1f} dB: BER=0 (no errors in {total_bits:,} bits)")

    # Find Eb/N0 at target BER by interpolation
    ebn0_ldpc_db = None
    for i in range(len(results) - 1):
        ebn0_1, ber1, _, _ = results[i]
        ebn0_2, ber2, _, _ = results[i + 1]

        if ber1 >= target_ber >= ber2 and ber1 > 0 and ber2 > 0:
            # Log-linear interpolation
            log_ber1, log_ber2 = np.log10(ber1), np.log10(ber2)
            log_target = np.log10(target_ber)
            ebn0_ldpc_db = ebn0_1 + (ebn0_2 - ebn0_1) * (log_target - log_ber1) / (log_ber2 - log_ber1)
            break
        elif ber2 == 0 and ber1 > 0 and ber1 <= target_ber:
            # Target is between last non-zero and zero
            ebn0_ldpc_db = ebn0_1
            print(f"\n  Note: BER dropped to 0 after Eb/N0={ebn0_1:.1f} dB, using as estimate")
            break

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Target BER:                  {target_ber:.0e}")
    print(f"Uncoded QPSK Eb/N0:          {ebn0_uncoded_db:.2f} dB (theoretical)")

    if ebn0_ldpc_db is not None:
        print(f"LDPC (α=0.75) Eb/N0:         {ebn0_ldpc_db:.2f} dB (simulated)")
        coding_gain = ebn0_uncoded_db - ebn0_ldpc_db
        print(f"Coding gain:                 {coding_gain:.2f} dB")
    else:
        # Find lowest BER achieved
        min_ber = min(r[1] for r in results if r[1] > 0) if any(r[1] > 0 for r in results) else 0
        if min_ber > 0:
            print(f"LDPC (α=0.75) Eb/N0:         Could not reach {target_ber:.0e}")
            print(f"  Lowest BER measured:       {min_ber:.2e}")
            print(f"  Need more trials or lower target BER")
        else:
            last_nonzero_ebn0 = max(r[0] for r in results if r[1] > 0) if any(r[1] > 0 for r in results) else results[0][0]
            print(f"LDPC (α=0.75) Eb/N0:         <{last_nonzero_ebn0:.1f} dB (all frames decoded)")
            coding_gain = ebn0_uncoded_db - last_nonzero_ebn0
            print(f"Coding gain:                 >{coding_gain:.1f} dB")

    print(f"{'='*60}\n")


def main():
    # Simulation parameters
    # Using Eb/N0 for accurate channel coding performance comparison
    ebn0_db_range = np.arange(-2, 12, 1.0)  # -2 to 12 dB, 1 dB steps

    n_trials = 100  # Number of frames per Eb/N0 point (more = smoother curves)
    n_bits_uncoded = 648  # Same as LDPC codeword length for fair comparison

    print("Running BER vs Eb/N0 simulation...")
    print(f"Eb/N0 range: {ebn0_db_range[0]} to {ebn0_db_range[-1]} dB")
    print(f"Trials per Eb/N0: {n_trials}")
    print()

    np.random.seed(42)

    # Run simulations
    ber_uncoded = simulate_uncoded_qpsk(ebn0_db_range, n_bits_uncoded, n_trials)

    # Standard min-sum (alpha=1.0)
    ber_std_minsum = simulate_ldpc_coded_qpsk(
        ebn0_db_range, n_trials, alpha=1.0, desc="LDPC (α=1.0)"
    )

    # Normalized min-sum (alpha=0.75)
    ber_norm_minsum = simulate_ldpc_coded_qpsk(
        ebn0_db_range, n_trials, alpha=0.75, desc="LDPC (α=0.75)"
    )

    ber_theory = theoretical_qpsk_ber(ebn0_db_range)

    # Measure coding gain at 10^-6
    measure_coding_gain_at_target_ber(target_ber=1e-6, n_trials=1000)

    # Calculate coding gain at BER = 1e-3
    def find_ebn0_at_ber(ebn0_range, ber, target_ber):
        """Interpolate to find Eb/N0 at target BER."""
        valid = ber > 0
        if not np.any(valid):
            return np.nan
        log_ber = np.log10(ber[valid])
        ebn0_valid = ebn0_range[valid]
        if np.min(log_ber) > np.log10(target_ber):
            return np.nan  # Never reaches target BER
        idx = np.searchsorted(-log_ber, -np.log10(target_ber))
        if idx == 0 or idx >= len(ebn0_valid):
            return ebn0_valid[min(idx, len(ebn0_valid)-1)]
        # Linear interpolation
        x1, x2 = ebn0_valid[idx-1], ebn0_valid[idx]
        y1, y2 = log_ber[idx-1], log_ber[idx]
        return x1 + (x2 - x1) * (np.log10(target_ber) - y1) / (y2 - y1)

    ebn0_uncoded_1e3 = find_ebn0_at_ber(ebn0_db_range, ber_uncoded, 1e-3)
    ebn0_std_1e3 = find_ebn0_at_ber(ebn0_db_range, ber_std_minsum, 1e-3)
    ebn0_norm_1e3 = find_ebn0_at_ber(ebn0_db_range, ber_norm_minsum, 1e-3)

    # Plot results
    plt.figure(figsize=(10, 7))

    # Plot simulated results
    plt.semilogy(ebn0_db_range, ber_uncoded, 'bo-', label='Uncoded QPSK (hard decision)',
                 markersize=4, linewidth=1.5)
    plt.semilogy(ebn0_db_range, ber_std_minsum, 'g^-',
                 label='LDPC R=1/2 + standard min-sum (α=1.0)',
                 markersize=4, linewidth=1.5)
    plt.semilogy(ebn0_db_range, ber_norm_minsum, 'rs-',
                 label='LDPC R=1/2 + normalized min-sum (α=0.75)',
                 markersize=4, linewidth=1.5)

    # Plot theoretical curve
    plt.semilogy(ebn0_db_range, ber_theory, 'k--', label='Theoretical QPSK',
                 linewidth=1.5, alpha=0.7)

    # Add coding gain annotations
    plt.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5)

    if not np.isnan(ebn0_uncoded_1e3) and not np.isnan(ebn0_norm_1e3):
        coding_gain_norm = ebn0_uncoded_1e3 - ebn0_norm_1e3
        plt.annotate(f'Coding gain (normalized): {coding_gain_norm:.1f} dB',
                    xy=(ebn0_norm_1e3, 1e-3),
                    xytext=(ebn0_norm_1e3 + 1, 3e-3),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    if not np.isnan(ebn0_std_1e3) and not np.isnan(ebn0_norm_1e3):
        norm_gain = ebn0_std_1e3 - ebn0_norm_1e3
        if norm_gain > 0.1:  # Only annotate if there's meaningful difference
            plt.annotate(f'Normalization gain: ~{norm_gain:.1f} dB',
                        xy=((ebn0_std_1e3 + ebn0_norm_1e3) / 2, 1e-3),
                        xytext=((ebn0_std_1e3 + ebn0_norm_1e3) / 2, 1e-4),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    plt.xlabel('Eb/N0 (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs Eb/N0: Standard vs Normalized Min-Sum LDPC Decoding\n'
              '(IEEE 802.11 LDPC, Rate 1/2, n=648)', fontsize=13)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    plt.xlim([ebn0_db_range[0], ebn0_db_range[-1]])
    plt.ylim([1e-5, 1])

    # Add waterfall region annotation
    plt.annotate('Waterfall\nregion', xy=(1.5, 2e-2), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = 'examples/ber_vs_ebn0.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "="*65)
    print("SIMULATION RESULTS")
    print("="*65)
    print(f"{'Eb/N0 (dB)':<12} {'Uncoded':<15} {'LDPC α=1.0':<15} {'LDPC α=0.75':<15}")
    print("-"*65)
    for ebn0, ber_u, ber_s, ber_n in zip(
        ebn0_db_range[::4], ber_uncoded[::4], ber_std_minsum[::4], ber_norm_minsum[::4]
    ):
        ber_u_str = f"{ber_u:.2e}" if ber_u > 0 else "0"
        ber_s_str = f"{ber_s:.2e}" if ber_s > 0 else "0"
        ber_n_str = f"{ber_n:.2e}" if ber_n > 0 else "0"
        print(f"{ebn0:<12.1f} {ber_u_str:<15} {ber_s_str:<15} {ber_n_str:<15}")

    print("-"*65)
    print("Eb/N0 @ BER=1e-3:")
    if not np.isnan(ebn0_uncoded_1e3):
        print(f"  Uncoded:              {ebn0_uncoded_1e3:.1f} dB")
    if not np.isnan(ebn0_std_1e3):
        print(f"  LDPC (α=1.0):         {ebn0_std_1e3:.1f} dB")
    if not np.isnan(ebn0_norm_1e3):
        print(f"  LDPC (α=0.75):        {ebn0_norm_1e3:.1f} dB")

    if not np.isnan(ebn0_uncoded_1e3) and not np.isnan(ebn0_norm_1e3):
        print(f"\nCoding gain (α=0.75):   {ebn0_uncoded_1e3 - ebn0_norm_1e3:.1f} dB")
    if not np.isnan(ebn0_std_1e3) and not np.isnan(ebn0_norm_1e3):
        print(f"Normalization benefit:  {ebn0_std_1e3 - ebn0_norm_1e3:.1f} dB")

    plt.show()


if __name__ == "__main__":
    main()
