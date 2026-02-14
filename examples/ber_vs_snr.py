#!/usr/bin/env python3
"""BER vs SNR simulation comparing LDPC-coded vs uncoded QPSK.

Demonstrates the "waterfall" effect of LDPC coding where BER drops
rapidly after a certain SNR threshold.

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


def simulate_uncoded_qpsk(snr_db_range: np.ndarray, n_bits: int, n_trials: int) -> np.ndarray:
    """Simulate uncoded QPSK with hard decision decoding."""
    qpsk = QPSK()
    ber = np.zeros(len(snr_db_range))

    for i, snr_db in enumerate(tqdm(snr_db_range, desc="Uncoded QPSK    ")):
        total_errors = 0
        total_bits = 0

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
    snr_db_range: np.ndarray,
    n_trials: int,
    alpha: float = 0.75,
    desc: str = "LDPC + QPSK",
) -> np.ndarray:
    """Simulate LDPC-coded QPSK with soft decision decoding.

    Args:
        snr_db_range: Array of SNR values to simulate
        n_trials: Number of frames per SNR point
        alpha: Min-sum normalization factor (0.75=normalized, 1.0=standard)
        desc: Description for progress bar
    """
    # LDPC config: rate 1/2, k=324 message bits, n=648 codeword bits
    config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
    ldpc = LDPC(config)
    qpsk = QPSK()

    # Warm up Numba JIT
    _ = ldpc.decode_numba(np.zeros(648), max_iterations=5, alpha=alpha)

    ber = np.zeros(len(snr_db_range))

    for i, snr_db in enumerate(tqdm(snr_db_range, desc=f"{desc:16}")):
        total_errors = 0
        total_bits = 0

        for trial in range(n_trials):
            # Generate random message bits
            message = np.random.randint(0, 2, ldpc.k)

            # LDPC encode
            codeword = ldpc.encode(message)

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

            # LDPC decode (Numba-accelerated)
            decoded = ldpc.decode_numba(llrs.flatten(), max_iterations=50, alpha=alpha)

            # Count errors (on message bits, not codeword)
            total_errors += np.sum(message != decoded)
            total_bits += ldpc.k

        ber[i] = total_errors / total_bits if total_bits > 0 else 0

    return ber


def theoretical_qpsk_ber(snr_db: np.ndarray) -> np.ndarray:
    """Theoretical BER for uncoded QPSK over AWGN channel."""
    from scipy.special import erfc
    snr_linear = 10 ** (snr_db / 10)
    # For QPSK: BER = 0.5 * erfc(sqrt(SNR))
    return 0.5 * erfc(np.sqrt(snr_linear))


def measure_coding_gain_at_target_ber(target_ber: float = 1e-6, n_trials: int = 10000):
    """Measure coding gain at a specific target BER using high trial count.

    Uses theoretical QPSK for uncoded (exact) and simulation for LDPC.
    """
    from scipy.special import erfcinv

    print(f"\n{'='*60}")
    print(f"MEASURING CODING GAIN AT BER = {target_ber:.0e}")
    print(f"{'='*60}")

    # Theoretical SNR for uncoded QPSK at target BER
    # BER = 0.5 * erfc(sqrt(SNR)) => SNR = erfcinv(2*BER)^2
    snr_uncoded_theory = erfcinv(2 * target_ber) ** 2
    snr_uncoded_db = 10 * np.log10(snr_uncoded_theory)
    print(f"\nUncoded QPSK @ BER={target_ber:.0e}: {snr_uncoded_db:.2f} dB (theoretical)")

    # For LDPC, simulate around expected waterfall region
    config = LDPCConfig(n=648, k=324, Z=27, code_rate=CodeRates.HALF_RATE)
    ldpc = LDPC(config)
    qpsk = QPSK()

    # Warm up Numba
    _ = ldpc.decode_numba(np.zeros(648), max_iterations=5, alpha=0.75)

    # Fine SNR sweep in waterfall region
    snr_test = np.arange(1.5, 3.5, 0.1)

    print(f"\nSimulating LDPC (α=0.75) with {n_trials} trials per SNR point...")
    print(f"Total bits per SNR point: {n_trials * 324:,}")
    print()

    results = []

    for snr_db in tqdm(snr_test, desc="LDPC measurement"):
        total_errors = 0
        total_bits = 0

        for trial in range(n_trials):
            message = np.random.randint(0, 2, ldpc.k)
            codeword = ldpc.encode(message)
            symbols = qpsk.bits2symbols(codeword)

            channel = ChannelModel(ChannelConfig(snr_db=snr_db, seed=trial))
            rx_symbols = channel.apply(symbols)

            sigma_sq = qpsk.estimate_noise_variance(rx_symbols)
            llrs = qpsk.symbols2bits_soft(rx_symbols, sigma_sq=sigma_sq)
            decoded = ldpc.decode_numba(llrs.flatten(), max_iterations=50, alpha=0.75)

            total_errors += np.sum(message != decoded)
            total_bits += ldpc.k

        ber = total_errors / total_bits if total_bits > 0 else 0
        results.append((snr_db, ber, total_errors, total_bits))

        if ber > 0:
            print(f"  SNR={snr_db:.1f} dB: BER={ber:.2e} ({total_errors} errors / {total_bits:,} bits)")
        else:
            print(f"  SNR={snr_db:.1f} dB: BER=0 (no errors in {total_bits:,} bits)")

    # Find SNR at target BER by interpolation
    snr_ldpc_db = None
    for i in range(len(results) - 1):
        snr1, ber1, _, _ = results[i]
        snr2, ber2, _, _ = results[i + 1]

        if ber1 >= target_ber >= ber2 and ber1 > 0 and ber2 > 0:
            # Log-linear interpolation
            log_ber1, log_ber2 = np.log10(ber1), np.log10(ber2)
            log_target = np.log10(target_ber)
            snr_ldpc_db = snr1 + (snr2 - snr1) * (log_target - log_ber1) / (log_ber2 - log_ber1)
            break
        elif ber2 == 0 and ber1 > 0 and ber1 <= target_ber:
            # Target is between last non-zero and zero
            snr_ldpc_db = snr1
            print(f"\n  Note: BER dropped to 0 after {snr1:.1f} dB, using as estimate")
            break

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Target BER:                  {target_ber:.0e}")
    print(f"Uncoded QPSK SNR:            {snr_uncoded_db:.2f} dB (theoretical)")

    if snr_ldpc_db is not None:
        print(f"LDPC (α=0.75) SNR:           {snr_ldpc_db:.2f} dB (simulated)")
        coding_gain = snr_uncoded_db - snr_ldpc_db
        print(f"Coding gain:                 {coding_gain:.2f} dB")
    else:
        # Find lowest BER achieved
        min_ber = min(r[1] for r in results if r[1] > 0) if any(r[1] > 0 for r in results) else 0
        if min_ber > 0:
            print(f"LDPC (α=0.75) SNR:           Could not reach {target_ber:.0e}")
            print(f"  Lowest BER measured:       {min_ber:.2e}")
            print(f"  Need more trials or lower target BER")
        else:
            last_nonzero_snr = max(r[0] for r in results if r[1] > 0) if any(r[1] > 0 for r in results) else results[0][0]
            print(f"LDPC (α=0.75) SNR:           <{last_nonzero_snr:.1f} dB (all frames decoded)")
            coding_gain = snr_uncoded_db - last_nonzero_snr
            print(f"Coding gain:                 >{coding_gain:.1f} dB")

    print(f"{'='*60}\n")


def main():
    # Simulation parameters
    # Fine resolution around waterfall region (0-4 dB), coarser elsewhere
    snr_low = np.arange(-2, 0, 0.5)       # -2 to 0 dB, 0.5 dB steps
    snr_waterfall = np.arange(0, 4, 0.2)  # 0 to 4 dB, 0.2 dB steps (fine resolution)
    snr_high = np.arange(4, 12, 0.5)      # 4 to 12 dB, 0.5 dB steps
    snr_db_range = np.concatenate([snr_low, snr_waterfall, snr_high])

    n_trials = 1000  # Number of frames per SNR point (more = smoother curves)
    n_bits_uncoded = 648  # Same as LDPC codeword length for fair comparison

    print("Running BER vs SNR simulation...")
    print(f"SNR range: {snr_db_range[0]} to {snr_db_range[-1]} dB")
    print(f"Trials per SNR: {n_trials}")
    print("Using Numba-accelerated decoder")
    print()

    np.random.seed(42)

    # Run simulations
    ber_uncoded = simulate_uncoded_qpsk(snr_db_range, n_bits_uncoded, n_trials)

    # Standard min-sum (alpha=1.0)
    ber_std_minsum = simulate_ldpc_coded_qpsk(
        snr_db_range, n_trials, alpha=1.0, desc="LDPC (α=1.0)"
    )

    # Normalized min-sum (alpha=0.75)
    ber_norm_minsum = simulate_ldpc_coded_qpsk(
        snr_db_range, n_trials, alpha=0.75, desc="LDPC (α=0.75)"
    )

    ber_theory = theoretical_qpsk_ber(snr_db_range)

    # Measure coding gain at 10^-6
    measure_coding_gain_at_target_ber(target_ber=1e-6, n_trials=10000)

    # Calculate coding gain at BER = 1e-3
    def find_snr_at_ber(snr_range, ber, target_ber):
        """Interpolate to find SNR at target BER."""
        valid = ber > 0
        if not np.any(valid):
            return np.nan
        log_ber = np.log10(ber[valid])
        snr_valid = snr_range[valid]
        if np.min(log_ber) > np.log10(target_ber):
            return np.nan  # Never reaches target BER
        idx = np.searchsorted(-log_ber, -np.log10(target_ber))
        if idx == 0 or idx >= len(snr_valid):
            return snr_valid[min(idx, len(snr_valid)-1)]
        # Linear interpolation
        x1, x2 = snr_valid[idx-1], snr_valid[idx]
        y1, y2 = log_ber[idx-1], log_ber[idx]
        return x1 + (x2 - x1) * (np.log10(target_ber) - y1) / (y2 - y1)

    snr_uncoded_1e3 = find_snr_at_ber(snr_db_range, ber_uncoded, 1e-3)
    snr_std_1e3 = find_snr_at_ber(snr_db_range, ber_std_minsum, 1e-3)
    snr_norm_1e3 = find_snr_at_ber(snr_db_range, ber_norm_minsum, 1e-3)

    # Plot results
    plt.figure(figsize=(10, 7))

    # Plot simulated results
    plt.semilogy(snr_db_range, ber_uncoded, 'bo-', label='Uncoded QPSK (hard decision)',
                 markersize=4, linewidth=1.5)
    plt.semilogy(snr_db_range, ber_std_minsum, 'g^-',
                 label='LDPC + standard min-sum (α=1.0)',
                 markersize=4, linewidth=1.5)
    plt.semilogy(snr_db_range, ber_norm_minsum, 'rs-',
                 label='LDPC + normalized min-sum (α=0.75)',
                 markersize=4, linewidth=1.5)

    # Plot theoretical curve
    plt.semilogy(snr_db_range, ber_theory, 'k--', label='Theoretical QPSK',
                 linewidth=1.5, alpha=0.7)

    # Add coding gain annotations
    plt.axhline(y=1e-3, color='gray', linestyle=':', alpha=0.5)

    if not np.isnan(snr_uncoded_1e3) and not np.isnan(snr_norm_1e3):
        coding_gain_norm = snr_uncoded_1e3 - snr_norm_1e3
        plt.annotate(f'Coding gain (normalized): {coding_gain_norm:.1f} dB',
                    xy=(snr_norm_1e3, 1e-3),
                    xytext=(snr_norm_1e3 + 1, 3e-3),
                    fontsize=9, ha='left',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    if not np.isnan(snr_std_1e3) and not np.isnan(snr_norm_1e3):
        norm_gain = snr_std_1e3 - snr_norm_1e3
        if norm_gain > 0.1:  # Only annotate if there's meaningful difference
            plt.annotate(f'Normalization gain: ~{norm_gain:.1f} dB',
                        xy=((snr_std_1e3 + snr_norm_1e3) / 2, 1e-3),
                        xytext=((snr_std_1e3 + snr_norm_1e3) / 2, 1e-4),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER vs SNR: Standard vs Normalized Min-Sum LDPC Decoding\n'
              '(IEEE 802.11 LDPC, Rate 1/2, n=648)', fontsize=13)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(loc='lower left', fontsize=9)
    plt.xlim([snr_db_range[0], snr_db_range[-1]])
    plt.ylim([1e-5, 1])

    # Add waterfall region annotation
    plt.annotate('Waterfall\nregion', xy=(1.5, 2e-2), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_path = 'examples/ber_vs_snr.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary
    print("\n" + "="*65)
    print("SIMULATION RESULTS")
    print("="*65)
    print(f"{'SNR (dB)':<10} {'Uncoded':<15} {'LDPC α=1.0':<15} {'LDPC α=0.75':<15}")
    print("-"*65)
    for snr, ber_u, ber_s, ber_n in zip(
        snr_db_range[::4], ber_uncoded[::4], ber_std_minsum[::4], ber_norm_minsum[::4]
    ):
        ber_u_str = f"{ber_u:.2e}" if ber_u > 0 else "0"
        ber_s_str = f"{ber_s:.2e}" if ber_s > 0 else "0"
        ber_n_str = f"{ber_n:.2e}" if ber_n > 0 else "0"
        print(f"{snr:<10.1f} {ber_u_str:<15} {ber_s_str:<15} {ber_n_str:<15}")

    print("-"*65)
    print("SNR @ BER=1e-3:")
    if not np.isnan(snr_uncoded_1e3):
        print(f"  Uncoded:              {snr_uncoded_1e3:.1f} dB")
    if not np.isnan(snr_std_1e3):
        print(f"  LDPC (α=1.0):         {snr_std_1e3:.1f} dB")
    if not np.isnan(snr_norm_1e3):
        print(f"  LDPC (α=0.75):        {snr_norm_1e3:.1f} dB")

    if not np.isnan(snr_uncoded_1e3) and not np.isnan(snr_norm_1e3):
        print(f"\nCoding gain (α=0.75):   {snr_uncoded_1e3 - snr_norm_1e3:.1f} dB")
    if not np.isnan(snr_std_1e3) and not np.isnan(snr_norm_1e3):
        print(f"Normalization benefit:  {snr_std_1e3 - snr_norm_1e3:.1f} dB")

    plt.show()


if __name__ == "__main__":
    main()
