import numpy as np
import matplotlib.pyplot as plt


def plot_iq(*signals, labels=None, title=None, sample_rate=None):
    """
    Plot I and Q components of complex signals.

    Args:
        *signals: Complex ndarrays to plot
        labels: List of labels for each signal (optional)
        title: Plot title (optional)
        sample_rate: If provided, x-axis shows time in seconds

    Example:
        plot_iq(tx_signal, rx_signal, labels=['TX', 'RX'])
        plot_iq(signal1, signal2, signal3)  # auto-labeled
    """
    if labels is None:
        labels = [f'Signal {i+1}' for i in range(len(signals))]

    fig, (ax_i, ax_q) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for sig, label in zip(signals, labels):
        if sample_rate:
            t = np.arange(len(sig)) / sample_rate * 1e6  # microseconds
            ax_i.plot(t, np.real(sig), label=label)
            ax_q.plot(t, np.imag(sig), label=label)
            ax_q.set_xlabel('Time [us]')
        else:
            ax_i.plot(np.real(sig), label=label)
            ax_q.plot(np.imag(sig), label=label)
            ax_q.set_xlabel('Sample')

    ax_i.set_ylabel('I (Real)')
    ax_q.set_ylabel('Q (Imag)')
    ax_i.legend()
    ax_q.legend()
    ax_i.grid(True, alpha=0.3)
    ax_q.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig, (ax_i, ax_q)


def plot_constellation(*signals, labels=None, title=None):
    """
    Plot constellation diagram of complex signals.

    Args:
        *signals: Complex ndarrays to plot
        labels: List of labels for each signal (optional)
        title: Plot title (optional)
    """
    if labels is None:
        labels = [f'Signal {i+1}' for i in range(len(signals))]

    fig, ax = plt.subplots(figsize=(6, 6))

    for sig, label in zip(signals, labels):
        ax.scatter(np.real(sig), np.imag(sig), label=label, alpha=0.6, s=10)

    ax.set_xlabel('I (Real)')
    ax.set_ylabel('Q (Imag)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_constellation_confidence(rx_symbols, llrs, tx_symbols=None, title=None):
    """
    Plot constellation with symbols colored by soft decision confidence.

    Args:
        rx_symbols: Received complex symbols
        llrs: LLR values from symbols2bits_soft, shape (N, bits_per_symbol)
        tx_symbols: Ideal constellation points to overlay (optional)
        title: Plot title (optional)

    The confidence is computed as min(|LLR|) across bits for each symbol.
    Low confidence (near decision boundaries) appears darker.
    """
    # Confidence = minimum |LLR| across bits for each symbol
    confidence = np.min(np.abs(llrs), axis=1)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot received symbols colored by confidence
    scatter = ax.scatter(
        np.real(rx_symbols), np.imag(rx_symbols),
        c=confidence, cmap='plasma', alpha=0.7, s=15,
        label='Received'
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence (min |LLR|)')

    # Overlay ideal constellation points
    if tx_symbols is not None:
        unique_tx = np.unique(tx_symbols)
        ax.scatter(
            np.real(unique_tx), np.imag(unique_tx),
            c='white', edgecolors='black', s=100, marker='s',
            linewidths=1.5, label='Ideal', zorder=10
        )

    # Decision boundaries
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('I (Real)')
    ax.set_ylabel('Q (Imag)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Constellation with Soft Decision Confidence')

    plt.tight_layout()
    return fig, ax


def plot_spectrum(*signals, sample_rate=1.0, labels=None, title=None):
    """
    Plot power spectrum of complex signals.

    Args:
        *signals: Complex ndarrays to plot
        sample_rate: Sample rate in Hz
        labels: List of labels for each signal (optional)
        title: Plot title (optional)
    """
    if labels is None:
        labels = [f'Signal {i+1}' for i in range(len(signals))]

    fig, ax = plt.subplots(figsize=(10, 4))

    for sig, label in zip(signals, labels):
        n = len(sig)
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1/sample_rate)) / 1e6  # MHz
        spectrum = np.fft.fftshift(np.fft.fft(sig))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-10)
        power_db -= np.max(power_db)  # normalize to 0 dB peak
        ax.plot(freq, power_db, label=label)

    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('Power [dB]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-60, 5])

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax
