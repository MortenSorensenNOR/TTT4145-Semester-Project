import numpy as np
import matplotlib.pyplot as plt

class BPSK:
    def __init__(self):
        self.symbols = np.array([-1 + 0j, 1 + 0j])

    def bits2symbols(self, bitstream: np.ndarray):
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        return self.symbols[bitstream]

    def symbols2bits(self, symbols: np.ndarray):
        if len(symbols) == 0:
            return np.array([], dtype=int)
        return np.argmin(np.abs(symbols[:, None] - self.symbols[None, :]), axis=1)

    def plot_constellation(self):
        for i in range(len(self.symbols)):
            plt.plot(np.real(self.symbols), np.imag(self.symbols), 'bo')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.text(
                np.real(self.symbols[i]), np.imag(self.symbols[i]) - 0.1,
                str(i),
                horizontalalignment='center',
                verticalalignment='top',
            )
        plt.plot([0, 0], [-1, 1], color=(0, 0, 0), linewidth=0.5)
        plt.plot([-1.5, 1.5], [0, 0], color=(0, 0, 0), linewidth=0.5)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

class QPSK:
    def __init__(self):
        # Gray-coded QPSK: 00 -> -1-1j, 01 -> -1+1j, 10 -> +1-1j, 11 -> +1+1j
        self.symbols = np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)
        self.symbol_mapping = self.symbols
        self.bit_mapping = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.bits_per_symbol = 2
        self.qam_order = 4

    def bits2symbols(self, bitstream: np.ndarray):
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(-1, 2)
        indices = bitstream[:, 0] * 2 + bitstream[:, 1]
        return self.symbols[indices]

    def symbols2bits(self, symbols: np.ndarray):
        if len(symbols) == 0:
            return np.array([], dtype=int)
        indices = np.argmin(np.abs(symbols[:, None] - self.symbols[None, :]), axis=1)
        return np.column_stack([indices // 2, indices % 2])

    def symbols2bits_soft(self, symbols: np.ndarray, sigma_sq = None):
        """
        Compute log-likelihood ratios (LLRs) for soft decision decoding.

        LLR > 0 means bit is more likely 0
        LLR < 0 means bit is more likely 1
        |LLR| indicates confidence

        Args:
            symbols: Received complex symbols
            sigma_sq: Noise variance. If None, estimated from data.

        Returns:
            LLRs as (N, 2) array for bit0 (I) and bit1 (Q)
        """
        if len(symbols) == 0:
            return np.array([], dtype=float)

        # Estimate noise variance if not provided
        if sigma_sq is None:
            indices = np.argmin(np.abs(symbols[:, None] - self.symbols[None, :]), axis=1)
            noise = symbols - self.symbols[indices]
            sigma_sq = np.mean(np.abs(noise)**2)
            sigma_sq = max(sigma_sq, 1e-10)

        # For this QPSK mapping:
        # bit0=0 when Re<0, bit0=1 when Re>0
        # bit1=0 when Im<0, bit1=1 when Im>0
        # LLR = ln(P(bit=0)/P(bit=1)), so LLR>0 means bit=0
        # LLR(bit0) = -2*√2/σ² * Re(y)  (negative Re -> bit0=0 -> positive LLR)
        # LLR(bit1) = -2*√2/σ² * Im(y)  (negative Im -> bit1=0 -> positive LLR)
        scale = 2.0 * np.sqrt(2) / sigma_sq
        llr_bit0 = -scale * np.real(symbols)
        llr_bit1 = -scale * np.imag(symbols)

        return np.column_stack([llr_bit0, llr_bit1])

    def estimate_noise_variance(self, symbols: np.ndarray):
        """Estimate noise variance from received symbols using hard decisions."""
        if len(symbols) == 0:
            return 0.0
        indices = np.argmin(np.abs(symbols[:, None] - self.symbols[None, :]), axis=1)
        noise = symbols - self.symbols[indices]
        return np.mean(np.abs(noise)**2)

    def plot_constellation(self):
        labels = ['00', '01', '10', '11']
        plt.plot(np.real(self.symbols), np.imag(self.symbols), 'bo')
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        for i, symbol in enumerate(self.symbols):
            plt.text(np.real(symbol), np.imag(symbol) - 0.1, labels[i],
                     horizontalalignment='center', verticalalignment='top')
        plt.plot([0, 0], [-1, 1], color=(0, 0, 0), linewidth=0.5)
        plt.plot([-1, 1], [0, 0], color=(0, 0, 0), linewidth=0.5)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

    def plot_llr_heatmap(self, sigma_sq: float = 0.1, grid_size: int = 100):
        """
        Plot heatmaps showing LLR values across the complex plane.

        Shows how confidence (|LLR|) decreases near decision boundaries
        (Re=0 for bit0, Im=0 for bit1).

        Args:
            sigma_sq: Noise variance for LLR calculation
            grid_size: Resolution of the grid
        """
        # Create grid of possible received symbols
        extent = 1.5
        re = np.linspace(-extent, extent, grid_size)
        im = np.linspace(-extent, extent, grid_size)
        Re, Im = np.meshgrid(re, im)
        symbols_grid = (Re + 1j * Im).flatten()

        # Calculate LLRs
        llrs = self.symbols2bits_soft(symbols_grid, sigma_sq=sigma_sq)
        llr_bit0 = llrs[:, 0].reshape(grid_size, grid_size)
        llr_bit1 = llrs[:, 1].reshape(grid_size, grid_size)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot LLR for bit 0 (I component)
        im0 = axes[0].imshow(llr_bit0, extent=[-extent, extent, -extent, extent],
                             origin='lower', cmap='RdBu', aspect='equal')
        axes[0].axvline(x=0, color='k', linestyle='--', linewidth=1, label='Decision boundary')
        axes[0].plot(np.real(self.symbols), np.imag(self.symbols), 'ko', markersize=8)
        axes[0].set_xlabel('Real')
        axes[0].set_ylabel('Imaginary')
        axes[0].set_title('LLR for Bit 0 (I)\nBlue: likely 0, Red: likely 1')
        plt.colorbar(im0, ax=axes[0], label='LLR')

        # Plot LLR for bit 1 (Q component)
        im1 = axes[1].imshow(llr_bit1, extent=[-extent, extent, -extent, extent],
                             origin='lower', cmap='RdBu', aspect='equal')
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1, label='Decision boundary')
        axes[1].plot(np.real(self.symbols), np.imag(self.symbols), 'ko', markersize=8)
        axes[1].set_xlabel('Real')
        axes[1].set_ylabel('Imaginary')
        axes[1].set_title('LLR for Bit 1 (Q)\nBlue: likely 0, Red: likely 1')
        plt.colorbar(im1, ax=axes[1], label='LLR')

        # Plot confidence (minimum |LLR|) - shows uncertainty near boundaries
        confidence = np.minimum(np.abs(llr_bit0), np.abs(llr_bit1))
        im2 = axes[2].imshow(confidence, extent=[-extent, extent, -extent, extent],
                             origin='lower', cmap='viridis', aspect='equal')
        axes[2].axvline(x=0, color='w', linestyle='--', linewidth=1)
        axes[2].axhline(y=0, color='w', linestyle='--', linewidth=1)
        axes[2].plot(np.real(self.symbols), np.imag(self.symbols), 'wo', markersize=8)
        axes[2].set_xlabel('Real')
        axes[2].set_ylabel('Imaginary')
        axes[2].set_title(f'Min Confidence |LLR| (σ²={sigma_sq})\nDark = uncertain')
        plt.colorbar(im2, ax=axes[2], label='|LLR|')

        plt.tight_layout()
        return fig, axes

class QAM:
    def __init__(self, qam_order):
        bits_per_symbol = int(np.log2(qam_order))

        # Gray coding
        iq = 2 * np.arange(np.sqrt(qam_order)) - np.sqrt(qam_order) + 1
        q_rep, i_rep = np.meshgrid(iq, iq)
        symbols = i_rep.reshape(qam_order) + 1j * q_rep.reshape(qam_order)
        symbols = symbols / np.sqrt(np.mean(np.abs(symbols)**2))

        a = int(np.sqrt(qam_order)/2)
        bitmapping_atom = np.hstack((np.ones(a), np.zeros(a)))
        for i in range(int(bits_per_symbol/2 - 1)):
            BitTemp = bitmapping_atom if i == 0 else bitmapping_atom[-1,:]
            bitmapping_atom = np.vstack((bitmapping_atom, np.hstack((BitTemp[::2], BitTemp[::-2]))))
        bitmapping_atom = bitmapping_atom.T
        bit_mapping = np.zeros((qam_order, bits_per_symbol))

        for x_iq in iq:
            index_i = np.nonzero(i_rep.reshape(qam_order) == x_iq)
            index_q = np.nonzero(q_rep.reshape(qam_order) == x_iq)

            if qam_order == 4:
                bit_mapping[index_i, 1] = bitmapping_atom
                bit_mapping[index_q, 0] = bitmapping_atom
            else:
                bit_mapping[index_i, 1::2] = bitmapping_atom
                bit_mapping[index_q,  ::2] = bitmapping_atom
        bin2dec = np.sum(bit_mapping * 2**np.arange(bits_per_symbol), axis=1, dtype=int)

        self.bit_mapping     = bit_mapping[np.argsort(bin2dec),:]
        self.symbol_mapping  = symbols[np.argsort(bin2dec)]
        self.bits_per_symbol = bits_per_symbol
        self.qam_order       = qam_order

    def bits2symbols(self, bitstream: np.ndarray):
        if len(bitstream) == 0:
            return np.array([], dtype=complex)
        bitstream = bitstream.reshape(int(np.size(bitstream)/self.bits_per_symbol), self.bits_per_symbol)
        return self.symbol_mapping[np.sum(bitstream * 2**np.arange(self.bits_per_symbol), axis=1, dtype=int)]

    def symbols2bits(self, symbols: np.ndarray):
        if len(symbols) == 0:
            return np.array([], dtype=int)
        distance_symbols_to_constellation = np.abs(symbols.reshape(np.size(symbols), 1, order='F') - self.symbol_mapping)
        return self.bit_mapping[np.argmin(distance_symbols_to_constellation, axis=1),:]

    def plot_constellation(self):
        for i in range(np.size(self.symbol_mapping)):
            plt.plot(np.real(self.symbol_mapping), np.imag(self.symbol_mapping),' bo')
            plt.xlabel('Real Part')
            plt.ylabel('Imaginary Part')
            plt.text(
                np.real(self.symbol_mapping[i]), np.imag(self.symbol_mapping[i])-0.02, 
                np.array2string(self.bit_mapping[i,:]).replace('[','').replace(']','').replace('.','').replace(' ',''),
                horizontalalignment='center',
                verticalalignment='top', 
             )
            plt.plot([0,0], [-1,1], color=(0,0,0), linewidth=0.5)
            plt.plot([-1,1], [0,0], color=(0,0,0), linewidth=0.5)
