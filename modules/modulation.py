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

class QPSK(QAM):
    def __init__(self):
        super().__init__(qam_order=4)
