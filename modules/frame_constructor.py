"""
This file contains code to take information bits and frame information from the internet protocol thing,
and construct the tx frame for the rest of the system. This includes header data, pilots for channel estimation,
error correction information, and actual data information.
"""

import numpy as np
from enum import Enum
from channel_coding import CodeRates

class ModulationSchemes(Enum):
    BPSK  = 1
    QPSK  = 2
    QAM16 = 3
    QAM64 = 4

class FrameHeader:
    def __init__(
        self,
        length,
        src,
        dst,
        mod_scheme,
        crc,
        crc_passed = True
    ):
        self.length = length
        self.src = src
        self.dst = dst
        self.mod_scheme = mod_scheme
        self.crc = crc
        self.crc_passed = crc_passed

class FrameHeaderGenerator:
    def __init__(
        self,
        length_bits: int,
        src_bits: int,
        dst_bits: int,
        mod_scheme_bits: int,
        crc_bits: int
    ):
        self.length_bits     = length_bits
        self.src_bits        = src_bits
        self.dst_bits        = dst_bits
        self.mod_scheme_bits = mod_scheme_bits
        self.crc_bits     = crc_bits

        self.header_length = self.length_bits + self.src_bits + self.dst_bits + self.mod_scheme_bits + self.crc_bits
        self.header_length = 2 * int(np.ceil(self.header_length / 2))

    def _crc_calc(self, data_bits, poly=0b10011):
        reg = data_bits << 4
        for i in range(data_bits.bit_length() - 1, -1, -1):
            if reg & (1 << (i + 4)):
                reg ^= poly << i
        return reg & 0b1111

    def encode(
        self, 
        length: int, 
        src: int, 
        dst: int, 
        mod_scheme: ModulationSchemes
    ):
        length_bits = int_to_bits(length, self.length_bits)
        src_bits    = int_to_bits(src, self.src_bits)
        dst_bits    = int_to_bits(dst, self.src_bits)
        mod_scheme_bits = int_to_bits(mod_scheme, self.mod_scheme_bits)

        header_data_bits = length_bits + src_bits + dst_bits + mod_scheme_bits + [0] # +1 to even out
        data_bits = "".join([str(bit) for bit in header_data_bits])
        crc_bits = self._crc_calc(data_bits) 
        header_data_bits += int_to_bits(crc_bits, 4)

        return header_data_bits

    def decode(self, header: np.ndarray) -> FrameHeader:
        self.idx = 0
        length = self._get_length_bits(header)
        src = self._get_src_dst_bits(header)
        dst = self._get_src_dst_bits(header)
        mod_scheme = self._get_mod_scheme_bits(header)
        crc = self._get_crc_bits(header)

        # check crc
        header_data_bits = "".join([str(bit) for bit in (length + src + dst + mod_scheme + [0])])
        calculated_crc = self._crc_calc(header_data_bits)
        calculated_crc = int_to_bits(calculated_crc, 4)
        print(f"Expected crc: {crc}    Got: {calculated_crc}")

        frame_header = FrameHeader(length, src, dst, mod_scheme, crc, crc == calculated_crc)
        return frame_header

    def _get_length_bits(self, header):
        length = header[self.idx:self.idx+self.length_bits]
        self.idx += self.length_bits
        return length

    def _get_src_dst_bits(self, header):
        assert(self.src_bits == self.dst_bits)
        sd = header[self.idx:self.idx+self.src_bits]
        self.idx += self.src_bits
        return sd

    def _get_mod_scheme_bits(self, header):
        scheme = header[self.idx:self.idx+self.mod_scheme_bits]
        self.idx += self.mod_scheme_bits
        return scheme

    def _get_crc_bits(self, header):
        crc = header[self.idx:self.idx+self.crc_bits]
        self.idx += self.crc_bits
        return crc

class FrameConstructor:
    def __init__(
        self,
        data_size: int,       # number of data bits per frame
        code_rate: CodeRates, # 
        pilots,               # TODO: Figure out this

        payload_length_bits: int = 8,
        src_bits: int = 4,
        dst_bits: int = 4,
        mod_scheme_bits: int = 3,
        crc_bits: int = 4,
    ):
        self.frame_header_generator = FrameHeaderGenerator(
            payload_length_bits,
            src_bits,
            dst_bits,
            mod_scheme_bits,
            crc_bits
        )

    def encode(self, data: np.ndarray):
        pass

    def decode(self, frame) -> np.ndarray:
        return np.array([], dtype=int)

# Util
def int_to_bits(n, length):
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]
