"""
This file contains code to take information bits and frame information from the internet protocol thing,
and construct the tx frame for the rest of the system. This includes header data, pilots for channel estimation,
error correction information, and actual data information.
"""

import numpy as np
from channel_coding import CodeRates

class FrameHeader:
    def __init__(
        self,
        length_bits: int,
        src_bits: int,
        dst_bits: int,
        mod_scheme_bits: int,
        parity_bits: int
    ):
        self.length_bits     = length_bits
        self.src_bits        = src_bits
        self.dst_bits        = dst_bits
        self.mod_scheme_bits = mod_scheme_bits
        self.parity_bits     = parity_bits

        self.header_length = self.length_bits + self.src_bits + self.dst_bits + self.mod_scheme_bits + self.parity_bits
        self.header_length = 2 * int(np.ceil(self.header_length / 2))

    def encode(self, length, src, dst, mod_scheme):
        pass 

    def decode(self, header: np.ndarray):
        pass

class FrameConstructor:
    def __init__(
        self,
        frame_header_data_length_bits: int,
        frame_header_src_bits: int,
        frame_header_dst_bits: int,
        frame_header_mod_scheme_bits: int,
        frame_header_parity_bits: int,

        data_size: int,       # number of data bits per frame
        code_rate: CodeRates, # 
        pilots                # TODO: Figure out this
    ):
        self.frame_header_generator = FrameHeader(
            frame_header_data_length_bits,
            frame_header_src_bits,
            frame_header_dst_bits,
            frame_header_mod_scheme_bits,
            frame_header_parity_bits
        )

    def encode(self, data: np.ndarray):
        pass

    def decode(self, frame) -> np.ndarray:
        return np.array([], dtype=int)
