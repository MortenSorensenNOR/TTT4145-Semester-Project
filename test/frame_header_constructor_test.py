import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameHeaderConstructor, ModulationSchemes


class TestFrameHeaderConstructor:
    header_constructor = FrameHeaderConstructor(
        length_bits=8,
        src_bits=4,
        dst_bits=4,
        mod_scheme_bits=3,
        crc_bits=4,
    )

    @given(
        length=st.integers(min_value=0, max_value=255),      # 8 bits
        src=st.integers(min_value=0, max_value=15),          # 4 bits
        dst=st.integers(min_value=0, max_value=15),          # 4 bits
        mod_scheme=st.sampled_from(ModulationSchemes),
    )
    def test_roundtrip(self, length, src, dst, mod_scheme):
        encoded = self.header_constructor.encode(length, src, dst, mod_scheme)
        decoded = self.header_constructor.decode(np.array(encoded))

        assert decoded.length == length
        assert decoded.src == src
        assert decoded.dst == dst
        assert decoded.mod_scheme == mod_scheme
        assert decoded.crc_passed

    @given(
        length=st.integers(min_value=0, max_value=255),
        src=st.integers(min_value=0, max_value=15),
        dst=st.integers(min_value=0, max_value=15),
        mod_scheme=st.sampled_from(ModulationSchemes),
        data=st.data(),  # for dynamic generation
    )
    def test_crc_detects_burst_errors(self, length, src, dst, mod_scheme, data):
        encoded = self.header_constructor.encode(length, src, dst, mod_scheme)

        # Position 19 is the padding bit which is skipped during decode
        padding_bit_pos = 8 + 4 + 4 + 3  # length + src + dst + mod_scheme

        # CRC-4 guarantees detection of burst errors up to 4 bits
        burst_len = data.draw(st.integers(min_value=1, max_value=4))

        # Pick a valid start position that avoids the padding bit
        # and doesn't go out of bounds
        max_start = len(encoded) - burst_len
        valid_starts = [
            i for i in range(max_start + 1)
            if padding_bit_pos not in range(i, i + burst_len)
        ]
        burst_start = data.draw(st.sampled_from(valid_starts))

        # Flip consecutive bits (burst error)
        for i in range(burst_start, burst_start + burst_len):
            encoded[i] ^= 1

        try:
            decoded = self.header_constructor.decode(np.array(encoded))
            # If decode succeeds, CRC should fail
            assert not decoded.crc_passed
        except ValueError:
            # Corruption caused invalid enum value - also counts as detection
            pass
