import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameHeaderConstructor, ModulationSchemes

LENGTH_BITS = 8
SRC_BITS = 4
DST_BITS = 4
MOD_SCHEME_BITS = 3
CRC_BITS = 4

LENGTH_MAX = (1 << LENGTH_BITS) - 1
SRC_MAX = (1 << SRC_BITS) - 1
DST_MAX = (1 << DST_BITS) - 1

PADDING_BIT_POS = LENGTH_BITS + SRC_BITS + DST_BITS + MOD_SCHEME_BITS


class TestFrameHeaderConstructor:
    header_constructor = FrameHeaderConstructor(
        length_bits=LENGTH_BITS,
        src_bits=SRC_BITS,
        dst_bits=DST_BITS,
        mod_scheme_bits=MOD_SCHEME_BITS,
        crc_bits=CRC_BITS,
    )

    @given(
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
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
        length=st.integers(min_value=0, max_value=LENGTH_MAX),
        src=st.integers(min_value=0, max_value=SRC_MAX),
        dst=st.integers(min_value=0, max_value=DST_MAX),
        mod_scheme=st.sampled_from(ModulationSchemes),
        data=st.data(),
    )
    def test_crc_detects_burst_errors(self, length, src, dst, mod_scheme, data):
        encoded = self.header_constructor.encode(length, src, dst, mod_scheme)

        padding_bit_pos = PADDING_BIT_POS

        # CRC guarantees detection of burst errors up to CRC_BITS
        burst_len = data.draw(st.integers(min_value=1, max_value=CRC_BITS))

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
