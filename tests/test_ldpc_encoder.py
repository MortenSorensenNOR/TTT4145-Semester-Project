"""Bit-exact correctness checks for the C++ structured LDPC encoder.

Compares `ldpc_encode` and `ldpc_encode_batch` against the reference
`message @ G % 2` matmul, and verifies syndrome zero against the same
permuted parity-check matrix the BP decoder consumes. Covers all 9
(n, code_rate) combinations the project supports.
"""

import numpy as np
import pytest

from modules.ldpc.channel_coding import CodeRates
from modules.ldpc.ldpc import (
    LDPCConfig,
    _get_encoding_structures,
    ldpc_clear_cache,
    ldpc_encode,
    ldpc_encode_batch,
)

CODE_RATES = [
    CodeRates.TWO_THIRDS_RATE,
    CodeRates.THREE_QUARTER_RATE,
    CodeRates.FIVE_SIXTH_RATE,
]
BLOCK_LENGTHS = [648, 1296, 1944]


def _config(n: int, code_rate: CodeRates) -> LDPCConfig:
    num, denom = code_rate.rate_fraction
    return LDPCConfig(k=n * num // denom, code_rate=code_rate)


@pytest.mark.parametrize("n", BLOCK_LENGTHS)
@pytest.mark.parametrize("code_rate", CODE_RATES)
def test_encoder_matches_reference_and_zero_syndrome(n, code_rate):
    """50 random messages: structured encoder == G-matmul, H @ codeword == 0."""
    ldpc_clear_cache()
    cfg = _config(n, code_rate)
    g_mat, h_perm = _get_encoding_structures(cfg)

    rng = np.random.default_rng(seed=hash((n, code_rate.value)) & 0xFFFF_FFFF)
    for _ in range(50):
        msg = rng.integers(0, 2, cfg.k, dtype=np.uint8)
        cw = ldpc_encode(msg, cfg)
        ref = (msg @ g_mat) % 2
        assert np.array_equal(cw, ref), "structured encoder disagrees with G-matmul reference"
        assert np.array_equal(cw[:cfg.k], msg), "encoder is not systematic"
        assert not (h_perm @ cw % 2).any(), "H @ codeword has nonzero syndrome"


@pytest.mark.parametrize("n", BLOCK_LENGTHS)
@pytest.mark.parametrize("code_rate", CODE_RATES)
def test_batch_encoder_matches_per_codeword(n, code_rate):
    """Batched encoder produces row-by-row identical output to the per-codeword path."""
    cfg = _config(n, code_rate)
    rng = np.random.default_rng(seed=(hash((n, code_rate.value, "batch")) & 0xFFFF_FFFF))
    n_cw = 7
    msgs = rng.integers(0, 2, (n_cw, cfg.k), dtype=np.uint8)

    batched = ldpc_encode_batch(msgs, cfg)
    assert batched.shape == (n_cw, cfg.n)

    for i in range(n_cw):
        single = ldpc_encode(msgs[i], cfg)
        assert np.array_equal(batched[i], single), f"row {i} differs from per-codeword encoder"


def test_encode_rejects_wrong_message_length():
    cfg = _config(1944, CodeRates.FIVE_SIXTH_RATE)
    with pytest.raises(ValueError):
        ldpc_encode(np.zeros(cfg.k - 1, dtype=np.uint8), cfg)


def test_batch_encode_rejects_non_multiple_length():
    cfg = _config(1944, CodeRates.FIVE_SIXTH_RATE)
    with pytest.raises(ValueError):
        ldpc_encode_batch(np.zeros(cfg.k + 1, dtype=np.uint8), cfg)
