"""LDPC encoding/decoding using IEEE 802.11 base matrices.

The hot paths are the C++ extension (modules.ldpc.ldpc_ext); this module
just builds the structures the extension needs (encoder packed tables,
decoder edge lists) and caches them per LDPCConfig.
"""

import numpy as np
from dataclasses import dataclass
from .channel_coding import CodeRates

import logging
logger = logging.getLogger(__name__)

try:
    from modules.ldpc import ldpc_ext as _ext
    logger.info("Loaded ldpc_ext pybind11 C++ extension.")
except ImportError as e:
    msg = "ldpc_ext C++ extension is required. Build it with: uv sync --reinstall"
    raise RuntimeError(msg) from e


@dataclass(frozen=True)
class LDPCConfig:
    """Configuration for LDPC encoding/decoding"""

    k: int
    code_rate: CodeRates

    @property
    def n(self) -> int:
        num, denom = self.code_rate.rate_fraction
        n = (self.k * denom) // num
        if n not in (648, 1296, 1944):
            msg = f"Invalid k={self.k} for {self.code_rate}: computed n={n} is not a valid block length (648, 1296, or 1944)."
            raise ValueError(msg)
        return n

    @property
    def z(self) -> int:
        """Circulant size derived from n"""
        return {648: 27, 1296: 54, 1944: 81}[self.n]


# Per-config caches for derived structures.
_encoding_cache: dict[LDPCConfig, tuple[np.ndarray, np.ndarray]] = {}
_encoder_tables_cache: dict[LDPCConfig, tuple] = {}
_decode_cache: dict[LDPCConfig, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def ldpc_clear_cache() -> None:
    """Clear all cached LDPC structures"""
    _encoding_cache.clear()
    _encoder_tables_cache.clear()
    _decode_cache.clear()


def interleave(bits: np.ndarray, n: int) -> np.ndarray:
    """Randomly permute bits using a seed derived from the codeword length.

    TX and RX produce identical permutations from n alone -- no signaling needed.
    """
    rng = np.random.default_rng(seed=n)
    perm = rng.permutation(len(bits))
    return bits[perm]


def deinterleave(bits: np.ndarray, n: int) -> np.ndarray:
    """Inverse of interleave(): restore original bit order."""
    rng = np.random.default_rng(seed=n)
    perm = rng.permutation(len(bits))
    out = np.empty_like(bits)
    out[perm] = bits
    return out


def ldpc_encode(message: np.ndarray, config: LDPCConfig) -> np.ndarray:
    """Encode a k-bit message to an n-bit codeword"""
    if len(message) != config.k:
        msg = f"Message length {len(message)} != expected {config.k}"
        raise ValueError(msg)
    msg_u8 = np.ascontiguousarray(message, dtype=np.uint8)
    h_s_packed, h_p_inv_cols, _k, n = _get_encoder_tables(config)
    return _ext.encode(msg_u8, h_s_packed, h_p_inv_cols, n)


def ldpc_encode_batch(messages: np.ndarray, config: LDPCConfig) -> np.ndarray:
    """Encode n_cw messages in a single C++ call"""
    flat = np.ascontiguousarray(messages, dtype=np.uint8).ravel()
    if len(flat) % config.k != 0:
        msg = f"messages length {len(flat)} not a multiple of k={config.k}"
        raise ValueError(msg)
    n_cw = len(flat) // config.k
    h_s_packed, h_p_inv_cols, _k, n = _get_encoder_tables(config)
    return _ext.encode_batch(flat, h_s_packed, h_p_inv_cols, n, n_cw)


def ldpc_decode_batch(
    llrs: np.ndarray,
    config: LDPCConfig,
    max_iterations: int = 50,
    alpha: float = 0.75,
) -> np.ndarray:
    """Decode n_cw concatenated codewords in a single C++ cal."""
    flat = np.ascontiguousarray(llrs, dtype=np.float32).ravel()
    if flat.size % config.n != 0:
        msg = f"llrs length {flat.size} not a multiple of n={config.n}"
        raise ValueError(msg)
    n_cw = flat.size // config.n
    edge_var, check_order, check_bounds = _get_decode_structures(config)
    return _ext.decode_batch(
        flat, edge_var, check_order, check_bounds,
        int(config.n), int(config.k), int(n_cw),
        int(max_iterations), float(alpha),
    )


def _get_encoder_tables(config: LDPCConfig) -> tuple[np.ndarray, np.ndarray, int, int]:
    if config in _encoder_tables_cache:
        return _encoder_tables_cache[config]

    _, h_permuted = _get_encoding_structures(config)
    m, n = h_permuted.shape
    k = n - m

    h_s = (h_permuted[:, :k] & 1).astype(np.uint8)
    h_p = (h_permuted[:, k:] & 1).astype(np.uint8)
    h_p_inv = _gf2_inverse(h_p)

    result = (
        _pack_bits_rows(h_s),
        _pack_bits_rows(h_p_inv.T),
        int(k), int(n),
    )
    _encoder_tables_cache[config] = result
    return result


def _get_decode_structures(config: LDPCConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the edge-list view of H that the BP decoder consumes"""
    if config in _decode_cache:
        return _decode_cache[config]

    _, h_permuted = _get_encoding_structures(config)
    num_checks = h_permuted.shape[0]

    rows, cols = np.nonzero(h_permuted)
    check_order = np.argsort(rows).astype(np.int64)
    check_bounds = np.searchsorted(rows[check_order], np.arange(num_checks + 1)).astype(np.int64)
    edge_var = cols.astype(np.int64)

    result = (edge_var, check_order, check_bounds)
    _decode_cache[config] = result
    return result


def _get_encoding_structures(config: LDPCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Compute (G, H_permuted) from the 802.11 base matrix; cached"""
    if config in _encoding_cache:
        return _encoding_cache[config]

    h_permuted, g_transposed = _coding_matrix_systematic(_expand_h(config))
    _encoding_cache[config] = (g_transposed.T, h_permuted)
    return _encoding_cache[config]


def _expand_h(config: LDPCConfig) -> np.ndarray:
    """Expand the base matrix into the full H matrix"""
    n, z = config.n, config.z
    h_base = get_ldpc_base_matrix(config.code_rate, n)
    num_block_rows, num_block_cols = h_base.shape

    h_mat = np.zeros((num_block_rows * z, num_block_cols * z), dtype=int)
    bi, bj = np.nonzero(h_base != -1)
    shifts = h_base[bi, bj]

    idx = np.arange(z)
    rows = (bi[:, None] * z + idx[None, :]).ravel()
    cols = (bj[:, None] * z + (idx[None, :] + shifts[:, None]) % z).ravel()
    h_mat[rows, cols] = 1
    return h_mat


def _coding_matrix_systematic(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GF(2) Gaussian elimination producing systematic generator and parity-check matrices"""
    m, n = h.shape
    k = n - m
    work = np.array(h, dtype=int) % 2
    col_perm = np.arange(n)

    for r in range(m):
        target_col = k + r
        found_row, found_col = _find_pivot(work, r, target_col, n, m)
        if found_row < 0:
            break
        if found_row != r:
            work[[r, found_row]] = work[[found_row, r]]
        if found_col != target_col:
            work[:, [target_col, found_col]] = work[:, [found_col, target_col]]
            col_perm[[target_col, found_col]] = col_perm[[found_col, target_col]]
        for row in range(m):
            if row != r and work[row, target_col]:
                work[row] ^= work[r]

    g = np.hstack([np.eye(k, dtype=int), work[:, :k].T])
    return h[:, col_perm], g.T


def _find_pivot(work: np.ndarray, r: int, target_col: int, n: int, m: int) -> tuple[int, int]:
    """Find a pivot row and column for one Gaussian-elimination step"""
    for c in list(range(target_col, n)) + list(range(r, target_col)):
        for row in range(r, m):
            if work[row, c]:
                return row, c
    return -1, -1


def _gf2_inverse(M: np.ndarray) -> np.ndarray:
    """Invert an n x n binary matrix over GF(2). Raises if singular"""
    n = M.shape[0]
    if M.shape != (n, n):
        msg = f"Matrix must be square, got {M.shape}"
        raise ValueError(msg)
    A = np.hstack([np.asarray(M, dtype=np.uint8) % 2, np.eye(n, dtype=np.uint8)])
    for r in range(n):
        if not A[r, r]:
            for rr in range(r + 1, n):
                if A[rr, r]:
                    A[[r, rr]] = A[[rr, r]]
                    break
            else:
                msg = "Matrix is singular over GF(2)"
                raise ValueError(msg)
        for rr in range(n):
            if rr != r and A[rr, r]:
                A[rr] ^= A[r]
    return A[:, n:]


def _pack_bits_rows(bits: np.ndarray) -> np.ndarray:
    """Pack a (R, C) {0,1} matrix into uint64 words (LSB-first along axis 1)"""
    R, C = bits.shape
    n_words = (C + 63) // 64
    pad = n_words * 64 - C
    if pad:
        bits = np.hstack([bits, np.zeros((R, pad), dtype=bits.dtype)])
    bits = bits.astype(np.uint64, copy=False)
    bit_pos = np.arange(64, dtype=np.uint64)
    return (bits.reshape(R, n_words, 64) << bit_pos).sum(axis=2).astype(np.uint64)


# 802.11 LDPC base matrices (IEEE Std 802.11-2020, Annex F, Tables F-1..F-3).

# n=648, Z=27
_N648_R23 = np.array(
    [
        [25, 26, 14, -1, 20, -1, 2, -1, 4, -1, -1, 8, -1, 16, -1, 18, 1, 0, -1, -1, -1, -1, -1, -1],
        [10, 9, 15, 11, -1, 0, -1, 1, -1, -1, 18, -1, 8, -1, 10, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [16, 2, 20, 26, 21, -1, 6, -1, 1, 26, -1, 7, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [10, 13, 5, 0, -1, 3, -1, 7, -1, -1, 26, -1, -1, 13, -1, 16, -1, -1, -1, 0, 0, -1, -1, -1],
        [23, 14, 24, -1, 12, -1, 19, -1, 17, -1, -1, -1, 20, -1, 21, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [6, 22, 9, 20, -1, 25, -1, 17, -1, 8, -1, 14, -1, 18, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [14, 23, 21, 11, 20, -1, 24, -1, 18, -1, 19, -1, -1, -1, -1, 22, -1, -1, -1, -1, -1, -1, 0, 0],
        [17, 11, 11, 20, -1, 21, -1, 26, -1, 3, -1, -1, 18, -1, 26, -1, 1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N648_R34 = np.array(
    [
        [16, 17, 22, 24, 9, 3, 14, -1, 4, 2, 7, -1, 26, -1, 2, -1, 21, -1, 1, 0, -1, -1, -1, -1],
        [25, 12, 12, 3, 3, 26, 6, 21, -1, 15, 22, -1, 15, -1, 4, -1, -1, 16, -1, 0, 0, -1, -1, -1],
        [25, 18, 26, 16, 22, 23, 9, -1, 0, -1, 4, -1, 4, -1, 8, 23, 11, -1, -1, -1, 0, 0, -1, -1],
        [9, 7, 0, 1, 17, -1, -1, 7, 3, -1, 3, 23, -1, 16, -1, -1, 21, -1, 0, -1, -1, 0, 0, -1],
        [24, 5, 26, 7, 1, -1, -1, 15, 24, 15, -1, 8, -1, 13, -1, 13, -1, 11, -1, -1, -1, -1, 0, 0],
        [2, 2, 19, 14, 24, 1, 15, 19, -1, 21, -1, 2, -1, 24, -1, 3, -1, 2, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N648_R56 = np.array(
    [
        [17, 13, 8, 21, 9, 3, 18, 12, 10, 0, 4, 15, 19, 2, 5, 10, 26, 19, 13, 13, 1, 0, -1, -1],
        [3, 12, 11, 14, 11, 25, 5, 18, 0, 9, 2, 26, 26, 10, 24, 7, 14, 20, 4, 2, -1, 0, 0, -1],
        [22, 16, 4, 3, 10, 21, 12, 5, 21, 14, 19, 5, -1, 8, 5, 18, 11, 5, 5, 15, 0, -1, 0, 0],
        [7, 7, 14, 14, 4, 16, 16, 24, 24, 10, 1, 7, 15, 6, 10, 26, 8, 18, 21, 14, 1, -1, -1, 0],
    ],
    dtype=int,
)

# n=1296, Z=54
_N1296_R23 = np.array(
    [
        [39, 31, 22, 43, -1, 40, 4, -1, 11, -1, -1, 50, -1, -1, -1, 6, 1, 0, -1, -1, -1, -1, -1, -1],
        [25, 52, 41, 2, 6, -1, 14, -1, 34, -1, -1, -1, 24, -1, 37, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [43, 31, 29, 0, 21, -1, 28, -1, -1, 2, -1, -1, 7, -1, 17, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [20, 33, 48, -1, 4, 13, -1, 26, -1, -1, 22, -1, -1, 46, 42, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [45, 7, 18, 51, 12, 25, -1, -1, -1, 50, -1, 5, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [35, 40, 32, 16, 5, -1, -1, 18, -1, -1, 43, 51, -1, 32, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [9, 24, 13, 22, 28, -1, -1, 37, -1, -1, 25, -1, -1, 52, -1, 13, -1, -1, -1, -1, -1, -1, 0, 0],
        [32, 22, 4, 21, 16, -1, -1, -1, 27, 28, -1, -1, 38, -1, -1, -1, 8, 1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1296_R34 = np.array(
    [
        [39, 40, 51, 41, 3, 29, 8, 36, -1, 14, -1, 6, -1, 33, -1, 11, -1, 4, 1, 0, -1, -1, -1, -1],
        [48, 21, 47, 9, 48, 35, 51, -1, 38, -1, 28, -1, 34, -1, 50, -1, 50, -1, -1, 0, 0, -1, -1, -1],
        [30, 39, 28, 42, 50, 39, 5, 17, -1, 6, -1, 18, -1, 20, -1, 15, -1, 40, -1, -1, 0, 0, -1, -1],
        [29, 0, 1, 43, 36, 30, 47, -1, 49, -1, 47, -1, 3, -1, 35, -1, 34, -1, 0, -1, -1, 0, 0, -1],
        [1, 32, 11, 23, 10, 44, 12, 7, -1, 48, -1, 4, -1, 9, -1, 17, -1, 16, -1, -1, -1, -1, 0, 0],
        [13, 7, 15, 47, 23, 16, 47, -1, 43, -1, 29, -1, 52, -1, 2, -1, 53, -1, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1296_R56 = np.array(
    [
        [48, 29, 37, 52, 2, 16, 6, 14, 53, 31, 34, 5, 18, 42, 53, 31, 45, -1, 46, 52, 1, 0, -1, -1],
        [17, 4, 30, 7, 43, 11, 24, 6, 14, 21, 6, 39, 17, 40, 47, 7, 15, 41, 19, -1, -1, 0, 0, -1],
        [7, 2, 51, 31, 46, 23, 16, 11, 53, 40, 10, 7, 46, 53, 33, 35, -1, 25, 35, 38, 0, -1, 0, 0],
        [19, 48, 41, 1, 10, 7, 36, 47, 5, 29, 52, 52, 31, 10, 26, 6, 3, 2, -1, 51, 1, -1, -1, 0],
    ],
    dtype=int,
)

# n=1944, Z=81
_N1944_R23 = np.array(
    [
        [61, 75, 4, 63, 56, -1, -1, -1, -1, -1, -1, 8, -1, 2, 17, 25, 1, 0, -1, -1, -1, -1, -1, -1],
        [56, 74, 77, 20, -1, -1, -1, 64, 24, 4, 67, -1, 7, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [28, 21, 68, 10, 7, 14, 65, -1, -1, -1, 23, -1, -1, -1, 75, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [48, 38, 43, 78, 76, -1, -1, -1, -1, 5, 36, -1, 15, 72, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [40, 2, 53, 25, -1, 52, 62, -1, 20, -1, -1, 44, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1],
        [69, 23, 64, 10, 22, -1, 21, -1, -1, -1, -1, -1, 68, 23, 29, -1, -1, -1, -1, -1, -1, 0, 0, -1],
        [12, 0, 68, 20, 55, 61, -1, 40, -1, -1, -1, 52, -1, -1, -1, 44, -1, -1, -1, -1, -1, -1, 0, 0],
        [58, 8, 34, 64, 78, -1, -1, 11, 78, 24, -1, -1, -1, -1, -1, 58, 1, -1, -1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1944_R34 = np.array(
    [
        [48, 29, 28, 39, 9, 61, -1, -1, -1, 63, 45, 80, -1, -1, -1, 37, 32, 22, 1, 0, -1, -1, -1, -1],
        [4, 49, 42, 48, 11, 30, -1, -1, -1, 49, 17, 41, 37, 15, -1, 54, -1, -1, -1, 0, 0, -1, -1, -1],
        [35, 76, 78, 51, 37, 35, 21, -1, 17, 64, -1, -1, -1, 59, 7, -1, -1, 32, -1, -1, 0, 0, -1, -1],
        [9, 65, 44, 9, 54, 56, 73, 34, 42, -1, -1, -1, 35, -1, -1, -1, 46, 39, 0, -1, -1, 0, 0, -1],
        [3, 62, 7, 80, 68, 26, -1, 80, 55, -1, 36, -1, 26, -1, 9, -1, 72, -1, -1, -1, -1, -1, 0, 0],
        [26, 75, 33, 21, 69, 59, 3, 38, -1, -1, -1, 35, -1, 62, 36, 26, -1, -1, 1, -1, -1, -1, -1, 0],
    ],
    dtype=int,
)

_N1944_R56 = np.array(
    [
        [13, 48, 80, 66, 4, 74, 7, 30, 76, 52, 37, 60, -1, 49, 73, 31, 74, 73, 23, -1, 1, 0, -1, -1],
        [69, 63, 74, 56, 64, 77, 57, 65, 6, 16, 51, -1, 64, -1, 68, 9, 48, 62, 54, 27, -1, 0, 0, -1],
        [51, 15, 0, 80, 24, 25, 42, 54, 44, 71, 71, 9, 67, 35, -1, 58, -1, 29, -1, 53, 0, -1, 0, 0],
        [16, 29, 36, 41, 44, 56, 59, 37, 50, 24, -1, 65, 4, 65, 52, -1, 4, -1, 73, 52, 1, -1, -1, 0],
    ],
    dtype=int,
)


def get_ldpc_base_matrix(code_rate: CodeRates, n: int) -> np.ndarray:
    """Return the 802.11 base matrix for the given code rate and block length."""
    matrices = {
        648: {
            CodeRates.TWO_THIRDS_RATE: _N648_R23,
            CodeRates.THREE_QUARTER_RATE: _N648_R34,
            CodeRates.FIVE_SIXTH_RATE: _N648_R56,
        },
        1296: {
            CodeRates.TWO_THIRDS_RATE: _N1296_R23,
            CodeRates.THREE_QUARTER_RATE: _N1296_R34,
            CodeRates.FIVE_SIXTH_RATE: _N1296_R56,
        },
        1944: {
            CodeRates.TWO_THIRDS_RATE: _N1944_R23,
            CodeRates.THREE_QUARTER_RATE: _N1944_R34,
            CodeRates.FIVE_SIXTH_RATE: _N1944_R56,
        },
    }
    if n not in matrices:
        msg = f"Unsupported block length n={n}. Must be 648, 1296, or 1944."
        raise ValueError(msg)
    if code_rate not in matrices[n]:
        msg = f"Unsupported code rate {code_rate} for n={n}."
        raise ValueError(msg)
    return matrices[n][code_rate]
