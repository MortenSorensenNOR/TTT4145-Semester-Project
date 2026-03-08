"""Channel coding: public API re-exporting Golay and LDPC implementations."""

from modules.golay import Golay
from modules.ldpc import (
    CodeRates,
    LDPCConfig,
    _decode_cache,
    _encoding_cache,
    _h_cache,
    deinterleave,
    get_ldpc_base_matrix,
    interleave,
    ldpc_clear_cache,
    ldpc_decode,
    ldpc_encode,
    ldpc_get_h_matrix,
    ldpc_get_supported_payload_lengths,
)

__all__ = [
    "CodeRates",
    "Golay",
    "LDPCConfig",
    "_decode_cache",
    "_encoding_cache",
    "_h_cache",
    "deinterleave",
    "get_ldpc_base_matrix",
    "interleave",
    "ldpc_clear_cache",
    "ldpc_decode",
    "ldpc_encode",
    "ldpc_get_h_matrix",
    "ldpc_get_supported_payload_lengths",
]
