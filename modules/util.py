"""Utility functions for radio communication simulation."""

import numpy as np
from numpy.typing import NDArray
from typing import Any # Added for np.ndarray type hints


def ebn0_to_snr(ebn0_db: np.float32, code_rate: np.float32, bits_per_symbol: int = 2) -> np.float32: # Changed type hints
    """Convert Eb/N0 (dB) to SNR per symbol (Es/N0) in dB.

    The relationship is:
        Es/N0 = Eb/N0 * code_rate * bits_per_symbol

    In dB:
        SNR (dB) = Eb/N0 (dB) + 10*log10(code_rate * bits_per_symbol)

    Source: https://en.wikipedia.org/wiki/Eb/N0#Relation_to_Es/N0

    """
    return ebn0_db + np.float32(10) * np.log10(code_rate * bits_per_symbol) # Explicitly cast to np.float32


def snr_to_ebn0(snr_db: np.float32, code_rate: np.float32, bits_per_symbol: int = 2) -> np.float32: # Changed type hints
    """Convert SNR per symbol (Es/N0) in dB to Eb/N0 (dB).

    The relationship is:
        Eb/N0 = Es/N0 / (code_rate * bits_per_symbol)

    In dB:
        Eb/N0 (dB) = SNR (dB) - 10*log10(code_rate * bits_per_symbol)

    """
    return snr_db - np.float32(10) * np.log10(code_rate * bits_per_symbol) # Explicitly cast to np.float32


def bytes_to_bits(data: bytes) -> NDArray[np.uint8]: # Added return type hint
    """Convert raw bytes to a bit array."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: NDArray[np.uint8]) -> bytes: # Changed type hint
    """Convert a bit array back to raw bytes."""
    remainder = len(bits) % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)]) # Changed to np.uint8
    return np.packbits(bits.astype(np.uint8)).tobytes()


def text_to_bits(text: str) -> NDArray[np.uint8]: # Added return type hint
    """Convert a UTF-8 string to a bit array."""
    return bytes_to_bits(text.encode("utf-8"))


def bits_to_text(bits: NDArray[np.uint8]) -> str: # Added return type hint
    """Convert a bit array back to a UTF-8 string."""
    return bits_to_bytes(bits).decode("utf-8", errors="replace")


def block_agc(
    symbols: NDArray[np.complex64], # Changed type hint
    block_size: int = 64,
    target_power: np.float32 = np.float32(1.0), # Changed type hint and default
) -> NDArray[np.complex64]: # Changed return type hint
    """Block-wise automatic gain control.

    Divides the symbol stream into blocks and normalizes each block
    to the target average power.
    """
    out = symbols.copy()
    for start in range(0, len(symbols), block_size):
        end = min(start + block_size, len(symbols))
        block = out[start:end]
        power = np.mean(np.abs(block) ** np.float32(2)) # Explicitly cast to np.float32
        if power > np.float32(0): # Explicitly cast to np.float32
            out[start:end] = block * np.sqrt(target_power / power)
    return out


def calculate_reference_power(reference_signal: NDArray[np.complex64]) -> np.float32: # Changed type hint
    """Calculate reference power from a representative signal.

    Use this to determine the transmit power of your signal chain
    (after modulation, upsampling, pulse shaping, etc.) for accurate
    SNR configuration in the channel model.

    """
    return np.float32(np.mean(np.abs(reference_signal) ** np.float32(2))) # Explicitly cast to np.float32