"""Channel coding enums and helpers."""

from enum import Enum


class CodeRates(Enum):
    """Supported channel coding rates."""

    HALF_RATE = 1
    QUATER_RATE = 2


class LDPC:
    """Placeholder LDPC codec implementation."""

    def __init__(self) -> None:
        """Initialize the LDPC codec placeholder."""
