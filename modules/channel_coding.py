"""Channel coding enums and helpers."""

from enum import Enum


class CodeRates(Enum):
    """Supported channel coding rates."""

    HALF_RATE = 1
    QUATER_RATE = 2
