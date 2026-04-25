from dataclasses import dataclass
from enum import Enum

class CodeRates(Enum):
    """Supported channel coding rates.

    Enum values are chosen so each fits in 2 bits, allowing the rate to be
    serialised in the frame header `coding_rate` field directly.
    """

    NONE = 0
    TWO_THIRDS_RATE = 1
    THREE_QUARTER_RATE = 2
    FIVE_SIXTH_RATE = 3

    @property
    def rate_fraction(self) -> tuple[int, int]:
        """Return the (numerator, denominator) tuple for this code rate."""
        fractions = {
            CodeRates.NONE: (1, 1),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }
        return fractions[self]

    @property
    def value_float(self) -> float:
        """Return the numeric code rate value (k/n ratio).

        Returns:
            The code rate as a float (e.g., 0.5 for HALF_RATE, 0.833... for FIVE_SIXTH_RATE).

        """
        num, denom = self.rate_fraction
        return num / denom
