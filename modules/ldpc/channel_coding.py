from enum import Enum

class CodeRates(Enum):
    """Supported channel coding rates."""
    NONE = 0
    TWO_THIRDS_RATE = 1
    THREE_QUARTER_RATE = 2
    FIVE_SIXTH_RATE = 3

    @property
    def rate_fraction(self) -> tuple[int, int]:
        fractions = {
            CodeRates.NONE: (1, 1),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }
        return fractions[self]

    @property
    def value_float(self) -> float:
        """Numeric code rate value (k/n ratio)."""
        num, denom = self.rate_fraction
        return num / denom
