from enum import Enum

class CodeRates(Enum):
    """Supported channel coding rates.

    Values fit in 3 bits so they can be serialised directly into the frame
    header `coding_rate` field.
    """
    NONE = 0
    HALF_RATE = 1
    TWO_THIRDS_RATE = 2
    THREE_QUARTER_RATE = 3
    FIVE_SIXTH_RATE = 4

    @property
    def rate_fraction(self) -> tuple[int, int]:
        fractions = {
            CodeRates.NONE: (1, 1),
            CodeRates.HALF_RATE: (1, 2),
            CodeRates.TWO_THIRDS_RATE: (2, 3),
            CodeRates.THREE_QUARTER_RATE: (3, 4),
            CodeRates.FIVE_SIXTH_RATE: (5, 6),
        }
        return fractions[self]

    @property
    def value_float(self) -> float:
        num, denom = self.rate_fraction
        return num / denom
