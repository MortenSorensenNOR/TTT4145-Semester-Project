def round_up(n: int, m: int = 8) -> int:
      return ((n + m - 1) // m) * m

def int_to_bits(n: int, length: int) -> list[int]:
    """Convert an integer to a fixed-width big-endian bit list."""
    return [(n >> (length - 1 - i)) & 1 for i in range(length)]
