"""PlutoSDR transmit and receive scripts."""

import adi


def create_pluto() -> object:
    """Create PlutoSDR instance."""
    return adi.Pluto("ip:192.168.2.1")  # type: ignore[abstract]
