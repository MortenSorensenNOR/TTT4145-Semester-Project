"""PlutoSDR transmit and receive scripts."""

from typing import Protocol, runtime_checkable

import adi
import numpy as np


@runtime_checkable
class SDRReceiver(Protocol):
    """Protocol for an SDR that can receive samples."""

    def rx(self) -> np.ndarray:
        """Receive a buffer of complex samples."""
        ...


@runtime_checkable
class SDRTransmitter(Protocol):
    """Protocol for an SDR that can transmit samples."""

    def tx(self, data: np.ndarray) -> None:
        """Transmit a buffer of complex samples."""
        ...

    def tx_destroy_buffer(self) -> None:
        """Release the transmit buffer."""
        ...


def create_pluto(uri: str = "ip:192.168.2.1") -> adi.Pluto:
    """Create PlutoSDR instance."""
    return adi.Pluto(uri)  # type: ignore[abstract]
