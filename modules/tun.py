"""Linux TUN device interface.

Opens /dev/net/tun and configures a TUN (Layer-3 IP tunnel) interface.
The calling process must have CAP_NET_ADMIN or be root.

Usage
-----
    with TunDevice("tun0") as tun:
        arq_node = ARQNode(tun, pluto_tx, pluto_rx, config)
        arq_node.start()
        ...
"""

import fcntl
import os
import select
import struct
import numpy as np

# ioctl constants for Linux TUN/TAP
_IFF_TUN   = 0x0001   # TUN device (no Ethernet header)
_IFF_NO_PI = 0x1000   # don't prepend packet info header
_TUNSETIFF = 0x400454CA


class TunDevice:
    """Wrapper around a Linux TUN character device.

    ``read()`` uses select with a short timeout so the ARQ TUN reader thread
    can respond to stop signals without blocking indefinitely.

    Parameters
    ----------
    name:
        Interface name, e.g. ``"tun0"``.
    mtu:
        Maximum IP packet size in bytes (default 1500).
    poll_timeout:
        Seconds select() will wait before returning None from read().
    """

    def __init__(
        self,
        name: str = "tun0",
        mtu: int = 1500,
        poll_timeout: float = 0.05,
    ) -> None:
        self.name = name
        self.mtu = mtu
        self._poll_timeout = poll_timeout

        self._fd = os.open("/dev/net/tun", os.O_RDWR)
        # struct ifreq: 16-byte name + 2-byte flags + 14-byte padding
        ifr = struct.pack("16sH14s", name.encode(), _IFF_TUN | _IFF_NO_PI, b"\x00" * 14)
        fcntl.ioctl(self._fd, _TUNSETIFF, ifr)

    def read(self) -> bytes | None:
        """Read one IP packet from the TUN interface.

        Returns ``None`` if no packet arrives within ``poll_timeout`` seconds.
        """
        ready, _, _ = select.select([self._fd], [], [], self._poll_timeout)
        if not ready:
            return None
        return os.read(self._fd, self.mtu)

    def write(self, data: bytes) -> None:
        """Inject one IP packet into the OS network stack."""
        os.write(self._fd, data)

    def close(self) -> None:
        os.close(self._fd)

    def __enter__(self) -> "TunDevice":
        return self

    def __exit__(self, *_) -> None:
        self.close()
