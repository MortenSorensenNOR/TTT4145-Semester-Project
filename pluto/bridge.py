"""TUN/TAP IP bridge over PlutoSDR.

Creates a virtual network interface (TUN device) and bridges IP packets
over a PlutoSDR radio link.  Two instances (one per Pluto) give full-duplex
IP connectivity using frequency-division duplexing (FDD).

Usage:
    # Machine A (10.0.0.1)
    sudo python -m pluto.bridge --node A

    # Machine B (10.0.0.2)
    sudo python -m pluto.bridge --node B

    # Then from machine A:
    ping 10.0.0.2
"""

import argparse
import fcntl
import logging
import os
import socket
import struct
import threading
from dataclasses import dataclass

import adi
import numpy as np

from modules.channel_coding import LDPCConfig, ldpc_decode, ldpc_encode, ldpc_get_supported_payload_lengths
from modules.frame_constructor import FrameConstructor
from modules.util import bytes_to_bits
from pluto import create_pluto
from pluto.config import (
    CODING_RATE,
    DAC_SCALE,
    DEFAULT_TX_GAIN,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    MOD_SCHEME,
    PIPELINE,
    configure_rx,
    configure_tx,
)
from pluto.decode import FrameResult, create_decoder
from pluto.receive import run_receiver
from pluto.transmit import build_tx_signal_from_bits, max_payload_bits

logger = logging.getLogger(__name__)

# ── Linux ioctl constants ─────────────────────────────────────────────────
# From linux/if_tun.h
TUNSETIFF = 0x400454CA
IFF_TUN = 0x0001
IFF_NO_PI = 0x1000

# From linux/sockios.h
SIOCGIFFLAGS = 0x8913
SIOCSIFFLAGS = 0x8914
SIOCSIFADDR = 0x8916
SIOCSIFNETMASK = 0x891C
SIOCSIFMTU = 0x8922

# From linux/if.h
IFF_UP = 0x1
IFF_RUNNING = 0x40

# Minimum valid IP packet size (IPv4 header without options)
MIN_IP_PACKET_SIZE = 20


@dataclass
class NodeConfig:
    """Per-node radio and IP configuration."""

    tx_freq: int
    rx_freq: int
    ip_addr: str
    peer_addr: str


NODE_CONFIGS = {
    "A": NodeConfig(tx_freq=FREQ_A_TO_B, rx_freq=FREQ_B_TO_A, ip_addr="10.0.0.1", peer_addr="10.0.0.2"),
    "B": NodeConfig(tx_freq=FREQ_B_TO_A, rx_freq=FREQ_A_TO_B, ip_addr="10.0.0.2", peer_addr="10.0.0.1"),
}


def open_tun(name: str = "pluto0") -> int:
    """Open a TUN device and return its file descriptor."""
    fd = os.open("/dev/net/tun", os.O_RDWR)
    ifr = struct.pack("16sH", name.encode(), IFF_TUN | IFF_NO_PI)
    fcntl.ioctl(fd, TUNSETIFF, ifr)
    return fd


def _sockaddr_in(ip_addr: str) -> bytes:
    """Pack an IPv4 address into a sockaddr_in struct (16 bytes)."""
    return struct.pack("HH4s8s", socket.AF_INET, 0, socket.inet_aton(ip_addr), b"\x00" * 8)


def configure_tun(name: str, ip_addr: str, peer_addr: str, mtu: int) -> None:
    """Configure the TUN device with IP address and MTU via ioctl."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    name_b = name.encode().ljust(16, b"\x00")[:16]

    # Set MTU
    fcntl.ioctl(sock, SIOCSIFMTU, name_b + struct.pack("I", mtu))

    # Set IP address
    fcntl.ioctl(sock, SIOCSIFADDR, name_b + _sockaddr_in(ip_addr))

    # Set netmask (/24)
    fcntl.ioctl(sock, SIOCSIFNETMASK, name_b + _sockaddr_in("255.255.255.0"))

    # Bring interface up
    ifr_flags = fcntl.ioctl(sock, SIOCGIFFLAGS, name_b + b"\x00" * 16)
    flags = struct.unpack_from("H", ifr_flags, 16)[0]
    fcntl.ioctl(sock, SIOCSIFFLAGS, name_b + struct.pack("H", flags | IFF_UP | IFF_RUNNING))

    sock.close()
    logger.info("TUN %s: %s/24 (peer %s, MTU %d)", name, ip_addr, peer_addr, mtu)


def _warm_tx_cache(frame_constructor: FrameConstructor) -> None:
    """Pre-warm LDPC encode cache for all supported payload sizes."""
    logger.info("TX: warming LDPC cache...")
    for k in ldpc_get_supported_payload_lengths(CODING_RATE):
        dummy_bits = np.zeros(int(k) - 16, dtype=np.uint8)  # -16 for CRC
        build_tx_signal_from_bits(dummy_bits, frame_constructor, MOD_SCHEME, CODING_RATE)
    logger.info("TX: LDPC cache ready")


def _warm_rx_cache() -> None:
    """Pre-warm LDPC decode cache for all supported payload sizes."""
    logger.info("RX: warming LDPC cache...")
    for k in ldpc_get_supported_payload_lengths(CODING_RATE):
        config = LDPCConfig(k=int(k), code_rate=CODING_RATE)
        dummy_msg = np.zeros(int(k), dtype=int)
        codeword = ldpc_encode(dummy_msg, config)
        llr = (1 - 2 * codeword).astype(float) * 5.0
        ldpc_decode(llr, config, max_iterations=1)
    logger.info("RX: LDPC cache ready")


def tx_thread(tun_fd: int, sdr: adi.Pluto, mtu: int, tx_buffer_len: int) -> None:
    """Read IP packets from TUN and transmit over PlutoSDR."""
    frame_constructor = FrameConstructor()
    max_bits = max_payload_bits(CODING_RATE)
    logger.info("TX thread started (max payload %d bytes)", max_bits // 8)

    _warm_tx_cache(frame_constructor)

    while True:
        try:
            packet = os.read(tun_fd, mtu)
        except OSError:
            break

        if not packet:
            continue

        payload_bits = bytes_to_bits(packet)
        if len(payload_bits) > max_bits:
            logger.warning("TX: dropping oversized packet (%d bytes > %d max)", len(packet), max_bits // 8)
            continue

        try:
            tx_signal = build_tx_signal_from_bits(payload_bits, frame_constructor, MOD_SCHEME, CODING_RATE)
            # Pad to fixed buffer length
            samples = np.zeros(tx_buffer_len, dtype=complex)
            samples[: len(tx_signal)] = tx_signal * DAC_SCALE
            sdr.tx(samples)  # type: ignore[union-attr]
            logger.info("TX: %d bytes", len(packet))
        except Exception:
            logger.exception("TX: failed to transmit packet")


def rx_thread_bridge(tun_fd: int, sdr: adi.Pluto) -> None:
    """Receive frames from PlutoSDR and write decoded IP packets to TUN."""
    decoder = create_decoder(PIPELINE)
    _warm_rx_cache()

    def on_frame(result: FrameResult) -> None:
        data = result.payload_bytes
        # Validate minimum IP packet: 20+ bytes, version nibble is 4 (IPv4) or 6 (IPv6)
        if len(data) < MIN_IP_PACKET_SIZE:
            logger.debug("RX: dropping short frame (%d bytes)", len(data))
            return
        version = (data[0] >> 4) & 0xF
        if version not in (4, 6):
            logger.debug("RX: dropping non-IP frame (version=%d)", version)
            return
        try:
            os.write(tun_fd, data)
            logger.info("RX: %d bytes (CFO=%+.0f Hz)", len(data), result.cfo_hz)
        except OSError:
            logger.exception("RX: TUN write failed")

    logger.info("RX thread started")
    run_receiver(sdr, decoder, on_frame)  # type: ignore[arg-type]


def main() -> None:
    """Entry point for the TUN/TAP bridge."""
    parser = argparse.ArgumentParser(description="PlutoSDR TUN/TAP IP bridge")
    parser.add_argument("--node", choices=["A", "B"], required=True, help="Node identity (A or B)")
    parser.add_argument("--tun", default="pluto0", help="TUN device name (default: pluto0)")
    parser.add_argument("--tx-gain", type=float, default=DEFAULT_TX_GAIN, help="TX gain in dB (default: %(default)s)")
    parser.add_argument(
        "--rx-cfo-offset", type=int, default=0, help="RX CFO offset in Hz (use test_measure_cfo.py to measure)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    node = NODE_CONFIGS[args.node]
    max_bytes = max_payload_bits(CODING_RATE) // 8
    tun_mtu = max_bytes

    # ── TUN setup ─────────────────────────────────────────────────────
    tun_fd = open_tun(args.tun)
    configure_tun(args.tun, node.ip_addr, node.peer_addr, tun_mtu)

    # ── SDR setup ─────────────────────────────────────────────────────
    sdr = create_pluto()
    configure_tx(sdr, freq=node.tx_freq, gain=args.tx_gain)

    # Compute TX buffer size from max-size frame (PlutoSDR requires constant buffer size)
    frame_constructor = FrameConstructor()
    max_bits = max_payload_bits(CODING_RATE)
    max_signal = build_tx_signal_from_bits(np.zeros(max_bits, dtype=np.uint8), frame_constructor, MOD_SCHEME, CODING_RATE)
    tx_buffer_len = len(max_signal)
    sdr.tx_buffer_size = tx_buffer_len

    rx_freq = node.rx_freq + args.rx_cfo_offset
    configure_rx(sdr, freq=rx_freq)

    sdr.rx()  # flush stale DMA buffer

    logger.info(
        "Node %s: TX %.0f MHz (%.0f dB) -> RX %.0f Hz (CFO offset %+d Hz)",
        args.node,
        node.tx_freq / 1e6,
        args.tx_gain,
        rx_freq,
        args.rx_cfo_offset,
    )

    # ── Launch TX and RX threads ──────────────────────────────────────
    # TX and RX use separate DMA channels in libiio, safe to use from different threads.
    t_tx = threading.Thread(target=tx_thread, args=(tun_fd, sdr, tun_mtu, tx_buffer_len), daemon=True, name="tx")
    t_rx = threading.Thread(target=rx_thread_bridge, args=(tun_fd, sdr), daemon=True, name="rx")
    t_tx.start()
    t_rx.start()

    try:
        t_tx.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        os.close(tun_fd)
        sdr.tx_destroy_buffer()


if __name__ == "__main__":
    main()
