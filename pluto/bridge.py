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

import numpy as np

from modules.channel_coding import CodeRates
from modules.frame_constructor import FrameConstructor, ModulationSchemes
from modules.pulse_shaping import rrc_filter
from modules.synchronization import Synchronizer
from modules.util import bytes_to_bits
from pluto import create_pluto
from pluto.config import (
    DAC_SCALE,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    PILOT_CONFIG,
    RRC_ALPHA,
    RRC_NUM_TAPS,
    SAMPLE_RATE,
    SPS,
    SYNC_CONFIG,
)
from pluto.receive import FrameDecoder
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

MOD_SCHEME = ModulationSchemes.QPSK
CODING_RATE = CodeRates.HALF_RATE
DEFAULT_TX_GAIN = -10
RX_BUFFER_SIZE = 2**16


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


def tx_thread(tun_fd: int, sdr: object, mtu: int) -> None:
    """Read IP packets from TUN and transmit over PlutoSDR."""
    max_bits = max_payload_bits(CODING_RATE)
    logger.info("TX thread started (max payload %d bytes)", max_bits // 8)

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
            tx_signal = build_tx_signal_from_bits(payload_bits, MOD_SCHEME, CODING_RATE)
            samples = tx_signal * DAC_SCALE
            sdr.tx(samples)  # type: ignore[union-attr]
            logger.debug("TX: %d bytes", len(packet))
        except Exception:
            logger.exception("TX: failed to transmit packet")


def rx_thread_bridge(tun_fd: int, sdr: object) -> None:
    """Receive frames from PlutoSDR and write decoded IP packets to TUN."""
    h_rrc = rrc_filter(SPS, RRC_ALPHA, RRC_NUM_TAPS)
    sync = Synchronizer(SYNC_CONFIG, sps=SPS, rrc_taps=h_rrc)
    fc = FrameConstructor()
    decoder = FrameDecoder(sync, fc, sample_rate=SAMPLE_RATE, sps=SPS, pilot_config=PILOT_CONFIG)

    max_frame_samples = decoder.max_frame_samples

    # Overlap-save matched filter state
    filter_overlap = len(h_rrc) - 1
    filter_state = np.zeros(filter_overlap, dtype=complex)

    # Receive buffer
    buf_capacity = max_frame_samples + RX_BUFFER_SIZE
    rx_buf = np.zeros(buf_capacity, dtype=complex)
    buf_len = 0
    abs_offset = 0

    logger.info("RX thread started")

    while True:
        try:
            rx = sdr.rx()  # type: ignore[union-attr]
        except Exception:
            logger.exception("RX: SDR read failed")
            break

        # Incremental matched filter
        chunk = np.concatenate([filter_state, rx])
        new_filtered = np.convolve(chunk, h_rrc, mode="valid")
        filter_state = rx[-filter_overlap:]

        n_new = len(new_filtered)
        if buf_len + n_new > buf_capacity:
            keep = max_frame_samples
            rx_buf[:keep] = rx_buf[buf_len - keep : buf_len]
            abs_offset += buf_len - keep
            buf_len = keep
        rx_buf[buf_len : buf_len + n_new] = new_filtered
        buf_len += n_new

        # Multi-frame detection
        while True:
            result = decoder.try_decode(rx_buf[:buf_len], abs_offset)
            if result is None:
                break

            packet = result.payload_bytes
            try:
                os.write(tun_fd, packet)
                logger.debug("RX: %d bytes (CFO=%+.0f Hz)", len(packet), result.cfo_hz)
            except OSError:
                logger.exception("RX: TUN write failed")

            consumed = result.consumed_samples
            remaining = buf_len - consumed
            rx_buf[:remaining] = rx_buf[consumed:buf_len]
            buf_len = remaining
            abs_offset += consumed

        if buf_len > max_frame_samples:
            trim = buf_len - max_frame_samples
            rx_buf[:max_frame_samples] = rx_buf[trim:buf_len]
            buf_len = max_frame_samples
            abs_offset += trim


def main() -> None:
    """Entry point for the TUN/TAP bridge."""
    parser = argparse.ArgumentParser(description="PlutoSDR TUN/TAP IP bridge")
    parser.add_argument("--node", choices=["A", "B"], required=True, help="Node identity (A or B)")
    parser.add_argument("--tun", default="pluto0", help="TUN device name (default: pluto0)")
    parser.add_argument("--tx-gain", type=float, default=DEFAULT_TX_GAIN, help="TX gain in dB (default: %(default)s)")
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
    sdr.sample_rate = int(SAMPLE_RATE)

    # TX config
    sdr.tx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.tx_lo = int(node.tx_freq)
    sdr.tx_hardwaregain_chan0 = args.tx_gain

    # RX config
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_lo = int(node.rx_freq)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = RX_BUFFER_SIZE

    sdr.rx()  # flush stale DMA buffer

    logger.info(
        "Node %s: TX %.0f MHz (%.0f dB) -> RX %.0f MHz",
        args.node,
        node.tx_freq / 1e6,
        args.tx_gain,
        node.rx_freq / 1e6,
    )

    # ── Launch TX and RX threads ──────────────────────────────────────
    t_tx = threading.Thread(target=tx_thread, args=(tun_fd, sdr, tun_mtu), daemon=True, name="tx")
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
