"""TUN/TAP IP bridge over PlutoSDR.

Creates a virtual network interface (TUN device) and bridges IP packets
over a PlutoSDR radio link using frequency-division duplexing (FDD).

Two instances running on separate machines (each with its own Pluto) give
full-duplex IP connectivity: Node A transmits on FREQ_A_TO_B and receives on
FREQ_B_TO_A; Node B does the opposite.

Usage:
    # Machine A (10.0.0.1) — Pluto at 192.168.2.1
    sudo .venv/bin/python -m pluto.bridge --node A --pluto-ip 192.168.2.1

    # Machine B (10.0.0.2) — Pluto at 192.168.3.1
    sudo .venv/bin/python -m pluto.bridge --node B --pluto-ip 192.168.3.1

    # Then from machine A:
    ping 10.0.0.2
    iperf3 -s          # on B
    iperf3 -c 10.0.0.2 # on A
"""

import argparse
import fcntl
import logging
import os
import queue
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass

import adi
import numpy as np

from modules.pipeline import TXPipeline, RXPipeline, PipelineConfig, Packet
from pluto.config import (
    DAC_SCALE,
    DEFAULT_TX_GAIN,
    MAX_PACKET_SIZE_BYTES,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    PIPELINE,
    configure_rx,
    configure_tx,
)
from pluto.sdr_stream import RxStream

logger = logging.getLogger(__name__)

# ── Linux ioctl constants ─────────────────────────────────────────────────
# From linux/if_tun.h
TUNSETIFF = 0x400454CA
IFF_TUN   = 0x0001
IFF_NO_PI = 0x1000

# From linux/sockios.h
SIOCGIFFLAGS  = 0x8913
SIOCSIFFLAGS  = 0x8914
SIOCSIFADDR   = 0x8916
SIOCSIFNETMASK = 0x891C
SIOCSIFMTU    = 0x8922

# From linux/if.h
IFF_UP      = 0x1
IFF_RUNNING = 0x40

# Minimum valid IPv4 header size
MIN_IP_PACKET_SIZE = 20


@dataclass
class NodeConfig:
    """Per-node radio and IP configuration."""
    tx_freq:   int
    rx_freq:   int
    ip_addr:   str
    peer_addr: str


NODE_CONFIGS = {
    "A": NodeConfig(tx_freq=FREQ_A_TO_B, rx_freq=FREQ_B_TO_A, ip_addr="10.0.0.1", peer_addr="10.0.0.2"),
    "B": NodeConfig(tx_freq=FREQ_B_TO_A, rx_freq=FREQ_A_TO_B, ip_addr="10.0.0.2", peer_addr="10.0.0.1"),
}


# ── TUN helpers ───────────────────────────────────────────────────────────

def open_tun(name: str = "pluto0") -> int:
    """Open a TUN device and return its file descriptor."""
    fd = os.open("/dev/net/tun", os.O_RDWR)
    ifr = struct.pack("16sH14s", name.encode(), IFF_TUN | IFF_NO_PI, b"\x00" * 14)
    fcntl.ioctl(fd, TUNSETIFF, ifr)
    return fd


def _sockaddr_in(ip_addr: str) -> bytes:
    """Pack an IPv4 address into a sockaddr_in struct (16 bytes)."""
    return struct.pack("HH4s8s", socket.AF_INET, 0, socket.inet_aton(ip_addr), b"\x00" * 8)


def configure_tun(name: str, ip_addr: str, peer_addr: str, mtu: int) -> None:
    """Configure the TUN device with IP address, netmask, and MTU via ioctl."""
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    name_b = name.encode().ljust(16, b"\x00")[:16]

    fcntl.ioctl(sock, SIOCSIFMTU,     name_b + struct.pack("I", mtu))
    fcntl.ioctl(sock, SIOCSIFADDR,    name_b + _sockaddr_in(ip_addr))
    fcntl.ioctl(sock, SIOCSIFNETMASK, name_b + _sockaddr_in("255.255.255.0"))

    ifr_flags = fcntl.ioctl(sock, SIOCGIFFLAGS, name_b + b"\x00" * 16)
    flags = struct.unpack_from("H", ifr_flags, 16)[0]
    fcntl.ioctl(sock, SIOCSIFFLAGS, name_b + struct.pack("H", flags | IFF_UP | IFF_RUNNING))

    sock.close()
    logger.info("TUN %s: %s/24  peer %s  MTU %d", name, ip_addr, peer_addr, mtu)


# ── Byte/bit conversion ───────────────────────────────────────────────────

def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert raw bytes to a uint8 bit array (MSB-first)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a uint8 bit array back to bytes, zero-padding to a multiple of 8."""
    remainder = len(bits) % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(8 - remainder, dtype=np.uint8)])
    return np.packbits(bits.astype(np.uint8)).tobytes()


# ── TX thread ─────────────────────────────────────────────────────────────

def _run_tx(config: PipelineConfig, tun_fd: int, sdr: adi.Pluto,
            mtu: int, tx_frame_len: int) -> None:
    """Read IP packets from TUN and transmit over PlutoSDR (runs forever).

    All frames are zero-padded to tx_frame_len samples so the PlutoSDR DMA
    buffer length stays constant across packets of varying sizes.
    """
    tx = TXPipeline(config)
    logger.info("TX thread started (MTU %d bytes, fixed frame %d samples)", mtu, tx_frame_len)

    while True:
        # select() so we can loop cleanly without blocking forever on a dead fd
        ready, _, _ = select.select([tun_fd], [], [], 0.1)
        if not ready:
            continue

        try:
            raw_ip = os.read(tun_fd, mtu)
        except OSError:
            logger.exception("TX: TUN read failed — stopping")
            break

        if not raw_ip:
            continue

        if len(raw_ip) > mtu:
            logger.warning("TX: dropping oversized packet (%d > %d)", len(raw_ip), mtu)
            continue

        payload_bits = bytes_to_bits(raw_ip)
        pkt = Packet(src_mac=0, dst_mac=0, type=0, seq_num=0,
                     length=len(raw_ip), payload=payload_bits)

        try:
            samples = tx.transmit(pkt)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak
            samples = (samples * DAC_SCALE).astype(np.complex64)

            # Pad to the fixed DMA buffer length (PlutoSDR rejects length changes)
            if len(samples) < tx_frame_len:
                samples = np.concatenate([samples,
                    np.zeros(tx_frame_len - len(samples), dtype=np.complex64)])

            sdr.tx(samples)
            logger.debug("TX: %d bytes  (%d samples)", len(raw_ip), tx_frame_len)
        except Exception:
            logger.exception("TX: transmit failed")


# ── RX thread ─────────────────────────────────────────────────────────────

def _run_rx(config: PipelineConfig, tun_fd: int, sdr: adi.Pluto,
            rx_buf_size: int, stop: threading.Event) -> None:
    """Drain PlutoSDR buffers, decode frames, write IP packets to TUN."""
    rx     = RXPipeline(config)
    stream = RxStream(sdr, maxsize=64, lossless=False)
    stream.start(flush=16)

    logger.info("RX thread started (buf %d samples / %.1f ms)",
                rx_buf_size, rx_buf_size / config.SAMPLE_RATE * 1e3)

    prev_buf:    np.ndarray | None = None
    search_from: int               = 0
    buf_count:   int               = 0

    try:
        while not stop.is_set():
            try:
                curr_buf = stream.get(timeout=0.1)
            except queue.Empty:
                continue

            buf_count += 1

            # Sliding window: overlap current buffer with previous so frames
            # that straddle a buffer boundary are not missed.
            raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
            prev_len = len(prev_buf) if prev_buf is not None else 0

            packets, max_det = rx.receive(raw, search_from=search_from)

            prev_buf = curr_buf
            if packets:
                last_ps    = max(p.sample_start for p in packets)
                search_from = max(0, max(last_ps, max_det) - prev_len)
            else:
                search_from = max(0, max_det - prev_len)

            for pkt in packets:
                if not pkt.valid:
                    logger.debug("RX: header CRC failed — dropping frame")
                    continue

                data = bits_to_bytes(pkt.payload)

                # Validate IPv4/IPv6 header
                if len(data) < MIN_IP_PACKET_SIZE:
                    logger.debug("RX: too short (%d bytes) — dropping", len(data))
                    continue
                version = (data[0] >> 4) & 0xF
                if version not in (4, 6):
                    logger.debug("RX: non-IP frame (version=%d) — dropping", version)
                    continue

                # Trim to the IP total-length field so we don't feed padding to the kernel
                if version == 4 and len(data) >= 4:
                    ip_total = (data[2] << 8) | data[3]
                    data = data[:ip_total]

                try:
                    os.write(tun_fd, data)
                    logger.debug("RX: %d bytes injected into TUN", len(data))
                except OSError:
                    logger.exception("RX: TUN write failed")

            if buf_count % 500 == 0:
                logger.info("RX: %d buffers processed  overruns=%d", buf_count, stream.overruns)

    finally:
        stream.stop()
        logger.info("RX thread stopped")


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node",        choices=["A", "B"], required=True, help="Node identity — A=10.0.0.1, B=10.0.0.2")
    parser.add_argument("--tun",         default="pluto0", help="TUN interface name (default: pluto0)")
    parser.add_argument("--tx-gain",     type=float, default=DEFAULT_TX_GAIN, help="TX hardware gain in dB (default: %(default)s)")
    parser.add_argument("--cfo-offset", type=int, default=0, help="LO offset in Hz applied to both RX and TX to compensate oscillator error (default: 0)")
    parser.add_argument("--pluto-ip",    default="192.168.2.1", help="PlutoSDR IP address (default: %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    node    = NODE_CONFIGS[args.node]
    config  = PIPELINE
    tun_mtu = MAX_PACKET_SIZE_BYTES

    # ── Compute rx_buf_size to comfortably hold at least one full frame ──
    # Use a dummy packet at MTU to get the worst-case frame length, then
    # round up to the next power of two and multiply by 16 (same heuristic
    # as loopback_threaded.py which cut drop rates from 48% → 4%).
    _probe_bits = np.zeros(tun_mtu * 8, dtype=np.uint8)
    _probe_pkt  = Packet(src_mac=0, dst_mac=0, type=0, seq_num=0,
                         length=tun_mtu, payload=_probe_bits)
    _tx_probe   = TXPipeline(config)
    _frame_len  = len(_tx_probe.transmit(_probe_pkt))
    rx_buf_size = 16 * int(2 ** np.ceil(np.log2(_frame_len)))
    logger.info("Frame len: %d samples (%.1f ms)  →  RX buf: %d samples (%.1f ms)",
                _frame_len, _frame_len / config.SAMPLE_RATE * 1e3,
                rx_buf_size, rx_buf_size / config.SAMPLE_RATE * 1e3)

    # ── TUN ──────────────────────────────────────────────────────────────
    tun_fd = open_tun(args.tun)
    configure_tun(args.tun, node.ip_addr, node.peer_addr, tun_mtu)

    # ── SDR ──────────────────────────────────────────────────────────────
    sdr = adi.Pluto(f"ip:{args.pluto_ip}")
    configure_tx(sdr, freq=node.tx_freq + args.cfo_offset, gain=args.tx_gain,
                 sample_rate=config.SAMPLE_RATE, cyclic=False)
    configure_rx(sdr, freq=node.rx_freq + args.cfo_offset,
                 sample_rate=config.SAMPLE_RATE, buffer_size=rx_buf_size)

    logger.info("Node %s  TX %.3f MHz (%.0f dB)  RX %.3f MHz (CFO %+d Hz)",
                args.node,
                (node.tx_freq + args.cfo_offset) / 1e6, args.tx_gain,
                (node.rx_freq + args.cfo_offset) / 1e6, args.cfo_offset)

    # ── Threads ──────────────────────────────────────────────────────────
    stop = threading.Event()

    t_tx = threading.Thread(
        target=_run_tx,
        args=(config, tun_fd, sdr, tun_mtu, _frame_len),
        name="TX", daemon=True,
    )
    t_rx = threading.Thread(
        target=_run_rx,
        args=(config, tun_fd, sdr, rx_buf_size, stop),
        name="RX", daemon=True,
    )
    t_rx.start()
    t_tx.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        stop.set()
        t_rx.join(timeout=3)
        try:
            os.close(tun_fd)
        except OSError:
            pass
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass


if __name__ == "__main__":
    main()
