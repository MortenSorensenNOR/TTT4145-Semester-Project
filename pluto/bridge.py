"""Full-duplex IP bridge over PlutoSDR with Go-Back-N ARQ.

Each node uses **two Plutos**: one dedicated to TX, one to RX. A single
USB-2 Pluto cannot sustain 4 Msps full-duplex, so the link is split across
two devices per side. The two still run on disjoint FDD frequencies so
local TX doesn't desensitize local RX through the coax / antenna coupling.
ARQ ("send N, ACK, retransmit from last received + 1") rides on top of the
DSP pipeline and provides reliable in-order delivery of IP packets.

Layout (per node)::

  TUN ──► ARQNode ──► TXPipeline ──► TxStream ──► Pluto-TX ─────► (air)
                                                                       │
                                                                       ▼
  TUN ◄── ARQNode ◄── RXPipeline ◄── RxStream ◄── Pluto-RX ◄─────── (air)

Node A and Node B are mirror images — A transmits on FREQ_A_TO_B and listens
on FREQ_B_TO_A; B does the opposite. Each node's ARQ has its own (src, dst)
pair so a node ignores its own TX leakage and any third party.

Radio IP convention (see pluto.config.NODE_RADIO_IPS): the 3rd octet N of
192.168.N.1 selects the node — even → A, odd → B. Defaults:

  * Node A — tx 192.168.4.1, rx 192.168.2.1
  * Node B — tx 192.168.3.1, rx 192.168.5.1

Usage::

    # Machine A (two Plutos attached)
    sudo .venv/bin/python -m pluto.bridge --node A

    # Machine B (two Plutos attached)
    sudo .venv/bin/python -m pluto.bridge --node B

    # Override IPs if needed:
    sudo .venv/bin/python -m pluto.bridge --node A \\
        --tx-ip 192.168.4.1 --rx-ip 192.168.2.1

    # From A:  ping 10.0.0.1   (B's TUN address)
    # From B:  ping 10.0.0.0   (A's TUN address)
"""

import argparse
import fcntl
import logging
import os
import queue
import select
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass

import adi
import numpy as np

from modules.arq import ARQConfig, ARQNode
from modules.pipeline import Packet, PipelineConfig, RXPipeline, TXPipeline
from pluto.cfo_config import CFO_CONFIG_PATH, load as load_cfo_calibration
from pluto.config import (
    DAC_SCALE,
    DEFAULT_TX_GAIN,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    MAX_PACKET_SIZE_BYTES,
    NODE_RADIO_IPS,
    PIPELINE,
    configure_rx,
    configure_tx,
)
from pluto.sdr_stream import RxStream, TxStream

logger = logging.getLogger(__name__)

# ── Linux ioctl constants ─────────────────────────────────────────────────
TUNSETIFF      = 0x400454CA
IFF_TUN        = 0x0001
IFF_NO_PI      = 0x1000
SIOCGIFFLAGS   = 0x8913
SIOCSIFFLAGS   = 0x8914
SIOCSIFADDR    = 0x8916
SIOCSIFNETMASK = 0x891C
SIOCSIFMTU     = 0x8922
IFF_UP         = 0x1
IFF_RUNNING    = 0x40

# Bridge IP plan — both addresses on the same /24 so the kernel routes
# directly out the TUN without needing a per-host route.
BRIDGE_NETMASK = "255.255.255.0"
NODE_IPS = {"A": "10.0.0.0", "B": "10.0.0.1"}


@dataclass
class NodeConfig:
    tx_freq:   int
    rx_freq:   int
    src:       int   # ARQ logical address (1 bit)
    dst:       int


NODE_CONFIGS = {
    "A": NodeConfig(tx_freq=FREQ_A_TO_B, rx_freq=FREQ_B_TO_A, src=0, dst=1),
    "B": NodeConfig(tx_freq=FREQ_B_TO_A, rx_freq=FREQ_A_TO_B, src=1, dst=0),
}


# ── TUN ───────────────────────────────────────────────────────────────────

class TunFd:
    """Minimal TUN wrapper with the read/write surface ARQNode expects."""

    def __init__(self, name: str, mtu: int, poll_timeout: float = 0.01) -> None:
        self.fd = os.open("/dev/net/tun", os.O_RDWR)
        ifr = struct.pack("16sH14s", name.encode(), IFF_TUN | IFF_NO_PI, b"\x00" * 14)
        fcntl.ioctl(self.fd, TUNSETIFF, ifr)
        self.mtu = mtu
        self._poll_timeout = poll_timeout

    def read(self) -> bytes | None:
        ready, _, _ = select.select([self.fd], [], [], self._poll_timeout)
        if not ready:
            return None
        return os.read(self.fd, self.mtu)

    def write(self, data: bytes) -> None:
        try:
            os.write(self.fd, data)
        except OSError as e:
            logger.warning("TUN write failed: %s", e)

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


def _sockaddr_in(ip_addr: str) -> bytes:
    return struct.pack("HH4s8s", socket.AF_INET, 0, socket.inet_aton(ip_addr), b"\x00" * 8)


def configure_tun_iface(name: str, ip_addr: str, mtu: int) -> None:
    """Set IP/netmask/MTU and bring the TUN interface up via ioctls."""
    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    name_b = name.encode().ljust(16, b"\x00")[:16]

    fcntl.ioctl(sock, SIOCSIFMTU,     name_b + struct.pack("I", mtu))
    fcntl.ioctl(sock, SIOCSIFADDR,    name_b + _sockaddr_in(ip_addr))
    fcntl.ioctl(sock, SIOCSIFNETMASK, name_b + _sockaddr_in(BRIDGE_NETMASK))

    ifr_flags = fcntl.ioctl(sock, SIOCGIFFLAGS, name_b + b"\x00" * 16)
    flags = struct.unpack_from("H", ifr_flags, 16)[0]
    fcntl.ioctl(sock, SIOCSIFFLAGS, name_b + struct.pack("H", flags | IFF_UP | IFF_RUNNING))

    sock.close()
    logger.info("TUN %s: %s/%s MTU %d", name, ip_addr, BRIDGE_NETMASK, mtu)


# ── DSP <-> ARQ glue ──────────────────────────────────────────────────────

def make_pluto_tx(tx_pipeline: TXPipeline, tx_stream: TxStream):
    """Return a ``pluto_tx(packet)`` callable that hands a Packet to the radio."""
    def fn(packet: Packet) -> None:
        samples = tx_pipeline.transmit(packet)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        samples = (samples * DAC_SCALE).astype(np.complex64)
        tx_stream.send(samples)
    return fn


def make_pluto_rx(rx_pipeline: RXPipeline, rx_stream: RxStream):
    """Return a ``pluto_rx() -> list[Packet]`` callable.

    Maintains a sliding-window buffer state across calls so frames straddling
    a buffer boundary are not lost (same approach as one_way_threaded.py /
    the previous bridge.py).
    """
    state = {"prev_buf": None, "search_from": 0}

    def fn() -> list[Packet]:
        try:
            curr_buf = rx_stream.get(timeout=0.1)
        except queue.Empty:
            return []

        prev_buf = state["prev_buf"]
        prev_len = len(prev_buf) if prev_buf is not None else 0
        raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

        packets, max_det = rx_pipeline.receive(raw, search_from=state["search_from"])

        state["prev_buf"] = curr_buf
        if packets:
            last_ps = max(p.sample_start for p in packets)
            state["search_from"] = max(0, max(last_ps, max_det) - prev_len)
        else:
            state["search_from"] = max(0, max_det - prev_len)
        return packets

    return fn


# ── Stats reporter ────────────────────────────────────────────────────────

def _stats_thread(arq: ARQNode, rx_stream: RxStream, tx_stream: TxStream,
                  stop: threading.Event, interval: float = 5.0) -> None:
    while not stop.wait(interval):
        s = arq.stats
        logger.info(
            "ARQ tun_in=%d tun_drop=%d tun_out=%d data_tx=%d (rtx=%d) ack_tx=%d "
            "data_rx=%d (buf=%d) ack_rx=%d (sack=%d) dup=%d foreign=%d timeouts=%d  "
            "rx_overruns=%d tx_bufs=%d",
            s.tun_in, s.tun_dropped, s.tun_out, s.data_tx, s.data_retransmit,
            s.ack_tx, s.data_rx_ok, s.data_rx_buffered, s.ack_rx, s.sack_rx,
            s.data_rx_dup, s.data_rx_foreign, s.timeouts,
            rx_stream.overruns, tx_stream.bufs_sent,
        )


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node",       choices=["A", "B"], required=True,
                        help="Node identity — A=10.0.0.0, B=10.0.0.1")
    parser.add_argument("--tun",        default="pluto0",   help="TUN interface name")
    parser.add_argument("--mtu",        type=int, default=MAX_PACKET_SIZE_BYTES,
                        help="TUN MTU in bytes")
    parser.add_argument("--tx-gain",    type=float, default=DEFAULT_TX_GAIN,
                        help="TX hardware gain in dB")
    parser.add_argument("--cfo-offset", type=int, default=None,
                        help="Manual override for the RX-LO CFO correction in "
                             "Hz. Default: value from pluto/cfo_calibration.json "
                             "(run scripts/cfo_calibrate.py to (re)generate it), "
                             "or 0 if no calibration file exists.")
    parser.add_argument("--tx-ip",      default=None,
                        help="IP of the Pluto used for TX. Default derived "
                             "from --node via pluto.config.NODE_RADIO_IPS.")
    parser.add_argument("--rx-ip",      default=None,
                        help="IP of the Pluto used for RX. Default derived "
                             "from --node via pluto.config.NODE_RADIO_IPS.")
    parser.add_argument("--rx-buf-mult", type=int, default=16,
                        help="RX buffer = mult × next-pow2(frame_len). Smaller "
                             "= lower latency, larger = better throughput. (default: 16)")
    parser.add_argument("--tx-buf-mult", type=int, default=8,
                        help="TX buffer = mult × next-pow2(frame_len) (default: 8)")
    parser.add_argument("--tx-queue-depth", type=int, default=1,
                        help="Packets buffered between ARQ and the TX packer "
                             "thread. Smaller = tighter backpressure and lower "
                             "latency; larger = smoother bursts. (default: 1)")
    parser.add_argument("--rx-queue-depth", type=int, default=2,
                        help="Raw-buffer prefetch depth between the SDR DMA and "
                             "the RX DSP pipeline. (default: 2)")
    parser.add_argument("--window",     type=int, default=3,
                        help="ARQ window size (< SEQ_SPACE/2 = 16). Smaller = "
                             "cleaner ping latency and less retransmit "
                             "cascade; larger = better iperf throughput on a "
                             "healthy link. (default: 3)")
    parser.add_argument("--retransmit-timeout", type=float, default=0.5,
                        help="ARQ retransmit timeout in seconds. Should sit "
                             "~2× above steady-state RTT: too short → false "
                             "retransmits bloat the TX queue; too long → real "
                             "losses take seconds to recover. (default: 0.5)")
    parser.add_argument("--arq-queue-depth", type=int, default=8,
                        help="ARQ TUN-ingress queue. Excess IP packets are "
                             "dropped at the TUN boundary rather than waiting "
                             "seconds to drain — keeps tail latency bounded "
                             "under overload. (default: 8)")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="Seconds between ARQ stats prints (0 = disable)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    node = NODE_CONFIGS[args.node]
    ip_addr = NODE_IPS[args.node]
    config: PipelineConfig = PIPELINE

    tx_ip = args.tx_ip or NODE_RADIO_IPS[args.node]["tx"]
    rx_ip = args.rx_ip or NODE_RADIO_IPS[args.node]["rx"]

    # Resolve RX-LO CFO offset: manual CLI override wins, otherwise use the
    # persisted calibration, otherwise 0. Only the RX LO is offset — the TX
    # transmits at its natural LO and the peer's receiver compensates.
    if args.cfo_offset is not None:
        rx_cfo_hz = args.cfo_offset
        cfo_src   = "cli"
    else:
        cal = load_cfo_calibration()
        if cal is None:
            rx_cfo_hz = 0
            cfo_src   = "unset"
            logger.warning("No CFO calibration found at %s — using 0 Hz. "
                           "Run 'uv run python scripts/cfo_calibrate.py' to generate one.",
                           CFO_CONFIG_PATH)
        else:
            rx_cfo_hz = cal.rx_offset_for(args.node)
            cfo_src   = f"calibration ({cal.measured_at or 'unknown date'})"

    # ── Probe to size buffers ───────────────────────────────────────────
    _probe_pkt = Packet(src_mac=node.src, dst_mac=node.dst, type=0, seq_num=0,
                        length=args.mtu, payload=np.zeros(args.mtu * 8, dtype=np.uint8))
    _tx_probe  = TXPipeline(config)
    frame_len  = len(_tx_probe.transmit(_probe_pkt))
    pow2       = int(2 ** np.ceil(np.log2(frame_len)))
    rx_buf_size = args.rx_buf_mult * pow2
    tx_buf_size = args.tx_buf_mult * pow2

    logger.info("Frame: %d samples (%.1f ms)  TX buf %d (%.1f ms)  RX buf %d (%.1f ms)",
                frame_len, frame_len / config.SAMPLE_RATE * 1e3,
                tx_buf_size, tx_buf_size / config.SAMPLE_RATE * 1e3,
                rx_buf_size, rx_buf_size / config.SAMPLE_RATE * 1e3)

    # ── TUN ─────────────────────────────────────────────────────────────
    tun = TunFd(args.tun, args.mtu)
    configure_tun_iface(args.tun, ip_addr, args.mtu)

    # ── SDR ─────────────────────────────────────────────────────────────
    # Two Plutos per node: one for TX, one for RX. Single-Pluto full-duplex
    # at 4 Msps saturates the USB-2 link and drops samples, so we split the
    # directions across dedicated radios.
    tx_sdr = adi.Pluto(f"ip:{tx_ip}")
    rx_sdr = adi.Pluto(f"ip:{rx_ip}")
    configure_tx(tx_sdr, freq=node.tx_freq,
                 gain=args.tx_gain, sample_rate=config.SAMPLE_RATE, cyclic=False)
    configure_rx(rx_sdr, freq=node.rx_freq + rx_cfo_hz,
                 sample_rate=config.SAMPLE_RATE, buffer_size=rx_buf_size)

    logger.info("Node %s  src=%d dst=%d  TX %.3f MHz (%.0f dB) @ %s  "
                "RX %.3f MHz (CFO %+d Hz, %s) @ %s",
                args.node, node.src, node.dst,
                node.tx_freq / 1e6, args.tx_gain, tx_ip,
                (node.rx_freq + rx_cfo_hz) / 1e6, rx_cfo_hz, cfo_src, rx_ip)

    # ── Pipelines + streams ─────────────────────────────────────────────
    tx_pipe = TXPipeline(config)
    rx_pipe = RXPipeline(config)

    tx_stream = TxStream(tx_sdr, config.SAMPLE_RATE, tx_buf_size,
                         maxsize=args.tx_queue_depth)
    rx_stream = RxStream(rx_sdr, maxsize=args.rx_queue_depth, lossless=False)

    tx_stream.start()
    rx_stream.start(flush=8)

    # ── ARQ ─────────────────────────────────────────────────────────────
    arq_cfg = ARQConfig(
        window_size=args.window,
        retransmit_timeout=args.retransmit_timeout,
        send_queue_maxsize=args.arq_queue_depth,
        src=node.src,
        dst=node.dst,
    )
    arq = ARQNode(
        tun_device=tun,
        pluto_tx=make_pluto_tx(tx_pipe, tx_stream),
        pluto_rx=make_pluto_rx(rx_pipe, rx_stream),
        config=arq_cfg,
    )
    arq.start()

    stop = threading.Event()
    stats_t = None
    if args.stats_interval > 0:
        stats_t = threading.Thread(
            target=_stats_thread,
            args=(arq, rx_stream, tx_stream, stop, args.stats_interval),
            name="stats", daemon=True,
        )
        stats_t.start()

    logger.info("Bridge up — node %s on %s/%s", args.node, ip_addr, BRIDGE_NETMASK)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down…")
    finally:
        stop.set()
        try:
            arq.stop()
        except Exception:
            logger.exception("ARQ stop failed")
        try:
            tx_stream.stop()
        except Exception:
            pass
        try:
            rx_stream.stop()
        except Exception:
            pass
        tun.close()
        try:
            tx_sdr.tx_destroy_buffer()
        except Exception:
            pass
        # Final stats line
        s = arq.stats
        logger.info("FINAL  data_tx=%d rtx=%d ack_tx=%d  data_rx=%d ack_rx=%d dup=%d foreign=%d timeouts=%d  tun_in=%d tun_out=%d",
                    s.data_tx, s.data_retransmit, s.ack_tx,
                    s.data_rx_ok, s.ack_rx, s.data_rx_dup, s.data_rx_foreign, s.timeouts,
                    s.tun_in, s.tun_out)


if __name__ == "__main__":
    sys.exit(main() or 0)
