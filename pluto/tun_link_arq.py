"""TUN-mode radio link with Selective-Repeat ARQ — TCP-capable, full-duplex.

Same shape as ``pluto.tun_link`` but with ``modules.arq.ARQNode`` providing
reliable, ordered delivery so TCP works across the radio link instead of just
UDP. ARQ requires both Plutos open on each node, so ``--mode`` is implicit
("both") and not exposed.

Default IP plan (matches tun_link):
  --node A → TUN pluto0 = 10.0.0.1/24
  --node B → TUN pluto0 = 10.0.0.2/24

Usage:
    sudo .venv/bin/python -m pluto.tun_link_arq                # node autodetected
    sudo .venv/bin/python -m pluto.tun_link_arq --video        # video FDD pair
    sudo .venv/bin/python -m pluto.tun_link_arq --node A       # force node A

UDP datagrams read from TUN take a fast-path: PacketType.RAW frames that
bypass ARQ entirely (no seq, no ACK, no retransmit). Disable with
``--no-bypass-udp``.
"""

import argparse
import queue
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet, PacketType
from modules.arq import ARQConfig, ARQNode, SEQ_SPACE
from modules.tun import TunDevice
from pluto.config import (
    DAC_SCALE,
    configure_rx,
    configure_tx,
    get_node_freqs,
)
from pluto.setup_config import SETUP_PATH, load_or_die as load_setup, resolve_node
from pluto.sdr_stream import RxStream, TxStream
from pluto.live_status import (
    LiveStatus, RateMeter, _fmt_rate, _fmt_bytes, _install_live_logging,
)
from utils.bit import round_up

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Per-frame HEADER info logs from the DSP pipeline drown the pinned status block under ARQ traffic.
logging.getLogger("modules.pipeline").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

NODE_ADDR      = {"A": 0, "B": 1}
DEFAULT_TUN_IP = {"A": "10.0.0.1", "B": "10.0.0.2"}

_IP_PROTO_UDP = 17


def _is_udp(ip_packet: bytes) -> bool:
    if len(ip_packet) < 20:
        return False
    version = ip_packet[0] >> 4
    if version == 4:
        return ip_packet[9] == _IP_PROTO_UDP
    if version == 6 and len(ip_packet) >= 40:
        return ip_packet[6] == _IP_PROTO_UDP
    return False


class BypassDemuxTun:
    """TUN wrapper that diverts UDP packets to ``on_udp`` and passes everything
    else through to ARQ. Sits between TunDevice and ARQ's TUN reader thread."""

    def __init__(self, tun: TunDevice, on_udp):
        self._tun = tun
        self._on_udp = on_udp
        self.mtu = tun.mtu

    def read(self) -> bytes | None:
        # Capped inner loop so non-UDP packets and stop-checks aren't starved.
        for _ in range(64):
            data = self._tun.read()
            if data is None:
                return None
            if _is_udp(data):
                self._on_udp(data)
                continue
            return data
        return None

    def write(self, data: bytes) -> None:
        self._tun.write(data)


class RadioTx:
    """Build & enqueue one Packet onto the live TxStream. Satisfies the
    ARQNode pluto_tx contract; ``send_raw`` is the bypass path for UDP."""

    def __init__(self, tx_pipe: TXPipeline, tx_stream: TxStream,
                 tx_rate: RateMeter, my_addr: int, peer_addr: int,
                 stats_dict: dict):
        self._pipe = tx_pipe
        self._stream = tx_stream
        self._rate = tx_rate
        self._my_addr = my_addr
        self._peer_addr = peer_addr
        self._stats = stats_dict

    def _push(self, packet: Packet) -> None:
        samples = self._pipe.transmit(packet)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        self._stream.send((samples * DAC_SCALE).astype(np.complex64))

    def __call__(self, packet: Packet) -> None:
        self._push(packet)
        # Only DATA bytes count as goodput; ACKs are protocol overhead.
        if packet.length > 0 and packet.type == int(PacketType.DATA):
            self._rate.add(packet.length)

    def send_raw(self, payload: bytes) -> None:
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pkt = Packet(
            src_mac=self._my_addr,
            dst_mac=self._peer_addr,
            type=int(PacketType.RAW),
            seq_num=0,
            length=len(payload),
            payload=bits,
            valid=True,
        )
        self._push(pkt)
        self._rate.add(len(payload))
        self._stats["udp_bypass_tx"] += 1


class RadioRx:
    """Drain RxStream → run RXPipeline → return list[Packet] for ARQNode.
    Buffer-stitching mirrors tun_link.rx_thread_fn so frames straddling the
    DMA boundary still decode. RAW frames bypass ARQ and go straight to TUN."""

    def __init__(self, rx_stream: RxStream, rx_pipe: RXPipeline,
                 rx_rate: RateMeter, stats_dict: dict, tun: TunDevice,
                 my_addr: int):
        self._stream = rx_stream
        self._pipe = rx_pipe
        self._rate = rx_rate
        self._stats = stats_dict
        self._tun = tun
        self._my_addr = my_addr
        self._prev_buf: np.ndarray | None = None
        self._search_from = 0

    def __call__(self) -> list[Packet]:
        try:
            curr_buf = self._stream.get(timeout=0.05)
        except queue.Empty:
            return []

        prev_len = len(self._prev_buf) if self._prev_buf is not None else 0
        raw = (np.concatenate([self._prev_buf, curr_buf])
               if self._prev_buf is not None else curr_buf)

        packets, max_det = self._pipe.receive(raw, search_from=self._search_from)
        self._stats["data_rx_payload_bad"] += self._pipe.last_payload_failures

        self._prev_buf = curr_buf
        if packets:
            last_ps = max(p.sample_start for p in packets)
            self._search_from = max(0, max(last_ps, max_det) - prev_len)
        else:
            self._search_from = max(0, max_det - prev_len)

        forwarded: list[Packet] = []
        for pkt in packets:
            if not pkt.valid:
                continue
            if pkt.type == int(PacketType.RAW):
                if pkt.dst_mac >= 0 and pkt.dst_mac != self._my_addr:
                    continue
                if pkt.length > 0:
                    payload = np.packbits(
                        pkt.payload[: pkt.length * 8].astype(np.uint8)
                    ).tobytes()
                    try:
                        self._tun.write(payload)
                    except OSError:
                        pass
                    self._rate.add(pkt.length)
                    self._stats["udp_bypass_rx"] += 1
                continue
            if pkt.length > 0 and pkt.type == int(PacketType.DATA):
                self._rate.add(pkt.length)
            forwarded.append(pkt)
        return forwarded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node",     type=str,   default=None,   help="Node identity A or B; picks default TX/RX IPs from pluto/setup.json and TUN IP from DEFAULT_TUN_IP. Autodetected from local Pluto subnet if omitted.")
    parser.add_argument("--gain",     type=float, default=-3,   help="TX gain in dB (default: -3)")
    parser.add_argument("--video",    action="store_true",        help="Use the video-mode FDD pair (2327/2390 MHz) instead of the default network pair (2470/2475 MHz).")
    parser.add_argument("--rx-gain-mode", type=str, default="manual",
                        choices=("slow_attack", "fast_attack", "hybrid", "manual"),
                        help="AD9361 RX AGC mode (default: manual). The auto "
                             "modes drift during the silence between bursts, "
                             "ramping gain up so the next packet clips the ADC "
                             "and the constellation widens 3–5×.")
    parser.add_argument("--rx-gain", type=float, default=45.0,
                        help="Fixed RX hardware gain in dB when "
                             "--rx-gain-mode=manual (default: 45, AD9361 range "
                             "~0–71). Ignored for any auto AGC mode.")
    parser.add_argument("--tx-buf-mult", type=float, default=1.05, help="TX buffer size as multiple of next-power-of-2 frame length")
    parser.add_argument("--rx-buf-mult", type=float, default=None,
                        help="RX buffer size as multiple of frame_len. If unset, uses "
                             "max(2.5 * frame_len, 60000) — the empirical floor below which "
                             "the AD9361/libiio refill seam corrupts straddling packets.")
    parser.add_argument("--rx-buf-samples", type=int, default=None,
                        help="Override rx_buffer_size with an exact sample count (ignores --rx-buf-mult).")
    parser.add_argument("--tun-name", type=str,   default="pluto0", help="TUN interface name (default: pluto0)")
    parser.add_argument("--tun-ip",   type=str,   default=None,   help="TUN IPv4 address with /24 implicit (default: 10.0.0.1 for A, 10.0.0.2 for B)")
    parser.add_argument("--mtu",      type=int,   default=1500,   help="TUN MTU in bytes (default: 1500)")
    parser.add_argument("--window-size", type=int, default=63,
                        help=f"Selective-Repeat sender window in frames (default: 63). "
                             f"Must satisfy 1 <= window < SEQ_SPACE/2 (={SEQ_SPACE // 2}).")
    parser.add_argument("--retransmit-timeout", type=float, default=0.1,
                        help="Seconds with no ACK before unacked seqs are retransmitted (default: 0.1).")
    parser.add_argument("--send-queue-maxsize", type=int, default=64,
                        help="TUN→ARQ queue depth before TUN reads are dropped (default: 64).")
    parser.add_argument("--no-bypass-udp", action="store_true",
                        help="Send UDP through ARQ too instead of via PacketType.RAW.")
    parser.add_argument("--tx-stream-maxsize", type=int, default=64,
                        help="TxStream packet queue depth (default: 64). Each "
                             "queued packet is ~one DMA buffer of air-time, so "
                             "64 ≈ 160 ms of hidden buffer. Lower (e.g. 8) to "
                             "tighten backpressure for timestamp-based "
                             "transports like SRT.")
    parser.add_argument("--txqueuelen", type=int, default=200,
                        help="Kernel TUN tx queue length in packets (default: "
                             "200). Lower values tighten backpressure but risk "
                             "I-frame burst drops on plain UDP video.")
    args = parser.parse_args()

    if args.window_size < 1 or args.window_size >= SEQ_SPACE // 2:
        print(f"ERROR: --window-size must satisfy 1 <= w < {SEQ_SPACE // 2}, got {args.window_size}",
              file=sys.stderr)
        sys.exit(1)

    setup = load_setup()
    args.node = resolve_node(setup, args.node)
    if args.node not in setup.nodes:
        print(f"ERROR: --node must be one of {sorted(setup.nodes)}, got '{args.node}'")
        sys.exit(1)
    if args.node not in NODE_ADDR:
        print(f"ERROR: --node must be 'A' or 'B' for the ARQ TUN bridge, got '{args.node}'")
        sys.exit(1)

    tx_uri = setup.tx_uri(args.node)
    rx_uri = setup.rx_uri(args.node)

    rx_cfo_hz = 0
    cfo_src   = "n/a"
    if setup.cfo is None:
        cfo_src   = "unset"
        print(f"  [warn] no CFO calibration in {SETUP_PATH} — using 0 Hz. "
              f"Run 'uv run python scripts/cfo_calibrate.py' to generate one.")
    else:
        rx_cfo_hz = setup.cfo.rx_offset_for(args.node)
        cfo_src   = f"calibration ({setup.cfo.measured_at or 'unknown date'})"

    peer        = "B" if args.node == "A" else "A"
    my_addr     = NODE_ADDR[args.node]
    peer_addr   = NODE_ADDR[peer]
    tun_ip      = args.tun_ip or DEFAULT_TUN_IP[args.node]
    peer_tun_ip = DEFAULT_TUN_IP[peer]

    pipe_cfg = PipelineConfig()
    tx_pipe = TXPipeline(pipe_cfg)
    rx_pipe = RXPipeline(pipe_cfg)

    rng = np.random.default_rng(0)

    # Probe one MTU-sized packet to learn frame_len for buffer sizing.
    _probe_bits    = rng.integers(0, 2, args.mtu * 8, dtype=np.uint8)
    _probe_pkt     = Packet(src_mac=my_addr, dst_mac=peer_addr, type=0, seq_num=0,
                            length=args.mtu, payload=_probe_bits)
    _probe_samples = tx_pipe.transmit(_probe_pkt)
    frame_len      = len(_probe_samples)
    # Below this samples/buf, the AD9361/libiio refill seam corrupts straddling packets.
    RX_BUF_FLOOR = 60000
    if args.rx_buf_samples is not None:
        if args.rx_buf_samples < frame_len:
            print(f"ERROR: --rx-buf-samples ({args.rx_buf_samples}) must be >= frame_len ({frame_len})")
            sys.exit(1)
        rx_buf_size = args.rx_buf_samples
    elif args.rx_buf_mult is not None:
        rx_buf_size = round_up(int(args.rx_buf_mult * frame_len))
    else:
        rx_buf_size = max(round_up(int(2.5 * frame_len)), RX_BUF_FLOOR)
    tx_buf_size    = round_up(int(args.tx_buf_mult * frame_len))

    node_freqs = get_node_freqs(args.node, video=args.video)
    tx_freq = node_freqs["tx"]
    rx_freq = node_freqs["rx"]

    bypass_udp = not args.no_bypass_udp

    print(f"Node      : {args.node}  (peer {peer})  mode=both (ARQ)  freq={'video' if args.video else 'network'}")
    print(f"TUN       : {args.tun_name} = {tun_ip}/24  (peer {peer_tun_ip})  MTU {args.mtu}  txqueuelen {args.txqueuelen}")
    print(f"TX radio  : {tx_uri}   @ {tx_freq / 1e6:.3f} MHz")
    if args.rx_gain_mode == "manual":
        rx_gain_desc = f"manual {args.rx_gain:.1f} dB"
    else:
        rx_gain_desc = f"AGC={args.rx_gain_mode}"
    print(f"RX radio  : {rx_uri}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
          f"(CFO {rx_cfo_hz:+d} Hz, {cfo_src}; {rx_gain_desc})")
    print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
    print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"TX buf    : {tx_buf_size} samples  ({tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)  "
          f"stream_q={args.tx_stream_maxsize} "
          f"(~{args.tx_stream_maxsize * tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.0f} ms hidden)")
    print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"TX gain   : {args.gain} dB")
    print(f"ARQ       : window={args.window_size} (selective-repeat, SEQ_SPACE={SEQ_SPACE}) "
          f"timeout={args.retransmit_timeout}s  send_q={args.send_queue_maxsize}")
    print(f"UDP fast  : {'BYPASS (RAW frames, no ARQ)' if bypass_udp else 'OFF — UDP runs through ARQ'}")
    print()

    tx_sdr = adi.Pluto(tx_uri)
    configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False)

    rx_sdr = adi.Pluto(rx_uri)
    configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz,
                 gain_mode=args.rx_gain_mode, gain=args.rx_gain)
    rx_sdr.rx_buffer_size = rx_buf_size

    def _ip(*ip_args):
        subprocess.run(["ip", *ip_args], check=True)

    tun = TunDevice(name=args.tun_name, mtu=args.mtu)
    try:
        _ip("link", "set", "dev", args.tun_name, "mtu", str(args.mtu))
        _ip("link", "set", "dev", args.tun_name, "txqueuelen", str(args.txqueuelen))
        _ip("addr", "add", f"{tun_ip}/24", "dev", args.tun_name)
        _ip("link", "set", args.tun_name, "up")
    except subprocess.CalledProcessError as e:
        tun.close()
        print(f"ERROR: failed to configure TUN {args.tun_name}: {e}", file=sys.stderr)
        print("       (need root + clean /24, e.g. no stale pluto0 from a prior run)", file=sys.stderr)
        sys.exit(1)

    status   = LiveStatus(n_lines=2)
    tx_rate  = RateMeter()
    rx_rate  = RateMeter()
    _install_live_logging(status)

    rx_stats = {
        "data_rx_payload_bad": 0,
        "udp_bypass_tx":       0,
        "udp_bypass_rx":       0,
    }

    tx_stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size,
                         maxsize=args.tx_stream_maxsize)
    rx_stream = RxStream(rx_sdr, maxsize=128, lossless=True)
    tx_stream.start()
    rx_stream.start(flush=16)

    radio_tx = RadioTx(tx_pipe, tx_stream, tx_rate, my_addr, peer_addr, rx_stats)
    radio_rx = RadioRx(rx_stream, rx_pipe, rx_rate, rx_stats, tun, my_addr)

    arq_tun = BypassDemuxTun(tun, radio_tx.send_raw) if bypass_udp else tun

    arq_cfg = ARQConfig(
        window_size=args.window_size,
        retransmit_timeout=args.retransmit_timeout,
        send_queue_maxsize=args.send_queue_maxsize,
        src=my_addr,
        dst=peer_addr,
    )
    arq = ARQNode(arq_tun, radio_tx, radio_rx, arq_cfg)
    arq.start()

    stop_event = threading.Event()
    LOG_INTERVAL_S = 1.0

    def _status_loop():
        last_log_t = 0.0
        while not stop_event.is_set():
            s = arq.stats
            tx_msg = (
                f"[TX] tun_in={s.tun_in:>7d} drop={s.tun_dropped:>4d} "
                f"data={s.data_tx:>7d} retx={s.data_retransmit:>5d} "
                f"raw_tx={rx_stats['udp_bypass_tx']:>6d} "
                f"acks_rx={s.ack_rx:>6d} sack={s.sack_rx:>5d} timeouts={s.timeouts:>4d}  "
                f"goodput={_fmt_rate(tx_rate.rate_bps)} "
                f"avg={_fmt_rate(tx_rate.avg_bps)} "
                f"total={_fmt_bytes(tx_rate.total_bytes)}"
            )
            rx_msg = (
                f"[RX] data_ok={s.data_rx_ok:>7d} buf={s.data_rx_buffered:>4d} "
                f"dup={s.data_rx_dup:>4d} foreign={s.data_rx_foreign:>4d} "
                f"raw_rx={rx_stats['udp_bypass_rx']:>6d} "
                f"pay_bad={rx_stats['data_rx_payload_bad']:>4d} "
                f"acks_tx={s.ack_tx:>6d} tun_out={s.tun_out:>7d}  "
                f"goodput={_fmt_rate(rx_rate.rate_bps)} "
                f"avg={_fmt_rate(rx_rate.avg_bps)} "
                f"total={_fmt_bytes(rx_rate.total_bytes)}"
            )
            status.set(0, "  " + tx_msg)
            status.set(1, "  " + rx_msg)
            now = time.monotonic()
            if now - last_log_t >= LOG_INTERVAL_S:
                logger.info(tx_msg)
                logger.info(rx_msg)
                last_log_t = now
            stop_event.wait(timeout=0.2)

    t_status = threading.Thread(target=_status_loop, name="status", daemon=True)
    t_status.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop TxStream first so any ARQ thread blocked in tx_stream.send() returns immediately; otherwise arq.stop() deadlocks.
        stop_event.set()
        try:
            tx_stream.stop()
        except Exception:
            pass
        arq.stop()
        try:
            rx_stream.stop()
        except Exception:
            pass
        status.stop()
        t_status.join(timeout=1.0)

        s = arq.stats
        print()
        print("=" * 50)
        print("FINAL STATS  (ARQ)")
        print("=" * 50)
        print(f"TUN in       : {s.tun_in}")
        print(f"TUN out      : {s.tun_out}")
        print(f"TUN dropped  : {s.tun_dropped}  (send queue full)")
        print(f"DATA tx      : {s.data_tx}  (incl. {s.data_retransmit} retransmits)")
        print(f"ACK  tx      : {s.ack_tx}")
        print(f"DATA rx ok   : {s.data_rx_ok}  (in-order + buffered)")
        print(f"DATA rx buf  : {s.data_rx_buffered}  (out-of-order, awaiting gap-fill)")
        print(f"DATA rx dup  : {s.data_rx_dup}")
        print(f"DATA rx for  : {s.data_rx_foreign}  (dst_mac != us)")
        print(f"DATA rx pbad : {rx_stats['data_rx_payload_bad']}  (header OK, payload CRC/LDPC failed)")
        print(f"ACK  rx      : {s.ack_rx}  (sack-confirmed: {s.sack_rx})")
        print(f"Timeouts     : {s.timeouts}")
        print(f"UDP raw tx   : {rx_stats['udp_bypass_tx']}  (bypassed ARQ)")
        print(f"UDP raw rx   : {rx_stats['udp_bypass_rx']}  (delivered straight to TUN)")
        print(f"TX goodput   : {_fmt_rate(tx_rate.avg_bps)}  total {_fmt_bytes(tx_rate.total_bytes)}")
        print(f"RX goodput   : {_fmt_rate(rx_rate.avg_bps)}  total {_fmt_bytes(rx_rate.total_bytes)}")
        print("=" * 50)

        subprocess.run(["ip", "link", "set", args.tun_name, "down"], check=False)
        tun.close()
        try:
            tx_sdr.tx_destroy_buffer()
        except Exception:
            pass
        del tx_sdr
        del rx_sdr

    sys.exit(0)
