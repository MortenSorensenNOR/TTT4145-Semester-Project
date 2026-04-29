"""TUN-mode radio link with Selective-Repeat ARQ — TCP-capable, full-duplex.

Same shape as ``pluto.tun_link`` (TUN device → SDR → TUN device) but with
``modules.arq.ARQNode`` providing reliable, ordered delivery so TCP works
across the radio link instead of just UDP.

Required hardware: ARQ needs both a TX *and* an RX Pluto on each node — the
sender retransmits on timeout / SACK gaps and the receiver streams ACKs back
in real time. ``--mode`` from ``tun_link`` is therefore implicit (always
"both") and not exposed.

Default IP plan (matches tun_link):
    --node A → TUN pluto0 = 10.0.0.1/24
    --node B → TUN pluto0 = 10.0.0.2/24

Usage:
    sudo .venv/bin/python -m pluto.tun_link_arq --node B
    sudo .venv/bin/python -m pluto.tun_link_arq --node A

Defaults match the reliable tun_link config (TX gain 0, RX manual 45 dB) so
both ends only need ``--node`` flipped — flags are otherwise symmetric.

After both ends are up, IP traffic between 10.0.0.1 ↔ 10.0.0.2 (ping, ssh,
iperf3 -t 0, etc.) flows through ARQ. TCP sees a clean, ordered, lossy-but-
recoverable pipe instead of the bare 1-5 % drop UDP path.
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
from modules.parallel_pipeline import apply_cpu_affinity, parse_cpu_spec
from modules.arq import ARQConfig, ARQNode, SEQ_SPACE
from modules.tun import TunDevice
from pluto.config import (
    DAC_SCALE,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    configure_rx,
    configure_tx,
)
from pluto.setup_config import SETUP_PATH, load_or_die as load_setup
from pluto.sdr_stream import RxStream, TxStream
from pluto.live_status import (
    LiveStatus, RateMeter, _fmt_rate, _fmt_bytes, _install_live_logging,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence the per-frame ``HEADER: crc=...`` info log from the DSP pipeline.
# Under ARQ both sides decode every retransmit + every ACK — at ~30 frames/s
# the volume drowns the pinned status block and slows the RX thread enough
# that ACKs queue up and look like real RTT.
logging.getLogger("modules.pipeline").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

NODE_FREQS = {
    "A": {"tx": FREQ_A_TO_B, "rx": FREQ_B_TO_A},
    "B": {"tx": FREQ_B_TO_A, "rx": FREQ_A_TO_B},
}

# 1-bit logical MAC — same scheme as tun_link so dst_mac filtering on the RX
# side rejects self-leakage and stray frames.
NODE_ADDR = {"A": 0, "B": 1}

DEFAULT_TUN_IP = {"A": "10.0.0.1", "B": "10.0.0.2"}


# ---------------------------------------------------------------------------
# IP-protocol fast-path: UDP bypasses ARQ.
#
# Why: UDP apps already accept loss; pushing them through ARQ wastes air on
# retransmits and adds the BDP-cap-induced latency to flows that don't want
# it. We tag UDP frames with PacketType.RAW so the receiver writes them
# straight to TUN — no seq, no ACK, just one-shot delivery.
# ---------------------------------------------------------------------------

_IP_PROTO_UDP = 17  # IPv4 protocol field / IPv6 next-header field


def _is_udp(ip_packet: bytes) -> bool:
    """True iff ``ip_packet`` (raw L3 bytes from TUN) is a UDP datagram.

    Handles IPv4 and the simple IPv6 case (no extension headers — fine for
    typical iperf3/video traffic; ESP/AH/HBH-fragmented IPv6 falls through to
    the ARQ path, which is conservative and harmless).
    """
    if len(ip_packet) < 20:
        return False
    version = ip_packet[0] >> 4
    if version == 4:
        return ip_packet[9] == _IP_PROTO_UDP
    if version == 6 and len(ip_packet) >= 40:
        return ip_packet[6] == _IP_PROTO_UDP
    return False


class BypassDemuxTun:
    """TUN wrapper: UDP packets divert to ``on_udp``; everything else passes
    through to ARQ as a normal ``read()``.

    Sits between :class:`modules.tun.TunDevice` and ARQ's internal TUN reader
    thread. ARQ never sees UDP packets, so they don't enter its window /
    consume seq numbers — they're sent on the air via ``on_udp`` (which the
    main module wires to a RAW-frame TxStream send).
    """

    def __init__(self, tun: TunDevice, on_udp):
        self._tun = tun
        self._on_udp = on_udp
        self.mtu = tun.mtu

    def read(self) -> bytes | None:
        # Loop only while UDP keeps arriving; bail out on the first read
        # that returns either a non-UDP packet or a None (no data this poll).
        # Capping the inner loop bounds latency for the caller's stop-checks.
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


# ---------------------------------------------------------------------------
# Radio adapters that satisfy the ARQNode pluto_tx / pluto_rx contract.
# ---------------------------------------------------------------------------

class RadioTx:
    """Build & enqueue one Packet onto the live TxStream.

    ARQNode hands us both DATA frames (default modulation / coding) and ACK
    frames (BPSK + uncoded — set on the Packet by ARQNode itself, the TX
    pipeline honours per-packet overrides). The bypass path also calls
    :meth:`send_raw` directly to inject UDP frames without going through ARQ.
    """

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
        """Bypass-ARQ send for UDP. Builds a RAW-typed Packet and queues it
        on the TxStream alongside ARQ's frames. No seq, no retransmit."""
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

    Mirrors the inline-mode buffer-stitching from ``tun_link.rx_thread_fn``:
    one previous buffer is kept around so frames straddling the DMA boundary
    still decode, and ``search_from`` is advanced past already-processed
    samples to avoid re-detecting the same preamble twice.

    Also intercepts ``PacketType.RAW`` frames before they reach ARQ — those
    are UDP bypass-mode packets and go straight to TUN with no ACK.
    """

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

        # Split: RAW frames go straight to TUN; DATA/ACK forward to ARQ.
        forwarded: list[Packet] = []
        for pkt in packets:
            if not pkt.valid:
                continue
            if pkt.type == int(PacketType.RAW):
                if pkt.dst_mac >= 0 and pkt.dst_mac != self._my_addr:
                    continue  # not for us — drop silently
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
            # DATA/ACK: ARQ handles dst_mac filter + delivery itself.
            if pkt.length > 0 and pkt.type == int(PacketType.DATA):
                self._rate.add(pkt.length)
            forwarded.append(pkt)
        return forwarded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--node",     type=str,   default="A",    help="Node identity A or B; picks default TX/RX IPs from pluto/setup.json and TUN IP from DEFAULT_TUN_IP")
    parser.add_argument("--gain",     type=float, default=0,      help="TX gain in dB (default: 0 — matches the reliable tun_link config)")
    parser.add_argument("--tx-ip",    type=str,   default=None,   help="Override TX Pluto IP (default: derived from --node via pluto/setup.json)")
    parser.add_argument("--rx-ip",    type=str,   default=None,   help="Override RX Pluto IP (default: derived from --node via pluto/setup.json)")
    parser.add_argument("--tx-freq",  type=float, default=None,   help="TX center frequency in Hz (default: NODE_FREQS[node]['tx'])")
    parser.add_argument("--rx-freq",  type=float, default=None,   help="RX center frequency in Hz (default: NODE_FREQS[node]['rx'])")
    parser.add_argument("--cfo-offset", type=int, default=None,   help="Manual override for the RX-LO CFO correction in Hz. Default: value from pluto/setup.json or 0.")
    parser.add_argument("--rx-gain-mode", type=str, default="manual",
                        choices=("slow_attack", "fast_attack", "hybrid", "manual"),
                        help="AD9361 RX AGC mode (default: manual — matches the reliable tun_link config; "
                             "auto AGC drifts during silence between bursts).")
    parser.add_argument("--rx-gain", type=float, default=45.0,
                        help="Fixed RX gain in dB when --rx-gain-mode=manual (default: 45, range ~0–71).")
    parser.add_argument("--tx-buf-mult", type=int, default=8,     help="TX buffer size as multiple of next-power-of-2 frame length (default: 8). "
                                                                       "Drop to 1-2 to slash RTT — bandwidth-delay product caps ARQ throughput, "
                                                                       "so halving the TX buffer doubles the throughput ceiling.")
    parser.add_argument("--rx-buf-mult", type=int, default=16,    help="RX buffer size as multiple of next-power-of-2 frame length (default: 16). "
                                                                       "Drop to 4 for low-latency ARQ; the ~22 ms cycle still fits a frame plus jitter. "
                                                                       "Same RTT trade-off as --tx-buf-mult.")
    parser.add_argument("--kernel-buffers", type=int, default=4,  help="libiio kernel-side DMA ring depth on both TX and RX (default: 4). "
                                                                       "Each slot adds tx_buf_mult × frame airtime of latency before the air. "
                                                                       "Going below 2 risks DMA underruns under USB scheduling jitter.")
    parser.add_argument("--tx-filler-amp", type=float, default=0.0,
                        help="Per-component amplitude of complex Gaussian noise filler emitted between packets. "
                             "0.0 = silent zero-fill. ~512 keeps RX AGC/Costas/Gardner engaged in sparse traffic.")
    parser.add_argument("--hardware-rrc", action="store_true",    help="Use the FPGA hardware RRC/4x interpolation path on TX.")
    parser.add_argument("--tun-name", type=str,   default="pluto0", help="TUN interface name (default: pluto0)")
    parser.add_argument("--tun-ip",   type=str,   default=None,   help="TUN IPv4 address with /24 implicit (default: 10.0.0.1 for A, 10.0.0.2 for B)")
    parser.add_argument("--mtu",      type=int,   default=1500,   help="TUN MTU in bytes (default: 1500)")
    # ARQ parameters --------------------------------------------------------
    parser.add_argument("--window-size", type=int, default=63,
                        help=f"Selective-Repeat sender window in frames (default: 63). "
                             f"Must satisfy 1 <= window < SEQ_SPACE/2 (={SEQ_SPACE // 2}). "
                             f"Larger windows lift the bandwidth-delay-product cap "
                             f"(throughput ≈ window × MTU / RTT).")
    parser.add_argument("--retransmit-timeout", type=float, default=0.5,
                        help="Seconds with no ACK before unacked seqs are retransmitted "
                             "(default: 0.5). With tx_buf_mult=8 and rx_buf=16x next_pow2, "
                             "RTT is ~250-300 ms (one TX buffer + one RX buffer per "
                             "direction); too short and the sender retransmits before the "
                             "ACK can land, flooding the queue and starving the ACK path.")
    parser.add_argument("--send-queue-maxsize", type=int, default=64,
                        help="TUN→ARQ queue depth before TUN reads are dropped (default: 64).")
    parser.add_argument("--no-bypass-udp", action="store_true",
                        help="Disable the UDP fast-path. By default UDP packets read "
                             "from TUN are sent as PacketType.RAW frames that bypass "
                             "ARQ entirely (no seq, no ACK, no retransmit) — UDP apps "
                             "tolerate loss but choke on the BDP-induced latency of "
                             "ARQ. Use this flag if you want every packet through ARQ "
                             "regardless of protocol.")
    # CPU pinning -----------------------------------------------------------
    parser.add_argument("--worker-cpus", type=str, default="0",
                        help="CPU IDs to pin the main process to. Same forms as tun_link "
                             "('0', '0-3', 'p-cores', '' to disable). Default: '0'.")
    args = parser.parse_args()

    if args.window_size < 1 or args.window_size >= SEQ_SPACE // 2:
        print(f"ERROR: --window-size must satisfy 1 <= w < {SEQ_SPACE // 2}, got {args.window_size}",
              file=sys.stderr)
        sys.exit(1)

    if args.worker_cpus == "":
        worker_cpus: list[int] | None = None
    else:
        worker_cpus = parse_cpu_spec(args.worker_cpus)
    apply_cpu_affinity(worker_cpus)

    setup = load_setup()
    if args.node not in setup.nodes:
        print(f"ERROR: --node must be one of {sorted(setup.nodes)}, got '{args.node}'")
        sys.exit(1)
    if args.node not in NODE_ADDR:
        print(f"ERROR: --node must be 'A' or 'B' for the ARQ TUN bridge, got '{args.node}'")
        sys.exit(1)

    tx_ip = args.tx_ip or setup.tx_ip(args.node)
    rx_ip = args.rx_ip or setup.rx_ip(args.node)

    rx_cfo_hz = 0
    cfo_src   = "n/a"
    if args.cfo_offset is not None:
        rx_cfo_hz = args.cfo_offset
        cfo_src   = "cli"
    elif setup.cfo is None:
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

    # ---------------------------------------------------------------------------
    # Pipelines
    # ---------------------------------------------------------------------------

    pipe_cfg = PipelineConfig(hardware_rrc=args.hardware_rrc)
    pipe_cfg.SYNC_CONFIG.fine_peak_ratio_min = np.float32(7.0)
    tx_pipe = TXPipeline(pipe_cfg)
    rx_pipe = RXPipeline(pipe_cfg)

    # ---------------------------------------------------------------------------
    # Buffer sizing — probe one MTU-sized packet to learn frame_len.
    # ---------------------------------------------------------------------------

    rng = np.random.default_rng(0)
    _probe_bits    = rng.integers(0, 2, args.mtu * 8, dtype=np.uint8)
    _probe_pkt     = Packet(src_mac=my_addr, dst_mac=peer_addr, type=0, seq_num=0,
                            length=args.mtu, payload=_probe_bits)
    _probe_samples = tx_pipe.transmit(_probe_pkt)
    frame_len      = len(_probe_samples)
    rx_buf_size    = args.rx_buf_mult * int(2 ** np.ceil(np.log2(frame_len)))
    tx_buf_size    = args.tx_buf_mult * int(2 ** np.ceil(np.log2(frame_len)))

    tx_freq = int(args.tx_freq) if args.tx_freq is not None else NODE_FREQS[args.node]["tx"]
    rx_freq = int(args.rx_freq) if args.rx_freq is not None else NODE_FREQS[args.node]["rx"]

    print(f"Node      : {args.node}  (peer {peer})  mode=both (ARQ)")
    print(f"TUN       : {args.tun_name} = {tun_ip}/24  (peer {peer_tun_ip})  MTU {args.mtu}")
    print(f"TX radio  : {tx_ip}   @ {tx_freq / 1e6:.3f} MHz")
    if args.rx_gain_mode == "manual":
        rx_gain_desc = f"manual {args.rx_gain:.1f} dB"
    else:
        rx_gain_desc = f"AGC={args.rx_gain_mode}"
    print(f"RX radio  : {rx_ip}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
          f"(CFO {rx_cfo_hz:+d} Hz, {cfo_src}; {rx_gain_desc})")
    print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
    print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"TX buf    : {tx_buf_size} samples  ({tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"TX gain   : {args.gain} dB")
    print(f"ARQ       : window={args.window_size} (selective-repeat, SEQ_SPACE={SEQ_SPACE}) "
          f"timeout={args.retransmit_timeout}s  send_q={args.send_queue_maxsize}")
    bypass_udp = not args.no_bypass_udp
    print(f"UDP fast  : {'BYPASS (RAW frames, no ARQ)' if bypass_udp else 'OFF — UDP runs through ARQ'}")
    print(f"CPU pin   : {worker_cpus if worker_cpus else 'off'}")
    print()

    # ---------------------------------------------------------------------------
    # SDR setup — full-duplex, both Plutos open.
    # ---------------------------------------------------------------------------

    tx_sdr = adi.Pluto("ip:" + tx_ip)
    configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False,
                 kernel_buffers_count=args.kernel_buffers)

    rx_sdr = adi.Pluto("ip:" + rx_ip)
    configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz,
                 gain_mode=args.rx_gain_mode, gain=args.rx_gain,
                 kernel_buffers_count=args.kernel_buffers)
    rx_sdr.rx_buffer_size = rx_buf_size

    # ---------------------------------------------------------------------------
    # TUN bring-up
    # ---------------------------------------------------------------------------

    def _ip(*ip_args):
        subprocess.run(["ip", *ip_args], check=True)

    tun = TunDevice(name=args.tun_name, mtu=args.mtu)
    try:
        _ip("link", "set", "dev", args.tun_name, "mtu", str(args.mtu))
        _ip("addr", "add", f"{tun_ip}/24", "dev", args.tun_name)
        _ip("link", "set", args.tun_name, "up")
    except subprocess.CalledProcessError as e:
        tun.close()
        print(f"ERROR: failed to configure TUN {args.tun_name}: {e}", file=sys.stderr)
        print("       (need root + clean /24, e.g. no stale pluto0 from a prior run)", file=sys.stderr)
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # Streams + status UI
    # ---------------------------------------------------------------------------

    status   = LiveStatus(n_lines=2)
    tx_rate  = RateMeter()
    rx_rate  = RateMeter()
    _install_live_logging(status)

    rx_stats = {
        "data_rx_payload_bad": 0,  # decode-time payload failures (LDPC/CRC)
        "udp_bypass_tx":       0,  # RAW frames sent (UDP fast-path)
        "udp_bypass_rx":       0,  # RAW frames received and delivered to TUN
    }

    tx_stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size,
                         filler_amp=args.tx_filler_amp)
    rx_stream = RxStream(rx_sdr, maxsize=128, lossless=True)
    tx_stream.start()
    rx_stream.start(flush=16)

    radio_tx = RadioTx(tx_pipe, tx_stream, tx_rate, my_addr, peer_addr, rx_stats)
    radio_rx = RadioRx(rx_stream, rx_pipe, rx_rate, rx_stats, tun, my_addr)

    # Wrap the TUN in the bypass-demux unless the user explicitly opted out.
    # ARQ's internal TUN reader will then never see UDP packets — they get
    # diverted through ``radio_tx.send_raw`` and ride RAW frames on the wire.
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

    # ---------------------------------------------------------------------------
    # Status thread — pinned 2-line summary that mirrors to log every second.
    # ---------------------------------------------------------------------------

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

    # ---------------------------------------------------------------------------
    # Main loop — daemons do the work, Ctrl-C to exit.
    # ---------------------------------------------------------------------------

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters: stop TxStream first so any ARQ thread blocked inside
        # tx_stream.send() (queue full) returns immediately. Otherwise
        # arq.stop() would deadlock joining a thread stuck in send().
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
