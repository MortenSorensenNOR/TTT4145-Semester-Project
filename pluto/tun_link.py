"""TUN-mode radio link — Linux network interface over the SDR (no ARQ).

Per-node bridge between a Linux TUN device and the radio. Defaults to
full-duplex (one TX Pluto + one RX Pluto open simultaneously) but
``--mode tx`` and ``--mode rx`` enable half-duplex one-way operation
when only one Pluto is plugged in for that node — useful for the
2-radio A→B test where node A only has its TX Pluto and node B only
has its RX Pluto.

Threads (only those needed for the chosen mode start):

  * tun-rd : reads IP packets from TUN, hands them to the TX queue   (TX only)
  * tx     : pulls packets, builds Packet, transmits via TxStream    (TX only)
  * rx     : drains RxStream, decodes frames, writes payload to TUN  (RX only)

Default IP plan:
  --node A → TUN pluto0 = 10.0.0.1/24
  --node B → TUN pluto0 = 10.0.0.2/24

Usage:
    sudo .venv/bin/python -m pluto.tun_link --node A                 # full-duplex
    sudo .venv/bin/python -m pluto.tun_link --node A --mode tx       # TX-only
    sudo .venv/bin/python -m pluto.tun_link --node B --mode rx       # RX-only
"""

import argparse
import collections
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

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from modules.parallel_pipeline import RXWorkerPool, TXWorkerPool, apply_cpu_affinity, parse_cpu_spec
from modules.pulse_shaping.pulse_shaping import match_filter
from modules.tun import TunDevice
from pluto.config import (
    DAC_SCALE,
    configure_rx,
    configure_tx,
    get_node_freqs,
)
from pluto.setup_config import SETUP_PATH, load_or_die as load_setup
from pluto.sdr_stream import RxStream, TxStream
from pluto.live_status import (
    LiveStatus, RateMeter, _fmt_rate, _fmt_bytes, _install_live_logging,
)

import logging
# Timestamps so log files (oneway-A.log / oneway-B.log) can be correlated to
# sweep cells / wall-clock test events without having to instrument each call
# site individually. msec precision is plenty (TX buf turnover is ~tens of ms).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# FDD frequency plan: A transmits on the A→B channel and listens on the B→A
# channel; B does the opposite. The actual frequencies depend on whether
# --video is set (see pluto.config.get_node_freqs).

# 1-bit logical MAC per node — copied into Packet.src_mac / .dst_mac so the
# RX side can drop frames whose dst doesn't match (self-reception / leakage).
NODE_ADDR = {"A": 0, "B": 1}

DEFAULT_TUN_IP = {"A": "10.0.0.1", "B": "10.0.0.2"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node",     type=str,   default="A",    help="Node identity A or B; picks default TX/RX IPs from pluto/setup.json and TUN IP from DEFAULT_TUN_IP")
    parser.add_argument("--mode",     type=str,   default="both",
                        choices=("tx", "rx", "both"),
                        help="Half-duplex mode. 'tx': open only the TX Pluto, "
                             "send TUN packets out the radio, never receive. "
                             "'rx': open only the RX Pluto, write received "
                             "packets to TUN, never transmit. 'both' (default): "
                             "full-duplex, open both Plutos.")
    parser.add_argument("--gain",     type=float, default=-10,    help="TX gain in dB (default: -10)")
    parser.add_argument("--tx-ip",    type=str,   default=None,   help="Override TX Pluto IP (default: derived from --node via pluto/setup.json). Implies --backend ip for the TX side.")
    parser.add_argument("--rx-ip",    type=str,   default=None,   help="Override RX Pluto IP (default: derived from --node via pluto/setup.json). Implies --backend ip for the RX side.")
    parser.add_argument("--backend",  type=str,   default="ip", choices=("ip", "usb"),
                        help="libiio backend used to open the Plutos. 'ip' (default) opens "
                             "ip:<addr> from setup.json; 'usb' resolves the per-node "
                             "tx_serial/rx_serial to a usb:<bus.dev.intf> URI.")
    parser.add_argument("--tx-freq",  type=float, default=None,   help="TX center frequency in Hz (default: derived from --node and --video)")
    parser.add_argument("--rx-freq",  type=float, default=None,   help="RX center frequency in Hz (default: derived from --node and --video)")
    parser.add_argument("--video",    action="store_true",        help="Use the video-mode FDD pair (2327/2390 MHz) instead of the default network pair (2470/2475 MHz).")
    parser.add_argument("--cfo-offset", type=int, default=None,
                        help="Manual override for the RX-LO CFO correction in Hz. "
                             "Default: value from the cfo block of pluto/setup.json "
                             "for --node (run scripts/cfo_calibrate.py to generate "
                             "it), or 0 if no calibration is present.")
    parser.add_argument("--rx-gain-mode", type=str, default="manual",
                        choices=("slow_attack", "fast_attack", "hybrid", "manual"),
                        help="AD9361 RX AGC mode (default: manual). The auto "
                             "modes drift during the silence between bursts, "
                             "ramping gain up so the next packet clips the ADC "
                             "and the constellation widens 3–5×.")
    parser.add_argument("--rx-gain", type=float, default=50.0,
                        help="Fixed RX hardware gain in dB when "
                             "--rx-gain-mode=manual (default: 50, AD9361 range "
                             "~0–71). Ignored for any auto AGC mode.")
    parser.add_argument("--tx-buf-mult", type=float, default=1.05, help="TX buffer size as multiple of next-power-of-2 frame length")
    parser.add_argument("--rx-buf-mult", type=float, default=1.75, help="RX buffer size as multiple of next-power-of-2 frame length")
    parser.add_argument("--tx-filler-amp", type=float, default=0.0,
                        help="Per-component amplitude of complex Gaussian noise "
                             "filler emitted between packets (DAC-scale units). "
                             "0.0 = silent zero-fill (default, original behaviour). "
                             "Recommended ~512 (DAC_SCALE/32, ~30 dB below packet "
                             "peak): keeps RX AGC/Costas/NDA-TED loops engaged "
                             "during sparse traffic so the receiver doesn't have "
                             "to re-converge on every preamble.")
    parser.add_argument("--hardware-rrc", action="store_true",    help="Use the FPGA hardware RRC/4x interpolation path on TX (toggles the pluto_custom firmware GPIO).")
    parser.add_argument("--tun-name", type=str,   default="pluto0", help="TUN interface name (default: pluto0)")
    parser.add_argument("--tun-ip",   type=str,   default=None,   help="TUN IPv4 address with /24 implicit (default: 10.0.0.1 for A, 10.0.0.2 for B)")
    parser.add_argument("--mtu",      type=int,   default=1500,   help="TUN MTU in bytes (default: 1500)")
    parser.add_argument("--queue-depth", type=int, default=64,    help="TUN→TX queue depth before drops (default: 64)")
    parser.add_argument("--workers",     type=int, default=0,
                        help="Number of worker processes used for TX packet build / "
                             "RX packet decode. 0 = run inline on the TX/RX threads "
                             "(default). Useful on hybrid Intel CPUs where pinning "
                             "work into separate processes lets the OS scheduler "
                             "park us on P-cores instead of bouncing one Python "
                             "thread between P/E cores.")
    parser.add_argument("--rx-slots",    type=int, default=4,
                        help="Number of shared-memory slots in the RX worker ring "
                             "(default: 4). Only applies when --workers > 0; "
                             "more slots = more buffers in flight before back-pressure.")
    parser.add_argument("--mp-start",    type=str, default=None,
                        choices=("spawn", "fork", "forkserver"),
                        help="multiprocessing start method for the worker pools. "
                             "Default: 'spawn' (safest). Override via "
                             "RADIO_MP_START_METHOD or this flag.")
    parser.add_argument("--worker-cpus", type=str, default="0",
                        help="CPU IDs to pin the main process AND workers to. "
                             "Linux only — uses os.sched_setaffinity(). Forms: "
                             "comma list ('0,1,2,3'), range ('0-3'), mix "
                             "('0-3,8'), the keyword 'p-cores' to auto-detect "
                             "Intel hybrid P-cores via "
                             "/sys/devices/cpu_core/cpus, or empty string '' "
                             "to disable pinning. Default: '0' — CPU 0 is "
                             "always a P-core on Intel hybrid systems and a "
                             "regular core elsewhere, so this gives "
                             "consistent single-thread performance "
                             "regardless of --workers. When --workers > 1, "
                             "override this with e.g. '0-3' or '0,2,4,6' to "
                             "give each worker its own physical P-core.")
    args = parser.parse_args()

    # Resolve the CPU set early — apply to the main process now, then later
    # forward the same set to every worker so the entire program runs on the
    # intended cores. Empty string disables pinning.
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
        print(f"ERROR: --node must be 'A' or 'B' for the TUN-link bridge, got '{args.node}'")
        sys.exit(1)

    do_tx = args.mode in ("tx", "both")
    do_rx = args.mode in ("rx", "both")

    # IP overrides force ip: for that side; otherwise --backend picks the URI.
    tx_uri = (
        (f"ip:{args.tx_ip}" if args.tx_ip else setup.tx_uri(args.node, args.backend))
        if do_tx else None
    )
    rx_uri = (
        (f"ip:{args.rx_ip}" if args.rx_ip else setup.rx_uri(args.node, args.backend))
        if do_rx else None
    )

    # Resolve RX-LO CFO offset: manual CLI override wins; otherwise pull the
    # measured value for this node from the calibration; otherwise 0.
    # Skipped entirely in TX-only mode (no RX LO to correct).
    rx_cfo_hz = 0
    cfo_src   = "n/a"
    if do_rx:
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

    peer       = "B" if args.node == "A" else "A"
    my_addr    = NODE_ADDR[args.node]
    peer_addr  = NODE_ADDR[peer]
    tun_ip     = args.tun_ip or DEFAULT_TUN_IP[args.node]
    peer_tun_ip = DEFAULT_TUN_IP[peer]

    # ---------------------------------------------------------------------------
    # Pipelines
    # ---------------------------------------------------------------------------

    pipe_cfg = PipelineConfig()
    tx_pipe = TXPipeline(pipe_cfg)
    rx_pipe = RXPipeline(pipe_cfg) if do_rx else None

    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------------------
    # Buffer sizing — probe one MTU-sized packet to learn frame_len.
    #
    # Run BEFORE opening the SDR so that the slow spawn-pool startup that
    # follows doesn't sit between configure_rx() and the first stream.refill()
    # — libiio's first-refill timeout is short enough that pool startup can
    # exceed it on slower laptops, killing the connection.
    # ---------------------------------------------------------------------------

    _probe_bits    = rng.integers(0, 2, args.mtu * 8, dtype=np.uint8)
    _probe_pkt     = Packet(src_mac=my_addr, dst_mac=peer_addr, type=0, seq_num=0,
                            length=args.mtu, payload=_probe_bits)
    _probe_samples = tx_pipe.transmit(_probe_pkt)
    frame_len      = len(_probe_samples)
    rx_buf_size    = int(args.rx_buf_mult * int(2 ** np.ceil(np.log2(frame_len))))
    tx_buf_size    = int(args.tx_buf_mult * int(2 ** np.ceil(np.log2(frame_len))))

    node_freqs = get_node_freqs(args.node, video=args.video)
    tx_freq = int(args.tx_freq) if args.tx_freq is not None else node_freqs["tx"]
    rx_freq = int(args.rx_freq) if args.rx_freq is not None else node_freqs["rx"]

    print(f"Node      : {args.node}  (peer {peer})  mode={args.mode}  freq={'video' if args.video else 'network'}")
    print(f"TUN       : {args.tun_name} = {tun_ip}/24  (peer {peer_tun_ip})  MTU {args.mtu}")
    if do_tx:
        print(f"TX radio  : {tx_uri}   @ {tx_freq / 1e6:.3f} MHz")
    else:
        print(f"TX radio  : disabled (--mode {args.mode})")
    if do_rx:
        if args.rx_gain_mode == "manual":
            rx_gain_desc = f"manual {args.rx_gain:.1f} dB"
        else:
            rx_gain_desc = f"AGC={args.rx_gain_mode}"
        print(f"RX radio  : {rx_uri}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
              f"(CFO {rx_cfo_hz:+d} Hz, {cfo_src}; {rx_gain_desc})")
    else:
        print(f"RX radio  : disabled (--mode {args.mode})")
    print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
    print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    if do_tx:
        print(f"TX buf    : {tx_buf_size} samples  ({tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    if do_rx:
        print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    if do_tx:
        print(f"TX gain   : {args.gain} dB")

    # ---------------------------------------------------------------------------
    # Worker pools — only created when --workers > 0.
    #
    # Created BEFORE opening the SDR (see comment above) so the spawn startup
    # latency lands while the radio is still idle.
    # ---------------------------------------------------------------------------

    tx_pool: TXWorkerPool | None = None
    rx_pool: RXWorkerPool | None = None

    if args.workers > 0:
        rx_slot_samples = 2 * rx_buf_size + 1024
        import atexit as _atexit
        if do_tx:
            tx_pool = TXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   start_method=args.mp_start,
                                   cpu_set=worker_cpus)
            _atexit.register(tx_pool.shutdown)
        if do_rx:
            rx_pool = RXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   slot_samples=rx_slot_samples,
                                   n_slots=args.rx_slots,
                                   start_method=args.mp_start,
                                   cpu_set=worker_cpus)
            _atexit.register(rx_pool.shutdown)
        pools = [name for name, on in (("TX", do_tx), ("RX", do_rx)) if on]
        print(f"Workers   : {args.workers}  ({'+'.join(pools)} pool"
              f"{'s' if len(pools) > 1 else ''}, "
              f"rx_slot={rx_slot_samples} samples × {args.rx_slots} slots, "
              f"start={args.mp_start or 'spawn'})")
    else:
        print("Workers   : 0  (inline TX build / RX decode on the threads)")
    print(f"CPU pin   : {worker_cpus if worker_cpus else 'off'}  "
          f"(applied to main process; also forwarded to workers)")
    print()

    # ---------------------------------------------------------------------------
    # SDR setup — always full-duplex (TX and RX Pluto open simultaneously).
    #
    # Done AFTER pool creation so the libiio connection isn't sitting idle
    # while spawn workers boot up.
    # ---------------------------------------------------------------------------

    tx_sdr = None
    rx_sdr = None
    if do_tx:
        tx_sdr = adi.Pluto(tx_uri)
        configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False)
    if do_rx:
        rx_sdr = adi.Pluto(rx_uri)
        configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz,
                     gain_mode=args.rx_gain_mode, gain=args.rx_gain)
        rx_sdr.rx_buffer_size = rx_buf_size

    # ---------------------------------------------------------------------------
    # TUN bring-up — open device, then assign address and bring the link up.
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
    # Threads
    # ---------------------------------------------------------------------------

    send_q: queue.Queue = queue.Queue(maxsize=args.queue_depth)
    stop_event = threading.Event()
    n_status_lines = (1 if do_tx else 0) + (1 if do_rx else 0)
    status = LiveStatus(n_lines=n_status_lines)
    tx_rate = RateMeter()
    rx_rate = RateMeter()
    _install_live_logging(status)
    # Status line indices depend on which threads are running.
    tx_line = 0 if do_tx else None
    rx_line = (1 if do_tx else 0) if do_rx else None

    stats = {
        "tun_in":      0,  # IP packets pulled from TUN
        "tun_out":     0,  # IP packets written back to TUN
        "tun_dropped": 0,  # TUN reads dropped because send_q was full
        "data_rx_ok":      0,  # valid frames received and accepted (header + payload OK, addressed to us)
        "data_rx_foreign": 0,  # valid frames whose dst_mac wasn't ours
        "data_rx_header_bad":  0,  # frames returned with valid=False (header CRC failed)
        "data_rx_payload_bad": 0,  # detections where header decoded but payload raised (CRC-16 mismatch / LDPC fail) — these never appear in the returned packets list, counted via rx_pipe.last_payload_failures
    }


    def tun_reader_thread():
        while not stop_event.is_set():
            try:
                data = tun.read()
            except OSError:
                return
            if data is None:
                continue
            stats["tun_in"] += 1
            try:
                send_q.put(data, timeout=0.05)
            except queue.Full:
                stats["tun_dropped"] += 1


    # Last-time-logged for the periodic [TX]/[RX] log mirror. Single-element
    # lists so the closures can mutate without `nonlocal` gymnastics.
    _last_tx_log_t = [0.0]
    _last_rx_log_t = [0.0]
    _LOG_INTERVAL_S = 1.0  # how often the [TX]/[RX] line is mirrored to logger.info

    def _tx_status(stream: TxStream) -> None:
        msg = (f"[TX] in={stats['tun_in']:>8d}  drop={stats['tun_dropped']:>4d}  "
               f"pending={stream.pending:>3d}  "
               f"rate={_fmt_rate(tx_rate.rate_bps)}  "
               f"avg={_fmt_rate(tx_rate.avg_bps)}  "
               f"total={_fmt_bytes(tx_rate.total_bytes)}")
        status.set(tx_line, "  " + msg)
        # Mirror to logger ~once/second so oneway-A.log captures stats per
        # second — invaluable for correlating sweep cells with TX behaviour.
        now = time.monotonic()
        if now - _last_tx_log_t[0] >= _LOG_INTERVAL_S:
            logger.info(msg)
            _last_tx_log_t[0] = now


    def _tx_build_inline(payload: bytes) -> np.ndarray:
        """In-thread packet build (no MP). Returns DAC-scaled complex64 samples."""
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pkt = Packet(src_mac=my_addr, dst_mac=peer_addr, type=0, seq_num=0,
                     length=len(payload), payload=bits)
        samples = tx_pipe.transmit(pkt)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        return (samples * DAC_SCALE).astype(np.complex64)


    def tx_thread_fn():
        stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size,
                          filler_amp=args.tx_filler_amp)
        stream.start()

        if tx_pool is None:
            # Inline path — original behaviour, no IPC.
            try:
                while not stop_event.is_set():
                    try:
                        payload = send_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    stream.send(_tx_build_inline(payload))
                    tx_rate.add(len(payload))
                    _tx_status(stream)
            finally:
                stream.stop()
            return

        # MP path — keep ~2× n_workers packet builds in flight so the pool stays
        # saturated and stream.send() pulls a finished sample-array on every
        # iteration rather than waiting on a single worker round-trip.
        inflight: collections.deque = collections.deque()
        inflight_depth = max(2, tx_pool.n_workers * 2)

        try:
            while not stop_event.is_set():
                # Top up by submitting any payloads currently waiting in send_q.
                # Non-blocking pulls so we don't stall when the queue is shallow.
                while len(inflight) < inflight_depth:
                    try:
                        payload = send_q.get_nowait()
                    except queue.Empty:
                        break
                    ar = tx_pool.submit(bytes(payload), my_addr, peer_addr,
                                        0, 0, float(DAC_SCALE))
                    inflight.append((ar, len(payload)))

                if inflight:
                    ar, payload_len = inflight.popleft()
                    try:
                        samples = ar.get(timeout=5.0)
                    except Exception as e:
                        import logging as _logging
                        _logging.warning(f"[tx_thread] worker raised: {e!r}")
                        continue
                    stream.send(samples)
                    tx_rate.add(payload_len)
                    _tx_status(stream)
                else:
                    # Block briefly when there's nothing to do.
                    try:
                        payload = send_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    ar = tx_pool.submit(bytes(payload), my_addr, peer_addr,
                                        0, 0, float(DAC_SCALE))
                    inflight.append((ar, len(payload)))
        finally:
            stream.stop()


    def _rx_status(stream: RxStream) -> None:
        msg = (f"[RX] ok={stats['data_rx_ok']:>8d}  "
               f"hdr_bad={stats['data_rx_header_bad']:>4d}  "
               f"pay_bad={stats['data_rx_payload_bad']:>4d}  "
               f"foreign={stats['data_rx_foreign']:>4d}  "
               f"q={stream._q.qsize():>3d}/{stream._q.maxsize}  "
               f"rate={_fmt_rate(rx_rate.rate_bps)}  "
               f"avg={_fmt_rate(rx_rate.avg_bps)}  "
               f"total={_fmt_bytes(rx_rate.total_bytes)}")
        status.set(rx_line, "  " + msg)
        now = time.monotonic()
        if now - _last_rx_log_t[0] >= _LOG_INTERVAL_S:
            logger.info(msg)
            _last_rx_log_t[0] = now


    def _rx_handle_packet(*, valid: bool, dst_mac: int, length: int,
                          payload_bytes: bytes) -> bool:
        """Apply MAC/validity filtering, write payload to TUN, update stats.

        Returns True normally, False if the TUN write OSError'd (caller should exit).
        """
        if not valid:
            stats["data_rx_header_bad"] += 1
            return True
        if dst_mac != my_addr:
            stats["data_rx_foreign"] += 1
            return True
        try:
            tun.write(payload_bytes)
        except OSError:
            return False
        stats["data_rx_ok"] += 1
        stats["tun_out"]    += 1
        rx_rate.add(length)
        return True


    def rx_thread_fn():
        stream = RxStream(rx_sdr, maxsize=128, lossless=True)
        stream.start(flush=16)

        if rx_pool is None:
            # Inline path — single-process receive loop, original behaviour.
            prev_buf = None
            search_from = 0
            try:
                while not stop_event.is_set():
                    try:
                        curr_buf = stream.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    prev_len = len(prev_buf) if prev_buf is not None else 0
                    raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

                    packets, max_det = rx_pipe.receive(raw, search_from=search_from)
                    stats["data_rx_payload_bad"] += rx_pipe.last_payload_failures

                    prev_buf = curr_buf
                    if packets:
                        last_ps = max(p.sample_start for p in packets)
                        search_from = max(0, max(last_ps, max_det) - prev_len)
                    else:
                        search_from = max(0, max_det - prev_len)

                    for pkt in packets:
                        payload_bytes = (
                            np.packbits(pkt.payload[:pkt.length * 8].astype(np.uint8)).tobytes()
                            if pkt.length > 0 else b""
                        )
                        if not _rx_handle_packet(valid=pkt.valid, dst_mac=pkt.dst_mac,
                                                 length=pkt.length, payload_bytes=payload_bytes):
                            return
                    _rx_status(stream)
            finally:
                stream.stop()
            return

        # MP path — match_filter + detect on this thread (cheap, single-threaded
        # path), decode in worker pool. Per-buffer parallelism: every detection
        # within a single rx buffer fans out across workers, but consecutive
        # buffers stay serialized so we can preserve search_from / tail-cutoff
        # semantics exactly.
        prev_buf = None
        search_from = 0
        try:
            while not stop_event.is_set():
                try:
                    curr_buf = stream.get(timeout=0.05)
                except queue.Empty:
                    continue

                prev_len = len(prev_buf) if prev_buf is not None else 0
                raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

                search_buf = raw[search_from:]
                filtered   = match_filter(search_buf, rx_pipe.rrc_taps)
                detections = rx_pipe.detect(filtered)

                prev_buf = curr_buf

                if not detections:
                    search_from = max(0, search_from - prev_len)
                    _rx_status(stream)
                    continue

                # Hand the post-match-filter buffer to the pool. abs_offsets are
                # detection.payload_start measured from the start of the *raw*
                # buffer (raw[search_from:] is what we filtered).
                sub = rx_pool.submit_buffer(filtered, detections, search_from_abs=search_from)

                packets_decoded: list = []   # (RXResult, abs_payload_start) tuples
                payload_failures = 0
                tail_cut         = False
                max_det          = search_from

                for ar, abs_payload_start in zip(sub.futures, sub.abs_offsets):
                    try:
                        result = ar.get(timeout=10.0)
                    except Exception as e:
                        import logging as _logging
                        _logging.warning(f"[rx_thread] worker raised: {e!r}")
                        payload_failures += 1
                        max_det = max(max_det, abs_payload_start)
                        continue

                    # Mirror the break-on-tail-cutoff semantics of RXPipeline.receive:
                    # once we see a tail cutoff, discard subsequent results from
                    # this buffer so they remain eligible for re-detection on the
                    # next iteration (with more data appended).
                    if tail_cut:
                        continue

                    if result.status == "ok":
                        packets_decoded.append((result, abs_payload_start))
                        max_det = max(max_det, abs_payload_start)
                    elif result.status == "tail_cutoff":
                        tail_cut = True
                    else:
                        # decode_error (header CRC fail, payload CRC/LDPC fail, ...)
                        payload_failures += 1
                        max_det = max(max_det, abs_payload_start)

                stats["data_rx_payload_bad"] += payload_failures

                if packets_decoded:
                    last_ps = max(abs_off for _, abs_off in packets_decoded)
                    search_from = max(0, max(last_ps, max_det) - prev_len)
                else:
                    search_from = max(0, max_det - prev_len)

                for result, _abs in packets_decoded:
                    if not _rx_handle_packet(valid=result.valid, dst_mac=result.dst_mac,
                                             length=result.length,
                                             payload_bytes=result.payload_bytes):
                        return
                _rx_status(stream)
        finally:
            stream.stop()


    threads: list[threading.Thread] = []
    if do_rx:
        t_rx = threading.Thread(target=rx_thread_fn, name="rx", daemon=True)
        t_rx.start()
        threads.append(t_rx)
    if do_tx:
        t_tx = threading.Thread(target=tx_thread_fn, name="tx", daemon=True)
        t_tx.start()
        threads.append(t_tx)
        t_tun = threading.Thread(target=tun_reader_thread, name="tun-rd", daemon=True)
        t_tun.start()
        threads.append(t_tun)

    # ---------------------------------------------------------------------------
    # Idle in main thread; daemons do the work. Ctrl-C exits.
    # ---------------------------------------------------------------------------

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        status.stop()
        for t in threads:
            t.join(timeout=2.0)

        print()
        print("=" * 50)
        print(f"FINAL STATS  (mode={args.mode})")
        print("=" * 50)
        if do_tx:
            print(f"TUN in       : {stats['tun_in']}")
            print(f"TUN dropped  : {stats['tun_dropped']}  (send queue full)")
            print(f"TX bytes     : {tx_rate.total_bytes}  (avg {_fmt_rate(tx_rate.avg_bps)})")
        if do_rx:
            print(f"TUN out      : {stats['tun_out']}")
            print(f"RX valid     : {stats['data_rx_ok']}")
            print(f"RX hdr bad   : {stats['data_rx_header_bad']}   (header CRC failed)")
            print(f"RX pay bad   : {stats['data_rx_payload_bad']}   (header OK, payload CRC/LDPC failed)")
            print(f"RX foreign   : {stats['data_rx_foreign']}  (dst_mac != us)")
            print(f"RX bytes     : {rx_rate.total_bytes}  (avg {_fmt_rate(rx_rate.avg_bps)})")
        print("=" * 50)

        subprocess.run(["ip", "link", "set", args.tun_name, "down"], check=False)
        tun.close()
        if tx_sdr is not None:
            try:
                tx_sdr.tx_destroy_buffer()
            except Exception:
                pass
            del tx_sdr
        if rx_sdr is not None:
            del rx_sdr

        if tx_pool is not None:
            tx_pool.shutdown()
        if rx_pool is not None:
            rx_pool.shutdown()

    sys.exit(0)
