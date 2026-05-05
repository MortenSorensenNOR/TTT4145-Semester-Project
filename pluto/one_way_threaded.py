"""Realistic threaded loopback test.

Spins up two threads — TX and RX — that run independently, mimicking a real
intermittent transmitter and a continuously listening receiver.

TX thread:
  - Continuously streams fixed-size TX buffers via TxStream
  - Builds packets with incrementing sequence numbers and queues them
  - TxStream packer thread greedily packs queued packets into each buffer
  - When queue is empty, silence (zeros) is transmitted to keep stream alive
  - Supports variable-length packets with --variable flag

RX thread:
  - Drains the SDR RX buffer via RxStream (lossless, large queue)
  - Searches each capture for decodable frames
  - Tracks sequence numbers and reports packet drop rate at the end

Usage:
    python pluto/one_way_threaded.py [options]

Options:
    --gain         TX hardware gain in dB        (default: -30)
    --payload      Payload size in bytes         (default: 10)
    --packets      Number of packets per burst   (default: 20)
    --interval     Inter-burst gap in ms         (default: 200)
    --node         Node identity (A or B)        (default: A)
    --tx-ip        Override TX Pluto IP          (default: derived from --node)
    --rx-ip        Override RX Pluto IP          (default: derived from --node)
    --mode         Operation mode: 'tx', 'rx', or 'both' (default: 'both')
    --variable     Randomize payload size per packet
    --min-payload  Min payload bytes (variable)  (default: 4)
    --tx-buf-mult  TX buf multiplier             (default: 8)

Radio convention: each node runs a dedicated TX Pluto and a dedicated RX
Pluto (a single USB-2 Pluto cannot sustain 4 Msps full-duplex). The per-
node TX/RX IP assignment is loaded from pluto/setup.json — see
pluto.setup_config for the schema.
"""

import argparse
import collections
import queue
import sys
import threading
import time

# Force line-buffered stdout so prints are visible even if the process is killed.
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from modules.parallel_pipeline import RXWorkerPool, TXWorkerPool, apply_cpu_affinity, parse_cpu_spec
from modules.pulse_shaping.pulse_shaping import match_filter
from pluto.config import (
    DAC_SCALE,
    PIPELINE,
    configure_rx,
    configure_tx,
    get_node_freqs,
)
from pluto.setup_config import SETUP_PATH, load_or_die as load_setup

# FDD frequency plan: A transmits on the A→B channel and listens on the B→A
# channel; B does the opposite, so a node-A and node-B process can run
# simultaneously without colliding on one channel. The actual frequencies
# depend on whether --video is set (see pluto.config.get_node_freqs).
from pluto.sdr_stream import RxStream, TxStream
from pluto.rx_agc import RxAGC
from pluto.live_status import (
    LiveStatus, RateMeter, _fmt_rate, _fmt_bytes, _install_live_logging,
)

import logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gain",     type=float, default=-10,           help="TX gain in dB (default: -10)")
    parser.add_argument("--payload",  type=int,   default=1000,          help="Payload bytes (default: 1000)")
    parser.add_argument("--packets",  type=int,   default=20,            help="Number of packets per TX burst (default: 20)")
    parser.add_argument("--interval", type=float, default=0,             help="Inter-burst gap in ms (default: 0)")
    parser.add_argument("--node",     type=str,   default="A",           help="Node identity A or B; picks default TX/RX IPs from pluto/setup.json")
    parser.add_argument("--tx-ip",    type=str,   default=None,          help="Override TX Pluto IP (default: derived from --node). Implies --backend ip for the TX side.")
    parser.add_argument("--rx-ip",    type=str,   default=None,          help="Override RX Pluto IP (default: derived from --node). Implies --backend ip for the RX side.")
    parser.add_argument("--backend",  type=str,   default="ip", choices=("ip", "usb"),
                        help="libiio backend used to open the Plutos. 'ip' (default) opens "
                             "ip:<addr> from setup.json; 'usb' resolves the per-node "
                             "tx_serial/rx_serial to a usb:<bus.dev.intf> URI. Use 'usb' "
                             "when running both Plutos directly off the host's USB ports.")
    parser.add_argument("--mode",     type=str,   default="both",        help="Mode: 'tx', 'rx', or 'both' (default: both)")
    parser.add_argument("--cfo-offset", type=int, default=None,
                        help="Manual override for the RX-LO CFO correction in Hz. "
                             "Default: value from the cfo block of pluto/setup.json "
                             "for --node (run scripts/cfo_calibrate.py to generate "
                             "it), or 0 if no calibration is present. Only affects RX.")
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
    parser.add_argument("--rx-agc", type=str, default="off",
                        choices=("auto", "off"),
                        help="Software AGC for --rx-gain-mode=manual. "
                             "'off' (default) leaves --rx-gain fixed. "
                             "'auto' watches the peak amplitude of every RX "
                             "buffer that produced a valid decode and nudges "
                             "rx_hardwaregain toward a target below ADC clip; "
                             "idle buffers are ignored so gain doesn't ramp up "
                             "during silence. No effect when --rx-gain-mode is "
                             "not 'manual' (the AD9361 is already running its "
                             "own AGC).")
    parser.add_argument("--tx-freq", type=float, default=None, help="TX center frequency in Hz (default: derived from --node and --video)")
    parser.add_argument("--rx-freq", type=float, default=None, help="RX center frequency in Hz (default: derived from --node and --video)")
    parser.add_argument("--video",   action="store_true",       help="Use the video-mode FDD pair (2327/2390 MHz) instead of the default network pair (2470/2475 MHz).")
    parser.add_argument("--constellation", action="store_true", help="Collect post-Costas PSK8 symbols and refresh a constellation plot. Live X11 by default; falls back to PNG-on-disk if X11 is unreachable (sudo+netns strips DISPLAY auth). Use --constellation-save to force the PNG path.")
    parser.add_argument("--constellation-save", type=str, default=None,
                        help="Force-save the constellation to this PNG path "
                             "instead of opening an X11 window. Refreshed "
                             "every few packets — view with `feh -R 1 PATH` "
                             "or `eog PATH` for live updates. Implies "
                             "--constellation.")
    parser.add_argument("--constellation-csv", type=str, default=None,
                        help="Stream post-Costas PSK8 symbols to this CSV "
                             "file (columns: seq,I,Q — one row per symbol, "
                             "seq is the per-packet 32-bit counter). "
                             "Independent of --constellation; use this for "
                             "offline plotting without matplotlib.")
    parser.add_argument("--variable", action="store_true", help="Randomize payload size per packet (between --min-payload and --payload)")
    parser.add_argument("--min-payload", type=int, default=4, help="Minimum payload bytes when --variable is set (default: 4, must hold seq number)")
    parser.add_argument("--tx-buf-mult", type=float, default=1.05, help="TX buffer size as multiple of next-power-of-2 frame length")
    parser.add_argument("--rx-buf-mult", type=float, default=1.75, help="RX buffer size as multiple of next-power-of-2 frame length")
    parser.add_argument("--tx-filler-amp", type=float, default=0.0,
                        help="Per-component amplitude of complex Gaussian noise filler emitted "
                             "between packets (DAC-scale units). 0.0 = silent zero-fill (default, "
                             "original behaviour). Recommended ~512 (DAC_SCALE/32, ~30 dB below "
                             "packet peak): keeps RX AGC/Costas/NDA-TED loops engaged during "
                             "sparse traffic so the receiver doesn't have to re-converge on "
                             "every preamble.")
    parser.add_argument("--hardware-rrc", action="store_true", help="Use the FPGA hardware RRC/4x interpolation path on TX (toggles the pluto_custom firmware GPIO). TX only — RX always uses software match filter.")
    parser.add_argument("--save-rx-buf", type=str, default=None,
                        help="Directory to dump raw RX buffers (.npz) for offline replay by "
                             "scripts/sweep_*_params.py. RX mode only.")
    parser.add_argument("--save-n",      type=int, default=4,
                        help="How many RX buffers to dump (default: 4). Only buffers that "
                             "produced ≥1 detected packet are saved.")
    parser.add_argument("--save-skip",   type=int, default=0,
                        help="Number of initial RX buffers to skip before dumping starts "
                             "(default: 0). Lets the Pluto's AGC/LO/clock warm up so the "
                             "saved captures are representative of steady-state operation.")
    parser.add_argument("--workers",     type=int, default=0,
                        help="Number of worker processes used for TX packet build / "
                             "RX packet decode. 0 = run inline on the TX/RX threads "
                             "(default). Useful on hybrid Intel CPUs where pinning "
                             "work into separate processes lets the OS scheduler "
                             "park us on P-cores instead of bouncing one Python "
                             "thread between P/E cores.")
    parser.add_argument("--rx-slots",    type=int, default=4,
                        help="Number of shared-memory slots in the RX worker ring "
                             "(default: 4). Only applies when --workers > 0.")
    parser.add_argument("--mp-start",    type=str, default=None,
                        choices=("spawn", "fork", "forkserver"),
                        help="multiprocessing start method for the worker pools "
                             "(default: 'spawn').")
    parser.add_argument("--worker-cpus", type=str, default="0",
                        help="Pin the main process AND workers to these CPU "
                             "IDs (Linux only). Forms: '0,1,2,3', '0-3', "
                             "'0-3,8', the keyword 'p-cores', or empty "
                             "string '' to disable. Default: '0' — CPU 0 is "
                             "always a P-core on Intel hybrid systems and a "
                             "regular core elsewhere, so this gives "
                             "consistent single-thread performance even "
                             "when --workers 0. Override with '0-3' or "
                             "'0,2,4,6' when scaling up.")
    parser.add_argument("--profile", action="store_true",
                        help="In --mode tx, log rolling build / send timings "
                             "every 100 packets so you can tell whether the "
                             "bottleneck is tx_pipe.transmit (CPU) or "
                             "stream.send / DMA push (radio).")
    args = parser.parse_args()

    # Resolve the CPU set early — apply to the main process now so the inline
    # path benefits, and forward the same set to every worker later.
    if args.worker_cpus == "":
        worker_cpus: list[int] | None = None
    else:
        worker_cpus = parse_cpu_spec(args.worker_cpus)
    apply_cpu_affinity(worker_cpus)

    if args.mode not in ("tx", "rx", "both"):
        print(f"ERROR: --mode must be 'tx', 'rx', or 'both', got '{args.mode}'")
        sys.exit(1)

    setup = load_setup()
    if args.node not in setup.nodes:
        print(f"ERROR: --node must be one of {sorted(setup.nodes)}, got '{args.node}'")
        sys.exit(1)

    # IP overrides force the ip: backend for that side; otherwise the global
    # --backend setting picks ip:<addr> or usb:<bus.dev.intf> via serial. Only
    # resolve the side(s) we actually intend to open — usb resolution scans
    # libiio and would fail in tx-only / rx-only modes if the unused Pluto
    # isn't plugged in.
    tx_uri = None
    rx_uri = None
    if args.mode in ("tx", "both"):
        tx_uri = f"ip:{args.tx_ip}" if args.tx_ip else setup.tx_uri(args.node, args.backend)
    if args.mode in ("rx", "both"):
        rx_uri = f"ip:{args.rx_ip}" if args.rx_ip else setup.rx_uri(args.node, args.backend)

    # Resolve RX-LO CFO offset: manual CLI override wins; otherwise pull the
    # measured value for this node from the calibration; otherwise 0. TX always
    # emits at its natural LO (split-radio convention — the peer's RX does the
    # compensating).
    if args.cfo_offset is not None:
        rx_cfo_hz = args.cfo_offset
        cfo_src   = "cli"
    elif args.mode == "tx":
        rx_cfo_hz = 0
        cfo_src   = "n/a (tx-only)"
    elif setup.cfo is None:
        rx_cfo_hz = 0
        cfo_src   = "unset"
        print(f"  [warn] no CFO calibration in {SETUP_PATH} — using 0 Hz. "
              f"Run 'uv run python scripts/cfo_calibrate.py' to generate one.")
    else:
        rx_cfo_hz = setup.cfo.rx_offset_for(args.node)
        cfo_src   = f"calibration ({setup.cfo.measured_at or 'unknown date'})"

    # ---------------------------------------------------------------------------
    # Pipelines
    # ---------------------------------------------------------------------------

    pipe_cfg = PipelineConfig(hardware_rrc=args.hardware_rrc)
    # Over coax, real-packet fine peak ratios sit at ~10–12 while spurious
    # detections on the ~7 kB of silence at the tail of each TX buffer fire with
    # ratio ~4–5. Bumping the gate to 7 cleanly rejects the latter — otherwise
    # they still pass the default 3.0 gate, fail at header decode, and advance
    # search_from past the real packet that followed them.
    pipe_cfg.SYNC_CONFIG.fine_peak_ratio_min = np.float32(7.0)
    tx_pipe  = TXPipeline(pipe_cfg)
    rx_pipe  = RXPipeline(pipe_cfg)

    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------------------
    # Buffer sizing — probe one packet to learn frame_len.
    #
    # Run BEFORE opening the SDR so that the slow spawn-pool startup that
    # follows doesn't sit between configure_rx() and stream.start() — libiio's
    # first-refill timeout is short enough that pool startup (esp. on hybrid
    # Intel laptops where the spawn imports get scheduled onto E-cores) can
    # exceed it and kill the connection.
    # ---------------------------------------------------------------------------

    _probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
    _probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
    _probe_samples = tx_pipe.transmit(_probe_pkt)
    frame_len      = len(_probe_samples)
    rx_buf_size    = int(args.rx_buf_mult * int(2 ** np.ceil(np.log2(frame_len))))
    tx_buf_size    = int(args.tx_buf_mult * int(2 ** np.ceil(np.log2(frame_len))))

    node_freqs = get_node_freqs(args.node, video=args.video)
    tx_freq = int(args.tx_freq) if args.tx_freq is not None else node_freqs["tx"]
    rx_freq = int(args.rx_freq) if args.rx_freq is not None else node_freqs["rx"]

    print(f"Mode      : {args.mode}  freq={'video' if args.video else 'network'}")
    print(f"Node      : {args.node}")
    if args.mode in ("tx", "both"):
        print(f"TX radio  : {tx_uri}   @ {tx_freq / 1e6:.3f} MHz")
    if args.mode in ("rx", "both"):
        print(f"RX radio  : {rx_uri}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
              f"(CFO {rx_cfo_hz:+d} Hz, {cfo_src})")
    print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
    if args.variable:
        print(f"Payload   : {args.min_payload}–{args.payload} bytes (variable)")
    else:
        print(f"Payload   : {args.payload} bytes  ({args.payload * 8} bits)")
    print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    if args.mode in ("tx", "both"):
        print(f"TX buf    : {tx_buf_size} samples  ({tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    if args.mode in ("rx", "both"):
        print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
    print(f"Gap       : {args.interval} ms")
    print(f"Packets   : {args.packets} per burst")
    print(f"TX gain   : {args.gain} dB")

    # ---------------------------------------------------------------------------
    # Worker pools — only created when --workers > 0. Spun up BEFORE the SDR
    # so spawn import latency lands while the radio is still idle.
    # ---------------------------------------------------------------------------

    tx_pool: TXWorkerPool | None = None
    rx_pool: RXWorkerPool | None = None

    if args.workers > 0:
        if args.mode in ("tx", "both"):
            tx_pool = TXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   start_method=args.mp_start,
                                   cpu_set=worker_cpus)
        if args.mode in ("rx", "both"):
            rx_slot_samples = 2 * rx_buf_size + 1024
            rx_pool = RXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   slot_samples=rx_slot_samples,
                                   n_slots=args.rx_slots,
                                   start_method=args.mp_start,
                                   cpu_set=worker_cpus)
            print(f"RX slots  : {args.rx_slots} × {rx_slot_samples} samples")
        # Atexit so the shared-memory ring gets unlinked even when the SDR
        # raises (libiio TimeoutError, USB unplug, etc.) and skips the
        # finally-cleanup at the bottom of the script.
        import atexit as _atexit
        if tx_pool is not None:
            _atexit.register(tx_pool.shutdown)
        if rx_pool is not None:
            _atexit.register(rx_pool.shutdown)
        print(f"Workers   : {args.workers}  (start={args.mp_start or 'spawn'})")
    else:
        print("Workers   : 0  (inline TX build / RX decode on the threads)")
    print(f"CPU pin   : {worker_cpus if worker_cpus else 'off'}  "
          f"(applied to main process; also forwarded to workers)")
    print()

    # ---------------------------------------------------------------------------
    # SDR setup — per direction. Each node runs a dedicated TX Pluto and a
    # dedicated RX Pluto; we only open the device(s) that the current mode
    # actually needs, so tx/rx-only invocations don't require both radios to
    # be plugged in.
    # ---------------------------------------------------------------------------

    tx_sdr = None
    rx_sdr = None

    if args.mode in ("tx", "both"):
        tx_sdr = adi.Pluto(tx_uri)
        configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False)

    if args.mode in ("rx", "both"):
        rx_sdr = adi.Pluto(rx_uri)
        configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz,
                     gain_mode=args.rx_gain_mode, gain=args.rx_gain)
        rx_sdr.rx_buffer_size = rx_buf_size

    # ---------------------------------------------------------------------------
    # TX helpers
    # ---------------------------------------------------------------------------

    def _make_payload_bytes(seq: int, payload_bytes: int) -> bytes:
        """Build the per-packet payload as bytes: 32-bit big-endian seq + random pad.

        Identical layout to the prior bit-level builder once unpacked at the
        modulator (np.unpackbits is MSB-first within each byte).
        """
        seq = seq % (2**32)
        if payload_bytes < 4:
            raise ValueError("payload must be at least 4 bytes (seq prefix)")
        head = np.array([(seq >> 24) & 0xFF, (seq >> 16) & 0xFF,
                         (seq >>  8) & 0xFF,  seq        & 0xFF], dtype=np.uint8)
        tail = rng.integers(0, 256, payload_bytes - 4, dtype=np.uint8)
        return np.concatenate([head, tail]).tobytes()


    def _build_packet_inline(payload: bytes) -> np.ndarray:
        """In-thread packet build (no MP). Returns DAC-scaled complex64 samples."""
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        pkt = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0,
                     length=len(payload), payload=bits)
        samples = tx_pipe.transmit(pkt)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples = samples / peak
        return (samples * DAC_SCALE).astype(np.complex64)


    def _build_packet(seq: int, payload_bytes: int) -> np.ndarray:
        """Inline build that takes the (seq, payload_size) tuple — kept for callers
        that don't bother with the pool path (build-once probes, etc.)."""
        return _build_packet_inline(_make_payload_bytes(seq, payload_bytes))


    # ---------------------------------------------------------------------------
    # RX MP shim — same return shape as rx_pipe.receive() so run_rx / the both-
    # mode RX thread don't need to know which path is in use.
    # ---------------------------------------------------------------------------

    def _result_to_packet(result, abs_payload_start: int) -> Packet:
        """Translate an RXResult back into a Packet so legacy consumers (run_rx,
        save_rx_buf, constellation plotter) keep working unchanged."""
        if result.payload_bytes:
            bits = np.unpackbits(np.frombuffer(result.payload_bytes, dtype=np.uint8))
            payload = bits.reshape(-1, 1).astype(int)
        else:
            payload = np.empty((0, 1), dtype=int)
        return Packet(
            src_mac=result.src_mac,
            dst_mac=result.dst_mac,
            type=result.type,
            seq_num=result.seq_num,
            length=result.length,
            payload=payload,
            valid=result.valid,
            sample_start=abs_payload_start,
            rx_symbols=result.rx_symbols,
        )


    def _mp_receive(raw: np.ndarray, search_from: int) -> tuple[list[Packet], int]:
        """Drop-in replacement for rx_pipe.receive() that decodes via rx_pool.

        Mirrors the break-on-tail-cutoff semantics: detections after a tail-cut
        are discarded so they remain eligible for retry on the next iteration.
        """
        search_buf = raw[search_from:]
        filtered   = match_filter(search_buf, rx_pipe.rrc_taps)
        detections = rx_pipe.detect(filtered)

        packets: list[Packet] = []
        payload_failures = 0
        max_det = search_from

        if not detections:
            rx_pipe.last_payload_failures = 0
            rx_pipe.last_tail_cutoffs = 0
            return packets, max_det

        sub = rx_pool.submit_buffer(filtered, detections, search_from_abs=search_from)

        tail_cut = False
        n_tail_cutoffs = 0
        for ar, abs_offset in zip(sub.futures, sub.abs_offsets):
            try:
                result = ar.get(timeout=10.0)
            except Exception:
                payload_failures += 1
                max_det = max(max_det, abs_offset)
                continue
            if tail_cut:
                continue
            if result.status == "ok":
                packets.append(_result_to_packet(result, abs_offset))
                max_det = max(max_det, abs_offset)
            elif result.status == "tail_cutoff":
                tail_cut = True
                n_tail_cutoffs += 1
            else:
                payload_failures += 1
                max_det = max(max_det, abs_offset)

        # Surface the per-call counters where rx_pipe.receive() would have set them
        # so existing stats-gathering callsites keep working.
        rx_pipe.last_payload_failures = payload_failures
        rx_pipe.last_tail_cutoffs     = n_tail_cutoffs
        return packets, max_det


    def _receive(raw: np.ndarray, search_from: int) -> tuple[list[Packet], int]:
        """Wrapper that dispatches between inline rx_pipe.receive and the MP pool."""
        if rx_pool is None:
            return rx_pipe.receive(raw, search_from=search_from)
        return _mp_receive(raw, search_from=search_from)


    # ---------------------------------------------------------------------------
    # TX mode — continuous streaming via TxStream
    # ---------------------------------------------------------------------------

    def run_tx():
        status = LiveStatus(n_lines=1)
        _install_live_logging(status)

        stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size,
                          filler_amp=args.tx_filler_amp)
        stream.start()
        status.log(f"  [TX] streaming (buf={tx_buf_size} samples / "
                   f"{tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms, "
                   f"variable={args.variable})")

        rate       = RateMeter()
        global_seq = 0
        burst_num  = 0

        # Bounded inflight queue for the MP path. Keep ~2 jobs/worker queued so the
        # pool stays saturated without hoarding memory.
        inflight: collections.deque = collections.deque()
        inflight_depth = max(2, tx_pool.n_workers * 2) if tx_pool else 0

        # --- profiling ----------------------------------------------------------
        prof_enabled  = args.profile
        prof_build_ms: collections.deque = collections.deque(maxlen=500)
        prof_send_ms:  collections.deque = collections.deque(maxlen=500)
        air_ms_per_pkt = frame_len / pipe_cfg.SAMPLE_RATE * 1e3

        def _maybe_log_profile() -> None:
            n = len(prof_build_ms)
            if n < 100:
                return
            bt = sorted(prof_build_ms)
            st = sorted(prof_send_ms)
            p99 = max(0, int(n * 0.99) - 1)
            label = "build" if tx_pool is None else "ar.get"
            status.log(
                f"  [TX-prof] last {n} pkts: "
                f"{label} avg={sum(bt)/n:.2f}ms p50={bt[n//2]:.2f}ms p99={bt[p99]:.2f}ms | "
                f"send avg={sum(st)/n:.2f}ms p50={st[n//2]:.2f}ms p99={st[p99]:.2f}ms | "
                f"air≈{air_ms_per_pkt:.2f}ms"
            )
            prof_build_ms.clear()
            prof_send_ms.clear()

        def _drain_inflight():
            """Pull one finished job off the head of inflight (blocks for it)."""
            ar, payload_len, seq = inflight.popleft()
            t0 = time.perf_counter()
            samples = ar.get(timeout=10.0)
            t1 = time.perf_counter()
            stream.send(samples)
            t2 = time.perf_counter()
            if prof_enabled:
                prof_build_ms.append((t1 - t0) * 1e3)
                prof_send_ms.append((t2 - t1) * 1e3)
                _maybe_log_profile()
            rate.add(payload_len)
            status.set(0, f"  [TX] seq={seq:>10d}  burst={burst_num:>4d}  "
                          f"pending={stream.pending:>3d}  rate={_fmt_rate(rate.rate_bps)}  "
                          f"avg={_fmt_rate(rate.avg_bps)}  total={_fmt_bytes(rate.total_bytes)}")

        try:
            while True:
                burst_num  += 1
                burst_start = global_seq
                t0 = time.perf_counter()

                for i in range(args.packets):
                    if args.variable:
                        pay_size = int(rng.integers(args.min_payload, args.payload + 1))
                    else:
                        pay_size = args.payload

                    if tx_pool is None:
                        tb0 = time.perf_counter()
                        samples = _build_packet(global_seq, pay_size)
                        tb1 = time.perf_counter()
                        stream.send(samples)
                        tb2 = time.perf_counter()
                        if prof_enabled:
                            prof_build_ms.append((tb1 - tb0) * 1e3)
                            prof_send_ms.append((tb2 - tb1) * 1e3)
                            _maybe_log_profile()
                        rate.add(pay_size)
                        status.set(0, f"  [TX] seq={global_seq+1:>10d}  burst={burst_num:>4d}  "
                                      f"pending={stream.pending:>3d}  rate={_fmt_rate(rate.rate_bps)}  "
                                      f"avg={_fmt_rate(rate.avg_bps)}  total={_fmt_bytes(rate.total_bytes)}")
                    else:
                        payload_bytes = _make_payload_bytes(global_seq, pay_size)
                        ar = tx_pool.submit(payload_bytes, 0, 1, 0, 0, float(DAC_SCALE))
                        inflight.append((ar, pay_size, global_seq + 1))
                        # Apply back-pressure: if the queue is at depth, drain one
                        # before submitting more.
                        while len(inflight) >= inflight_depth:
                            _drain_inflight()

                    global_seq += 1

                # Only drain to zero when there's an inter-burst gap to honour.
                # Continuous TX (interval=0) keeps the worker pipeline saturated
                # across burst boundaries — draining here would idle the pool
                # for ~inflight_depth × per_packet_time on every burst.
                if args.interval > 0:
                    while inflight:
                        _drain_inflight()

                t1 = time.perf_counter()
                status.log(f"  [TX] burst {burst_num}: {args.packets} pkts "
                           f"(seq {burst_start}–{global_seq - 1}) submitted in {t1 - t0:.3f}s  "
                           f"(inflight={len(inflight)}, pending={stream.pending}, "
                           f"bufs_sent={stream.bufs_sent})")

                if args.interval > 0:
                    time.sleep(args.interval / 1000.0)
        finally:
            status.stop()

    # ---------------------------------------------------------------------------
    # RX mode — run forever, print every decoded packet
    # ---------------------------------------------------------------------------

    def _setup_constellation_plot():
        """Set up a constellation figure. Returns (fig, ax, scatter, save_path).

        When ``save_path`` is None, the plot updates live in an X11 window;
        when set (either via --constellation-save, or auto-fallback when X11
        is unavailable under sudo+netns), the plot is rendered to that PNG on
        every refresh.
        """
        import os
        import matplotlib

        save_path = args.constellation_save
        live = save_path is None and "DISPLAY" in os.environ

        if live:
            try:
                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt
                plt.ion()
                _probe = plt.figure()    # cheap test that the backend is actually usable
                plt.close(_probe)
            except Exception as e:
                logger.warning(
                    "X11 unavailable (%s) — falling back to PNG save", e)
                live = False
                save_path = save_path or "oneway-constellation.png"

        if not live:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            save_path = save_path or "oneway-constellation.png"
            logger.info("constellation will be written to %s "
                        "(view with `feh -R 1 %s` or `eog %s` for live updates)",
                        save_path, save_path, save_path)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.set_title("PSK8 constellation — accumulating…")
        ax.grid(True, alpha=0.25)

        # Decision boundary lines at every π/4
        for k in range(8):
            angle = k * np.pi / 4 + np.pi / 8
            ax.plot([0, 1.55 * np.cos(angle)], [0, 1.55 * np.sin(angle)],
                    "k--", alpha=0.15, linewidth=0.8)

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 256)
        ax.plot(np.cos(theta), np.sin(theta), "k-", alpha=0.08, linewidth=1)

        # Ideal 8-PSK points
        ideal = rx_pipe.psk8.symbol_mapping
        ax.scatter(ideal.real, ideal.imag, s=200, marker="*", c="red",
                   zorder=6, label="Ideal", linewidths=0)

        # Received symbols — start empty
        scat = ax.scatter([], [], s=6, alpha=0.35, c="steelblue",
                          label="Received", linewidths=0)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        if live:
            plt.pause(0.01)
        else:
            fig.savefig(save_path)
        return fig, ax, scat, save_path


    def run_rx():
        status = LiveStatus(n_lines=1)
        _install_live_logging(status)

        # Lossless stream: large queue so the hardware reader never stalls while
        # the decoder is busy.
        stream = RxStream(rx_sdr, maxsize=128, lossless=True)
        stream.start(flush=16)

        # Software AGC — only useful when the AD9361's own AGC is off.
        agc = None
        if args.rx_gain_mode == "manual" and args.rx_agc == "auto":
            agc = RxAGC(rx_sdr, initial_gain_db=args.rx_gain)
            status.log(f"  [RX] software AGC enabled (start gain {args.rx_gain} dB)")

        status.log("  [RX] listening …")

        # --- optional constellation plot ---
        _fig = _ax = _scat = None
        _const_save_path: str | None = None  # set if PNG-save mode is active
        _sym_buf: list[np.ndarray] = []   # post-Costas symbols accumulated across packets
        _pkt_count = 0                    # valid packets since last plot refresh
        _PLOT_EVERY = 10                   # update plot every N valid packets
        # --constellation-save implies --constellation
        if args.constellation_save and not args.constellation:
            args.constellation = True
        if args.constellation:
            import matplotlib.pyplot as plt
            _fig, _ax, _scat, _const_save_path = _setup_constellation_plot()

        _csv_file = None
        if args.constellation_csv:
            _csv_file = open(args.constellation_csv, "w", buffering=1)
            _csv_file.write("seq,I,Q\n")
            status.log(f"  [RX] streaming constellation symbols to {args.constellation_csv}")

        rate        = RateMeter()
        n_total     = 0
        n_valid     = 0
        n_dropped   = 0
        last_seq    = None
        prev_buf    = None
        search_from = 0

        # Processing time stats (logged every 500 buffers)
        _proc_times: list[float] = []
        _buf_count  = 0

        # --- optional raw-buffer dump for offline sweep replay ---
        _save_dir: Path | None = None
        _saved_n  = 0
        if args.save_rx_buf:
            _save_dir = Path(args.save_rx_buf)
            _save_dir.mkdir(parents=True, exist_ok=True)
            _skip_msg = f" after skipping first {args.save_skip}" if args.save_skip > 0 else ""
            status.log(f"  [RX] dumping up to {args.save_n} buffers to {_save_dir}/{_skip_msg}")

        try:
            while True:
                try:
                    curr_buf = stream.get(timeout=0.05)
                except queue.Empty:
                    if args.constellation and _fig is not None and _const_save_path is None:
                        import matplotlib.pyplot as plt
                        plt.pause(0.001)
                    continue

                prev_len = len(prev_buf) if prev_buf is not None else 0
                raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

                _t0 = time.perf_counter()
                _rx_search_from = search_from
                packets, max_det = _receive(raw, search_from=search_from)
                _proc_ms = (time.perf_counter() - _t0) * 1e3
                _proc_times.append(_proc_ms)
                _buf_count += 1

                if (_save_dir is not None and _saved_n < args.save_n
                    and _buf_count > args.save_skip
                    and len(packets) > 0):
                    _n_valid_buf = sum(1 for p in packets if p.valid)
                    # Ground truth for offline replay: seq_nums of every valid
                    # packet decoded during capture. Sweep scripts compare each
                    # combo's decoded seqs against this set to catch silent LDPC
                    # divergence (CRC-16 false-pass prob is ~2⁻¹⁶ but non-zero).
                    _seqs_in_buf: list[int] = []
                    for p in packets:
                        if not p.valid or p.payload is None or p.payload.size < 32:
                            continue
                        sb = np.packbits(p.payload[:32].astype(np.uint8))
                        _seqs_in_buf.append(
                            (int(sb[0]) << 24) | (int(sb[1]) << 16)
                            | (int(sb[2]) << 8) | int(sb[3])
                        )
                    _path = _save_dir / f"rxbuf_{_saved_n:04d}.npz"
                    np.savez(
                        _path,
                        samples=raw.astype(np.complex64),
                        search_from=np.int64(_rx_search_from),
                        sample_rate=np.int64(pipe_cfg.SAMPLE_RATE),
                        sps=np.int64(pipe_cfg.SPS),
                        mod_scheme=pipe_cfg.MOD_SCHEME.name,
                        code_rate=pipe_cfg.CODING_RATE.name,
                        n_valid_in_buf=np.int64(_n_valid_buf),
                        n_total_in_buf=np.int64(len(packets)),
                        seq_nums=np.array(_seqs_in_buf, dtype=np.int64),
                    )
                    _saved_n += 1
                    status.log(f"  [RX] saved {_path.name} "
                               f"({len(raw)} samples, {_n_valid_buf}/{len(packets)} valid, "
                               f"seqs={_seqs_in_buf})")
                if _buf_count % 500 == 0:
                    buf_dur_ms = rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3
                    status.log(f"  [RX perf] proc={np.mean(_proc_times):.2f} ms avg / "
                               f"{max(_proc_times):.2f} ms max  (buf={buf_dur_ms:.2f} ms, "
                               f"queue≈{stream._q.qsize()}/{stream._q.maxsize})")
                    _proc_times.clear()

                prev_buf = curr_buf
                if packets:
                    last_ps = max(pkt.sample_start for pkt in packets)
                    search_from = max(0, max(last_ps, max_det) - prev_len)
                else:
                    search_from = max(0, max_det - prev_len)

                for pkt in packets:
                    n_total += 1
                    if pkt.valid:
                        seq_bits = pkt.payload[:32]
                        b        = np.packbits(seq_bits)
                        seq      = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])
                        n_valid += 1
                        rate.add(pkt.length)

                        gap = 0
                        if last_seq is not None:
                            gap = (seq - last_seq - 1) % (2**32)
                            if gap > 0:
                                n_dropped += gap
                        last_seq = seq

                        if gap > 0:
                            status.log(f"  [RX] *** GAP: {gap} dropped before seq={seq} ***")

                        gain_str = (f"  gain={agc.gain_db:>4.1f}dB" if agc is not None
                                    else (f"  gain={args.rx_gain:>4.1f}dB" if args.rx_gain_mode == "manual" else ""))
                        status.set(0, f"  [RX] #{n_total:>6d}  seq={seq:>10d}  valid=True   "
                                      f"(ok={n_valid}, dropped≈{n_dropped})  "
                                      f"q={stream._q.qsize():>3d}/{stream._q.maxsize}{gain_str}  "
                                      f"rate={_fmt_rate(rate.rate_bps)}  "
                                      f"avg={_fmt_rate(rate.avg_bps)}  total={_fmt_bytes(rate.total_bytes)}")

                        if _csv_file is not None and pkt.rx_symbols is not None:
                            syms = pkt.rx_symbols
                            lines = "".join(
                                f"{seq},{z.real:.7g},{z.imag:.7g}\n" for z in syms
                            )
                            _csv_file.write(lines)

                        # Collect symbols for constellation plot
                        if args.constellation and pkt.rx_symbols is not None:
                            _sym_buf.append(pkt.rx_symbols)
                            _pkt_count += 1
                            if _pkt_count >= _PLOT_EVERY:
                                import matplotlib.pyplot as plt
                                all_syms = np.concatenate(_sym_buf)
                                _scat.set_offsets(np.column_stack([all_syms.real, all_syms.imag]))
                                _ax.set_title(
                                    f"PSK8 constellation — last {_pkt_count} pkts "
                                    f"({len(all_syms)} symbols)"
                                )
                                if _const_save_path is None:
                                    _fig.canvas.flush_events()
                                    plt.pause(0.001)
                                else:
                                    _fig.savefig(_const_save_path)
                                _sym_buf.clear()
                                _pkt_count = 0
                    else:
                        status.log(f"  [RX] #{n_total}  header CRC failed  "
                                   f"(ok={n_valid}, dropped≈{n_dropped})")

                if agc is not None:
                    n_valid_in_buf = sum(1 for p in packets if p.valid)
                    agc.update(curr_buf, n_valid_in_buf)

                # Refresh the live line every buffer (even on idle ones) so
                # the displayed gain tracks the AGC in real time instead of
                # only updating when a valid packet lands.
                gain_str = (f"  gain={agc.gain_db:>4.1f}dB" if agc is not None
                            else (f"  gain={args.rx_gain:>4.1f}dB" if args.rx_gain_mode == "manual" else ""))
                last_seq_str = f"{last_seq:>10d}" if last_seq is not None else "         -"
                status.set(0, f"  [RX] #{n_total:>6d}  seq={last_seq_str}            "
                              f"(ok={n_valid}, dropped≈{n_dropped})  "
                              f"q={stream._q.qsize():>3d}/{stream._q.maxsize}{gain_str}  "
                              f"rate={_fmt_rate(rate.rate_bps)}  "
                              f"avg={_fmt_rate(rate.avg_bps)}  total={_fmt_bytes(rate.total_bytes)}")

        except KeyboardInterrupt:
            status.log(f"  [RX] interrupted — decoded {n_valid} valid / {n_total} total frames, ~{n_dropped} dropped by seq gap")
        finally:
            status.stop()
            stream.stop()
            if _csv_file is not None:
                _csv_file.close()
                print(f"[info] constellation CSV written to {args.constellation_csv}")
            if args.constellation and _fig is not None:
                import matplotlib.pyplot as plt
                if _const_save_path is None:
                    plt.ioff()
                else:
                    _fig.savefig(_const_save_path)
                    print(f"[info] constellation saved to {_const_save_path}")
                plt.show()  # keep window open after run ends

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------

    if args.mode == "tx":
        try:
            run_tx()
        except KeyboardInterrupt:
            print("\n[TX] interrupted")

    elif args.mode == "rx":
        run_rx()

    else:  # "both" — original threaded loopback behaviour
        # ---------------------------------------------------------------------------
        # Shared state
        # ---------------------------------------------------------------------------

        rx_results: list[dict] = []
        rx_lock      = threading.Lock()
        tx_done      = threading.Event()
        rx_ready     = threading.Event()
        _decoded_seqs: set[int] = set()
        _all_decoded = threading.Event()

        # Estimate total air time for all packets (continuous stream, no batch padding).
        # Add margin for TX buffer packing overhead (silence between packed packets).
        _air_time_ms   = int(frame_len * args.packets / pipe_cfg.SAMPLE_RATE * 1000)
        _buf_ms        = int(rx_buf_size / pipe_cfg.SAMPLE_RATE * 1000)
        _BUFS_AFTER_TX = int(np.ceil(_air_time_ms / _buf_ms)) + 8

        print(f"Post-TX   : {_BUFS_AFTER_TX} bufs needed after tx_done  "
              f"(air={_air_time_ms} ms / buf={_buf_ms} ms + 8 margin)\n")

        # Two-line live status: line 0 = TX, line 1 = RX. Pipeline log records
        # (DECODE ERROR, etc.) scroll above via _install_live_logging.
        status   = LiveStatus(n_lines=2)
        tx_rate  = RateMeter()
        rx_rate  = RateMeter()
        _install_live_logging(status)

        # -----------------------------------------------------------------------
        # TX thread
        # -----------------------------------------------------------------------

        def tx_thread():
            rx_ready.wait()

            stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size,
                          filler_amp=args.tx_filler_amp)
            stream.start()

            # MP path: keep ~2× n_workers builds in flight; non-MP: build inline.
            inflight: collections.deque = collections.deque()
            inflight_depth = max(2, tx_pool.n_workers * 2) if tx_pool else 0

            def _drain_one():
                ar, payload_len, seq_disp = inflight.popleft()
                samples = ar.get(timeout=10.0)
                stream.send(samples)
                tx_rate.add(payload_len)
                status.set(0, f"  [TX] seq={seq_disp:>10d}/{args.packets}  "
                              f"pending={stream.pending:>3d}  "
                              f"rate={_fmt_rate(tx_rate.rate_bps)}  "
                              f"avg={_fmt_rate(tx_rate.avg_bps)}  "
                              f"total={_fmt_bytes(tx_rate.total_bytes)}")

            t0 = time.perf_counter()
            for seq in range(args.packets):
                if args.variable:
                    pay_size = int(rng.integers(args.min_payload, args.payload + 1))
                else:
                    pay_size = args.payload

                if tx_pool is None:
                    samples = _build_packet(seq, pay_size)
                    stream.send(samples)
                    tx_rate.add(pay_size)
                    status.set(0, f"  [TX] seq={seq+1:>10d}/{args.packets}  "
                                  f"pending={stream.pending:>3d}  "
                                  f"rate={_fmt_rate(tx_rate.rate_bps)}  "
                                  f"avg={_fmt_rate(tx_rate.avg_bps)}  "
                                  f"total={_fmt_bytes(tx_rate.total_bytes)}")
                else:
                    payload_bytes = _make_payload_bytes(seq, pay_size)
                    ar = tx_pool.submit(payload_bytes, 0, 1, 0, 0, float(DAC_SCALE))
                    inflight.append((ar, pay_size, seq + 1))
                    while len(inflight) >= inflight_depth:
                        _drain_one()

            # Drain any remaining inflight builds before reporting timing.
            while inflight:
                _drain_one()

            # Wait for the stream to drain the queue
            while stream.pending > 0:
                time.sleep(0.01)
            # One more air_time to let the last buffer finish transmitting
            time.sleep(tx_buf_size / pipe_cfg.SAMPLE_RATE)

            t1 = time.perf_counter()
            stream.stop()

            status.log(f"  [TX] done in {t1 - t0:.3f}s — sent {args.packets} pkts, "
                       f"avg {_fmt_rate(tx_rate.avg_bps)}")
            tx_done.set()

        # -----------------------------------------------------------------------
        # RX thread
        # -----------------------------------------------------------------------

        def rx_thread():
            stream = RxStream(rx_sdr, maxsize=128, lossless=True)
            stream.start(flush=16)
            rx_ready.set()

            agc = None
            if args.rx_gain_mode == "manual" and args.rx_agc == "auto":
                agc = RxAGC(rx_sdr, initial_gain_db=args.rx_gain)
                status.log(f"  [RX] software AGC enabled (start gain {args.rx_gain} dB)")

            prev_buf    = None
            search_from = 0
            n_after_tx  = 0
            n_total     = 0
            n_valid     = 0

            while not _all_decoded.is_set():
                if tx_done.is_set() and n_after_tx >= _BUFS_AFTER_TX:
                    break
                try:
                    curr_buf = stream.get(timeout=0.05)
                except queue.Empty:
                    continue

                if tx_done.is_set():
                    n_after_tx += 1

                prev_len = len(prev_buf) if prev_buf is not None else 0
                raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf

                packets, max_det = _receive(raw, search_from=search_from)

                prev_buf = curr_buf
                if packets:
                    last_ps = max(pkt.sample_start for pkt in packets)
                    search_from = max(0, max(last_ps, max_det) - prev_len)
                else:
                    search_from = max(0, max_det - prev_len)

                for pkt in packets:
                    n_total += 1
                    if pkt.valid:
                        seq_bits = pkt.payload[:32]
                        b        = np.packbits(seq_bits)
                        seq      = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])
                        n_valid += 1
                        rx_rate.add(pkt.length)
                        gain_str = (f"  gain={agc.gain_db:>4.1f}dB" if agc is not None
                                    else (f"  gain={args.rx_gain:>4.1f}dB" if args.rx_gain_mode == "manual" else ""))
                        status.set(1, f"  [RX] #{n_total:>6d}  seq={seq:>10d}  valid=True   "
                                      f"(ok={n_valid})  "
                                      f"q={stream._q.qsize():>3d}/{stream._q.maxsize}{gain_str}  "
                                      f"rate={_fmt_rate(rx_rate.rate_bps)}  "
                                      f"avg={_fmt_rate(rx_rate.avg_bps)}  "
                                      f"total={_fmt_bytes(rx_rate.total_bytes)}")
                    else:
                        seq = -1
                        status.log(f"  [RX] #{n_total}  header CRC failed (ok={n_valid})")

                    entry = {"seq_num": seq, "valid": pkt.valid, "time": time.perf_counter()}
                    with rx_lock:
                        rx_results.append(entry)
                        if pkt.valid:
                            _decoded_seqs.add(seq)
                            if len(_decoded_seqs) >= args.packets:
                                _all_decoded.set()

                if agc is not None:
                    n_valid_in_buf = sum(1 for p in packets if p.valid)
                    agc.update(curr_buf, n_valid_in_buf)

                # Refresh the live line every buffer so the gain field reflects
                # AGC adjustments in real time, not only when a valid packet lands.
                gain_str = (f"  gain={agc.gain_db:>4.1f}dB" if agc is not None
                            else (f"  gain={args.rx_gain:>4.1f}dB" if args.rx_gain_mode == "manual" else ""))
                status.set(1, f"  [RX] #{n_total:>6d}            "
                              f"(ok={n_valid})  "
                              f"q={stream._q.qsize():>3d}/{stream._q.maxsize}{gain_str}  "
                              f"rate={_fmt_rate(rx_rate.rate_bps)}  "
                              f"avg={_fmt_rate(rx_rate.avg_bps)}  "
                              f"total={_fmt_bytes(rx_rate.total_bytes)}")

            stream.stop()
            status.log(f"  [RX] done — decoded {n_valid}/{n_total} (avg {_fmt_rate(rx_rate.avg_bps)})")

        # -----------------------------------------------------------------------
        # Launch threads
        # -----------------------------------------------------------------------

        t_tx = threading.Thread(target=tx_thread, name="TX", daemon=True)
        t_rx = threading.Thread(target=rx_thread, name="RX", daemon=True)

        t_rx.start()
        t_tx.start()

        t_tx.join()
        t_rx.join(timeout=args.packets * args.interval / 1000.0 + _air_time_ms * 4 / 1000.0 + 15.0)

        status.stop()

        # -----------------------------------------------------------------------
        # Report
        # -----------------------------------------------------------------------

        print()
        print("=" * 50)
        print("RESULTS")
        print("=" * 50)

        n_tx  = args.packets
        n_rx  = len(rx_results)
        valid = [r for r in rx_results if r["valid"]]

        seen_seqs    = set()
        unique_valid = []
        for r in valid:
            if r["seq_num"] not in seen_seqs:
                seen_seqs.add(r["seq_num"])
                unique_valid.append(r)

        n_unique  = len(unique_valid)
        n_dropped = n_tx - n_unique
        drop_rate = n_dropped / n_tx if n_tx > 0 else float("nan")

        missing_seqs = sorted(set(range(n_tx)) - seen_seqs)

        if len(unique_valid) >= 2:
            rx_duration   = unique_valid[-1]["time"] - unique_valid[0]["time"]
            rx_throughput = n_unique * args.payload / rx_duration if rx_duration > 0 else float("nan")
        else:
            rx_duration   = float("nan")
            rx_throughput = float("nan")

        print(f"Transmitted : {n_tx} packets")
        print(f"RX captures : {n_rx}  (raw, incl. duplicates)")
        print(f"Decoded OK  : {n_unique} unique valid packets")
        print(f"Dropped     : {n_dropped} packets")
        print(f"Drop rate   : {drop_rate * 100:.1f}%")
        print(f"RX duration : {rx_duration:.3f} s")
        print(f"RX throughput: {rx_throughput*8 / 1_000_000:.2f} mbps")
        if missing_seqs:
            print(f"Missing seq : {missing_seqs}")
        else:
            print("Missing seq : none")
        print("=" * 50)

    # ---------------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------------

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

