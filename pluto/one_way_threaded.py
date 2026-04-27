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
from modules.parallel_pipeline import RXWorkerPool, TXWorkerPool
from modules.pulse_shaping.pulse_shaping import match_filter
from pluto.config import (
    DAC_SCALE,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    PIPELINE,
    configure_rx,
    configure_tx,
)
from pluto.setup_config import SETUP_PATH, load_or_die as load_setup

# FDD frequency plan: A transmits on FREQ_A_TO_B and listens on FREQ_B_TO_A;
# B does the opposite, so a node-A and node-B process can run simultaneously
# without colliding on one channel.
NODE_FREQS = {
    "A": {"tx": FREQ_A_TO_B, "rx": FREQ_B_TO_A},
    "B": {"tx": FREQ_B_TO_A, "rx": FREQ_A_TO_B},
}
from pluto.sdr_stream import RxStream, TxStream
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
    parser.add_argument("--tx-ip",    type=str,   default=None,          help="Override TX Pluto IP (default: derived from --node)")
    parser.add_argument("--rx-ip",    type=str,   default=None,          help="Override RX Pluto IP (default: derived from --node)")
    parser.add_argument("--mode",     type=str,   default="both",        help="Mode: 'tx', 'rx', or 'both' (default: both)")
    parser.add_argument("--cfo-offset", type=int, default=None,
                        help="Manual override for the RX-LO CFO correction in Hz. "
                             "Default: value from the cfo block of pluto/setup.json "
                             "for --node (run scripts/cfo_calibrate.py to generate "
                             "it), or 0 if no calibration is present. Only affects RX.")
    parser.add_argument("--tx-freq", type=float, default=None, help="TX center frequency in Hz (default: derived from --node via NODE_FREQS)")
    parser.add_argument("--rx-freq", type=float, default=None, help="RX center frequency in Hz (default: derived from --node via NODE_FREQS)")
    parser.add_argument("--constellation", action="store_true", help="Show live PSK8 constellation plot (RX mode only)")
    parser.add_argument("--variable", action="store_true", help="Randomize payload size per packet (between --min-payload and --payload)")
    parser.add_argument("--min-payload", type=int, default=4, help="Minimum payload bytes when --variable is set (default: 4, must hold seq number)")
    parser.add_argument("--tx-buf-mult", type=int, default=8, help="TX buffer size as multiple of next-power-of-2 frame length (default: 8)")
    parser.add_argument("--hardware-rrc", action="store_true", help="Use the FPGA hardware RRC/4x interpolation path on TX (toggles the pluto_custom firmware GPIO). TX only — RX always uses software match filter.")
    parser.add_argument("--save-rx-buf", type=str, default=None,
                        help="Directory to dump raw RX buffers (.npz) for offline replay by "
                             "scripts/sweep_*_params.py. RX mode only.")
    parser.add_argument("--save-n",      type=int, default=4,
                        help="How many RX buffers to dump (default: 4). Only buffers that "
                             "produced ≥1 detected packet are saved.")
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
    args = parser.parse_args()

    if args.mode not in ("tx", "rx", "both"):
        print(f"ERROR: --mode must be 'tx', 'rx', or 'both', got '{args.mode}'")
        sys.exit(1)

    setup = load_setup()
    if args.node not in setup.nodes:
        print(f"ERROR: --node must be one of {sorted(setup.nodes)}, got '{args.node}'")
        sys.exit(1)

    tx_ip = args.tx_ip or setup.tx_ip(args.node)
    rx_ip = args.rx_ip or setup.rx_ip(args.node)

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
    # SDR setup — per direction. Each node runs a dedicated TX Pluto and a
    # dedicated RX Pluto; we only open the device(s) that the current mode
    # actually needs, so tx/rx-only invocations don't require both radios to
    # be plugged in.
    # ---------------------------------------------------------------------------

    tx_sdr = None
    rx_sdr = None

    tx_freq = int(args.tx_freq) if args.tx_freq is not None else NODE_FREQS[args.node]["tx"]
    rx_freq = int(args.rx_freq) if args.rx_freq is not None else NODE_FREQS[args.node]["rx"]

    if args.mode in ("tx", "both"):
        tx_sdr = adi.Pluto("ip:" + tx_ip)
        configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False)

    if args.mode in ("rx", "both"):
        rx_sdr = adi.Pluto("ip:" + rx_ip)
        configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz, gain_mode="slow_attack")

    # ---------------------------------------------------------------------------
    # RX buffer sizing (needed for RX and both modes, and for the TX probe)
    # ---------------------------------------------------------------------------

    _probe_bits    = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)
    _probe_pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=args.payload, payload=_probe_bits)
    _probe_samples = tx_pipe.transmit(_probe_pkt)
    frame_len      = len(_probe_samples)
    rx_buf_size    = 16 * int(2 ** np.ceil(np.log2(frame_len)))

    if args.mode in ("rx", "both"):
        rx_sdr.rx_buffer_size = rx_buf_size

    tx_buf_size = args.tx_buf_mult * int(2 ** np.ceil(np.log2(frame_len)))

    print(f"Mode      : {args.mode}")
    print(f"Node      : {args.node}")
    if args.mode in ("tx", "both"):
        print(f"TX radio  : {tx_ip}   @ {tx_freq / 1e6:.3f} MHz")
    if args.mode in ("rx", "both"):
        print(f"RX radio  : {rx_ip}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
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
    # Worker pools — only created when --workers > 0.
    # ---------------------------------------------------------------------------

    tx_pool: TXWorkerPool | None = None
    rx_pool: RXWorkerPool | None = None

    if args.workers > 0:
        if args.mode in ("tx", "both"):
            tx_pool = TXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   start_method=args.mp_start)
        if args.mode in ("rx", "both"):
            rx_slot_samples = 2 * rx_buf_size + 1024
            rx_pool = RXWorkerPool(pipe_cfg, n_workers=args.workers,
                                   slot_samples=rx_slot_samples,
                                   n_slots=args.rx_slots,
                                   start_method=args.mp_start)
            print(f"RX slots  : {args.rx_slots} × {rx_slot_samples} samples")
        print(f"Workers   : {args.workers}  (start={args.mp_start or 'spawn'})")
    else:
        print("Workers   : 0  (inline TX build / RX decode on the threads)")
    print()

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

        stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size)
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

        def _drain_inflight():
            """Pull one finished job off the head of inflight (blocks for it)."""
            ar, payload_len, seq = inflight.popleft()
            samples = ar.get(timeout=10.0)
            stream.send(samples)
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
                        samples = _build_packet(global_seq, pay_size)
                        stream.send(samples)
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

                # End of burst: drain remaining inflight before reporting timing,
                # otherwise the burst time is misleading (work is still queued).
                while inflight:
                    _drain_inflight()

                t1 = time.perf_counter()
                status.log(f"  [TX] burst {burst_num}: {args.packets} pkts "
                           f"(seq {burst_start}–{global_seq - 1}) queued in {t1 - t0:.3f}s  "
                           f"(pending={stream.pending}, bufs_sent={stream.bufs_sent})")

                time.sleep(args.interval / 1000.0)
        finally:
            status.stop()

    # ---------------------------------------------------------------------------
    # RX mode — run forever, print every decoded packet
    # ---------------------------------------------------------------------------

    def _setup_constellation_plot():
        """Set up a live PSK8 constellation figure. Returns (fig, ax, scatter_handle)."""
        import matplotlib
        matplotlib.use("TkAgg" if "DISPLAY" in __import__("os").environ else "Agg")
        import matplotlib.pyplot as plt

        plt.ion()
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
        plt.pause(0.01)
        return fig, ax, scat


    def run_rx():
        status = LiveStatus(n_lines=1)
        _install_live_logging(status)

        # Lossless stream: large queue so the hardware reader never stalls while
        # the decoder is busy.
        stream = RxStream(rx_sdr, maxsize=128, lossless=True)
        stream.start(flush=16)

        status.log("  [RX] listening …")

        # --- optional constellation plot ---
        _fig = _ax = _scat = None
        _sym_buf: list[np.ndarray] = []   # post-Costas symbols accumulated across packets
        _pkt_count = 0                    # valid packets since last plot refresh
        _PLOT_EVERY = 20                  # update plot every N valid packets
        if args.constellation:
            import matplotlib.pyplot as plt
            _fig, _ax, _scat = _setup_constellation_plot()

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
            status.log(f"  [RX] dumping up to {args.save_n} buffers to {_save_dir}/")

        try:
            while True:
                try:
                    curr_buf = stream.get(timeout=0.05)
                except queue.Empty:
                    if args.constellation and _fig is not None:
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

                        status.set(0, f"  [RX] #{n_total:>6d}  seq={seq:>10d}  valid=True   "
                                      f"(ok={n_valid}, dropped≈{n_dropped})  "
                                      f"q={stream._q.qsize():>3d}/{stream._q.maxsize}  "
                                      f"rate={_fmt_rate(rate.rate_bps)}  "
                                      f"avg={_fmt_rate(rate.avg_bps)}  total={_fmt_bytes(rate.total_bytes)}")

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
                                _fig.canvas.flush_events()
                                plt.pause(0.001)
                                _sym_buf.clear()
                                _pkt_count = 0
                    else:
                        status.log(f"  [RX] #{n_total}  header CRC failed  "
                                   f"(ok={n_valid}, dropped≈{n_dropped})")

        except KeyboardInterrupt:
            status.log(f"  [RX] interrupted — decoded {n_valid} valid / {n_total} total frames, ~{n_dropped} dropped by seq gap")
        finally:
            status.stop()
            stream.stop()
            if args.constellation and _fig is not None:
                import matplotlib.pyplot as plt
                plt.ioff()
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

            stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size)
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
                        status.set(1, f"  [RX] #{n_total:>6d}  seq={seq:>10d}  valid=True   "
                                      f"(ok={n_valid})  "
                                      f"q={stream._q.qsize():>3d}/{stream._q.maxsize}  "
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

