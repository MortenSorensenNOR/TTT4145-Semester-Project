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
Pluto (a single USB-2 Pluto cannot sustain 4 Msps full-duplex). The 3rd
octet N in 192.168.N.1 picks the node — even → A, odd → B. Defaults come
from pluto.config.NODE_RADIO_IPS.
"""

import argparse
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
from pluto.cfo_config import CFO_CONFIG_PATH, load as load_cfo_calibration
from pluto.config import NODE_RADIO_IPS, PIPELINE, DAC_SCALE, configure_rx, configure_tx
from pluto.rrc_ctrl import set_hardware_rrc as _set_pluto_hardware_rrc, get_hardware_rrc as _get_pluto_hardware_rrc
from pluto.sdr_stream import RxStream, TxStream

import logging
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",     type=float, default=-30,           help="TX gain in dB (default: -30)")
parser.add_argument("--payload",  type=int,   default=10,            help="Payload bytes (default: 10)")
parser.add_argument("--packets",  type=int,   default=20,            help="Number of packets per TX burst (default: 20)")
parser.add_argument("--interval", type=float, default=200,           help="Inter-burst gap in ms (default: 200)")
parser.add_argument("--node",     type=str,   default="A",           help="Node identity A or B; picks default TX/RX IPs from NODE_RADIO_IPS")
parser.add_argument("--tx-ip",    type=str,   default=None,          help="Override TX Pluto IP (default: derived from --node)")
parser.add_argument("--rx-ip",    type=str,   default=None,          help="Override RX Pluto IP (default: derived from --node)")
parser.add_argument("--mode",     type=str,   default="both",        help="Mode: 'tx', 'rx', or 'both' (default: both)")
parser.add_argument("--cfo-offset", type=int, default=None,
                    help="Manual override for the RX-LO CFO correction in Hz. "
                         "Default: value from pluto/cfo_calibration.json for "
                         "--node (run scripts/cfo_calibrate.py to generate it), "
                         "or 0 if no calibration file exists. Only affects RX.")
parser.add_argument("--freq", type=float, default=PIPELINE.CENTER_FREQ, help="Center frequency")
parser.add_argument("--constellation", action="store_true", help="Show live PSK8 constellation plot (RX mode only)")
parser.add_argument("--variable", action="store_true", help="Randomize payload size per packet (between --min-payload and --payload)")
parser.add_argument("--min-payload", type=int, default=4, help="Minimum payload bytes when --variable is set (default: 4, must hold seq number)")
parser.add_argument("--tx-buf-mult", type=int, default=8, help="TX buffer size as multiple of next-power-of-2 frame length (default: 8)")
parser.add_argument("--hardware-rrc", action="store_true", help="Use the FPGA hardware RRC/4x interpolation path on TX (toggles the pluto_custom firmware GPIO). TX only — RX always uses software match filter.")
args = parser.parse_args()

if args.mode not in ("tx", "rx", "both"):
    print(f"ERROR: --mode must be 'tx', 'rx', or 'both', got '{args.mode}'")
    sys.exit(1)

if args.node not in NODE_RADIO_IPS:
    print(f"ERROR: --node must be one of {sorted(NODE_RADIO_IPS)}, got '{args.node}'")
    sys.exit(1)

tx_ip = args.tx_ip or NODE_RADIO_IPS[args.node]["tx"]
rx_ip = args.rx_ip or NODE_RADIO_IPS[args.node]["rx"]

# Resolve RX-LO CFO offset: manual CLI override wins; otherwise pull the
# measured value for this node from the calibration file; otherwise 0. TX
# always emits at its natural LO (split-radio convention — the peer's RX
# does the compensating).
if args.cfo_offset is not None:
    rx_cfo_hz = args.cfo_offset
    cfo_src   = "cli"
elif args.mode == "tx":
    rx_cfo_hz = 0
    cfo_src   = "n/a (tx-only)"
else:
    cal = load_cfo_calibration()
    if cal is None:
        rx_cfo_hz = 0
        cfo_src   = "unset"
        print(f"  [warn] no CFO calibration at {CFO_CONFIG_PATH} — using 0 Hz. "
              f"Run 'uv run python scripts/cfo_calibrate.py' to generate one.")
    else:
        rx_cfo_hz = cal.rx_offset_for(args.node)
        cfo_src   = f"calibration ({cal.measured_at or 'unknown date'})"

# ---------------------------------------------------------------------------
# Hardware-RRC GPIO: the FPGA applies RRC+4× interpolation iff the pluto_custom
# GPIO is set. Software mode and GPIO state MUST match — mismatch means either
# double-filtering (GPIO on + SW upsample) or no filtering at all, both of
# which destroy the signal. Toggle it over SSH on the TX Pluto before we open
# the iio context. RX never uses the FPGA RRC (removed from the RX path) so
# skip the RX Pluto.
# ---------------------------------------------------------------------------

# Only drive the FPGA GPIO when the user actually asked for it. We leave
# whatever state it's in otherwise — meaning you must pass --hardware-rrc
# on every invocation after flashing the custom firmware, otherwise the
# FPGA may still be in hardware-RRC mode from a previous run while TX
# generates 4× upsampled samples → double-filtered signal.
if args.hardware_rrc and args.mode in ("tx", "both"):
    try:
        _set_pluto_hardware_rrc(tx_ip, True)
        print(f"  [pluto@{tx_ip}] hardware_rrc={'on' if _get_pluto_hardware_rrc(tx_ip) else 'off'}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

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

frequency = args.freq
if args.mode in ("tx", "both"):
    tx_sdr = adi.Pluto("ip:" + tx_ip)
    configure_tx(tx_sdr, freq=frequency, gain=args.gain, cyclic=False)

if args.mode in ("rx", "both"):
    rx_sdr = adi.Pluto("ip:" + rx_ip)
    configure_rx(rx_sdr, freq=frequency + rx_cfo_hz, gain_mode="slow_attack")

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
    print(f"TX radio  : {tx_ip}   @ {frequency / 1e6:.3f} MHz")
if args.mode in ("rx", "both"):
    print(f"RX radio  : {rx_ip}   @ {(frequency + rx_cfo_hz) / 1e6:.3f} MHz  "
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
print(f"TX gain   : {args.gain} dB\n")

# ---------------------------------------------------------------------------
# TX helpers
# ---------------------------------------------------------------------------

def _build_packet(seq: int, payload_bytes: int) -> np.ndarray:
    """Build a single packet with the given seq number and payload size.

    Returns DAC-scaled complex64 samples ready for TxStream.send().
    """
    seq = seq % (2**32)
    seq_bytes     = np.array([(seq >> 24) & 0xFF, (seq >> 16) & 0xFF,
                               (seq >>  8) & 0xFF,  seq        & 0xFF], dtype=np.uint8)
    sequence_bits = np.unpackbits(seq_bytes)
    random_bits   = rng.integers(0, 2, payload_bytes * 8 - 32, dtype=np.uint8)
    payload_bits  = np.concatenate([sequence_bits, random_bits])
    pkt     = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=payload_bytes, payload=payload_bits)
    samples = tx_pipe.transmit(pkt)
    peak    = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak
    return (samples * DAC_SCALE).astype(np.complex64)


# ---------------------------------------------------------------------------
# TX mode — continuous streaming via TxStream
# ---------------------------------------------------------------------------

def run_tx():
    stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size)
    stream.start()
    print(f"  [TX] streaming (buf={tx_buf_size} samples / "
          f"{tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms, "
          f"variable={args.variable})")

    global_seq = 0
    burst_num  = 0

    while True:
        burst_num  += 1
        burst_start = global_seq
        t0 = time.perf_counter()

        for i in range(args.packets):
            if args.variable:
                pay_size = rng.integers(args.min_payload, args.payload + 1)
            else:
                pay_size = args.payload
            samples = _build_packet(global_seq, pay_size)
            stream.send(samples)
            global_seq += 1

        t1 = time.perf_counter()
        print(f"  [TX] burst {burst_num}: {args.packets} packets "
              f"(seq {burst_start}–{global_seq - 1}) queued in {t1 - t0:.3f} s  "
              f"(pending={stream.pending}, bufs_sent={stream.bufs_sent})")

        time.sleep(args.interval / 1000.0)

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
    # Lossless stream: large queue so the hardware reader never stalls while
    # the decoder is busy.
    stream = RxStream(rx_sdr, maxsize=128, lossless=True)
    stream.start(flush=16)

    print("  [RX] listening …")

    # --- optional constellation plot ---
    _fig = _ax = _scat = None
    _sym_buf: list[np.ndarray] = []   # post-Costas symbols accumulated across packets
    _pkt_count = 0                    # valid packets since last plot refresh
    _PLOT_EVERY = 20                  # update plot every N valid packets
    if args.constellation:
        import matplotlib.pyplot as plt
        _fig, _ax, _scat = _setup_constellation_plot()

    n_total     = 0
    n_valid     = 0
    n_dropped   = 0
    last_seq    = None
    prev_buf    = None
    search_from = 0

    # Processing time stats (printed every 500 buffers)
    _proc_times: list[float] = []
    _buf_count  = 0

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
            packets, max_det = rx_pipe.receive(raw, search_from=search_from)
            _proc_ms = (time.perf_counter() - _t0) * 1e3
            _proc_times.append(_proc_ms)
            _buf_count += 1
            if _buf_count % 500 == 0:
                buf_dur_ms = rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3
                print(f"  [RX perf] proc={np.mean(_proc_times):.2f} ms avg / "
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

                    gap = 0
                    if last_seq is not None:
                        gap = (seq - last_seq - 1) % (2**32)
                        if gap > 0:
                            n_dropped += gap
                    last_seq = seq

                    gap_str = f"  *** GAP: {gap} dropped ***" if gap > 0 else ""
                    print(f"  [RX] #{n_total}  seq={seq:10d}  valid=True   "
                          f"(ok={n_valid}, dropped≈{n_dropped}){gap_str}")

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
                    print(f"  [RX] #{n_total}  header CRC failed  "
                          f"(ok={n_valid}, dropped≈{n_dropped})")

    except KeyboardInterrupt:
        print(f"\n  [RX] interrupted — decoded {n_valid} valid / {n_total} total frames, ~{n_dropped} dropped by seq gap")
    finally:
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

    # -----------------------------------------------------------------------
    # TX thread
    # -----------------------------------------------------------------------

    def tx_thread():
        rx_ready.wait()

        stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size)
        stream.start()

        t0 = time.perf_counter()
        for seq in range(args.packets):
            if args.variable:
                pay_size = rng.integers(args.min_payload, args.payload + 1)
            else:
                pay_size = args.payload
            samples = _build_packet(seq, pay_size)
            stream.send(samples)

        # Wait for the stream to drain the queue
        while stream.pending > 0:
            time.sleep(0.01)
        # One more air_time to let the last buffer finish transmitting
        time.sleep(tx_buf_size / pipe_cfg.SAMPLE_RATE)

        t1 = time.perf_counter()
        stream.stop()

        print(f"Took: {t1 - t0:.3f} seconds. Throughput: {args.packets * args.payload / (t1 - t0):.0f} B/s")
        tx_done.set()
        print("  [TX] done")

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

            packets, max_det = rx_pipe.receive(raw, search_from=search_from)

            prev_buf = curr_buf
            if packets:
                last_ps = max(pkt.sample_start for pkt in packets)
                search_from = max(0, max(last_ps, max_det) - prev_len)
            else:
                search_from = max(0, max_det - prev_len)

            for pkt in packets:
                if pkt.valid:
                    seq_bits = pkt.payload[:32]
                    b        = np.packbits(seq_bits)
                    seq      = (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])
                else:
                    seq = -1
                entry    = {"seq_num": seq, "valid": pkt.valid, "time": time.perf_counter()}
                if pkt.valid:
                    print(f"  [RX] decoded seq={seq:10d}, valid={pkt.valid}")
                else:
                    print(f"  [RX] frame found but header CRC failed")
                with rx_lock:
                    rx_results.append(entry)
                    if pkt.valid:
                        _decoded_seqs.add(seq)
                        if len(_decoded_seqs) >= args.packets:
                            _all_decoded.set()

        stream.stop()
        print("  [RX] done")

    # -----------------------------------------------------------------------
    # Launch threads
    # -----------------------------------------------------------------------

    t_tx = threading.Thread(target=tx_thread, name="TX", daemon=True)
    t_rx = threading.Thread(target=rx_thread, name="RX", daemon=True)

    t_rx.start()
    t_tx.start()

    t_tx.join()
    t_rx.join(timeout=args.packets * args.interval / 1000.0 + _air_time_ms * 4 / 1000.0 + 15.0)

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

sys.exit(0)

