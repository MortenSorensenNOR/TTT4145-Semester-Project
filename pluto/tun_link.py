"""TUN-mode radio link — Linux network interface over the SDR (no ARQ).

Per-node, full-duplex bridge between a Linux TUN device and the radio.
Packets that fail to decode are dropped — there's no retransmit layer.

Each node opens a dedicated TX Pluto and a dedicated RX Pluto (IPs from
pluto/setup.json), creates a TUN at /dev/net/tun, and runs three threads:

  * tun-rd : reads IP packets from TUN, hands them to the TX queue
  * tx     : pulls packets, builds Packet, transmits via TxStream
  * rx     : drains RxStream, decodes frames, writes payload to TUN

Default IP plan:
  --node A → TUN pluto0 = 10.0.0.1/24
  --node B → TUN pluto0 = 10.0.0.2/24

Usage:
    sudo .venv/bin/python -m pluto.tun_link --node A
    sudo .venv/bin/python -m pluto.tun_link --node B
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

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
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
logging.basicConfig(level=logging.INFO)

# FDD frequency plan: same as pluto.one_way_threaded — A transmits on
# FREQ_A_TO_B and listens on FREQ_B_TO_A; B does the opposite.
NODE_FREQS = {
    "A": {"tx": FREQ_A_TO_B, "rx": FREQ_B_TO_A},
    "B": {"tx": FREQ_B_TO_A, "rx": FREQ_A_TO_B},
}

# 1-bit logical MAC per node — copied into Packet.src_mac / .dst_mac so the
# RX side can drop frames whose dst doesn't match (self-reception / leakage).
NODE_ADDR = {"A": 0, "B": 1}

DEFAULT_TUN_IP = {"A": "10.0.0.1", "B": "10.0.0.2"}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--node",     type=str,   default="A",    help="Node identity A or B; picks default TX/RX IPs from pluto/setup.json and TUN IP from DEFAULT_TUN_IP")
parser.add_argument("--gain",     type=float, default=-10,    help="TX gain in dB (default: -10)")
parser.add_argument("--tx-ip",    type=str,   default=None,   help="Override TX Pluto IP (default: derived from --node via pluto/setup.json)")
parser.add_argument("--rx-ip",    type=str,   default=None,   help="Override RX Pluto IP (default: derived from --node via pluto/setup.json)")
parser.add_argument("--tx-freq",  type=float, default=None,   help="TX center frequency in Hz (default: NODE_FREQS[node]['tx'])")
parser.add_argument("--rx-freq",  type=float, default=None,   help="RX center frequency in Hz (default: NODE_FREQS[node]['rx'])")
parser.add_argument("--cfo-offset", type=int, default=None,
                    help="Manual override for the RX-LO CFO correction in Hz. "
                         "Default: value from the cfo block of pluto/setup.json "
                         "for --node (run scripts/cfo_calibrate.py to generate "
                         "it), or 0 if no calibration is present.")
parser.add_argument("--tx-buf-mult", type=int, default=8,     help="TX buffer size as multiple of next-power-of-2 frame length (default: 8)")
parser.add_argument("--hardware-rrc", action="store_true",    help="Use the FPGA hardware RRC/4x interpolation path on TX (toggles the pluto_custom firmware GPIO).")
parser.add_argument("--tun-name", type=str,   default="pluto0", help="TUN interface name (default: pluto0)")
parser.add_argument("--tun-ip",   type=str,   default=None,   help="TUN IPv4 address with /24 implicit (default: 10.0.0.1 for A, 10.0.0.2 for B)")
parser.add_argument("--mtu",      type=int,   default=1500,   help="TUN MTU in bytes (default: 1500)")
parser.add_argument("--queue-depth", type=int, default=64,    help="TUN→TX queue depth before drops (default: 64)")
args = parser.parse_args()

setup = load_setup()
if args.node not in setup.nodes:
    print(f"ERROR: --node must be one of {sorted(setup.nodes)}, got '{args.node}'")
    sys.exit(1)
if args.node not in NODE_ADDR:
    print(f"ERROR: --node must be 'A' or 'B' for the TUN-link bridge, got '{args.node}'")
    sys.exit(1)

tx_ip = args.tx_ip or setup.tx_ip(args.node)
rx_ip = args.rx_ip or setup.rx_ip(args.node)

# Resolve RX-LO CFO offset: manual CLI override wins; otherwise pull the
# measured value for this node from the calibration; otherwise 0.
if args.cfo_offset is not None:
    rx_cfo_hz = args.cfo_offset
    cfo_src   = "cli"
elif setup.cfo is None:
    rx_cfo_hz = 0
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

pipe_cfg = PipelineConfig(hardware_rrc=args.hardware_rrc)
# Same gate as one_way_threaded: rejects spurious detections on TX-buffer tail
# silence whose ratios sit ~4–5 vs ~10–12 for real packets.
pipe_cfg.SYNC_CONFIG.fine_peak_ratio_min = np.float32(7.0)
tx_pipe = TXPipeline(pipe_cfg)
rx_pipe = RXPipeline(pipe_cfg)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# SDR setup — always full-duplex (TX and RX Pluto open simultaneously).
# ---------------------------------------------------------------------------

tx_freq = int(args.tx_freq) if args.tx_freq is not None else NODE_FREQS[args.node]["tx"]
rx_freq = int(args.rx_freq) if args.rx_freq is not None else NODE_FREQS[args.node]["rx"]

tx_sdr = adi.Pluto("ip:" + tx_ip)
configure_tx(tx_sdr, freq=tx_freq, gain=args.gain, cyclic=False)

rx_sdr = adi.Pluto("ip:" + rx_ip)
configure_rx(rx_sdr, freq=rx_freq + rx_cfo_hz, gain_mode="slow_attack")

# ---------------------------------------------------------------------------
# Buffer sizing — probe one MTU-sized packet to learn frame_len.
# ---------------------------------------------------------------------------

_probe_bits    = rng.integers(0, 2, args.mtu * 8, dtype=np.uint8)
_probe_pkt     = Packet(src_mac=my_addr, dst_mac=peer_addr, type=0, seq_num=0,
                        length=args.mtu, payload=_probe_bits)
_probe_samples = tx_pipe.transmit(_probe_pkt)
frame_len      = len(_probe_samples)
rx_buf_size    = 16 * int(2 ** np.ceil(np.log2(frame_len)))
rx_sdr.rx_buffer_size = rx_buf_size
tx_buf_size    = args.tx_buf_mult * int(2 ** np.ceil(np.log2(frame_len)))

print(f"Node      : {args.node}  (peer {peer})")
print(f"TUN       : {args.tun_name} = {tun_ip}/24  (peer {peer_tun_ip})  MTU {args.mtu}")
print(f"TX radio  : {tx_ip}   @ {tx_freq / 1e6:.3f} MHz")
print(f"RX radio  : {rx_ip}   @ {(rx_freq + rx_cfo_hz) / 1e6:.3f} MHz  "
      f"(CFO {rx_cfo_hz:+d} Hz, {cfo_src})")
print(f"Pipeline  : SPS={pipe_cfg.SPS}, alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
print(f"Frame len : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"TX buf    : {tx_buf_size} samples  ({tx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"RX buf    : {rx_buf_size} samples  ({rx_buf_size / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"TX gain   : {args.gain} dB\n")

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
status = LiveStatus(n_lines=2)
tx_rate = RateMeter()
rx_rate = RateMeter()
_install_live_logging(status)

stats = {
    "tun_in":      0,  # IP packets pulled from TUN
    "tun_out":     0,  # IP packets written back to TUN
    "tun_dropped": 0,  # TUN reads dropped because send_q was full
    "data_rx_ok":  0,  # valid frames received and accepted
    "data_rx_foreign": 0,  # valid frames whose dst_mac wasn't ours
    "data_rx_invalid": 0,  # frames whose payload CRC failed
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


def tx_thread_fn():
    stream = TxStream(tx_sdr, pipe_cfg.SAMPLE_RATE, tx_buf_size)
    stream.start()
    try:
        while not stop_event.is_set():
            try:
                payload = send_q.get(timeout=0.05)
            except queue.Empty:
                continue

            bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
            pkt = Packet(
                src_mac=my_addr,
                dst_mac=peer_addr,
                type=0,
                seq_num=0,
                length=len(payload),
                payload=bits,
            )
            samples = tx_pipe.transmit(pkt)
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak
            stream.send((samples * DAC_SCALE).astype(np.complex64))

            tx_rate.add(len(payload))
            status.set(0, f"  [TX] in={stats['tun_in']:>8d}  drop={stats['tun_dropped']:>4d}  "
                          f"pending={stream.pending:>3d}  "
                          f"rate={_fmt_rate(tx_rate.rate_bps)}  "
                          f"avg={_fmt_rate(tx_rate.avg_bps)}  "
                          f"total={_fmt_bytes(tx_rate.total_bytes)}")
    finally:
        stream.stop()


def rx_thread_fn():
    stream = RxStream(rx_sdr, maxsize=128, lossless=True)
    stream.start(flush=16)

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

            prev_buf = curr_buf
            if packets:
                last_ps = max(p.sample_start for p in packets)
                search_from = max(0, max(last_ps, max_det) - prev_len)
            else:
                search_from = max(0, max_det - prev_len)

            for pkt in packets:
                if not pkt.valid:
                    stats["data_rx_invalid"] += 1
                    continue
                if pkt.dst_mac != my_addr:
                    stats["data_rx_foreign"] += 1
                    continue

                data = np.packbits(pkt.payload[:pkt.length * 8].astype(np.uint8)).tobytes()
                try:
                    tun.write(data)
                except OSError:
                    return

                stats["data_rx_ok"] += 1
                stats["tun_out"]    += 1
                rx_rate.add(pkt.length)
                status.set(1, f"  [RX] ok={stats['data_rx_ok']:>8d}  "
                              f"bad={stats['data_rx_invalid']:>4d}  "
                              f"foreign={stats['data_rx_foreign']:>4d}  "
                              f"q={stream._q.qsize():>3d}/{stream._q.maxsize}  "
                              f"rate={_fmt_rate(rx_rate.rate_bps)}  "
                              f"avg={_fmt_rate(rx_rate.avg_bps)}  "
                              f"total={_fmt_bytes(rx_rate.total_bytes)}")
    finally:
        stream.stop()


t_tun = threading.Thread(target=tun_reader_thread, name="tun-rd", daemon=True)
t_tx  = threading.Thread(target=tx_thread_fn,      name="tx",     daemon=True)
t_rx  = threading.Thread(target=rx_thread_fn,      name="rx",     daemon=True)

t_rx.start()
t_tx.start()
t_tun.start()

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
    for t in (t_tun, t_tx, t_rx):
        t.join(timeout=2.0)

    print()
    print("=" * 50)
    print("FINAL STATS")
    print("=" * 50)
    print(f"TUN in       : {stats['tun_in']}")
    print(f"TUN out      : {stats['tun_out']}")
    print(f"TUN dropped  : {stats['tun_dropped']}  (send queue full)")
    print(f"RX valid     : {stats['data_rx_ok']}")
    print(f"RX bad CRC   : {stats['data_rx_invalid']}")
    print(f"RX foreign   : {stats['data_rx_foreign']}  (dst_mac != us)")
    print(f"TX bytes     : {tx_rate.total_bytes}  (avg {_fmt_rate(tx_rate.avg_bps)})")
    print(f"RX bytes     : {rx_rate.total_bytes}  (avg {_fmt_rate(rx_rate.avg_bps)})")
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
