"""Loopback bit-accuracy test.

Builds a packet with a known random payload, runs it through the full TX
pipeline (frame construction → modulation → RRC pulse shaping), transmits
over the Pluto loopback, then runs the full RX pipeline (frame sync →
match filter → Costas → demodulation) and verifies the decoded bits
match what was sent.

Usage:
    python pluto/loopback_packet.py [--gain <dB>] [--payload <bytes>] [--trials <n>]

Flags (all optional):
    --gain    TX hardware gain in dB  (default: -30, range: -90 to 0)
    --payload Payload size in bytes   (default: 100)
    --trials  Number of RX captures   (default: 3)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
import adi

from utils.plotting import *
from modules.pulse_shaping import *

from modules.pipeline import PipelineConfig, TXPipeline, RXPipeline, Packet
from pluto.config import (
    CENTER_FREQ,
    DAC_SCALE,
    configure_rx,
    configure_tx,
)
from pluto.sdr_stream import RxStream

N_FLUSH  = 10   # RX flushes before capture to discard stale samples

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain",    type=float, default=-30,  help="TX gain in dB (default: -30)")
parser.add_argument("--payload", type=int,   default=10,  help="Payload bytes (default: 100)")
parser.add_argument("--trials",  type=int,   default=3,    help="Number of RX captures (default: 3)")
parser.add_argument("--ip", type=str, default="192.168.2.1", help="ip of pluto")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Build pipelines
# ---------------------------------------------------------------------------

pipe_cfg = PipelineConfig(hardware_rrc=True)
tx_pipe  = TXPipeline(pipe_cfg)
rx_pipe  = RXPipeline(pipe_cfg)

# ---------------------------------------------------------------------------
# Build test packet with known random payload bits
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)
payload_bits = rng.integers(0, 2, args.payload * 8, dtype=np.uint8)

packet = Packet(
    src_mac=0,
    dst_mac=1,
    type=0,
    seq_num=0,
    length=args.payload,
    payload=payload_bits,
)

# ---------------------------------------------------------------------------
# TX: encode → modulate → RRC pulse shaping
# ---------------------------------------------------------------------------
tx_samples = tx_pipe.transmit(packet)

# Normalize to [-1, 1] then scale to DAC range
peak = np.max(np.abs(tx_samples))
if peak > 0:
    tx_samples = tx_samples / peak
tx_samples = (tx_samples * DAC_SCALE).astype(np.complex64)

frame_len = len(tx_samples)

# RX buffer holds ~1 frame.  The RX loop concatenates the previous buffer with
# the current one (2-buffer sliding window) so frames that straddle a boundary
# are still caught.  Round up to the next power of 2.
rx_buf_size = int(2 ** np.ceil(np.log2(frame_len)))

print(f"\nPipeline config : SPS={pipe_cfg.SPS}, RRC_alpha={pipe_cfg.RRC_ALPHA}, mod={pipe_cfg.MOD_SCHEME.name}")
print(f"Payload         : {args.payload} bytes  ({args.payload * 8} bits)")
print(f"TX frame length : {frame_len} samples  ({frame_len / pipe_cfg.SAMPLE_RATE * 1e3:.1f} ms)")
print(f"RX buffer size  : {rx_buf_size} samples")
print(f"TX gain         : {args.gain} dB\n")

# ---------------------------------------------------------------------------
# Configure SDR
# ---------------------------------------------------------------------------

PLUTO_IP = "ip:" + args.ip
sdr = adi.Pluto(PLUTO_IP)
configure_tx(sdr, freq=CENTER_FREQ, gain=args.gain, cyclic=True)
configure_rx(sdr, freq=CENTER_FREQ, gain_mode="manual")
sdr.rx_buffer_size = rx_buf_size  # override default from configure_rx

# Start cyclic TX (continuously loops the frame)
sdr.tx(tx_samples)

stream = RxStream(sdr)
stream.start()
stream.flush(N_FLUSH)

# ---------------------------------------------------------------------------
# RX trials
# ---------------------------------------------------------------------------

passed = 0
failed = 0
prev_buf = None

for trial in range(args.trials):
    curr_buf = stream.get()
    rx_raw = np.concatenate([prev_buf, curr_buf]) if prev_buf is not None else curr_buf
    prev_buf = curr_buf

    start = time.perf_counter()
    packets = rx_pipe.receive(rx_raw)

    end = time.perf_counter()
    time_rx = end - start
    print(f"Koden tok {time_rx:.4f} sekunder å kjøre.")

    if not packets:
        print(f"[Trial {trial + 1}/{args.trials}] FAIL — no frame detected")
        failed += 1
        continue

    pkt = packets[0]

    if not pkt.valid:
        print(f"[Trial {trial + 1}/{args.trials}] FAIL — frame detected but header CRC failed")
        failed += 1
        continue

    # payload_decode returns raw demodulated bits (shape: (N_syms, bits_per_sym)).
    # The first payload_bytes*8 bits correspond to the original payload.
    rx_bits = pkt.payload.ravel()[: args.payload * 8].astype(np.uint8)
    n_errors = int(np.sum(rx_bits != payload_bits[: len(rx_bits)]))
    ber = n_errors / len(payload_bits) if len(payload_bits) > 0 else float("nan")

    if n_errors == 0:
        print(f"[Trial {trial + 1}/{args.trials}] PASS — {len(rx_bits)} bits correct, BER = 0.0")
        passed += 1
    else:
        print(f"[Trial {trial + 1}/{args.trials}] FAIL — {n_errors}/{len(payload_bits)} bit errors, BER = {ber:.4f}")
        failed += 1

stream.stop()
sdr.tx_destroy_buffer()

print(f"\nResult: {passed}/{args.trials} passed")
sys.exit(0 if failed == 0 else 1)
