"""Transmit a frame continuously (cyclic) for spectrum / loopback testing.

Usage:
    python pluto/tx_continuous.py [--gain <dB>] [--ip <addr>]

Press Ctrl+C to stop.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import adi

from modules.pipeline import PipelineConfig, TXPipeline, Packet
from pluto.config import CENTER_FREQ, DAC_SCALE, configure_tx

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--gain", type=float, default=-30, help="TX gain in dB (default: -30)")
parser.add_argument("--ip",   default="ip:localhost",  help="Pluto URI (default: ip:localhost)")
args = parser.parse_args()

cfg     = PipelineConfig(hardware_rrc=True)
tx_pipe = TXPipeline(cfg)

packet = Packet(src_mac=0, dst_mac=1, type=0, seq_num=0, length=100,
                payload=np.random.randint(0, 2, 100 * 8, dtype=np.uint8))

tx_samples = tx_pipe.transmit(packet)
peak = np.max(np.abs(tx_samples))
tx_samples = (tx_samples / peak * DAC_SCALE).astype(np.complex64)

sdr = adi.Pluto(args.ip)
configure_tx(sdr, freq=CENTER_FREQ, gain=args.gain, cyclic=True)

sdr.tx(tx_samples)
print(f"Transmitting {len(tx_samples)} samples continuously on {CENTER_FREQ/1e9:.3f} GHz at {args.gain} dB gain.")
print("Press Ctrl+C to stop.")

try:
    while True:
        pass
except KeyboardInterrupt:
    sdr.tx_destroy_buffer()
    print("\nStopped.")
