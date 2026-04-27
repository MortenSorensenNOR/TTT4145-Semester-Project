"""Measure the per-direction CFO between the two nodes' radios.

For each air path (A_TX → B_RX, B_TX → A_RX) the script:

  1. Transmits a CW tone from the source Pluto at a known baseband offset.
  2. Captures one long buffer on the receiving Pluto.
  3. FFT-peak-picks the spectrum.
  4. Subtracts the intentional baseband offset — what's left is the LO
     frequency error between the two oscillators.

The two measurements are medianed across several captures and written
into the ``cfo`` block of ``pluto/setup.json``. The bridge reads the
file at startup and auto-applies the correction to each node's RX LO,
so you don't have to pass ``--cfo-offset`` every run.

Usage::

    uv run python scripts/cfo_calibrate.py            # measure both directions
    uv run python scripts/cfo_calibrate.py --captures 20

Assumes all four Plutos are reachable from this host at the IPs given
in the ``nodes`` block of ``pluto/setup.json`` (same layout the bridge
uses).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import adi
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pluto.config import (
    DAC_SCALE,
    FREQ_A_TO_B,
    FREQ_B_TO_A,
    PIPELINE,
)
from pluto.setup_config import SETUP_PATH, CFOCalibration, load_or_die as load_setup, save_cfo

SAMPLE_RATE    = PIPELINE.SAMPLE_RATE
BUF_SIZE       = 131_072         # ~30 Hz FFT bin at 4 Msps — fine enough for LO drift
TONE_OFFSET_HZ = 100_000         # baseband tone offset (avoids DC-leakage peak)
RX_GAIN_DB     = 50              # manual RX gain — plenty of margin over coax / short link
TX_GAIN_DB     = -20             # a strong tone; the receiver has attenuator headroom
FLUSH_BUFFERS  = 8               # discard this many RX buffers to clear DMA + settle AGC
SETTLE_SECONDS = 0.2             # give TX DMA a moment before first RX capture


def _make_tone(buf_size: int, tone_hz: float) -> np.ndarray:
    """DAC-scaled complex CW tone at `tone_hz` in baseband (one period repeats cleanly)."""
    t = np.arange(buf_size, dtype=np.float64) / SAMPLE_RATE
    tone = np.exp(2j * np.pi * tone_hz * t)
    # Use 0.9 * DAC_SCALE so we stay well below the clipping limit.
    return (0.9 * DAC_SCALE * tone).astype(np.complex64)


def _configure_tx(sdr: adi.Pluto, freq_hz: int, gain_db: float) -> None:
    sdr.sample_rate = SAMPLE_RATE
    sdr.tx_rf_bandwidth = SAMPLE_RATE
    sdr.tx_lo = int(freq_hz)
    sdr.tx_hardwaregain_chan0 = gain_db
    sdr.tx_cyclic_buffer = True


def _configure_rx(sdr: adi.Pluto, freq_hz: int) -> None:
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
    sdr.rx_lo = int(freq_hz)
    sdr.sample_rate = SAMPLE_RATE
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_buffer_size = BUF_SIZE


def _peak_offset_hz(samples: np.ndarray) -> float:
    """FFT peak frequency of the given samples, in Hz (signed)."""
    window   = np.hanning(len(samples))
    spectrum = np.fft.fftshift(np.fft.fft(samples * window))
    freqs    = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1.0 / SAMPLE_RATE))
    power    = np.abs(spectrum)
    peak_bin = int(np.argmax(power))
    # Quadratic interpolation around the peak for sub-bin resolution.
    if 0 < peak_bin < len(power) - 1:
        a, b, c = power[peak_bin - 1], power[peak_bin], power[peak_bin + 1]
        denom = (a - 2 * b + c)
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    else:
        shift = 0.0
    bin_hz = SAMPLE_RATE / len(samples)
    return float(freqs[peak_bin]) + shift * bin_hz


def measure_path(
    tx_sdr: adi.Pluto,
    rx_sdr: adi.Pluto,
    freq_hz: int,
    *,
    captures: int,
    label: str,
) -> int:
    """Measure the LO error along one TX→RX path. Returns median CFO in Hz."""
    print(f"\n── {label}  @ {freq_hz / 1e6:.3f} MHz ──")

    _configure_tx(tx_sdr, freq_hz, TX_GAIN_DB)
    _configure_rx(rx_sdr, freq_hz)

    tone = _make_tone(BUF_SIZE, TONE_OFFSET_HZ)
    tx_sdr.tx(tone)              # cyclic buffer — transmits continuously

    try:
        time.sleep(SETTLE_SECONDS)
        for _ in range(FLUSH_BUFFERS):
            rx_sdr.rx()

        offsets: list[float] = []
        for i in range(captures):
            samples = np.array(rx_sdr.rx(), dtype=np.complex64)
            peak = _peak_offset_hz(samples)
            cfo  = peak - TONE_OFFSET_HZ
            offsets.append(cfo)
            print(f"  capture {i+1:3d}/{captures}: peak={peak:+10.1f} Hz  cfo={cfo:+10.1f} Hz")

        med = float(np.median(offsets))
        std = float(np.std(offsets))
        print(f"  → median CFO = {med:+.1f} Hz  (σ = {std:.1f} Hz)")
        return int(round(med))
    finally:
        try:
            tx_sdr.tx_destroy_buffer()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", type=int, default=10,
                        help="Buffers to median over, per direction (default: 10)")
    parser.add_argument("--output",   type=Path, default=SETUP_PATH,
                        help=f"Setup JSON path to update (default: {SETUP_PATH})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Measure and print, but don't overwrite the calibration file.")
    args = parser.parse_args()

    setup = load_setup(args.output)
    a_tx_ip = setup.tx_ip("A")
    a_rx_ip = setup.rx_ip("A")
    b_tx_ip = setup.tx_ip("B")
    b_rx_ip = setup.rx_ip("B")

    print("Opening radios:")
    print(f"  A TX @ {a_tx_ip}")
    print(f"  A RX @ {a_rx_ip}")
    print(f"  B TX @ {b_tx_ip}")
    print(f"  B RX @ {b_rx_ip}")

    a_tx = adi.Pluto(f"ip:{a_tx_ip}")
    a_rx = adi.Pluto(f"ip:{a_rx_ip}")
    b_tx = adi.Pluto(f"ip:{b_tx_ip}")
    b_rx = adi.Pluto(f"ip:{b_rx_ip}")

    try:
        a_to_b = measure_path(a_tx, b_rx, FREQ_A_TO_B,
                              captures=args.captures, label="A_TX → B_RX")
        b_to_a = measure_path(b_tx, a_rx, FREQ_B_TO_A,
                              captures=args.captures, label="B_TX → A_RX")
    finally:
        # Paranoid cleanup — either TX destroy may have already fired.
        for sdr in (a_tx, b_tx):
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass

    cal = CFOCalibration(a_to_b_cfo_hz=a_to_b, b_to_a_cfo_hz=b_to_a)
    print("\n── Result ──")
    print(f"  A_TX → B_RX :  {cal.a_to_b_cfo_hz:+d} Hz   (B's RX LO will be tuned up by this)")
    print(f"  B_TX → A_RX :  {cal.b_to_a_cfo_hz:+d} Hz   (A's RX LO will be tuned up by this)")

    if args.dry_run:
        print("\n[dry-run] not writing calibration file")
    else:
        save_cfo(cal, args.output)
        print(f"\nUpdated cfo block in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
