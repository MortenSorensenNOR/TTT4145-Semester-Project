"""Measure the per-direction CFO between the two nodes' radios.

For each air path (A_TX → B_RX, B_TX → A_RX) the script:

  1. Transmits a CW tone from the source Pluto at a known baseband offset.
  2. Captures one long buffer on the receiving Pluto.
  3. FFT-peak-picks the spectrum.
  4. Subtracts the intentional baseband offset — what's left is the LO
     frequency error between the two oscillators.

The two measurements are medianed across several captures and written
into the ``cfo`` block of ``pluto/setup.json``. The bridge reads the
file at startup and auto-applies the correction to each node's RX LO.

Usage::

    uv run python scripts/cfo_calibrate.py            # both directions
    uv run python scripts/cfo_calibrate.py --node A   # only A_TX → B_RX
    uv run python scripts/cfo_calibrate.py --node B   # only B_TX → A_RX
    uv run python scripts/cfo_calibrate.py --captures 20

    # Split-host: TX and RX Plutos hang off different machines.
    # On the TX host (start this first, leave it running):
    uv run python scripts/cfo_calibrate.py --node A --role tx
    # On the RX host (this measures and updates the local setup.json):
    uv run python scripts/cfo_calibrate.py --node A --role rx

Assumes the Plutos involved in the requested direction(s) are reachable
from this host at the IPs in the ``nodes`` block of ``pluto/setup.json``.
The default (no ``--node``) requires all four; ``--node A`` only opens
A's TX and B's RX; ``--node B`` only opens B's TX and A's RX. When only
one direction is measured, the other direction's existing value in the
calibration file is preserved. With ``--role tx`` only the source Pluto
is opened (transmits cyclically until Ctrl-C); with ``--role rx`` only
the destination Pluto is opened, and the result is written to the local
calibration file.
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
    PIPELINE,
    get_node_freqs,
)
from pluto.setup_config import SETUP_PATH, CFOCalibration, load_or_die as load_setup, save_cfo

SAMPLE_RATE    = PIPELINE.SAMPLE_RATE
BUF_SIZE       = 131_072         # ~30 Hz FFT bin at 4 Msps — fine enough for LO drift
TONE_OFFSET_HZ = 100_000         # baseband tone offset (avoids DC-leakage peak)
TX_GAIN_DB     = -10             # a strong tone; the receiver has attenuator headroom
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
    sdr.gain_control_mode_chan0 = "slow_attack"
    # sdr.gain_control_mode_chan0 = "manual"
    # sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
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


def transmit_only(tx_sdr: adi.Pluto, freq_hz: int, label: str) -> None:
    """Configure TX, kick off the cyclic tone, block until Ctrl-C."""
    print(f"\n── {label}  @ {freq_hz / 1e6:.3f} MHz  (TX, Ctrl-C to stop) ──")
    _configure_tx(tx_sdr, freq_hz, TX_GAIN_DB)
    tone = _make_tone(BUF_SIZE, TONE_OFFSET_HZ)
    tx_sdr.tx(tone)
    print(f"  transmitting CW tone at baseband +{TONE_OFFSET_HZ / 1e3:.1f} kHz")
    print("  press Ctrl-C once the RX side has reported its result")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  stopping TX")


def receive_only(rx_sdr: adi.Pluto, freq_hz: int, *, captures: int, label: str) -> int:
    """Configure RX, capture, return the median CFO in Hz."""
    print(f"\n── {label}  @ {freq_hz / 1e6:.3f} MHz  (RX) ──")
    _configure_rx(rx_sdr, freq_hz)

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


def _run_split_host(args: argparse.Namespace, setup) -> int:
    """Handle --role tx|rx: open only one Pluto on this host."""
    label    = "A_TX → B_RX" if args.node == "A" else "B_TX → A_RX"
    freq_hz  = get_node_freqs(args.node, video=args.video)["tx"]
    print(f"Frequency plan: {'video' if args.video else 'network'} "
          f"({label} {freq_hz / 1e6:.3f} MHz)")

    if args.role == "tx":
        uri = setup.tx_uri(args.node)
        print(f"Opening TX radio: {uri}")
        sdr = adi.Pluto(uri)
        try:
            transmit_only(sdr, freq_hz, label)
        finally:
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
        return 0

    # role == "rx": open the *destination* node's RX Pluto.
    rx_node = "B" if args.node == "A" else "A"
    uri = setup.rx_uri(rx_node)
    print(f"Opening RX radio: {uri}")
    sdr = adi.Pluto(uri)
    cfo_hz = receive_only(sdr, freq_hz, captures=args.captures, label=label)

    prev = setup.cfo
    if args.node == "A":
        a_to_b = cfo_hz
        if prev is not None:
            b_to_a = prev.b_to_a_cfo_hz
        else:
            b_to_a = 0
            print("[warn] no existing B_TX→A_RX calibration; saving 0 for that direction")
    else:
        b_to_a = cfo_hz
        if prev is not None:
            a_to_b = prev.a_to_b_cfo_hz
        else:
            a_to_b = 0
            print("[warn] no existing A_TX→B_RX calibration; saving 0 for that direction")

    cal = CFOCalibration(a_to_b_cfo_hz=a_to_b, b_to_a_cfo_hz=b_to_a)
    print("\n── Result ──")
    a_marker = "  " if args.node == "A" else "* "
    b_marker = "  " if args.node == "B" else "* "
    print(f"{a_marker}A_TX → B_RX :  {cal.a_to_b_cfo_hz:+d} Hz   (B's RX LO will be tuned up by this)")
    print(f"{b_marker}B_TX → A_RX :  {cal.b_to_a_cfo_hz:+d} Hz   (A's RX LO will be tuned up by this)")
    print("  (* = preserved from prior calibration, not re-measured)")

    if args.dry_run:
        print("\n[dry-run] not writing calibration file")
    else:
        save_cfo(cal, args.output)
        print(f"\nUpdated cfo block in {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node",     choices=["A", "B"], default=None,
                        help="Only calibrate one direction: 'A' measures A_TX→B_RX, "
                             "'B' measures B_TX→A_RX. Default: both directions.")
    parser.add_argument("--role",     choices=["tx", "rx"], default=None,
                        help="Split-host mode: 'tx' opens only the source Pluto and "
                             "transmits the calibration tone cyclically until Ctrl-C; "
                             "'rx' opens only the destination Pluto, measures, and "
                             "writes the result for that one direction. Requires --node.")
    parser.add_argument("--captures", type=int, default=10,
                        help="Buffers to median over, per direction (default: 10)")
    parser.add_argument("--output",   type=Path, default=SETUP_PATH,
                        help=f"Setup JSON path to update (default: {SETUP_PATH})")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Measure and print, but don't overwrite the calibration file.")
    parser.add_argument("--video",    action="store_true",
                        help="Calibrate against the video-mode FDD pair (2327/2390 MHz) "
                             "instead of the default network pair (2470/2475 MHz).")
    args = parser.parse_args()

    if args.role is not None and args.node is None:
        parser.error("--role requires --node (which direction this host is handling)")

    setup = load_setup(args.output)

    if args.role is not None:
        return _run_split_host(args, setup)

    do_a   = args.node in (None, "A")
    do_b   = args.node in (None, "B")

    a_tx_uri = setup.tx_uri("A") if do_a else None
    b_rx_uri = setup.rx_uri("B") if do_a else None
    b_tx_uri = setup.tx_uri("B") if do_b else None
    a_rx_uri = setup.rx_uri("A") if do_b else None

    print("Opening radios:")
    if do_a:
        print(f"  A TX @ {a_tx_uri}")
        print(f"  B RX @ {b_rx_uri}")
    if do_b:
        print(f"  B TX @ {b_tx_uri}")
        print(f"  A RX @ {a_rx_uri}")

    a_tx = adi.Pluto(a_tx_uri) if do_a else None
    b_rx = adi.Pluto(b_rx_uri) if do_a else None
    b_tx = adi.Pluto(b_tx_uri) if do_b else None
    a_rx = adi.Pluto(a_rx_uri) if do_b else None

    # A's TX channel == B's RX channel == "tx" entry of node A's freq plan.
    freq_a_to_b = get_node_freqs("A", video=args.video)["tx"]
    freq_b_to_a = get_node_freqs("B", video=args.video)["tx"]
    print(f"Frequency plan: {'video' if args.video else 'network'} "
          f"(A→B {freq_a_to_b / 1e6:.3f} MHz, B→A {freq_b_to_a / 1e6:.3f} MHz)")

    a_to_b: int | None = None
    b_to_a: int | None = None
    try:
        if do_a:
            a_to_b = measure_path(a_tx, b_rx, freq_a_to_b,
                                  captures=args.captures, label="A_TX → B_RX")
        if do_b:
            b_to_a = measure_path(b_tx, a_rx, freq_b_to_a,
                                  captures=args.captures, label="B_TX → A_RX")
    finally:
        # Paranoid cleanup — either TX destroy may have already fired.
        for sdr in (a_tx, b_tx):
            if sdr is not None:
                try:
                    sdr.tx_destroy_buffer()
                except Exception:
                    pass

    # When only one direction was measured, preserve the other from any
    # existing calibration so we don't clobber a previously-good value with 0.
    prev = setup.cfo
    if a_to_b is None:
        a_to_b = prev.a_to_b_cfo_hz if prev is not None else 0
        if prev is None:
            print("[warn] no existing A_TX→B_RX calibration; saving 0 for that direction")
    if b_to_a is None:
        b_to_a = prev.b_to_a_cfo_hz if prev is not None else 0
        if prev is None:
            print("[warn] no existing B_TX→A_RX calibration; saving 0 for that direction")

    cal = CFOCalibration(a_to_b_cfo_hz=a_to_b, b_to_a_cfo_hz=b_to_a)
    print("\n── Result ──")
    a_marker = "  " if do_a else "* "  # mark preserved-from-prior values
    b_marker = "  " if do_b else "* "
    print(f"{a_marker}A_TX → B_RX :  {cal.a_to_b_cfo_hz:+d} Hz   (B's RX LO will be tuned up by this)")
    print(f"{b_marker}B_TX → A_RX :  {cal.b_to_a_cfo_hz:+d} Hz   (A's RX LO will be tuned up by this)")
    if not (do_a and do_b):
        print("  (* = preserved from prior calibration, not re-measured)")

    if args.dry_run:
        print("\n[dry-run] not writing calibration file")
    else:
        save_cfo(cal, args.output)
        print(f"\nUpdated cfo block in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
