"""Persistent CFO calibration.

Each node uses split TX/RX Plutos whose LO oscillators drift independently.
Instead of eyeballing an offset and passing it on the command line on every
run, ``scripts/cfo_calibrate.py`` measures the two air-path CFOs (A→B and
B→A) once and writes them here; the bridge then auto-loads the file and
applies the measured offset to each node's **RX LO** (the TX LO stays at
its natural frequency, and the receiver tunes to meet the transmitter).

The drift is dominated by per-oscillator part-to-part variation, so values
are stable across boots for a fixed set of Plutos.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default calibration file lives next to this module.
CFO_CONFIG_PATH: Path = Path(__file__).resolve().parent / "cfo_calibration.json"


@dataclass
class CFOCalibration:
    """Measured per-direction CFO, in Hz.

    ``a_to_b_cfo_hz``
        Offset of A's TX carrier as seen at B's RX (positive = A's emitted
        frequency is higher than B's RX nominal LO). B should tune its RX
        LO up by this amount to centre on the signal.
    ``b_to_a_cfo_hz``
        Same, flipped — A tunes its RX LO by this to hear B.
    """

    a_to_b_cfo_hz: int
    b_to_a_cfo_hz: int
    measured_at: str = ""   # ISO-8601 timestamp, informational
    version: int = 1

    def rx_offset_for(self, node: str) -> int:
        """Offset to add to the node's RX LO (Hz)."""
        if node == "A":
            return self.b_to_a_cfo_hz
        if node == "B":
            return self.a_to_b_cfo_hz
        raise ValueError(f"unknown node {node!r} (expected 'A' or 'B')")


def save(cal: CFOCalibration, path: Path = CFO_CONFIG_PATH) -> None:
    """Write the calibration to ``path`` as pretty-printed JSON."""
    if not cal.measured_at:
        cal.measured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cal), indent=2) + "\n")


def load(path: Path = CFO_CONFIG_PATH) -> CFOCalibration | None:
    """Load the calibration from ``path``. Returns None if missing / unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return CFOCalibration(
            a_to_b_cfo_hz=int(data["a_to_b_cfo_hz"]),
            b_to_a_cfo_hz=int(data["b_to_a_cfo_hz"]),
            measured_at=str(data.get("measured_at", "")),
            version=int(data.get("version", 1)),
        )
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("Could not read CFO calibration %s: %s", path, exc)
        return None
