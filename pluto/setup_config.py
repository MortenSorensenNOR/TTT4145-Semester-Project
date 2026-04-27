"""Persistent radio setup: per-node TX/RX Pluto IPs and CFO calibration.

A single JSON file (``pluto/setup.json``) holds everything that varies per
deployment — which Pluto plays which role on each node, plus the measured
LO offsets between the four oscillators. The bridge and one_way scripts
load this on startup; ``scripts/cfo_calibrate.py`` rewrites just the cfo
section after a measurement run.

File schema (version 2)::

    {
      "version": 2,
      "nodes": {
        "A": {"tx_ip": "...", "rx_ip": "..."},
        "B": {"tx_ip": "...", "rx_ip": "..."}
      },
      "cfo": {
        "a_to_b_cfo_hz": <int>,
        "b_to_a_cfo_hz": <int>,
        "measured_at": "<ISO-8601>"
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SETUP_PATH: Path = Path(__file__).resolve().parent / "setup.json"


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
    measured_at: str = ""

    def rx_offset_for(self, node: str) -> int:
        """Offset to add to the node's RX LO (Hz)."""
        if node == "A":
            return self.b_to_a_cfo_hz
        if node == "B":
            return self.a_to_b_cfo_hz
        raise ValueError(f"unknown node {node!r} (expected 'A' or 'B')")


@dataclass
class Setup:
    """Full radio-setup config: TX/RX IPs per node + optional CFO cal."""

    nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    cfo: CFOCalibration | None = None
    version: int = 2

    def tx_ip(self, node: str) -> str:
        return self.nodes[node]["tx"]

    def rx_ip(self, node: str) -> str:
        return self.nodes[node]["rx"]


def load(path: Path = SETUP_PATH) -> Setup:
    """Load the setup from ``path``. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pluto setup file missing: {path}. "
            f"Create it (see pluto/setup_config.py for the schema)."
        )
    data = json.loads(path.read_text())
    nodes = {
        node: {"tx": entry["tx_ip"], "rx": entry["rx_ip"]}
        for node, entry in data["nodes"].items()
    }
    cfo_data = data.get("cfo")
    cfo: CFOCalibration | None = None
    if cfo_data and cfo_data.get("measured_at"):
        cfo = CFOCalibration(
            a_to_b_cfo_hz=int(cfo_data["a_to_b_cfo_hz"]),
            b_to_a_cfo_hz=int(cfo_data["b_to_a_cfo_hz"]),
            measured_at=str(cfo_data.get("measured_at", "")),
        )
    return Setup(nodes=nodes, cfo=cfo, version=int(data.get("version", 2)))


def load_or_die(path: Path = SETUP_PATH) -> Setup:
    """Load the setup or print a clear error and exit."""
    try:
        return load(path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"ERROR loading {path}: {exc}", file=sys.stderr)
        sys.exit(1)


def save(setup: Setup, path: Path = SETUP_PATH) -> None:
    """Write the full setup to ``path`` as pretty-printed JSON."""
    cfo_block: dict | None = None
    if setup.cfo is not None:
        if not setup.cfo.measured_at:
            setup.cfo.measured_at = datetime.now().astimezone().isoformat(timespec="seconds")
        cfo_block = {
            "a_to_b_cfo_hz": setup.cfo.a_to_b_cfo_hz,
            "b_to_a_cfo_hz": setup.cfo.b_to_a_cfo_hz,
            "measured_at":   setup.cfo.measured_at,
        }
    data = {
        "version": setup.version,
        "nodes": {
            node: {"tx_ip": entry["tx"], "rx_ip": entry["rx"]}
            for node, entry in setup.nodes.items()
        },
        "cfo": cfo_block,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_cfo(cfo: CFOCalibration, path: Path = SETUP_PATH) -> None:
    """Update only the cfo section, leaving node IPs untouched."""
    setup = load(path)
    setup.cfo = cfo
    save(setup, path)


def _print_shell_export(setup: Setup) -> None:
    """Print bash-style ``KEY=VALUE`` lines so shell scripts can ``eval`` us."""
    for node, entry in setup.nodes.items():
        print(f"PLUTO_{node}_TX_IP={entry['tx']}")
        print(f"PLUTO_{node}_RX_IP={entry['rx']}")
    if setup.cfo is not None:
        print(f"CFO_A_TO_B_HZ={setup.cfo.a_to_b_cfo_hz}")
        print(f"CFO_B_TO_A_HZ={setup.cfo.b_to_a_cfo_hz}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect / export pluto/setup.json")
    parser.add_argument("--shell-export", action="store_true",
                        help="Emit bash KEY=VALUE lines for `eval` in shell scripts")
    parser.add_argument("--path", type=Path, default=SETUP_PATH,
                        help=f"Setup file (default: {SETUP_PATH})")
    args = parser.parse_args()

    setup = load_or_die(args.path)
    if args.shell_export:
        _print_shell_export(setup)
    else:
        print(json.dumps({
            "nodes": setup.nodes,
            "cfo": (
                None if setup.cfo is None
                else {"a_to_b_cfo_hz": setup.cfo.a_to_b_cfo_hz,
                      "b_to_a_cfo_hz": setup.cfo.b_to_a_cfo_hz,
                      "measured_at":   setup.cfo.measured_at}
            ),
            "version": setup.version,
        }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
