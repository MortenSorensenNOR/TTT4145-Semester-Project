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
        "A": {"tx_ip": "...", "rx_ip": "...",
              "tx_serial": "...", "rx_serial": "..."},
        "B": {"tx_ip": "...", "rx_ip": "...",
              "tx_serial": "...", "rx_serial": "..."}
      },
      "cfo": {
        "a_to_b_cfo_hz": <int>,
        "b_to_a_cfo_hz": <int>,
        "measured_at": "<ISO-8601>"
      }
    }

``tx_serial``/``rx_serial`` are optional. They hold the 8-char hex serial
burned into each Pluto's firmware (also on the back-of-device sticker)
and are used when opening the radios via the libiio ``usb:`` backend
instead of ``ip:`` — USB bus IDs are reassigned on every replug, so the
serial is the only stable identifier for a given physical board.

Run ``python -m pluto.setup_config --list-usb`` with both Plutos plugged
in to see the discovered serials, then paste them into setup.json.
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


def usb_uri_for_serial(serial: str) -> str:
    """Resolve an 8-char hex Pluto serial to a libiio ``usb:bus.dev.intf`` URI.

    USB bus/device numbers are kernel-assigned and change on every replug,
    so the serial is the only stable handle for a particular physical
    Pluto. This scans whatever libiio sees right now and matches by serial.
    """
    import iio  # lazy: only required when actually using the usb backend
    contexts = iio.scan_contexts()
    matches = [
        uri for uri, desc in contexts.items()
        if uri.startswith("usb:") and serial.lower() in desc.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(
            f"No USB Pluto found with serial {serial!r}. "
            f"libiio scan returned: {dict(contexts)!r}. "
            f"Check the device is plugged in and udev permissions are set."
        )
    raise RuntimeError(
        f"Multiple USB Plutos matched serial {serial!r}: {matches}. "
        f"Serials should be unique per board."
    )


@dataclass
class Setup:
    """Full radio-setup config: TX/RX IPs per node + optional CFO cal."""

    nodes: dict[str, dict[str, str | None]] = field(default_factory=dict)
    cfo: CFOCalibration | None = None
    version: int = 2

    def tx_ip(self, node: str) -> str:
        return self.nodes[node]["tx"]

    def rx_ip(self, node: str) -> str:
        return self.nodes[node]["rx"]

    def tx_serial(self, node: str) -> str | None:
        return self.nodes[node].get("tx_serial")

    def rx_serial(self, node: str) -> str | None:
        return self.nodes[node].get("rx_serial")

    def tx_uri(self, node: str, backend: str = "ip") -> str:
        return self._uri(node, role="tx", backend=backend)

    def rx_uri(self, node: str, backend: str = "ip") -> str:
        return self._uri(node, role="rx", backend=backend)

    def _uri(self, node: str, *, role: str, backend: str) -> str:
        if backend == "ip":
            return f"ip:{self.nodes[node]['tx' if role == 'tx' else 'rx']}"
        if backend == "usb":
            serial = self.nodes[node].get(f"{role}_serial")
            if not serial:
                raise ValueError(
                    f"node {node!r} has no {role}_serial in setup.json. "
                    f"Run 'python -m pluto.setup_config --list-usb' to discover "
                    f"serials, then add them to setup.json."
                )
            return usb_uri_for_serial(serial)
        raise ValueError(f"unknown backend {backend!r} (expected 'ip' or 'usb')")


def load(path: Path = SETUP_PATH) -> Setup:
    """Load the setup from ``path``. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Pluto setup file missing: {path}. "
            f"Create it (see pluto/setup_config.py for the schema)."
        )
    data = json.loads(path.read_text())
    nodes = {
        node: {
            "tx":        entry["tx_ip"],
            "rx":        entry["rx_ip"],
            "tx_serial": entry.get("tx_serial"),
            "rx_serial": entry.get("rx_serial"),
        }
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
    def _node_block(entry: dict) -> dict:
        block = {"tx_ip": entry["tx"], "rx_ip": entry["rx"]}
        if entry.get("tx_serial"):
            block["tx_serial"] = entry["tx_serial"]
        if entry.get("rx_serial"):
            block["rx_serial"] = entry["rx_serial"]
        return block

    data = {
        "version": setup.version,
        "nodes": {node: _node_block(entry) for node, entry in setup.nodes.items()},
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


def _print_usb_scan() -> None:
    """List every libiio context currently visible, highlighting USB Plutos.

    Run this with both Plutos plugged in to discover the 8-char serials,
    then paste them into ``setup.json`` under ``nodes.<X>.tx_serial`` /
    ``rx_serial``.
    """
    import iio
    contexts = iio.scan_contexts()
    if not contexts:
        print("No libiio contexts visible.")
        print("  - is the Pluto plugged in?")
        print("  - is the udev rule installed (53-adi-plutosdr-usb.rules)?")
        return
    print(f"Discovered {len(contexts)} libiio context(s):")
    for uri, desc in contexts.items():
        marker = "  USB " if uri.startswith("usb:") else "      "
        print(f"{marker}{uri}")
        print(f"        {desc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect / export pluto/setup.json")
    parser.add_argument("--shell-export", action="store_true",
                        help="Emit bash KEY=VALUE lines for `eval` in shell scripts")
    parser.add_argument("--list-usb", action="store_true",
                        help="Scan libiio for USB Plutos and print their URIs + "
                             "serials. Run this with both Plutos plugged in to "
                             "discover the 8-char serials to paste into setup.json.")
    parser.add_argument("--path", type=Path, default=SETUP_PATH,
                        help=f"Setup file (default: {SETUP_PATH})")
    args = parser.parse_args()

    if args.list_usb:
        _print_usb_scan()
        return 0

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
