"""Sweep target UDP bitrate × datagram size through the one-way radio link.

Runs iperf2 client in arq-a → server in arq-b for every (rate, size) cell on
the sweep grid, parses the per-connection summary the server prints, and emits
a CSV plus an optional loss% heatmap. Useful for characterising where the link
falls over (low-bitrate AGC/DC drift, MTU-large packet fragmentation, raw
capacity ceiling, etc.) and for picking sane defaults for downstream apps
(ffmpeg bitrate, control-channel packet size, …).

Pre-req: link must already be up via scripts/oneway_netns.sh up. The script
does NOT touch the netns or tun_link processes — it just runs iperf inside
the existing namespaces.

Usage (must be root, as ip-netns-exec requires CAP_SYS_ADMIN):
    sudo .venv/bin/python scripts/sweep_throughput.py
    sudo .venv/bin/python scripts/sweep_throughput.py --duration 5 \\
        --rates 100k,500k,1M,2M,3M --sizes 256,512,1024,1316
    sudo .venv/bin/python scripts/sweep_throughput.py --plot sweep.png

The default grid is 8 × 6 = 48 cells × 8 s ≈ 7 minutes (plus settle gaps).
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────
NS_A    = "arq-a"
NS_B    = "arq-b"
SRV_IP  = "10.0.0.2"
PORT    = 5001

# Defaults cover roughly the link's working range at SAMPLE_RATE=6_000_000,
# SPS=4, PSK8, LDPC 5/6 — useful capacity ~3.75 Mbps continuous (less after
# per-packet preamble/header/LDPC-pad overhead, hence 4M stresses the ceiling).
DEFAULT_RATES = ["50k", "100k", "200k", "500k", "1M", "2M", "3M", "4M"]
DEFAULT_SIZES = [128, 256, 512, 1024, 1316, 1400]


# ── iperf2 server-summary parser ──────────────────────────────────────────
# Matches the final per-connection line iperf2 prints, e.g.:
#   [  1] 0.0000-10.0763 sec  5.00 MBytes  4.20 Mbits/sec   6.351 ms 363/3747 (9.7%)
SUMMARY_RE = re.compile(
    r"\[\s*\d+\]\s+"
    r"(?P<t0>[\d.]+)-(?P<t1>[\d.]+)\s+sec\s+"
    r"(?P<bytes>[\d.]+)\s+(?P<bunit>[KMG]?Bytes)\s+"
    r"(?P<rate>[\d.]+)\s+(?P<runit>[KMG]?bits/sec)\s+"
    r"(?P<jitter>[\d.]+)\s+ms\s+"
    r"(?P<lost>\d+)/(?P<total>\d+)\s+"
    r"\((?P<lossp>[\d.]+)%\)"
)

_BUNIT = {"Bytes": 1, "KBytes": 1e3, "MBytes": 1e6, "GBytes": 1e9}
_RUNIT = {"bits/sec": 1, "Kbits/sec": 1e3, "Mbits/sec": 1e6, "Gbits/sec": 1e9}


@dataclass
class CellResult:
    rate_arg: str           # e.g. "500k", as passed to iperf
    size: int               # bytes per datagram
    duration_s: float
    offered_bps: float      # what iperf claims it sent (best-effort, from client)
    delivered_bps: float    # what server actually saw
    loss_pct: float
    jitter_ms: float
    lost: int
    total: int


# ── iperf invocation helpers ──────────────────────────────────────────────
def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def parse_rate_arg(arg: str) -> float:
    """'500k' → 500_000, '1M' → 1_000_000. Same units iperf accepts."""
    m = re.fullmatch(r"([\d.]+)([kmg]?)", arg.strip().lower())
    if not m:
        raise ValueError(f"unrecognised rate {arg!r} — use e.g. '500k', '1M', '3.5M'")
    num = float(m.group(1))
    suffix = m.group(2)
    return num * {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}[suffix]


def start_server(server_log: Path) -> subprocess.Popen:
    """Persistent iperf2 UDP server in arq-b. Each client connection adds a
    summary line to its stdout; we tail the file later to align per-cell."""
    server_log.write_text("")
    fh = server_log.open("w", buffering=1)
    p = subprocess.Popen(
        ["ip", "netns", "exec", NS_B, "iperf", "-s", "-u", "-i", "1"],
        stdout=fh, stderr=subprocess.STDOUT,
    )
    # Give it a beat to bind the port. If it died immediately, surface that now.
    time.sleep(1.0)
    if p.poll() is not None:
        log = server_log.read_text()
        raise RuntimeError(
            f"iperf server died on startup (exit={p.returncode}):\n{log}"
        )
    return p


def stop_server(p: subprocess.Popen) -> None:
    p.terminate()
    try:
        p.wait(timeout=3)
    except subprocess.TimeoutExpired:
        p.kill()


def run_client(rate_arg: str, size: int, duration: float) -> str:
    """iperf2 client in arq-a. Returns the client's stdout (final summary
    has the 'Sent N datagrams' line; loss/jitter only appears on the server)."""
    cmd = [
        "ip", "netns", "exec", NS_A,
        "iperf", "-c", SRV_IP, "-u",
        "-b", rate_arg, "-l", str(size),
        "-t", str(duration), "-i", "0",  # no per-interval client output
    ]
    cp = run(cmd)
    if cp.returncode != 0:
        sys.stderr.write(f"[warn] client returned {cp.returncode}\n")
        sys.stderr.write(cp.stderr or cp.stdout)
    return cp.stdout


# ── Sweep core ────────────────────────────────────────────────────────────
def precheck() -> None:
    """Bail loudly if the netns aren't up — the user forgot to run oneway_netns.sh."""
    cp = run(["ip", "netns", "list"])
    if NS_A not in cp.stdout or NS_B not in cp.stdout:
        sys.exit(f"ERROR: {NS_A} / {NS_B} not present. Run "
                 f"'sudo ./scripts/oneway_netns.sh up' first.")


def _to_bps(val: float, unit: str) -> float:
    return val * _RUNIT[unit]


def _to_bytes(val: float, unit: str) -> float:
    return val * _BUNIT[unit]


def parse_server_summaries(server_log: Path) -> list[dict]:
    """One dict per client connection seen by the server."""
    out = []
    for line in server_log.read_text().splitlines():
        m = SUMMARY_RE.search(line)
        if not m:
            continue
        out.append({
            "t0":      float(m.group("t0")),
            "t1":      float(m.group("t1")),
            "bytes":   _to_bytes(float(m.group("bytes")), m.group("bunit")),
            "bps":     _to_bps(float(m.group("rate")),    m.group("runit")),
            "jitter":  float(m.group("jitter")),
            "lost":    int(m.group("lost")),
            "total":   int(m.group("total")),
            "loss_pct": float(m.group("lossp")),
        })
    return out


def sweep(rates: list[str], sizes: list[int], duration: float,
          server_log: Path) -> list[CellResult]:
    """Run iperf for every (rate, size) cell in a deterministic order
    (rates outer, sizes inner). Server is left running across all cells."""
    proc = start_server(server_log)
    results: list[CellResult] = []
    summaries_seen = 0

    try:
        total = len(rates) * len(sizes)
        cell  = 0
        for rate_arg in rates:
            for size in sizes:
                cell += 1
                print(f"[{cell:>3d}/{total}] rate={rate_arg:>5s} size={size:>4d}B "
                      f"duration={duration:.0f}s …", end="", flush=True)

                t0 = time.time()
                run_client(rate_arg, size, duration)
                # Server prints its summary AFTER the client closes; small grace.
                time.sleep(0.5)
                summaries = parse_server_summaries(server_log)
                if len(summaries) <= summaries_seen:
                    print(" no server-side summary (lost completely?)")
                    results.append(CellResult(
                        rate_arg=rate_arg, size=size, duration_s=duration,
                        offered_bps=parse_rate_arg(rate_arg),
                        delivered_bps=0.0, loss_pct=100.0,
                        jitter_ms=float("nan"), lost=0, total=0,
                    ))
                else:
                    s = summaries[-1]
                    results.append(CellResult(
                        rate_arg=rate_arg, size=size, duration_s=duration,
                        offered_bps=parse_rate_arg(rate_arg),
                        delivered_bps=s["bps"], loss_pct=s["loss_pct"],
                        jitter_ms=s["jitter"], lost=s["lost"], total=s["total"],
                    ))
                    summaries_seen = len(summaries)
                    elapsed = time.time() - t0
                    print(f" got {s['bps']/1e3:>6.0f} kbps  loss={s['loss_pct']:5.1f}%  "
                          f"({elapsed:.1f}s)")
                # Settle: let the link drain + AGC re-base before the next cell.
                time.sleep(1.0)
    finally:
        stop_server(proc)

    return results


# ── Output ────────────────────────────────────────────────────────────────
def write_csv(results: list[CellResult], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rate_arg", "size_bytes", "duration_s",
                    "offered_bps", "delivered_bps", "loss_pct",
                    "jitter_ms", "lost", "total"])
        for r in results:
            w.writerow([r.rate_arg, r.size, r.duration_s,
                        r.offered_bps, r.delivered_bps, r.loss_pct,
                        r.jitter_ms, r.lost, r.total])
    print(f"[ok] wrote {path} ({len(results)} cells)")


def plot_heatmap(results: list[CellResult], rates: list[str], sizes: list[int],
                 path: Path) -> None:
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[warn] matplotlib not installed — skipping {path}")
        return

    grid = np.full((len(rates), len(sizes)), float("nan"))
    by_key = {(r.rate_arg, r.size): r for r in results}
    for i, ra in enumerate(rates):
        for j, sz in enumerate(sizes):
            r = by_key.get((ra, sz))
            if r is not None:
                grid[i, j] = r.loss_pct

    fig, ax = plt.subplots(figsize=(1.2 * len(sizes) + 2, 0.7 * len(rates) + 2))
    im = ax.imshow(grid, cmap="magma_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f"{s}B" for s in sizes])
    ax.set_yticks(range(len(rates))); ax.set_yticklabels(rates)
    ax.set_xlabel("UDP datagram size")
    ax.set_ylabel("Target bitrate")
    ax.set_title("Loss % across (rate × size) — lower is better")
    for i in range(len(rates)):
        for j in range(len(sizes)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v > 40 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="loss %")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[ok] wrote {path}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rates", default=",".join(DEFAULT_RATES),
                    help=f"comma-separated iperf -b values "
                         f"(default: {','.join(DEFAULT_RATES)})")
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                    help=f"comma-separated iperf -l values in bytes "
                         f"(default: {','.join(str(s) for s in DEFAULT_SIZES)})")
    ap.add_argument("--duration", type=float, default=8.0,
                    help="seconds per cell (default: 8)")
    ap.add_argument("--out", type=Path, default=Path("sweep.csv"),
                    help="CSV output path (default: sweep.csv)")
    ap.add_argument("--plot", type=Path, default=None,
                    help="optional heatmap PNG (requires matplotlib)")
    ap.add_argument("--server-log", type=Path, default=Path("/tmp/iperf_sweep_server.log"),
                    help="where the server's stdout goes (default: /tmp/iperf_sweep_server.log)")
    args = ap.parse_args()

    rates = [r.strip() for r in args.rates.split(",") if r.strip()]
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"sweep: {len(rates)} rates × {len(sizes)} sizes = {len(rates)*len(sizes)} cells, "
          f"{args.duration:.0f}s each (~{len(rates)*len(sizes)*(args.duration+1.5)/60:.1f} min)")

    precheck()
    results = sweep(rates, sizes, args.duration, args.server_log)
    write_csv(results, args.out)
    if args.plot is not None:
        plot_heatmap(results, rates, sizes, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
