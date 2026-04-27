#!/usr/bin/env bash
# Bridge benchmark — two netns, *two Plutos in each* (one TX, one RX),
# measure latency / throughput / post-ARQ reliability over the radio link.
#
# Split-radio layout: single USB-2 Pluto can't sustain 4 Msps full-duplex,
# so each node runs a dedicated TX radio and a dedicated RX radio. Host
# IPs follow the 192.168.N.1 convention (even N → node A, odd N → node B).
#
# Needs root (netns + moving interfaces into netns).  Pluto IPs come from
# pluto/setup.json (loaded via `python -m pluto.setup_config --shell-export`);
# the host-side endpoint on each Pluto USB net is the same /24 with .10 in
# the last octet. Assumes the project venv lives at .venv/bin/python.
#
# Usage:
#   sudo ./scripts/bridge_benchmark.sh                        # ping only (default)
#   sudo ./scripts/bridge_benchmark.sh --iperf                # also run iperf
#   sudo ./scripts/bridge_benchmark.sh --ping-count 30        # tune
#   sudo ./scripts/bridge_benchmark.sh --tx-gain -20          # stronger TX
#   sudo ./scripts/bridge_benchmark.sh --shape-rate 2mbit     # TCP shaping cap
#   sudo ./scripts/bridge_benchmark.sh --no-shape             # disable shaping
#   sudo ./scripts/bridge_benchmark.sh --cleanup              # tear down only
#
# TCP shaping: a `cake` qdisc is installed on each TUN after the bridge comes
# up, capping TCP at --shape-rate (default 1500kbit). Without this, TCP slow-
# start inflates cwnd far past the air-link BDP, queues overflow in bursts,
# and iperf samples oscillate between ~5 Mbps spikes and multi-second 0 bps
# stalls. Matching the cap to actual air throughput gives TCP a real back-
# pressure signal and smooths the rate.
#
# CFO compensation: per-direction LO offsets live in the cfo block of
# pluto/setup.json — generate them once with
#   uv run python scripts/cfo_calibrate.py
# and the bridge applies the correction to each node's RX LO on startup.
#
# After completion the script tears down both netns and moves the Pluto
# USB interfaces back to the root namespace.
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
PING_COUNT=30
PING_INTERVAL=0.5       # -i 0.5 to keep from starving ARQ
IPERF_TIME=30
TX_GAIN=-20             # dB, passed to both nodes
MTU=1500
WINDOW=15               # ARQ in-flight frames; must be < SEQ_SPACE/2 = 16
SHAPE_RATE="1500kbit"   # TUN egress cap (cake qdisc); "off" to skip shaping
TUN_NAME="pluto0"       # TUN iface name used by pluto.bridge (its --tun default)
SKIP_IPERF=0            # iperf off by default — link must ping cleanly first
SKIP_PING=0
STARTUP_WAIT=6          # seconds to let bridges come up before traffic
CLEANUP_ONLY=0          # --cleanup: tear down state from a prior aborted run

NS_A=arq-a
NS_B=arq-b
IP_A=10.0.0.0
IP_B=10.0.0.1

# Per-node TX/RX Pluto IPs are loaded from pluto/setup.json below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJ_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "ERROR: $PYTHON not found — run 'uv sync' first" >&2; exit 1; }
eval "$("$PYTHON" -m pluto.setup_config --shell-export)"
: "${PLUTO_A_TX_IP:?missing PLUTO_A_TX_IP}"
: "${PLUTO_A_RX_IP:?missing PLUTO_A_RX_IP}"
: "${PLUTO_B_TX_IP:?missing PLUTO_B_TX_IP}"
: "${PLUTO_B_RX_IP:?missing PLUTO_B_RX_IP}"

# Host-side endpoints share each Pluto's /24, with .10 in the last octet.
host_ip_for() { sed 's/\.[0-9]\+$/.10/' <<<"$1"; }
HOST_A_TX_IP=$(host_ip_for "$PLUTO_A_TX_IP")
HOST_A_RX_IP=$(host_ip_for "$PLUTO_A_RX_IP")
HOST_B_TX_IP=$(host_ip_for "$PLUTO_B_TX_IP")
HOST_B_RX_IP=$(host_ip_for "$PLUTO_B_RX_IP")

# ── Argparse-lite ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ping-count)       PING_COUNT=$2; shift 2 ;;
        --ping-interval)    PING_INTERVAL=$2; shift 2 ;;
        --iperf-time)       IPERF_TIME=$2; shift 2 ;;
        --tx-gain)          TX_GAIN=$2; shift 2 ;;
        --mtu)              MTU=$2; shift 2 ;;
        --window)           WINDOW=$2; shift 2 ;;
        --shape-rate)       SHAPE_RATE=$2; shift 2 ;;
        --no-shape)         SHAPE_RATE="off"; shift ;;
        --skip-iperf)       SKIP_IPERF=1; shift ;;
        --iperf)            SKIP_IPERF=0; shift ;;
        --skip-ping)        SKIP_PING=1; shift ;;
        --startup-wait)     STARTUP_WAIT=$2; shift 2 ;;
        --cleanup)          CLEANUP_ONLY=1; shift ;;
        -h|--help)
            sed -n '1,25p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (needs netns + ip link set netns)" >&2
    exit 1
fi

# ── Standalone cleanup (--cleanup) ───────────────────────────────────────
# Safe to re-run: kills any stray bridge processes, drains each netns back
# to root (preserving whatever interface names were inside), restores the
# Pluto-USB IPs by name, then deletes both netns. No auto-detection up
# front, because when the interfaces are still inside the netns they don't
# show up in `ip addr` at the root.
cleanup_only() {
    echo "[info] Cleanup-only mode"

    # 1. Kill any python -m pluto.bridge processes still alive
    if pgrep -f "pluto\.bridge" >/dev/null; then
        echo "[info]   killing stray bridge processes…"
        pkill -INT -f "pluto\.bridge" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "pluto\.bridge" 2>/dev/null || true
    fi

    # 2. For each netns we own, move every non-lo iface back to root,
    #    remembering the IPv4 address it held *inside* the netns so we can
    #    re-add it afterwards (moving an iface across netns drops its IPs).
    #    Each netns now owns two ifaces (TX radio + RX radio).
    declare -A restore_ip
    for ns in "$NS_A" "$NS_B"; do
        ip netns list 2>/dev/null | grep -q "^$ns" || continue

        echo "[info]   draining $ns → root…"
        while IFS= read -r ifname; do
            [[ -z "$ifname" || "$ifname" == "lo" ]] && continue
            want_ip=$(ip -n "$ns" -o -4 addr show dev "$ifname" 2>/dev/null \
                        | awk '{print $4}' | head -n1 | cut -d/ -f1)
            if ip -n "$ns" link set "$ifname" netns 1 2>/dev/null; then
                [[ -n "$want_ip" ]] && restore_ip["$ifname"]="$want_ip"
            fi
        done < <(ip -n "$ns" -o link show 2>/dev/null \
                   | awk -F': ' '{print $2}' | sed 's/@.*//' | sort -u)

        ip netns del "$ns" 2>/dev/null || true
    done

    # 3. Re-add each iface's original host IP.
    for ifname in "${!restore_ip[@]}"; do
        ipaddr="${restore_ip[$ifname]}"
        ip addr flush dev "$ifname" 2>/dev/null || true
        ip addr add "${ipaddr}/24" dev "$ifname" 2>/dev/null || true
        ip link set "$ifname" up 2>/dev/null || true
        echo "[info]   $ifname ← $ipaddr/24 (up)"
    done

    # 4. Verify Pluto reachability — if any of these fails the USB link is
    #    wedged; unplug / replug is usually the fix.
    ok=1
    for pl in "$PLUTO_A_TX_IP" "$PLUTO_A_RX_IP" "$PLUTO_B_TX_IP" "$PLUTO_B_RX_IP"; do
        if ping -c1 -W1 "$pl" >/dev/null 2>&1; then
            echo "[info]   $pl reachable"
        else
            echo "[warn]   $pl NOT reachable — unplug/replug the USB Pluto if needed"
            ok=0
        fi
    done

    [[ $ok -eq 1 ]] && echo "[info] Cleanup complete." \
                    || echo "[info] Cleanup done with warnings."
}

if [[ $CLEANUP_ONLY -eq 1 ]]; then
    cleanup_only
    exit 0
fi

# ── Auto-detect USB-ethernet interfaces ──────────────────────────────────
detect_iface() {
    # $1 = host IPv4; print the iface name that owns it, or fail
    local want=$1
    ip -o -4 addr show | awk -v ip="$want" '$4 ~ ("^" ip "/") {print $2; exit}'
}

IFACE_A_TX=$(detect_iface "$HOST_A_TX_IP") || true
IFACE_A_RX=$(detect_iface "$HOST_A_RX_IP") || true
IFACE_B_TX=$(detect_iface "$HOST_B_TX_IP") || true
IFACE_B_RX=$(detect_iface "$HOST_B_RX_IP") || true

missing=""
[[ -z "${IFACE_A_TX:-}" ]] && missing="$missing $HOST_A_TX_IP"
[[ -z "${IFACE_A_RX:-}" ]] && missing="$missing $HOST_A_RX_IP"
[[ -z "${IFACE_B_TX:-}" ]] && missing="$missing $HOST_B_TX_IP"
[[ -z "${IFACE_B_RX:-}" ]] && missing="$missing $HOST_B_RX_IP"
if [[ -n "$missing" ]]; then
    echo "ERROR: could not find USB-ethernet interfaces with IPs:$missing" >&2
    echo "       Check 'ip -4 addr' — are all four Plutos plugged in?" >&2
    exit 1
fi

echo "[info] Node A  TX iface: $IFACE_A_TX  ($HOST_A_TX_IP ↔ $PLUTO_A_TX_IP)"
echo "[info] Node A  RX iface: $IFACE_A_RX  ($HOST_A_RX_IP ↔ $PLUTO_A_RX_IP)"
echo "[info] Node B  TX iface: $IFACE_B_TX  ($HOST_B_TX_IP ↔ $PLUTO_B_TX_IP)"
echo "[info] Node B  RX iface: $IFACE_B_RX  ($HOST_B_RX_IP ↔ $PLUTO_B_RX_IP)"

# ── Cleanup trap ─────────────────────────────────────────────────────────
cleanup() {
    set +e
    echo
    echo "[info] Cleaning up…"

    # Kill bridge processes (inside netns) by signalling the netns PIDs we stored
    for pid in "${BRIDGE_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -INT "$pid" 2>/dev/null
    done
    # Give the bridges a moment to flush stats, then force kill
    sleep 2
    for pid in "${BRIDGE_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null
    done

    # Move USB ifaces back to root netns if still inside ours (two per node).
    if ip netns list | grep -q "^$NS_A"; then
        ip -n "$NS_A" link set "$IFACE_A_TX" netns 1 2>/dev/null || true
        ip -n "$NS_A" link set "$IFACE_A_RX" netns 1 2>/dev/null || true
        ip netns del "$NS_A" 2>/dev/null || true
    fi
    if ip netns list | grep -q "^$NS_B"; then
        ip -n "$NS_B" link set "$IFACE_B_TX" netns 1 2>/dev/null || true
        ip -n "$NS_B" link set "$IFACE_B_RX" netns 1 2>/dev/null || true
        ip netns del "$NS_B" 2>/dev/null || true
    fi

    # Restore IPs on the host interfaces (moving back doesn't re-add them)
    ip addr replace "$HOST_A_TX_IP/24" dev "$IFACE_A_TX" 2>/dev/null || true
    ip addr replace "$HOST_A_RX_IP/24" dev "$IFACE_A_RX" 2>/dev/null || true
    ip addr replace "$HOST_B_TX_IP/24" dev "$IFACE_B_TX" 2>/dev/null || true
    ip addr replace "$HOST_B_RX_IP/24" dev "$IFACE_B_RX" 2>/dev/null || true
    for ifn in "$IFACE_A_TX" "$IFACE_A_RX" "$IFACE_B_TX" "$IFACE_B_RX"; do
        ip link set "$ifn" up 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

# ── Build netns and move Pluto ifaces into them ──────────────────────────
echo "[info] Creating netns $NS_A, $NS_B …"
ip netns add "$NS_A"
ip netns add "$NS_B"

ip link set "$IFACE_A_TX" netns "$NS_A"
ip link set "$IFACE_A_RX" netns "$NS_A"
ip link set "$IFACE_B_TX" netns "$NS_B"
ip link set "$IFACE_B_RX" netns "$NS_B"

ip -n "$NS_A" link set lo up
ip -n "$NS_B" link set lo up

ip -n "$NS_A" link set "$IFACE_A_TX" up
ip -n "$NS_A" link set "$IFACE_A_RX" up
ip -n "$NS_B" link set "$IFACE_B_TX" up
ip -n "$NS_B" link set "$IFACE_B_RX" up
ip -n "$NS_A" addr add "$HOST_A_TX_IP/24" dev "$IFACE_A_TX"
ip -n "$NS_A" addr add "$HOST_A_RX_IP/24" dev "$IFACE_A_RX"
ip -n "$NS_B" addr add "$HOST_B_TX_IP/24" dev "$IFACE_B_TX"
ip -n "$NS_B" addr add "$HOST_B_RX_IP/24" dev "$IFACE_B_RX"

# Sanity-check Pluto reachability from inside each netns (all 4 radios).
for entry in \
    "$NS_A:$PLUTO_A_TX_IP:A-TX" "$NS_A:$PLUTO_A_RX_IP:A-RX" \
    "$NS_B:$PLUTO_B_TX_IP:B-TX" "$NS_B:$PLUTO_B_RX_IP:B-RX"; do
    ns="${entry%%:*}"; rest="${entry#*:}"
    pluto="${rest%%:*}"; tag="${rest##*:}"
    if ! ip netns exec "$ns" ping -c1 -W2 "$pluto" >/dev/null; then
        echo "ERROR: Pluto $tag ($pluto) unreachable from $ns" >&2
        exit 1
    fi
done
echo "[info] All four Plutos reachable inside their netns."

# ── Launch bridge processes ──────────────────────────────────────────────
LOG_A="$PROJ_ROOT/bridge-A.log"
LOG_B="$PROJ_ROOT/bridge-B.log"
: > "$LOG_A"
: > "$LOG_B"

BRIDGE_PIDS=()

echo "[info] Starting bridge A (window=$WINDOW, log: $LOG_A)…"
cd "$PROJ_ROOT"
ip netns exec "$NS_A" "$PYTHON" -m pluto.bridge \
    --node A --tx-ip "$PLUTO_A_TX_IP" --rx-ip "$PLUTO_A_RX_IP" \
    --tx-gain "$TX_GAIN" --mtu "$MTU" --window "$WINDOW" \
    >"$LOG_A" 2>&1 &
BRIDGE_PIDS+=($!)

echo "[info] Starting bridge B (window=$WINDOW, log: $LOG_B)…"
ip netns exec "$NS_B" "$PYTHON" -m pluto.bridge \
    --node B --tx-ip "$PLUTO_B_TX_IP" --rx-ip "$PLUTO_B_RX_IP" \
    --tx-gain "$TX_GAIN" --mtu "$MTU" --window "$WINDOW" \
    >"$LOG_B" 2>&1 &
BRIDGE_PIDS+=($!)

echo "[info] Waiting ${STARTUP_WAIT}s for bridges to come up…"
sleep "$STARTUP_WAIT"

# Verify each bridge process is still alive
for pid in "${BRIDGE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "ERROR: bridge process $pid died during startup — see logs" >&2
        echo "--- bridge-A.log (last 30 lines) ---"; tail -n 30 "$LOG_A"
        echo "--- bridge-B.log (last 30 lines) ---"; tail -n 30 "$LOG_B"
        exit 1
    fi
done

# ── TCP shaping on each TUN (bufferbloat mitigation) ─────────────────────
# Install a `cake` qdisc inside each netns so TCP's cwnd can't grow past the
# air-link BDP. Without this, cwnd balloons to hundreds of KB on a ~12 KB
# BDP link, the kernel + ARQ queues saturate, and iperf samples alternate
# between ~5 Mbps spikes and multi-second stalls.
if [[ "$SHAPE_RATE" != "off" ]]; then
    for ns in "$NS_A" "$NS_B"; do
        if ! ip netns exec "$ns" tc qdisc replace dev "$TUN_NAME" root cake bandwidth "$SHAPE_RATE" 2>/dev/null; then
            echo "[warn] cake qdisc install failed on $ns/$TUN_NAME — falling back to tbf"
            ip netns exec "$ns" tc qdisc replace dev "$TUN_NAME" root tbf \
                rate "$SHAPE_RATE" burst 16kb latency 50ms \
                || { echo "ERROR: tbf fallback also failed on $ns/$TUN_NAME" >&2; exit 1; }
        fi
    done
    echo "[info] TCP shaping: $SHAPE_RATE on $TUN_NAME in both netns"
else
    echo "[info] TCP shaping disabled (--no-shape)"
fi

# ── Tests ────────────────────────────────────────────────────────────────
echo
echo "========================================================================"
echo "  BRIDGE BENCHMARK  (A=$IP_A  ↔  B=$IP_B)"
echo "========================================================================"

RESULT_PING=""
RESULT_THROUGHPUT=""
RESULT_RELIABILITY=""

if [[ $SKIP_PING -eq 0 ]]; then
    echo
    echo "── Latency + reliability (ping A→B, $PING_COUNT pkts @ ${PING_INTERVAL}s) ──"
    # tee keeps output visible while we parse the summary below
    PING_OUT=$(ip netns exec "$NS_A" ping -c "$PING_COUNT" -i "$PING_INTERVAL" \
                 -W 2 "$IP_B" | tee /dev/stderr) || true

    # Parse loss % and rtt avg from the final two ping lines.
    RESULT_RELIABILITY=$(printf '%s\n' "$PING_OUT" | awk '/packet loss/ {for(i=1;i<=NF;i++) if ($i ~ /%/) {print $i; exit}}')
    RESULT_PING=$(printf '%s\n' "$PING_OUT" | awk -F'/' '/min\/avg\/max/ {print $5 " ms (avg)"}')
fi

if [[ $SKIP_IPERF -eq 0 ]]; then
    echo
    echo "── Throughput (iperf3, ${IPERF_TIME}s TCP, A→B) ──"

    # Server on B — -1 makes it exit after one test completes.
    ip netns exec "$NS_B" iperf3 -s -1 >/dev/null 2>&1 &
    IPERF_PID=$!
    sleep 1

    IPERF_OUT=$(ip netns exec "$NS_A" iperf3 -c "$IP_B" -t "$IPERF_TIME" -i 2 2>&1 | tee /dev/stderr) || true
    wait "$IPERF_PID" 2>/dev/null || true

    # Extract the receiver throughput (last line with "receiver")
    RESULT_THROUGHPUT=$(printf '%s\n' "$IPERF_OUT" | awk '/receiver/ && /bits\/sec/ {for(i=1;i<=NF;i++) if ($(i+1) ~ /bits\/sec/) {print $i " " $(i+1); exit}}')
fi

# ── Summary ──────────────────────────────────────────────────────────────
echo
echo "========================================================================"
echo "  SUMMARY"
echo "========================================================================"
[[ -n "$RESULT_PING" ]]         && echo "  Ping RTT (avg) : $RESULT_PING"
[[ -n "$RESULT_RELIABILITY" ]]  && echo "  Post-ARQ loss  : $RESULT_RELIABILITY"
[[ -n "$RESULT_THROUGHPUT" ]]   && echo "  Throughput     : $RESULT_THROUGHPUT"
echo "  Bridge logs    : $LOG_A  $LOG_B"
echo "========================================================================"
