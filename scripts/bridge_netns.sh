#!/usr/bin/env bash
# Persistent bridge netns — bring up the two-node radio link so you can
# drive traffic through it manually (ffmpeg, iperf, ping, whatever).
#
# This is the same netns + bridge stack that bridge_benchmark.sh builds,
# pulled out into an up/down controller. After `up` the bridges keep
# running in the background; run your traffic tool inside either netns:
#
#     sudo ip netns exec arq-a ffmpeg -re -i input.mp4 ... 10.0.0.2
#     sudo ip netns exec arq-b ffplay udp://0.0.0.0:5000
#
# Usage:
#     sudo ./scripts/bridge_netns.sh up                      # bring link up
#     sudo ./scripts/bridge_netns.sh up --shape-rate 2mbit   # custom shaping
#     sudo ./scripts/bridge_netns.sh up --no-shape           # no qdisc
#     sudo ./scripts/bridge_netns.sh status                  # what's running?
#     sudo ./scripts/bridge_netns.sh down                    # tear down
#
# Pluto IPs are loaded from pluto/setup.json (single source of truth, also
# used by the bridge process itself). Generated log files:
#     bridge-A.log   bridge-B.log    (project root)
# Bridge PIDs are stored in /tmp/bridge-{A,B}.pid so `down` can kill them
# even after the shell that ran `up` has exited.
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────
NS_A=arq-a
NS_B=arq-b
IP_A=10.0.0.1
IP_B=10.0.0.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJ_ROOT/.venv/bin/python"

# Pull PLUTO_{A,B}_{TX,RX}_IP from pluto/setup.json.
[[ -x "$PYTHON" ]] || { echo "ERROR: $PYTHON not found — run 'uv sync' first" >&2; exit 1; }
eval "$("$PYTHON" -m pluto.setup_config --shell-export)"
: "${PLUTO_A_TX_IP:?missing PLUTO_A_TX_IP}"
: "${PLUTO_A_RX_IP:?missing PLUTO_A_RX_IP}"
: "${PLUTO_B_TX_IP:?missing PLUTO_B_TX_IP}"
: "${PLUTO_B_RX_IP:?missing PLUTO_B_RX_IP}"

# Host-side endpoints share the Pluto's /24 with .10 in the last octet.
host_ip_for() { sed 's/\.[0-9]\+$/.10/' <<<"$1"; }
HOST_A_TX_IP=$(host_ip_for "$PLUTO_A_TX_IP")
HOST_A_RX_IP=$(host_ip_for "$PLUTO_A_RX_IP")
HOST_B_TX_IP=$(host_ip_for "$PLUTO_B_TX_IP")
HOST_B_RX_IP=$(host_ip_for "$PLUTO_B_RX_IP")

TUN_NAME=pluto0

TX_GAIN=-20
MTU=1500
SHAPE_RATE="1500kbit"   # "off" to skip; override with --shape-rate / --no-shape
STARTUP_WAIT=6

PID_A=/tmp/bridge-A.pid
PID_B=/tmp/bridge-B.pid

LOG_A="$PROJ_ROOT/bridge-A.log"
LOG_B="$PROJ_ROOT/bridge-B.log"

# ── Subcommand + arg parsing ─────────────────────────────────────────────
SUBCMD="${1:-}"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tx-gain)    TX_GAIN=$2; shift 2 ;;
        --mtu)        MTU=$2; shift 2 ;;
        --shape-rate) SHAPE_RATE=$2; shift 2 ;;
        --no-shape)   SHAPE_RATE="off"; shift ;;
        --startup-wait) STARTUP_WAIT=$2; shift 2 ;;
        -h|--help)
            sed -n '1,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$SUBCMD" in
    up|down|status) ;;
    ""|-h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "Unknown subcommand: $SUBCMD (expected up|down|status)" >&2; exit 2 ;;
esac

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (needs netns + ip link set netns)" >&2
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────
detect_iface() {
    # $1 = host IPv4 → iface name, or empty.
    ip -o -4 addr show | awk -v ip="$1" '$4 ~ ("^" ip "/") {print $2; exit}'
}

# Map { netns → { ifname → ipv4 } } for restoration after teardown.
drain_netns() {
    # Drains every non-lo iface from the given netns back into the root netns,
    # restoring whatever IPv4 address each iface held inside the netns.
    local ns=$1
    ip netns list 2>/dev/null | grep -q "^$ns" || return 0

    echo "[info]   draining $ns → root…"
    while IFS= read -r ifname; do
        [[ -z "$ifname" || "$ifname" == "lo" ]] && continue
        local want_ip
        want_ip=$(ip -n "$ns" -o -4 addr show dev "$ifname" 2>/dev/null \
                    | awk '{print $4}' | head -n1 | cut -d/ -f1)
        if ip -n "$ns" link set "$ifname" netns 1 2>/dev/null && [[ -n "$want_ip" ]]; then
            ip addr flush dev "$ifname" 2>/dev/null || true
            ip addr add "${want_ip}/24" dev "$ifname" 2>/dev/null || true
            ip link set "$ifname" up 2>/dev/null || true
            echo "[info]     $ifname ← $want_ip/24 (up)"
        fi
    done < <(ip -n "$ns" -o link show 2>/dev/null \
               | awk -F': ' '{print $2}' | sed 's/@.*//' | sort -u)

    ip netns del "$ns" 2>/dev/null || true
}

kill_bridge() {
    # $1 = pidfile
    local pf=$1
    [[ -f "$pf" ]] || return 0
    local pid
    pid=$(cat "$pf")
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.4
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    rm -f "$pf"
}

# ── Subcommand: down ─────────────────────────────────────────────────────
cmd_down() {
    echo "[info] Tearing down bridge netns…"

    # 1. Kill tracked bridges; best-effort sweep for anything leftover.
    kill_bridge "$PID_A"
    kill_bridge "$PID_B"
    if pgrep -f "pluto\.tun_link" >/dev/null; then
        pkill -INT -f "pluto\.tun_link" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "pluto\.tun_link" 2>/dev/null || true
    fi

    # 2. Drain each netns and restore the USB iface IPs.
    drain_netns "$NS_A"
    drain_netns "$NS_B"

    # 3. Sanity: are the Plutos reachable again?
    local ok=1
    for pl in "$PLUTO_A_TX_IP" "$PLUTO_A_RX_IP" "$PLUTO_B_TX_IP" "$PLUTO_B_RX_IP"; do
        if ping -c1 -W1 "$pl" >/dev/null 2>&1; then
            echo "[info]   $pl reachable"
        else
            echo "[warn]   $pl NOT reachable — unplug/replug the USB Pluto if needed"
            ok=0
        fi
    done

    [[ $ok -eq 1 ]] && echo "[info] Teardown complete." \
                    || echo "[info] Teardown done with warnings."
}

# ── Subcommand: status ───────────────────────────────────────────────────
cmd_status() {
    echo "── bridge netns status ─────────────────────────────"
    for ns in "$NS_A" "$NS_B"; do
        if ip netns list | grep -q "^$ns"; then
            printf "  %-8s present\n" "$ns:"
            ip -n "$ns" -o -4 addr show \
                | awk -v ns="$ns" '$2 != "lo" {printf "      %-10s %s\n", $2, $4}'
        else
            printf "  %-8s absent\n" "$ns:"
        fi
    done
    for entry in "A:$PID_A" "B:$PID_B"; do
        tag="${entry%%:*}"; pf="${entry##*:}"
        if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
            echo "  bridge-$tag: running  (pid $(cat "$pf"))"
        else
            echo "  bridge-$tag: not running"
        fi
    done
    echo "───────────────────────────────────────────────────"
}

# ── Subcommand: up ───────────────────────────────────────────────────────
cmd_up() {
    # Refuse if anything is already up — force the user to `down` first so
    # we don't silently stack two bridge processes on the same Pluto.
    if ip netns list | grep -qE "^($NS_A|$NS_B)\b"; then
        echo "ERROR: netns $NS_A or $NS_B already exists — run 'down' first" >&2
        exit 1
    fi

    [[ -x "$PYTHON" ]] || { echo "ERROR: $PYTHON not found — run 'uv sync' first" >&2; exit 1; }

    # 1. Find each Pluto's USB iface by its host-side IP.
    local IFACE_A_TX IFACE_A_RX IFACE_B_TX IFACE_B_RX
    IFACE_A_TX=$(detect_iface "$HOST_A_TX_IP")
    IFACE_A_RX=$(detect_iface "$HOST_A_RX_IP")
    IFACE_B_TX=$(detect_iface "$HOST_B_TX_IP")
    IFACE_B_RX=$(detect_iface "$HOST_B_RX_IP")
    local missing=""
    [[ -z "$IFACE_A_TX" ]] && missing="$missing $HOST_A_TX_IP"
    [[ -z "$IFACE_A_RX" ]] && missing="$missing $HOST_A_RX_IP"
    [[ -z "$IFACE_B_TX" ]] && missing="$missing $HOST_B_TX_IP"
    [[ -z "$IFACE_B_RX" ]] && missing="$missing $HOST_B_RX_IP"
    if [[ -n "$missing" ]]; then
        echo "ERROR: could not find USB-ethernet interfaces with IPs:$missing" >&2
        echo "       Check 'ip -4 addr' — are all four Plutos plugged in?" >&2
        exit 1
    fi
    echo "[info] Plutos:"
    echo "  A TX: $IFACE_A_TX ($HOST_A_TX_IP ↔ $PLUTO_A_TX_IP)"
    echo "  A RX: $IFACE_A_RX ($HOST_A_RX_IP ↔ $PLUTO_A_RX_IP)"
    echo "  B TX: $IFACE_B_TX ($HOST_B_TX_IP ↔ $PLUTO_B_TX_IP)"
    echo "  B RX: $IFACE_B_RX ($HOST_B_RX_IP ↔ $PLUTO_B_RX_IP)"

    # 2. Build netns and move ifaces in.
    echo "[info] Creating netns $NS_A, $NS_B …"
    ip netns add "$NS_A"
    ip netns add "$NS_B"

    # On any failure past this point, tear down cleanly rather than leaving
    # Plutos stranded inside half-built namespaces.
    trap 'echo "[err] setup aborted — tearing down" >&2; cmd_down; exit 1' ERR

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

    # 3. Sanity-check every Pluto is reachable from its netns.
    for entry in \
        "$NS_A:$PLUTO_A_TX_IP:A-TX" "$NS_A:$PLUTO_A_RX_IP:A-RX" \
        "$NS_B:$PLUTO_B_TX_IP:B-TX" "$NS_B:$PLUTO_B_RX_IP:B-RX"; do
        ns="${entry%%:*}"; rest="${entry#*:}"
        pl="${rest%%:*}";  tag="${rest##*:}"
        if ! ip netns exec "$ns" ping -c1 -W2 "$pl" >/dev/null; then
            echo "ERROR: Pluto $tag ($pl) unreachable from $ns" >&2
            exit 1
        fi
    done
    echo "[info] All four Plutos reachable inside their netns."

    # 4. Launch bridge processes.
    : > "$LOG_A"; : > "$LOG_B"
    cd "$PROJ_ROOT"

    echo "[info] Starting bridge A (log: $LOG_A)…"
    ip netns exec "$NS_A" "$PYTHON" -m pluto.tun_link \
        --node A --tx-ip "$PLUTO_A_TX_IP" --rx-ip "$PLUTO_A_RX_IP" \
        --gain "$TX_GAIN" --mtu "$MTU" --tun-name "$TUN_NAME" \
        >"$LOG_A" 2>&1 &
    echo $! > "$PID_A"

    echo "[info] Starting bridge B (log: $LOG_B)…"
    ip netns exec "$NS_B" "$PYTHON" -m pluto.tun_link \
        --node B --tx-ip "$PLUTO_B_TX_IP" --rx-ip "$PLUTO_B_RX_IP" \
        --gain "$TX_GAIN" --mtu "$MTU" --tun-name "$TUN_NAME" \
        >"$LOG_B" 2>&1 &
    echo $! > "$PID_B"

    echo "[info] Waiting ${STARTUP_WAIT}s for bridges to come up…"
    sleep "$STARTUP_WAIT"

    for pf in "$PID_A" "$PID_B"; do
        pid=$(cat "$pf")
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: bridge pid $pid died during startup — see logs" >&2
            echo "--- $LOG_A (last 30 lines) ---"; tail -n 30 "$LOG_A"
            echo "--- $LOG_B (last 30 lines) ---"; tail -n 30 "$LOG_B"
            exit 1
        fi
    done

    # 5. Optional TCP shaping on each TUN.
    if [[ "$SHAPE_RATE" != "off" ]]; then
        for ns in "$NS_A" "$NS_B"; do
            if ! ip netns exec "$ns" tc qdisc replace dev "$TUN_NAME" root \
                    cake bandwidth "$SHAPE_RATE" 2>/dev/null; then
                echo "[warn] cake qdisc failed on $ns/$TUN_NAME — trying tbf"
                ip netns exec "$ns" tc qdisc replace dev "$TUN_NAME" root \
                    tbf rate "$SHAPE_RATE" burst 16kb latency 50ms
            fi
        done
        echo "[info] TCP shaping: $SHAPE_RATE on $TUN_NAME in both netns"
    else
        echo "[info] TCP shaping disabled (--no-shape)"
    fi

    trap - ERR  # setup succeeded; don't auto-teardown on later shell errors

    cat <<EOF

========================================================================
  Bridge link is up.

  Node A (netns $NS_A):  TUN $TUN_NAME = $IP_A
  Node B (netns $NS_B):  TUN $TUN_NAME = $IP_B

  Drive traffic through the link with:
    sudo ip netns exec $NS_A <cmd>    # A-side
    sudo ip netns exec $NS_B <cmd>    # B-side

  Examples:
    sudo ip netns exec $NS_A ping $IP_B
    sudo ip netns exec $NS_B iperf3 -s -1 &
    sudo ip netns exec $NS_A iperf3 -c $IP_B -t 20

    # Video — sender in A, receiver in B
    sudo ip netns exec $NS_A ffmpeg -re -i input.mp4 -c:v copy \\
        -f mpegts udp://$IP_B:5000
    sudo ip netns exec $NS_B ffplay udp://0.0.0.0:5000

  Bridge logs:  $LOG_A  $LOG_B
  Tear down:    sudo $0 down
========================================================================
EOF
}

case "$SUBCMD" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    status) cmd_status ;;
esac
