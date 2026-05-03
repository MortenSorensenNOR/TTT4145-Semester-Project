#!/usr/bin/env bash
# One-way netns bring-up — A→B only, for the 2-radio setup where you have
# A's TX Pluto and B's RX Pluto plugged in but no return-path radios.
#
# Builds two netns (arq-a / arq-b), each with a TUN at 10.0.0.x:
#
#     arq-a:  pluto0 = 10.0.0.1   (running tun_link --mode tx)
#     arq-b:  pluto0 = 10.0.0.2   (running tun_link --mode rx)
#
# After `up` you can drive UDP from A to B:
#
#     # in arq-b: start a UDP iperf3 server
#     sudo ip netns exec arq-b iperf3 -s -1
#     # in arq-a: send 100 kbps UDP for 30s
#     sudo ip netns exec arq-a iperf3 -c 10.0.0.2 -u -b 100k -t 30
#
#     # or simpler with netcat
#     sudo ip netns exec arq-b nc -u -l 5000
#     echo "hello" | sudo ip netns exec arq-a nc -u 10.0.0.2 5000
#
# TCP won't work in this setup — the return-path ACKs have no radio to come
# back on. UDP, ffmpeg/ffplay, etc. are fine.
#
# Usage:
#     sudo ./scripts/oneway_netns.sh up
#     sudo ./scripts/oneway_netns.sh up --only A     # node A only (10.0.0.1, TX)
#     sudo ./scripts/oneway_netns.sh up --only B     # node B only (10.0.0.2, RX)
#     sudo ./scripts/oneway_netns.sh up --tx-gain -15
#     sudo ./scripts/oneway_netns.sh up --rx-gain-mode slow_attack    # for dense traffic only
#     sudo ./scripts/oneway_netns.sh up --rx-gain 55                  # tweak manual gain
#     sudo ./scripts/oneway_netns.sh up --tx-filler-amp 4096          # noise filler for sparse traffic
#     sudo ./scripts/oneway_netns.sh status
#     sudo ./scripts/oneway_netns.sh down
#
# Defaults to manual RX gain (50 dB). slow_attack AGC drifts during the
# silence between packet bursts, ramping gain up so the next packet clips
# the ADC and the constellation widens 3–5×. Override only when running
# continuous / dense traffic where AGC stays converged.
#
# --tx-filler-amp emits low-amplitude complex Gaussian noise between packets
# to keep the receiver's Costas (carrier phase) and NDA-TED (symbol timing)
# loops, plus AD9361 hardware DC tracking, engaged during the silence.
# Manual RX gain alone fixes AGC drift; filler is the next layer for
# bursty / low-bitrate workloads (e.g. ffmpeg at <1 Mbps) where you'd
# otherwise see worse decode quality at lower bitrates than at higher ones.
set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────
NS_A=arq-a
NS_B=arq-b
IP_A=10.0.0.1
IP_B=10.0.0.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJ_ROOT/.venv/bin/python"

[[ -x "$PYTHON" ]] || { echo "ERROR: $PYTHON not found — run 'uv sync' first" >&2; exit 1; }
eval "$("$PYTHON" -m pluto.setup_config --shell-export)"
: "${PLUTO_A_TX_IP:?missing PLUTO_A_TX_IP}"
: "${PLUTO_B_RX_IP:?missing PLUTO_B_RX_IP}"

host_ip_for() { sed 's/\.[0-9]\+$/.10/' <<<"$1"; }
HOST_A_TX_IP=$(host_ip_for "$PLUTO_A_TX_IP")
HOST_B_RX_IP=$(host_ip_for "$PLUTO_B_RX_IP")

TUN_NAME=pluto0

TX_GAIN=-20
RX_GAIN_MODE="manual"   # sparse-traffic default — slow_attack drifts during inter-packet silence
RX_GAIN=50              # dB, used when RX_GAIN_MODE=manual
TX_FILLER_AMP=0         # 0 = silent zeros between packets (legacy). Bump to ~2048-4096
                        # to keep RX Costas/NDA-TED/DC-tracking loops engaged when traffic
                        # is bursty (e.g. low-bitrate ffmpeg). DAC_SCALE/4 ≈ 4096.
MTU=1500
STARTUP_WAIT=6
ONLY=""                 # "" = both nodes, "A" = TX-only side, "B" = RX-only side

PID_A=/tmp/oneway-A.pid
PID_B=/tmp/oneway-B.pid

LOG_A="$PROJ_ROOT/oneway-A.log"
LOG_B="$PROJ_ROOT/oneway-B.log"

# ── Subcommand + arg parsing ─────────────────────────────────────────────
SUBCMD="${1:-}"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tx-gain)       TX_GAIN=$2; shift 2 ;;
        --rx-gain-mode)  RX_GAIN_MODE=$2; shift 2 ;;
        --rx-gain)       RX_GAIN=$2; shift 2 ;;
        --tx-filler-amp) TX_FILLER_AMP=$2; shift 2 ;;
        --mtu)           MTU=$2; shift 2 ;;
        --startup-wait)  STARTUP_WAIT=$2; shift 2 ;;
        --only)          ONLY=$2; shift 2 ;;
        -h|--help)       sed -n '1,32p' "$0"; exit 0 ;;
        *)               echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$SUBCMD" in
    up|down|status) ;;
    ""|-h|--help) sed -n '1,32p' "$0"; exit 0 ;;
    *) echo "Unknown subcommand: $SUBCMD (expected up|down|status)" >&2; exit 2 ;;
esac

case "$ONLY" in
    ""|A|B) ;;
    *) echo "Unknown --only value: $ONLY (expected A or B)" >&2; exit 2 ;;
esac

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (needs netns + ip link set netns)" >&2
    exit 1
fi

# ── Helpers ──────────────────────────────────────────────────────────────
detect_iface() {
    ip -o -4 addr show | awk -v ip="$1" '$4 ~ ("^" ip "/") {print $2; exit}'
}

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
    echo "[info] Tearing down one-way netns…"

    kill_bridge "$PID_A"
    kill_bridge "$PID_B"
    if pgrep -f "pluto\.tun_link" >/dev/null; then
        pkill -INT -f "pluto\.tun_link" 2>/dev/null || true
        sleep 1
        pkill -KILL -f "pluto\.tun_link" 2>/dev/null || true
    fi

    drain_netns "$NS_A"
    drain_netns "$NS_B"

    local ok=1
    for pl in "$PLUTO_A_TX_IP" "$PLUTO_B_RX_IP"; do
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
    echo "── one-way netns status ────────────────────────────"
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
            echo "  oneway-$tag: running  (pid $(cat "$pf"))"
        else
            echo "  oneway-$tag: not running"
        fi
    done
    echo "───────────────────────────────────────────────────"
}

# ── Subcommand: up ───────────────────────────────────────────────────────
cmd_up() {
    local do_a=1 do_b=1
    case "$ONLY" in
        A) do_b=0 ;;
        B) do_a=0 ;;
    esac

    if (( do_a )) && ip netns list | grep -qE "^$NS_A\b"; then
        echo "ERROR: netns $NS_A already exists — run 'down' first" >&2
        exit 1
    fi
    if (( do_b )) && ip netns list | grep -qE "^$NS_B\b"; then
        echo "ERROR: netns $NS_B already exists — run 'down' first" >&2
        exit 1
    fi

    # 1. Find the USB ifaces by host-side IP (only the side(s) we need).
    local IFACE_A_TX="" IFACE_B_RX=""
    local missing=""
    if (( do_a )); then
        IFACE_A_TX=$(detect_iface "$HOST_A_TX_IP")
        [[ -z "$IFACE_A_TX" ]] && missing="$missing $HOST_A_TX_IP (A-TX, $PLUTO_A_TX_IP)"
    fi
    if (( do_b )); then
        IFACE_B_RX=$(detect_iface "$HOST_B_RX_IP")
        [[ -z "$IFACE_B_RX" ]] && missing="$missing $HOST_B_RX_IP (B-RX, $PLUTO_B_RX_IP)"
    fi
    if [[ -n "$missing" ]]; then
        echo "ERROR: could not find USB-ethernet interface(s) for:$missing" >&2
        echo "       Check 'ip -4 addr' — Pluto plugged in?" >&2
        exit 1
    fi
    echo "[info] Plutos:"
    (( do_a )) && echo "  A TX: $IFACE_A_TX ($HOST_A_TX_IP ↔ $PLUTO_A_TX_IP)"
    (( do_b )) && echo "  B RX: $IFACE_B_RX ($HOST_B_RX_IP ↔ $PLUTO_B_RX_IP)"

    # 2. Build netns and move ifaces in.
    echo "[info] Creating netns…"
    (( do_a )) && ip netns add "$NS_A"
    (( do_b )) && ip netns add "$NS_B"

    trap 'echo "[err] setup aborted — tearing down" >&2; cmd_down; exit 1' ERR

    if (( do_a )); then
        ip link set "$IFACE_A_TX" netns "$NS_A"
        ip -n "$NS_A" link set lo up
        ip -n "$NS_A" link set "$IFACE_A_TX" up
        ip -n "$NS_A" addr add "$HOST_A_TX_IP/24" dev "$IFACE_A_TX"
    fi
    if (( do_b )); then
        ip link set "$IFACE_B_RX" netns "$NS_B"
        ip -n "$NS_B" link set lo up
        ip -n "$NS_B" link set "$IFACE_B_RX" up
        ip -n "$NS_B" addr add "$HOST_B_RX_IP/24" dev "$IFACE_B_RX"
    fi

    # 3. Sanity-check Pluto(s) reachable from their netns.
    if (( do_a )) && ! ip netns exec "$NS_A" ping -c1 -W2 "$PLUTO_A_TX_IP" >/dev/null; then
        echo "ERROR: Pluto A-TX ($PLUTO_A_TX_IP) unreachable from $NS_A" >&2
        exit 1
    fi
    if (( do_b )) && ! ip netns exec "$NS_B" ping -c1 -W2 "$PLUTO_B_RX_IP" >/dev/null; then
        echo "ERROR: Pluto B-RX ($PLUTO_B_RX_IP) unreachable from $NS_B" >&2
        exit 1
    fi
    echo "[info] Pluto(s) reachable inside their netns."

    # 4. Launch tun_link in TX-only / RX-only mode.
    cd "$PROJ_ROOT"

    if (( do_a )); then
        : > "$LOG_A"
        echo "[info] Starting node A (TX-only, log: $LOG_A)…"
        ip netns exec "$NS_A" "$PYTHON" -m pluto.tun_link \
            --node A --mode tx --tx-ip "$PLUTO_A_TX_IP" \
            --gain "$TX_GAIN" --tx-filler-amp "$TX_FILLER_AMP" \
            --mtu "$MTU" --tun-name "$TUN_NAME" \
            >"$LOG_A" 2>&1 &
        echo $! > "$PID_A"
    fi

    if (( do_b )); then
        : > "$LOG_B"
        echo "[info] Starting node B (RX-only, log: $LOG_B)…"
        ip netns exec "$NS_B" "$PYTHON" -m pluto.tun_link \
            --node B --mode rx --rx-ip "$PLUTO_B_RX_IP" \
            --rx-gain-mode "$RX_GAIN_MODE" --rx-gain "$RX_GAIN" \
            --mtu "$MTU" --tun-name "$TUN_NAME" \
            >"$LOG_B" 2>&1 &
        echo $! > "$PID_B"
    fi

    echo "[info] Waiting ${STARTUP_WAIT}s for tun_link to come up…"
    sleep "$STARTUP_WAIT"

    local entry tag pf log run pid
    for entry in "A:$PID_A:$LOG_A:$do_a" "B:$PID_B:$LOG_B:$do_b"; do
        IFS=: read -r tag pf log run <<<"$entry"
        (( run )) || continue
        pid=$(cat "$pf")
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: tun_link node $tag (pid $pid) died during startup — see logs" >&2
            echo "--- $log (last 30 lines) ---"; tail -n 30 "$log"
            exit 1
        fi
    done

    trap - ERR

    echo
    echo "========================================================================"
    echo "  One-way link is up."
    echo
    if (( do_a )); then
        echo "  Node A (netns $NS_A):  TUN $TUN_NAME = $IP_A   (TX-only)"
        echo "  TUN log:               $LOG_A"
    fi
    if (( do_b )); then
        echo "  Node B (netns $NS_B):  TUN $TUN_NAME = $IP_B   (RX-only)"
        echo "  TUN log:               $LOG_B"
    fi
    if (( do_a && do_b )); then
        echo
        echo "  Drive UDP traffic A → B:"
        echo "    sudo ip netns exec $NS_B iperf3 -s -1 &"
        echo "    sudo ip netns exec $NS_A iperf3 -c $IP_B -u -b 100k -t 30"
        echo
        echo "  Or with netcat:"
        echo "    sudo ip netns exec $NS_B nc -u -l 5000"
        echo "    echo hello | sudo ip netns exec $NS_A nc -u $IP_B 5000"
    fi
    echo
    echo "  Tear down:   sudo $0 down"
    echo "========================================================================"
}

case "$SUBCMD" in
    up)     cmd_up ;;
    down)   cmd_down ;;
    status) cmd_status ;;
esac
