#!/bin/bash
# setup_namespaces.sh — Network namespace helpers for PlutoSDR bridge testing
#
# Namespace "ns_a" gets eth1 (192.168.2.x → Pluto A at 192.168.2.1)
# Namespace "ns_b" gets eth2 (192.168.3.x → Pluto B at 192.168.3.1)
#
# Usage:
#   sudo ./setup_namespaces.sh setup                  # create namespaces, move interfaces
#   sudo ./setup_namespaces.sh teardown                # restore everything
#   sudo ./setup_namespaces.sh calibrate               # measure CFO (A transmits, B measures)
#   sudo ./setup_namespaces.sh bridge A [extra args]   # run bridge in ns_a
#   sudo ./setup_namespaces.sh bridge B [extra args]   # run bridge in ns_b
#   sudo ./setup_namespaces.sh exec A <command...>     # run arbitrary command in ns_a
#   sudo ./setup_namespaces.sh exec B <command...>     # run arbitrary command in ns_b

set -euo pipefail

NS_A="ns_a"
NS_B="ns_b"
ETH_A="eth1"
ETH_B="eth2"
PLUTO_A="192.168.2.1"
PLUTO_B="192.168.3.1"
UV="$(which uv 2>/dev/null || echo /home/radiotester/.local/bin/uv)"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Run a command inside a namespace, from the project directory
ns_exec() {
    local ns="$1"; shift
    ip netns exec "$ns" bash -c "cd $PROJECT_DIR && $*"
}

# Run a python module inside a namespace
ns_python() {
    local ns="$1"; shift
    ns_exec "$ns" "$UV run python -m $*"
}

setup() {
    echo "=== Creating network namespaces ==="

    ip netns add "$NS_A" 2>/dev/null || echo "  $NS_A already exists"
    ip netns add "$NS_B" 2>/dev/null || echo "  $NS_B already exists"

    echo "  Moving $ETH_A → $NS_A"
    ip link set "$ETH_A" netns "$NS_A"

    echo "  Moving $ETH_B → $NS_B"
    ip link set "$ETH_B" netns "$NS_B"

    echo "  Configuring $ETH_A in $NS_A"
    ip netns exec "$NS_A" ip addr add 192.168.2.10/24 dev "$ETH_A"
    ip netns exec "$NS_A" ip link set "$ETH_A" up
    ip netns exec "$NS_A" ip link set lo up

    echo "  Configuring $ETH_B in $NS_B"
    ip netns exec "$NS_B" ip addr add 192.168.3.10/24 dev "$ETH_B"
    ip netns exec "$NS_B" ip link set "$ETH_B" up
    ip netns exec "$NS_B" ip link set lo up

    echo ""
    echo "=== Verifying Pluto connectivity ==="
    echo -n "  $NS_A → $PLUTO_A: "
    ip netns exec "$NS_A" ping -c 1 -W 2 "$PLUTO_A" > /dev/null 2>&1 && echo "OK" || echo "FAIL"
    echo -n "  $NS_B → $PLUTO_B: "
    ip netns exec "$NS_B" ping -c 1 -W 2 "$PLUTO_B" > /dev/null 2>&1 && echo "OK" || echo "FAIL"

    echo ""
    echo "=== Setup complete ==="
}

teardown() {
    echo "=== Tearing down namespaces ==="
    ip netns exec "$NS_A" ip link set "$ETH_A" netns 1 2>/dev/null || true
    ip netns exec "$NS_B" ip link set "$ETH_B" netns 1 2>/dev/null || true
    ip netns del "$NS_A" 2>/dev/null || true
    ip netns del "$NS_B" 2>/dev/null || true
    ip link set "$ETH_A" up 2>/dev/null || true
    ip link set "$ETH_B" up 2>/dev/null || true
    echo "  Done. May need to restart NetworkManager/dhclient for DHCP."
}

calibrate() {
    local direction="${1:-AB}"
    case "$direction" in
        AB|ab)
            echo "=== CFO Calibration (A→B) ==="
            echo "  Starting continuous TX on node A..."
            ns_python "$NS_A" "pluto.transmit 'CFO_CAL' --tx-gain -50 --pluto-ip $PLUTO_A" &
            TX_PID=$!
            sleep 2
            echo "  Measuring CFO on node B (Ctrl-C to stop)..."
            ns_python "$NS_B" "pluto.test.test_measure_cfo --pluto-ip $PLUTO_B" || true
            ;;
        BA|ba)
            echo "=== CFO Calibration (B→A) ==="
            echo "  Starting continuous TX on node B..."
            ns_python "$NS_B" "pluto.transmit 'CFO_CAL' --tx-gain -50 --pluto-ip $PLUTO_B" &
            TX_PID=$!
            sleep 2
            echo "  Measuring CFO on node A (Ctrl-C to stop)..."
            ns_python "$NS_A" "pluto.test.test_measure_cfo --pluto-ip $PLUTO_A" || true
            ;;
        *)  echo "Usage: sudo $0 calibrate [AB|BA]"; exit 1 ;;
    esac

    echo "  Stopping transmitter..."
    kill "$TX_PID" 2>/dev/null || true
    wait "$TX_PID" 2>/dev/null || true
    echo ""
    echo "Pass the measured CFO to the bridge with --rx-cfo-offset <hz>"
}

bridge() {
    local node="$1"; shift
    local ns pluto_ip

    case "$node" in
        A|a) ns="$NS_A"; pluto_ip="$PLUTO_A" ;;
        B|b) ns="$NS_B"; pluto_ip="$PLUTO_B" ;;
        *)   echo "Unknown node: $node (use A or B)"; exit 1 ;;
    esac

    node="${node^^}"  # uppercase
    echo "=== Starting bridge node $node in $ns (pluto $pluto_ip) ==="
    ns_python "$ns" "pluto.bridge --node $node --pluto-ip $pluto_ip $*"
}

ns_exec_cmd() {
    local node="$1"; shift
    local ns

    case "$node" in
        A|a) ns="$NS_A" ;;
        B|b) ns="$NS_B" ;;
        *)   echo "Unknown node: $node (use A or B)"; exit 1 ;;
    esac

    ns_exec "$ns" "$*"
}

case "${1:-}" in
    setup)      setup ;;
    teardown)   teardown ;;
    calibrate)  shift; calibrate "${1:-AB}" ;;
    bridge)     shift; bridge "$@" ;;
    exec)       shift; ns_exec_cmd "$@" ;;
    *)
        echo "Usage: sudo $0 {setup|teardown|calibrate|bridge|exec}"
        echo ""
        echo "  setup                    Create namespaces and move interfaces"
        echo "  teardown                 Restore interfaces and delete namespaces"
        echo "  calibrate [AB|BA]        Measure CFO offset (default: A→B)"
        echo "  bridge {A|B} [args...]   Run pluto.bridge in the node's namespace"
        echo "  exec {A|B} <cmd...>      Run arbitrary command in node's namespace"
        echo ""
        echo "Examples:"
        echo "  sudo $0 setup"
        echo "  sudo $0 calibrate"
        echo "  sudo $0 bridge A"
        echo "  sudo $0 bridge B --rx-cfo-offset 15503"
        echo "  sudo $0 exec A ping 10.0.0.2"
        echo "  sudo $0 exec B iperf3 -s"
        exit 1
        ;;
esac
