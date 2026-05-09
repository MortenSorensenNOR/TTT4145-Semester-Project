#!/usr/bin/env bash
# Turn this machine into a NAT gateway: forward traffic from $LINK_IF out via $WAN_IF.
# Usage: sudo ./start_router.sh [link_iface] [wan_iface]
# Defaults: link=pluto0, wan=interface holding the default route.
set -euo pipefail

LINK_IF="${1:-pluto0}"
WAN_IF="${2:-$(ip route show default | awk '/default/ {print $5; exit}')}"

if [[ -z "$WAN_IF" ]]; then
    echo "Could not detect WAN interface (no default route). Pass it as arg 2." >&2
    exit 1
fi

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo -E "$0" "$LINK_IF" "$WAN_IF"
fi

echo "Routing $LINK_IF  ->  $WAN_IF"

PREV_FORWARD=$(sysctl -n net.ipv4.ip_forward)

cleanup() {
    echo
    echo "Tearing down..."
    iptables -t nat -D POSTROUTING -o "$WAN_IF" -j MASQUERADE 2>/dev/null || true
    iptables -D FORWARD -i "$LINK_IF" -o "$WAN_IF" -j ACCEPT 2>/dev/null || true
    iptables -D FORWARD -i "$WAN_IF" -o "$LINK_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true
    sysctl -w "net.ipv4.ip_forward=$PREV_FORWARD" >/dev/null
    echo "Done."
}
trap cleanup EXIT INT TERM

sysctl -w net.ipv4.ip_forward=1 >/dev/null
iptables -t nat -A POSTROUTING -o "$WAN_IF" -j MASQUERADE
iptables -A FORWARD -i "$LINK_IF" -o "$WAN_IF" -j ACCEPT
iptables -A FORWARD -i "$WAN_IF" -o "$LINK_IF" -m state --state RELATED,ESTABLISHED -j ACCEPT

LINK_IP=$(ip -4 -br addr show "$LINK_IF" | awk '{print $3}' | cut -d/ -f1)
echo
echo "Gateway is up. On the other machine:"
echo "  - default gateway:  $LINK_IP"
echo "  - DNS:              1.1.1.1   (or 8.8.8.8)"
echo
echo "Press Ctrl-C to stop and undo all changes."
echo

# Idle until interrupted; cleanup runs via trap.
sleep infinity
