#!/usr/bin/env bash
# Route this machine's traffic through a peer that's running start_router.sh.
# Usage: sudo ./start_client.sh [gateway_ip] [link_iface] [dns]
# Defaults: gateway=10.0.0.2, link=pluto0, dns=1.1.1.1
set -euo pipefail

GW_IP="${1:-10.0.0.2}"
LINK_IF="${2:-pluto0}"
DNS="${3:-1.1.1.1}"

if [[ $EUID -ne 0 ]]; then
    echo "Re-running with sudo..."
    exec sudo -E "$0" "$GW_IP" "$LINK_IF" "$DNS"
fi

if ! ip -br link show "$LINK_IF" >/dev/null 2>&1; then
    echo "Interface $LINK_IF does not exist." >&2
    exit 1
fi

echo "Routing default traffic via $GW_IP on $LINK_IF (DNS $DNS)"

# Snapshot current state so we can restore it.
PREV_DEFAULTS=$(ip route show default || true)
RESOLV=/etc/resolv.conf
RESOLV_BACKUP=$(mktemp /tmp/resolv.conf.bak.XXXXXX)
RESOLV_WAS_SYMLINK=0
RESOLV_LINK_TARGET=""
if [[ -L "$RESOLV" ]]; then
    RESOLV_WAS_SYMLINK=1
    RESOLV_LINK_TARGET=$(readlink "$RESOLV")
fi
cp -a "$RESOLV" "$RESOLV_BACKUP" 2>/dev/null || true

cleanup() {
    echo
    echo "Tearing down..."
    ip route del default via "$GW_IP" dev "$LINK_IF" 2>/dev/null || true
    # Restore prior default routes (there may have been more than one).
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        ip route add $line 2>/dev/null || true
    done <<< "$PREV_DEFAULTS"
    # Restore resolv.conf.
    rm -f "$RESOLV"
    if [[ "$RESOLV_WAS_SYMLINK" -eq 1 ]]; then
        ln -s "$RESOLV_LINK_TARGET" "$RESOLV"
    else
        cp -a "$RESOLV_BACKUP" "$RESOLV"
    fi
    rm -f "$RESOLV_BACKUP"
    echo "Done."
}
trap cleanup EXIT INT TERM

# Replace default route. `ip route replace` handles both the "no default" and
# "already have a default" cases.
ip route replace default via "$GW_IP" dev "$LINK_IF"

# Overwrite resolv.conf (breaking the symlink if there was one — restored on exit).
rm -f "$RESOLV"
printf 'nameserver %s\n' "$DNS" > "$RESOLV"

echo
echo "Up. Test with:"
echo "  curl -s https://ifconfig.me ; echo"
echo "  ping -c 3 1.1.1.1"
echo
echo "Press Ctrl-C to stop and restore the previous routing/DNS."
echo

sleep infinity
