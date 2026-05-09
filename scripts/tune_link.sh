#!/usr/bin/env bash
# Tune the host's TCP/UDP stack for the radio TUN.
#
# Run once per boot per node, AFTER `tun_link_arq.py` has brought pluto0 up
# (the script needs the interface to exist to attach the per-route congctl
# and MSS clamp).
#
#   sudo scripts/tune_link.sh                # iface=pluto0, peer auto-detected
#   sudo scripts/tune_link.sh pluto0 10.0.0.2
#
# Reverts on reboot. To undo manually: `sysctl -w net.core.rmem_max=4194304`
# (or whatever the prior value was), `ip route del <peer>/32`, and
# `iptables -t mangle -D ...` for the two MSS-clamp rules below.
set -euo pipefail

IFACE="${1:-pluto0}"
PEER_IP="${2:-}"

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root" >&2
    exit 1
fi

if ! ip link show "$IFACE" >/dev/null 2>&1; then
    echo "ERROR: interface '$IFACE' not found — bring the radio link up first" >&2
    exit 1
fi

if [[ -z "$PEER_IP" ]]; then
    LOCAL_IP=$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | cut -d/ -f1 | head -1)
    case "$LOCAL_IP" in
        10.0.0.1) PEER_IP=10.0.0.2 ;;
        10.0.0.2) PEER_IP=10.0.0.1 ;;
        *) echo "ERROR: cannot auto-detect peer (local=$LOCAL_IP); pass it explicitly" >&2; exit 1 ;;
    esac
fi

# ffplay/ffmpeg request 8 MB UDP socket buffers in run_ffplay_hevc.sh; setsockopt
# is silently clamped to net.core.rmem_max. With 4 MB the I-frame microburst can
# still fit, but with FEC overhead or any other UDP load it stops fitting.
sysctl -wq net.core.rmem_max=16777216
sysctl -wq net.core.wmem_max=16777216

# BBR is scoped to the per-host route to the peer so it doesn't change global
# congestion control. fq is the qdisc BBR pacing was designed against.
modprobe tcp_bbr 2>/dev/null || true
if grep -qw bbr /proc/sys/net/ipv4/tcp_available_congestion_control; then
    tc qdisc replace dev "$IFACE" root fq 2>/dev/null || true
    ip route replace "$PEER_IP/32" dev "$IFACE" congctl bbr rto_min 75ms initcwnd 20
    CONG="bbr"
else
    # CUBIC over a 50 ms RTT link with ~0.5 % residual loss collapses cwnd often;
    # at minimum lower rto_min so recovery happens in <1 RTT instead of 200 ms.
    ip route replace "$PEER_IP/32" dev "$IFACE" rto_min 75ms initcwnd 20
    CONG="(BBR not in kernel — kept default, lowered rto_min)"
fi

# MSS clamp on SYNs going out via the radio. PMTU discovery often breaks across
# TUN, leaving TCP at the 536-byte fallback MSS — disastrous goodput on a link
# whose effective MTU is 1500. OUTPUT covers locally-originated traffic
# (ffmpeg, iperf, ssh on this host); FORWARD covers anything routed via us.
for CHAIN in OUTPUT FORWARD; do
    iptables -t mangle -C "$CHAIN" -o "$IFACE" -p tcp --tcp-flags SYN,RST SYN \
        -j TCPMSS --clamp-mss-to-pmtu 2>/dev/null || \
    iptables -t mangle -A "$CHAIN" -o "$IFACE" -p tcp --tcp-flags SYN,RST SYN \
        -j TCPMSS --clamp-mss-to-pmtu
done

cat <<EOF
[ok] $IFACE tuned
     rmem_max/wmem_max  = 16 MB
     route $PEER_IP/32  = congctl=$CONG  rto_min=75ms  initcwnd=20
     MSS clamp          = OUTPUT + FORWARD on $IFACE
EOF
