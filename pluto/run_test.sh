#!/usr/bin/env bash
# run_test.sh — start RX and TX, run for a while, print stats.
#
# Usage:
#   ./pluto/run_test.sh [options passed to one_way_threaded.py]
#
# Examples:
#   ./pluto/run_test.sh --payload 1000 --interval 0
#   ./pluto/run_test.sh --payload 100 --duration 60
#   ./pluto/run_test.sh --payload 10 --tx-gain 0
#
# Options forwarded to one_way_threaded.py:
#   --payload N       Payload size in bytes       (default: 10)
#   --interval N      Inter-burst gap in ms        (default: 200)
#   --packets N       Packets per TX burst         (default: 20)
#   --batch-size N    Packets per DMA call         (default: 8)
#   --gain N          TX hardware gain dB          (default: -30)
#   --cfo-offset N    CFO offset Hz                (default: 15200)
#
# Note: --constellation requires running RX directly (not via this script),
#   since this script redirects RX output to a log file and backgrounds it:
#   uv run pluto/one_way_threaded.py --ip <RX_IP> --mode rx --constellation
#
# Script-only options:
#   --duration N      Run time in seconds          (default: 60)
#   --rx-ip IP        RX Pluto IP                  (default: from pluto/setup.json node A)
#   --tx-ip IP        TX Pluto IP                  (default: from pluto/setup.json node B)

set -euo pipefail
cd "$(dirname "$0")/.."

# Pull defaults from pluto/setup.json so the IPs follow the canonical config.
eval "$(uv run python -m pluto.setup_config --shell-export)"
RX_IP="${PLUTO_A_RX_IP:?missing PLUTO_A_RX_IP from setup.json}"
TX_IP="${PLUTO_B_TX_IP:?missing PLUTO_B_TX_IP from setup.json}"
DURATION=60
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --duration)  DURATION="$2";  shift 2 ;;
        --rx-ip)     RX_IP="$2";     shift 2 ;;
        --tx-ip)     TX_IP="$2";     shift 2 ;;
        *)           PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

RX_LOG=$(mktemp /tmp/rx_XXXXXX.txt)
TX_LOG=$(mktemp /tmp/tx_XXXXXX.txt)

cleanup() {
    kill "$RX_PID" "$TX_PID" 2>/dev/null || true
    wait "$RX_PID" "$TX_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Kill any leftover processes from previous runs
pkill -f one_way_threaded.py 2>/dev/null || true
sleep 1

echo "=== Starting RX (${RX_IP}) ==="
uv run pluto/one_way_threaded.py --ip "$RX_IP" --mode rx "${PASSTHROUGH_ARGS[@]}" \
    > "$RX_LOG" 2>&1 &
RX_PID=$!

sleep 2

echo "=== Starting TX (${TX_IP}) ==="
uv run pluto/one_way_threaded.py --ip "$TX_IP" --mode tx "${PASSTHROUGH_ARGS[@]}" \
    > "$TX_LOG" 2>&1 &
TX_PID=$!

echo "=== Running for ${DURATION}s … (Ctrl-C to stop early) ==="
sleep "$DURATION"

echo ""
echo "=== Stopping ==="
kill "$RX_PID" "$TX_PID" 2>/dev/null || true
wait "$RX_PID" "$TX_PID" 2>/dev/null || true
trap - EXIT INT TERM

echo ""
echo "=== TX log (tail) ==="
tail -5 "$TX_LOG"

echo ""
echo "=== RX results ==="
grep "dropped≈" "$RX_LOG" | tail -1 || true

TOTAL=$(grep -c "valid=True" "$RX_LOG" 2>/dev/null) || TOTAL=0
DROPS=$(grep "dropped≈" "$RX_LOG" | tail -1 | grep -oP "dropped≈\K[0-9]+" 2>/dev/null) || DROPS=0
if [[ ${TOTAL:-0} -gt 0 ]]; then
    SENT=$((TOTAL + DROPS))
    python3 -c "print(f'Decoded: {$TOTAL}, Dropped: {$DROPS}, Total: {$SENT}, Drop rate: {$DROPS/$SENT*100:.2f}%')"
fi

echo ""
echo "Gap distribution:"
grep "GAP" "$RX_LOG" | grep -oP "GAP: \K[0-9]+" | sort -n | uniq -c || true

echo ""
PERF_LINES=$(grep "\[RX perf\]" "$RX_LOG" || true)
if [[ -n "$PERF_LINES" ]]; then
    echo "Processing time (last sample):"
    echo "$PERF_LINES" | tail -1
fi

echo ""
echo "CRC mismatches : $(grep -c 'CRC-16 mismatch' "$RX_LOG" 2>/dev/null || echo 0)"
echo "Tail cutoffs   : $(grep -c 'outside of buffer' "$RX_LOG" 2>/dev/null || echo 0)"

echo ""
echo "Logs: RX=$RX_LOG  TX=$TX_LOG"
