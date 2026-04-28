#!/usr/bin/env bash
# Receive an MPEG-TS UDP video stream, save it to disk, and play it live.
#
# Runs ffmpeg in the background (capture → file + relay to localhost) and
# ffplay in the foreground (reads the relay). Ctrl-C in ffplay tears down
# both. The saved file is the lossy stream exactly as the receiver saw it,
# usable as input for VMAF/PSNR/SSIM against the source.
#
# Usage:
#   scripts/record_and_play.sh                          # /tmp/received.ts, port 5000
#   scripts/record_and_play.sh /tmp/myrun.ts            # custom output path
#   scripts/record_and_play.sh /tmp/myrun.ts 6000       # custom listen port
set -euo pipefail

OUT="${1:-/tmp/received.ts}"
PORT="${2:-5000}"
RELAY_PORT=$((PORT + 50))

cleanup() {
    [[ -n "${FFMPEG_PID:-}" ]] && kill "$FFMPEG_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[info] capture → $OUT  (relay → udp://127.0.0.1:$RELAY_PORT)"

ffmpeg -hide_banner -loglevel warning \
    -i "udp://@:${PORT}?fifo_size=2000000&overrun_nonfatal=1" \
    -map 0 -c copy -f tee \
    "[f=mpegts]${OUT}|[f=mpegts]udp://127.0.0.1:${RELAY_PORT}?pkt_size=1316" &
FFMPEG_PID=$!

# Give ffmpeg a moment to bind and start emitting on the relay port.
sleep 0.5

ffplay -hide_banner -loglevel warning \
    -hwaccel auto -framedrop -fflags nobuffer \
    "udp://@:${RELAY_PORT}?fifo_size=2000000"
