#!/usr/bin/env bash
# Two-way video call over the radio TUN: send webcam+mic out to the peer AND
# play the peer's incoming stream simultaneously. Run this on BOTH nodes
# (A and B) — each side passes the *other side's* TUN IP as the destination.
#
# The send socket uses an ephemeral source port; the recv socket binds the
# local PORT in listen mode. No port collision — same PORT works for both
# directions.
#
# Receiver path (hevc_cuvid + low-latency ffplay) matches scripts/run_ffplay_hevc.sh.
#
# Usage (from node A, sending to B at 10.0.0.2):
#   scripts/two_way_webcam.sh 10.0.0.2
# Usage (from node B, sending to A at 10.0.0.1):
#   scripts/two_way_webcam.sh 10.0.0.1
#
# Optional:
#   scripts/two_way_webcam.sh 10.0.0.2 5000          # explicit shared port
#   scripts/two_way_webcam.sh 10.0.0.2 5000 5001     # split TX/RX ports
#
# Inherits all stream_webcam.sh env vars (VIDEO_DEV, WIDTH, HEIGHT, FRAMERATE,
# INPUT_FORMAT, AUDIO, PULSE_SOURCE, VAAPI_DEVICE).
set -euo pipefail

DEST="${1:?usage: $0 <peer-ip> [tx-port] [rx-port]}"
TX_PORT="${2:-5000}"
RX_PORT="${3:-$TX_PORT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Track child PIDs so Ctrl-C / peer hangup tears both sides down cleanly
# instead of leaving an orphaned ffmpeg/ffplay holding the camera or socket.
pids=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# RX first so the listening socket is bound before the peer's first packets
# arrive. Without this, an unlucky race burns ~1s of video while ffplay
# initializes and the kernel drops the early packets.
"$SCRIPT_DIR/run_ffplay_hevc.sh" "$RX_PORT" &
pids+=("$!")

# Small head-start for ffplay before we kick off the encoder.
sleep 0.5

"$SCRIPT_DIR/stream_webcam.sh" "$DEST" "$TX_PORT" &
pids+=("$!")

# Exit as soon as either side dies — typical case is the user hitting q in
# ffplay, which should also stop the outbound encode.
wait -n
