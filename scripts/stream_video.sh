#!/usr/bin/env bash
# Stream a video file over the radio TUN with libx264 (CPU H.264) over one-way
# UDP. Input decode and encode both run on CPU; libx264 at -preset slower
# gives substantially better quality-per-bit than NVENC at this bitrate.
# (NVDEC was tried as input decoder but its 1088-row internal buffer for
# 1080p didn't always crop cleanly on download to CPU, leaving an 8-row
# "dragged-line" band at the bottom of the picture. CPU decode avoids it.)
#
# Targets 2.5 Mbps video, capped at 2.9 Mbps.
#
# Usage:
#   scripts/stream_video.sh ~/videos/sintel.mkv                 # → 10.0.0.1:5000
#   scripts/stream_video.sh ~/videos/sintel.mkv 10.0.0.2 5000   # explicit dst:port
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video.sh ...        # halve framerate (huge win on motion)
#   HEIGHT=720 scripts/stream_video.sh ...    # drop to 720p
#   PRESET=slow scripts/stream_video.sh ...   # back off if CPU can't keep up at slower
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"
PRESET="${PRESET:-slower}"

# CPU-side scale (lanczos for quality) + format conversion. Frames come down
# from NVDEC into system RAM here, where libx264 picks them up.
VF="scale=-2:${HEIGHT}:flags=lanczos,format=yuv420p"
[[ -n "$FPS" ]] && VF="${VF},fps=${FPS}"

exec ffmpeg -re \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v libx264 \
    -preset "$PRESET" -profile:v high -level 4.2 \
    -b:v 2500k -maxrate 3.2M -bufsize 5M \
    -x264-params "keyint=60:min-keyint=60:scenecut=40:bframes=3:b-adapt=2:ref=4:slices=4:rc-lookahead=60:aq-mode=3:psy-rd=1.0,0.15" \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
