#!/usr/bin/env bash
# Stream a video file with H.264 hardware encode on the AMD iGPU (Radeon 780M
# / VCN 4) via VAAPI. Realtime at 1080p60 without CPU load — useful when
# laptop is on USB-C power and CPU can't sustain libx264 -preset slower.
#
# Targets 2.5 Mbps video, capped at 2.9 Mbps.
#
# Usage:
#   scripts/stream_video_vaapi.sh ~/videos/sintel.mkv
#   scripts/stream_video_vaapi.sh ~/videos/sintel.mkv 10.0.0.2 5000
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video_vaapi.sh ...        # halve framerate
#   HEIGHT=720 scripts/stream_video_vaapi.sh ...    # drop to 720p
#   VAAPI_DEVICE=/dev/dri/renderD128 ...            # override device
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

# Defaults are for the system observed: AMD Radeon 780M at renderD129,
# routed through Mesa's radeonsi VAAPI driver (not the libva-nvidia shim
# which would otherwise hijack it).
VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME="${LIBVA_DRIVER_NAME:-radeonsi}"

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"

# CPU decode → upload to VAAPI surface → scale on iGPU → encode on iGPU.
VF="format=nv12|vaapi,hwupload,scale_vaapi=w=-2:h=${HEIGHT}"
[[ -n "$FPS" ]] && VF="fps=${FPS},${VF}"

exec ffmpeg -re \
    -vaapi_device "$VAAPI_DEVICE" \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v h264_vaapi \
    -profile:v high -level 4.2 \
    -rc_mode VBR -b:v 2500k -maxrate 2.9M -bufsize 5M \
    -quality 0 \
    -bf 2 -refs 4 -g 60 -forced-idr 1 \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
