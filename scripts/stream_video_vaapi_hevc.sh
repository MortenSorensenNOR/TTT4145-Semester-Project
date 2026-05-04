#!/usr/bin/env bash
# Stream a video file with HEVC hardware encode on the AMD iGPU (Radeon 780M
# / VCN 4) via VAAPI. HEVC has roughly 30-50% better compression efficiency
# than H.264 at low bitrates, so quality at 2.5 Mbps should be visibly cleaner
# than h264_vaapi for the same content.
#
# Targets 2.5 Mbps video, capped at 2.9 Mbps.
#
# Usage:
#   scripts/stream_video_vaapi_hevc.sh ~/videos/sintel.mkv
#   scripts/stream_video_vaapi_hevc.sh ~/videos/sintel.mkv 10.0.0.2 5000
#
# IMPORTANT: receiver must use hevc_cuvid (or hevc software decode), not
# h264_cuvid as in run_ffplay.sh. Quick one-liner for the receiver:
#   ffplay -fflags nobuffer -flags low_delay -framedrop -vcodec hevc_cuvid \
#     'udp://0.0.0.0:5000?listen&buffer_size=8388608&fifo_size=8388608&overrun_nonfatal=1'
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video_vaapi_hevc.sh ...
#   HEIGHT=720 scripts/stream_video_vaapi_hevc.sh ...
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"

# HEVC's coding block sizes are flexible; no mod-16 padding hack needed.
# Plain mod-2 width preservation is enough.
VF="scale=-2:${HEIGHT}:flags=lanczos,format=nv12,hwupload"
[[ -n "$FPS" ]] && VF="fps=${FPS},${VF}"

exec ffmpeg -re \
    -vaapi_device "$VAAPI_DEVICE" \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v 2500k \
    -g 60 \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
