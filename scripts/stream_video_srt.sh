#!/usr/bin/env bash
# Stream a video file with HEVC hardware encode on the AMD iGPU (Radeon 780M
# / VCN 4) via VAAPI, transported over SRT. SRT is bidirectional UDP with
# ARQ retransmit inside the latency window.
#
# Targets 2.5 Mbps video.
#
# Usage:
#   scripts/stream_video_srt.sh ~/oled.mp4                        # → 10.0.0.1:5000
#   scripts/stream_video_srt.sh ~/oled.mp4 192.168.0.169 1234     # explicit dst:port
#   scripts/stream_video_srt.sh ~/oled.mp4 192.168.0.169 1234 300 # custom latency (ms)
#
# IMPORTANT: receiver must decode HEVC, not H.264. See run_ffplay_srt.sh.
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video_srt.sh ...
#   HEIGHT=720 scripts/stream_video_srt.sh ...
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port] [latency-ms]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"
# Rule of thumb: SRT latency >= 4*RTT. Radio link RTT ~50 ms, so 250 ms gives
# room for two NAK round-trips before TLPKTDROP starts trashing frames.
LATENCY_MS="${4:-250}"

LATENCY_US=$(( LATENCY_MS * 1000 ))

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"

# HEVC's coding block sizes are flexible; no mod-16 padding hack needed.
VF="scale=-2:${HEIGHT}:flags=lanczos,format=nv12,hwupload"
[[ -n "$FPS" ]] && VF="fps=${FPS},${VF}"

exec ffmpeg -re \
    -vaapi_device "$VAAPI_DEVICE" \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v 3000k -maxrate 3300k -bufsize 1000k \
    -g 60 \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "srt://${DEST}:${PORT}?mode=caller&latency=${LATENCY_US}&pkt_size=1316&maxbw=440000"
