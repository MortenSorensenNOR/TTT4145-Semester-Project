#!/usr/bin/env bash
# Stream HEVC/VAAPI over SRT with FEC packet filter and ARQ disabled. Designed
# for radio links where ARQ creates a feedback loop (retransmits compete for
# the same constrained downstream queue and amplify loss). FEC sends ~20% of
# parity packets in the forward direction; receiver recovers losses without
# any reverse-path retransmit traffic.
#
# Matrix: 10x10 staircase. Recovers a contiguous burst of up to ~10 lost
# packets per 100-packet block (≈ 333 ms at 300 pps). Bandwidth budget:
#   2300k video + 96k audio + ~5 % mpegts overhead ≈ 2.5 Mbps payload
#   + 20 % FEC parity                              ≈ 3.0 Mbps on the wire
# Comfortably under the ~3.6 Mbps link.
#
# Usage:
#   scripts/stream_video_vaapi_hevc_fec.sh ~/oled.mp4
#   scripts/stream_video_vaapi_hevc_fec.sh ~/oled.mp4 10.0.0.1 5000
#   scripts/stream_video_vaapi_hevc_fec.sh ~/oled.mp4 10.0.0.1 5000 750
#
# Receiver: run_ffplay_fec.sh — must use matching matrix and arq mode.
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video_vaapi_hevc_fec.sh ...
#   HEIGHT=720 scripts/stream_video_vaapi_hevc_fec.sh ...
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port] [latency-ms]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"
# 750 ms covers one full FEC matrix at 300 pps (~333 ms) plus the SDR pipeline
# buffer chain plus jitter. No ARQ retransmit budget needed (arq:never).
LATENCY_MS="${4:-750}"

LATENCY_US=$(( LATENCY_MS * 1000 ))

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"

VF="scale=-2:${HEIGHT}:flags=lanczos,format=nv12,hwupload"
[[ -n "$FPS" ]] && VF="fps=${FPS},${VF}"

exec ffmpeg -re \
    -vaapi_device "$VAAPI_DEVICE" \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v 2300k -maxrate 2500k -bufsize 750k \
    -g 60 \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "srt://${DEST}:${PORT}?mode=caller&latency=${LATENCY_US}&pkt_size=1316&maxbw=440000&packetfilter=fec,cols:10,rows:10,layout:staircase,arq:never"
