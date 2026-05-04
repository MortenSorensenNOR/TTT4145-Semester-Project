#!/usr/bin/env bash
# Stream a video file over the radio TUN with HEVC (NVENC) + loss-resilience
# knobs tuned for one-way UDP (no ARQ): short closed GOPs, no B-frames,
# multi-slice frames, repeated PAT/PMT/headers.
# Targets 2.5 Mbps video, capped at 3 Mbps.
#
# Usage:
#   scripts/stream_video.sh ~/oled.mp4                 # → 10.0.0.1:5000
#   scripts/stream_video.sh ~/oled.mp4 10.0.0.2 5000   # explicit dst:port
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

# GPU-side scale + format conversion (frames stay in VRAM end-to-end).
VF="scale_cuda=-2:1080:format=yuv420p"

exec ffmpeg -re \
    -hwaccel cuda -hwaccel_output_format cuda \
    -i "$INPUT" \
    -vf "$VF" -r 60 \
    -c:v hevc_nvenc \
    -preset p5 -tune ll \
    -rc vbr -b:v 2500k -maxrate 3M -bufsize 3M \
    -bf 0 -g 30 -forced-idr 1 -no-scenecut 1 \
    -rc-lookahead 20 -spatial_aq 1 -slices 4 \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
