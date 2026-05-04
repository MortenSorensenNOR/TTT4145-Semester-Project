#!/usr/bin/env bash
# Stream a video file with HEVC (NVENC) over SRT — bidirectional UDP transport
# with ARQ retransmit. Use when the link allows return packets so SRT can
# recover the occasional drop within its latency window.
#
# Compared to stream_video.sh (one-way UDP): SRT handles loss recovery, so we
# trade short GOPs / multi-slice / no-B-frames for higher quality settings —
# longer GOP, B-frames as references, full hq tune.
#
# Targets 2.5 Mbps video, capped at 3 Mbps.
#
# Usage:
#   scripts/stream_video_srt.sh ~/oled.mp4                        # → 10.0.0.1:5000
#   scripts/stream_video_srt.sh ~/oled.mp4 192.168.0.169 1234     # explicit dst:port
#   scripts/stream_video_srt.sh ~/oled.mp4 192.168.0.169 1234 200 # custom latency (ms)
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port] [latency-ms]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"
LATENCY_MS="${4:-120}"

LATENCY_US=$(( LATENCY_MS * 1000 ))

# GPU-side scale + format conversion (frames stay in VRAM end-to-end).
VF="scale_cuda=-2:1080:format=yuv420p"

exec ffmpeg -re \
    -hwaccel cuda -hwaccel_output_format cuda \
    -i "$INPUT" \
    -vf "$VF" -r 60 \
    -c:v hevc_nvenc \
    -preset p6 -tune hq \
    -rc vbr -b:v 2500k -maxrate 2.9M -bufsize 6M \
    -rc-lookahead 32 -spatial_aq 1 -temporal_aq 1 -aq-strength 8 \
    -bf 4 -b_ref_mode middle -refs 4 \
    -g 240 \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts \
    "srt://${DEST}:${PORT}?mode=caller&latency=${LATENCY_US}&pkt_size=1316"
