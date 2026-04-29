#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

VF="scale=-2:1080,format=yuv420p"

exec ffmpeg -re -i "$INPUT" \
    -c:v hevc_nvenc \
    -preset p4 -tune ull \
    -rc cbr -b:v 2500k -maxrate 2500k -bufsize 2500k \
    -g 60 -forced-idr 1 \
    -vf "$VF" -r 60 \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    -c:a libopus -b:a 96k -ac 2 \
    -f mpegts \
    "udp://${DEST}:${PORT}?pkt_size=1316"
