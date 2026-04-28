#!/usr/bin/env bash
# Stream a video file over the radio TUN with HEVC + max loss-resilience.
# Targets ~2.4 Mbps total (video + audio) for a ~3 Mbps lossy link.
#
# Usage:
#   scripts/stream_video.sh ~/oled.mp4                 # → 10.0.0.2:5000
#   scripts/stream_video.sh ~/oled.mp4 10.0.0.2 5000   # explicit dst:port
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"

# Light filter chain: just scale to 1080p, 8-bit. Pre-transcode the source
# to SDR/8-bit (see README/notes) so we don't pay for HDR→SDR every frame.
VF="scale=-2:1080,format=yuv420p"

exec ffmpeg -re -i "$INPUT" \
    -c:v libx264 -preset fast -tune zerolatency \
    -b:v 2000k -maxrate 2800k -bufsize 4800k \
    -x264-params "keyint=30:min-keyint=30:scenecut=0:bframes=0:slices=4:rc-lookahead=20" \
    -vf "$VF" -r 60 \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -fec 1 -packet_loss 10 \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
