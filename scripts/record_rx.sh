#!/usr/bin/env bash
# Capture the incoming UDP video stream to a file for offline VMAF/SSIM/PSNR
# scoring against the original source. Stops on Ctrl+C, or when the sender
# stops and the UDP timeout fires.
#
# Usage:
#   scripts/record_rx.sh                       # → ./recv.mkv on port 5000
#   scripts/record_rx.sh recv.mkv              # custom output
#   scripts/record_rx.sh recv.mkv 5000         # custom output + port
#
# Notes:
#   - `-c copy` keeps the received bitstream verbatim; transmission errors and
#     dropped TS packets stay in the file so the scoring tool sees what the
#     receiver actually got.
#   - Pair with the original input to stream_video.sh as the reference.
set -euo pipefail

OUT="${1:-recv.mkv}"
PORT="${2:-5000}"

exec ffmpeg -y \
    -fflags nobuffer \
    -i "udp://@:${PORT}?fifo_size=1000000&overrun_nonfatal=1&timeout=5000000" \
    -c copy -copyts \
    "$OUT"
