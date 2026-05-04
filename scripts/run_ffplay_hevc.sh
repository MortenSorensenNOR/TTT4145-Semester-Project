#!/bin/bash
# Receive HEVC over MPEG-TS/UDP with NVDEC (hevc_cuvid) + low-latency ffplay.
# Pair with stream_video_vaapi_hevc.sh (or any HEVC sender).
#
# Usage:
#   scripts/run_ffplay_hevc.sh         # listen on 5000
#   scripts/run_ffplay_hevc.sh 5000    # explicit port
set -euo pipefail

PORT="${1:-5000}"

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec hevc_cuvid \
    "udp://0.0.0.0:${PORT}?listen&buffer_size=8388608&fifo_size=8388608&overrun_nonfatal=1"
