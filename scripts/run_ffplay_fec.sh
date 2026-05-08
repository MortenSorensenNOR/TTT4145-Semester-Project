#!/bin/bash
# Receive HEVC/MPEG-TS over SRT with FEC packet filter (no ARQ).
# Pair with stream_video_vaapi_hevc_fec.sh. Matrix params and arq mode
# must match the sender exactly.
#
# Usage:
#   scripts/run_ffplay_fec.sh              # listen on 5000, 750 ms latency
#   scripts/run_ffplay_fec.sh 1234         # explicit port
#   scripts/run_ffplay_fec.sh 1234 600     # custom latency (ms)
set -euo pipefail

PORT="${1:-5000}"
LATENCY_MS="${2:-750}"
LATENCY_US=$(( LATENCY_MS * 1000 ))

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec hevc_cuvid \
    "srt://0.0.0.0:${PORT}?mode=listener&latency=${LATENCY_US}&packetfilter=fec,cols:10,rows:10,layout:staircase,arq:never"
