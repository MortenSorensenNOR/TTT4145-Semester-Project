#!/bin/bash
# Receive HEVC over MPEG-TS/SRT (listener) with NVDEC (hevc_cuvid) +
# low-latency ffplay. SRT runs over UDP but is bidirectional — the link
# must allow packets in both directions on this port.
#
# Usage:
#   scripts/run_ffplay_srt.sh              # listen on 5000, 120 ms latency
#   scripts/run_ffplay_srt.sh 1234         # explicit port
#   scripts/run_ffplay_srt.sh 1234 200     # custom latency (ms)
set -euo pipefail

PORT="${1:-5000}"
LATENCY_MS="${2:-120}"
LATENCY_US=$(( LATENCY_MS * 1000 ))

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec hevc_cuvid \
    "srt://0.0.0.0:${PORT}?mode=listener&latency=${LATENCY_US}"
