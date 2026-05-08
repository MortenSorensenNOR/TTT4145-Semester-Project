#!/bin/bash
# Receive HEVC/MPEG-TS over SRT with FEC packet filter (no ARQ).
# Pair with stream_video_vaapi_hevc_fec.sh. Matrix and arq mode must match.
#
# Pipeline:  peer ──SRT+FEC──▶ srt-live-transmit ──UDP/lo──▶ ffplay
#
# Usage:
#   scripts/run_ffplay_fec.sh              # listen on 5000, 750 ms latency
#   scripts/run_ffplay_fec.sh 1234         # explicit port
#   scripts/run_ffplay_fec.sh 1234 600     # custom latency (ms)
set -euo pipefail

PORT="${1:-5000}"
LATENCY_MS="${2:-750}"
LATENCY_US=$(( LATENCY_MS * 1000 ))
LOCAL_PORT="${LOCAL_PORT:-6001}"

cleanup() {
    [[ -n "${RELAY_PID:-}" ]] && kill "$RELAY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

srt-live-transmit \
    "srt://:${PORT}?mode=listener&latency=${LATENCY_US}&packetfilter=fec,cols:10,rows:10,layout:staircase,arq:never" \
    "udp://127.0.0.1:${LOCAL_PORT}" &
RELAY_PID=$!

sleep 0.3

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec hevc_cuvid \
    "udp://127.0.0.1:${LOCAL_PORT}?listen&buffer_size=8388608&fifo_size=8388608&overrun_nonfatal=1"
