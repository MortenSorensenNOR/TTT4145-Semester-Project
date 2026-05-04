#!/bin/bash
# HEVC receiver tuned for minimum glass-to-glass latency.
# Stacks every low-latency ffplay flag: skips probing, ignores corrupt frames,
# syncs to the system clock instead of the demuxer's. Trade-offs:
#   - audio/video may drift slightly (sync ext is less smooth than master)
#   - corrupt frames are dropped on sight (no recovery attempt)
#   - assumes the codec is HEVC (no probe → can't auto-detect a different one)
#
# Use this for live demos where latency matters more than perfect smoothness.
# For "good enough" low-latency with cleaner audio sync, use run_ffplay_hevc.sh.
#
# Note: buffer_size on the UDP URL is the kernel socket buffer, not a playback
# buffer — it absorbs VBR bursts and does NOT add latency. Keep it large.
#
# Usage:
#   scripts/run_ffplay_hevc_lowlatency.sh           # listen on 5000
#   scripts/run_ffplay_hevc_lowlatency.sh 5000      # explicit port
set -euo pipefail

PORT="${1:-5000}"

exec ffplay \
    -probesize 32 -analyzeduration 0 \
    -fflags 'nobuffer+discardcorrupt' \
    -flags low_delay \
    -framedrop \
    -sync ext \
    -vcodec hevc_cuvid \
    "udp://0.0.0.0:${PORT}?listen&buffer_size=8388608&fifo_size=8388608&overrun_nonfatal=1"
