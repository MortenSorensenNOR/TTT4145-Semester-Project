#!/bin/bash
# Receive HEVC over MPEG-TS/UDP with NVDEC (hevc_cuvid) + low-latency ffplay.
#
# Usage:
#   scripts/run_ffplay.sh          # listen on 5000
#   scripts/run_ffplay.sh 5000     # explicit port
set -euo pipefail

PORT="${1:-5000}"

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec hevc_cuvid \
    "udp://0.0.0.0:${PORT}?listen"
