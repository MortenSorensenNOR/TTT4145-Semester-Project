#!/bin/bash
# Receive H.264 over MPEG-TS/UDP with NVDEC (h264_cuvid) + low-latency ffplay.
# buffer_size=8 MB on the UDP socket to absorb VBR bursts; without it, intense
# scenes overrun Linux's default ~256 KB UDP receive buffer and ffplay reports
# "Packet corrupt" as MPEG-TS sync is lost.
#
# Usage:
#   scripts/run_ffplay.sh          # listen on 5000
#   scripts/run_ffplay.sh 5000     # explicit port
#
# If you still see "Packet corrupt" under heavy load, also raise the kernel cap:
#   sudo sysctl -w net.core.rmem_max=16777216
set -euo pipefail

PORT="${1:-5000}"

exec ffplay \
    -fflags nobuffer -flags low_delay -framedrop \
    -vcodec h264_cuvid \
    "udp://0.0.0.0:${PORT}?listen&buffer_size=8388608&fifo_size=8388608&overrun_nonfatal=1"
