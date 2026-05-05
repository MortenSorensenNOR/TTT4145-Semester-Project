#!/usr/bin/env bash
# GStreamer fallback receiver — when mpv's "soft" buffer flags can't keep
# data from piling up. GStreamer gives hard, explicit guarantees:
#
#   queue leaky=downstream max-size-buffers=1   → drop OLD frames the moment
#                                                 a new one arrives, never queue
#   sink sync=false                             → render the instant a frame
#                                                 reaches the sink, no clock
#
# Pair with stream_webcam.sh / stream_webcam_nvenc.sh.
#
# Usage:
#   scripts/run_gst_hevc.sh         # listen on 5000
#   scripts/run_gst_hevc.sh 5000    # explicit port
#
# Knobs:
#   DECODER=vaapih265dec   # explicit hardware decode (AMD/Intel)
#   DECODER=nvh265dec      # NVIDIA hardware decode
#   DECODER=avdec_h265     # software fallback (default; works everywhere)
set -euo pipefail

PORT="${1:-5000}"
DECODER="${DECODER:-avdec_h265}"

# udpsrc buffer-size sets the kernel SO_RCVBUF; 128 KiB ≈ 1 s headroom at
# 1 Mbps before the kernel drops packets (acceptable for live).
exec gst-launch-1.0 -v \
    udpsrc address=0.0.0.0 port="$PORT" buffer-size=131072 \
        caps="application/x-rtp-stream,media=(string)video" \
    ! tsdemux \
    ! h265parse \
    ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream \
    ! "$DECODER" \
    ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream \
    ! videoconvert \
    ! autovideosink sync=false
