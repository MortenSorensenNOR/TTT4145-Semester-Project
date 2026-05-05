#!/usr/bin/env bash
# Receive HEVC over MPEG-TS/UDP with mpv in low-latency mode.
# Replacement for run_ffplay_hevc.sh — ffplay's A/V sync queue floors latency
# at 200–400 ms; mpv's --profile=low-latency typically sits well under 100 ms
# of receiver-side delay.
#
# Pair with stream_webcam.sh (VAAPI) or stream_webcam_nvenc.sh (NVENC).
#
# Usage:
#   scripts/run_mpv_hevc.sh         # listen on 5000
#   scripts/run_mpv_hevc.sh 5000    # explicit port
#
# Knobs:
#   HWDEC=auto-safe   # default; use HWDEC=nvdec / vaapi to pin a backend,
#                     # HWDEC=no to force software decode
set -euo pipefail

PORT="${1:-5000}"
HWDEC="${HWDEC:-auto-safe}"

# What --profile=low-latency does for us:
#   --audio-buffer=0, --vd-lavc-threads=1, --cache-pause=no,
#   --demuxer-lavf-o-add=fflags=+nobuffer, --video-sync=audio,
#   --interpolation=no, --video-latency-hacks=yes, --stream-buffer-size=4k
#
# Extras we layer on:
#   --no-cache                 → disable demuxer ring entirely (live UDP, no seeking)
#   --untimed=no               → keep A/V sync (we have audio); flip to yes for video-only
#   --demuxer-lavf-probesize/analyzeduration → cut startup probing
#   --stream-lavf-o            → pass UDP socket buffer flags through to libavformat
exec mpv \
    --profile=low-latency \
    --no-cache \
    --hwdec="$HWDEC" \
    --demuxer-lavf-probesize=32 \
    --demuxer-lavf-analyzeduration=0 \
    --demuxer-lavf-o-add=fflags=+nobuffer+discardcorrupt \
    --stream-lavf-o-add=buffer_size=8388608 \
    --stream-lavf-o-add=fifo_size=8388608 \
    --stream-lavf-o-add=overrun_nonfatal=1 \
    "udp://0.0.0.0:${PORT}?listen"
