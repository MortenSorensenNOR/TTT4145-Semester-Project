#!/usr/bin/env bash
# Receive HEVC over MPEG-TS/UDP with mpv in low-latency mode.
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
#   AUDIO=0           # drop the audio track on the receiver
set -euo pipefail

PORT="${1:-5000}"
HWDEC="${HWDEC:-auto-safe}"
AUDIO="${AUDIO:-1}"

# Two flags that actually matter for live UDP HEVC:
#   --video-sync=display-desync → don't snap frames to integer display-refresh
#                                 cycles. Default behavior on a 165 Hz monitor
#                                 picks 6 vblanks per 30 fps frame = 27.5 fps
#                                 consume rate, so the buffer grows ~8 % per
#                                 second. display-desync just shows each frame
#                                 at the next vblank, no rate alignment.
#   --framedrop=vo              → drop late frames AT THE OUTPUT only. Never
#                                 use decoder framedrop with HEVC: P-frames
#                                 reference their predecessor (even with
#                                 -bf 0), so skipping a decode produces
#                                 "could not ref with poc" until the next IDR.
audio_args=(--no-audio)
if [[ "$AUDIO" == "1" ]]; then
    audio_args=(--audio-buffer=0 --audio-pitch-correction=no)
fi

exec mpv \
    --profile=low-latency \
    --no-cache \
    --hwdec="$HWDEC" \
    --video-sync=display-desync \
    --framedrop=vo \
    --no-interpolation \
    --demuxer-lavf-probesize=32 \
    --demuxer-lavf-analyzeduration=0 \
    --demuxer-lavf-o-add=fflags=+nobuffer+discardcorrupt \
    --stream-lavf-o-add=buffer_size=262144 \
    --stream-lavf-o-add=fifo_size=262144 \
    --stream-lavf-o-add=overrun_nonfatal=1 \
    "${audio_args[@]}" \
    "udp://0.0.0.0:${PORT}?listen"
