#!/usr/bin/env bash
# Receive HEVC over MPEG-TS/UDP with mpv tuned to ffplay-style "be live, drop
# late frames" behavior. The default --profile=low-latency alone is NOT enough
# — it leaves --video-sync=audio, which pegs video latency to whatever the
# audio output buffer happens to be (typically 100–500 ms on PulseAudio).
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
#   AUDIO=0           # drop the audio track entirely on the receiver — the
#                     # tightest live latency mode. Default 1 keeps audio but
#                     # forces video-sync=desync so video doesn't wait for it.
set -euo pipefail

PORT="${1:-5000}"
HWDEC="${HWDEC:-auto-safe}"
AUDIO="${AUDIO:-1}"

# Why each flag matters:
#   --untimed                  → ignore PTS pacing, render frame the instant
#                                the decoder spits it out (this is the big one)
#   --video-sync=desync        → don't drop/dup to match a clock; just display
#                                (overrides the low-latency profile's =audio)
#   --framedrop=decoder+vo     → drop late frames at both stages, like ffplay
#                                -framedrop, so we stay near wall-clock
#   --no-correct-pts           → don't reorder by PTS; trust decode order
#   --speed=1.01               → tiny over-speed nudges the audio clock forward
#                                so the buffer drains and we don't accumulate
#                                latency over time (only used when audio is on)
#   --no-cache + --demuxer-max-bytes=512KiB → no demuxer-side ring buffer
#   --vo=gpu --gpu-context=auto → modern renderer; --no-interpolation kills
#                                the 1-frame motion-interpolation buffer
audio_args=()
if [[ "$AUDIO" == "1" ]]; then
    # Smallest sane PulseAudio buffer; --speed=1.01 keeps it drained.
    audio_args=(
        --audio-buffer=0
        --speed=1.01
        --audio-pitch-correction=no
    )
else
    audio_args=(--no-audio)
fi

exec mpv \
    --profile=low-latency \
    --no-cache \
    --demuxer-max-bytes=512KiB \
    --demuxer-max-back-bytes=0 \
    --hwdec="$HWDEC" \
    --untimed \
    --video-sync=desync \
    --framedrop=decoder+vo \
    --no-correct-pts \
    --no-interpolation \
    --vd-lavc-threads=1 \
    --demuxer-lavf-probesize=32 \
    --demuxer-lavf-analyzeduration=0 \
    --demuxer-lavf-o-add=fflags=+nobuffer+discardcorrupt \
    --stream-lavf-o-add=buffer_size=8388608 \
    --stream-lavf-o-add=fifo_size=8388608 \
    --stream-lavf-o-add=overrun_nonfatal=1 \
    "${audio_args[@]}" \
    "udp://0.0.0.0:${PORT}?listen"
