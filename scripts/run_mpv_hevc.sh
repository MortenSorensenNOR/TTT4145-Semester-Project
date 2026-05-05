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
#   --video-sync=display-desync → don't snap frames to integer display-refresh
#                                cycles. On a 144 Hz monitor playing 30 fps,
#                                integer alignment picks 5 vblanks per frame =
#                                28.8 fps display rate = 4 % drift. Buffer grows
#                                forever. This option just displays each frame
#                                ASAP at the next vblank with no rate matching.
#   --untimed                  → ignore PTS pacing, render frame the instant
#                                the decoder spits it out
#   --framedrop=decoder+vo     → drop late frames at both stages, like ffplay
#                                -framedrop, so we stay near wall-clock
#   --no-correct-pts           → don't reorder by PTS; trust decode order
#   --speed=1.01               → tiny over-speed nudges the audio clock forward
#                                so the buffer drains and we don't accumulate
#                                latency over time (only used when audio is on)
#   --no-cache + small demuxer buffers → no demuxer-side ring buffer
#   --no-interpolation        → kill the 1-frame motion-interpolation buffer
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

# UDP recv buffer was 8 MB — at 2.5 Mbps that's ~25 s of accumulation capacity.
# 256 KiB is ~0.8 s headroom; if we fall behind that, the kernel drops packets
# (visible as artifacts) instead of silently growing latency.
exec mpv \
    --profile=low-latency \
    --no-cache \
    --cache-secs=0 \
    --demuxer-max-bytes=128KiB \
    --demuxer-max-back-bytes=0 \
    --demuxer-readahead-secs=0 \
    --hwdec="$HWDEC" \
    --untimed \
    --video-sync=display-desync \
    --framedrop=decoder+vo \
    --no-correct-pts \
    --no-interpolation \
    --vd-lavc-threads=1 \
    --demuxer-lavf-probesize=32 \
    --demuxer-lavf-analyzeduration=0 \
    --demuxer-lavf-o-add=fflags=+nobuffer+discardcorrupt \
    --stream-lavf-o-add=buffer_size=262144 \
    --stream-lavf-o-add=fifo_size=262144 \
    --stream-lavf-o-add=overrun_nonfatal=1 \
    "${audio_args[@]}" \
    "udp://0.0.0.0:${PORT}?listen"
