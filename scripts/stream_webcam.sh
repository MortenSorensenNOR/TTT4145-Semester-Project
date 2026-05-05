#!/usr/bin/env bash
# Stream a webcam (V4L2) + microphone (PulseAudio) with HEVC hardware encode
# on the AMD iGPU. Live source — no `-re` needed; the camera paces itself.
#
# Targets 2.5 Mbps video, capped at 2.9 Mbps (CBR), plus 96 kbps Opus audio.
#
# Receiver: scripts/run_ffplay_hevc.sh
#
# Usage:
#   scripts/stream_webcam.sh                      # → 10.0.0.1:5000
#   scripts/stream_webcam.sh 10.0.0.2 5000        # explicit dst:port
#
# Knobs via env vars (defaults work for most laptop webcams):
#   VIDEO_DEV=/dev/video2                         # different camera
#   WIDTH=1920 HEIGHT=1080                        # if webcam supports it
#   FRAMERATE=30
#   INPUT_FORMAT=yuyv422                          # if webcam doesn't do MJPEG
#   PULSE_SOURCE=alsa_input.usb-...               # specific mic
#   AUDIO=0                                       # mute (video only)
#
# To list what your webcam actually supports:
#   v4l2-ctl --device=/dev/video0 --list-formats-ext
# To list audio sources:
#   pactl list sources short
set -euo pipefail

DEST="${1:-10.0.0.1}"
PORT="${2:-5000}"

VIDEO_DEV="${VIDEO_DEV:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FRAMERATE="${FRAMERATE:-30}"
INPUT_FORMAT="${INPUT_FORMAT:-mjpeg}"
AUDIO="${AUDIO:-1}"
PULSE_SOURCE="${PULSE_SOURCE:-default}"

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

audio_input=""
audio_encode=""
if [[ "$AUDIO" == "1" ]]; then
    audio_input="-thread_queue_size 512 -fflags +nobuffer -f pulse -i $PULSE_SOURCE"
    # `-application lowdelay` shrinks Opus algorithmic delay (~6.5 ms vs. 22.5 ms
    # for the `audio` profile) at the cost of some coding efficiency — fine at 96k.
    audio_encode="-c:a libopus -b:a 96k -ac 2 -application lowdelay -frame_duration 20"
fi

# Low-latency notes:
#   -g $FRAMERATE      → IDR every 1 s (was 60 = 2 s); receiver can lock on faster
#   -bf 0              → no B-frames, kills reorder buffer at encoder + decoder
#   -async_depth 1     → encoder doesn't queue frames internally
#   -muxdelay/preload 0 → MPEG-TS doesn't pre-buffer at start of stream
#   -flush_packets 1   → push every TS packet to UDP immediately
# shellcheck disable=SC2086  # word splitting on audio_* args is intentional
exec ffmpeg \
    -fflags +nobuffer -flags +low_delay \
    -f v4l2 -input_format "$INPUT_FORMAT" \
    -video_size "${WIDTH}x${HEIGHT}" -framerate "$FRAMERATE" \
    -i "$VIDEO_DEV" \
    $audio_input \
    -vaapi_device "$VAAPI_DEVICE" \
    -vf 'format=nv12,hwupload' \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v 2500k \
    -g "$FRAMERATE" -bf 0 -async_depth 1 \
    $audio_encode \
    -muxdelay 0 -muxpreload 0 -flush_packets 1 \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
