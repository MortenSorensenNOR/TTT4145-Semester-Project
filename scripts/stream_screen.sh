#!/usr/bin/env bash
# Stream the desktop (X11 / XWayland) + microphone (PulseAudio) with HEVC
# hardware encode on the AMD iGPU. Live source — no `-re` needed; the
# screen-grabber paces itself at FRAMERATE.
#
# Targets 2.5 Mbps video, capped at 2.9 Mbps (CBR), plus 96 kbps Opus audio.
#
# Receiver: scripts/run_ffplay_hevc.sh
#
# Usage:
#   scripts/stream_screen.sh                      # → 10.0.0.1:5000
#   scripts/stream_screen.sh 10.0.0.2 5000        # explicit dst:port
#
# Knobs via env vars:
#   DISPLAY_DEV=:1                                # XWayland display (default :1, falls back to $DISPLAY)
#   CAPTURE_W=2560 CAPTURE_H=1600                 # source region (defaults to full screen)
#   OFFSET_X=0 OFFSET_Y=0                         # top-left of capture region
#   OUT_W=1280 OUT_H=800                          # downscaled encode resolution
#   FRAMERATE=30
#   PULSE_SOURCE=alsa_input.usb-...               # specific mic
#   AUDIO=0                                       # mute (video only)
#   DRAW_MOUSE=0                                  # hide cursor
#
# To capture a region instead of the full screen, set CAPTURE_W/H + OFFSET_X/Y.
# To list audio sources:
#   pactl list sources short
set -euo pipefail

DEST="${1:-10.0.0.1}"
PORT="${2:-5000}"

DISPLAY_DEV="${DISPLAY_DEV:-${DISPLAY:-:1}}"
CAPTURE_W="${CAPTURE_W:-2560}"
CAPTURE_H="${CAPTURE_H:-1600}"
OFFSET_X="${OFFSET_X:-0}"
OFFSET_Y="${OFFSET_Y:-0}"
OUT_W="${OUT_W:-1280}"
OUT_H="${OUT_H:-800}"
FRAMERATE="${FRAMERATE:-30}"
DRAW_MOUSE="${DRAW_MOUSE:-1}"
AUDIO="${AUDIO:-1}"
PULSE_SOURCE="${PULSE_SOURCE:-default}"

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

audio_input=""
audio_encode=""
if [[ "$AUDIO" == "1" ]]; then
    audio_input="-f pulse -i $PULSE_SOURCE"
    audio_encode="-c:a libopus -b:a 96k -ac 2 -application audio"
fi

# Scale to OUT_W x OUT_H, ensure even dimensions, then upload to VAAPI surface.
# format=nv12 happens before hwupload because VAAPI HEVC encode wants NV12 input.
VF="scale=${OUT_W}:${OUT_H}:flags=lanczos,format=nv12,hwupload"

# shellcheck disable=SC2086  # word splitting on audio_* args is intentional
exec ffmpeg \
    -f x11grab -framerate "$FRAMERATE" -draw_mouse "$DRAW_MOUSE" \
    -video_size "${CAPTURE_W}x${CAPTURE_H}" \
    -i "${DISPLAY_DEV}+${OFFSET_X},${OFFSET_Y}" \
    $audio_input \
    -vaapi_device "$VAAPI_DEVICE" \
    -vf "$VF" \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v 2500k \
    -g 60 \
    $audio_encode \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
