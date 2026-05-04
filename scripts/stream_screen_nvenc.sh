#!/usr/bin/env bash
# Stream the desktop (X11 / XWayland) + microphone (PulseAudio) with HEVC
# hardware encode on an NVIDIA GPU (NVENC). NVENC counterpart of
# stream_screen.sh for machines without an AMD iGPU / VAAPI.
#
# Live source — no `-re`. Targets ~2.5 Mbps video CBR (override via BITRATE)
# plus 96 kbps Opus audio. `-tune ll` + small GOP keep latency low.
#
# Receiver: scripts/run_ffplay_hevc.sh
#
# Usage:
#   scripts/stream_screen_nvenc.sh                      # → 10.0.0.1:5000
#   scripts/stream_screen_nvenc.sh 10.0.0.2 5000        # explicit dst:port
#
# Knobs via env vars:
#   DISPLAY_DEV=:0                                # X display to grab (defaults to $DISPLAY or :0)
#   CAPTURE_W=1920 CAPTURE_H=1080                 # source region (default = full screen via xrandr probe)
#   OFFSET_X=0 OFFSET_Y=0                         # top-left of capture region
#   OUT_W=1280 OUT_H=720                          # downscaled encode resolution
#   FRAMERATE=30
#   PULSE_SOURCE=alsa_input.usb-...               # specific mic
#   AUDIO=0                                       # mute (video only)
#   DRAW_MOUSE=0                                  # hide cursor
#   BITRATE=2500k                                 # video target bitrate
#   PRESET=p4                                     # p1 fastest .. p7 slowest/best
set -euo pipefail

DEST="${1:-10.0.0.1}"
PORT="${2:-5000}"

DISPLAY_DEV="${DISPLAY_DEV:-${DISPLAY:-:0}}"

# Probe the screen size off xrandr if the caller didn't set it. Falls back to
# 1920x1080 if xrandr is missing (headless boxes don't usually have it).
if [[ -z "${CAPTURE_W:-}" || -z "${CAPTURE_H:-}" ]]; then
    if command -v xrandr >/dev/null 2>&1; then
        read -r probe_w probe_h < <(DISPLAY="$DISPLAY_DEV" xrandr 2>/dev/null \
            | awk '/^Screen 0/ {gsub(",",""); for(i=1;i<=NF;i++) if($i=="current"){print $(i+1), $(i+3); exit}}')
    fi
    CAPTURE_W="${CAPTURE_W:-${probe_w:-1920}}"
    CAPTURE_H="${CAPTURE_H:-${probe_h:-1080}}"
fi

OFFSET_X="${OFFSET_X:-0}"
OFFSET_Y="${OFFSET_Y:-0}"
OUT_W="${OUT_W:-1280}"
OUT_H="${OUT_H:-720}"
FRAMERATE="${FRAMERATE:-30}"
DRAW_MOUSE="${DRAW_MOUSE:-1}"
AUDIO="${AUDIO:-1}"
PULSE_SOURCE="${PULSE_SOURCE:-default}"
BITRATE="${BITRATE:-2500k}"
PRESET="${PRESET:-p4}"

MAXRATE_NUM=$(awk -v b="$BITRATE" 'BEGIN{
    n=b; sub(/[Kk]/,"",n); sub(/[Mm]/,"",n);
    if (b ~ /[Mm]/) n=n*1000;
    printf "%d", n*1.15
}')
MAXRATE="${MAXRATE_NUM}k"
BUFSIZE="${MAXRATE_NUM}k"

audio_input=""
audio_encode=""
if [[ "$AUDIO" == "1" ]]; then
    audio_input="-f pulse -i $PULSE_SOURCE"
    audio_encode="-c:a libopus -b:a 96k -ac 2 -application audio"
fi

VF="scale=${OUT_W}:${OUT_H}:flags=lanczos,format=yuv420p"

# shellcheck disable=SC2086
exec ffmpeg \
    -f x11grab -framerate "$FRAMERATE" -draw_mouse "$DRAW_MOUSE" \
    -video_size "${CAPTURE_W}x${CAPTURE_H}" \
    -i "${DISPLAY_DEV}+${OFFSET_X},${OFFSET_Y}" \
    $audio_input \
    -vf "$VF" \
    -c:v hevc_nvenc \
    -preset "$PRESET" -tune ll \
    -rc cbr -b:v "$BITRATE" -maxrate "$MAXRATE" -bufsize "$BUFSIZE" \
    -g 60 \
    -zerolatency 1 \
    $audio_encode \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
