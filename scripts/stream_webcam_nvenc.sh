#!/usr/bin/env bash
# Stream a webcam (V4L2) + microphone (PulseAudio) with HEVC hardware encode
# on an NVIDIA GPU (NVENC). Mirror of stream_webcam.sh for machines that have
# NVIDIA but no AMD iGPU / VAAPI.
#
# Live source — no `-re` needed; the camera paces itself.
# Targets ~2.5 Mbps video CBR (override with BITRATE), plus 96 kbps Opus audio.
# `-tune ll` + small GOP keeps end-to-end latency low at the cost of some
# coding efficiency vs. the hq tune used in stream_video_srt.sh.
#
# Receiver: scripts/run_ffplay_hevc.sh
#
# Usage:
#   scripts/stream_webcam_nvenc.sh                      # → 10.0.0.1:5000
#   scripts/stream_webcam_nvenc.sh 10.0.0.2 5000        # explicit dst:port
#
# Knobs via env vars:
#   VIDEO_DEV=/dev/video2                         # different camera
#   WIDTH=1920 HEIGHT=1080                        # if webcam supports it
#   FRAMERATE=30
#   INPUT_FORMAT=yuyv422                          # if webcam doesn't do MJPEG
#   PULSE_SOURCE=alsa_input.usb-...               # specific mic
#   AUDIO=0                                       # mute (video only)
#   BITRATE=2500k                                 # video target bitrate
#   PRESET=p4                                     # p1 fastest .. p7 slowest/best
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
BITRATE="${BITRATE:-2500k}"
PRESET="${PRESET:-p4}"

# CBR maxrate sits ~15% above target so the encoder has room to track motion
# without overshooting the radio's ~1.5 Mbps shaped window when BITRATE is
# dialed back for two-way operation.
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

# shellcheck disable=SC2086  # word splitting on audio_* args is intentional
exec ffmpeg \
    -f v4l2 -input_format "$INPUT_FORMAT" \
    -video_size "${WIDTH}x${HEIGHT}" -framerate "$FRAMERATE" \
    -i "$VIDEO_DEV" \
    $audio_input \
    -c:v hevc_nvenc \
    -preset "$PRESET" -tune ll \
    -rc cbr -b:v "$BITRATE" -maxrate "$MAXRATE" -bufsize "$BUFSIZE" \
    -g 60 \
    -zerolatency 1 \
    $audio_encode \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
