#!/usr/bin/env bash
# Stream a webcam (V4L2) + microphone (PulseAudio) with HEVC VAAPI encode,
# tuned for Microsoft-Teams-class glass-to-glass latency (~150 ms target with
# a 50 ms-RTT link, no FEC because the physical layer is lossless).
#
# Receiver: scripts/run_mpv_hevc.sh
#
# Usage:
#   scripts/stream_webcam.sh                      # → 10.0.0.1:5000
#   scripts/stream_webcam.sh 10.0.0.2 5000
#
# Latency-tilted knobs (defaults pick the low-latency option):
#   FRAMERATE=60          # higher fps = shorter frame interval = lower latency
#   HEIGHT=480            # smaller frames encode/decode/transmit faster
#   BITRATE=1500k         # lower CBR target = smaller frames = less serialisation
#   AUDIO=0               # drop mic; saves Opus algorithmic delay (~6.5 ms)
#   VIDEO_DEV=/dev/video2 PULSE_SOURCE=...
set -euo pipefail

DEST="${1:-10.0.0.1}"
PORT="${2:-5000}"

VIDEO_DEV="${VIDEO_DEV:-/dev/video0}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FRAMERATE="${FRAMERATE:-30}"
INPUT_FORMAT="${INPUT_FORMAT:-mjpeg}"
BITRATE="${BITRATE:-2000k}"
AUDIO="${AUDIO:-1}"
PULSE_SOURCE="${PULSE_SOURCE:-default}"

VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD129}"
export LIBVA_DRIVER_NAME=radeonsi

audio_input=""
audio_encode=""
if [[ "$AUDIO" == "1" ]]; then
    # nobuffer + small thread_queue: don't let pulse stack up audio packets
    # while waiting for the first video frame.
    audio_input="-thread_queue_size 64 -fflags +nobuffer -f pulse -i $PULSE_SOURCE"
    # Opus -application lowdelay drops the algorithmic delay from ~22.5 ms
    # (audio profile) to ~6.5 ms; -frame_duration 10 caps packetisation latency
    # at one Opus frame instead of three (default 20 ms).
    audio_encode="-c:a libopus -b:a 64k -ac 1 -application lowdelay -frame_duration 10"
fi

# Latency budget per stage (HEVC VAAPI on Radeon 780M, 30 fps, 720p):
#   capture (MJPEG decode)  ~5 ms
#   encoder (async_depth=1) ~1 frame = 33 ms
#   network (RTT/2)         ~25 ms
#   demux + decode (mpv)    ~10 ms
#   vsync                   ~8 ms (one half-refresh on average)
# Bigger latency wins come from FRAMERATE=60 (frame interval halved) or
# HEIGHT=480 (camera, encoder and decoder all proportionally faster).
#
# Why each flag matters:
#   -fflags nobuffer +flags low_delay → libavformat doesn't wait for stream
#                                       probing on the input side
#   -bf 0                → no B-frames, no reorder delay
#   -g $FRAMERATE        → IDR every 1 s; receiver locks on within a second
#                          even if it joins late
#   -async_depth 1       → VAAPI returns each frame before queuing the next
#   -rc_mode CBR -maxrate=b:v -bufsize one-frame-worth → the rate controller
#                          can't accumulate a fat VBV; every frame is sized for
#                          this instant's bandwidth, no IDR-spike backlog
#   -muxdelay/preload 0  → MPEG-TS doesn't pre-buffer at start of stream
#   -flush_packets 1     → flush each TS packet to the UDP socket immediately
#   -max_delay 0         → libav muxer doesn't reorder for B-frames (we have none)
BUFSIZE_KBPS="$(( ${BITRATE%k} / FRAMERATE ))"

# shellcheck disable=SC2086  # word splitting on audio_* args is intentional
exec ffmpeg \
    -fflags nobuffer -flags low_delay \
    -f v4l2 -input_format "$INPUT_FORMAT" \
    -video_size "${WIDTH}x${HEIGHT}" -framerate "$FRAMERATE" \
    -i "$VIDEO_DEV" \
    $audio_input \
    -vaapi_device "$VAAPI_DEVICE" \
    -vf 'format=nv12,hwupload' \
    -c:v hevc_vaapi -rc_mode CBR \
    -b:v "$BITRATE" -maxrate "$BITRATE" -bufsize "${BUFSIZE_KBPS}k" \
    -g "$FRAMERATE" -bf 0 -async_depth 1 \
    $audio_encode \
    -max_delay 0 -muxdelay 0 -muxpreload 0 -flush_packets 1 \
    -f mpegts -mpegts_flags +resend_headers -pat_period 0.1 -sdt_period 0.1 \
    "udp://${DEST}:${PORT}?pkt_size=1316"
