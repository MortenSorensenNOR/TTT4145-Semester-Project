#!/usr/bin/env bash
# Stream a video file with HEVC hardware encode on an NVIDIA GPU (NVENC),
# transported over SRT. NVENC counterpart of stream_video_srt.sh for boxes
# without an AMD iGPU / VAAPI. SRT runs over UDP but is bidirectional, with
# ARQ retransmit inside the latency window — so we can afford a longer GOP
# than the one-way UDP variant.
#
# Targets 3 Mbps video CBR.
#
# Usage:
#   scripts/stream_video_srt_nvenc.sh ~/oled.mp4                        # → 10.0.0.1:5000
#   scripts/stream_video_srt_nvenc.sh ~/oled.mp4 192.168.0.169 1234     # explicit dst:port
#   scripts/stream_video_srt_nvenc.sh ~/oled.mp4 192.168.0.169 1234 200 # custom latency (ms)
#
# IMPORTANT: receiver must decode HEVC, not H.264. See run_ffplay_srt.sh.
#
# Quality knobs via env vars:
#   FPS=30 scripts/stream_video_srt_nvenc.sh ...
#   HEIGHT=720 scripts/stream_video_srt_nvenc.sh ...
#   PRESET=p6 scripts/stream_video_srt_nvenc.sh ...   # p1 fastest .. p7 slowest/best
#   BFRAMES=4 scripts/stream_video_srt_nvenc.sh ...   # Turing+ NVENC only; Volta-gen
#                                                     # (GTX 1650 TU117, Pascal, older)
#                                                     # has no HEVC B-frame support
set -euo pipefail

INPUT="${1:?usage: $0 <input-file> [dest-ip] [dest-port] [latency-ms]}"
DEST="${2:-10.0.0.1}"
PORT="${3:-5000}"
LATENCY_MS="${4:-120}"

LATENCY_US=$(( LATENCY_MS * 1000 ))

HEIGHT="${HEIGHT:-1080}"
FPS="${FPS:-}"
PRESET="${PRESET:-p6}"
BFRAMES="${BFRAMES:-0}"
TEMPORAL_AQ="${TEMPORAL_AQ:-0}"
REFS="${REFS:-}"

# GPU-side scale + format conversion (frames stay in VRAM end-to-end).
VF="scale_cuda=-2:${HEIGHT}:format=yuv420p"
[[ -n "$FPS" ]] && VF="fps=${FPS},${VF}"

bframe_args=( -bf "$BFRAMES" )
(( BFRAMES > 0 )) && bframe_args+=( -b_ref_mode middle )

refs_args=()
[[ -n "$REFS" ]] && refs_args=( -refs "$REFS" )

exec ffmpeg -re \
    -hwaccel cuda -hwaccel_output_format cuda \
    -i "$INPUT" \
    -vf "$VF" \
    -c:v hevc_nvenc \
    -preset "$PRESET" -tune hq \
    -rc cbr -b:v 3000k \
    -rc-lookahead 32 -spatial_aq 1 -temporal_aq "$TEMPORAL_AQ" -aq-strength 8 \
    "${bframe_args[@]}" "${refs_args[@]}" \
    -g 240 \
    -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
    -c:a libopus -b:a 96k -ac 2 -application audio \
    -f mpegts \
    "srt://${DEST}:${PORT}?mode=caller&latency=${LATENCY_US}&pkt_size=1316"
