#!/usr/bin/env bash
# Pre-encode a video file for HTTP streaming over the radio TUN.
#
# Two-pass libx265 -preset veryslow with an explicit bitrate target so the
# output average lands within a few percent of the requested kbps. CRF mode
# would beat 2-pass slightly on quality-per-bit, but CRF's average bitrate
# is unpredictable per source — for "fits the link" we want predictable.
#
# Usage:
#   scripts/encode_for_link.sh INPUT.mp4
#   scripts/encode_for_link.sh INPUT.mp4 OUTPUT.mp4
#   scripts/encode_for_link.sh INPUT.mp4 OUTPUT.mp4 2200             # total target kbps
#   scripts/encode_for_link.sh INPUT.mp4 OUTPUT.mp4 2200 2500        # target + hard cap
#   scripts/encode_for_link.sh INPUT.mp4 OUTPUT.mp4 2200 2500 1080   # + target height (px)
#
# Defaults:
#   OUTPUT         = INPUT with "_streaming" suffix
#   TOTAL_KBPS     = 2200   (target average)
#   TOTAL_MAX_KBPS = 2500   (hard ceiling enforced via VBV; ~96 % of ~2.6 Mbit/s link)
#   HEIGHT         = 1080   (target height in pixels; width auto-scaled to keep aspect.
#                            Pass 0 to disable scaling and keep input resolution.)
#
# Audio is fixed at 96 kbps AAC; video target = TOTAL_KBPS - 100,
# video VBV maxrate/bufsize = TOTAL_MAX_KBPS - 100. The bufsize == maxrate
# choice means "never exceed the cap averaged over any 1-second window",
# which is what stops champagne-bubbles-grade scenes from choking the link.
set -euo pipefail

INPUT="${1:?usage: $0 INPUT.mp4 [OUTPUT.mp4] [TOTAL_KBPS] [TOTAL_MAX_KBPS] [HEIGHT]}"
TOTAL_KBPS="${3:-2200}"
TOTAL_MAX_KBPS="${4:-2500}"
HEIGHT="${5:-1080}"

if (( TOTAL_MAX_KBPS < TOTAL_KBPS )); then
    echo "ERROR: TOTAL_MAX_KBPS ($TOTAL_MAX_KBPS) must be >= TOTAL_KBPS ($TOTAL_KBPS)" >&2
    exit 1
fi

if [[ ! -f "$INPUT" ]]; then
    echo "ERROR: input '$INPUT' not found" >&2
    exit 1
fi

if [[ -n "${2:-}" ]]; then
    OUTPUT="$2"
else
    DIR=$(dirname "$INPUT")
    BASE=$(basename "$INPUT")
    NAME="${BASE%.*}"
    OUTPUT="$DIR/${NAME}_streaming.mp4"
fi

VIDEO_KBPS=$(( TOTAL_KBPS - 100 ))
VIDEO_MAX_KBPS=$(( TOTAL_MAX_KBPS - 100 ))
# Per-encode stats path so two encodes in the same CWD don't trample each other.
PASSLOG=$(mktemp -u "${TMPDIR:-/tmp}/encode_link.XXXXXX")
cleanup() {
    rm -f "${PASSLOG}.log" "${PASSLOG}.log.cutree" \
          "${PASSLOG}-0.log" "${PASSLOG}-0.log.cutree" 2>/dev/null || true
}
trap cleanup EXIT

# scale=-2:H keeps aspect ratio and forces width to an even number (codec requirement).
# HEIGHT=0 disables scaling.
if (( HEIGHT > 0 )); then
    SCALE_ARGS=(-vf "scale=-2:${HEIGHT}")
    SCALE_DESC="${HEIGHT}p (width auto, aspect preserved)"
else
    SCALE_ARGS=()
    SCALE_DESC="unchanged (input resolution)"
fi

echo "Input    : $INPUT"
echo "Output   : $OUTPUT"
echo "Scale    : $SCALE_DESC"
echo "Target   : ${TOTAL_KBPS} kbps total  (${VIDEO_KBPS} kbps video + 96 kbps AAC)"
echo "Cap      : ${TOTAL_MAX_KBPS} kbps total  (${VIDEO_MAX_KBPS} kbps video VBV maxrate/bufsize)"
echo "Encoder  : libx265 -preset veryslow, 2-pass"
echo

X265_VBV="vbv-maxrate=${VIDEO_MAX_KBPS}:vbv-bufsize=${VIDEO_MAX_KBPS}"

echo "[1/2] analysis pass ..."
ffmpeg -y -hide_banner -loglevel warning -stats \
    -i "$INPUT" \
    "${SCALE_ARGS[@]}" \
    -c:v libx265 -preset veryslow \
    -b:v "${VIDEO_KBPS}k" \
    -x265-params "pass=1:stats=${PASSLOG}.log:${X265_VBV}" \
    -an -f null /dev/null

echo
echo "[2/2] encode pass ..."
ffmpeg -y -hide_banner -loglevel warning -stats \
    -i "$INPUT" \
    "${SCALE_ARGS[@]}" \
    -c:v libx265 -preset veryslow \
    -b:v "${VIDEO_KBPS}k" \
    -x265-params "pass=2:stats=${PASSLOG}.log:${X265_VBV}" \
    -c:a aac -b:a 96k -ac 2 \
    -movflags +faststart \
    "$OUTPUT"

ACTUAL_BPS=$(ffprobe -v error -show_entries format=bit_rate -of default=nk=1:nw=1 "$OUTPUT")
ACTUAL_KBPS=$(( ACTUAL_BPS / 1000 ))
SIZE=$(du -h "$OUTPUT" | cut -f1)

echo
echo "[ok] $OUTPUT  ($SIZE)"
echo "     actual avg bitrate: ${ACTUAL_KBPS} kbps  (target ${TOTAL_KBPS} kbps)"
echo
echo "Serve:   cd $(dirname "$OUTPUT") && python -m http.server 8000 --bind <local-tun-ip>"
echo "Watch:   http://<sender-tun-ip>:8000/$(basename "$OUTPUT")"
