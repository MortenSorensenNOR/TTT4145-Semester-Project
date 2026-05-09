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
#   scripts/encode_for_link.sh INPUT.mp4 OUTPUT.mp4 2400        # total kbps
#
# Defaults:
#   OUTPUT     = INPUT with "_streaming" suffix
#   TOTAL_KBPS = 2400   (~ 90 % of the measured ~2.59 Mbit/s HTTP ceiling)
#
# Audio is fixed at 96 kbps AAC; video gets TOTAL_KBPS - 100.
set -euo pipefail

INPUT="${1:?usage: $0 INPUT.mp4 [OUTPUT.mp4] [TOTAL_KBPS]}"
TOTAL_KBPS="${3:-2400}"

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
# Per-encode stats path so two encodes in the same CWD don't trample each other.
PASSLOG=$(mktemp -u "${TMPDIR:-/tmp}/encode_link.XXXXXX")
cleanup() {
    rm -f "${PASSLOG}.log" "${PASSLOG}.log.cutree" \
          "${PASSLOG}-0.log" "${PASSLOG}-0.log.cutree" 2>/dev/null || true
}
trap cleanup EXIT

echo "Input    : $INPUT"
echo "Output   : $OUTPUT"
echo "Target   : ${TOTAL_KBPS} kbps total  (${VIDEO_KBPS} kbps video + 96 kbps AAC)"
echo "Encoder  : libx265 -preset veryslow, 2-pass"
echo

echo "[1/2] analysis pass ..."
ffmpeg -y -hide_banner -loglevel warning -stats \
    -i "$INPUT" \
    -c:v libx265 -preset veryslow \
    -b:v "${VIDEO_KBPS}k" \
    -x265-params "pass=1:stats=${PASSLOG}.log" \
    -an -f null /dev/null

echo
echo "[2/2] encode pass ..."
ffmpeg -y -hide_banner -loglevel warning -stats \
    -i "$INPUT" \
    -c:v libx265 -preset veryslow \
    -b:v "${VIDEO_KBPS}k" \
    -x265-params "pass=2:stats=${PASSLOG}.log" \
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
