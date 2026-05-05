#!/usr/bin/env bash
# Receive HEVC over MPEG-TS/UDP with mpv in low-latency mode and a live
# buffer/drop readout in the terminal — easy way to spot when latency creeps
# up because the demuxer cache is filling rather than draining.
#
# Pair with scripts/stream_webcam.sh.
#
# Usage:
#   scripts/run_mpv_hevc.sh         # listen on 5000
#   scripts/run_mpv_hevc.sh 5000
#
# Knobs:
#   HWDEC=auto-safe  (default)   HWDEC=vaapi / nvdec / no
#   AUDIO=1                       enable audio (default off — assumes no
#                                 speaker on the receive side; mpv otherwise
#                                 underruns the (absent) sink)
#
# Hot keys while playing:
#   i   toggle the stats overlay (full demuxer + decoder timings)
#   I   pin the stats overlay
set -euo pipefail

PORT="${1:-5000}"
HWDEC="${HWDEC:-auto-safe}"
AUDIO="${AUDIO:-0}"

# Why each flag matters:
#   --video-sync=display-desync → don't snap each frame to an integer count of
#                                 display refreshes. On a 165 Hz screen showing
#                                 30 fps, integer-snap picks 6 vblanks = 27.5
#                                 fps consume rate, so the demuxer cache grows
#                                 ~8 % per second. desync just shows each frame
#                                 at the next vblank with no rate matching.
#   --framedrop=vo              → drop late frames AT THE OUTPUT only. Decoder
#                                 framedrop on HEVC breaks the P-frame ref
#                                 chain (no B-frames here, but P still refs
#                                 the previous frame) and produces "could not
#                                 ref with poc" until the next IDR.
#   --no-correct-pts +          → mpv plays whatever the decoder produces in
#   --container-fps-override=…    arrival order, paced by the source fps. With
#                                 correct-pts on, MPEG-TS PCR jitter forces
#                                 mpv to re-time frames, which adds buffering.
#   --demuxer-readahead-secs=0  → no read-ahead — keep the demuxer cache near
#                                 zero so anything that does sit in it shows
#                                 up immediately as added latency.
#   --term-status-msg           → live terminal readout: demuxer cache
#                                 duration, dropped frames, vo-delayed frames.
#                                 If "cache" climbs above ~50 ms, the network
#                                 jitter buffer is filling; if drops climb,
#                                 the encoder is over the link's capacity.
audio_args=(--no-audio)
if [[ "$AUDIO" == "1" ]]; then
    audio_args=(--audio-buffer=0 --audio-pitch-correction=no)
fi

exec mpv \
    --profile=low-latency \
    --no-cache \
    --demuxer-readahead-secs=0 \
    --hwdec="$HWDEC" \
    --video-sync=display-desync \
    --framedrop=vo \
    --no-interpolation \
    --demuxer-lavf-probesize=32 \
    --demuxer-lavf-analyzeduration=0 \
    --demuxer-lavf-o-add=fflags=+nobuffer+discardcorrupt \
    --stream-lavf-o-add=buffer_size=262144 \
    --stream-lavf-o-add=fifo_size=262144 \
    --stream-lavf-o-add=overrun_nonfatal=1 \
    --term-status-msg='cache=${demuxer-cache-duration} drops=vo:${vo-drop-frame-count}/dec:${decoder-frame-drop-count} delayed=${vo-delayed-frame-count}' \
    "${audio_args[@]}" \
    "udp://0.0.0.0:${PORT}?listen"
