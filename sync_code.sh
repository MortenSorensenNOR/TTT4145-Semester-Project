#!/bin/bash
# sync.sh — Push local code to the Raspberry Pi
# Usage: ./sync.sh [file_or_dir...]
#   ./sync.sh                    # sync entire project
#   ./sync.sh pluto/test/test_packet_loss.py   # sync one file

REMOTE="radiotester@100.114.51.4"
REMOTE_DIR="~/TTT4145-Semester-Project"

if [ $# -eq 0 ]; then
    rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude 'vendor' --exclude "uv.lock" \
        ./ "$REMOTE:$REMOTE_DIR/"
else
    for f in "$@"; do
        rsync -avz "$f" "$REMOTE:$REMOTE_DIR/$f"
    done
fi
