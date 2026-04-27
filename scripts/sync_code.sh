#!/bin/bash
# sync.sh — Push local code to remote targets
# Usage:
#   ./sync.sh                          # sync entire project (default remote)
#   ./sync.sh file_or_dir              # sync specific file/dir (default remote)
#   ./sync.sh pluto A                  # sync to physical Pluto at 192.168.2.1
#   ./sync.sh pluto B                  # sync to physical Pluto at 192.168.3.1
# (The A/B labels are just shorthand for the two Plutos; the TX/RX role on
#  each node is whatever pluto/setup.json currently says.)

DEFAULT_REMOTE="radiotester@100.114.51.4"
REMOTE_DIR="~/TTT4145-Semester-Project"

REMOTE="$DEFAULT_REMOTE"

# =========================
# Handle Pluto mode
# =========================
if [ "$1" == "pluto" ]; then
    if [ "$2" == "A" ]; then
        REMOTE="root@192.168.2.1"
    elif [ "$2" == "B" ]; then
        REMOTE="root@192.168.3.1"
    else
        echo "Usage: ./sync.sh pluto [A|B]"
        exit 1
    fi

    echo "Syncing to Pluto $2 ($REMOTE)..."

    sshpass -p analog rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude 'vendor' --exclude "uv.lock" \
        ./ "$REMOTE:$REMOTE_DIR/"

    exit 0
fi

# =========================
# Default behavior
# =========================
echo "Syncing to default remote ($REMOTE)..."

if [ $# -eq 0 ]; then
    rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' --exclude 'vendor' --exclude "uv.lock" \
        ./ "$REMOTE:$REMOTE_DIR/"

    echo "Syncing from $REMOTE to Pluto A and B..."
    ssh "$REMOTE" "cd $REMOTE_DIR && ./sync_code.sh pluto A && ./sync_code.sh pluto B"
else
    for f in "$@"; do
        rsync -avz "$f" "$REMOTE:$REMOTE_DIR/$f"
    done
fi
