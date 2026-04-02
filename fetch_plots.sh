#!/bin/bash
# fetch_plots.sh
# Fetch plots from Pluto via radiotester

set -e

REMOTE="radiotester@100.114.51.4"
PLUTO_IP="192.168.2.1"
PLUTO_USER="root"
PLUTO_PASS="analog"

REMOTE_DIR="~/TTT4145-Semester-Project"
PLOTS_SUBDIR="pluto/plots"
PLOTS_PARENT="pluto"

echo "Fetching plots from Pluto ($PLUTO_IP) via radiotester..."

ssh "$REMOTE" << EOF
set -e

echo "[Radiotester] Ensuring target dir exists..."
mkdir -p $REMOTE_DIR/$PLOTS_PARENT

echo "[Radiotester] Pulling plots from Pluto..."
sshpass -p '$PLUTO_PASS' scp -O -r \
    $PLUTO_USER@$PLUTO_IP:$REMOTE_DIR/$PLOTS_SUBDIR \
    $REMOTE_DIR/$PLOTS_PARENT/

echo "[Radiotester] Contents now:"
ls -la $REMOTE_DIR/$PLOTS_SUBDIR || true
EOF

echo "Copying plots from radiotester to local..."
mkdir -p ./pluto
scp -r "$REMOTE:$REMOTE_DIR/$PLOTS_SUBDIR" ./pluto/

echo "Local contents now:"
ls -la ./pluto/plots || true

echo "Done."
