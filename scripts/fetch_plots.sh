#!/bin/bash
# fetch_plots.sh
# Fetch plots from Pluto via radiotester, replacing old plot directories

set -euo pipefail

REMOTE="radiotester@100.114.51.4"
PLUTO_IP="192.168.3.1"
PLUTO_USER="root"
PLUTO_PASS="analog"

REMOTE_DIR="~/TTT4145-Semester-Project"
PLOTS_SUBDIR="pluto/plots"
PLOTS_PARENT="pluto"

echo "Fetching plots from Pluto ($PLUTO_IP) via radiotester..."

ssh "$REMOTE" << EOF
set -euo pipefail

echo "[Radiotester] Ensuring parent dir exists..."
mkdir -p $REMOTE_DIR/$PLOTS_PARENT

echo "[Radiotester] Removing old plots dir..."
rm -rf $REMOTE_DIR/$PLOTS_SUBDIR

echo "[Radiotester] Pulling fresh plots from Pluto..."
sshpass -p '$PLUTO_PASS' scp -O -r \
    $PLUTO_USER@$PLUTO_IP:$REMOTE_DIR/$PLOTS_SUBDIR \
    $REMOTE_DIR/$PLOTS_PARENT/

echo "[Radiotester] Contents now:"
ls -la $REMOTE_DIR/$PLOTS_SUBDIR || true
EOF

echo "Replacing local plots with fresh copy from radiotester..."
mkdir -p ./pluto
rm -rf ./pluto/plots
scp -r "$REMOTE:$REMOTE_DIR/$PLOTS_SUBDIR" ./pluto/

echo "Local contents now:"
ls -la ./pluto/plots || true

echo "Done."
