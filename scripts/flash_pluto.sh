#!/bin/bash
# Flash custom firmware onto a PlutoSDR if it isn't already running it.
#
# Usage: ./flash_pluto.sh [A|B]
#   A -> 192.168.2.1 (node A, TX Pluto in current setup.json)
#   B -> 192.168.3.1 (node B, RX Pluto in current setup.json)
#
# The A/B argument selects a physical Pluto by IP; the role (TX/RX) on each
# node is whatever pluto/setup.json currently says — flashing is per-device,
# not per-role, so the mapping above is just a label.
#
# The script checks whether the device is already running the custom build
# (detected by "dirty" in the version string).  If not, it calls
# upload_and_test.sh and re-checks.  Up to MAX_ATTEMPTS flashing rounds are
# made because the DFU flash does not always take on the first try.

set -euo pipefail

UPLOAD_SCRIPT="/home/morten/school/plutosdr_fw/upload_and_test.sh"
MAX_ATTEMPTS=3

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [ "$#" -ne 1 ] || { [ "$1" != "A" ] && [ "$1" != "B" ]; }; then
    echo "Usage: $0 [A|B]"
    exit 1
fi

NODE="$1"
if [ "$NODE" == "A" ]; then
    IPADDR=192.168.2.1
else
    IPADDR=192.168.3.1
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ssh_version() {
    # Try common locations where PlutoSDR stores the version string.
    # /etc/motd contains the login banner (including the version line).
    sshpass -p analog ssh \
        -oStrictHostKeyChecking=no \
        -oUserKnownHostsFile=/dev/null \
        -oCheckHostIP=no \
        -oConnectTimeout=5 \
        root@"${IPADDR}" \
        "cat /etc/motd 2>/dev/null || cat /etc/version 2>/dev/null || cat /etc/build 2>/dev/null || echo unknown" 2>/dev/null || true
}

is_custom() {
    local ver
    ver=$(ssh_version)
    echo "  Firmware version reported: ${ver:-<no response>}"
    [[ "${ver}" == *dirty* ]]
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "==> Checking firmware on Pluto ${NODE} (${IPADDR})"

if is_custom; then
    echo "==> Already running custom firmware. Nothing to do."
    exit 0
fi

echo "==> Stock firmware detected. Starting flash loop (max ${MAX_ATTEMPTS} attempts)."

attempt=1
while [ "${attempt}" -le "${MAX_ATTEMPTS}" ]; do
    echo ""
    echo "--- Flash attempt ${attempt}/${MAX_ATTEMPTS} ---"
    # upload_and_test.sh needs to run from its own directory so relative paths work.
    (cd /home/morten/school/plutosdr_fw && bash "${UPLOAD_SCRIPT}" "${NODE}")

    echo ""
    echo "==> Verifying firmware after attempt ${attempt}..."
    if is_custom; then
        echo "==> Custom firmware confirmed. Flash successful."
        exit 0
    fi

    echo "==> Flash did not take (still showing stock version)."
    ((attempt++))
done

echo ""
echo "==> ERROR: Custom firmware not confirmed after ${MAX_ATTEMPTS} attempts."
exit 1
