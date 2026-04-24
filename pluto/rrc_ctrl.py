"""Host-side helper to toggle the FPGA hardware-RRC path on pluto_custom.

The custom bitstream exposes an AXI GPIO at physical address 0x41200000
whose bit 0 selects the signal path between the AD9363 and DMA:

    0 → bypass      (software-RRC mode; host sends 4× upsampled, SW-filtered)
    1 → hardware-RRC (host sends raw symbols; FPGA does 4× interp + RRC)

Software mode and GPIO state MUST match — mismatch either double-filters
(GPIO=1 but host upsampled+filtered) or loses the pulse shape entirely
(GPIO=0 but host sent raw symbols). Neither recovers.

This module runs on the *host*, not the Pluto. It reaches the register
over SSH + busybox `devmem`, because the kernel in the v0.39 custom build
lacks the `gpio-xilinx` driver, so `/sys/class/gpio` has no chip for this
IP. The AXI GPIO is synthesised with all-outputs, so the TRI register is
hard-wired — writing DATA alone is sufficient.
"""

from __future__ import annotations

import subprocess

_AXI_GPIO_DATA = "0x41200000"
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=5",
]


def _ssh(ip: str, remote_cmd: str) -> str:
    cmd = ["sshpass", "-p", "analog", "ssh", *_SSH_OPTS, f"root@{ip}", remote_cmd]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        msg = result.stderr.strip() or f"ssh exit {result.returncode}"
        raise RuntimeError(f"{ip}: {msg}")
    return result.stdout.strip()


def set_hardware_rrc(ip: str, enable: bool) -> None:
    """Drive bit 0 of the FPGA AXI GPIO on the Pluto at ``ip``."""
    val = "0x00000001" if enable else "0x00000000"
    _ssh(ip, f"busybox devmem {_AXI_GPIO_DATA} 32 {val}")


def get_hardware_rrc(ip: str) -> bool:
    """Return True if the FPGA hardware-RRC path is currently enabled."""
    out = _ssh(ip, f"busybox devmem {_AXI_GPIO_DATA}")
    return int(out, 16) & 1 == 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[2] not in ("on", "off", "status"):
        print("usage: rrc_ctrl.py <pluto-ip> {on|off|status}", file=sys.stderr)
        sys.exit(2)
    ip, cmd = sys.argv[1], sys.argv[2]
    if cmd == "status":
        print("on" if get_hardware_rrc(ip) else "off")
    else:
        set_hardware_rrc(ip, cmd == "on")
        print(f"{ip}: hardware_rrc={'on' if get_hardware_rrc(ip) else 'off'}")
