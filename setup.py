# setup.py  (project root) building the pybind extensions
import os
import platform
import subprocess
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def _cpu_info() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return f.read().lower()
    except OSError:
        pass
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        ).decode().lower()
    except Exception:
        return platform.processor().lower()


def _compile_flags() -> list[str]:
    import sys
    # MSVC doesn't understand GCC flags — return MSVC equivalents instead.
    if sys.platform == "win32":
        return ["/O2", "/fp:fast", "/GL"]

    cross = os.environ.get("CROSS_COMPILE", "").lower()
    machine = platform.machine().lower() if not cross else cross.split("-")[0]
    cpu = os.environ.get("TARGET_CPU", _cpu_info()).lower()
    base = ["-O3", "-ffast-math", "-funroll-loops", "-fomit-frame-pointer"]

    if machine.startswith("arm"):
        if "cortex-a9" in cpu or "zynq" in cpu:
            return base + ["-march=armv7-a", "-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
        if "cortex-a7" in cpu:
            return base + ["-march=armv7-a", "-mcpu=cortex-a7", "-mfpu=neon-vfpv4", "-mfloat-abi=hard"]
        return base + ["-march=armv7-a", "-mfpu=neon", "-mfloat-abi=hard"]

    if machine in ("aarch64", "arm64"):
        if "cortex-a53" in cpu or "raspberry pi 3" in cpu:
            return base + ["-march=armv8-a", "-mcpu=cortex-a53"]
        if "cortex-a72" in cpu or "raspberry pi 4" in cpu:
            return base + ["-march=armv8-a+simd", "-mcpu=cortex-a72"]
        if "cortex-a76" in cpu or "raspberry pi 5" in cpu:
            return base + ["-march=armv8.2-a+simd", "-mcpu=cortex-a76"]
        return base + ["-march=native"]

    return base + ["-march=native"]


_EXTENSIONS = [
    ("modules.costas_loop.costas_ext", "modules/costas_loop/costas_pybind11.cpp"),
    ("modules.gardner_ted.gardner_ext", "modules/gardner_ted/gardner_ext.cpp"),
    ("modules.frame_sync.frame_sync_ext", "modules/frame_sync/frame_sync_ext.cpp"),
    ("modules.frame_constructor.frame_constructor_ext", "modules/frame_constructor/frame_constructor_pybind11.cpp"),
    ("modules.pulse_shaping.pulse_shaping_ext", "modules/pulse_shaping/pulse_shaping_ext.cpp"),
    ("modules.ldpc.ldpc_ext", "modules/ldpc/ldpc_ext.cpp"),
    ("modules.modulators_ext", "modules/modulators_ext.cpp"),
]

setup(
    ext_modules=[
        Pybind11Extension(
            name=name,
            sources=[src],
            extra_compile_args=_compile_flags(),
            cxx_std=17,
        )
        for name, src in _EXTENSIONS
    ],
    cmdclass={"build_ext": build_ext},
)

