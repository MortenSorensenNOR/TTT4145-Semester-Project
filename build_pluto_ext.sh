#!/bin/bash
# Cross-compile pybind11 extensions for PlutoSDR (ARM Cortex-A9)
# Requires a completed plutosdr_fw buildroot build for toolchain + Python headers
set -e

PLUTOSDR_FW="${PLUTOSDR_FW:-$HOME/school/plutosdr_fw}"
BUILDROOT="$PLUTOSDR_FW/buildroot/output"
HOST_BIN="$BUILDROOT/host/bin"
STAGING="$BUILDROOT/staging"

CXX="$HOST_BIN/arm-linux-gnueabihf-g++"
PYTHON_INCLUDE="$BUILDROOT/host/arm-buildroot-linux-gnueabihf/sysroot/usr/include/python3.11"
PYBIND_INCLUDE="$(uv run python -c 'import pybind11; print(pybind11.get_include())')"

FLAGS="-O3 -ffast-math -funroll-loops -fomit-frame-pointer \
       -march=armv7-a -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard \
       -std=c++17 -fPIC -fvisibility=hidden \
       -I$PYTHON_INCLUDE -I$PYBIND_INCLUDE"

if [ ! -x "$CXX" ]; then
    echo "error: cross-compiler not found at $CXX"
    echo "Run a full plutosdr_fw build first, or set PLUTOSDR_FW to your firmware repo path"
    exit 1
fi

echo "==> Compiler: $CXX"
echo "==> Python headers: $PYTHON_INCLUDE"
echo "==> pybind11 headers: $PYBIND_INCLUDE"

compile() {
    local src=$1
    local out=$2
    echo "==> Compiling $src"
    $CXX $FLAGS -shared "$src" -o "$out"
}

compile modules/costas_loop/costas_pybind11.cpp \
        modules/costas_loop/costas_ext.cpython-311-arm-linux-gnueabihf.so

compile modules/gardner_ted/gardner_ext.cpp \
        modules/gardner_ted/gardner_ext.cpython-311-arm-linux-gnueabihf.so

echo "==> Done"
