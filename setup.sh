#!/bin/bash
set -e

echo "==> Initializing submodules..."
git submodule update --init --recursive

echo "==> Building libiio..."
cd vendor/libiio
mkdir -p build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../../..

echo "==> Building libad9361-iio..."
cd vendor/libad9361-iio
mkdir -p build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j$(nproc) ad9361
sudo cp -P libad9361.so* /usr/local/lib/
sudo cp ../ad9361.h /usr/local/include/
sudo ldconfig
cd ../../..

echo "==> Syncing Python dependencies..."
uv sync

echo "==> Done!"
