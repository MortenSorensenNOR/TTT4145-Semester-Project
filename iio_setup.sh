#!/bin/bash
set -e

echo "==> Initializing submodules..."
git submodule update --init --recursive

echo "==> Ensuring /usr/local/lib is in library path..."
if [ ! -f /etc/ld.so.conf.d/local.conf ]; then
    echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/local.conf
fi

echo "==> Removing any conflicting libiio v1.x libraries..."
sudo rm -f /usr/lib/libiio.so.1* /usr/local/lib/libiio.so.1*

echo "==> Building libiio v0.25..."
cd vendor/libiio
git checkout v0.25
rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
sudo ldconfig
cd ../../..

echo "==> Building libad9361-iio..."
cd vendor/libad9361-iio
rm -rf build && mkdir build && cd build
cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DLIBIIO_INCLUDEDIR=/usr/local/include \
    -DLIBIIO_LIBRARIES=/usr/local/lib/libiio.so \
    ..
make -j$(nproc) ad9361
sudo cp -P libad9361.so* /usr/local/lib/
sudo cp ../ad9361.h /usr/local/include/
sudo ldconfig
cd ../../..

echo "==> Syncing Python dependencies..."
uv sync

echo "==> Done!"
