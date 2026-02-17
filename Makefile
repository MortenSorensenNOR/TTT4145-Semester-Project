.PHONY: deps deps-system libiio libad9361 sync submodules clean setup

# Full setup: submodules + C libs + Python deps
setup: submodules deps
	uv sync

# Just sync Python deps (after C libs are installed)
sync:
	uv sync

# Initialize/update git submodules
submodules:
	git submodule update --init --recursive

# Build C library dependencies
deps: libiio libad9361

# Install system dependencies (Arch Linux)
deps-system:
	sudo pacman -S --needed base-devel git libxml2 bison flex cmake libusb avahi libaio

libiio:
	cd vendor/libiio && mkdir -p build && cd build && \
		cmake .. && make -j$$(nproc) && sudo make install
	sudo ldconfig

libad9361:
	cd vendor/libad9361-iio && mkdir -p build && cd build && \
		cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
			-DLIBIIO_INCLUDEDIR=$$(pkg-config --variable=includedir libiio)/iio \
			.. && make -j$$(nproc) ad9361 && \
		sudo cp -P libad9361.so* /usr/local/lib/ && \
		sudo cp ../ad9361.h /usr/local/include/ && \
		sudo ldconfig

# Clean C library builds
clean:
	rm -rf vendor/libiio/build vendor/libad9361-iio/build
