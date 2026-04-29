#!/bin/bash

gst-launch-1.0 udpsrc port=5000 ! tsdemux ! av1parse ! dav1dec ! autovideosink sync=false
