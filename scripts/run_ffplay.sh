#!/bin/bash

ffplay -fflags nobuffer "udp://@:5000?fifo_size=1000000&overrun_nonfatal=1"
