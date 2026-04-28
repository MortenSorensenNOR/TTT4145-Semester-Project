ffplay -nodisp -fflags nobuffer -sync ext \
    "udp://@:5001?fifo_size=1000000&overrun_nonfatal=1"
