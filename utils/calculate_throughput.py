import time
import numpy as np
import matplotlib.pyplot as plt
from modules.pipeline import *

NUM_PACKETS  = 100
PAYLOAD_SIZE = 150
RX_DECODE_RUNS = 20

config  = PipelineConfig()
tx_pipe = TXPipeline(config)
rx_pipe = RXPipeline(config)

data = np.random.randint(0, 2, PAYLOAD_SIZE * 8, dtype=np.uint8)
pkt = Packet(0, 1, 0, 0, length=PAYLOAD_SIZE, payload=data)
samples = tx_pipe.transmit(pkt)
signal = np.tile(samples, NUM_PACKETS)

time_signal = len(signal) / config.SAMPLE_RATE
throughput  = NUM_PACKETS * PAYLOAD_SIZE * 8 / time_signal

print(f"Time per 100 packets (air): {time_signal * 1_000.0:.3f} ms")
print(f"Throughput: {throughput / 1_000_000:.2f} mbps")

# try decode performance
times = []
for _ in range(RX_DECODE_RUNS):
    t0 = time.perf_counter()
    _ = rx_pipe.receive(signal)
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg_time = np.mean(np.array(times))
print(f"Average receive time RX: {avg_time * 1_000:.3f} ms")
