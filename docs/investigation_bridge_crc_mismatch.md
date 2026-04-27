# Investigation: bridge.py shows frequent payload CRC mismatches; one_way_threaded.py does not

## The task

Figure out why `pluto/bridge.py` gets a steady stream of `Payload CRC-16 mismatch`
and `IndexError: payload end is outside of buffer` errors from the RX pipeline,
while `pluto/one_way_threaded.py` — running essentially the same DSP pipeline on
the same Pluto hardware over the same coax — does **not**. The user observes 0%
post-ARQ drops at 1.8 Mbps with one_way_threaded, but the bridge's ARQ layer
shows ~14–26% retransmit rate even at low bitrate (~400 kbit/s goodput) with
many underlying frame decode failures.

**This isn't an air-layer SNR problem.** Coarse sync detections fire with
`ratio ≈ 10–11` (well above the `fine_peak_ratio_min = 7.0` gate) and CFO is
measured at 100–150 Hz (tiny, inside the capture range of ~13.5 kHz).
Preambles are being found cleanly. Payloads are corrupt or truncated.

## Symptom (from user's bridge-A.log)

```
01:36:40 [arq-rx] DECODE ERROR (cfo=127 Hz, ratio=11.2): IndexError: payload end is outside of buffer
01:36:40 [arq-rx] 1/1 detections failed decode
01:36:40 [arq-rx] DECODE ERROR (cfo=127 Hz, ratio=11.2): RuntimeError: Payload CRC-16 mismatch: got 0x3be1, expected 0xc4df
01:36:40 [arq-rx] DECODE ERROR (cfo=147 Hz, ratio=10.4): IndexError: payload end is outside of buffer
01:36:40 [arq-rx] 2/2 detections failed decode
01:36:40 [arq-rx] DECODE ERROR (cfo=147 Hz, ratio=10.4): RuntimeError: Payload CRC-16 mismatch: got 0x9f3a, expected 0x5c88
```

Both failure modes (CRC mismatch + IndexError) coexist. The IndexError says
"detected a preamble, but the payload ran off the end of the buffer I was
handed." The CRC mismatch says "decoded the whole payload but got garbage
bits in the middle." Either one suggests a **sample-alignment or
sample-loss problem**, not a demod-quality problem.

## Setup

- Split-radio bridge: each node has one TX Pluto and one RX Pluto
  (per-node IPs in the `nodes` block of `pluto/setup.json`). Connected via
  coax, no shared antenna, so no self-interference.
- Bidirectional ARQ with selective-repeat (`modules/arq.py`), MTU 1500,
  window 7, retransmit-timeout 0.15 s, arq-queue-depth 64.
- CFO calibration is current and applied to RX LO only (see the `cfo`
  block of `pluto/setup.json`).
- Tests pass (`uv run pytest tests/`).

## Reproduction

Shell A:
```bash
sudo ./scripts/bridge_netns.sh up --no-shape --arq-queue-depth 64 \
    --retransmit-timeout 0.15
```

Shell B (TCP mpegts receiver):
```bash
sudo -E ip netns exec arq-b ffplay -fflags nobuffer -flags low_delay \
    'tcp://0.0.0.0:5000?listen=1'
```

Shell C (TCP mpegts sender):
```bash
sudo ip netns exec arq-a ffmpeg \
    -re -i /home/morten/downloads/badapple.mp4 \
    -vf scale=854:480 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -b:v 500k -maxrate 500k -bufsize 250k \
    -c:a aac -b:a 64k -ac 1 -ar 44100 \
    -f mpegts tcp://10.0.0.2:5000
```

Watch `bridge-A.log` for DECODE ERROR lines and the periodic ARQ stats:
```bash
tail -f bridge-A.log | grep -E 'DECODE|rtx=|tun_drop'
```

For the control (one_way_threaded), bring the bridge down first, then:

```bash
# Terminal 1
uv run python pluto/one_way_threaded.py --node A --mode tx --packets 1000 \
    --payload 1500 --interval 0 --variable
# Terminal 2
uv run python pluto/one_way_threaded.py --node B --mode rx
```

The user reports ~0% post-coarse-sync decode failures here.

## Key file map

- `pluto/bridge.py` — ARQ glue, `make_pluto_rx()` does the sliding-window
  concat of `prev_buf + curr_buf` and advances `search_from` between calls.
- `pluto/one_way_threaded.py` — does the **same** sliding-window logic in its
  RX loop (lines around `run_rx()` / the `rx_thread` for "both" mode).
- `pluto/sdr_stream.py` — `RxStream` is the threaded DMA drainer. Lossy by
  default (`lossless=False`) — drops oldest buffer when the consumer is
  behind. `overruns` counter tracks how often that fires.
- `modules/pipeline.py` — `RXPipeline.receive(buffer, search_from=...)`
  returns `(packets, max_det)`. Payload decoding lives in
  `payload_decode()` which is what raises `IndexError` / `RuntimeError`.
- `modules/arq.py` — where the errors get logged (the `arq-rx` thread wraps
  `pluto_rx()` calls). The traceback will point into the pipeline.

## Critical difference between bridge and one_way_threaded

Look at **RX buffer size**. This is the smoking-gun suspect.

`pluto/one_way_threaded.py`:
```python
rx_buf_size = 16 * int(2 ** np.ceil(np.log2(frame_len)))
```
That's **16×** the next power of two above a single frame. For an MTU-1500
frame (~16 k samples raw → pow2 = 32 768), RX buffer = ~524 k samples ≈
**131 ms** of audio. Each DMA buffer contains ~16 frames end-to-end.

`pluto/bridge.py`:
```python
rx_buf_size = args.rx_buf_mult * pow2   # default --rx-buf-mult 1
```
That's **1×** pow2 = ~32 k samples ≈ **8 ms**. Each DMA buffer contains
~1–2 frames, and packets frequently straddle the buffer seam.

**Hypothesis:** with the tiny RX buffer, the sliding-window concat +
`search_from` advancement has a corner case that corrupts or truncates a
straddling frame — especially when two frames arrive back-to-back (which
happens for ARQ DATA → reverse-direction ACK → DATA).

## What to investigate (in priority order)

### 1. Test the buffer-size hypothesis first — cheapest experiment

Re-run the bridge with `--rx-buf-mult 16` (match one_way_threaded) and see
if CRC mismatches drop to near zero:

```bash
# Hack the default in scripts/bridge_netns.sh or pass --rx-buf-mult 16
# through the bridge CLI. Easiest: add --rx-buf-mult 16 to the python -m
# pluto.bridge invocation inside cmd_up().
```

If this **fixes it**, the root cause is the sliding-window logic in
`make_pluto_rx` at `pluto/bridge.py:~165`. The next step is to trace
exactly which frames straddle the seam and fail. A useful instrumentation:

```python
# In make_pluto_rx, inside the `fn()` closure
logger.debug("rx: prev_len=%d curr_len=%d search_from=%d "
             "raw_len=%d max_det=%d packets=%d",
             prev_len, len(curr_buf), state["search_from"],
             len(raw), max_det, len(packets))
```

If this **doesn't fix it**, the problem is elsewhere — proceed to #2.

### 2. Check RxStream overruns

In the bridge stats line (printed every 5 s), look for `rx_overruns=N`.
If N is growing, the DMA is dropping whole buffers because the consumer
thread is behind. Any frame that straddled the dropped buffer is corrupt.

The bridge uses `RxStream(..., maxsize=args.rx_queue_depth, lossless=False)`
which silently overwrites old buffers on overrun. Try raising
`--rx-queue-depth 8` (from default 2) and see if overruns go to zero.

### 3. Look for a pipeline bug specific to the bridge's call pattern

The bridge calls `rx_pipe.receive(raw, search_from=...)` repeatedly with
`raw = prev_buf + curr_buf`. That's fine when `prev_buf` is None (first
call). But on every subsequent call, `raw[:prev_len]` is the *already-
searched* region from the previous iteration, and `search_from` should
point somewhere in `raw[prev_len:]` (the new samples).

Verify by printing `search_from - prev_len` — it should always be ≥ 0.
If it's ever negative, we're re-searching old samples, which could
re-trigger the same preamble and try to decode a payload that's now
offset incorrectly.

Also check: `pluto/bridge.py:~180`:
```python
if packets:
    last_ps = max(p.sample_start for p in packets)
    state["search_from"] = max(0, max(last_ps, max_det) - prev_len)
else:
    state["search_from"] = max(0, max_det - prev_len)
```
The `max(last_ps, max_det)` takes whichever is further — `last_ps` is the
start of a decoded packet, `max_det` is the last position the coarse-sync
scanned to. If `max_det < last_ps`, we advance past `last_ps` (the start
of the last packet we found). But that skips the full payload span! If
`last_ps + frame_len > prev_len + len(curr_buf)`, the next iteration
might restart mid-payload.

Cross-reference with the exact same logic in `one_way_threaded.py` —
it's identical, so if this is the bug both should fail. Unless the
difference is buffer size making it manifest only for small buffers.

### 4. CPU / timing

Run `htop` during the bridge and check CPU usage of the Python process.
If it's >90% on one core, the process is CPU-bound and the RX thread
can't keep up with DMA → overruns → buffer drops → corrupt frames.

`one_way_threaded --mode rx` only runs an RX pipeline; bridge runs
TX + RX + ARQ + TUN in one process. CPU budget is much tighter.

If CPU-bound: consider running the TX and RX in separate processes
talking over a socket, or profile `rx_pipe.receive()` and see if
there's a hot loop that can move to the C++ extension.

### 5. Hex-dump a specific failing frame

Add a debug dump when a CRC mismatch fires:

```python
# in modules/pipeline.py, wherever payload_decode raises the CRC mismatch
logger.debug("CRC mismatch — dump: header=%s payload[:32]=%s payload[-32:]=%s "
             "buf_range=[%d:%d] of %d",
             header, payload_bits[:32], payload_bits[-32:],
             payload_start, payload_end, len(buffer))
```

Look for: payload that's mostly zeros in the middle (DMA dropped samples),
or a payload whose length matches an earlier frame (misaligned start), or
a payload truncated at the buffer end (spans past `len(buffer)`).

## What NOT to go down rat-holes on

- **CFO drift.** Measured CFO in the error lines is single-digit Hz, way
  inside capture range. Not the problem.
- **SNR / physical layer.** Coax, ratio=10+ detections, no fading.
- **Selective-repeat bugs.** ARQ layer is above the pipeline; if the
  pipeline is returning garbage, ARQ just (correctly) sees invalid frames
  and drops them.

## Deliverable

Write up findings in a comment at the top of `pluto/bridge.py` (what the
root cause is) and open a minimal fix. If the fix is "match the RX buffer
size," make that the default in the bridge (and document why in a comment
near `rx_buf_size = ...`).
