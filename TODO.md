# PlutoSDR ARM Port TODO

## Current Architecture

```
┌──────────────── HOST ────────────────┐
│ TUN/TAP ↔ TX/RX DSP ↔ PlutoSDR USB   │
└──────────────────────────────────────┘
```

## Target Architecture

```
┌─── HOST ───┐         ┌────── PLUTO ARM ──────┐
│ TUN/TAP    │ ←TCP→   │ TX/RX DSP ↔ IIO Radio │
│ Relay      │         │                       │
└────────────┘         └───────────────────────┘
```

## What's Portable to PlutoSDR

**All of `/modules/`** - pure Python/NumPy, no hardware dependencies:
- `modulation.py`, `frame_constructor.py`, `channel_coding.py`
- `synchronization.py`, `pulse_shaping.py`, `pilots.py`, `equalization.py`
- `transmit.py`, `receive.py` pipelines

**Key challenge:** `pyldpc` for LDPC decoding - need this compiled for ARM or use an alternative.

## What Needs to be Created

### 1. Host-side relay (`host/relay.py`)
- [ ] TUN/TAP device creation (reuse ioctl logic from `bridge.py:15-40`)
- [ ] Socket server to communicate with Pluto
- [ ] Forward packets: TUN → socket → Pluto
- [ ] Receive decoded packets: Pluto → socket → TUN

### 2. Pluto-side main (`pluto_main.py`)
- [ ] Socket client connecting to host
- [ ] Replace `pyadi-iio` with native IIO access (PlutoSDR has `/dev/iio:device*`)
- [ ] TX thread: socket → DSP pipeline → IIO TX
- [ ] RX thread: IIO RX → DSP pipeline → socket

### 3. Dependencies on Pluto

Already available (patched firmware):
- [x] Python
- [x] NumPy
- [x] pysdr
- [x] Pluto Python libraries

Still needed:
- [ ] `scipy` for FFT/convolution (or reimplement with NumPy)
- [ ] `pyldpc` compiled for ARM (or alternative LDPC library)
- [x] Native IIO bindings (libiio Python bindings or direct sysfs access)

## Files to Modify/Create

| File | Action |
|------|--------|
| `pluto/bridge.py` | Split into host relay + pluto main |
| `pluto/__init__.py` | Replace pyadi-iio with native IIO |
| `pluto/config.py` | Add network config (host IP, ports) |
| New: `host/relay.py` | TUN/TAP + socket relay on host |
| New: `pluto/main.py` | Entry point for Pluto ARM |

## Performance Concerns

From timing analysis:
- LDPC decoding takes **100-200ms per frame** on x86
- On ARM @ 600MHz this could be **significantly slower**
- Consider: lower code rates, smaller payloads, or FPGA offload for LDPC

## Implementation Order

1. [ ] Create `host/relay.py` with TUN/TAP + socket server
2. [ ] Create `pluto/iio_native.py` to replace pyadi-iio
3. [ ] Create `pluto/main.py` entry point for ARM
4. [ ] Test socket communication between host and Pluto
5. [ ] Port modules to Pluto and test DSP pipeline
6. [ ] Integrate full TX/RX with IIO streaming
7. [ ] Performance testing and optimization
