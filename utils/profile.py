"""Lightweight per-stage timing profiler.

Use `Profiler` to record per-stage elapsed times via a context manager, then
periodically `dump()` p50/p95/p99/max in milliseconds. `NULL_PROFILER` is a
no-op singleton so call sites can stay unconditional when profiling is off.

    prof = Profiler() if args.profile else NULL_PROFILER
    with prof.stage("rx.receive"):
        rx_pipe.receive(...)
    if prof.should_dump():
        logger.info(prof.dump())
"""
from __future__ import annotations

import collections
import threading
import time


class _NullCtx:
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *_): return False


_NULL_CTX = _NullCtx()


class _NullProfiler:
    """No-op profiler. All methods are cheap and return harmless values."""
    enabled = False
    __slots__ = ()
    def stage(self, _name): return _NULL_CTX
    def record(self, _name, _dt_ms): pass
    def should_dump(self, _interval_s=2.0): return False
    def dump(self): return ""


NULL_PROFILER = _NullProfiler()


class _Stage:
    __slots__ = ("_p", "_name", "_t0")
    def __init__(self, p: "Profiler", name: str):
        self._p = p
        self._name = name
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        self._p.record(self._name, (time.perf_counter() - self._t0) * 1e3)
        return False


class Profiler:
    """Rolling-window timing recorder. Thread-safe.

    Per-stage history is bounded by `window` samples. `dump()` reports
    count, p50, p95, p99 and max for every stage seen since construction
    (or since the last `reset()`).
    """
    enabled = True

    def __init__(self, window: int = 4096):
        self._stages: dict[str, collections.deque[float]] = {}
        self._window = window
        self._lock = threading.Lock()
        self._last_dump_t = time.monotonic()

    def stage(self, name: str) -> _Stage:
        return _Stage(self, name)

    def record(self, name: str, dt_ms: float) -> None:
        with self._lock:
            dq = self._stages.get(name)
            if dq is None:
                dq = collections.deque(maxlen=self._window)
                self._stages[name] = dq
            dq.append(dt_ms)

    def should_dump(self, interval_s: float = 2.0) -> bool:
        now = time.monotonic()
        if now - self._last_dump_t >= interval_s:
            self._last_dump_t = now
            return True
        return False

    def dump(self) -> str:
        with self._lock:
            snapshot = {k: list(v) for k, v in self._stages.items() if v}
        if not snapshot:
            return "[profile] (no samples)"
        name_w = max(len(n) for n in snapshot)
        lines = ["[profile] stage timings (ms)  — recent window per stage"]
        lines.append(f"  {'stage':<{name_w}}  {'n':>5}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}")
        for name in sorted(snapshot):
            samples = sorted(snapshot[name])
            n = len(samples)
            def q(p):
                idx = min(n - 1, max(0, int(round(p * (n - 1)))))
                return samples[idx]
            lines.append(
                f"  {name:<{name_w}}  {n:>5d}  "
                f"{q(0.50):>7.3f}  {q(0.95):>7.3f}  {q(0.99):>7.3f}  {samples[-1]:>7.3f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
