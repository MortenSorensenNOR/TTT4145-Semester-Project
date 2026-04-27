"""Pinned-status terminal UI shared by the threaded radio scripts.

A multi-line status block stays pinned at the bottom of the terminal while
log records and one-shot prints scroll above it.  Falls back to plain prints
when stdout is not a TTY.
"""

import collections
import logging
import shutil
import sys
import threading
import time


class RateMeter:
    """Sliding-window throughput meter (bytes / sec)."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._events: collections.deque = collections.deque()
        self._t_start = time.perf_counter()
        self.total_bytes = 0

    def add(self, n_bytes: int) -> None:
        now = time.perf_counter()
        self._events.append((now, n_bytes))
        self.total_bytes += n_bytes
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    @property
    def rate_bps(self) -> float:
        if not self._events:
            return 0.0
        b   = sum(n for _, n in self._events)
        win = max(time.perf_counter() - self._events[0][0], 1e-3)
        return b / win

    @property
    def avg_bps(self) -> float:
        elapsed = time.perf_counter() - self._t_start
        return self.total_bytes / elapsed if elapsed > 0 else 0.0


def _fmt_rate(bps: float) -> str:
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if bps < 1024 or unit == "GB/s":
            return f"{bps:7.1f} {unit}"
        bps /= 1024
    return f"{bps:7.1f} GB/s"


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} GB"


class LiveStatus:
    """Multi-line status pinned at the bottom; logs scroll above.

    On a TTY, uses ANSI cursor-up + clear-line so :meth:`log` can print
    above the status without clobbering it.  On a non-TTY (pipe / file),
    :meth:`set` is silent and :meth:`log` is a plain ``print``.
    """

    def __init__(self, n_lines: int = 1, stream=None) -> None:
        self.n_lines = n_lines
        self.lines = [""] * n_lines
        self.stream = stream if stream is not None else sys.stdout
        self.is_tty = self.stream.isatty()
        self._rendered = False
        self._lock = threading.Lock()

    def set(self, idx: int, text: str) -> None:
        with self._lock:
            self.lines[idx] = text
            if self.is_tty:
                self._refresh_locked()

    def log(self, msg: str) -> None:
        with self._lock:
            if self.is_tty and self._rendered:
                self._clear_locked()
            self.stream.write(str(msg) + "\n")
            self.stream.flush()
            if self.is_tty:
                self._refresh_locked()

    def stop(self) -> None:
        """Release the terminal — leaves the last status visible above the cursor."""
        with self._lock:
            if self.is_tty and self._rendered:
                self.stream.write("\n")
                self.stream.flush()
            self._rendered = False

    def _clear_locked(self) -> None:
        for _ in range(self.n_lines):
            self.stream.write("\033[F\033[K")
        self._rendered = False

    def _refresh_locked(self) -> None:
        if self._rendered:
            self._clear_locked()
        cols = shutil.get_terminal_size((120, 24)).columns
        for line in self.lines:
            line_trim = line if len(line) <= cols else line[: cols - 1] + "…"
            self.stream.write(line_trim + "\n")
        self.stream.flush()
        self._rendered = True


class _StatusLogHandler(logging.Handler):
    """Routes log records through LiveStatus.log so they scroll above the
    pinned status without disturbing it."""
    def __init__(self, status: LiveStatus) -> None:
        super().__init__()
        self.status = status

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.status.log(self.format(record))
        except Exception:  # never let a logging error kill the run
            self.handleError(record)


def _install_live_logging(status: LiveStatus, level=logging.INFO) -> None:
    """Replace any default basicConfig handlers with one that writes through `status`."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = _StatusLogHandler(status)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
