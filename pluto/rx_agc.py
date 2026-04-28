"""Software AGC for manual-mode PlutoSDR RX gain.

The AD9361 ships with three hardware AGC modes (slow/fast attack, hybrid),
but the user has observed those drift the gain during silence between
sparse bursts so the next packet then clips. The split-radio one-way test
therefore runs with ``--rx-gain-mode=manual``, which holds whatever fixed
gain we set — but a fixed gain only matches one operating point.

:class:`RxAGC` closes the loop in software: it inspects the peak amplitude
of each RX buffer that produced a valid packet decode and nudges
``sdr.rx_hardwaregain_chan0`` toward a target peak. Buffers that produced
no valid decode (silence, noise spike, header CRC fail) are ignored, so
we never ramp gain up while waiting for traffic and clip the next packet.

Sample-scale reminder: :class:`pluto.sdr_stream.RxStream` returns samples
as int16 * 2 / DAC_SCALE = int16 / 8192. The AD9361 12-bit ADC fills
int16 (±32767), so |x| ≈ 4.0 in those normalised units corresponds to
ADC clip. We aim for a target peak well below that.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RxAGCConfig:
    """Tunable thresholds for :class:`RxAGC`.

    All amplitude values are in the same normalised units as the samples
    delivered by :class:`pluto.sdr_stream.RxStream` (int16 * 2/DAC_SCALE,
    so ADC clip ≈ 4.0).
    """

    target_peak: float = 2.0          # ~ −6 dBFS, comfortable midpoint
    clip_threshold: float = 3.5       # peak above this counts as clipping
    low_threshold: float = 0.5        # peak below this means gain is too low
    max_step_db: float = 3.0          # cap on a single adjustment
    min_step_db: float = 0.5          # below this, don't bother changing
    min_gain_db: float = 0.0          # AD9361 lower rail
    max_gain_db: float = 71.0         # AD9361 upper rail
    cooldown_bufs: int = 4            # buffers to skip between changes


class RxAGC:
    """Software AGC operating on a manual-gain Pluto RX.

    The caller invokes :meth:`update` once per RX buffer, passing the
    samples and how many valid packets came out of the buffer. The AGC
    adjusts ``sdr.rx_hardwaregain_chan0`` in place.
    """

    def __init__(self, sdr, *, initial_gain_db: float,
                 cfg: RxAGCConfig | None = None):
        self._sdr = sdr
        self._cfg = cfg or RxAGCConfig()
        self._gain = float(initial_gain_db)
        self._cooldown = 0
        self._adjustments = 0

    @property
    def gain_db(self) -> float:
        return self._gain

    @property
    def adjustments(self) -> int:
        return self._adjustments

    def update(self, buf: np.ndarray, n_valid: int) -> tuple[float, float] | None:
        """Inspect one RX buffer; maybe adjust the hardware gain.

        Args:
            buf:     complex64 RX samples (already normalised by RxStream).
            n_valid: number of validly-decoded packets that came out of
                     this buffer.

        Returns:
            (peak, new_gain_db) when the gain was changed, else None.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        # Per the user's requirement: only act on buffers where the
        # receiver actually decoded a packet. This avoids ramping gain
        # up during silence (which would clip on the next real packet)
        # and avoids reacting to a one-off noise spike.
        if n_valid <= 0:
            return None

        peak = float(np.max(np.abs(buf))) if buf.size else 0.0
        if peak <= 0.0:
            return None

        cfg = self._cfg
        clipping = peak >= cfg.clip_threshold
        too_low = peak <= cfg.low_threshold
        if not (clipping or too_low):
            return None

        # Linear gain change in dB to bring this buffer's peak to target.
        delta_db = 20.0 * math.log10(cfg.target_peak / peak)
        if delta_db > cfg.max_step_db:
            delta_db = cfg.max_step_db
        elif delta_db < -cfg.max_step_db:
            delta_db = -cfg.max_step_db

        new_gain = max(cfg.min_gain_db, min(cfg.max_gain_db, self._gain + delta_db))
        if abs(new_gain - self._gain) < cfg.min_step_db:
            return None

        try:
            self._sdr.rx_hardwaregain_chan0 = float(new_gain)
        except Exception as e:
            logger.warning("[RX-AGC] failed to set gain to %.1f dB: %s",
                           new_gain, e)
            return None

        reason = "clip" if clipping else "low"
        logger.info("[RX-AGC] %s peak=%.2f  gain %.1f dB -> %.1f dB",
                    reason, peak, self._gain, new_gain)
        self._gain = new_gain
        self._cooldown = cfg.cooldown_bufs
        self._adjustments += 1
        return peak, new_gain
