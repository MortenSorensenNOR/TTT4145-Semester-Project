# Localized Plateau Analysis in coarse_sync

## Problem

`coarse_sync` computes the Schmidl-Cox timing metric `m_d` over the entire input
buffer, then uses a global `argmax` and relative threshold to build `plateau_mask`.
When the buffer contains multiple frames (as in a stream-buffer receive loop),
the mask can span multiple preambles, mixing their CFO contributions.

Industry implementations (GNU Radio `ofdm_sync_sc_cfb`, MATLAB WLAN Toolbox,
gr-ieee802-11 `sync_short`) avoid this by processing one frame at a time via
stream-based triggers or iterate-and-advance with minimum skip.

## Solution

Localize the plateau and CFO computation to a window of +/- one preamble span
around the detected peak. This makes `coarse_sync` correct regardless of how
many frames the caller's buffer contains.

### Changes to `frame_sync.py`

1. **Rename `rx` to `samples`** in `coarse_sync` and `fine_timing` signatures
   and bodies. `samples` is source-neutral (works for raw SDR output or
   stream-buffer slices).

2. **Localize plateau analysis** in `coarse_sync` (replaces lines 141-146):

   ```python
   peak_idx = np.argmax(m_d)
   preamble_span = cfg.short_preamble_nsym * cfg.short_preamble_nreps * samples_per_symbol
   local_start = max(0, peak_idx - preamble_span)
   local_end = min(len(m_d), peak_idx + preamble_span)

   local_mask = np.zeros_like(m_d, dtype=bool)
   local_mask[local_start:local_end] = (
       m_d[local_start:local_end] >= cfg.plateau_edge_fraction * m_d[peak_idx]
   )
   d_hat = np.argmax(local_mask)
   phi_hat = np.angle(np.mean(p_d[local_mask]))
   ```

### Changes to `test_frame_sync_pipeline.py`

1. **Multi-frame plateau isolation** -- buffer with two frames at different
   offsets; verify CFO estimate matches the first frame only.

2. **Iterate-and-advance compatibility** -- after removing frame 1 samples
   from buffer, verify `coarse_sync` on the remainder finds frame 2 correctly.

3. Existing single-frame tests remain unchanged.

## Context for future work

This change prepares `frame_sync.py` for a stream-buffer receive loop
(802.11 iterate-and-advance pattern) in `pipeline.py`. That loop, a
`StreamBuffer` class, TX guard intervals, and minimum-skip logic are
out of scope here and tracked separately.

## References

- Schmidl & Cox, "Robust Frequency and Timing Synchronization for OFDM",
  IEEE Trans. Comm., Vol. 45, No. 12, 1997.
- GNU Radio `ofdm_sync_sc_cfb` -- stream-based local trigger.
- gr-ieee802-11 `sync_short.cc` -- `MIN_GAP = 480` minimum skip.
- MATLAB WLAN Toolbox -- `searchOffset += lstfLen` iterate-and-advance.
