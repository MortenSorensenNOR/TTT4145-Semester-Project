# Localized Plateau Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Make `coarse_sync` robust to multi-frame buffers by localizing plateau analysis and renaming `rx` to `samples`.

**Architecture:** Replace the global `plateau_mask` in `coarse_sync` with a local window of +/- one preamble span around the detected peak. This isolates the CFO estimate to a single frame regardless of buffer contents. Rename the `rx` parameter to `samples` in both `coarse_sync` and `fine_timing` for source-neutrality.

**Tech Stack:** Python 3.13, NumPy, pytest

---

### Task 1: Rename `rx` to `samples` in `frame_sync.py`

**Files:**
- Modify: `modules/frame_sync.py:92-97` (coarse_sync signature)
- Modify: `modules/frame_sync.py:120-131` (coarse_sync body — all `rx` refs)
- Modify: `modules/frame_sync.py:155-162` (fine_timing signature)
- Modify: `modules/frame_sync.py:164-182` (fine_timing body — all `rx` refs)

**Step 1: Rename in `coarse_sync`**

In `coarse_sync`, replace parameter `rx` with `samples` in the signature and all references in the function body:
- Line 93: `rx: np.ndarray` → `samples: np.ndarray`
- Line 120: `np.iscomplexobj(rx)` → `np.iscomplexobj(samples)`
- Line 121: error message `"rx must be complex..."` → `"samples must be complex..."`
- Line 129: `len(rx)` → `len(samples)`
- Line 130: error message with `rx` → `samples`
- Line 133: `rx[:-sample_cnt]` and `rx[sample_cnt:]` → `samples[...]`
- Line 136: `rx[sample_cnt:]` → `samples[sample_cnt:]`

**Step 2: Rename in `fine_timing`**

In `fine_timing`, replace parameter `rx` with `samples`:
- Line 156: `rx: np.ndarray` → `samples: np.ndarray`
- Line 164: `np.iscomplexobj(rx)` → `np.iscomplexobj(samples)`
- Line 165: error message → `"samples must be complex..."`
- Line 173: `len(rx)` → `len(samples)`
- Line 182: `rx[search_start:search_end]` → `samples[search_start:search_end]`

**Step 3: Run existing tests to verify no regressions**

Run: `uv run pytest tests/test_frame_sync_pipeline.py -v`
Expected: 6 passed, 15 xfailed (identical to baseline)

**Step 4: Commit**

```bash
git add modules/frame_sync.py
git commit -m "refactor: rename rx to samples in coarse_sync and fine_timing"
```

---

### Task 2: Localize plateau analysis in `coarse_sync`

**Files:**
- Modify: `modules/frame_sync.py:141-146`

**Step 1: Write the failing test**

Add to `tests/test_frame_sync_pipeline.py`:

```python
def test_multi_frame_plateau_isolation() -> None:
    """CFO estimate must come from the first frame only, not a mix of both.

    Places two identical frames (with different CFOs) in a single buffer.
    The old global plateau_mask would average both CFOs.
    The localized version must return only the first frame's CFO.
    """
    num_taps = 2 * SPS * SPAN + 1
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, num_taps)

    cfo_frame1 = 5_000   # 5 kHz
    cfo_frame2 = 15_000  # 15 kHz — deliberately different

    rng = np.random.default_rng(42)
    preamble = generate_preamble(SYNC_CFG)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    frame_syms = np.concatenate([preamble, payload])

    # Build two frames with different CFOs
    tx1 = upsample(frame_syms, SPS, rrc_taps)
    tx2 = upsample(frame_syms, SPS, rrc_taps)

    tx1 *= np.exp(2j * np.pi * cfo_frame1 / SAMPLE_RATE * np.arange(len(tx1)))

    gap = np.zeros(500, dtype=complex)  # guard interval
    offset2 = SAMPLE_OFFSET + len(tx1) + len(gap)
    tx2_phase_start = offset2
    tx2 *= np.exp(2j * np.pi * cfo_frame2 / SAMPLE_RATE * (tx2_phase_start + np.arange(len(tx2))))

    buffer = np.concatenate([
        np.zeros(SAMPLE_OFFSET, dtype=complex),
        tx1,
        gap,
        tx2,
    ])

    coarse = coarse_sync(buffer, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse.m_peak >= MIN_DETECTION_CONFIDENCE
    cfo_err = abs(float(coarse.cfo_hat) - cfo_frame1)
    assert cfo_err < MAX_CFO_ERROR_HZ, (
        f"CFO error {cfo_err:.0f} Hz — plateau likely mixed both frames"
    )
```

Run: `uv run pytest tests/test_frame_sync_pipeline.py::test_multi_frame_plateau_isolation -v`
Expected: FAIL (global plateau averages both CFOs, error > 200 Hz)

**Step 2: Implement localized plateau**

In `modules/frame_sync.py`, replace lines 141-146:

```python
    # OLD (global):
    # peak_idx = np.argmax(m_d)
    # plateau_mask = m_d >= cfg.plateau_edge_fraction * m_d[peak_idx]
    # d_hat = np.argmax(plateau_mask)
    # phi_hat = np.angle(np.mean(p_d[plateau_mask]))
    # cfo_hat = phi_hat * fs / (2 * np.pi * sample_cnt)

    # NEW (localized around peak):
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
    cfo_hat = phi_hat * fs / (2 * np.pi * sample_cnt)
```

**Step 3: Run the new test**

Run: `uv run pytest tests/test_frame_sync_pipeline.py::test_multi_frame_plateau_isolation -v`
Expected: PASS

**Step 4: Run ALL tests to verify no regressions**

Run: `uv run pytest tests/test_frame_sync_pipeline.py -v`
Expected: 7 passed, 15 xfailed

**Step 5: Commit**

```bash
git add modules/frame_sync.py tests/test_frame_sync_pipeline.py
git commit -m "feat: localize plateau analysis in coarse_sync for multi-frame buffers"
```

---

### Task 3: Add iterate-and-advance compatibility test

**Files:**
- Modify: `tests/test_frame_sync_pipeline.py`

**Step 1: Write the test**

```python
def test_iterate_and_advance() -> None:
    """After consuming frame 1, coarse_sync on the remainder finds frame 2.

    Simulates the 802.11 iterate-and-advance receive pattern:
    detect frame 1 → consume → detect frame 2 on buffer[offset:].
    """
    num_taps = 2 * SPS * SPAN + 1
    group_delay = (num_taps - 1) // 2
    rrc_taps = rrc_filter(SPS, RRC_ALPHA, num_taps)
    long_ref = build_long_ref(SYNC_CFG, SPS, rrc_taps)

    cfo_hz = 10_000

    rng = np.random.default_rng(99)
    preamble = generate_preamble(SYNC_CFG)
    payload1 = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    payload2 = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)

    frame1 = upsample(np.concatenate([preamble, payload1]), SPS, rrc_taps)
    frame2 = upsample(np.concatenate([preamble, payload2]), SPS, rrc_taps)

    gap = np.zeros(500, dtype=complex)
    buffer = np.concatenate([
        np.zeros(SAMPLE_OFFSET, dtype=complex),
        frame1,
        gap,
        frame2,
        np.zeros(SAMPLE_OFFSET, dtype=complex),
    ])
    buffer *= np.exp(2j * np.pi * cfo_hz / SAMPLE_RATE * np.arange(len(buffer)))

    # --- Detect frame 1 ---
    coarse1 = coarse_sync(buffer, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse1.m_peak >= MIN_DETECTION_CONFIDENCE
    ft1 = fine_timing(buffer, long_ref, coarse1, SAMPLE_RATE, SPS, SYNC_CFG)

    # Consume: advance past frame 1 (preamble + payload)
    preamble_samples = len(upsample(preamble, SPS, rrc_taps))
    payload_samples = len(upsample(payload1, SPS, rrc_taps))
    consumed = int(ft1) + len(long_ref) + (N_PAYLOAD_SYMBOLS * SPS)

    # --- Detect frame 2 on remainder ---
    remainder = buffer[consumed:]
    coarse2 = coarse_sync(remainder, SAMPLE_RATE, SPS, SYNC_CFG)
    assert coarse2.m_peak >= MIN_DETECTION_CONFIDENCE, (
        f"frame 2 not detected after consume (m_peak={coarse2.m_peak:.3f})"
    )

    cfo_err2 = abs(float(coarse2.cfo_hat) - cfo_hz)
    assert cfo_err2 < MAX_CFO_ERROR_HZ, f"frame 2 CFO error {cfo_err2:.0f} Hz"
```

**Step 2: Run the test**

Run: `uv run pytest tests/test_frame_sync_pipeline.py::test_iterate_and_advance -v`
Expected: PASS (localized plateau from Task 2 already handles this)

**Step 3: Run full suite**

Run: `uv run pytest tests/test_frame_sync_pipeline.py -v`
Expected: 8 passed, 15 xfailed

**Step 4: Commit**

```bash
git add tests/test_frame_sync_pipeline.py
git commit -m "test: add iterate-and-advance multi-frame test for coarse_sync"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (no regressions in other test files)

**Step 2: Verify git status is clean**

Run: `git status`
Expected: clean working tree, 3 new commits on `simplification_testing`
