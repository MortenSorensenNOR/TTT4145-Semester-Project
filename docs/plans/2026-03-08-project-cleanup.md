# Project Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Make the codebase more concise and intuitive by splitting a 649-line file, removing duplication, and relocating example-only code.

**Architecture:** Keep `channel_coding.py` as a stable public API facade that re-exports from `golay.py` and `ldpc.py`. Move shared soft-demod logic into the base class. Remove dead code and relocate files that don't belong in the module tree.

**Tech Stack:** Python 3.13, numpy, scipy, uv, ruff

---

### Task 1: Split channel_coding.py into golay.py + ldpc.py + thin facade

**Files:**
- Create: `modules/golay.py`
- Create: `modules/ldpc.py`
- Rewrite: `modules/channel_coding.py`

**Step 1: Create `modules/golay.py`**

Extract the `Golay` class (lines 56-155 of current channel_coding.py) into its own file. Only dependency is numpy.

**Step 2: Create `modules/ldpc.py`**

Extract everything LDPC-related:
- `_njit` fallback (lines 6-14)
- All IEEE 802.11 base matrices (`_N648_R12` through `_N1944_R56`, lines 156-318)
- `get_ldpc_base_matrix()` (lines 321-349)
- `LDPCConfig` dataclass (lines 352-382)
- Module-level caches (`_h_cache`, `_encoding_cache`, `_decode_cache`)
- All LDPC functions: `ldpc_get_supported_payload_lengths`, `ldpc_get_h_matrix`, `ldpc_clear_cache`, `ldpc_encode`, `ldpc_decode`
- All internal helpers: `_check_node_update`, `_find_pivot`, `_coding_matrix_systematic`, `_get_encoding_structures`, `_get_decode_structures`, `_expand_h`
- Dependencies: numpy, scipy.sparse, dataclasses, enum (for CodeRates import)

**Step 3: Rewrite `modules/channel_coding.py` as thin facade**

```python
"""Channel coding: public API re-exporting Golay and LDPC implementations."""

from modules.golay import Golay
from modules.ldpc import (
    CodeRates,
    LDPCConfig,
    deinterleave,
    interleave,
    ldpc_clear_cache,
    ldpc_decode,
    ldpc_encode,
    ldpc_get_h_matrix,
    ldpc_get_supported_payload_lengths,
)

__all__ = [
    "CodeRates",
    "Golay",
    "LDPCConfig",
    "deinterleave",
    "interleave",
    "ldpc_clear_cache",
    "ldpc_decode",
    "ldpc_encode",
    "ldpc_get_h_matrix",
    "ldpc_get_supported_payload_lengths",
]
```

Note: `CodeRates` and `interleave`/`deinterleave` move to `ldpc.py` since they're only used by the LDPC pipeline. If a future coding scheme needs them, they can be promoted to a shared module then (YAGNI).

**Step 4: Run tests**

```bash
uv run pytest test/ -x -q
uv run ruff check --select ALL
```

Expected: 266 tests pass, zero ruff issues.

---

### Task 2: Extract shared soft-demod into _ModulatorBase

**Files:**
- Modify: `modules/modulation.py`

**Step 1: Add default `symbols2bits_soft` to `_ModulatorBase`**

Move the max-log-MAP implementation (currently duplicated in EightPSK lines 220-253 and QAM lines 335-371) into `_ModulatorBase`. This becomes the generic fallback that works for any constellation.

```python
class _ModulatorBase:
    """Shared base for all modulators."""

    symbol_mapping: np.ndarray
    bit_mapping: np.ndarray
    bits_per_symbol: int

    def estimate_noise_variance(self, symbols: np.ndarray) -> float:
        """Estimate noise variance from received symbols."""
        return estimate_noise_variance(symbols, self.symbol_mapping)

    def symbols2bits_soft(
        self,
        symbols: np.ndarray,
        sigma_sq: float | None = None,
    ) -> np.ndarray:
        """Compute LLRs using max-log-MAP approximation.

        For each bit position, LLR = min distance to constellation point with bit=1
        minus min distance to constellation point with bit=0, scaled by 1/sigma_sq.
        """
        if len(symbols) == 0:
            return np.array([], dtype=float)

        if sigma_sq is None:
            sigma_sq = estimate_noise_variance(symbols, self.symbol_mapping)

        distances_sq = np.abs(symbols.reshape(-1, 1) - self.symbol_mapping[np.newaxis, :]) ** 2

        llrs = np.zeros((len(symbols), self.bits_per_symbol))
        for bit_idx in range(self.bits_per_symbol):
            bit_is_zero = self.bit_mapping[:, bit_idx] == 0
            bit_is_one = ~bit_is_zero
            min_dist_zero = np.min(distances_sq[:, bit_is_zero], axis=1)
            min_dist_one = np.min(distances_sq[:, bit_is_one], axis=1)
            llrs[:, bit_idx] = (min_dist_one - min_dist_zero) / sigma_sq

        return llrs
```

**Step 2: Delete `symbols2bits_soft` from EightPSK and QAM**

Both classes inherit the base implementation. BPSK and QPSK keep their closed-form overrides (they're genuinely different algorithms, not duplication).

**Step 3: Run tests**

```bash
uv run pytest test/ -x -q
uv run ruff check --select ALL
```

---

### Task 3: Remove costas_loop.py __main__ block

**Files:**
- Modify: `modules/costas_loop.py`

**Step 1: Delete lines 164-216**

Remove the entire `if __name__ == "__main__":` block. It contains:
- A test harness with hardcoded path (`examples/data/phase.png`)
- Empty `pass` statements (dead convergence check)
- Redundant `import numpy as np` (already imported at module level)

The Costas loop is thoroughly tested in `test/test_synchronization.py`.

**Step 2: Verify unused imports can be removed**

After removing `__main__`, the `time` and `logging` imports at the top may become unused. Check and remove if so.

**Step 3: Run tests**

```bash
uv run pytest test/ -x -q
uv run ruff check --select ALL
```

---

### Task 4: Move plotting.py to examples/ and visualize.py to pluto/scripts/

**Files:**
- Move: `modules/plotting.py` → `examples/plotting.py`
- Move: `pluto/visualize.py` → `pluto/scripts/visualize.py`
- Modify: `examples/soft_decision_demo.py`

**Step 1: Move files**

```bash
mv modules/plotting.py examples/plotting.py
mkdir -p pluto/scripts
mv pluto/visualize.py pluto/scripts/visualize.py
```

**Step 2: Update import in `examples/soft_decision_demo.py`**

Change `from modules.plotting import plot_llr_heatmap` to a relative import or sys.path adjustment appropriate for a script in examples/.

**Step 3: Run tests**

```bash
uv run pytest test/ -x -q
uv run ruff check --select ALL
```

---

### Task 5: Final verification and commit

**Step 1: Full test suite**

```bash
uv run pytest test/ -x -q
```

Expected: 266 tests pass.

**Step 2: Linting**

```bash
uv run ruff check --select ALL
uv run ruff format --check
```

Expected: All checks passed.

**Step 3: Commit and push**

```bash
git add -A
git commit -m "refactor: split channel_coding, deduplicate soft-demod, remove dead code, relocate example files"
git push
```
