"""Compute EVM (Error Vector Magnitude) and derived SNR from a saved
constellation of post-Costas PSK8 symbols.

  EVM_rms = sqrt(mean(|rx_norm - nearest_ideal|^2)) / sqrt(mean(|ideal|^2))
  SNR_dB  = -20 * log10(EVM_rms)

The receiver doesn't fix amplitude, so rx is rescaled to match the unit-norm
PSK8 reference before snapping. Constellation orientation falls out of the
nearest-ideal snap (Costas leaves an 8-fold rotation ambiguity).
"""
from __future__ import annotations

import argparse
import numpy as np

from modules.pipeline import RXPipeline, PipelineConfig


def load_symbols(path: str) -> np.ndarray:
    """Load symbols from .npy. Accepts either:
      - flat complex array of symbols
      - real (N, 3) array with columns [seq, I, Q] (matches CSV dump format)
      - real (N, 2) array with columns [I, Q]
    """
    obj = np.load(path, allow_pickle=True)
    if np.iscomplexobj(obj):
        return obj.astype(np.complex128).ravel()
    if obj.ndim == 2 and obj.shape[1] == 3:
        return (obj[:, 1] + 1j * obj[:, 2]).astype(np.complex128)
    if obj.ndim == 2 and obj.shape[1] == 2:
        return (obj[:, 0] + 1j * obj[:, 1]).astype(np.complex128)
    raise ValueError(f"unsupported constellation layout: dtype={obj.dtype}, shape={obj.shape}")


def evm_psk8(rx: np.ndarray, ideal: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # Match rx RMS amplitude to the ideal RMS amplitude (=1 for unit PSK).
    rx_rms = np.sqrt(np.mean(np.abs(rx) ** 2))
    ref_rms = np.sqrt(np.mean(np.abs(ideal) ** 2))
    rx_n = rx * (ref_rms / rx_rms)

    # Snap each rx symbol to the closest ideal point.
    d = np.abs(rx_n[:, None] - ideal[None, :])
    nearest = ideal[np.argmin(d, axis=1)]

    err = rx_n - nearest
    evm = np.sqrt(np.mean(np.abs(err) ** 2)) / ref_rms
    return float(evm), rx_n, nearest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/constellation.npy")
    args = ap.parse_args()

    rx = load_symbols(args.path)
    print(f"loaded {rx.size} symbols from {args.path}")
    print(f"  rx |z| mean={np.mean(np.abs(rx)):.4f}  rms={np.sqrt(np.mean(np.abs(rx)**2)):.4f}")

    ideal = np.asarray(RXPipeline(PipelineConfig()).psk8.symbol_mapping, dtype=np.complex128)
    print(f"  ideal points: {ideal.size} (unit magnitude: {np.allclose(np.abs(ideal), 1.0)})")

    evm, rx_n, nearest = evm_psk8(rx, ideal)
    snr_lin = 1.0 / (evm ** 2)
    snr_db = 10.0 * np.log10(snr_lin)

    err = rx_n - nearest
    err_per_sym = np.abs(err)

    print()
    print(f"EVM (RMS)        : {evm:.6f}  ({evm * 100:.3f} %)")
    print(f"EVM (peak)       : {err_per_sym.max():.6f}")
    print(f"SNR (from EVM)   : {snr_db:.3f} dB   (linear {snr_lin:.2f})")
    print()
    print("symbol-error magnitude percentiles:")
    for q in (50, 90, 99):
        print(f"  p{q:>2d}: {np.percentile(err_per_sym, q):.4f}")


if __name__ == "__main__":
    main()
