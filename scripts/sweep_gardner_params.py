"""Sweep Gardner-TED parameters and rank by post-Costas constellation tightness.

Operates on RX buffers captured from a real Pluto run by
``pluto/one_way_threaded.py --save-rx-buf <dir>``.  For every (BnTs, ζ, L)
combination on the sweep grid, the script re-runs ``RXPipeline.receive()``
on each saved buffer and computes the **EVM** of the post-Costas symbols
(RMS distance to the nearest ideal constellation point, % of unit magnitude).
Lower EVM = tighter point cloud around each ideal point.

Capture buffers first:
    # node B (or wherever)
    uv run python pluto/one_way_threaded.py --mode tx
    # node A
    uv run python pluto/one_way_threaded.py --mode rx --save-rx-buf rx_dump --save-n 4

Then sweep:
    uv run python scripts/sweep_gardner_params.py rx_dump
    uv run python scripts/sweep_gardner_params.py rx_dump --plot
    uv run python scripts/sweep_gardner_params.py rx_dump \
        --bnts 0.0001,0.0005,0.001,0.002 --zetas 0.5,0.707,1.0 --ls 1,2,3,4

The same input buffers are used across all parameter combinations, so the
only thing changing between runs is the Gardner config — making the ranking
directly meaningful.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.modulators.modulators import BPSK, QPSK, PSK8, PSK16
from modules.pipeline import PipelineConfig, RXPipeline


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


_MOD_INSTANCE = {
    ModulationSchemes.BPSK: BPSK(),
    ModulationSchemes.QPSK: QPSK(),
    ModulationSchemes.PSK8: PSK8(),
    ModulationSchemes.PSK16: PSK16(),
}


def _mod_for(scheme):
    if isinstance(scheme, str):
        scheme = ModulationSchemes[scheme]
    if isinstance(scheme, int):
        scheme = ModulationSchemes(scheme)
    return _MOD_INSTANCE[scheme]


def _squared_errors(rx_symbols: np.ndarray, mod) -> np.ndarray:
    """Per-symbol squared distance from each rx_symbol to the nearest ideal
    constellation point.  All PSK constellations live on the unit circle, so
    the result is already normalised — sqrt(mean(...)) gives EVM directly."""
    ideal = mod.symbol_mapping
    distances = np.abs(rx_symbols[:, None] - ideal[None, :])
    nearest = ideal[np.argmin(distances, axis=1)]
    err = rx_symbols - nearest
    return np.abs(err) ** 2


def _resolve_buffer_paths(arg: str) -> list[str]:
    """Accept a directory, a glob pattern, or a single .npz file."""
    if os.path.isdir(arg):
        paths = sorted(glob.glob(os.path.join(arg, "*.npz")))
    else:
        paths = sorted(glob.glob(arg))
    if not paths:
        raise FileNotFoundError(f"No .npz buffers matched '{arg}'")
    return paths


def load_rx_buffers(paths: list[str]) -> list[dict]:
    """Load each .npz dumped by one_way_threaded.py --save-rx-buf into a dict
    of {samples, search_from, sample_rate, sps, mod_scheme, code_rate, ...}."""
    bufs: list[dict] = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        bufs.append({
            "path":         p,
            "samples":      d["samples"],
            "search_from":  int(d["search_from"]),
            "sample_rate":  int(d["sample_rate"]),
            "sps":          int(d["sps"]),
            "mod_scheme":   str(d["mod_scheme"]),
            "code_rate":    str(d["code_rate"]),
            "n_valid_in_buf": int(d["n_valid_in_buf"]),
            "n_total_in_buf": int(d["n_total_in_buf"]),
        })
    return bufs


def sweep(rx: RXPipeline, buffers: list[dict],
          bnts_list: list[float], zeta_list: list[float], l_list: list[int],
          verbose: bool = False, progress_every: int | None = None) -> list[dict]:
    """For each (BnTs, ζ, L) combo, decode every loaded buffer with that
    Gardner config and aggregate EVM across all valid packets.

    verbose=True prints a line per combo. Otherwise prints a progress line
    every ``progress_every`` combos (auto-chosen ~100 lines if None)."""
    results: list[dict] = []
    total = len(bnts_list) * len(zeta_list) * len(l_list)
    if progress_every is None:
        progress_every = max(1, total // 100)
    t_start = time.perf_counter()
    best_so_far = float("inf")

    for i, (bnts, zeta, L) in enumerate(product(bnts_list, zeta_list, l_list), start=1):
        rx.config.GARDNER_BN_TS = bnts
        rx.config.GARDNER_ZETA  = zeta
        rx.config.GARDNER_L     = L

        sq_err_chunks: list[np.ndarray] = []
        n_decoded = 0
        n_total   = 0
        n_syms    = 0
        per_mod_n: dict[str, int] = {}
        t0 = time.perf_counter()

        for b in buffers:
            try:
                packets, _ = rx.receive(b["samples"], search_from=b["search_from"])
            except Exception as e:
                if verbose:
                    print(f"  [{i:5d}/{total}] BnTs={bnts:.5f} ζ={zeta:.3f} L={L}: "
                          f"buffer {os.path.basename(b['path'])} raised "
                          f"{type(e).__name__}: {e}")
                continue
            n_total += len(packets)
            for pkt in packets:
                if not pkt.valid or pkt.rx_symbols is None or pkt.rx_symbols.size == 0:
                    continue
                mod = _mod_for(pkt.mod_scheme) if pkt.mod_scheme is not None \
                      else _mod_for(b["mod_scheme"])
                sq_err_chunks.append(_squared_errors(pkt.rx_symbols, mod))
                n_decoded += 1
                n_syms    += pkt.rx_symbols.size
                key = pkt.mod_scheme.name if pkt.mod_scheme is not None else b["mod_scheme"]
                per_mod_n[key] = per_mod_n.get(key, 0) + pkt.rx_symbols.size

        dt = time.perf_counter() - t0
        if sq_err_chunks:
            evm = 100.0 * float(np.sqrt(np.mean(np.concatenate(sq_err_chunks))))
        else:
            evm = float("nan")

        results.append({
            "bnts": bnts, "zeta": zeta, "L": L,
            "evm_pct":   evm,
            "n_decoded": n_decoded,
            "n_total":   n_total,
            "n_syms":    n_syms,
            "per_mod_n": per_mod_n,
            "elapsed_s": dt,
        })

        if not np.isnan(evm) and evm < best_so_far:
            best_so_far = evm

        if verbose:
            evm_s = f"{evm:6.2f}%" if not np.isnan(evm) else "   nan"
            print(f"  [{i:5d}/{total}] BnTs={bnts:.5f} ζ={zeta:.3f} L={L}: "
                  f"decoded {n_decoded:>3d}/{n_total:<3d}  EVM={evm_s}  "
                  f"({n_syms} syms, {dt*1000:.0f} ms)")
        elif i % progress_every == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            eta = elapsed * (total - i) / i if i > 0 else 0.0
            best_s = f"{best_so_far:.2f}%" if best_so_far < float("inf") else "—"
            print(f"  [{i:5d}/{total}]  best EVM so far: {best_s}  "
                  f"elapsed {elapsed:.1f}s  eta {eta:.1f}s")

    results.sort(key=lambda r: (np.isnan(r["evm_pct"]), r["evm_pct"]))
    return results


def _decimate_ticks(vals: list, max_ticks: int = 12) -> tuple[list[int], list[str]]:
    """Pick at most ``max_ticks`` evenly-spaced indices into ``vals`` and return
    (positions, labels) so heatmap axes don't get smothered by 100 ticks."""
    n = len(vals)
    if n <= max_ticks:
        positions = list(range(n))
    else:
        positions = list(np.linspace(0, n - 1, max_ticks).round().astype(int))
        # de-dup while preserving order
        seen = set()
        positions = [p for p in positions if not (p in seen or seen.add(p))]
    labels = [f"{vals[p]:g}" for p in positions]
    return positions, labels


def maybe_plot_heatmap(results: list[dict], out_path: Path, *, title: str) -> None:
    """Two stacked heatmaps:
       (top)    EVM(BnTs, ζ) at the L that gave the global minimum.
       (bottom) min EVM over L for each (BnTs, ζ) — the "best you can do"."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping heatmap")
        return

    bnts_vals = sorted({r["bnts"] for r in results})
    L_vals    = sorted({r["L"]    for r in results})
    zeta_vals = sorted({r["zeta"] for r in results})

    if len(bnts_vals) < 2 or len(zeta_vals) < 2:
        print("  [plot] grid too sparse for heatmap — skipping")
        return

    # results is already sorted ascending by EVM; first finite entry is global best.
    best = next((r for r in results if not np.isnan(r["evm_pct"])), None)
    if best is None:
        print("  [plot] no finite EVM samples — skipping")
        return

    # Grid for "L = best L"
    grid_best_L = np.full((len(zeta_vals), len(bnts_vals)), np.nan)
    # Grid for "min EVM over all L"
    grid_min_L  = np.full((len(zeta_vals), len(bnts_vals)), np.nan)

    for r in results:
        if np.isnan(r["evm_pct"]):
            continue
        zi = zeta_vals.index(r["zeta"])
        bi = bnts_vals.index(r["bnts"])
        if r["L"] == best["L"]:
            grid_best_L[zi, bi] = r["evm_pct"]
        if np.isnan(grid_min_L[zi, bi]) or r["evm_pct"] < grid_min_L[zi, bi]:
            grid_min_L[zi, bi] = r["evm_pct"]

    finite = np.concatenate([
        grid_best_L[np.isfinite(grid_best_L)],
        grid_min_L[np.isfinite(grid_min_L)],
    ])
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9))

    bnts_pos, bnts_lbl = _decimate_ticks(bnts_vals, max_ticks=12)
    zeta_pos, zeta_lbl = _decimate_ticks(zeta_vals, max_ticks=10)

    for ax, grid, sub in [
        (ax_top, grid_best_L, f"L = {best['L']} (the L of the global best)"),
        (ax_bot, grid_min_L,  "min EVM over all L (per BnTs, ζ)"),
    ]:
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(bnts_pos);  ax.set_xticklabels(bnts_lbl, rotation=30, ha="right")
        ax.set_yticks(zeta_pos);  ax.set_yticklabels(zeta_lbl)
        ax.set_xlabel("BnTs")
        ax.set_ylabel("ζ")
        ax.set_title(sub)
        # Mark the global minimum on each panel
        ax.scatter([bnts_vals.index(best["bnts"])], [zeta_vals.index(best["zeta"])],
                   marker="*", s=180, edgecolors="white", facecolors="red", zorder=10,
                   label=(f"best: BnTs={best['bnts']:g}, ζ={best['zeta']:g}, "
                          f"L={best['L']}, EVM={best['evm_pct']:.2f}%"))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(title)
    fig.colorbar(im, ax=[ax_top, ax_bot], label="EVM (%)", shrink=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved heatmap → {out_path}")


def _arange_inclusive(lo: float, hi: float, step: float, decimals: int = 8) -> list[float]:
    """Inclusive linspace built from a (min, max, step) triple, robust to FP drift."""
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, decimals) for i in range(max(n, 1))]


def _fmt_list(vals: list, max_show: int = 8) -> str:
    if len(vals) <= max_show:
        return str(vals)
    head = ", ".join(str(v) for v in vals[:3])
    tail = ", ".join(str(v) for v in vals[-2:])
    return f"[{head}, …, {tail}] ({len(vals)} values)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rx_buffers", help="directory, glob, or .npz file dumped by --save-rx-buf")

    # Default grid (~16500 combos = ~117× finer than the previous 140-combo default).
    p.add_argument("--bnts-min",  type=float, default=0.001, help="BnTs min (default: 0.0001)")
    p.add_argument("--bnts-max",  type=float, default=0.02,   help="BnTs max (default: 0.01)")
    p.add_argument("--bnts-step", type=float, default=0.001, help="BnTs step (default: 0.0001)")
    p.add_argument("--zeta-min",  type=float, default=0.6,    help="ζ min (default: 0.4)")
    p.add_argument("--zeta-max",  type=float, default=2.0,    help="ζ max (default: 2.0)")
    p.add_argument("--zeta-step", type=float, default=0.04,   help="ζ step (default: 0.05)")
    p.add_argument("--l-min",     type=int,   default=1,      help="L min (default: 1)")
    p.add_argument("--l-max",     type=int,   default=5,      help="L max (default: 5)")

    # Power-user overrides — if given, win over the corresponding min/max/step.
    p.add_argument("--bnts",  type=str, default=None,
                   help="explicit comma-separated BnTs list (overrides --bnts-min/max/step)")
    p.add_argument("--zetas", type=str, default=None,
                   help="explicit comma-separated ζ list (overrides --zeta-min/max/step)")
    p.add_argument("--ls",    type=str, default=None,
                   help="explicit comma-separated L list (overrides --l-min/max)")

    p.add_argument("--top",     type=int, default=15,
                   help="how many of the best combos to print at the end (default: 15)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print one line per combination (default: progress lines only)")
    p.add_argument("--plot",    action="store_true",
                   help="save EVM heatmap to tests/plots/sweep_gardner.png")
    args = p.parse_args()

    bnts_list = _parse_floats(args.bnts) if args.bnts else \
                _arange_inclusive(args.bnts_min, args.bnts_max, args.bnts_step)
    zeta_list = _parse_floats(args.zetas) if args.zetas else \
                _arange_inclusive(args.zeta_min, args.zeta_max, args.zeta_step)
    L_list    = _parse_ints(args.ls) if args.ls else \
                list(range(args.l_min, args.l_max + 1))

    paths = _resolve_buffer_paths(args.rx_buffers)
    bufs  = load_rx_buffers(paths)

    print("=== Gardner-TED sweep (real captures) ===")
    print(f"Buffers    : {len(bufs)} files from '{args.rx_buffers}'")
    n_samples_total = sum(len(b["samples"]) for b in bufs)
    print(f"             {n_samples_total} samples total "
          f"({n_samples_total / bufs[0]['sample_rate'] * 1e3:.1f} ms @ "
          f"{bufs[0]['sample_rate']/1e6:g} Msps, sps={bufs[0]['sps']})")
    mod_set = sorted({b["mod_scheme"] for b in bufs})
    code_set = sorted({b["code_rate"] for b in bufs})
    print(f"Capture cfg: mod={','.join(mod_set)}  code={','.join(code_set)}")
    n_valid_capt = sum(b["n_valid_in_buf"] for b in bufs)
    n_total_capt = sum(b["n_total_in_buf"] for b in bufs)
    print(f"At capture : {n_valid_capt}/{n_total_capt} valid packets")
    print(f"Grid       : BnTs ∈ {_fmt_list(bnts_list)}")
    print(f"             ζ    ∈ {_fmt_list(zeta_list)}")
    print(f"             L    ∈ {_fmt_list(L_list)}")
    print(f"             (total {len(bnts_list)*len(zeta_list)*len(L_list)} combinations)")
    print()

    # All buffers must share sample_rate / sps for one RXPipeline to handle them.
    sr_set  = {b["sample_rate"] for b in bufs}
    sps_set = {b["sps"]         for b in bufs}
    if len(sr_set) != 1 or len(sps_set) != 1:
        print(f"ERROR: buffers have inconsistent sample_rate ({sr_set}) "
              f"or sps ({sps_set}). Cannot mix.")
        return 1

    config = PipelineConfig()
    config.SAMPLE_RATE = bufs[0]["sample_rate"]
    config.SPS         = bufs[0]["sps"]
    rx = RXPipeline(config)

    print("Sweeping …")
    results = sweep(rx, bufs, bnts_list, zeta_list, L_list, verbose=args.verbose)

    # ----- top-N table -----
    print()
    print(f"Best {min(args.top, len(results))} combinations (lowest EVM):")
    print(f"  {'BnTs':>9s}  {'ζ':>6s}  {'L':>3s}  {'decoded':>9s}  {'EVM (%)':>9s}  {'#syms':>7s}")
    for r in results[:args.top]:
        evm_str = f"{r['evm_pct']:.2f}" if not np.isnan(r["evm_pct"]) else "  nan"
        print(f"  {r['bnts']:>9.5f}  {r['zeta']:>6.3f}  {r['L']:>3d}  "
              f"{r['n_decoded']:>3d}/{r['n_total']:<3d}    {evm_str:>9s}  {r['n_syms']:>7d}")

    if args.plot:
        out = Path(__file__).resolve().parents[1] / "tests" / "plots" / "sweep_gardner.png"
        title = (f"Gardner sweep — {','.join(mod_set)}/{','.join(code_set)}  "
                 f"({len(bufs)} buffers, {n_valid_capt} pkts at capture)")
        maybe_plot_heatmap(results, out, title=title)

    return 0


if __name__ == "__main__":
    sys.exit(main())
