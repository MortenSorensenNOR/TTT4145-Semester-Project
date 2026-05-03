"""Sweep Costas-loop parameters and rank by post-Costas constellation tightness.

Operates on RX buffers captured from a real Pluto run by
``pluto/one_way_threaded.py --save-rx-buf <dir>``.  For every (Bn, ζ)
combination on the sweep grid, the script re-runs ``RXPipeline.receive()``
on each saved buffer and computes the **EVM** of the post-Costas symbols
(RMS distance to the nearest ideal constellation point, % of unit magnitude).
Lower EVM = tighter point cloud around each ideal point.

Capture buffers first:
    # node B (or wherever)
    uv run python pluto/one_way_threaded.py --mode tx
    # node A
    uv run python pluto/one_way_threaded.py --mode rx --save-rx-buf rx_dump --save-n 4

Then sweep Costas only:
    uv run python scripts/sweep_costas_params.py rx_dump
    uv run python scripts/sweep_costas_params.py rx_dump --plot
    uv run python scripts/sweep_costas_params.py rx_dump \
        --bns 0.001,0.005,0.008,0.02 --zetas 0.5,0.707,1.0

Or sweep Costas + Gardner jointly (slower, larger grid):
    uv run python scripts/sweep_costas_params.py rx_dump --with-gardner --plot

The same input buffers are used across all parameter combinations, so the
only thing changing between runs is the loop config — making the ranking
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

from modules.costas_loop.costas import CostasConfig
from modules.frame_constructor.frame_constructor import ModulationSchemes
from modules.modulators.modulators import BPSK, PSK8, PSK16, QPSK
from modules.pipeline import PipelineConfig, RXPipeline


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


_MOD_INSTANCE = {
    ModulationSchemes.BPSK:  BPSK(),
    ModulationSchemes.QPSK:  QPSK(),
    ModulationSchemes.PSK8:  PSK8(),
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
    bufs: list[dict] = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        # seq_nums was added later — gracefully treat older captures as having
        # no ground truth.  When None, the sweep falls back to ranking by EVM
        # alone; when present, it ranks by (n_match desc, EVM asc).
        seq_nums = (np.asarray(d["seq_nums"], dtype=np.int64)
                    if "seq_nums" in d.files else None)
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
            "seq_nums":     seq_nums,
        })
    return bufs


def _seq_from_payload(payload: np.ndarray) -> int | None:
    """Extract the 32-bit big-endian sequence number from a payload.  Mirrors
    the encoding in pluto/one_way_threaded.py:_build_packet."""
    if payload is None or payload.size < 32:
        return None
    sb = np.packbits(payload[:32].astype(np.uint8))
    return ((int(sb[0]) << 24) | (int(sb[1]) << 16)
            | (int(sb[2]) << 8)  |  int(sb[3]))


def _run_one_combo(rx: RXPipeline, buffers: list[dict],
                   verbose_prefix: str | None,
                   collect_per_buffer: bool = False) -> dict:
    """Decode every buffer with the current rx.config and aggregate stats.

    Returns dict with: evm_pct, n_decoded, n_total, n_syms, per_mod_n,
    elapsed_s, n_match, n_expected, [per_buffer].

    n_match / n_expected are populated only for buffers that carry
    ground-truth ``seq_nums`` — they count how many of the originally-decoded
    seqs were also recovered with the current config.  This catches silent
    LDPC divergence (random bits passing CRC-16) that EVM alone can't see."""
    sq_err_chunks: list[np.ndarray] = []
    n_decoded = 0
    n_total   = 0
    n_syms    = 0
    n_match     = 0
    n_expected  = 0
    per_mod_n: dict[str, int] = {}
    per_buffer: list[dict] = []
    t0 = time.perf_counter()

    for b in buffers:
        try:
            packets, _ = rx.receive(b["samples"], search_from=b["search_from"])
        except Exception as e:
            if verbose_prefix is not None:
                print(f"{verbose_prefix} buffer {os.path.basename(b['path'])} raised "
                      f"{type(e).__name__}: {e}")
            if collect_per_buffer:
                per_buffer.append({
                    "path": b["path"], "n_decoded": 0, "n_total": 0,
                    "n_match": 0, "n_expected":
                        len(b["seq_nums"]) if b.get("seq_nums") is not None else 0,
                    "evm_pct": float("nan"),
                    "decoded_seqs": [], "expected_seqs":
                        list(b["seq_nums"]) if b.get("seq_nums") is not None else [],
                    "error": f"{type(e).__name__}: {e}",
                })
            continue

        buf_decoded_seqs: list[int] = []
        buf_sq_err: list[np.ndarray] = []
        buf_n_decoded = 0
        for pkt in packets:
            if not pkt.valid or pkt.rx_symbols is None or pkt.rx_symbols.size == 0:
                continue
            mod = _mod_for(pkt.mod_scheme) if pkt.mod_scheme is not None \
                  else _mod_for(b["mod_scheme"])
            errs = _squared_errors(pkt.rx_symbols, mod)
            buf_sq_err.append(errs)
            sq_err_chunks.append(errs)
            buf_n_decoded += 1
            n_syms    += pkt.rx_symbols.size
            key = pkt.mod_scheme.name if pkt.mod_scheme is not None else b["mod_scheme"]
            per_mod_n[key] = per_mod_n.get(key, 0) + pkt.rx_symbols.size
            seq = _seq_from_payload(pkt.payload)
            if seq is not None:
                buf_decoded_seqs.append(seq)

        n_total   += len(packets)
        n_decoded += buf_n_decoded

        # Ground-truth check: how many of the captured seqs did this combo
        # also recover correctly?
        if b.get("seq_nums") is not None:
            expected = set(int(s) for s in b["seq_nums"])
            decoded  = set(buf_decoded_seqs)
            buf_n_match    = len(expected & decoded)
            buf_n_expected = len(expected)
            n_match    += buf_n_match
            n_expected += buf_n_expected
        else:
            buf_n_match = 0
            buf_n_expected = 0

        if collect_per_buffer:
            buf_evm = (100.0 * float(np.sqrt(np.mean(np.concatenate(buf_sq_err))))
                       if buf_sq_err else float("nan"))
            per_buffer.append({
                "path": b["path"], "n_decoded": buf_n_decoded, "n_total": len(packets),
                "n_match": buf_n_match, "n_expected": buf_n_expected,
                "evm_pct": buf_evm,
                "decoded_seqs": sorted(set(buf_decoded_seqs)),
                "expected_seqs": (sorted(int(s) for s in b["seq_nums"])
                                  if b.get("seq_nums") is not None else []),
                "error": None,
            })

    dt = time.perf_counter() - t0
    if sq_err_chunks:
        evm = 100.0 * float(np.sqrt(np.mean(np.concatenate(sq_err_chunks))))
    else:
        evm = float("nan")
    return {
        "evm_pct": evm, "n_decoded": n_decoded, "n_total": n_total, "n_syms": n_syms,
        "per_mod_n": per_mod_n, "elapsed_s": dt,
        "n_match": n_match, "n_expected": n_expected,
        "per_buffer": per_buffer,
    }


def sweep(rx: RXPipeline, buffers: list[dict],
          bn_list: list[float], zeta_list: list[float],
          gardner_grid: list[tuple[float, float, int]] | None,
          verbose: bool = False, progress_every: int | None = None,
          collect_per_buffer: bool = False) -> list[dict]:
    """Sweep Costas (Bn, ζ).  If ``gardner_grid`` is given, also sweep over
    Gardner (BnTs, ζ_g, L) — full Cartesian product.  ``gardner_grid`` is
    a list of (bnts, zeta_g, L) triples; pass ``None`` to skip Gardner.

    When ground-truth seq_nums are present in the buffers, results are sorted
    by (n_match desc, EVM asc) — configs that recover more captured packets
    win, ties broken by tighter constellation."""
    results: list[dict] = []
    g_combos = gardner_grid if gardner_grid is not None else [None]
    total = len(bn_list) * len(zeta_list) * len(g_combos)
    if progress_every is None:
        progress_every = max(1, total // 100)
    t_start = time.perf_counter()
    best_so_far = float("inf")
    have_truth = any(b.get("seq_nums") is not None for b in buffers)

    for i, (bn, zeta_c, g) in enumerate(
        product(bn_list, zeta_list, g_combos), start=1
    ):
        rx.config.COSTAS_CONFIG = CostasConfig(
            loop_noise_bandwidth_normalized=bn, damping_factor=zeta_c,
        )
        if g is not None:
            bnts, zeta_g, L = g
            rx.config.GARDNER_BN_TS = bnts
            rx.config.GARDNER_ZETA  = zeta_g
            rx.config.GARDNER_L     = L

        prefix = (f"  [{i:5d}/{total}] Bn={bn:.5f} ζ={zeta_c:.3f}"
                  + (f" BnTs={g[0]:.5f} ζg={g[1]:.3f} L={g[2]}" if g is not None else "")
                  + ":") if verbose else None

        stats = _run_one_combo(rx, buffers, prefix, collect_per_buffer=collect_per_buffer)

        row: dict = {
            "bn": bn, "zeta": zeta_c,
            **stats,
        }
        if g is not None:
            row["bnts"], row["zeta_g"], row["L"] = g
        results.append(row)

        evm = stats["evm_pct"]
        if not np.isnan(evm) and evm < best_so_far:
            best_so_far = evm

        if verbose:
            evm_s = f"{evm:6.2f}%" if not np.isnan(evm) else "   nan"
            match_s = (f"  match={stats['n_match']:>3d}/{stats['n_expected']:<3d}"
                       if have_truth else "")
            print(f"{prefix} decoded {stats['n_decoded']:>3d}/{stats['n_total']:<3d}{match_s}  "
                  f"EVM={evm_s}  ({stats['n_syms']} syms, {stats['elapsed_s']*1000:.0f} ms)")
        elif i % progress_every == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            eta = elapsed * (total - i) / i if i > 0 else 0.0
            best_s = f"{best_so_far:.2f}%" if best_so_far < float("inf") else "—"
            print(f"  [{i:5d}/{total}]  best EVM so far: {best_s}  "
                  f"elapsed {elapsed:.1f}s  eta {eta:.1f}s")

    if have_truth:
        results.sort(key=lambda r: (-r["n_match"], np.isnan(r["evm_pct"]), r["evm_pct"]))
    else:
        results.sort(key=lambda r: (np.isnan(r["evm_pct"]), r["evm_pct"]))
    return results


def _decimate_ticks(vals: list, max_ticks: int = 12) -> tuple[list[int], list[str]]:
    n = len(vals)
    if n <= max_ticks:
        positions = list(range(n))
    else:
        positions = list(np.linspace(0, n - 1, max_ticks).round().astype(int))
        seen = set()
        positions = [p for p in positions if not (p in seen or seen.add(p))]
    labels = [f"{vals[p]:g}" for p in positions]
    return positions, labels


def _project_min(results: list[dict], x_key: str, y_key: str
                 ) -> tuple[list, list, np.ndarray]:
    """Build a 2D grid of min EVM over all other params, projected onto (x, y)."""
    x_vals = sorted({r[x_key] for r in results})
    y_vals = sorted({r[y_key] for r in results})
    grid = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in results:
        if np.isnan(r["evm_pct"]):
            continue
        xi = x_vals.index(r[x_key])
        yi = y_vals.index(r[y_key])
        if np.isnan(grid[yi, xi]) or r["evm_pct"] < grid[yi, xi]:
            grid[yi, xi] = r["evm_pct"]
    return x_vals, y_vals, grid


def _heatmap_panel(ax, x_vals, y_vals, grid, *, x_label, y_label, sub, vmin, vmax,
                   best_x=None, best_y=None, best_label=None):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax)
    xpos, xlbl = _decimate_ticks(x_vals, max_ticks=12)
    ypos, ylbl = _decimate_ticks(y_vals, max_ticks=10)
    ax.set_xticks(xpos);  ax.set_xticklabels(xlbl, rotation=30, ha="right")
    ax.set_yticks(ypos);  ax.set_yticklabels(ylbl)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(sub)
    if best_x is not None and best_y is not None and best_x in x_vals and best_y in y_vals:
        ax.scatter([x_vals.index(best_x)], [y_vals.index(best_y)],
                   marker="*", s=180, edgecolors="white", facecolors="red", zorder=10,
                   label=best_label or "best")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    return im


def maybe_plot_heatmap(results: list[dict], out_path: Path, *,
                       title: str, with_gardner: bool) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available — skipping heatmap")
        return

    best = next((r for r in results if not np.isnan(r["evm_pct"])), None)
    if best is None:
        print("  [plot] no finite EVM samples — skipping")
        return

    finite_evms = [r["evm_pct"] for r in results if not np.isnan(r["evm_pct"])]
    vmin = float(np.min(finite_evms))
    vmax = float(np.max(finite_evms))

    x_vals, y_vals, grid_costas = _project_min(results, "bn", "zeta")
    if len(x_vals) < 2 or len(y_vals) < 2:
        print("  [plot] Costas grid too sparse for heatmap — skipping")
        return

    if not with_gardner:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        im = _heatmap_panel(
            ax, x_vals, y_vals, grid_costas,
            x_label="Bn (loop noise BW, normalised)",
            y_label="ζ (damping factor)",
            sub="EVM(Bn, ζ)",
            vmin=vmin, vmax=vmax,
            best_x=best["bn"], best_y=best["zeta"],
            best_label=(f"best: Bn={best['bn']:g}, ζ={best['zeta']:g}, "
                        f"EVM={best['evm_pct']:.2f}%"),
        )
        fig.suptitle(title)
        fig.colorbar(im, ax=ax, label="EVM (%)")
    else:
        gx, gy, grid_gardner = _project_min(results, "bnts", "zeta_g")
        if len(gx) < 2 or len(gy) < 2:
            # Gardner grid was a single point — fall back to Costas-only plot.
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            im = _heatmap_panel(
                ax, x_vals, y_vals, grid_costas,
                x_label="Bn", y_label="ζ_costas",
                sub="EVM(Bn, ζ_costas)  — Gardner grid is 1D",
                vmin=vmin, vmax=vmax,
                best_x=best["bn"], best_y=best["zeta"],
                best_label=f"best: EVM={best['evm_pct']:.2f}%",
            )
            fig.suptitle(title)
            fig.colorbar(im, ax=ax, label="EVM (%)")
        else:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 9))
            im_top = _heatmap_panel(
                ax_top, x_vals, y_vals, grid_costas,
                x_label="Bn", y_label="ζ_costas",
                sub="min EVM over Gardner params, per (Bn, ζ_costas)",
                vmin=vmin, vmax=vmax,
                best_x=best["bn"], best_y=best["zeta"],
                best_label=(f"best: Bn={best['bn']:g}, ζ_c={best['zeta']:g}, "
                            f"EVM={best['evm_pct']:.2f}%"),
            )
            _heatmap_panel(
                ax_bot, gx, gy, grid_gardner,
                x_label="BnTs", y_label="ζ_gardner",
                sub="min EVM over Costas params, per (BnTs, ζ_gardner)",
                vmin=vmin, vmax=vmax,
                best_x=best["bnts"], best_y=best["zeta_g"],
                best_label=(f"best: BnTs={best['bnts']:g}, ζ_g={best['zeta_g']:g}, "
                            f"L={best['L']}, EVM={best['evm_pct']:.2f}%"),
            )
            fig.suptitle(title)
            fig.colorbar(im_top, ax=[ax_top, ax_bot], label="EVM (%)", shrink=0.8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved heatmap → {out_path}")


def _arange_inclusive(lo: float, hi: float, step: float, decimals: int = 8) -> list[float]:
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

    # Costas grid (default ~330 combos).
    p.add_argument("--bn-min",    type=float, default=0.001, help="Bn min (default: 0.001)")
    p.add_argument("--bn-max",    type=float, default=0.03,  help="Bn max (default: 0.03)")
    p.add_argument("--bn-step",   type=float, default=0.001, help="Bn step (default: 0.001)")
    p.add_argument("--zeta-min",  type=float, default=0.5,   help="ζ min (default: 0.5)")
    p.add_argument("--zeta-max",  type=float, default=1.5,   help="ζ max (default: 1.5)")
    p.add_argument("--zeta-step", type=float, default=0.1,   help="ζ step (default: 0.1)")
    p.add_argument("--bns",       type=str, default=None,
                   help="explicit comma-separated Bn list (overrides --bn-min/max/step)")
    p.add_argument("--zetas",     type=str, default=None,
                   help="explicit comma-separated ζ list (overrides --zeta-min/max/step)")

    # Combined-mode (Gardner) grid — defaults are deliberately coarse.
    p.add_argument("--with-gardner", action="store_true",
                   help="also sweep Gardner-TED (BnTs, ζ_g, L) jointly with Costas")
    p.add_argument("--gardner-bnts-min",  type=float, default=0.001)
    p.add_argument("--gardner-bnts-max",  type=float, default=0.005)
    p.add_argument("--gardner-bnts-step", type=float, default=0.001)
    p.add_argument("--gardner-zeta-min",  type=float, default=0.707)
    p.add_argument("--gardner-zeta-max",  type=float, default=2.0)
    p.add_argument("--gardner-zeta-step", type=float, default=0.25)
    p.add_argument("--gardner-l-min",     type=int,   default=2)
    p.add_argument("--gardner-l-max",     type=int,   default=3)
    p.add_argument("--gardner-bnts",  type=str, default=None,
                   help="explicit comma-separated Gardner BnTs list")
    p.add_argument("--gardner-zetas", type=str, default=None,
                   help="explicit comma-separated Gardner ζ list")
    p.add_argument("--gardner-ls",    type=str, default=None,
                   help="explicit comma-separated Gardner L list")

    p.add_argument("--top",     type=int, default=15,
                   help="how many of the best combos to print at the end (default: 15)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print one line per combination (default: progress lines only)")
    p.add_argument("--plot",    action="store_true",
                   help="save EVM heatmap to tests/plots/sweep_costas[_gardner].png")
    args = p.parse_args()

    bn_list = _parse_floats(args.bns) if args.bns else \
              _arange_inclusive(args.bn_min, args.bn_max, args.bn_step)
    zeta_list = _parse_floats(args.zetas) if args.zetas else \
                _arange_inclusive(args.zeta_min, args.zeta_max, args.zeta_step)

    gardner_grid: list[tuple[float, float, int]] | None = None
    if args.with_gardner:
        bnts_list = _parse_floats(args.gardner_bnts) if args.gardner_bnts else \
                    _arange_inclusive(args.gardner_bnts_min, args.gardner_bnts_max,
                                      args.gardner_bnts_step)
        gz_list = _parse_floats(args.gardner_zetas) if args.gardner_zetas else \
                  _arange_inclusive(args.gardner_zeta_min, args.gardner_zeta_max,
                                    args.gardner_zeta_step)
        gl_list = _parse_ints(args.gardner_ls) if args.gardner_ls else \
                  list(range(args.gardner_l_min, args.gardner_l_max + 1))
        gardner_grid = list(product(bnts_list, gz_list, gl_list))

    paths = _resolve_buffer_paths(args.rx_buffers)
    bufs  = load_rx_buffers(paths)

    label = "Costas + Gardner" if args.with_gardner else "Costas-loop"
    print(f"=== {label} sweep (real captures) ===")
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
    print(f"Costas grid: Bn ∈ {_fmt_list(bn_list)}")
    print(f"             ζ  ∈ {_fmt_list(zeta_list)}")
    if gardner_grid is not None:
        bnts_unique = sorted({g[0] for g in gardner_grid})
        gz_unique   = sorted({g[1] for g in gardner_grid})
        gl_unique   = sorted({g[2] for g in gardner_grid})
        print(f"Gardner    : BnTs ∈ {_fmt_list(bnts_unique)}")
        print(f"             ζ    ∈ {_fmt_list(gz_unique)}")
        print(f"             L    ∈ {_fmt_list(gl_unique)}")
    total_combos = len(bn_list) * len(zeta_list) * (len(gardner_grid) if gardner_grid else 1)
    print(f"             (total {total_combos} combinations)")
    print()

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

    have_truth = any(b.get("seq_nums") is not None for b in bufs)
    if not have_truth:
        print("  [warn] no ground-truth seq_nums in any buffer — ranking by EVM "
              "only.  Recapture with the updated one_way_threaded.py to enable "
              "the LDPC-convergence cross-check.")

    is_single = total_combos == 1
    if is_single:
        print("Single-config run (1×1 grid) — collecting per-buffer detail.")
    else:
        print("Sweeping …")
    results = sweep(rx, bufs, bn_list, zeta_list, gardner_grid,
                    verbose=args.verbose, collect_per_buffer=is_single)

    # ----- single-config detailed report -----
    if is_single:
        r = results[0]
        print()
        print("Per-buffer breakdown:")
        hdr = f"  {'buffer':<24s}  {'decoded':>9s}"
        if have_truth:
            hdr += f"  {'match':>9s}"
        hdr += f"  {'EVM (%)':>9s}"
        print(hdr)
        for pb in r["per_buffer"]:
            name = os.path.basename(pb["path"])[:24]
            evm_s = f"{pb['evm_pct']:.2f}" if not np.isnan(pb["evm_pct"]) else "  nan"
            line  = f"  {name:<24s}  {pb['n_decoded']:>3d}/{pb['n_total']:<3d}    "
            if have_truth:
                line += f"{pb['n_match']:>3d}/{pb['n_expected']:<3d}    "
            line += f"{evm_s:>9s}"
            if pb["error"]:
                line += f"   ERROR: {pb['error']}"
            print(line)
            if have_truth and pb["expected_seqs"]:
                missed = sorted(set(pb["expected_seqs"]) - set(pb["decoded_seqs"]))
                extra  = sorted(set(pb["decoded_seqs"])  - set(pb["expected_seqs"]))
                if missed:
                    print(f"      missed seqs: {missed}")
                if extra:
                    print(f"      extra  seqs (decoded now, not at capture): {extra}")
        print()
        print("Aggregate:")
        evm_s = f"{r['evm_pct']:.2f}%" if not np.isnan(r['evm_pct']) else "nan"
        print(f"  decoded     : {r['n_decoded']}/{r['n_total']}")
        if have_truth:
            frac = (r["n_match"] / r["n_expected"]) if r["n_expected"] else float("nan")
            print(f"  ground-truth: {r['n_match']}/{r['n_expected']} match  "
                  f"({frac*100:.1f}%) — LDPC convergence cross-check")
        print(f"  EVM         : {evm_s}  ({r['n_syms']} symbols)")
        if r["per_mod_n"]:
            print(f"  per-mod syms: " + ", ".join(f"{k}={v}" for k, v in r["per_mod_n"].items()))
        print(f"  elapsed     : {r['elapsed_s']*1000:.0f} ms")
    else:
        # ----- top-N table -----
        print()
        rank_label = ("highest match, then lowest EVM" if have_truth
                      else "lowest EVM")
        print(f"Best {min(args.top, len(results))} combinations ({rank_label}):")
        match_col = f"  {'match':>9s}" if have_truth else ""
        if gardner_grid is None:
            print(f"  {'Bn':>9s}  {'ζ':>6s}  {'decoded':>9s}{match_col}  {'EVM (%)':>9s}  {'#syms':>7s}")
            for r in results[:args.top]:
                evm_str = f"{r['evm_pct']:.2f}" if not np.isnan(r["evm_pct"]) else "  nan"
                m_str = (f"  {r['n_match']:>3d}/{r['n_expected']:<3d}    "
                         if have_truth else "")
                print(f"  {r['bn']:>9.5f}  {r['zeta']:>6.3f}  "
                      f"{r['n_decoded']:>3d}/{r['n_total']:<3d}    {m_str}{evm_str:>9s}  {r['n_syms']:>7d}")
        else:
            print(f"  {'Bn':>9s}  {'ζ_c':>5s}  {'BnTs':>9s}  {'ζ_g':>5s}  {'L':>3s}  "
                  f"{'decoded':>9s}{match_col}  {'EVM (%)':>9s}  {'#syms':>7s}")
            for r in results[:args.top]:
                evm_str = f"{r['evm_pct']:.2f}" if not np.isnan(r["evm_pct"]) else "  nan"
                m_str = (f"  {r['n_match']:>3d}/{r['n_expected']:<3d}    "
                         if have_truth else "")
                print(f"  {r['bn']:>9.5f}  {r['zeta']:>5.3f}  {r['bnts']:>9.5f}  "
                      f"{r['zeta_g']:>5.3f}  {r['L']:>3d}  "
                      f"{r['n_decoded']:>3d}/{r['n_total']:<3d}    {m_str}{evm_str:>9s}  {r['n_syms']:>7d}")

    if args.plot and not is_single:
        out_name = "sweep_costas_gardner.png" if args.with_gardner else "sweep_costas.png"
        out = Path(__file__).resolve().parents[1] / "tests" / "plots" / out_name
        title = (f"{label} sweep — {','.join(mod_set)}/{','.join(code_set)}  "
                 f"({len(bufs)} buffers, {n_valid_capt} pkts at capture)")
        maybe_plot_heatmap(results, out, title=title, with_gardner=args.with_gardner)

    return 0


if __name__ == "__main__":
    sys.exit(main())
