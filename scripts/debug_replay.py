"""Per-detection diagnostic replay of saved RX buffers.

Loads buffers from --save-rx-buf and walks RXPipeline.receive() with
verbose per-detection prints: position-in-raw, NCC peak, CFO estimate,
phase, decode outcome, and reason for any IndexError tail-cut.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.pipeline import PipelineConfig, RXPipeline
from modules.frame_sync.frame_sync import full_buffer_xcorr_sync
from modules.pulse_shaping.pulse_shaping import match_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("buf_dir")
    args = ap.parse_args()

    cfg = PipelineConfig()
    rx = RXPipeline(cfg)

    files = sorted(glob.glob(str(Path(args.buf_dir) / "*.npz")))
    print(f"Loaded {len(files)} buffers from {args.buf_dir}\n")

    n_ref = len(rx.preamble_ref)
    rrc_tail = rx.num_taps - 1
    print(f"frame_len(estimate via header_n_bits): cfg.SPS={cfg.SPS}  num_taps={rx.num_taps}  preamble_ref_len={n_ref}\n")

    for f in files:
        d = np.load(f, allow_pickle=False)
        raw = d["samples"].astype(np.complex64)
        search_from = int(d["search_from"])
        expected_seqs = list(d["seq_nums"]) if "seq_nums" in d.files else []
        print(f"=== {Path(f).name}  len={len(raw)}  search_from={search_from}  expected_valid_seqs={expected_seqs}")

        search_buf = raw[search_from:]
        filtered = match_filter(search_buf, rx.rrc_taps)
        fine, cfos = full_buffer_xcorr_sync(
            filtered, rx.preamble_ref, rx.preamble_ref_rev,
            float(cfg.SYNC_CONFIG.ncc_threshold), cfg.SAMPLE_RATE,
        )

        n_det = fine.sample_idxs.size
        print(f"  {n_det} detections (search_buf len={len(search_buf)}, filtered len={len(filtered)})")

        for k in range(n_det):
            preamble_peak = int(fine.sample_idxs[k])
            payload_start = preamble_peak + n_ref          # in filtered coords
            abs_payload_start = search_from + payload_start  # in raw coords
            ncc = float(fine.peak_ratios[k])
            cfo = float(cfos[k])
            phase = float(fine.phase_estimates[k])

            slice_avail = len(filtered) - payload_start
            full_overlap_avail = (len(search_buf) - rrc_tail) - payload_start

            rx_syms = filtered[payload_start:]
            outcome = "?"
            err = ""
            try:
                pkt = rx.decode(rx_syms, np.float32(cfo), np.float32(phase))
                if pkt.valid:
                    sb = np.packbits(pkt.payload[:32].astype(np.uint8))
                    seq = (int(sb[0]) << 24) | (int(sb[1]) << 16) | (int(sb[2]) << 8) | int(sb[3])
                    outcome = f"VALID  seq={seq}"
                else:
                    outcome = f"CRC_FAIL  reason='{pkt.err_reason}'"
            except IndexError as e:
                outcome = "INDEX_ERROR (tail-cut)"
                err = str(e)
            except Exception as e:
                outcome = f"DECODE_EXC  {type(e).__name__}: {e}"

            print(f"  [{k}] preamble_peak={preamble_peak:6d}  abs_pl_start={abs_payload_start:6d}  "
                  f"ncc={ncc:.3f}  cfo={cfo:+7.1f}Hz  phase={phase:+.2f}  "
                  f"avail={slice_avail:6d}  full_ovl={full_overlap_avail:6d}  -> {outcome}{(' '+err) if err else ''}")
        print()


if __name__ == "__main__":
    main()
