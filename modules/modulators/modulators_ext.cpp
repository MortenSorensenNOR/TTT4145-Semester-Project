// modules/modulators_ext.cpp
//
// Fast max-log LLR demapper for unit-norm PSK constellations (BPSK, QPSK, PSK8,
// PSK16).  For unit-norm constellations |s|² is constant, so:
//
//   |r - s|² = |r|² − 2·Re(r·conj(s)) + 1
//
// and minimising |r-s|² over a subset is equivalent to maximising
// Re(r·conj(s)) = r.real·s.real + r.imag·s.imag over that subset.  The
// |r|² term cancels in (min_C1 − min_C0).
//
// LLR(b) = min_{s∈C1}|r-s|² − min_{s∈C0}|r-s|²
//        = 2·(max_{s∈C0} Re(r·conj(s)) − max_{s∈C1} Re(r·conj(s)))
//
// Compared to the numpy chain (allocate (N,M) complex diff, abs, square, min)
// this kernel:
//   - holds the constellation (M ≤ 16) in registers / L1
//   - streams symbols sequentially with no extra allocations
//   - computes per-bit max in the same loop as the constellation projection
//
// Output is multiplied by 2 to match the d² formulation downstream.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;
using c64 = std::complex<float>;
using f32 = float;

// Compute max-log LLR for N symbols against an M-point unit-norm constellation.
//
// sym_re/sym_im : float32[M]      — constellation real/imag parts
// c0_idx[b]     : list of indices in C0 for bit b  (variable length)
// c1_idx[b]     : list of indices in C1 for bit b
// nbits         : bits per symbol (1, 2, 3 or 4)
// nsyms         : number of symbols
// Output : float32[N, nbits]
//
// Per-bit fan-in is up to M/2 indices; we precompute fixed-size arrays to keep
// the inner loops branch-free.
static py::array_t<f32> psk_llr_unit_norm(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols_in,
    py::array_t<f32, py::array::c_style | py::array::forcecast> sym_re_in,
    py::array_t<f32, py::array::c_style | py::array::forcecast> sym_im_in,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> c0_idx_in,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> c1_idx_in,
    int nbits)
{
    const int M = static_cast<int>(sym_re_in.size());
    const int N = static_cast<int>(symbols_in.size());

    if (sym_im_in.size() != M) {
        throw std::invalid_argument("sym_re/sym_im size mismatch");
    }
    if (c0_idx_in.shape(0) != nbits || c1_idx_in.shape(0) != nbits) {
        throw std::invalid_argument("c0/c1 idx must have nbits rows");
    }
    if (M > 16) {
        throw std::invalid_argument("constellation > 16 points not supported");
    }
    const int per_bit = static_cast<int>(c0_idx_in.shape(1));
    if (c1_idx_in.shape(1) != per_bit) {
        throw std::invalid_argument("c0/c1 idx must have same #cols");
    }

    py::array_t<f32> out({(py::ssize_t)N, (py::ssize_t)nbits});
    f32* out_ptr = out.mutable_data(0, 0);

    const c64* sym = symbols_in.data();
    const f32* sym_re_ptr = sym_re_in.data();
    const f32* sym_im_ptr = sym_im_in.data();
    const int32_t* c0 = c0_idx_in.data();
    const int32_t* c1 = c1_idx_in.data();

    {
        py::gil_scoped_release nogil;

        // Local copy of constellation in fixed-size arrays — held in registers.
        f32 sr[16], si[16];
        for (int i = 0; i < M; ++i) {
            sr[i] = sym_re_ptr[i];
            si[i] = sym_im_ptr[i];
        }

        // Constellation indices per bit, padded to per_bit length.  Padding is
        // handled by Python with -1 to mean "ignore" — but we just store actual
        // indices since that's already the convention from numpy.
        // c0/c1 are int32[nbits, per_bit].

        for (int n = 0; n < N; ++n) {
            const f32 rr = sym[n].real();
            const f32 ri = sym[n].imag();

            // Compute Re(r · conj(s_k)) = rr*sr_k + ri*si_k for all k.
            f32 dot[16];
            for (int k = 0; k < M; ++k) {
                dot[k] = rr * sr[k] + ri * si[k];
            }

            for (int b = 0; b < nbits; ++b) {
                const int32_t* idx0 = c0 + b * per_bit;
                const int32_t* idx1 = c1 + b * per_bit;
                f32 max0 = -std::numeric_limits<f32>::infinity();
                f32 max1 = -std::numeric_limits<f32>::infinity();
                for (int j = 0; j < per_bit; ++j) {
                    const f32 d0 = dot[idx0[j]];
                    const f32 d1 = dot[idx1[j]];
                    if (d0 > max0) max0 = d0;
                    if (d1 > max1) max1 = d1;
                }
                // LLR(b) = min_C1|r-s|² − min_C0|r-s|² = 2·(max_C0 dot − max_C1 dot)
                out_ptr[n * nbits + b] = 2.0f * (max0 - max1);
            }
        }
    }
    return out;
}

PYBIND11_MODULE(modulators_ext, m, py::mod_gil_not_used()) {
    m.doc() = "Fast max-log LLR demapper for unit-norm PSK constellations.";

    m.def("psk_llr_unit_norm", &psk_llr_unit_norm,
          py::arg("symbols"),
          py::arg("sym_re"),
          py::arg("sym_im"),
          py::arg("c0_idx"),
          py::arg("c1_idx"),
          py::arg("nbits"),
          "Max-log LLR demapper for unit-norm PSK constellations.");
}
