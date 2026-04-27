// File: modules/ldpc/ldpc_ext.cpp
//
// C++ pybind11 extension for LDPC encode + min-sum BP decode.
//
// encode  — bit-packed XOR over the systematic generator G.  For every set
//           bit in the message, XOR the corresponding G row (packed into
//           uint64 words) into the accumulator.  Replaces the dense
//           `message @ G % 2` matmul, which is the main encode bottleneck.
//
// decode  — min-sum belief propagation in the edge-message representation
//           used by the Python reference.  Same algorithm; same intermediate
//           values; just C++ instead of Python+numba.
//
// Compiled with -O3 -ffast-math; ARM: also -mcpu=cortex-a9 -mfpu=neon
// (see setup.py for the per-target flag table).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------
//
// message:  uint8[k]            (each element 0/1)
// g_packed: uint64[k, n_words]  (row i = G[i,:] bits packed LSB-first)
// n:        codeword length in bits
// returns:  uint8[n]            (each element 0/1)
//
// Loop invariant: acc holds the running XOR sum of G rows selected by msg
// bits 0..i-1.  After the loop, acc[w] bit b is codeword[w*64 + b].

static py::array_t<uint8_t> encode_ext(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> message,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> g_packed,
    int n
) {
    auto msg_buf = message.unchecked<1>();
    auto g_buf   = g_packed.unchecked<2>();
    const ssize_t k = msg_buf.shape(0);
    const ssize_t n_words = g_buf.shape(1);

    if (g_buf.shape(0) != k) {
        throw std::invalid_argument("g_packed.shape[0] must equal len(message)");
    }
    if ((ssize_t)((n + 63) / 64) != n_words) {
        throw std::invalid_argument("g_packed.shape[1] inconsistent with n");
    }

    py::array_t<uint8_t> out(n);
    auto out_buf = out.mutable_unchecked<1>();

    {
        py::gil_scoped_release nogil;
        std::vector<uint64_t> acc(n_words, 0);
        for (ssize_t i = 0; i < k; ++i) {
            if (msg_buf(i) & 1) {
                const uint64_t* row = &g_buf(i, 0);
                for (ssize_t w = 0; w < n_words; ++w) {
                    acc[w] ^= row[w];
                }
            }
        }
        for (int j = 0; j < n; ++j) {
            out_buf(j) = (uint8_t)((acc[j >> 6] >> (j & 63)) & 1ULL);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Decoder — min-sum belief propagation
// ---------------------------------------------------------------------------
//
// llr:          float32[n]           channel LLR (positive ⇒ bit 0)
// edge_var:     int64[E]             variable index for each edge
// check_order:  int64[E]             edge indices grouped by check
// check_bounds: int64[num_checks+1]  check-edge ranges in check_order
// k:            message length (return first k bits)
// max_iter:     iteration cap
// alpha:        min-sum scaling factor
//
// Returns uint8[k].  Matches the Python `ldpc_decode` reference bit-for-bit
// (modulo float rounding) — same min-sum check update, same scatter-add for
// the variable update, same per-iteration syndrome early-exit.

static py::array_t<uint8_t> decode_ext(
    py::array_t<float, py::array::c_style | py::array::forcecast>   llr_in,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> edge_var_in,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> check_order_in,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> check_bounds_in,
    int k,
    int max_iter,
    float alpha
) {
    auto llr_buf  = llr_in.unchecked<1>();
    auto var_buf  = edge_var_in.unchecked<1>();
    auto ord_buf  = check_order_in.unchecked<1>();
    auto bnd_buf  = check_bounds_in.unchecked<1>();

    const ssize_t n         = llr_buf.shape(0);
    const ssize_t num_edges = var_buf.shape(0);
    const ssize_t num_checks = bnd_buf.shape(0) - 1;

    if (ord_buf.shape(0) != num_edges) {
        throw std::invalid_argument("check_order length must match edge_var");
    }

    py::array_t<uint8_t> out(k);
    auto out_buf = out.mutable_unchecked<1>();

    // Pull arrays into local pointers — the Python objects don't move during
    // the call, but the pointer-deref shaves a small amount of indirection.
    const float*   llr  = llr_buf.data(0);
    const int64_t* ev   = var_buf.data(0);
    const int64_t* co   = ord_buf.data(0);
    const int64_t* bnd  = bnd_buf.data(0);

    py::gil_scoped_release nogil;

    std::vector<float> v2c(num_edges);
    std::vector<float> c2v(num_edges, 0.0f);
    std::vector<float> l_total(n);
    std::vector<uint8_t> hard(n);

    // v2c initialized to llr[edge_var[e]]
    for (ssize_t e = 0; e < num_edges; ++e) {
        v2c[e] = llr[ev[e]];
    }

    constexpr int MIN_CHECK_DEGREE = 2;

    for (int iter = 0; iter < max_iter; ++iter) {
        // ---- check node update (min-sum) ----
        for (ssize_t ci = 0; ci < num_checks; ++ci) {
            const int64_t start = bnd[ci];
            const int64_t end   = bnd[ci + 1];
            const int64_t d     = end - start;
            if (d < MIN_CHECK_DEGREE) {
                for (int64_t j = start; j < end; ++j) c2v[co[j]] = 0.0f;
                continue;
            }

            float total_sign = 1.0f;
            float min1 = std::numeric_limits<float>::infinity();
            float min2 = std::numeric_limits<float>::infinity();
            int64_t argmin_local = 0;

            for (int64_t j = start; j < end; ++j) {
                const float msg = v2c[co[j]];
                if (msg < 0.0f) total_sign = -total_sign;
                const float mag = std::fabs(msg);
                if (mag < min1) {
                    min2 = min1;
                    min1 = mag;
                    argmin_local = j - start;
                } else if (mag < min2) {
                    min2 = mag;
                }
            }

            for (int64_t j = start; j < end; ++j) {
                const float msg = v2c[co[j]];
                const float sign = (msg < 0.0f) ? -1.0f : 1.0f;
                const float sign_excl = total_sign * sign;
                const float min_excl  = ((j - start) != argmin_local) ? min1 : min2;
                c2v[co[j]] = alpha * sign_excl * min_excl;
            }
        }

        // ---- variable node update + hard decision ----
        // l_total = llr + scatter-add(c2v, edge_var)
        std::copy(llr, llr + n, l_total.begin());
        for (ssize_t e = 0; e < num_edges; ++e) {
            l_total[ev[e]] += c2v[e];
        }
        // v2c[e] = l_total[ev[e]] - c2v[e]
        for (ssize_t e = 0; e < num_edges; ++e) {
            v2c[e] = l_total[ev[e]] - c2v[e];
        }
        for (ssize_t v = 0; v < n; ++v) {
            hard[v] = (l_total[v] < 0.0f) ? 1 : 0;
        }

        // ---- syndrome check (same edge groupings as check update) ----
        bool all_zero = true;
        for (ssize_t ci = 0; ci < num_checks && all_zero; ++ci) {
            uint8_t parity = 0;
            const int64_t start = bnd[ci];
            const int64_t end   = bnd[ci + 1];
            for (int64_t j = start; j < end; ++j) {
                parity ^= hard[ev[co[j]]];
            }
            if (parity) all_zero = false;
        }
        if (all_zero) break;
    }

    for (int v = 0; v < k; ++v) out_buf(v) = hard[v];
    return out;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(ldpc_ext, m) {
    m.doc() = R"pbdoc(
        LDPC C++ extension.

        encode — bit-packed XOR encoder over a packed systematic generator.
        decode — min-sum belief propagation matching the Python reference.
    )pbdoc";

    m.def("encode", &encode_ext,
          py::arg("message"), py::arg("g_packed"), py::arg("n"),
          "Encode message bits to codeword via bit-packed XOR over G rows.");

    m.def("decode", &decode_ext,
          py::arg("llr"), py::arg("edge_var"),
          py::arg("check_order"), py::arg("check_bounds"),
          py::arg("k"), py::arg("max_iter"), py::arg("alpha"),
          "Min-sum BP decode using edge-grouped check structures.");
}
