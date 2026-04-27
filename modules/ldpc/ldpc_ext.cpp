// File: modules/ldpc/ldpc_ext.cpp
//
// C++ pybind11 extension for LDPC encode + min-sum BP decode.
//
// encode  — structured encoder via H_p^{-1}.  Computes the syndrome
//           syn = H_s · message (popcount-and over bit-packed H_s rows),
//           then derives parity by p = H_p^{-1} · syn (XOR of precomputed
//           H_p^{-1} columns selected by the set bits of syn).  The
//           codeword is [message | p].
//
//           This replaces the previous dense G-matrix encoder, which did
//           ~k full-row XORs of n bits per codeword.  The new path moves
//           ~k×k bits of work into ~m × (k + m) bits — for the typical
//           rate 5/6, n=1944 case (k=1620, m=324) that's a ~5–10×
//           reduction in per-codeword XOR work, plus a much smaller
//           working set (the parity-only structures fit comfortably in
//           L1/L2 instead of spilling out of cache).
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
// message:      uint8[k]                (each element 0/1)
// h_s_packed:   uint64[m, k_words]      H_s rows packed LSB-first
// h_p_inv_cols: uint64[m, m_words]      row j = column j of H_p^{-1} packed
// n:            codeword length (= k + m)
// returns:      uint8[n]                each element 0/1
//
// Algorithm:
//   1. Pack message into k_words = ceil(k/64) uint64s.
//   2. For each row i of H_s, syn[i] = popcount(H_s[i] & m_packed) mod 2.
//      Accumulate the set bits of syn into a packed m-bit vector.
//   3. For each set bit j of syn, XOR h_p_inv_cols[j] into a packed
//      m-bit accumulator p.
//   4. Output codeword = [message | p].

// Single-codeword core; reused by the batched entry point below.
// Caller is responsible for any GIL handling around this — it does no Python work.
static inline void encode_one(
    const uint8_t*  msg,        // [k]
    const uint64_t* hs,         // [m, k_words]
    const uint64_t* hpi,        // [m, m_words]
    ssize_t k, ssize_t m,
    ssize_t k_words, ssize_t m_words,
    uint8_t*        out         // [k + m]
) {
    // Step 1: pack message into k_words uint64s.
    std::vector<uint64_t> m_packed(k_words, 0);
    for (ssize_t i = 0; i < k; ++i) {
        m_packed[i >> 6] |= ((uint64_t)(msg[i] & 1)) << (i & 63);
    }

    // Step 2: syn[i] = popcount(H_s[i] & m_packed) mod 2.
    std::vector<uint64_t> syn_packed(m_words, 0);
    for (ssize_t i = 0; i < m; ++i) {
        const uint64_t* row = hs + i * k_words;
        uint64_t parity_acc = 0;
        for (ssize_t w = 0; w < k_words; ++w) {
            parity_acc ^= row[w] & m_packed[w];
        }
        uint64_t bit = (uint64_t)__builtin_parityll(parity_acc) & 1ULL;
        syn_packed[i >> 6] |= bit << (i & 63);
    }

    // Step 3: p = H_p^{-1} · syn — for each set bit j of syn, XOR col j into p.
    std::vector<uint64_t> p_packed(m_words, 0);
    for (ssize_t w = 0; w < m_words; ++w) {
        uint64_t bits = syn_packed[w];
        while (bits) {
            int b = __builtin_ctzll(bits);
            ssize_t j = (ssize_t)w * 64 + b;
            if (j >= m) break;  // last word may carry spurious bits past m
            const uint64_t* col = hpi + j * m_words;
            for (ssize_t ww = 0; ww < m_words; ++ww) {
                p_packed[ww] ^= col[ww];
            }
            bits &= bits - 1;
        }
    }

    // Step 4: write codeword [message | p].
    for (ssize_t i = 0; i < k; ++i) {
        out[i] = msg[i] & 1;
    }
    for (ssize_t j = 0; j < m; ++j) {
        out[k + j] = (uint8_t)((p_packed[j >> 6] >> (j & 63)) & 1ULL);
    }
}

static py::array_t<uint8_t> encode_ext(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> message,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> h_s_packed,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> h_p_inv_cols,
    int n
) {
    auto msg_buf = message.unchecked<1>();
    auto hs_buf  = h_s_packed.unchecked<2>();
    auto hpi_buf = h_p_inv_cols.unchecked<2>();

    const ssize_t k = msg_buf.shape(0);
    const ssize_t m = (ssize_t)n - k;
    const ssize_t k_words = hs_buf.shape(1);
    const ssize_t m_words = hpi_buf.shape(1);

    if (m <= 0 || k <= 0) {
        throw std::invalid_argument("invalid k/n");
    }
    if (hs_buf.shape(0) != m || hpi_buf.shape(0) != m) {
        throw std::invalid_argument("h_s_packed/h_p_inv_cols row count must equal m = n-k");
    }
    if ((ssize_t)((k + 63) / 64) != k_words) {
        throw std::invalid_argument("h_s_packed.shape[1] inconsistent with k");
    }
    if ((ssize_t)((m + 63) / 64) != m_words) {
        throw std::invalid_argument("h_p_inv_cols.shape[1] inconsistent with m");
    }

    py::array_t<uint8_t> out(n);
    auto out_buf = out.mutable_unchecked<1>();

    const uint8_t*  msg = msg_buf.data(0);
    const uint64_t* hs  = hs_buf.data(0, 0);
    const uint64_t* hpi = hpi_buf.data(0, 0);
    uint8_t*        out_ptr = out_buf.mutable_data(0);

    {
        py::gil_scoped_release nogil;
        encode_one(msg, hs, hpi, k, m, k_words, m_words, out_ptr);
    }
    return out;
}

// Batched encoder: one Python ↔ C++ transition for all `n_cw` codewords.
// Per-call pybind11/numpy marshalling was the dominant cost on the TX path.
static py::array_t<uint8_t> encode_batch_ext(
    py::array_t<uint8_t,  py::array::c_style | py::array::forcecast> messages,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> h_s_packed,
    py::array_t<uint64_t, py::array::c_style | py::array::forcecast> h_p_inv_cols,
    int n,
    int n_cw
) {
    auto msg_buf = messages.unchecked<1>();
    auto hs_buf  = h_s_packed.unchecked<2>();
    auto hpi_buf = h_p_inv_cols.unchecked<2>();

    const ssize_t m = (ssize_t)hs_buf.shape(0);
    const ssize_t k = (ssize_t)n - m;
    const ssize_t k_words = hs_buf.shape(1);
    const ssize_t m_words = hpi_buf.shape(1);

    if (k <= 0 || m <= 0 || n_cw <= 0) {
        throw std::invalid_argument("invalid k, m, or n_cw");
    }
    if (msg_buf.shape(0) != (ssize_t)n_cw * k) {
        throw std::invalid_argument("messages length must equal n_cw * k");
    }
    if (hpi_buf.shape(0) != m) {
        throw std::invalid_argument("h_p_inv_cols rows must equal m = n-k");
    }
    if ((ssize_t)((k + 63) / 64) != k_words) {
        throw std::invalid_argument("h_s_packed.shape[1] inconsistent with k");
    }
    if ((ssize_t)((m + 63) / 64) != m_words) {
        throw std::invalid_argument("h_p_inv_cols.shape[1] inconsistent with m");
    }

    py::array_t<uint8_t> out({(ssize_t)n_cw, (ssize_t)n});
    auto out_buf = out.mutable_unchecked<2>();

    const uint8_t*  msg = msg_buf.data(0);
    const uint64_t* hs  = hs_buf.data(0, 0);
    const uint64_t* hpi = hpi_buf.data(0, 0);
    uint8_t*        out_ptr = out_buf.mutable_data(0, 0);

    {
        py::gil_scoped_release nogil;
        for (int i = 0; i < n_cw; ++i) {
            encode_one(msg + (ssize_t)i * k, hs, hpi, k, m, k_words, m_words,
                       out_ptr + (ssize_t)i * n);
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

PYBIND11_MODULE(ldpc_ext, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        LDPC C++ extension.

        encode — structured encoder via H_p^{-1}: syndrome popcount + column XOR.
        decode — min-sum belief propagation matching the Python reference.
    )pbdoc";

    m.def("encode", &encode_ext,
          py::arg("message"), py::arg("h_s_packed"), py::arg("h_p_inv_cols"), py::arg("n"),
          "Encode message bits to codeword via H_p^{-1}-based structured encoder.");

    m.def("encode_batch", &encode_batch_ext,
          py::arg("messages"), py::arg("h_s_packed"), py::arg("h_p_inv_cols"),
          py::arg("n"), py::arg("n_cw"),
          "Encode n_cw concatenated messages in one call. Returns uint8[n_cw, n].");

    m.def("decode", &decode_ext,
          py::arg("llr"), py::arg("edge_var"),
          py::arg("check_order"), py::arg("check_bounds"),
          py::arg("k"), py::arg("max_iter"), py::arg("alpha"),
          "Min-sum BP decode using edge-grouped check structures.");
}
