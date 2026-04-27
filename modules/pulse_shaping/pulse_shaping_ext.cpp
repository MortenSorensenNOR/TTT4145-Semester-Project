// modules/pulse_shaping_ext.cpp
//
// C++ FIR kernels for complex64 signals.
//
// match_filter  — FIR convolution of complex64 signal with real float32 taps.
//                 Deinterleaves re/im so the inner loop is a plain float dot
//                 product → auto-vectorises to SSE2/AVX2 on x86 or NEON on ARM.
//
// upsample      — Polyphase upsampler: avoids redundant zero-multiply iterations
//                 by decomposing the FIR into sps sub-filters applied to the
//                 symbol array directly (≈ sps× fewer operations than direct).
//
// Compiled with -O3 -ffast-math -funroll-loops; ARM: -mfpu=neon (see setup.py).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace py = pybind11;
using c64 = std::complex<float>;
using f32 = float;

// ---------------------------------------------------------------------------
// match_filter
//
// Equivalent to:  np.convolve(signal, taps, mode="full")[M-1:]
//
// Mathematical identity used:
//   result[i] = sum_{m=0}^{M-1} signal[i+m] * trev[m]
// where trev[m] = taps[M-1-m]  (reversed taps for forward-sequential access).
// signal[j] = 0 for j >= N (trailing zero-pad embedded in re_in / im_in).
// ---------------------------------------------------------------------------

py::array_t<c64> match_filter(
    py::array_t<c64, py::array::c_style | py::array::forcecast> signal_in,
    py::array_t<f32, py::array::c_style | py::array::forcecast> taps_in)
{
    const int N = static_cast<int>(signal_in.size());
    const int M = static_cast<int>(taps_in.size());
    const c64* sig  = signal_in.data();
    const f32* taps = taps_in.data();

    py::array_t<c64> out_np(N);
    f32* out = reinterpret_cast<f32*>(out_np.mutable_data());

    {
        py::gil_scoped_release nogil;

        // Reversed taps — inner loop becomes a forward dot-product (cache-friendly)
        std::vector<f32> trev(M);
        for (int k = 0; k < M; ++k) trev[k] = taps[M - 1 - k];

        // Deinterleaved input padded with M trailing zeros (handles trailing edge
        // where signal[i+m] would be out-of-bounds — just reads zeros instead).
        const int buf_len = N + M;
        std::vector<f32> re_in(buf_len, 0.0f), im_in(buf_len, 0.0f);
        for (int i = 0; i < N; ++i) {
            re_in[i] = sig[i].real();
            im_in[i] = sig[i].imag();
        }

        const f32* re = re_in.data();
        const f32* im = im_in.data();
        const f32* tr = trev.data();

        // Simple inner dot-product: -O3 -ffast-math -march=native causes GCC/Clang to
        // auto-vectorize this into AVX2 FMAs with multiple implicit accumulators.
        // Manual unrolling here tends to increase register pressure and hurt performance.
        for (int i = 0; i < N; ++i) {
            f32 re_acc = 0.0f, im_acc = 0.0f;
            const f32* rs = re + i;
            const f32* is_ = im + i;
            for (int m = 0; m < M; ++m) {
                f32 t   = tr[m];
                re_acc += rs[m]  * t;
                im_acc += is_[m] * t;
            }
            out[2 * i]     = re_acc;
            out[2 * i + 1] = im_acc;
        }
    }
    return out_np;
}

// ---------------------------------------------------------------------------
// upsample (polyphase)
//
// Equivalent to:
//   up = np.zeros(N_sym * sps, complex64);  up[::sps] = symbols
//   np.convolve(up, taps, mode="full")
//
// Output length: N_sym * sps + M - 1
//
// Polyphase decomposition avoids the zero-multiplies in the upsampled FIR.
// The filter of length M is split into sps sub-filters of length ≈ M/sps.
// Each sub-filter is applied to the (compact) symbol array, writing to every
// sps-th output position — O(N_sym × M) total instead of O(N_sym × sps × M).
// ---------------------------------------------------------------------------

py::array_t<c64> upsample(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols_in,
    int sps,
    py::array_t<f32, py::array::c_style | py::array::forcecast> taps_in)
{
    const int Ns = static_cast<int>(symbols_in.size());
    const int M  = static_cast<int>(taps_in.size());
    const c64* sym  = symbols_in.data();
    const f32* taps = taps_in.data();

    const int Nout = Ns * sps + M - 1;

    py::array_t<c64> out_np(Nout);
    f32* out = reinterpret_cast<f32*>(out_np.mutable_data());

    {
        py::gil_scoped_release nogil;

        // Deinterleave symbols
        std::vector<f32> sym_re(Ns), sym_im(Ns);
        for (int i = 0; i < Ns; ++i) {
            sym_re[i] = sym[i].real();
            sym_im[i] = sym[i].imag();
        }

        std::fill(out, out + 2 * Nout, 0.0f);

        // For each phase p ∈ [0, sps-1]:
        //   sub-filter: h_p[k] = taps[p + k*sps],  k = 0 .. sub_len-1
        //   output positions: p, p+sps, p+2*sps, ... (every sps samples)
        //   For output index n = q*sps + p:
        //     out[n] = sum_{k=0}^{sub_len-1} symbols[q-k] * h_p[k]   (causal)
        //
        // This is a full convolution of symbols with h_p, written at offset p
        // and stride sps.

        for (int p = 0; p < sps; ++p) {
            // Build sub-filter for phase p
            int sub_len = (M - p + sps - 1) / sps;  // ceil((M-p)/sps)
            std::vector<f32> hp(sub_len);
            for (int k = 0; k < sub_len; ++k) hp[k] = taps[p + k * sps];

            // Full convolution of symbols with hp; output length Ns + sub_len - 1
            int conv_len = Ns + sub_len - 1;
            for (int q = 0; q < conv_len; ++q) {
                f32 re_acc = 0.0f, im_acc = 0.0f;
                const int k_lo = std::max(0, q - (Ns - 1));
                const int k_hi = std::min(sub_len, q + 1);
                for (int k = k_lo; k < k_hi; ++k) {
                    int s = q - k;
                    f32 h = hp[k];
                    re_acc += sym_re[s] * h;
                    im_acc += sym_im[s] * h;
                }
                int out_idx = q * sps + p;
                if (out_idx < Nout) {
                    out[2 * out_idx]     = re_acc;
                    out[2 * out_idx + 1] = im_acc;
                }
            }
        }
    }
    return out_np;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(pulse_shaping_ext, m, py::mod_gil_not_used()) {
    m.doc() = "C++ FIR filter kernels for complex64 signals (pulse shaping).";

    m.def("match_filter", &match_filter,
          py::arg("signal"), py::arg("taps"),
          R"(FIR match filter.

Equivalent to np.convolve(signal, taps, mode='full')[M-1:].
Deinterleaves re/im for SIMD auto-vectorisation of the inner dot-product.)");

    m.def("upsample", &upsample,
          py::arg("symbols"), py::arg("sps"), py::arg("taps"),
          R"(Polyphase FIR upsampler.

Equivalent to:
  up = np.zeros(N*sps, complex64); up[::sps] = symbols
  np.convolve(up, taps, mode='full')

Uses polyphase decomposition to skip zero-inserts (~sps× fewer ops).)");
}
