// File: modules/nda_ted/nda_ted_ext.cpp
//
// NDA (Non-Data-Aided) symbol timing synchroniser — pybind11 C++ extension.
//
// Algorithm: M. Rice, "Digital Communications: A Discrete-Time Approach",
//            Prentice Hall, 2009.  NDA TED with Farrow cubic interpolator.
//
// Optimisations for ARM Cortex-A9:
//   - Farrow cubic filter computed with Horner's method (4 muls vs 9)
//   - SoA real/imag split — enables NEON auto-vectorisation
//   - No std::complex in the inner loop
//   - No trig functions (angle via atan2f on smoothed c1 accumulator only)
//   - -O3 -ffast-math applied via setup.py

#include <cmath>
#include <complex>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using c64 = std::complex<float>;
using f32 = float;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Farrow cubic interpolator (I_ord = 3), Horner form.
// Coefficients match Rice p.clients 362.
//
//   v3 = ( 1/6)*z[n+2] + (-1/2)*z[n+1] + ( 1/2)*z[n] + (-1/6)*z[n-1]
//   v2 = ( 0  )*z[n+2] + ( 1/2)*z[n+1] + (-1  )*z[n] + ( 1/2)*z[n-1]
//   v1 = (-1/6)*z[n+2] + ( 1  )*z[n+1] + (-1/2)*z[n] + (-1/3)*z[n-1]
//   v0 = z[n]
//   out = ((mu*v3 + v2)*mu + v1)*mu + v0
// ---------------------------------------------------------------------------

static inline void farrow3(
    f32 zm1_r, f32 z0_r, f32 z1_r, f32 z2_r,   // real parts: z[n-1..n+2]
    f32 zm1_i, f32 z0_i, f32 z1_i, f32 z2_i,   // imag parts
    f32 mu,
    f32& out_r, f32& out_i)
{
    // real
    f32 v3r = ( 1.f/6.f)*z2_r + (-1.f/2.f)*z1_r + ( 1.f/2.f)*z0_r + (-1.f/6.f)*zm1_r;
    f32 v2r = (          0.f) + ( 1.f/2.f)*z1_r + (-1.f    )*z0_r + ( 1.f/2.f)*zm1_r;
    f32 v1r = (-1.f/6.f)*z2_r + ( 1.f    )*z1_r + (-1.f/2.f)*z0_r + (-1.f/3.f)*zm1_r;
    f32 v0r = z0_r;
    out_r   = ((mu*v3r + v2r)*mu + v1r)*mu + v0r;

    // imag
    f32 v3i = ( 1.f/6.f)*z2_i + (-1.f/2.f)*z1_i + ( 1.f/2.f)*z0_i + (-1.f/6.f)*zm1_i;
    f32 v2i = (          0.f) + ( 1.f/2.f)*z1_i + (-1.f    )*z0_i + ( 1.f/2.f)*zm1_i;
    f32 v1i = (-1.f/6.f)*z2_i + ( 1.f    )*z1_i + (-1.f/2.f)*z0_i + (-1.f/3.f)*zm1_i;
    f32 v0i = z0_i;
    out_i   = ((mu*v3i + v2i)*mu + v1i)*mu + v0i;
}

// ---------------------------------------------------------------------------
// NDA symbol timing synchroniser
//
// Parameters
// ----------
// symbols  : RRC matched-filter output at Ns samples/symbol
// Ns       : nominal samples per symbol
// L        : TED smoothing half-length (2*L+1 samples averaged)
// BnTs     : normalised loop bandwidth (loop_bw * symbol_period)
// zeta     : loop damping factor (0.707 = Butterworth)
//
// Returns
// -------
// timing-corrected symbols at 1 sample/symbol
// ---------------------------------------------------------------------------

py::array_t<c64> nda_symb_sync(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
    int Ns, int L, f32 BnTs, f32 zeta,
    bool prepend_first = false)
{
    auto in = symbols.unchecked<1>();
    int  N  = static_cast<int>(in.shape(0));

    // Layout always prepends one zero sample (matches Python hstack([0], z)).
    // When prepend_first is true the kernel additionally duplicates the first
    // input sample, replacing what was previously a Python-side concat.
    int prepend = prepend_first ? 2 : 1;
    int n_total = N + prepend;

    std::vector<f32> re(n_total, 0.f), im(n_total, 0.f);

    // Output size depends on the inner loop's progress, so accumulate in
    // std::vectors here and allocate the py::array_t once we know the length.
    int max_out = n_total / Ns + 2;
    std::vector<f32> out_r(max_out, 0.f), out_i(max_out, 0.f);
    int mm = 1;  // output write index (index 0 left as zero, matches Python)

    {
    py::gil_scoped_release nogil;
    if (N > 0 && prepend_first) {
        // index 0 is the structural zero; index 1 duplicates input[0].
        re[1] = in(0).real();
        im[1] = in(0).imag();
    }
    for (int i = 0; i < N; ++i) {
        re[i + prepend] = in(i).real();
        im[i + prepend] = in(i).imag();
    }

    // Loop filter gains (Rice eq. 8.89)
    f32 K0 = -1.f;
    f32 Kp =  1.f;
    f32 denom = zeta + 1.f / (4.f * zeta);
    f32 K1 = 4.f * zeta / denom * BnTs / Ns / Kp / K0;
    f32 K2 = 4.f / (denom * denom) * (BnTs / Ns) * (BnTs / Ns) / Kp / K0;

    // NDA TED twiddle factors: exp(-j*2*pi/Ns*kk) for kk in 0..Ns-1.
    // These depend only on Ns; precompute once instead of calling cos/sin
    // every output sample × every kk.
    std::vector<f32> ted_cos(Ns), ted_sin(Ns);
    {
        const f32 base = -2.f * static_cast<f32>(M_PI) / static_cast<f32>(Ns);
        for (int kk = 0; kk < Ns; ++kk) {
            f32 angle = base * static_cast<f32>(kk);
            ted_cos[kk] = std::cos(angle);
            ted_sin[kk] = std::sin(angle);
        }
    }
    const f32* tcos = ted_cos.data();
    const f32* tsin = ted_sin.data();
    const f32 inv_Ns = 1.f / static_cast<f32>(Ns);
    const f32 inv_two_pi = -1.f / (2.f * static_cast<f32>(M_PI));

    // c1 smoothing buffer (circular, length 2*L+1)
    int  buf_len  = 2 * L + 1;
    std::vector<f32> c1_buf_r(buf_len, 0.f), c1_buf_i(buf_len, 0.f);
    int  buf_ptr  = 0;  // circular write pointer
    const f32 inv_buf_len = 1.f / static_cast<f32>(buf_len);

    f32 vi        = 0.f;
    f32 CNT_next  = 0.f;
    f32 mu_next   = 0.f;
    int underflow = 0;
    f32 epsilon   = 0.f;
    f32 mu        = 0.f;
    f32 CNT       = 0.f;

    int loop_end = Ns * (n_total / Ns - (Ns - 1));

    for (int nn = 1; nn < loop_end; ++nn) {
        CNT = CNT_next;
        mu  = mu_next;

        if (underflow == 1) {
            // --- Decimated interpolant at current strobe ---
            f32 zr, zi;
            // bounds: need nn-1, nn, nn+1, nn+2
            if (nn >= 1 && nn + 2 < n_total) {
                farrow3(re[nn-1], re[nn], re[nn+1], re[nn+2],
                        im[nn-1], im[nn], im[nn+1], im[nn+2],
                        mu, zr, zi);
            } else {
                zr = re[nn]; zi = im[nn];
            }

            // --- NDA TED: average |z_interp|^2 * exp(-j*2*pi/Ns*kk) over kk=0..Ns-1 ---
            f32 c1r = 0.f, c1i = 0.f;
            for (int kk = 0; kk < Ns; ++kk) {
                int idx = nn + kk;
                f32 tr, ti;
                if (idx >= 1 && idx + 2 < n_total) {
                    farrow3(re[idx-1], re[idx], re[idx+1], re[idx+2],
                            im[idx-1], im[idx], im[idx+1], im[idx+2],
                            mu, tr, ti);
                } else {
                    tr = re[std::min(idx, n_total-1)];
                    ti = im[std::min(idx, n_total-1)];
                }
                f32 mag2  = tr*tr + ti*ti;
                // exp(-j*2*pi/Ns*kk) twiddle precomputed (depends only on Ns).
                c1r += mag2 * tcos[kk];
                c1i += mag2 * tsin[kk];
            }
            c1r *= inv_Ns;
            c1i *= inv_Ns;

            // Update circular smoothing buffer
            c1_buf_r[buf_ptr] = c1r;
            c1_buf_i[buf_ptr] = c1i;
            buf_ptr = (buf_ptr + 1 == buf_len) ? 0 : (buf_ptr + 1);

            // Smoothed c1 sum
            f32 sum_r = 0.f, sum_i = 0.f;
            for (int k = 0; k < buf_len; ++k) {
                sum_r += c1_buf_r[k];
                sum_i += c1_buf_i[k];
            }
            sum_r *= inv_buf_len;
            sum_i *= inv_buf_len;

            epsilon = inv_two_pi * std::atan2(sum_i, sum_r);

            // Store output
            if (mm < max_out) {
                out_r[mm] = zr;
                out_i[mm] = zi;
            }
            mm++;
        }

        // Loop filter
        f32 vp = K1 * epsilon;
        vi    += K2 * epsilon;
        f32 v  = vp + vi;
        f32 W  = 1.f / static_cast<f32>(Ns) + v;

        // Modulo-1 counter
        CNT_next = CNT - W;
        if (CNT_next < 0.f) {
            CNT_next  = 1.f + CNT_next;
            underflow = 1;
            mu_next   = CNT / W;
        } else {
            underflow = 0;
            mu_next   = mu;
        }
    }
    }  // end gil_scoped_release

    // Trim to actual output length (mm-1 valid symbols, index 1..mm-1)
    int out_len = mm - 1;
    auto result = py::array_t<c64>(out_len);
    auto ptr    = result.mutable_unchecked<1>();

    {
        py::gil_scoped_release nogil;

        // Compute std for normalisation (matches Python zz /= np.std(zz))
        f32 mean_r = 0.f, mean_i = 0.f;
        for (int i = 1; i <= out_len; ++i) { mean_r += out_r[i]; mean_i += out_i[i]; }
        mean_r /= out_len; mean_i /= out_len;
        f32 var = 0.f;
        for (int i = 1; i <= out_len; ++i) {
            f32 dr = out_r[i] - mean_r, di = out_i[i] - mean_i;
            var += dr*dr + di*di;
        }
        f32 std_val = std::sqrt(var / out_len);
        if (std_val < 1e-10f) std_val = 1.f;

        for (int i = 0; i < out_len; ++i)
            ptr(i) = c64((out_r[i+1]) / std_val, (out_i[i+1]) / std_val);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(nda_ted_ext, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        NDA symbol timing synchroniser — optimised pybind11 C++ extension.

        Algorithm: Rice (2009) NDA TED with Farrow cubic interpolator.

        Optimisations:
          - Farrow cubic via Horner's method  — fewer multiplies
          - SoA real/imag layout              — NEON vectorisation on ARM A9
          - No std::complex in inner loop
          - -O3 -ffast-math via setup.py

        Signature:
            nda_ted(symbols, sps, BnTs=0.01, zeta=0.707, L=2)
                -> np.ndarray[complex64]
    )pbdoc";

    m.def("nda_ted",
        [](py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
           int sps, f32 BnTs, f32 zeta, int L, bool prepend_first) {
            return nda_symb_sync(symbols, sps, L, BnTs, zeta, prepend_first);
        },
        py::arg("symbols"),
        py::arg("sps"),
        py::arg("BnTs")  = 0.01f,
        py::arg("zeta")  = 0.707f,
        py::arg("L")     = 2,
        py::arg("prepend_first") = false,
        "NDA symbol timing sync with Farrow cubic interpolation. "
        "prepend_first=True duplicates the first input sample internally to "
        "cancel the new-NDA 1-sample bias (saves a Python-side concatenate).");
}
