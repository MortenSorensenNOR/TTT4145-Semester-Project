// File: modules/gardner_ted/gardner_ext.cpp
//
// Gardner timing error detector C++ extension via pybind11.
//
// Optimisations applied:
//   - No complex exponentials — cubic interpolation uses only multiply/add
//   - SoA (struct-of-arrays) layout for NEON auto-vectorisation on ARM A9
//   - Catmull-Rom cubic interpolation split into real/imag to avoid std::complex overhead
//   - Horner's method for polynomial evaluation — fewer multiplies
//   - Compiled with -O3 -ffast-math on all platforms
//   - Additional -mcpu=cortex-a9 -mfpu=neon flags applied on ARM (see gardner_setup.py)

#include <cmath>
#include <complex>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using c64 = std::complex<float>;
using f32 = float;

// ---------------------------------------------------------------------------
// Catmull-Rom cubic interpolation via Horner's method.
// Real and imag computed separately — avoids std::complex overhead,
// enables NEON auto-vectorisation on A9.
// ---------------------------------------------------------------------------

static inline f32 cubic_interp_f32(
    f32 v0, f32 v1, f32 v2, f32 v3, f32 mu)
{
    f32 c0 =  v1;
    f32 c1 = -0.5f*v0 + 0.5f*v2;
    f32 c2 =  v0 - 2.5f*v1 + 2.0f*v2 - 0.5f*v3;
    f32 c3 = -0.5f*v0 + 1.5f*v1 - 1.5f*v2 + 0.5f*v3;
    return c0 + mu*(c1 + mu*(c2 + mu*c3));
}

static inline void interp_cx(
    const f32* re, const f32* im, int idx, f32 mu, int n,
    f32& out_re, f32& out_im)
{
    if (idx < 1 || idx + 2 >= n) {
        int c  = idx < 0 ? 0 : (idx >= n ? n-1 : idx);
        out_re = re[c];
        out_im = im[c];
        return;
    }
    out_re = cubic_interp_f32(re[idx-1], re[idx], re[idx+1], re[idx+2], mu);
    out_im = cubic_interp_f32(im[idx-1], im[idx], im[idx+1], im[idx+2], mu);
}

// ---------------------------------------------------------------------------
// Gardner TED
//
// Error detector:
//   e = Re{ (y[k] - y[k-T]) * conj(y[k-T/2]) }
//
// Loop filter (type-1, proportional only):
//   mu  = clamp(mu + gain * e,  -0.5, 0.5)
//
// Timing advance:
//   k  += sps + mu
// ---------------------------------------------------------------------------

py::array_t<c64> gardner_ted(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
    int sps,
    f32 gain)
{
    auto in = symbols.unchecked<1>();
    int  n  = static_cast<int>(in.shape(0));

    // SoA split — lets the compiler vectorise inner multiplies with NEON
    std::vector<f32> re(n), im(n);
    for (int i = 0; i < n; ++i) {
        re[i] = in(i).real();
        im[i] = in(i).imag();
    }

    std::vector<c64> out;
    out.reserve(n / sps + 2);

    f32 mu = 0.0f;
    f32 k  = static_cast<f32>(sps);

    while (true) {
        int   k_int     = static_cast<int>(k);
        f32   k_mid_f   = k - sps * 0.5f;
        int   k_mid_int = static_cast<int>(k_mid_f);
        int   k_prev    = k_int - sps;

        if (k_int + 2 >= n || k_mid_int < 0 || k_prev < 0)
            break;

        f32 cr, ci, mr, mi, pr, pi;
        interp_cx(re.data(), im.data(), k_int,     k     - k_int,         n, cr, ci);
        interp_cx(re.data(), im.data(), k_mid_int, k_mid_f - k_mid_int,   n, mr, mi);
        interp_cx(re.data(), im.data(), k_prev,    k     - k_int,         n, pr, pi);

        // Re{ (curr - prev) * conj(mid) }  =  (cr-pr)*mr + (ci-pi)*mi
        f32 dr = cr - pr;
        f32 di = ci - pi;
        f32 e  = dr * mr + di * mi;

        mu += gain * e;
        if      (mu >  0.5f) mu =  0.5f;
        else if (mu < -0.5f) mu = -0.5f;

        k += static_cast<f32>(sps) + mu;

        out.emplace_back(cr, ci);
    }

    auto result = py::array_t<c64>(static_cast<py::ssize_t>(out.size()));
    auto ptr    = result.mutable_unchecked<1>();
    for (size_t i = 0; i < out.size(); ++i)
        ptr(i) = out[i];

    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(gardner_ext, m) {
    m.doc() = R"pbdoc(
        Gardner timing error detector — optimised pybind11 C++ extension.

        Optimisations:
          - SoA real/imag layout     — enables NEON auto-vectorisation on ARM A9
          - Horner's method          — fewer multiplies in cubic interpolation
          - No std::complex in loop  — avoids ABI overhead on A9
          - No complex exponentials  — pure multiply/add in hot path
          - -O3 -ffast-math          — all platforms

        Signature:
            gardner_ted(symbols, sps, gain=0.001) -> np.ndarray[complex64]
    )pbdoc";

    m.def("gardner_ted", &gardner_ted,
        py::arg("symbols"),
        py::arg("sps"),
        py::arg("gain") = 0.001f,
        "Gardner TED with Catmull-Rom cubic interpolation");
}
