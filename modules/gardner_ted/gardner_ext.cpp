/*
 * gardner_ext.cpp
 * ---------------
 * Pybind11 C++ extension implementing the Gardner Timing Error Detector (TED)
 * for BPSK, QPSK, and 8-PSK modulation schemes at arbitrary samples-per-symbol.
 *
 * Gardner TED equation (non-data-aided, complex baseband):
 *
 *   BPSK :  e[m] = Re(z_mid) * ( Re(z_prev) − Re(z_curr) )
 *   QPSK :  e[m] = Re{ conj(z_mid) * (z_prev − z_curr) }
 *   8-PSK:  same as QPSK, normalised by |z_mid|
 *
 * where:
 *   z_curr = interpolated sample at the current symbol centre
 *   z_prev = interpolated sample at the previous symbol centre
 *   z_mid  = interpolated sample at the midpoint between z_prev and z_curr
 *             (i.e. sps/2 samples before z_curr)
 *
 * Arbitrary-SPS interpolation
 * ---------------------------
 * mu tracks the fractional timing offset in units of *one symbol period*.
 * A strobe counter fires once per sps samples; at each strobe we interpolate
 * both the on-time and mid-symbol samples using a 4-tap cubic Farrow filter,
 * compute the TED error, update the PI loop filter, and fold mu back into the
 * strobe counter so the next strobe fires at the corrected timing instant.
 *
 * Farrow cubic interpolator
 * -------------------------
 * 4-tap cubic Lagrange interpolation (Erup, Gardner & Harris 1993):
 *   p(eta) = c0 + eta*(c1 + eta*(c2 + eta*c3))   for eta in [0,1)
 * where the coefficients are derived from 4 surrounding raw samples.
 *
 * Loop filter (2nd-order PI):
 *   integrator += beta  * e
 *   mu         += alpha * e + integrator
 *   mu          = clamp(mu, -0.5, 0.5)   // symbol-period units
 *
 * Input contract:
 *   samples    – complex64, arbitrary SPS (>= 2)
 *   alpha      – proportional gain
 *   beta       – integral gain
 *   sps        – samples per symbol (integer >= 2)
 *   mu         – initial fractional timing offset, symbol-period units [-0.5, 0.5]
 *   integrator – initial PI integrator state
 *
 * Returns:
 *   out_symbols – complex64, one per symbol
 *   out_mu      – float32,   timing offset estimate per symbol (diagnostic)
 *
 * Build:
 *   uv run python setup.py build_ext --inplace
 *
 * References:
 *   F. M. Gardner, "A BPSK/QPSK Timing-Error Detector for Sampled Receivers,"
 *   IEEE Trans. Commun., vol. COM-34, pp. 423-429, May 1986.
 *
 *   L. Erup, F. M. Gardner, R. A. Harris, "Interpolation in Digital Modems—
 *   Part II: Implementation and Performance," IEEE Trans. Commun., 1993.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

using cx64 = std::complex<float>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Bounds-safe sample accessor (clamp to first/last sample at edges)
static inline cx64 xget(const cx64* x, py::ssize_t idx, py::ssize_t N)
{
    if (idx < 0)  return x[0];
    if (idx >= N) return x[N - 1];
    return x[idx];
}

// 4-tap cubic Farrow interpolator
// eta in [0,1): 0 -> x0, approaching 1 -> x1
static inline cx64 farrow_interp(cx64 xm1, cx64 x0, cx64 x1, cx64 x2, float eta)
{
    cx64 c0 =  x0;
    cx64 c1 = -cx64(1.0f/6.0f)*xm1 - cx64(0.5f)*x0  + x1           - cx64(1.0f/3.0f)*x2;
    cx64 c2 =  cx64(0.5f)*xm1      - x0              + cx64(0.5f)*x1;
    cx64 c3 = -cx64(1.0f/6.0f)*xm1 + cx64(0.5f)*x0  - cx64(0.5f)*x1 + cx64(1.0f/6.0f)*x2;
    return c0 + eta * (c1 + eta * (c2 + eta * c3));
}

// Interpolate at a (possibly negative or fractional) index relative to base
static inline cx64 interp_at(const cx64* x, py::ssize_t N,
                              py::ssize_t base, float frac_offset)
{
    // frac_offset may be outside [0,1); split into integer + fractional parts
    py::ssize_t ioff = static_cast<py::ssize_t>(std::floor(frac_offset));
    float eta = frac_offset - static_cast<float>(ioff);
    py::ssize_t n = base + ioff;
    return farrow_interp(
        xget(x, n - 1, N),
        xget(x, n,     N),
        xget(x, n + 1, N),
        xget(x, n + 2, N),
        eta
    );
}

// ---------------------------------------------------------------------------
// Modulation-specific timing error detectors
// ---------------------------------------------------------------------------

static inline float ted_bpsk(cx64 prev, cx64 mid, cx64 curr)
{
    // Original Gardner 1986 — real-only; correct for BPSK where Im carries no data.
    return mid.real() * (prev.real() - curr.real());
}

static inline float ted_qpsk(cx64 prev, cx64 mid, cx64 curr)
{
    // Complex Gardner: Re{ conj(mid) * (prev - curr) }
    // Exploits both I and Q branches equally.
    cx64 diff = prev - curr;
    return mid.real() * diff.real() + mid.imag() * diff.imag();
}

static inline float ted_8psk(cx64 prev, cx64 mid, cx64 curr)
{
    // Same complex form as QPSK, but amplitude-normalised.
    // 8-PSK has constant |z|, yet the inner-product projection onto the
    // error axis varies with symbol angle; normalisation keeps the S-curve
    // slope (TED gain) approximately constant regardless of which symbol is
    // being received.
    cx64  diff = prev - curr;
    float e    = mid.real() * diff.real() + mid.imag() * diff.imag();
    float ampl = std::abs(mid);
    return (ampl > 1e-6f) ? e / ampl : e;
}

// ---------------------------------------------------------------------------
// Core Gardner loop — templated on the TED function
// ---------------------------------------------------------------------------
template<float (*TED)(cx64, cx64, cx64)>
static std::pair<py::array_t<cx64>, py::array_t<float>>
gardner_loop(
    py::array_t<cx64, py::array::c_style | py::array::forcecast> samples,
    float alpha,
    float beta,
    int   sps,
    float mu,           // fractional timing offset, symbol-period units [-0.5, 0.5]
    float integrator    // PI integrator state
)
{
    if (sps < 2)
        throw std::runtime_error("gardner_loop: sps must be >= 2");

    auto buf = samples.request();
    if (buf.ndim != 1)
        throw std::runtime_error("gardner_loop: samples must be 1-D");

    const cx64*    x = static_cast<const cx64*>(buf.ptr);
    const py::ssize_t N = buf.shape[0];

    if (N < static_cast<py::ssize_t>(2 * sps))
        throw std::runtime_error("gardner_loop: need at least 2*sps input samples");

    // Reserve output vectors (upper bound)
    std::vector<cx64>  v_syms;  v_syms.reserve(N / sps + 2);
    std::vector<float> v_mu;    v_mu.reserve(N / sps + 2);

    // -----------------------------------------------------------------------
    // Strobe-based timing loop
    // -----------------------------------------------------------------------
    // `strobe` accumulates sample-domain progress.  We advance it by 1 per
    // input sample.  When strobe >= sps we have crossed a symbol boundary:
    //   - remainder strobe - sps is how far we have overrun
    //   - the interpolation fractional offset (in sample units) for the
    //     on-time sample is:  (strobe - sps) + mu*sps
    //
    // Using `float sps_f` for the threshold allows mu to shift the strobe
    // firing point continuously.
    // -----------------------------------------------------------------------

    const float sps_f = static_cast<float>(sps);
    float strobe = sps_f;   // fire immediately on first symbol

    cx64 prev_sym = xget(x, 0, N);   // bootstrap: treat sample 0 as previous on-time

    for (py::ssize_t i = 1; i < N; ++i)
    {
        strobe += 1.0f;
        if (strobe < sps_f)
            continue;   // not yet at symbol boundary

        // ----- Strobe fired -----
        strobe -= sps_f;

        // Fractional overshoot (samples past the nominal boundary) + mu adjustment
        // eta_on is measured relative to sample i (the sample that triggered the strobe)
        float eta_on = strobe + mu * sps_f;  // sample-domain fractional offset from i

        // On-time sample: interpolate at i + eta_on (in sample units)
        cx64 on_time = interp_at(x, N, i, eta_on);

        // Mid-symbol sample: sps/2 samples before on-time
        float eta_mid = eta_on - sps_f * 0.5f;
        cx64  mid_sym = interp_at(x, N, i, eta_mid);

        // ----- Timing error -----
        float e = TED(prev_sym, mid_sym, on_time);

        // ----- PI loop filter -----
        integrator += beta  * e;
        mu         += alpha * e + integrator;

        // Wrap mu to (-0.5, 0.5] — symbol-period units
        if      (mu >  0.5f) mu -= 1.0f;
        else if (mu < -0.5f) mu += 1.0f;

        // ----- Store -----
        prev_sym = on_time;
        v_syms.push_back(on_time);
        v_mu.push_back(mu);
    }

    // Copy to numpy arrays
    py::ssize_t K = static_cast<py::ssize_t>(v_syms.size());
    auto out_syms = py::array_t<cx64>(K);
    auto out_mu   = py::array_t<float>(K);
    std::copy(v_syms.begin(), v_syms.end(),
              static_cast<cx64*>(out_syms.request().ptr));
    std::copy(v_mu.begin(), v_mu.end(),
              static_cast<float*>(out_mu.request().ptr));

    return {out_syms, out_mu};
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(gardner_ext, m)
{
    m.doc() = R"(
Gardner Timing Error Detector (TED) — pybind11 C++ extension.

Second-order PI-filtered Gardner TED for BPSK, QPSK, and 8-PSK at arbitrary
samples-per-symbol (sps >= 2).  Sub-sample interpolation uses a 4-tap cubic
Farrow filter.

Functions
---------
gardner_loop_bpsk(samples, alpha, beta, sps, mu, integrator) -> (symbols, mu_track)
gardner_loop_qpsk(samples, alpha, beta, sps, mu, integrator) -> (symbols, mu_track)
gardner_loop_8psk(samples, alpha, beta, sps, mu, integrator) -> (symbols, mu_track)

Parameters
----------
samples    : np.ndarray[complex64]  — oversampled input (matched-filter output)
alpha      : float                  — proportional gain
beta       : float                  — integral gain
sps        : int                    — samples per symbol (>= 2)
mu         : float                  — initial fractional timing offset [-0.5, 0.5], symbol units
integrator : float                  — initial PI integrator state

Returns
-------
symbols   : np.ndarray[complex64]  — one timing-corrected sample per symbol
mu_track  : np.ndarray[float32]    — fractional timing offset per symbol (diagnostic)
)";

    m.def("gardner_loop_bpsk", &gardner_loop<ted_bpsk>,
          py::arg("samples"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("sps"),
          py::arg("mu")         = 0.0f,
          py::arg("integrator") = 0.0f,
          "Gardner TED for BPSK — real-only error detector.");

    m.def("gardner_loop_qpsk", &gardner_loop<ted_qpsk>,
          py::arg("samples"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("sps"),
          py::arg("mu")         = 0.0f,
          py::arg("integrator") = 0.0f,
          "Gardner TED for QPSK — complex (I+Q) error detector.");

    m.def("gardner_loop_8psk", &gardner_loop<ted_8psk>,
          py::arg("samples"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("sps"),
          py::arg("mu")         = 0.0f,
          py::arg("integrator") = 0.0f,
          "Gardner TED for 8-PSK — amplitude-normalised complex error detector.");
}

