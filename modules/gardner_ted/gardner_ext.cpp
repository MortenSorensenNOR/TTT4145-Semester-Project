/*
 * gardner_ext.cpp
 * ---------------
 * Pybind11 C++ extension: Gardner Timing Error Detector for BPSK, QPSK, 8-PSK
 * at arbitrary samples-per-symbol (sps >= 2).
 *
 * Gardner TED (non-data-aided, complex baseband):
 *   BPSK :  e[m] = Re(z_mid) * ( Re(z_prev) - Re(z_curr) )
 *   QPSK :  e[m] = Re{ conj(z_mid) * (z_prev - z_curr) }
 *   8-PSK:  same as QPSK, normalised by |z_mid|
 *
 * Strobe model
 * ------------
 * strobe starts at (sps - 1).  It increments by 1 each input sample.
 * When strobe >= sps a symbol boundary has been crossed.
 *
 *   strobe pre-load = sps - 1
 *   i = 0  : strobe becomes sps   → fires, overshoot = 0
 *             on_time_pos = 0 - 0 + mu*sps = 0        (== simple decimation)
 *   i = sps: strobe becomes sps   → fires, overshoot = 0
 *             on_time_pos = sps                        (== simple decimation)
 *   ...and so on.
 *
 * With mu != 0 the interpolation point shifts by mu*sps samples from the
 * nominal position; the Farrow filter handles sub-sample accuracy.
 *
 * The mid-symbol point sits sps/2 samples before the on-time point.
 *
 * Bootstrap
 * ---------
 * The TED needs prev_sym.  On the very first strobe we record the on-time
 * sample as prev_sym but skip the TED error (have_prev = false).
 * From symbol[1] onward the loop runs normally.
 *
 * Loop filter (2nd-order PI):
 *   integrator += beta  * e
 *   mu         += alpha * e + integrator
 *   mu          = clamp(mu, -0.5, 0.5)   // symbol-period units
 *
 * Build:
 *   uv run python setup.py build_ext --inplace
 *
 * References:
 *   F. M. Gardner, IEEE Trans. Commun., vol. COM-34, pp. 423-429, May 1986.
 *   L. Erup, F. M. Gardner, R. A. Harris, IEEE Trans. Commun., 1993.
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

static inline cx64 xget(const cx64* x, py::ssize_t idx, py::ssize_t N)
{
    if (idx < 0)  return x[0];
    if (idx >= N) return x[N - 1];
    return x[idx];
}

// 4-tap cubic Farrow interpolator.
// eta in [0, 1): eta=0 → x0, eta→1 → x1.
static inline cx64 farrow(cx64 xm1, cx64 x0, cx64 x1, cx64 x2, float eta)
{
    cx64 c0 =  x0;
    cx64 c1 = -cx64(1.f/6.f)*xm1 - cx64(0.5f)*x0 +          x1 - cx64(1.f/3.f)*x2;
    cx64 c2 =  cx64(0.5f)   *xm1 -             x0 + cx64(0.5f)*x1;
    cx64 c3 = -cx64(1.f/6.f)*xm1 + cx64(0.5f)*x0  - cx64(0.5f)*x1 + cx64(1.f/6.f)*x2;
    return c0 + eta*(c1 + eta*(c2 + eta*c3));
}

// Interpolate at absolute (possibly fractional) sample position `pos`.
static inline cx64 interp(const cx64* x, py::ssize_t N, float pos)
{
    auto  n   = static_cast<py::ssize_t>(std::floor(pos));
    float eta = pos - static_cast<float>(n);
    return farrow(xget(x,n-1,N), xget(x,n,N), xget(x,n+1,N), xget(x,n+2,N), eta);
}

// ---------------------------------------------------------------------------
// Timing error detectors
// ---------------------------------------------------------------------------

static inline float ted_bpsk(cx64 prev, cx64 mid, cx64 curr)
{
    // Real-only — correct for BPSK where the imaginary axis carries no data.
    return mid.real() * (prev.real() - curr.real());
}

static inline float ted_qpsk(cx64 prev, cx64 mid, cx64 curr)
{
    // Re{ conj(mid) * (prev - curr) } — uses both I and Q branches.
    cx64 d = prev - curr;
    return mid.real()*d.real() + mid.imag()*d.imag();
}
/*
static inline float ted_8psk(cx64 prev, cx64 mid, cx64 curr)
{
    // Plain complex Gardner — identical to QPSK.
    // The variable TED gain across 8-PSK transition types averages out over
    // many symbols and does not prevent convergence.
    // Previous normalisations (/|mid| or /|diff|^2) produced unstable S-curves.
    cx64 d = prev - curr;
    return mid.real()*d.real() + mid.imag()*d.imag();
}
*/
static inline float ted_8psk(cx64 prev, cx64 mid, cx64 curr)
{
    // Same as QPSK but amplitude-normalised to stabilise TED gain across the
    // 8 constellation points whose projections onto the error axis vary.
    cx64  d    = prev - curr;
    float e    = mid.real()*d.real() + mid.imag()*d.imag();
    float ampl = std::abs(mid);
    return (ampl > 1e-6f) ? e / ampl : e;
}

// ---------------------------------------------------------------------------
// Core Gardner loop (templated on TED)
// ---------------------------------------------------------------------------
template<float (*TED)(cx64, cx64, cx64)>
static std::pair<py::array_t<cx64>, py::array_t<float>>
gardner_loop(
    py::array_t<cx64, py::array::c_style | py::array::forcecast> samples,
    float alpha,
    float beta,
    int   sps,
    float mu,           // fractional timing offset, symbol-period units [-0.5, 0.5]
    float integrator
)
{
    if (sps < 2)
        throw std::runtime_error("gardner_loop: sps must be >= 2");

    auto buf = samples.request();
    if (buf.ndim != 1)
        throw std::runtime_error("gardner_loop: samples must be 1-D");

    const cx64*       x = static_cast<const cx64*>(buf.ptr);
    const py::ssize_t N = buf.shape[0];

    if (N < static_cast<py::ssize_t>(2 * sps))
        throw std::runtime_error("gardner_loop: need at least 2*sps input samples");

    std::vector<cx64>  v_syms;  v_syms.reserve(N / sps + 2);
    std::vector<float> v_mu;    v_mu.reserve(N / sps + 2);

    const float sps_f = static_cast<float>(sps);

    // Pre-load strobe so the first fire occurs at i=0, matching simple
    // decimation (on-time positions: 0, sps, 2*sps, ...).
    float strobe   = sps_f - 1.0f;
    cx64  prev_sym = {};
    bool  have_prev = false;

    for (py::ssize_t i = 0; i < N; ++i)
    {
        strobe += 1.0f;

        if (strobe < sps_f)
            continue;

        // ---- Strobe fired ----
        strobe -= sps_f;

        // `strobe` is now the overshoot in samples (0 when perfectly on-time).
        // The nominal on-time absolute position is (i - overshoot); mu shifts
        // it by mu*sps samples.
        float on_time_pos = static_cast<float>(i) - strobe + mu * sps_f;
        float mid_pos     = on_time_pos - sps_f * 0.5f;

        cx64 on_time = interp(x, N, on_time_pos);
        cx64 mid_sym = interp(x, N, mid_pos);

        if (have_prev)
        {
            float e    = TED(prev_sym, mid_sym, on_time);
            integrator += beta  * e;
            mu         += alpha * e + integrator;
            if      (mu >  0.5f) mu -= 1.0f;
            else if (mu < -0.5f) mu += 1.0f;
        }

        prev_sym  = on_time;
        have_prev = true;
        v_syms.push_back(on_time);
        v_mu.push_back(mu);
    }

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

With mu=0 and no timing error the output exactly reproduces simple decimation
(x[0], x[sps], x[2*sps], ...).

Functions
---------
gardner_loop_bpsk(samples, alpha, beta, sps, mu=0, integrator=0) -> (symbols, mu_track)
gardner_loop_qpsk(samples, alpha, beta, sps, mu=0, integrator=0) -> (symbols, mu_track)
gardner_loop_8psk(samples, alpha, beta, sps, mu=0, integrator=0) -> (symbols, mu_track)

Parameters
----------
samples    : np.ndarray[complex64]  — oversampled input (matched-filter output)
alpha      : float                  — proportional gain
beta       : float                  — integral gain
sps        : int                    — samples per symbol (>= 2)
mu         : float                  — initial fractional timing offset [-0.5, 0.5] (symbol units)
integrator : float                  — initial PI integrator state

Returns
-------
symbols   : np.ndarray[complex64]  — one timing-corrected sample per symbol
mu_track  : np.ndarray[float32]    — fractional timing offset per symbol (diagnostic)
)";

    m.def("gardner_loop_bpsk", &gardner_loop<ted_bpsk>,
          py::arg("samples"), py::arg("alpha"), py::arg("beta"), py::arg("sps"),
          py::arg("mu") = 0.0f, py::arg("integrator") = 0.0f,
          "Gardner TED for BPSK — real-only error detector.");

    m.def("gardner_loop_qpsk", &gardner_loop<ted_qpsk>,
          py::arg("samples"), py::arg("alpha"), py::arg("beta"), py::arg("sps"),
          py::arg("mu") = 0.0f, py::arg("integrator") = 0.0f,
          "Gardner TED for QPSK — complex (I+Q) error detector.");

    m.def("gardner_loop_8psk", &gardner_loop<ted_8psk>,
          py::arg("samples"), py::arg("alpha"), py::arg("beta"), py::arg("sps"),
          py::arg("mu") = 0.0f, py::arg("integrator") = 0.0f,
          "Gardner TED for 8-PSK — amplitude-normalised complex error detector.");
}

