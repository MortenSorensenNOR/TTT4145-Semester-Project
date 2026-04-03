#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;
using cf32 = std::complex<float>;

// ---------------- Catmull-Rom cubic interpolation ----------------
static inline float interp1(float v0, float v1, float v2, float v3, float mu) {
    float c0 = v1;
    float c1 = -0.5f*v0 + 0.5f*v2;
    float c2 = v0 - 2.5f*v1 + 2.0f*v2 - 0.5f*v3;
    float c3 = -0.5f*v0 + 1.5f*v1 - 1.5f*v2 + 0.5f*v3;
    return c0 + mu*(c1 + mu*(c2 + mu*c3));
}

static inline cf32 cubic_interp(const float* re, const float* im, int idx, float mu, int n) {
    if (idx < 1 || idx + 2 >= n) {
        int c = std::max(0, std::min(idx, n-1));
        return cf32(re[c], im[c]);
    }
    float r = interp1(re[idx-1], re[idx], re[idx+1], re[idx+2], mu);
    float i = interp1(im[idx-1], im[idx], im[idx+1], im[idx+2], mu);
    return cf32(r, i);
}

// ---------------- Gardner TED ----------------
py::array_t<cf32> gardner_ted(py::array signal_in, int sps, float gain) {
    py::array_t<cf32> arr(signal_in);  // Cast input to complex64
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Input must be 1-D");
    const int n = static_cast<int>(buf.size);
    const cf32* sig = static_cast<const cf32*>(buf.ptr);

    // Split real/imag planes
    std::vector<float> re(n), im(n);
    for (int i = 0; i < n; ++i) { re[i] = sig[i].real(); im[i] = sig[i].imag(); }

    std::vector<cf32> out;
    out.reserve(n / sps + 8);

    float k = static_cast<float>(sps); // start after first symbol
    float mu_acc = 0.0f;

    while (k < n - 1) {
        int k_int = static_cast<int>(k);
        float k_frac = k - k_int;

        int k_prev_int = k_int - sps;
        float mu_prev = k - sps - k_prev_int;

        int k_mid_int = static_cast<int>(k - sps*0.5f);
        float mu_mid = k - sps*0.5f - k_mid_int;

        // Interpolate
        cf32 curr = cubic_interp(re.data(), im.data(), k_int, k_frac, n);
        cf32 mid  = cubic_interp(re.data(), im.data(), k_mid_int, mu_mid, n);
        cf32 prev = cubic_interp(re.data(), im.data(), k_prev_int, mu_prev, n);

        // Gardner error
        float e = (curr.real() - prev.real())*mid.real() + (curr.imag() - prev.imag())*mid.imag();

        mu_acc = std::clamp(mu_acc + gain*e, -0.5f, 0.5f);

        out.push_back(curr);

        k += static_cast<float>(sps) + mu_acc;
    }

    py::array_t<cf32> result(out.size());
    cf32* dst = static_cast<cf32*>(result.request().ptr);
    std::copy(out.begin(), out.end(), dst);
    return result;
}

PYBIND11_MODULE(gardner_ext, m) {
    m.doc() = "Gardner TED C++ extension with Catmull-Rom interpolation";
    m.def("gardner_ted", &gardner_ted, py::arg("signal"), py::arg("sps"), py::arg("gain"));
}

