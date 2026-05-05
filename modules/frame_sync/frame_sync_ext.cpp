// File: modules/frame_sync/frame_sync_ext.cpp
//
// C++ pybind11 extension: post-FFT NCC + peak finding for the single-stage
// full-buffer cross-correlation detector (full_buffer_xcorr_sync).
//
// Compiled with -O3 -ffast-math; ARM: also -mcpu=cortex-a9 -mfpu=neon (see setup.py).

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace py = pybind11;
using c64 = std::complex<float>;
using f32 = float;
using f64 = double;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// xcorr_ncc_post — post-FFT NCC + peak finding for full_buffer_xcorr_sync.
//
// Replaces a chain of numpy ops (z_mag2 = z.real²+z.imag²; sig_pwr cumsum;
// NCC = z_mag2/(sig_energy·ref_energy); silent-region masking; threshold +
// per-cluster argmax + CFO half-window split + phase projection) with a
// single tight C++ pass.  Inputs:
//   z          : complex64[n_z]  — already convolved (samples ⊛ conj(ref[::-1]))
//   samples    : complex64[N]    — RX buffer (for sliding |s|² and CFO win)
//   preamble_ref   : complex64[n_ref]
//   threshold, fs, n_ref          — scalars
//   silent_frac                  — fraction of max sig_energy to call silent
// Returns parallel arrays (sample_idxs, ncc_peaks, cfo_hats, phase_estimates).
// ---------------------------------------------------------------------------

struct XCorrResult {
    py::array_t<ssize_t> sample_idxs;
    py::array_t<f32>     peak_ratios;
    py::array_t<f32>     phase_estimates;
    py::array_t<f32>     cfo_hats;
};

static XCorrResult xcorr_ncc_post(
    py::array_t<c64, py::array::c_style | py::array::forcecast> z_in,
    py::array_t<c64, py::array::c_style | py::array::forcecast> samples_in,
    py::array_t<c64, py::array::c_style | py::array::forcecast> preamble_ref_in,
    f32 threshold,
    int fs,
    f32 silent_frac)
{
    const c64* z   = z_in.data();
    const c64* sig = samples_in.data();
    const c64* ref = preamble_ref_in.data();

    const ssize_t n_z   = z_in.size();
    const ssize_t N     = samples_in.size();
    const ssize_t n_ref = preamble_ref_in.size();
    const ssize_t half  = n_ref / 2;

    auto empty = []() -> XCorrResult {
        return {
            py::array_t<ssize_t>(0),
            py::array_t<f32>(0),
            py::array_t<f32>(0),
            py::array_t<f32>(0),
        };
    };

    if (n_z == 0 || N < n_ref) return empty();

    // Local accumulators, copied into freshly-allocated py::array_t at the end.
    std::vector<ssize_t> idx_vec;
    std::vector<f32>     peak_vec;
    std::vector<f32>     phase_vec;
    std::vector<f32>     cfo_vec;
    bool early_exit = false;
    {
        py::gil_scoped_release nogil;

        // Reference energy.
        f64 ref_energy = 0.0;
        for (ssize_t k = 0; k < n_ref; ++k) {
            ref_energy += static_cast<f64>(ref[k].real()) * ref[k].real()
                        + static_cast<f64>(ref[k].imag()) * ref[k].imag();
        }
        if (ref_energy <= 0.0) early_exit = true;

        // Sliding |s|² windowed energy via running sum.
        std::vector<f32> sig_energy;
        f32 max_sig = 0.0f;
        if (!early_exit) {
            sig_energy.resize(n_z);
            f64 running = 0.0;
            for (ssize_t k = 0; k < n_ref; ++k) {
                running += static_cast<f64>(sig[k].real()) * sig[k].real()
                         + static_cast<f64>(sig[k].imag()) * sig[k].imag();
            }
            sig_energy[0] = static_cast<f32>(running);
            for (ssize_t i = 1; i < n_z; ++i) {
                running += static_cast<f64>(sig[i + n_ref - 1].real()) * sig[i + n_ref - 1].real()
                         + static_cast<f64>(sig[i + n_ref - 1].imag()) * sig[i + n_ref - 1].imag()
                         - static_cast<f64>(sig[i - 1].real()) * sig[i - 1].real()
                         - static_cast<f64>(sig[i - 1].imag()) * sig[i - 1].imag();
                sig_energy[i] = static_cast<f32>(running);
            }
            for (ssize_t i = 0; i < n_z; ++i) {
                if (sig_energy[i] > max_sig) max_sig = sig_energy[i];
            }
            if (max_sig <= 0.0f) early_exit = true;
        }

        if (early_exit) goto done;

        {
        const f32 silent_thr = silent_frac * max_sig;
        const f32 inv_ref_e = 1.0f / static_cast<f32>(ref_energy);

        // NCC + threshold check, building above-threshold list inline.
        std::vector<f32> ncc(n_z);
        std::vector<int> above;
        above.reserve(n_z / 32);
        for (ssize_t i = 0; i < n_z; ++i) {
            const f32 zr = z[i].real();
            const f32 zi = z[i].imag();
            const f32 zm2 = zr * zr + zi * zi;
            const f32 se  = sig_energy[i];
            f32 v = 0.0f;
            if (se >= silent_thr) {
                v = zm2 * inv_ref_e / std::max(se, std::numeric_limits<f32>::min());
            }
            ncc[i] = v;
            if (v > threshold) above.push_back(static_cast<int>(i));
        }

        if (above.empty()) goto done;

        // Cluster peaks: gaps > n_ref start a new cluster.
        // For each cluster keep the argmax NCC; project phase + CFO at peak.
        ssize_t cluster_argmax = above[0];
        f32     cluster_max    = ncc[above[0]];
        auto emit = [&](ssize_t peak) {
            if (peak + n_ref > N) return;  // window runs off the end
            // CFO via half-window split: window = samples · conj(preamble_ref),
            // p = vdot(window[:half], window[half:n_ref]) → CFO = angle(p)·fs/(π·n_ref)
            f64 p_re = 0.0, p_im = 0.0;
            for (ssize_t k = 0; k < half; ++k) {
                // a = sig[peak + k]      · conj(ref[k])
                // b = sig[peak + k+half] · conj(ref[k+half])
                // accumulate conj(a) · b
                const f32 ar = sig[peak + k].real();
                const f32 ai = sig[peak + k].imag();
                const f32 rkr = ref[k].real();
                const f32 rki = ref[k].imag();
                const f32 a_r = ar * rkr + ai * rki;
                const f32 a_i = ai * rkr - ar * rki;

                const f32 br = sig[peak + k + half].real();
                const f32 bi = sig[peak + k + half].imag();
                const f32 rkr2 = ref[k + half].real();
                const f32 rki2 = ref[k + half].imag();
                const f32 b_r = br * rkr2 + bi * rki2;
                const f32 b_i = bi * rkr2 - br * rki2;

                p_re += static_cast<f64>(a_r) * b_r + static_cast<f64>(a_i) * b_i;
                p_im += static_cast<f64>(a_r) * b_i - static_cast<f64>(a_i) * b_r;
            }
            const f32 cfo_hat = static_cast<f32>(std::atan2(p_im, p_re))
                              * static_cast<f32>(fs)
                              / (static_cast<f32>(M_PI) * static_cast<f32>(n_ref));

            const f32 phase_mid = std::atan2(z[peak].imag(), z[peak].real());
            f32 phase_at_payload = phase_mid
                + 2.0f * static_cast<f32>(M_PI) * cfo_hat / static_cast<f32>(fs)
                * (static_cast<f32>(n_ref) / 2.0f);
            const f32 two_pi = 2.0f * static_cast<f32>(M_PI);
            phase_at_payload -= std::floor(phase_at_payload / two_pi) * two_pi;

            idx_vec.push_back(peak);
            peak_vec.push_back(ncc[peak]);
            phase_vec.push_back(phase_at_payload);
            cfo_vec.push_back(cfo_hat);
        };

        for (size_t i = 1; i < above.size(); ++i) {
            if (above[i] - above[i - 1] > n_ref) {
                emit(cluster_argmax);
                cluster_argmax = above[i];
                cluster_max = ncc[above[i]];
            } else if (ncc[above[i]] > cluster_max) {
                cluster_argmax = above[i];
                cluster_max = ncc[above[i]];
            }
        }
        emit(cluster_argmax);
        }  // end inner block

      done:;
    }

    if (idx_vec.empty()) return empty();

    const ssize_t n = static_cast<ssize_t>(idx_vec.size());
    auto out_idx   = py::array_t<ssize_t>(n);
    auto out_peak  = py::array_t<f32>(n);
    auto out_phase = py::array_t<f32>(n);
    auto out_cfo   = py::array_t<f32>(n);
    auto pi  = out_idx.mutable_unchecked<1>();
    auto pp  = out_peak.mutable_unchecked<1>();
    auto pph = out_phase.mutable_unchecked<1>();
    auto pc  = out_cfo.mutable_unchecked<1>();
    for (ssize_t i = 0; i < n; ++i) {
        pi(i)  = idx_vec[i];
        pp(i)  = peak_vec[i];
        pph(i) = phase_vec[i];
        pc(i)  = cfo_vec[i];
    }
    return {out_idx, out_peak, out_phase, out_cfo};
}

PYBIND11_MODULE(frame_sync_ext, m, py::mod_gil_not_used()) {
    m.doc() = R"pbdoc(
        Frame synchronization C++ extension.

        xcorr_ncc_post — post-FFT NCC + peak finding for the single-stage
                         full-buffer cross-correlation detector.
    )pbdoc";

    py::class_<XCorrResult>(m, "XCorrResult")
        .def_readonly("sample_idxs",     &XCorrResult::sample_idxs)
        .def_readonly("peak_ratios",     &XCorrResult::peak_ratios)
        .def_readonly("phase_estimates", &XCorrResult::phase_estimates)
        .def_readonly("cfo_hats",        &XCorrResult::cfo_hats)
        .def("__iter__", [](const XCorrResult& r) {
            return py::iter(py::make_tuple(r.sample_idxs, r.peak_ratios,
                                            r.phase_estimates, r.cfo_hats));
        });

    m.def("xcorr_ncc_post", &xcorr_ncc_post,
        py::arg("z"), py::arg("samples"), py::arg("preamble_ref"),
        py::arg("threshold"), py::arg("fs"),
        py::arg("silent_frac") = 0.05f,
        "Post-FFT NCC + peak finding for full_buffer_xcorr_sync. "
        "Replaces the numpy chain (z²+sig_pwr cumsum+NCC+mask+peaks+CFO) "
        "with a single C++ pass.");
}
