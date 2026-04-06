// File: modules/frame_sync_ext.cpp
//
// C++ pybind11 extension: coarse_sync and fine_timing for frame synchronization.
//
// coarse_sync  — Schmidl-Cox sliding-window metric with no cumsum allocation.
//                Clusters plateau indices to support multi-frame detection.
//                CFO estimate: angle(mean P(d)) * fs / (2π * L).
//
// fine_timing  — Direct cross-correlation (no FFT, no allocation in inner loop).
//                CFO correction via LUT sin/cos (same 1024-entry LUT as costas).
//                NEON auto-vectorises the 904-sample inner dot-product under -O3.
//
// Compiled with -O3 -ffast-math; ARM: also -mcpu=cortex-a9 -mfpu=neon (see setup.py).

#include <algorithm>
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
// Sin/cos LUT  (1024 entries = 8 KB, fits in A9 32 KB L1 data cache)
// ---------------------------------------------------------------------------

static constexpr int LUT_SIZE = 1024;
static f32 sin_lut[LUT_SIZE];
static f32 cos_lut[LUT_SIZE];

static void init_lut() {
    for (int i = 0; i < LUT_SIZE; ++i) {
        f32 a = (2.0f * f32(M_PI) * i) / LUT_SIZE;
        sin_lut[i] = std::sin(a);
        cos_lut[i] = std::cos(a);
    }
}

// ---------------------------------------------------------------------------
// Return types
// ---------------------------------------------------------------------------

struct CoarseResult {
    py::array_t<ssize_t> d_hats;
    py::array_t<f32>     cfo_hats;
    py::array_t<f32>     m_peaks;
};

struct FineResult {
    py::array_t<ssize_t> sample_idxs;
    py::array_t<f32>     peak_ratios;
    py::array_t<f32>     phase_estimates;
};

// ---------------------------------------------------------------------------
// coarse_sync
//
// Schmidl-Cox on a complex64 buffer (typically decimated to symbol rate).
//
//   P(d) = Σ_{m=0}^{L-1}  conj(s[d+m]) · s[d+m+L]
//   R(d) = Σ_{m=0}^{L-1}  |s[d+m+L]|²
//   M(d) = |P(d)|² / R(d)²
//
// Sliding window: O(N) time, no allocations per step.
// ---------------------------------------------------------------------------

CoarseResult coarse_sync_ext(
    py::array_t<c64, py::array::c_style | py::array::forcecast> samples_in,
    int fs,
    int sps,
    int short_nsym,
    int short_nreps,
    int long_nsym,
    f32 energy_floor,
    f32 detection_threshold,
    f32 energy_gate_fraction)
{
    auto s   = samples_in.unchecked<1>();
    int  N   = static_cast<int>(s.shape(0));
    int  L   = short_nsym * sps;   // window length (= sample_cnt in Python)
    int  min_gap = short_nsym * sps * short_nreps + long_nsym * sps;

    auto make_empty = [&]() -> CoarseResult {
        return {py::array_t<ssize_t>(0), py::array_t<f32>(0), py::array_t<f32>(0)};
    };

    int md_len = N - 2 * L + 1;   // number of valid d positions
    if (md_len <= 0) return make_empty();

    // ---- build P(d) and R(d) via sliding window ----

    std::vector<c64> p_d(md_len);
    std::vector<f32> r_d(md_len);

    c64 p(0.0f, 0.0f);
    f32 r = 0.0f;
    for (int m = 0; m < L; ++m) {
        p += std::conj(s(m)) * s(m + L);
        r += std::norm(s(m + L));   // |s[m+L]|^2
    }
    p_d[0] = p;
    r_d[0] = r;

    for (int d = 0; d < md_len - 1; ++d) {
        p += std::conj(s(d + L)) * s(d + 2 * L) - std::conj(s(d)) * s(d + L);
        r += std::norm(s(d + 2 * L)) - std::norm(s(d + L));
        p_d[d + 1] = p;
        r_d[d + 1] = r;
    }

    // ---- energy gate threshold ----

    f32 max_r = *std::max_element(r_d.begin(), r_d.end());
    // r_gate = -1 disables the gate (r_d[d] > -1 is always true)
    f32 r_gate = (energy_gate_fraction > 0.0f && max_r > 0.0f)
                     ? max_r * energy_gate_fraction
                     : -1.0f;

    // ---- compute M(d) and collect above-threshold indices ----

    std::vector<f32> m_d(md_len);
    std::vector<int> above;
    above.reserve(md_len / 4);

    for (int d = 0; d < md_len; ++d) {
        f32 r2   = r_d[d] * r_d[d];
        m_d[d]   = std::norm(p_d[d]) / std::max(r2, energy_floor);
        if (m_d[d] > detection_threshold && r_d[d] > r_gate)
            above.push_back(d);
    }

    if (above.empty()) return make_empty();

    // ---- cluster by gap > min_gap; take first index per cluster ----

    std::vector<int> splits = {0};
    for (int i = 1; i < static_cast<int>(above.size()); ++i) {
        if (above[i] - above[i - 1] > min_gap)
            splits.push_back(i);
    }
    splits.push_back(static_cast<int>(above.size()));  // sentinel

    int n_frames = static_cast<int>(splits.size()) - 1;

    auto out_d   = py::array_t<ssize_t>(n_frames);
    auto out_cfo = py::array_t<f32>(n_frames);
    auto out_m   = py::array_t<f32>(n_frames);
    auto pd      = out_d.mutable_unchecked<1>();
    auto pc      = out_cfo.mutable_unchecked<1>();
    auto pm      = out_m.mutable_unchecked<1>();

    for (int ci = 0; ci < n_frames; ++ci) {
        int s0 = splits[ci], s1 = splits[ci + 1];

        pd(ci) = static_cast<ssize_t>(above[s0]);   // first index in cluster

        c64  psum(0.0f, 0.0f);
        f32  m_peak = 0.0f;
        for (int j = s0; j < s1; ++j) {
            int d = above[j];
            psum += p_d[d];
            if (m_d[d] > m_peak) m_peak = m_d[d];
        }
        pm(ci) = m_peak;
        // CFO = angle(mean P) * fs / (2π * L)
        // angle(psum / cnt) == angle(psum)  (cnt > 0 is a positive real)
        pc(ci) = std::atan2(psum.imag(), psum.real())
                 * static_cast<f32>(fs)
                 / (2.0f * f32(M_PI) * static_cast<f32>(L));
    }

    return {out_d, out_cfo, out_m};
}

// ---------------------------------------------------------------------------
// fine_timing
//
// For each frame:
//   1. Compute start = clip(d_hat + nreps*L - margin, 0, N-window_len)
//   2. Extract window of length (2*margin + ref_len); apply CFO correction
//      via LUT sin/cos with accumulator (no repeated floor/fmod calls).
//   3. Direct sliding cross-correlation: valid_len × ref_len complex MACs.
//      The inner loop (ref_len ≈ 904 iterations) is NEON auto-vectorised.
//   4. Peak → sample_idx, channel_phase, peak_ratio.
//   5. Project phase to payload start for Costas initialisation.
// ---------------------------------------------------------------------------

FineResult fine_timing_ext(
    py::array_t<c64,    py::array::c_style | py::array::forcecast> samples_in,
    py::array_t<c64,    py::array::c_style | py::array::forcecast> long_ref_in,
    py::array_t<ssize_t,py::array::c_style | py::array::forcecast> d_hats_in,
    py::array_t<f32,    py::array::c_style | py::array::forcecast> cfo_hats_in,
    int fs,
    int sps,
    int short_nsym,
    int short_nreps,
    int long_margin_nsym)
{
    auto s       = samples_in.unchecked<1>();
    auto ref     = long_ref_in.unchecked<1>();
    auto d_hats  = d_hats_in.unchecked<1>();
    auto cfo_arr = cfo_hats_in.unchecked<1>();

    int N        = static_cast<int>(s.shape(0));
    int ref_len  = static_cast<int>(ref.shape(0));
    int n_frames = static_cast<int>(d_hats.shape(0));

    int samples_per_rep = short_nsym * sps;
    int sample_margin   = long_margin_nsym * sps;
    int window_len      = 2 * sample_margin + ref_len;
    int valid_len       = 2 * sample_margin + 1;    // == window_len - ref_len + 1

    // Precompute conjugate of long_ref (stays in L1 cache for inner loop)
    std::vector<c64> conj_ref(ref_len);
    for (int n = 0; n < ref_len; ++n)
        conj_ref[n] = std::conj(ref(n));

    auto out_sidx  = py::array_t<ssize_t>(n_frames);
    auto out_ratio = py::array_t<f32>(n_frames);
    auto out_phase = py::array_t<f32>(n_frames);
    auto p_sidx    = out_sidx.mutable_unchecked<1>();
    auto p_ratio   = out_ratio.mutable_unchecked<1>();
    auto p_phase   = out_phase.mutable_unchecked<1>();

    std::vector<c64> window(window_len);
    std::vector<c64> xcorr(valid_len);

    for (int i = 0; i < n_frames; ++i) {

        // ---- compute start (clip to valid range) ----

        int raw_start = static_cast<int>(d_hats(i))
                        + short_nreps * samples_per_rep
                        - sample_margin;
        int start = std::max(0, std::min(raw_start, N - window_len));

        // ---- CFO correction via LUT accumulator ----
        // phase[k] = phase_per_sample * (start + k)
        // Represented as fractional cycles in [0,1) to avoid repeated floor.

        f32 cfo            = cfo_arr(i);
        f32 phase_per_samp = -2.0f * f32(M_PI) * cfo / static_cast<f32>(fs);

        // Initial fractional cycle for sample index 'start'
        f32 phase0 = phase_per_samp * static_cast<f32>(start);
        f32 frac   = phase0 / (2.0f * f32(M_PI));
        frac       = frac - std::floor(frac);           // normalise to [0, 1)
        f32 dfrac  = phase_per_samp / (2.0f * f32(M_PI)); // step per sample

        for (int k = 0; k < window_len; ++k) {
            int lut_idx = static_cast<int>(frac * LUT_SIZE) & (LUT_SIZE - 1);
            f32 c = cos_lut[lut_idx];
            f32 ss = sin_lut[lut_idx];
            c64 samp = s(start + k);
            // corrected = samp * (c + j*ss)
            window[k] = c64(samp.real() * c - samp.imag() * ss,
                             samp.real() * ss + samp.imag() * c);
            frac += dfrac;
            if (frac >= 1.0f) frac -= 1.0f;
            else if (frac < 0.0f) frac += 1.0f;
        }

        // ---- direct sliding cross-correlation ----
        // xcorr[k] = Σ_n  window[k+n] · conj_ref[n]
        // Inner loop auto-vectorises with NEON under -O3 -ffast-math.

        const c64* wp = window.data();
        const c64* rp = conj_ref.data();
        for (int k = 0; k < valid_len; ++k) {
            c64 acc(0.0f, 0.0f);
            const c64* wk = wp + k;
            for (int n = 0; n < ref_len; ++n)
                acc += wk[n] * rp[n];
            xcorr[k] = acc;
        }

        // ---- find peak and mean magnitude ----

        f32 peak_mag = 0.0f;
        int peak_idx = 0;
        f32 mean_mag = 0.0f;
        for (int k = 0; k < valid_len; ++k) {
            f32 mag = std::abs(xcorr[k]);
            mean_mag += mag;
            if (mag > peak_mag) { peak_mag = mag; peak_idx = k; }
        }
        mean_mag /= static_cast<f32>(valid_len);

        // ---- compute outputs ----

        ssize_t sample_idx    = static_cast<ssize_t>(start + peak_idx);
        f32 channel_phase     = std::atan2(xcorr[peak_idx].imag(), xcorr[peak_idx].real());
        ssize_t payload_pos   = sample_idx + static_cast<ssize_t>(ref_len);

        // Project phase to payload start (same formula as Python)
        f32 phase_at_payload  = channel_phase
                                + 2.0f * f32(M_PI) * (cfo / static_cast<f32>(fs))
                                  * static_cast<f32>(payload_pos);

        // Python's % (2π) — always non-negative
        f32 two_pi   = 2.0f * f32(M_PI);
        f32 mod_phase = phase_at_payload
                        - std::floor(phase_at_payload / two_pi) * two_pi;

        p_sidx(i)  = sample_idx;
        p_ratio(i) = (mean_mag > 0.0f) ? peak_mag / mean_mag : 0.0f;
        p_phase(i) = mod_phase;
    }

    return {out_sidx, out_ratio, out_phase};
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(frame_sync_ext, m) {
    m.doc() = R"pbdoc(
        Frame synchronization C++ extension.

        coarse_sync — Schmidl-Cox coarse timing + CFO (sliding P/R, no cumsum).
        fine_timing — Direct cross-correlation with long preamble (no FFT).

        Both fall back transparently to Python if the .so is missing.
    )pbdoc";

    init_lut();

    py::class_<CoarseResult>(m, "CoarseResult")
        .def_readonly("d_hats",   &CoarseResult::d_hats)
        .def_readonly("cfo_hats", &CoarseResult::cfo_hats)
        .def_readonly("m_peaks",  &CoarseResult::m_peaks)
        .def("__iter__", [](const CoarseResult& r) {
            return py::iter(py::make_tuple(r.d_hats, r.cfo_hats, r.m_peaks));
        });

    py::class_<FineResult>(m, "FineResult")
        .def_readonly("sample_idxs",     &FineResult::sample_idxs)
        .def_readonly("peak_ratios",     &FineResult::peak_ratios)
        .def_readonly("phase_estimates", &FineResult::phase_estimates)
        .def("__iter__", [](const FineResult& r) {
            return py::iter(py::make_tuple(r.sample_idxs, r.peak_ratios, r.phase_estimates));
        });

    m.def("coarse_sync", &coarse_sync_ext,
        py::arg("samples"),
        py::arg("fs"),
        py::arg("sps"),
        py::arg("short_nsym"),
        py::arg("short_nreps"),
        py::arg("long_nsym"),
        py::arg("energy_floor"),
        py::arg("detection_threshold"),
        py::arg("energy_gate_fraction"),
        "Schmidl-Cox coarse timing + CFO estimation (sliding window, no allocations).");

    m.def("fine_timing", &fine_timing_ext,
        py::arg("samples"),
        py::arg("long_ref"),
        py::arg("d_hats"),
        py::arg("cfo_hats"),
        py::arg("fs"),
        py::arg("sps"),
        py::arg("short_nsym"),
        py::arg("short_nreps"),
        py::arg("long_margin_nsym"),
        "Fine timing via direct cross-correlation with long preamble (LUT CFO correction, NEON-friendly).");
}
