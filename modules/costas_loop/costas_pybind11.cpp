// File: modules/costas_loop/costas_pybind11.cpp
//
// Optimised Costas loop C++ extension via pybind11.
//
// Optimisations applied:
//   - LUT-based sin/cos (avoids libm on every symbol — critical on ARM A9)
//   - 8PSK: repeated multiply instead of std::pow (avoids log/exp)
//   - SoA (struct-of-arrays) layout in the inner loop for NEON auto-vectorisation
//   - Compiled with -O3 -ffast-math on all platforms
//   - Additional -mcpu=cortex-a9 -mfpu=neon flags applied on ARM (see costas_setup.py)

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
        f32 a = (2.0f * static_cast<f32>(M_PI) * i) / LUT_SIZE;
        sin_lut[i] = std::sin(a);
        cos_lut[i] = std::cos(a);
    }
}

// Map a phase in [-π, π] to a LUT index
static inline int phase_to_idx(f32 phase) {
    f32 norm = (phase + static_cast<f32>(M_PI)) / (2.0f * static_cast<f32>(M_PI));
    return static_cast<int>(norm * LUT_SIZE) & (LUT_SIZE - 1);
}

// Wrap phase to [-π, π]
static inline f32 wrap(f32 phase) {
    return std::fmod(phase + static_cast<f32>(M_PI),
                     2.0f * static_cast<f32>(M_PI))
           - static_cast<f32>(M_PI);
}

static inline f32 fsign(f32 x) { return (x > 0.0f) ? 1.0f : -1.0f; }

// ---------------------------------------------------------------------------
// Return type — unpackable as (corrected_symbols, phase_estimates)
// ---------------------------------------------------------------------------

struct LoopResult {
    py::array_t<c64> corrected_symbols;
    py::array_t<f32> phase_estimates;
};

// ---------------------------------------------------------------------------
// BPSK  —  error = Im{y'} * sign(Re{y'})
// ---------------------------------------------------------------------------

LoopResult costas_loop_bpsk(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
    f32 alpha, f32 beta,
    f32 phase_estimate = 0.0f,
    f32 integrator     = 0.0f)
{
    auto in = symbols.unchecked<1>();
    int  n  = static_cast<int>(in.shape(0));

    // SoA split — lets the compiler vectorise the correction multiply with NEON
    std::vector<f32> re(n), im(n);
    for (int i = 0; i < n; ++i) { re[i] = in(i).real(); im[i] = in(i).imag(); }

    std::vector<f32> out_re(n), out_im(n);
    auto out_phase = py::array_t<f32>(n);
    auto phase_ptr = out_phase.mutable_unchecked<1>();

    for (int i = 0; i < n; ++i) {
        int idx = phase_to_idx(phase_estimate);
        f32 cr  =  re[i] * cos_lut[idx] + im[i] * sin_lut[idx];
        f32 ci  =  -re[i] * sin_lut[idx] + im[i] * cos_lut[idx];

        f32 error       = ci * fsign(cr);
        integrator     += beta  * error;
        phase_estimate += alpha * error + integrator;
        phase_estimate  = wrap(phase_estimate);

        out_re[i] = cr; out_im[i] = ci; phase_ptr(i) = phase_estimate;
    }

    auto out_syms = py::array_t<c64>(n);
    auto syms_ptr = out_syms.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) syms_ptr(i) = c64(out_re[i], out_im[i]);
    return {std::move(out_syms), std::move(out_phase)};
}

// ---------------------------------------------------------------------------
// QPSK  —  error = Im{y'}*sign(Re{y'}) - Re{y'}*sign(Im{y'})
// ---------------------------------------------------------------------------

LoopResult costas_loop_qpsk(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
    f32 alpha, f32 beta,
    f32 phase_estimate = 0.0f,
    f32 integrator     = 0.0f)
{
    auto in = symbols.unchecked<1>();
    int  n  = static_cast<int>(in.shape(0));

    std::vector<f32> re(n), im(n);
    for (int i = 0; i < n; ++i) { re[i] = in(i).real(); im[i] = in(i).imag(); }

    std::vector<f32> out_re(n), out_im(n);
    auto out_phase = py::array_t<f32>(n);
    auto phase_ptr = out_phase.mutable_unchecked<1>();

    for (int i = 0; i < n; ++i) {
        int idx = phase_to_idx(phase_estimate);
        f32 cr  =  re[i] * cos_lut[idx] + im[i] * sin_lut[idx];
        f32 ci  = -re[i] * sin_lut[idx] + im[i] * cos_lut[idx];

        f32 error       = ci * fsign(cr) - cr * fsign(ci);
        integrator     += beta  * error;
        phase_estimate += alpha * error + integrator;
        phase_estimate  = wrap(phase_estimate);

        out_re[i] = cr; out_im[i] = ci; phase_ptr(i) = phase_estimate;
    }

    auto out_syms = py::array_t<c64>(n);
    auto syms_ptr = out_syms.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) syms_ptr(i) = c64(out_re[i], out_im[i]);
    return {std::move(out_syms), std::move(out_phase)};
}

// ---------------------------------------------------------------------------
// 8PSK  —  Mth-power error detector,  error = angle(y'^8) / 8
//
// std::pow(y, 8) replaced with 3 complex multiplies:
//   y^2 = y*y,  y^4 = y^2*y^2,  y^8 = y^4*y^4
// This avoids the log/exp path inside std::pow, which is ~200 cycles on A9.
// ---------------------------------------------------------------------------

LoopResult costas_loop_8psk(
    py::array_t<c64, py::array::c_style | py::array::forcecast> symbols,
    f32 alpha, f32 beta,
    f32 phase_estimate = 0.0f,
    f32 integrator     = 0.0f)
{
    auto in = symbols.unchecked<1>();
    int  n  = static_cast<int>(in.shape(0));

    std::vector<f32> re(n), im(n);
    for (int i = 0; i < n; ++i) { re[i] = in(i).real(); im[i] = in(i).imag(); }

    std::vector<f32> out_re(n), out_im(n);
    auto out_phase = py::array_t<f32>(n);
    auto phase_ptr = out_phase.mutable_unchecked<1>();

    for (int i = 0; i < n; ++i) {
        int idx = phase_to_idx(phase_estimate);
        f32 cr  =  re[i] * cos_lut[idx] + im[i] * sin_lut[idx];
        f32 ci  = -re[i] * sin_lut[idx] + im[i] * cos_lut[idx];

        c64 y(cr, ci);
        c64 y2 = y  * y;
        c64 y4 = y2 * y2;
        c64 y8 = y4 * y4;
        f32 error = std::arg(y8) / 8.0f;

        integrator     += beta  * error;
        phase_estimate += alpha * error + integrator;
        phase_estimate  = wrap(phase_estimate);

        out_re[i] = cr; out_im[i] = ci; phase_ptr(i) = phase_estimate;
    }

    auto out_syms = py::array_t<c64>(n);
    auto syms_ptr = out_syms.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) syms_ptr(i) = c64(out_re[i], out_im[i]);
    return {std::move(out_syms), std::move(out_phase)};
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(costas_ext, m) {
    m.doc() = R"pbdoc(
        Costas loop carrier phase recovery — optimised pybind11 C++ extension.

        Optimisations vs naive implementation:
          - LUT sin/cos           — avoids libm on every symbol
          - 8PSK y^8 via 3 muls  — no log/exp
          - SoA layout            — enables NEON auto-vectorisation on ARM
          - -O3 -ffast-math       — applied on all platforms (see costas_setup.py)

        Signature for all three functions:
            (symbols, alpha, beta, phase_estimate=0.0, integrator=0.0)
                -> LoopResult  (iterable as corrected_symbols, phase_estimates)
    )pbdoc";

    init_lut();

    py::class_<LoopResult>(m, "LoopResult")
        .def_readonly("corrected_symbols", &LoopResult::corrected_symbols)
        .def_readonly("phase_estimates",   &LoopResult::phase_estimates)
        .def("__iter__", [](const LoopResult& r) {
            return py::iter(py::make_tuple(r.corrected_symbols, r.phase_estimates));
        });

    m.def("costas_loop_bpsk", &costas_loop_bpsk,
        py::arg("symbols"), py::arg("alpha"), py::arg("beta"),
        py::arg("phase_estimate") = 0.0f, py::arg("integrator") = 0.0f,
        "BPSK Costas loop");

    m.def("costas_loop_qpsk", &costas_loop_qpsk,
        py::arg("symbols"), py::arg("alpha"), py::arg("beta"),
        py::arg("phase_estimate") = 0.0f, py::arg("integrator") = 0.0f,
        "QPSK Costas loop");

    m.def("costas_loop_8psk", &costas_loop_8psk,
        py::arg("symbols"), py::arg("alpha"), py::arg("beta"),
        py::arg("phase_estimate") = 0.0f, py::arg("integrator") = 0.0f,
        "8PSK Costas loop — Mth-power error detector");
}
