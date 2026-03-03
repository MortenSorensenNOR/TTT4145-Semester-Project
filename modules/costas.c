// File: modules/costas.c
#include <complex.h>
#include <math.h>

// M_PI is not standard C, so define it if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to get the sign of a float, returns -1.0, 0.0, or 1.0
static inline float sign(float x) {
    if (x > 0.0f) return 1.0f;
    if (x < 0.0f) return -1.0f;
    return 0.0f;
}

void costas_loop_bpsk(
    const float complex* symbols,
    int n_symbols,
    float alpha,
    float beta,
    float* current_phase_estimate,
    float* current_frequency_offset,
    float complex* corrected_symbols,
    float* phase_estimates) {

    float phase_estimate = *current_phase_estimate;
    float integrator = *current_frequency_offset;

    for (int i = 0; i < n_symbols; i++) {
        // 1. Correct phase
        corrected_symbols[i] = symbols[i] * cexpf(-I * phase_estimate);

        // 2. Calculate phase error for BPSK: error = Im{y'} * sign(Re{y'})
        float error = cimagf(corrected_symbols[i]) * sign(crealf(corrected_symbols[i]));

        // 3. Update loop filter
        integrator += beta * error;
        float proportional = alpha * error;

        // 4. Update phase estimate
        phase_estimate += proportional + integrator;
        
        // 5. Wrap phase
        phase_estimate = fmodf(phase_estimate + M_PI, 2 * M_PI) - M_PI;
        
        phase_estimates[i] = phase_estimate;
    }
    *current_phase_estimate = phase_estimate;
    *current_frequency_offset = integrator;
}

void costas_loop_qpsk(
    const float complex* symbols,
    int n_symbols,
    float alpha,
    float beta,
    float* current_phase_estimate,
    float* current_frequency_offset,
    float complex* corrected_symbols,
    float* phase_estimates) {

    float phase_estimate = *current_phase_estimate;
    float integrator = *current_frequency_offset;

    for (int i = 0; i < n_symbols; i++) {
        // 1. Correct phase
        corrected_symbols[i] = symbols[i] * cexpf(-I * phase_estimate);

        // 2. Calculate phase error for QPSK
        // error = Im{y'}*sign(Re{y'}) - Re{y'}*sign(Im{y'})
        float real_part = crealf(corrected_symbols[i]);
        float imag_part = cimagf(corrected_symbols[i]);
        float error = imag_part * sign(real_part) - real_part * sign(imag_part);

        // 3. Update loop filter
        integrator += beta * error;
        float proportional = alpha * error;

        // 4. Update phase estimate
        phase_estimate += proportional + integrator;

        // 5. Wrap phase
        phase_estimate = fmodf(phase_estimate + M_PI, 2 * M_PI) - M_PI;
        
        phase_estimates[i] = phase_estimate;
    }
    *current_phase_estimate = phase_estimate;
    *current_frequency_offset = integrator;
}


void costas_loop_8psk(
    const float complex* symbols,
    int n_symbols,
    float alpha,
    float beta,
    float* current_phase_estimate,
    float* current_frequency_offset,
    float complex* corrected_symbols,
    float* phase_estimates) {

    float phase_estimate = *current_phase_estimate;
    float integrator = *current_frequency_offset;

    for (int i = 0; i < n_symbols; i++) {
        // 1. Correct phase
        corrected_symbols[i] = symbols[i] * cexpf(-I * phase_estimate);

        // 2. 8PSK phase error detector (Mth-power method)
        // error = angle(y'^8) / 8
        float complex powered_sym = cpowf(corrected_symbols[i], 8);
        float error = cargf(powered_sym) / 8.0f;

        // 3. Update loop filter
        integrator += beta * error;
        float proportional = alpha * error;

        // 4. Update phase estimate
        phase_estimate += proportional + integrator;

        // 5. Wrap phase
        phase_estimate = fmodf(phase_estimate + M_PI, 2 * M_PI) - M_PI;

        phase_estimates[i] = phase_estimate;
    }
    *current_phase_estimate = phase_estimate;
    *current_frequency_offset = integrator;
}
