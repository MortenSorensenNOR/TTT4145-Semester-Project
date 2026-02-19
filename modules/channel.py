"""Channel Model for Radio Communication Simulation."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def pluto_max_cfo(carrier_hz: float = 433e6, num_devices: int = 2) -> float:
    """Calculate maximum CFO for Pluto SDR (25 ppm oscillator)."""
    ppm = 25e-6  # 25 ppm oscillator tolerance
    return ppm * carrier_hz * num_devices


def delay_ns_to_samples(delay_ns: float, sample_rate: float) -> float:
    """Convert delay from nanoseconds to samples."""
    return delay_ns * 1e-9 * sample_rate


def samples_to_delay_ns(delay_samples: float, sample_rate: float) -> float:
    """Convert delay from samples to nanoseconds."""
    return delay_samples / sample_rate * 1e9


# Speed of light in meters per second
SPEED_OF_LIGHT = 299_792_458.0


def distance_to_delay(distance_m: float) -> float:
    """Calculate propagation delay given a distance."""
    return distance_m / SPEED_OF_LIGHT


# Magic number constants for channel processing
FRACTIONAL_DELAY_THRESHOLD = 1e-9
FRACTIONAL_DELAY_LINEAR_THRESHOLD = 1e-6
SCO_THRESHOLD = 1e-9


@dataclass
class ChannelConfig:
    """Configuration for the channel model.

    Features activate automatically based on their parameter values:
    - CFO: active when cfo_hz != 0
    - Phase offset: active when initial_phase_rad != 0 or phase_drift_hz != 0
    - Delay: active when delay_samples > 0
    - SCO: active when abs(sco_ppm) > threshold
    - Fading: active when doppler_hz > 0 (requires enable_multipath)

    Explicit enable flags are kept only for features that have no natural
    "off" value: multipath and phase noise.
    """

    # Basic parameters
    sample_rate: float = 1e6
    snr_db: float = 20.0
    reference_power: float = 1.0  # Reference signal power for SNR calculation

    # Multipath configuration
    enable_multipath: bool = False
    multipath_delays_samples: tuple[float, ...] = (0.0,)
    multipath_gains_db: tuple[float, ...] = (0.0,)

    # Fading configuration (only applies when enable_multipath is True)
    doppler_hz: float = 0.0
    fading_type: str = "rayleigh"  # "rayleigh" or "rician"
    rician_k_db: float = 10.0  # Rician K-factor in dB

    # CFO configuration (active when cfo_hz != 0)
    cfo_hz: float = 0.0

    # Phase offset configuration (active when initial_phase_rad or phase_drift_hz != 0)
    initial_phase_rad: float = 0.0
    phase_drift_hz: float = 0.0

    # Propagation delay configuration (active when delay_samples > 0)
    delay_samples: float = 0.0

    # Phase noise configuration
    enable_phase_noise: bool = False
    phase_noise_psd_dbchz: float = -100.0  # Phase noise PSD in dBc/Hz at 1 kHz offset

    # Sample clock offset configuration (active when sco_ppm != 0)
    sco_ppm: float = 0.0

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.multipath_delays_samples) != len(self.multipath_gains_db):
            msg = "multipath_delays_samples and multipath_gains_db must have same length"
            raise ValueError(msg)
        if self.fading_type not in ("rayleigh", "rician"):
            msg = "fading_type must be 'rayleigh' or 'rician'"
            raise ValueError(msg)


@dataclass
class StreamState:
    """Maintains state for streaming/block-by-block processing."""

    sample_index: int = 0

    # Fading state (for sum-of-sinusoids model)
    fading_phases: NDArray[np.float64] | None = None

    # Delay buffer for fractional delay filter
    delay_buffer: NDArray[np.complex128] | None = None

    # Multipath delay line buffer
    multipath_buffer: NDArray[np.complex128] | None = None

    # Phase noise state
    phase_noise_state: float = 0.0

    # SCO state (for resampling)
    sco_phase: float = 0.0
    sco_buffer: NDArray[np.complex128] | None = None

    # Random number generator
    rng: np.random.Generator | None = None


def _apply_integer_delay(
    x: NDArray[np.complex128],
    int_delay: int,
    buffer: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Apply integer-sample delay and return a shifted signal with same length."""
    n_samples = len(x)
    y = np.zeros(n_samples, dtype=x.dtype)

    if buffer is not None and len(buffer) > 0:
        buffer_samples_needed = min(int_delay, len(buffer), n_samples)
        if buffer_samples_needed > 0:
            y[:buffer_samples_needed] = buffer[-buffer_samples_needed:]

    if int_delay < n_samples:
        samples_to_copy = n_samples - int_delay
        y[int_delay:] = x[:samples_to_copy]

    return y


def _lagrange_coefficients(frac_delay: float, order: int = 4) -> NDArray[np.float64]:
    """Calculate Lagrange interpolation coefficients for a fractional delay.

    Source: https://en.wikipedia.org/wiki/Lagrange_polynomial
    """
    n_taps = order + 1
    coefficients = np.zeros(n_taps)
    for k in range(n_taps):
        coefficients[k] = 1.0
        for m in range(n_taps):
            if m != k:
                coefficients[k] *= (frac_delay - (m - order // 2)) / (k - m)
    return coefficients


def _build_delay_buffer(
    x: NDArray[np.complex128],
    int_delay: int,
    buffer: NDArray[np.complex128] | None,
) -> NDArray[np.complex128]:
    """Create the updated delay buffer for block-based processing."""
    n_samples = len(x)
    buffer_size = max(int_delay + 5, 5)
    new_buffer = np.zeros(buffer_size, dtype=x.dtype)
    if n_samples >= buffer_size:
        new_buffer[:] = x[-buffer_size:]
        return new_buffer

    if buffer is not None and len(buffer) >= buffer_size - n_samples:
        new_buffer[: buffer_size - n_samples] = buffer[-(buffer_size - n_samples) :]
    new_buffer[buffer_size - n_samples :] = x
    return new_buffer


def apply_fractional_delay(
    x: NDArray[np.complex128],
    delay_samples: float,
    buffer: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Apply fractional sample delay using Lagrange interpolation."""
    if delay_samples == 0.0:
        new_buf = buffer if buffer is not None else np.zeros(4, dtype=x.dtype)
        return x.copy(), cast("NDArray[np.complex128]", new_buf)

    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    y = _apply_integer_delay(x, int_delay, buffer)

    # Apply fractional delay using Lagrange interpolation if needed
    if frac_delay > FRACTIONAL_DELAY_THRESHOLD:
        h = _lagrange_coefficients(frac_delay)
        n_taps = len(h)

        # Apply fractional delay filter
        # Pad to handle filter edge effects
        y_padded = np.concatenate([np.zeros(n_taps - 1, dtype=x.dtype), y])
        y_filtered = signal.lfilter(h, 1.0, y_padded)
        y = cast("NDArray[np.complex128]", y_filtered[n_taps - 1 : n_taps - 1 + len(x)])

    new_buffer = _build_delay_buffer(x, int_delay, buffer)

    return y, new_buffer


def apply_multipath(
    x: NDArray[np.complex128],
    delays_samples: tuple[float, ...],
    gains_linear: NDArray[np.float64],
    fading_gains: NDArray[np.complex128] | None = None,
    buffer: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Apply multipath channel using tapped delay line."""
    n_samples = len(x)
    max_delay = int(np.ceil(max(delays_samples))) + 1

    # Initialize buffer
    if buffer is None:
        buffer = np.zeros(max_delay, dtype=x.dtype)
    elif len(buffer) < max_delay:
        new_buffer = np.zeros(max_delay, dtype=x.dtype)
        new_buffer[: len(buffer)] = buffer
        buffer = new_buffer

    # Extend input with buffer
    x_extended = np.concatenate([buffer, x])

    # Apply tapped delay line
    y = np.zeros(n_samples, dtype=x.dtype)
    for i, (delay, gain) in enumerate(zip(delays_samples, gains_linear, strict=True)):
        int_delay = int(np.floor(delay))
        frac_delay = delay - int_delay

        # Get delayed samples
        if frac_delay < FRACTIONAL_DELAY_LINEAR_THRESHOLD:
            # Integer delay - simple indexing
            delayed = x_extended[max_delay - int_delay : max_delay - int_delay + n_samples]
        else:
            # Fractional delay - linear interpolation for simplicity
            idx = max_delay - int_delay
            delayed = (1 - frac_delay) * x_extended[idx : idx + n_samples] + frac_delay * x_extended[
                idx - 1 : idx - 1 + n_samples
            ]

        # Apply gain (static or time-varying)
        if fading_gains is not None:
            y += gain * fading_gains[i, :] * delayed
        else:
            y += gain * delayed

    # Update buffer
    new_buffer = x_extended[-max_delay:]

    return y, new_buffer


def generate_fading_gains(
    config: ChannelConfig,
    n_samples: int,
    phases: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Generate time-varying fading gains using Clarke's sum-of-sinusoids model.

    Source: https://en.wikipedia.org/wiki/Rayleigh_fading#Clarke's_model
    """
    if rng is None:
        rng = np.random.default_rng()

    n_taps = len(config.multipath_delays_samples)
    rician_k_linear = 10 ** (config.rician_k_db / 10)
    n_sinusoids = 16  # Number of sinusoids in sum-of-sinusoids model

    # Initialize phases if not provided
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, (n_taps, n_sinusoids, 2))  # 2 for I/Q

    t = np.arange(n_samples) / config.sample_rate

    gains = np.zeros((n_taps, n_samples), dtype=complex)

    for tap in range(n_taps):
        inphase = np.zeros(n_samples)
        quadrature = np.zeros(n_samples)
        for k in range(n_sinusoids):
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            freq_k = config.doppler_hz * np.cos(alpha_k)
            inphase += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 0])
            quadrature += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 1])
        inphase /= np.sqrt(n_sinusoids)
        quadrature /= np.sqrt(n_sinusoids)

        if config.fading_type == "rician":
            los_amplitude = np.sqrt(rician_k_linear / (rician_k_linear + 1))
            scatter_amplitude = np.sqrt(1 / (rician_k_linear + 1))
            gains[tap] = los_amplitude + scatter_amplitude * (inphase + 1j * quadrature)
        else:
            gains[tap] = (inphase + 1j * quadrature) / np.sqrt(2)

    # Update phases for streaming continuity
    new_phases = phases.copy()
    phase_increment = 2 * np.pi * config.doppler_hz * n_samples / config.sample_rate
    for tap in range(n_taps):
        for k in range(n_sinusoids):
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            new_phases[tap, k, :] += phase_increment * np.cos(alpha_k)

    return gains, new_phases


def apply_cfo_and_phase(
    x: NDArray[np.complex128],
    config: ChannelConfig,
    sample_index: int,
) -> NDArray[np.complex128]:
    """Apply carrier frequency offset and phase offset."""
    n = len(x)
    t = np.arange(sample_index, sample_index + n) / config.sample_rate

    phase = config.initial_phase_rad + 2 * np.pi * config.cfo_hz * t + 2 * np.pi * config.phase_drift_hz * t

    return x * np.exp(1j * phase)


def apply_phase_noise(
    x: NDArray[np.complex128],
    psd_dbchz: float,
    sample_rate: float,
    state: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.complex128], float]:
    """Apply phase noise using a filtered random walk model."""
    n = len(x)

    # Convert PSD to variance per sample
    # Phase noise PSD: L(f) = sigma^2 / (pi * f^2) for random walk
    # At offset f0: L(f0) = 10^(psd_dbchz/10)
    # sigma^2 = L(f0) * pi * f0^2
    f_offset = 1000.0  # Reference offset frequency (1 kHz)
    l_f0 = 10 ** (psd_dbchz / 10)
    sigma_sq = l_f0 * np.pi * f_offset**2

    # Generate phase noise as integrated white noise (random walk)
    noise_variance = sigma_sq / sample_rate
    phase_increments = rng.normal(0, np.sqrt(noise_variance), n)

    # Integrate to get phase noise
    phase_noise = np.cumsum(phase_increments) + state

    # Apply phase noise
    y = x * np.exp(1j * phase_noise)

    return y, phase_noise[-1]


def apply_sco(
    x: NDArray[np.complex128],
    sco_ppm: float,
    phase: float,
    buffer: NDArray[np.complex128] | None = None,
) -> tuple[NDArray[np.complex128], float, NDArray[np.complex128]]:
    """Apply sample clock offset via resampling using Catmull-Rom interpolation.

    Source: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
    """
    if abs(sco_ppm) < SCO_THRESHOLD:
        buffer_out = buffer if buffer is not None else np.zeros(4, dtype=x.dtype)
        return x.copy(), phase, buffer_out

    # Resampling ratio
    ratio = 1.0 + sco_ppm * 1e-6

    # Initialize buffer for interpolation
    interp_order = 3
    if buffer is None:
        buffer = np.zeros(interp_order + 1, dtype=x.dtype)

    # Prepend buffer
    x_extended = np.concatenate([buffer, x])

    # Calculate output indices
    n_in = len(x)
    n_out = int(np.floor(n_in / ratio))

    # Output sample positions in input coordinates
    out_positions = phase + np.arange(n_out) * ratio

    # Vectorized cubic interpolation (Catmull-Rom)
    y = np.zeros(n_out, dtype=x.dtype)
    idx = np.floor(out_positions).astype(int) + interp_order
    frac = out_positions - np.floor(out_positions)

    valid = (idx >= interp_order) & (idx < len(x_extended) - 1)
    vi = idx[valid]
    vf = frac[valid]

    p0 = x_extended[vi - 1]
    p1 = x_extended[vi]
    p2 = x_extended[vi + 1]
    p3 = x_extended[np.minimum(vi + 2, len(x_extended) - 1)]

    term = 3 * (p1 - p2) + p3 - p0
    y[valid] = p1 + 0.5 * vf * (p2 - p0 + vf * (2 * p0 - 5 * p1 + 4 * p2 - p3 + vf * term))

    # Update phase for next block
    new_phase = (phase + n_out * ratio) % 1.0

    # Update buffer
    new_buffer = x_extended[-(interp_order + 1) :]

    return y, new_phase, new_buffer


def apply_awgn(
    x: NDArray[np.complex128],
    snr_db: float,
    rng: np.random.Generator,
    reference_power: float = 1.0,
) -> NDArray[np.complex128]:
    """Add AWGN to achieve specified SNR."""
    # Calculate noise power for desired SNR using reference power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = reference_power / snr_linear

    # Generate complex AWGN
    noise_std = np.sqrt(noise_power / 2)  # /2 for complex noise (I and Q)
    noise = noise_std * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))

    return x + noise


class ChannelModel:
    """Modular channel model for radio communication simulation.

    Supports both batch processing and streaming (block-by-block) modes.

    Processing order:
    1. Propagation delay (fractional sample delay)
    2. Multipath/fading (tapped delay line)
    3. CFO and phase offset (complex exponential rotation)
    4. Phase noise (optional)
    5. AWGN (added at receiver)
    6. SCO (optional resampling)
    """

    def __init__(self, config: ChannelConfig) -> None:
        """Initialize channel model."""
        self.config = config
        self._state: StreamState | None = None

        # Pre-compute linear gains from dB
        self._multipath_gains_linear = 10 ** (np.array(config.multipath_gains_db) / 20)

    def reset(self) -> None:
        """Reset channel state for new transmission."""
        self._state = None

    def _init_state(self) -> StreamState:
        """Initialize streaming state."""
        rng = np.random.default_rng(self.config.seed)

        return StreamState(
            sample_index=0,
            fading_phases=None,
            delay_buffer=None,
            multipath_buffer=None,
            phase_noise_state=0.0,
            sco_phase=0.0,
            sco_buffer=None,
            rng=rng,
        )

    def _get_state(self) -> StreamState:
        """Get or create streaming state."""
        if self._state is None:
            self._state = self._init_state()
        return self._state

    def apply(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply channel model to entire signal (batch mode)."""
        self.reset()
        return self.process_block(x)

    def process_block(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Process a single block of samples (streaming mode)."""
        state = self._get_state()
        y = x.astype(np.complex128)
        n_samples = len(x)
        cfg = self.config

        # 1. Propagation delay
        if cfg.delay_samples > 0:
            y, state.delay_buffer = apply_fractional_delay(
                y,
                cfg.delay_samples,
                state.delay_buffer,
            )

        # 2. Multipath and fading
        if cfg.enable_multipath:
            fading_gains = None
            if cfg.doppler_hz > 0:
                fading_gains, state.fading_phases = generate_fading_gains(
                    cfg,
                    n_samples,
                    state.fading_phases,
                    state.rng,
                )

            y, state.multipath_buffer = apply_multipath(
                y,
                cfg.multipath_delays_samples,
                self._multipath_gains_linear,
                fading_gains,
                state.multipath_buffer,
            )

        # 3. CFO and phase offset
        if cfg.cfo_hz != 0 or cfg.initial_phase_rad != 0 or cfg.phase_drift_hz != 0:
            y = apply_cfo_and_phase(y, cfg, state.sample_index)

        # 4. Phase noise
        if cfg.enable_phase_noise:
            if state.rng is None:
                state.rng = np.random.default_rng(cfg.seed)
            y, state.phase_noise_state = apply_phase_noise(
                y,
                cfg.phase_noise_psd_dbchz,
                cfg.sample_rate,
                state.phase_noise_state,
                state.rng,
            )

        # 5. AWGN
        if state.rng is None:
            state.rng = np.random.default_rng(cfg.seed)
        y = apply_awgn(y, cfg.snr_db, state.rng, cfg.reference_power)

        # 6. SCO
        if abs(cfg.sco_ppm) > SCO_THRESHOLD:
            y, state.sco_phase, state.sco_buffer = apply_sco(
                y,
                cfg.sco_ppm,
                state.sco_phase,
                state.sco_buffer,
            )

        # Update sample index
        state.sample_index += n_samples

        return y

    def __repr__(self) -> str:
        """Return string representation."""
        cfg = self.config
        parts = [f"ChannelModel(snr={cfg.snr_db}dB"]

        if cfg.enable_multipath:
            parts.append(f"multipath={len(cfg.multipath_delays_samples)} taps")

        if cfg.doppler_hz > 0:
            parts.append(f"fading={cfg.fading_type}")

        if cfg.cfo_hz != 0:
            parts.append(f"cfo={cfg.cfo_hz}Hz")

        if cfg.delay_samples > 0:
            parts.append(f"delay={cfg.delay_samples:.2f} samples")

        if cfg.enable_phase_noise:
            parts.append(f"phase_noise={cfg.phase_noise_psd_dbchz}dBc/Hz")

        if abs(cfg.sco_ppm) > SCO_THRESHOLD:
            parts.append(f"sco={cfg.sco_ppm}ppm")

        return ", ".join(parts) + ")"
