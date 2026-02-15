"""Channel Model for Radio Communication Simulation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

# =============================================================================
# Helper Functions
# =============================================================================


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


# =============================================================================
# Configuration Classes
# =============================================================================

# Magic number constants for channel processing
FRACTIONAL_DELAY_THRESHOLD = 1e-9
FRACTIONAL_DELAY_LINEAR_THRESHOLD = 1e-6
SCO_THRESHOLD = 1e-9


class ChannelProfile(Enum):
    """Preset channel profiles for common scenarios."""

    IDEAL = "ideal"  # AWGN only
    INDOOR_SMALL = "indoor_small"  # Small office/home
    INDOOR_MEDIUM = "indoor_medium"  # Conference room, larger spaces
    URBAN_LOS = "urban_los"  # Outdoor with line-of-sight
    URBAN_NLOS = "urban_nlos"  # Outdoor, obstructed path


@dataclass
class ChannelConfig:
    """Configuration for the channel model."""

    # Basic parameters
    sample_rate: float = 1e6
    snr_db: float = 20.0
    reference_power: float = 1.0  # Reference signal power for SNR calculation

    # Multipath configuration
    enable_multipath: bool = False
    multipath_delays_samples: tuple[float, ...] = (0.0,)
    multipath_gains_db: tuple[float, ...] = (0.0,)

    # Fading configuration
    enable_fading: bool = False
    doppler_hz: float = 0.0
    fading_type: str = "rayleigh"  # "rayleigh" or "rician"
    rician_k_db: float = 10.0  # Rician K-factor in dB

    # CFO configuration
    enable_cfo: bool = False
    cfo_hz: float = 0.0

    # Phase offset configuration
    enable_phase_offset: bool = False
    initial_phase_rad: float = 0.0
    phase_drift_hz: float = 0.0  # Slow phase drift rate

    # Propagation delay configuration
    enable_delay: bool = False
    delay_samples: float = 0.0  # Fractional delay in samples

    # Phase noise configuration
    enable_phase_noise: bool = False
    phase_noise_psd_dbchz: float = -100.0  # Phase noise PSD in dBc/Hz at 1 kHz offset

    # Sample clock offset configuration
    enable_sco: bool = False
    sco_ppm: float = 0.0  # Sample clock offset in ppm

    # Reproducibility
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.multipath_delays_samples) != len(self.multipath_gains_db):
            msg = (
                "multipath_delays_samples and multipath_gains_db must have same length"
            )
            raise ValueError(msg)
        if self.fading_type not in ("rayleigh", "rician"):
            msg = "fading_type must be 'rayleigh' or 'rician'"
            raise ValueError(msg)


@dataclass
class StreamState:
    """Maintains state for streaming/block-by-block processing."""

    sample_index: int = 0
    phase_accumulator: float = 0.0

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


@dataclass(frozen=True)
class ProfileOverrides:
    """Optional overrides applied when building a profile configuration."""

    cfo_hz: float | None = None
    phase_offset_rad: float | None = None
    delay_samples: float | None = None


@dataclass(frozen=True)
class ProfileRequest:
    """Input parameters used to create a channel configuration from a profile."""

    sample_rate: float
    snr_db: float
    seed: int | None = None
    reference_power: float = 1.0  # Reference signal power for SNR calculation
    overrides: ProfileOverrides = field(default_factory=ProfileOverrides)


@dataclass(frozen=True)
class FadingParams:
    """Parameters for fading gain generation."""

    n_samples: int
    n_taps: int
    doppler_hz: float
    sample_rate: float
    fading_type: str
    rician_k_linear: float


@dataclass(frozen=True)
class CfoPhaseParams:
    """Parameters for CFO and phase rotation."""

    cfo_hz: float
    sample_rate: float
    initial_phase: float
    phase_drift_hz: float
    sample_index: int


# =============================================================================
# Profile Presets
# =============================================================================


def get_profile_config(
    profile: ChannelProfile,
    request: ProfileRequest,
) -> ChannelConfig:
    """Create a ChannelConfig from a preset profile."""
    # Get profile-specific configuration
    config_kwargs = _create_profile_config_kwargs(
        profile,
        cfo_hz=request.overrides.cfo_hz,
        phase_offset_rad=request.overrides.phase_offset_rad,
        delay_samples=request.overrides.delay_samples,
    )

    # Add basic parameters
    config_kwargs["sample_rate"] = request.sample_rate
    config_kwargs["snr_db"] = request.snr_db
    config_kwargs["seed"] = request.seed
    config_kwargs["reference_power"] = request.reference_power

    return ChannelConfig(**config_kwargs)


# =============================================================================
# Signal Processing Functions
# =============================================================================


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
    """Calculate Lagrange interpolation coefficients for a fractional delay."""
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
        return x.copy(), buffer if buffer is not None else np.zeros(4, dtype=x.dtype)

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
        lfilter_result = signal.lfilter(h, 1.0, y_padded)
        y_filtered = (
            lfilter_result[0] if isinstance(lfilter_result, tuple) else lfilter_result
        )
        y = y_filtered[n_taps - 1 : n_taps - 1 + len(x)]

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
            delayed = x_extended[
                max_delay - int_delay : max_delay - int_delay + n_samples
            ]
        else:
            # Fractional delay - linear interpolation for simplicity
            idx = max_delay - int_delay
            delayed = (1 - frac_delay) * x_extended[
                idx : idx + n_samples
            ] + frac_delay * x_extended[idx - 1 : idx - 1 + n_samples]

        # Apply gain (static or time-varying)
        if fading_gains is not None:
            y += gain * fading_gains[i, :] * delayed
        else:
            y += gain * delayed

    # Update buffer
    new_buffer = x_extended[-max_delay:]

    return y, new_buffer


def _create_profile_config_kwargs(
    profile: ChannelProfile,
    cfo_hz: float | None = None,
    phase_offset_rad: float | None = None,
    delay_samples: float | None = None,
) -> dict[str, Any]:
    """Create profile-specific configuration kwargs."""
    profile_configs: dict[ChannelProfile, dict[str, Any]] = {
        ChannelProfile.IDEAL: {},
        ChannelProfile.INDOOR_SMALL: {
            "enable_multipath": True,
            "multipath_delays_samples": (0.0, 5.0, 15.0),
            "multipath_gains_db": (0.0, -3.0, -8.0),
            "enable_fading": True,
            "doppler_hz": 5.0,
            "fading_type": "rayleigh",
        },
        ChannelProfile.INDOOR_MEDIUM: {
            "enable_multipath": True,
            "multipath_delays_samples": (0.0, 10.0, 25.0, 50.0),
            "multipath_gains_db": (0.0, -5.0, -10.0, -15.0),
            "enable_fading": True,
            "doppler_hz": 3.0,
            "fading_type": "rayleigh",
        },
        ChannelProfile.URBAN_LOS: {
            "enable_multipath": True,
            "multipath_delays_samples": (0.0, 3.0),
            "multipath_gains_db": (0.0, -8.0),
            "enable_fading": False,
        },
        ChannelProfile.URBAN_NLOS: {
            "enable_multipath": True,
            "multipath_delays_samples": (0.0, 5.0, 15.0, 30.0),
            "multipath_gains_db": (-5.0, -8.0, -15.0, -20.0),
            "enable_fading": True,
            "doppler_hz": 10.0,
            "fading_type": "rician",
            "rician_k_db": 7.0,
        },
    }

    config_kwargs = profile_configs[profile].copy()

    # Apply overrides if specified
    if cfo_hz is not None:
        config_kwargs["enable_cfo"] = True
        config_kwargs["cfo_hz"] = cfo_hz

    if phase_offset_rad is not None:
        config_kwargs["enable_phase_offset"] = True
        config_kwargs["initial_phase_rad"] = phase_offset_rad

    if delay_samples is not None:
        config_kwargs["enable_delay"] = True
        config_kwargs["delay_samples"] = delay_samples

    return config_kwargs


def generate_fading_gains(
    params: FadingParams,
    phases: NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Generate time-varying fading gains using Clarke's sum-of-sinusoids model."""
    if rng is None:
        rng = np.random.default_rng()

    n_sinusoids = 16  # Number of sinusoids in sum-of-sinusoids model

    # Initialize phases if not provided
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, (params.n_taps, n_sinusoids, 2))  # 2 for I/Q

    t = np.arange(params.n_samples) / params.sample_rate
    gains = np.zeros((params.n_taps, params.n_samples), dtype=np.complex128)

    for tap in range(params.n_taps):
        # Sum of sinusoids for I and Q components
        inphase = np.zeros(params.n_samples)
        quadrature = np.zeros(params.n_samples)

        for k in range(n_sinusoids):
            # Jakes model: frequencies uniformly distributed in Doppler spectrum
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            freq_k = params.doppler_hz * np.cos(alpha_k)

            # I component
            inphase += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 0])
            # Q component
            quadrature += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 1])

        # Normalize
        inphase /= np.sqrt(n_sinusoids)
        quadrature /= np.sqrt(n_sinusoids)

        if params.fading_type == "rician":
            # Add LOS component
            k_factor = params.rician_k_linear
            los_amplitude = np.sqrt(k_factor / (k_factor + 1))
            scatter_amplitude = np.sqrt(1 / (k_factor + 1))
            gains[tap, :] = los_amplitude + scatter_amplitude * (
                inphase + 1j * quadrature
            )
        else:
            # Pure Rayleigh
            gains[tap, :] = (inphase + 1j * quadrature) / np.sqrt(2)

    # Update phases for streaming continuity
    phase_increment = (
        2 * np.pi * params.doppler_hz * params.n_samples / params.sample_rate
    )
    new_phases = phases.copy()
    for tap in range(params.n_taps):
        for k in range(n_sinusoids):
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            new_phases[tap, k, :] += phase_increment * np.cos(alpha_k)

    return gains, new_phases


def apply_cfo_and_phase(
    x: NDArray[np.complex128],
    params: CfoPhaseParams,
) -> tuple[NDArray[np.complex128], float]:
    """Apply carrier frequency offset and phase offset."""
    n = len(x)
    t = (np.arange(params.sample_index, params.sample_index + n)) / params.sample_rate

    # Total phase: initial + CFO + drift
    phase = (
        params.initial_phase
        + 2 * np.pi * params.cfo_hz * t
        + 2 * np.pi * params.phase_drift_hz * t
    )

    y = x * np.exp(1j * phase)

    # Return final phase for continuity (modulo 2*pi for numerical stability)
    final_phase = phase[-1] % (2 * np.pi)

    return y, final_phase


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
    """Apply sample clock offset via resampling."""
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

    # Cubic interpolation
    y = np.zeros(n_out, dtype=x.dtype)
    for i, pos in enumerate(out_positions):
        idx = int(np.floor(pos)) + interp_order
        if idx < interp_order or idx >= len(x_extended) - 1:
            continue

        frac = pos - np.floor(pos)

        # Cubic interpolation (Catmull-Rom)
        p0 = x_extended[idx - 1]
        p1 = x_extended[idx]
        p2 = x_extended[idx + 1]
        p3 = x_extended[idx + 2] if idx + 2 < len(x_extended) else p2

        term = 3 * (p1 - p2) + p3 - p0
        y[i] = p1 + 0.5 * frac * (
            p2 - p0 + frac * (2 * p0 - 5 * p1 + 4 * p2 - p3 + frac * term)
        )

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
    """Add AWGN to achieve specified SNR.

    Args:
        x: Input signal.
        snr_db: Desired SNR in dB.
        rng: Random number generator.
        reference_power: Reference signal power for SNR calculation.
            Use calculate_reference_power() from util.py to compute this
            from a representative signal (e.g., pulse-shaped symbols).
            Defaults to 1.0 (unit power).

    Returns:
        Signal with added AWGN.
    """
    # Calculate noise power for desired SNR using reference power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = reference_power / snr_linear

    # Generate complex AWGN
    noise_std = np.sqrt(noise_power / 2)  # /2 for complex noise (I and Q)
    noise = noise_std * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))

    return x + noise


# =============================================================================
# Main Channel Model Class
# =============================================================================


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

        # Pre-compute Rician K-factor in linear scale
        self._rician_k_linear = 10 ** (config.rician_k_db / 10)

    @classmethod
    def from_profile(
        cls,
        profile: ChannelProfile,
        request: ProfileRequest,
    ) -> "ChannelModel":
        """Create channel model from preset profile."""
        config = get_profile_config(
            profile=profile,
            request=request,
        )
        return cls(config)

    def reset(self) -> None:
        """Reset channel state for new transmission."""
        self._state = None

    def _init_state(self) -> StreamState:
        """Initialize streaming state."""
        rng = np.random.default_rng(self.config.seed)

        return StreamState(
            sample_index=0,
            phase_accumulator=self.config.initial_phase_rad,
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

        # 1. Propagation delay
        if self.config.enable_delay and self.config.delay_samples > 0:
            y, state.delay_buffer = apply_fractional_delay(
                y,
                self.config.delay_samples,
                state.delay_buffer,
            )

        # 2. Multipath and fading
        if self.config.enable_multipath:
            # Generate fading gains if enabled
            fading_gains = None
            if self.config.enable_fading and self.config.doppler_hz > 0:
                fading_gains, state.fading_phases = generate_fading_gains(
                    params=FadingParams(
                        n_samples=n_samples,
                        n_taps=len(self.config.multipath_delays_samples),
                        doppler_hz=self.config.doppler_hz,
                        sample_rate=self.config.sample_rate,
                        fading_type=self.config.fading_type,
                        rician_k_linear=self._rician_k_linear,
                    ),
                    phases=state.fading_phases,
                    rng=state.rng,
                )

            y, state.multipath_buffer = apply_multipath(
                y,
                self.config.multipath_delays_samples,
                self._multipath_gains_linear,
                fading_gains,
                state.multipath_buffer,
            )

        # 3. CFO and phase offset
        if self.config.enable_cfo or self.config.enable_phase_offset:
            cfo = self.config.cfo_hz if self.config.enable_cfo else 0.0
            drift = (
                self.config.phase_drift_hz if self.config.enable_phase_offset else 0.0
            )
            # Always use initial_phase_rad - sample_index provides streaming continuity
            y, state.phase_accumulator = apply_cfo_and_phase(
                y,
                CfoPhaseParams(
                    cfo_hz=cfo,
                    sample_rate=self.config.sample_rate,
                    initial_phase=self.config.initial_phase_rad,
                    phase_drift_hz=drift,
                    sample_index=state.sample_index,
                ),
            )

        # 4. Phase noise
        if self.config.enable_phase_noise:
            if state.rng is None:
                state.rng = np.random.default_rng(self.config.seed)
            y, state.phase_noise_state = apply_phase_noise(
                y,
                self.config.phase_noise_psd_dbchz,
                self.config.sample_rate,
                state.phase_noise_state,
                state.rng,
            )

        # 5. AWGN
        if state.rng is None:
            state.rng = np.random.default_rng(self.config.seed)
        y = apply_awgn(y, self.config.snr_db, state.rng, self.config.reference_power)

        # 6. SCO
        if self.config.enable_sco and abs(self.config.sco_ppm) > SCO_THRESHOLD:
            y, state.sco_phase, state.sco_buffer = apply_sco(
                y,
                self.config.sco_ppm,
                state.sco_phase,
                state.sco_buffer,
            )

        # Update sample index
        state.sample_index += n_samples

        return y

    def get_impulse_response(
        self,
        _n_samples: int = 256,
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Get the channel impulse response (multipath only, no noise)."""
        delays = np.array(self.config.multipath_delays_samples)
        gains = self._multipath_gains_linear.astype(np.complex128)
        return delays, gains

    def __repr__(self) -> str:
        """Return string representation."""
        parts = [f"ChannelModel(snr={self.config.snr_db}dB"]

        if self.config.enable_multipath:
            parts.append(f"multipath={len(self.config.multipath_delays_samples)} taps")

        if self.config.enable_fading:
            parts.append(f"fading={self.config.fading_type}")

        if self.config.enable_cfo:
            parts.append(f"cfo={self.config.cfo_hz}Hz")

        if self.config.enable_delay:
            parts.append(f"delay={self.config.delay_samples:.2f} samples")

        if self.config.enable_phase_noise:
            parts.append(f"phase_noise={self.config.phase_noise_psd_dbchz}dBc/Hz")

        if self.config.enable_sco:
            parts.append(f"sco={self.config.sco_ppm}ppm")

        return ", ".join(parts) + ")"
