"""
Channel Model for Radio Communication Simulation

Implements a modular channel model for simulating indoor and urban outdoor
short-range radio communications with:
- Simple multipath (tapped delay line)
- RF impairments: CFO, phase offset, time delay, phase noise, SCO
- AWGN
- Streaming/live simulation support for sync and frame detection testing

Designed with realistic defaults for Pluto SDR (25 ppm oscillator).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from scipy import signal


# =============================================================================
# Helper Functions
# =============================================================================


def pluto_max_cfo(carrier_hz: float = 433e6, num_devices: int = 2) -> float:
    """
    Calculate maximum CFO for Pluto SDR (25 ppm oscillator).

    Parameters
    ----------
    carrier_hz : float
        Carrier frequency in Hz (default: 433 MHz ISM band)
    num_devices : int
        Number of devices (2 for TX+RX, each with its own oscillator)

    Returns
    -------
    float
        Maximum CFO in Hz

    Examples
    --------
    >>> pluto_max_cfo(433e6, 2)  # Two Plutos at 433 MHz
    21650.0
    >>> pluto_max_cfo(915e6, 2)  # Two Plutos at 915 MHz
    45750.0
    """
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
    """
    Calculate propagation delay given a distance.

    Parameters
    ----------
    distance_m : float
        Distance in meters

    Returns
    -------
    float
        Propagation delay in seconds

    Examples
    --------
    >>> distance_to_delay(300)  # 300 meters
    1.0006922855944561e-06
    >>> distance_to_delay(1000)  # 1 km ~ 3.33 microseconds
    3.335640951981521e-06
    """
    return distance_m / SPEED_OF_LIGHT


# =============================================================================
# Configuration Classes
# =============================================================================


class ChannelProfile(Enum):
    """Preset channel profiles for common scenarios."""

    IDEAL = "ideal"  # AWGN only
    INDOOR_SMALL = "indoor_small"  # Small office/home
    INDOOR_MEDIUM = "indoor_medium"  # Conference room, larger spaces
    URBAN_LOS = "urban_los"  # Outdoor with line-of-sight
    URBAN_NLOS = "urban_nlos"  # Outdoor, obstructed path


@dataclass
class ChannelConfig:
    """
    Configuration for the channel model.

    All impairments can be individually enabled/disabled via enable_* flags.
    """

    # Basic parameters
    sample_rate: float = 1e6
    snr_db: float = 20.0

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
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if len(self.multipath_delays_samples) != len(self.multipath_gains_db):
            raise ValueError(
                "multipath_delays_samples and multipath_gains_db must have same length"
            )
        if self.fading_type not in ("rayleigh", "rician"):
            raise ValueError("fading_type must be 'rayleigh' or 'rician'")


@dataclass
class StreamState:
    """
    Maintains state for streaming/block-by-block processing.

    This allows continuous processing of signal blocks while maintaining
    phase continuity and filter states across block boundaries.
    """

    sample_index: int = 0
    phase_accumulator: float = 0.0

    # Fading state (for sum-of-sinusoids model)
    fading_phases: Optional[NDArray[np.float64]] = None

    # Delay buffer for fractional delay filter
    delay_buffer: Optional[NDArray[np.complex128]] = None

    # Multipath delay line buffer
    multipath_buffer: Optional[NDArray[np.complex128]] = None

    # Phase noise state
    phase_noise_state: float = 0.0

    # SCO state (for resampling)
    sco_phase: float = 0.0
    sco_buffer: Optional[NDArray[np.complex128]] = None

    # Random number generator
    rng: Optional[np.random.Generator] = None


# =============================================================================
# Profile Presets
# =============================================================================


def get_profile_config(
    profile: ChannelProfile,
    sample_rate: float,
    snr_db: float,
    cfo_hz: Optional[float] = None,
    phase_offset_rad: Optional[float] = None,
    delay_samples: Optional[float] = None,
    seed: Optional[int] = None,
) -> ChannelConfig:
    """
    Create a ChannelConfig from a preset profile.

    Parameters
    ----------
    profile : ChannelProfile
        The preset profile to use
    sample_rate : float
        Sample rate in Hz
    snr_db : float
        Signal-to-noise ratio in dB
    cfo_hz : float, optional
        Override CFO (default: 0 for all profiles)
    phase_offset_rad : float, optional
        Override initial phase offset
    delay_samples : float, optional
        Override propagation delay
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    ChannelConfig
        Configuration object for the channel model
    """
    # Base configuration
    config_kwargs = {
        "sample_rate": sample_rate,
        "snr_db": snr_db,
        "seed": seed,
    }

    # Apply CFO if specified
    if cfo_hz is not None:
        config_kwargs["enable_cfo"] = True
        config_kwargs["cfo_hz"] = cfo_hz

    # Apply phase offset if specified
    if phase_offset_rad is not None:
        config_kwargs["enable_phase_offset"] = True
        config_kwargs["initial_phase_rad"] = phase_offset_rad

    # Apply delay if specified
    if delay_samples is not None:
        config_kwargs["enable_delay"] = True
        config_kwargs["delay_samples"] = delay_samples

    # Profile-specific settings
    if profile == ChannelProfile.IDEAL:
        # AWGN only - no multipath, no fading
        pass

    elif profile == ChannelProfile.INDOOR_SMALL:
        # Small office/home: 10-25 ns delay spread, no fading
        # Typical 2-tap model
        delays_ns = (0.0, 15.0)  # Direct path + one reflection
        delays_samples = tuple(delay_ns_to_samples(d, sample_rate) for d in delays_ns)
        config_kwargs.update(
            {
                "enable_multipath": True,
                "multipath_delays_samples": delays_samples,
                "multipath_gains_db": (0.0, -8.0),  # Second tap 8 dB down
            }
        )

    elif profile == ChannelProfile.INDOOR_MEDIUM:
        # Conference room: 20-80 ns delay spread, Rician fading K=10dB, Doppler 2 Hz
        delays_ns = (0.0, 30.0, 60.0)
        delays_samples = tuple(delay_ns_to_samples(d, sample_rate) for d in delays_ns)
        config_kwargs.update(
            {
                "enable_multipath": True,
                "multipath_delays_samples": delays_samples,
                "multipath_gains_db": (0.0, -6.0, -12.0),
                "enable_fading": True,
                "fading_type": "rician",
                "rician_k_db": 10.0,
                "doppler_hz": 2.0,
            }
        )

    elif profile == ChannelProfile.URBAN_LOS:
        # Urban outdoor with LOS: 30-150 ns, Rician K=6dB, Doppler 10 Hz
        delays_ns = (0.0, 50.0, 100.0, 150.0)
        delays_samples = tuple(delay_ns_to_samples(d, sample_rate) for d in delays_ns)
        config_kwargs.update(
            {
                "enable_multipath": True,
                "multipath_delays_samples": delays_samples,
                "multipath_gains_db": (0.0, -4.0, -8.0, -14.0),
                "enable_fading": True,
                "fading_type": "rician",
                "rician_k_db": 6.0,
                "doppler_hz": 10.0,
            }
        )

    elif profile == ChannelProfile.URBAN_NLOS:
        # Urban outdoor NLOS: 50-350 ns, Rayleigh fading, Doppler 20 Hz
        delays_ns = (0.0, 80.0, 180.0, 300.0)
        delays_samples = tuple(delay_ns_to_samples(d, sample_rate) for d in delays_ns)
        config_kwargs.update(
            {
                "enable_multipath": True,
                "multipath_delays_samples": delays_samples,
                "multipath_gains_db": (0.0, -3.0, -6.0, -10.0),
                "enable_fading": True,
                "fading_type": "rayleigh",
                "doppler_hz": 20.0,
            }
        )

    return ChannelConfig(**config_kwargs)


# =============================================================================
# Signal Processing Functions
# =============================================================================


def apply_fractional_delay(
    x: NDArray[np.complex128],
    delay_samples: float,
    buffer: Optional[NDArray[np.complex128]] = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Apply fractional sample delay using Lagrange interpolation.

    Uses 4th-order (5-tap) Lagrange interpolation for accurate fractional delay.
    The output signal is shifted in time: zeros are inserted at the beginning
    and the signal is truncated at the end to maintain the same length.

    Parameters
    ----------
    x : ndarray
        Input signal
    delay_samples : float
        Delay in samples (can be fractional)
    buffer : ndarray, optional
        State buffer from previous block (for streaming)

    Returns
    -------
    y : ndarray
        Delayed signal (same length as input)
    new_buffer : ndarray
        Updated state buffer for next block
    """
    if delay_samples == 0.0:
        return x.copy(), buffer if buffer is not None else np.zeros(4, dtype=x.dtype)

    n_samples = len(x)
    int_delay = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    # Handle integer delay: shift signal and pad with zeros/buffer
    y = np.zeros(n_samples, dtype=x.dtype)

    if buffer is not None and len(buffer) > 0:
        # Use samples from buffer for the delayed portion
        buffer_samples_needed = min(int_delay, len(buffer), n_samples)
        if buffer_samples_needed > 0:
            y[:buffer_samples_needed] = buffer[-buffer_samples_needed:]

    # Copy input signal shifted by int_delay
    if int_delay < n_samples:
        samples_to_copy = n_samples - int_delay
        y[int_delay:] = x[:samples_to_copy]

    # Apply fractional delay using Lagrange interpolation if needed
    if frac_delay > 1e-9:
        # Lagrange interpolation coefficients (4th order, 5 taps)
        order = 4
        n_taps = order + 1

        # Calculate Lagrange coefficients for fractional delay
        h = np.zeros(n_taps)
        for k in range(n_taps):
            h[k] = 1.0
            for m in range(n_taps):
                if m != k:
                    h[k] *= (frac_delay - (m - order // 2)) / (k - m)

        # Apply fractional delay filter
        # Pad to handle filter edge effects
        y_padded = np.concatenate([np.zeros(n_taps - 1, dtype=x.dtype), y])
        y_filtered = signal.lfilter(h, 1.0, y_padded)
        y = y_filtered[n_taps - 1 : n_taps - 1 + n_samples]

    # Update buffer with end of input for streaming continuity
    buffer_size = max(int_delay + 5, 5)
    new_buffer = np.zeros(buffer_size, dtype=x.dtype)
    if n_samples >= buffer_size:
        new_buffer[:] = x[-buffer_size:]
    else:
        if buffer is not None and len(buffer) >= buffer_size - n_samples:
            new_buffer[: buffer_size - n_samples] = buffer[-(buffer_size - n_samples) :]
        new_buffer[buffer_size - n_samples :] = x

    return y.astype(x.dtype), new_buffer


def apply_multipath(
    x: NDArray[np.complex128],
    delays_samples: tuple[float, ...],
    gains_linear: NDArray[np.float64],
    fading_gains: Optional[NDArray[np.complex128]] = None,
    buffer: Optional[NDArray[np.complex128]] = None,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Apply multipath channel using tapped delay line.

    Parameters
    ----------
    x : ndarray
        Input signal
    delays_samples : tuple
        Tap delays in samples
    gains_linear : ndarray
        Tap gains in linear scale
    fading_gains : ndarray, optional
        Time-varying fading gains per tap (shape: [n_taps, n_samples])
    buffer : ndarray, optional
        Delay line buffer from previous block

    Returns
    -------
    y : ndarray
        Output signal with multipath
    new_buffer : ndarray
        Updated delay line buffer
    """
    n_samples = len(x)
    n_taps = len(delays_samples)
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
    for i, (delay, gain) in enumerate(zip(delays_samples, gains_linear)):
        int_delay = int(np.floor(delay))
        frac_delay = delay - int_delay

        # Get delayed samples
        if frac_delay < 1e-6:
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
    n_samples: int,
    n_taps: int,
    doppler_hz: float,
    sample_rate: float,
    fading_type: str,
    rician_k_linear: float,
    phases: Optional[NDArray[np.float64]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    Generate time-varying fading gains using Clarke's sum-of-sinusoids model.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_taps : int
        Number of multipath taps
    doppler_hz : float
        Maximum Doppler frequency
    sample_rate : float
        Sample rate in Hz
    fading_type : str
        'rayleigh' or 'rician'
    rician_k_linear : float
        Rician K-factor in linear scale
    phases : ndarray, optional
        Initial phases for each sinusoid (for streaming continuity)
    rng : Generator, optional
        Random number generator

    Returns
    -------
    gains : ndarray
        Complex fading gains, shape [n_taps, n_samples]
    new_phases : ndarray
        Updated phases for next block
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sinusoids = 16  # Number of sinusoids in sum-of-sinusoids model

    # Initialize phases if not provided
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, (n_taps, n_sinusoids, 2))  # 2 for I/Q

    t = np.arange(n_samples) / sample_rate
    gains = np.zeros((n_taps, n_samples), dtype=np.complex128)

    for tap in range(n_taps):
        # Sum of sinusoids for I and Q components
        inphase = np.zeros(n_samples)
        quadrature = np.zeros(n_samples)

        for k in range(n_sinusoids):
            # Jakes model: frequencies uniformly distributed in Doppler spectrum
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            freq_k = doppler_hz * np.cos(alpha_k)

            # I component
            inphase += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 0])
            # Q component
            quadrature += np.cos(2 * np.pi * freq_k * t + phases[tap, k, 1])

        # Normalize
        inphase /= np.sqrt(n_sinusoids)
        quadrature /= np.sqrt(n_sinusoids)

        if fading_type == "rician":
            # Add LOS component
            k_factor = rician_k_linear
            los_amplitude = np.sqrt(k_factor / (k_factor + 1))
            scatter_amplitude = np.sqrt(1 / (k_factor + 1))
            gains[tap, :] = los_amplitude + scatter_amplitude * (inphase + 1j * quadrature)
        else:
            # Pure Rayleigh
            gains[tap, :] = (inphase + 1j * quadrature) / np.sqrt(2)

    # Update phases for streaming continuity
    phase_increment = 2 * np.pi * doppler_hz * n_samples / sample_rate
    new_phases = phases.copy()
    for tap in range(n_taps):
        for k in range(n_sinusoids):
            alpha_k = (2 * np.pi * k + phases[tap, k, 0]) / n_sinusoids
            new_phases[tap, k, :] += phase_increment * np.cos(alpha_k)

    return gains, new_phases


def apply_cfo_and_phase(
    x: NDArray[np.complex128],
    cfo_hz: float,
    sample_rate: float,
    initial_phase: float,
    phase_drift_hz: float,
    sample_index: int,
) -> tuple[NDArray[np.complex128], float]:
    """
    Apply carrier frequency offset and phase offset.

    Parameters
    ----------
    x : ndarray
        Input signal
    cfo_hz : float
        Carrier frequency offset in Hz
    sample_rate : float
        Sample rate in Hz
    initial_phase : float
        Initial phase offset in radians
    phase_drift_hz : float
        Slow phase drift rate in Hz
    sample_index : int
        Starting sample index (for streaming continuity)

    Returns
    -------
    y : ndarray
        Signal with CFO and phase applied
    final_phase : float
        Phase at end of block (for streaming)
    """
    n = len(x)
    t = (np.arange(sample_index, sample_index + n)) / sample_rate

    # Total phase: initial + CFO + drift
    phase = initial_phase + 2 * np.pi * cfo_hz * t + 2 * np.pi * phase_drift_hz * t

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
    """
    Apply phase noise using a filtered random walk model.

    Parameters
    ----------
    x : ndarray
        Input signal
    psd_dbchz : float
        Phase noise PSD in dBc/Hz at 1 kHz offset
    sample_rate : float
        Sample rate in Hz
    state : float
        Phase noise state from previous block
    rng : Generator
        Random number generator

    Returns
    -------
    y : ndarray
        Signal with phase noise
    new_state : float
        Updated phase noise state
    """
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
    buffer: Optional[NDArray[np.complex128]] = None,
) -> tuple[NDArray[np.complex128], float, NDArray[np.complex128]]:
    """
    Apply sample clock offset via resampling.

    Parameters
    ----------
    x : ndarray
        Input signal
    sco_ppm : float
        Sample clock offset in ppm
    phase : float
        Resampling phase from previous block
    buffer : ndarray, optional
        Buffer for interpolation

    Returns
    -------
    y : ndarray
        Resampled signal
    new_phase : float
        Updated resampling phase
    new_buffer : ndarray
        Updated interpolation buffer
    """
    if abs(sco_ppm) < 1e-9:
        return x.copy(), phase, buffer if buffer is not None else np.zeros(4, dtype=x.dtype)

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

        y[i] = (
            p1
            + 0.5
            * frac
            * (
                p2
                - p0
                + frac * (2 * p0 - 5 * p1 + 4 * p2 - p3 + frac * (3 * (p1 - p2) + p3 - p0))
            )
        )

    # Update phase for next block
    new_phase = (phase + n_out * ratio) % 1.0

    # Update buffer
    new_buffer = x_extended[-(interp_order + 1) :]

    return y, new_phase, new_buffer


def apply_awgn(
    x: NDArray[np.complex128], snr_db: float, rng: np.random.Generator
) -> NDArray[np.complex128]:
    """
    Add AWGN to achieve specified SNR.

    Parameters
    ----------
    x : ndarray
        Input signal
    snr_db : float
        Desired SNR in dB
    rng : Generator
        Random number generator

    Returns
    -------
    y : ndarray
        Signal with added noise
    """
    # Calculate signal power
    signal_power = np.mean(np.abs(x) ** 2)

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex AWGN
    noise_std = np.sqrt(noise_power / 2)  # /2 for complex noise (I and Q)
    noise = noise_std * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))

    return x + noise


# =============================================================================
# Main Channel Model Class
# =============================================================================


class ChannelModel:
    """
    Modular channel model for radio communication simulation.

    Supports both batch processing and streaming (block-by-block) modes.

    Processing order:
    1. Propagation delay (fractional sample delay)
    2. Multipath/fading (tapped delay line)
    3. CFO and phase offset (complex exponential rotation)
    4. Phase noise (optional)
    5. AWGN (added at receiver)
    6. SCO (optional resampling)
    """

    def __init__(self, config: ChannelConfig):
        """
        Initialize channel model.

        Parameters
        ----------
        config : ChannelConfig
            Channel configuration
        """
        self.config = config
        self._state: Optional[StreamState] = None

        # Pre-compute linear gains from dB
        self._multipath_gains_linear = 10 ** (np.array(config.multipath_gains_db) / 20)

        # Pre-compute Rician K-factor in linear scale
        self._rician_k_linear = 10 ** (config.rician_k_db / 10)

    @classmethod
    def from_profile(
        cls,
        profile: ChannelProfile,
        sample_rate: float,
        snr_db: float,
        cfo_hz: Optional[float] = None,
        phase_offset_rad: Optional[float] = None,
        delay_samples: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "ChannelModel":
        """
        Create channel model from preset profile.

        Parameters
        ----------
        profile : ChannelProfile
            Preset profile to use
        sample_rate : float
            Sample rate in Hz
        snr_db : float
            Signal-to-noise ratio in dB
        cfo_hz : float, optional
            Carrier frequency offset in Hz
        phase_offset_rad : float, optional
            Initial phase offset in radians
        delay_samples : float, optional
            Propagation delay in samples
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        ChannelModel
            Configured channel model instance
        """
        config = get_profile_config(
            profile=profile,
            sample_rate=sample_rate,
            snr_db=snr_db,
            cfo_hz=cfo_hz,
            phase_offset_rad=phase_offset_rad,
            delay_samples=delay_samples,
            seed=seed,
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
        """
        Apply channel model to entire signal (batch mode).

        This resets the channel state before processing.

        Parameters
        ----------
        x : ndarray
            Input signal (complex baseband)

        Returns
        -------
        y : ndarray
            Output signal after channel effects
        """
        self.reset()
        return self.process_block(x)

    def process_block(self, x: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Process a single block of samples (streaming mode).

        Maintains state across calls for continuous streaming simulation.

        Parameters
        ----------
        x : ndarray
            Input signal block (complex baseband)

        Returns
        -------
        y : ndarray
            Output signal block after channel effects
        """
        state = self._get_state()
        y = x.astype(np.complex128)
        n_samples = len(x)

        # 1. Propagation delay
        if self.config.enable_delay and self.config.delay_samples > 0:
            y, state.delay_buffer = apply_fractional_delay(
                y, self.config.delay_samples, state.delay_buffer
            )

        # 2. Multipath and fading
        if self.config.enable_multipath:
            # Generate fading gains if enabled
            fading_gains = None
            if self.config.enable_fading and self.config.doppler_hz > 0:
                fading_gains, state.fading_phases = generate_fading_gains(
                    n_samples=n_samples,
                    n_taps=len(self.config.multipath_delays_samples),
                    doppler_hz=self.config.doppler_hz,
                    sample_rate=self.config.sample_rate,
                    fading_type=self.config.fading_type,
                    rician_k_linear=self._rician_k_linear,
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
            drift = self.config.phase_drift_hz if self.config.enable_phase_offset else 0.0
            # Always use initial_phase_rad - sample_index provides streaming continuity
            initial_phase = self.config.initial_phase_rad

            y, state.phase_accumulator = apply_cfo_and_phase(
                y,
                cfo,
                self.config.sample_rate,
                initial_phase,
                drift,
                state.sample_index,
            )

        # 4. Phase noise
        if self.config.enable_phase_noise:
            y, state.phase_noise_state = apply_phase_noise(
                y,
                self.config.phase_noise_psd_dbchz,
                self.config.sample_rate,
                state.phase_noise_state,
                state.rng,
            )

        # 5. AWGN
        y = apply_awgn(y, self.config.snr_db, state.rng)

        # 6. SCO
        if self.config.enable_sco and abs(self.config.sco_ppm) > 1e-9:
            y, state.sco_phase, state.sco_buffer = apply_sco(
                y, self.config.sco_ppm, state.sco_phase, state.sco_buffer
            )

        # Update sample index
        state.sample_index += n_samples

        return y

    def get_impulse_response(
        self, n_samples: int = 256
    ) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Get the channel impulse response (multipath only, no noise).

        Parameters
        ----------
        n_samples : int
            Length of impulse response

        Returns
        -------
        delays : ndarray
            Delay values in samples
        gains : ndarray
            Complex gains at each delay
        """
        delays = np.array(self.config.multipath_delays_samples)
        gains = self._multipath_gains_linear.astype(np.complex128)
        return delays, gains

    def __repr__(self) -> str:
        """String representation."""
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
