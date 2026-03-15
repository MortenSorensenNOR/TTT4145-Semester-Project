"""Sync chain test.

Verifies RXPipeline.detect() finds the preamble
through RRC + CFO + AWGN.
"""

import numpy as np
import pytest

from modules.frame_sync import generate_preamble
from modules.modulators import QPSK
from modules.pipeline import PipelineConfig, RXPipeline
from modules.pulse_shaping import rrc_filter, upsample

# Test constants
RNG_SEED = 42
SAMPLE_OFFSET = 200
NOISE_SCALE = 0.1
N_PAYLOAD_SYMBOLS = 100
CFO_TOLERANCE_HZ = 100
MIN_DETECTION_CONFIDENCE = 0.5
TIMING_TOLERANCE_TAPS = 2


@pytest.mark.parametrize("cfo_hz", [0.0, 150.0, 250.0])
def test_sync_chain(cfo_hz: float) -> None:
    """Full coarse -> fine through RRC + CFO + AWGN via RXPipeline.detect()."""
    config = PipelineConfig()
    sps = config.SPS
    cfg = config.SYNC_CONFIG
    num_taps = 2 * sps * config.SPAN + 1
    rrc_taps = rrc_filter(sps, config.RRC_ALPHA, num_taps)
    rng = np.random.default_rng(RNG_SEED)

    # -- TX --
    preamble = generate_preamble(cfg)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    tx = upsample(np.concatenate([preamble, payload]), sps, rrc_taps)

    # -- channel: delay + CFO + noise --
    rx = np.concatenate([np.zeros(SAMPLE_OFFSET, dtype=complex), tx])
    rx *= np.exp(2j * np.pi * cfo_hz / config.SAMPLE_RATE * np.arange(len(rx)))
    rx += NOISE_SCALE * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))

    # -- RX --
    rx_pipe = RXPipeline(config)
    result = rx_pipe.detect(rx)

    # -- checks --
    assert result.cfo_estimate == pytest.approx(cfo_hz, abs=CFO_TOLERANCE_HZ)
    assert result.confidence > MIN_DETECTION_CONFIDENCE

    # payload_start ≈ offset + TX group delay + short preamble + long_ref
    tx_group_delay = (num_taps - 1) // 2
    short_samples = cfg.short_preamble_nsym * cfg.short_preamble_nreps * sps
    expected_start = SAMPLE_OFFSET + tx_group_delay + short_samples + len(rx_pipe.long_ref)
    assert result.payload_start == pytest.approx(expected_start, abs=TIMING_TOLERANCE_TAPS * num_taps)

    # -- false positive: noise-only must not produce high confidence --
    noise_only = NOISE_SCALE * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))
    assert rx_pipe.detect(noise_only).confidence < MIN_DETECTION_CONFIDENCE
