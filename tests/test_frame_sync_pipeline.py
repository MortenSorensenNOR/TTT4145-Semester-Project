"""Sync chain test.

Verifies RXPipeline.detect() finds the preamble
through RRC + CFO + AWGN.
"""

import numpy as np

from modules.frame_sync import generate_preamble
from modules.modulators import QPSK
from modules.pipeline import PipelineConfig, RXPipeline
from modules.pulse_shaping import rrc_filter, upsample

# Test thresholds
CFO_TOLERANCE_HZ = 100
MIN_DETECTION_CONFIDENCE = 0.5
N_PAYLOAD_SYMBOLS = 100


def test_sync_chain() -> None:
    """Full coarse -> fine through RRC + CFO + AWGN via RXPipeline.detect()."""
    config = PipelineConfig()
    sps = config.SPS
    cfg = config.SYNC_CONFIG
    num_taps = 2 * sps * config.SPAN + 1
    rrc_taps = rrc_filter(sps, config.RRC_ALPHA, num_taps)
    rng = np.random.default_rng(42)

    # -- TX --
    preamble = generate_preamble(cfg)
    payload = rng.choice(QPSK().symbol_mapping, N_PAYLOAD_SYMBOLS)
    tx = upsample(np.concatenate([preamble, payload]), sps, rrc_taps)

    # -- channel: delay + CFO + noise --
    offset = 200
    cfo_hz = 150.0
    rx = np.concatenate([np.zeros(offset, dtype=complex), tx])
    rx *= np.exp(2j * np.pi * cfo_hz / config.SAMPLE_RATE * np.arange(len(rx)))
    rx += 0.1 * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))

    # -- RX --
    rx_pipe = RXPipeline(config)
    result = rx_pipe.detect(rx)

    # -- checks --
    assert abs(result.cfo_estimate - cfo_hz) < CFO_TOLERANCE_HZ
    assert result.confidence > MIN_DETECTION_CONFIDENCE

    # payload_start ≈ offset + TX group delay + short preamble + long_ref
    tx_group_delay = (num_taps - 1) // 2
    short_samples = cfg.short_preamble_nsym * cfg.short_preamble_nreps * sps
    expected_start = offset + tx_group_delay + short_samples + len(rx_pipe.long_ref)
    assert abs(result.payload_start - expected_start) < 2 * num_taps
