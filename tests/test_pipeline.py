from numpy import shape
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules import pipeline
from modules.frame_constructor import ModulationSchemes
from modules.pipeline import *
from modules.pulse_shaping import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]

@composite
def random_pipeline_config(draw):
    return PipelineConfig(
        MOD_SCHEME = draw(st.sampled_from(MOD_SCHEMES))
    )

@composite
def random_packet_length(draw):
    return draw(st.integers(min_value=1, max_value=(2**12)-1))

# --- Tests ---
@given(pipeline_config = random_pipeline_config(), packet_length = random_packet_length())
def test_simple(pipeline_config, packet_length):

    tx = TXPipeline(pipeline_config)
    rx = RXPipeline(pipeline_config)
    
    rng   = np.random.default_rng(42)
    packet = Packet(
        src_mac=0,
        dst_mac=1,
        type=0,
        seq_num=0,
        length=packet_length,
        payload=rng.integers(0, 2, packet_length*8)
    )

    print(packet.payload.shape)

    tx_signal = tx.transmit(packet)
    print(tx_signal.shape)
    
    detection = DetectionResult(len(tx.sync_syms), valid=True)
    tx_signal = downsample(tx_signal, pipeline_config.SPS, tx.rrc_taps)
    rx_packet = rx.decode(tx_signal, detection)

    assert packet.payload.all() == rx_packet.payload.all()
