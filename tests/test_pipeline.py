import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.frame_constructor import *
from modules.pipeline import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]

@composite
def random_frame_header(draw):
    return FrameHeader(
        length=draw(st.integers(min_value=1, max_value=(2**12)-1)),
        src=0,
        dst=1,
        frame_type=0,
        mod_scheme=draw(st.sampled_from(MOD_SCHEMES)),
        sequence_number=0,
    )

# --- Fixtures ---

@pytest.fixture
def tx_instance():
    config = PipelineConfig()
    return TXPipeline(config)

@pytest.fixture
def rx_instance():
    config = PipelineConfig()
    return RXPipeline(config)

# --- Tests ---
def test_simple(tx_instance, rx_instance):
    pass
