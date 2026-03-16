import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.frame_constructor import *

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
def frame_header_constructor_instance():
    config = FrameHeaderConfig()
    return FrameHeaderConstructor(config)

@pytest.fixture
def frame_constructor_instance():
    config = FrameHeaderConfig()
    return FrameConstructor(config)

# --- Tests ---

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(frame_header=random_frame_header())
def test_frame_header_constructor(frame_header_constructor_instance: FrameHeaderConstructor, frame_header: FrameHeader):
    encoded_header = frame_header_constructor_instance.encode(frame_header)
    result = frame_header_constructor_instance.decode(encoded_header)

    assert result == frame_header

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(frame_header=random_frame_header())
def test_frame_constructor(frame_constructor_instance: FrameConstructor, frame_header: FrameHeader):
    rng   = np.random.default_rng(42)
    payload_bits  = rng.integers(0, 2, size=frame_header.length*8)

    frame_encoded = frame_constructor_instance.encode(frame_header, payload_bits)
    result_payload = frame_constructor_instance.decode_payload(frame_header, frame_encoded[1])
    result_header = frame_constructor_instance.decode_header(frame_encoded[0])

    assert result_payload.all() == payload_bits.all() 
    assert result_header == frame_header

