import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules.frame_constructor.frame_constructor import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]
_MAX_PAYLOAD_LEN = (1 << FrameHeaderConfig().payload_length_bits) - 1


@composite
def random_frame_header(draw):
    return FrameHeader(
        length=draw(st.integers(min_value=1, max_value=_MAX_PAYLOAD_LEN)),
        src=0,
        dst=1,
        frame_type=0,
        mod_scheme=draw(st.sampled_from(MOD_SCHEMES)),
        sequence_number=0,
    )


@pytest.fixture
def frame_header_constructor_instance():
    return FrameHeaderConstructor(FrameHeaderConfig())


@pytest.fixture
def frame_constructor_instance():
    return FrameConstructor(FrameHeaderConfig())


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(frame_header=random_frame_header())
def test_frame_header_roundtrip(frame_header_constructor_instance, frame_header):
    encoded = frame_header_constructor_instance.encode(frame_header)
    assert frame_header_constructor_instance.decode(encoded) == frame_header


@settings(deadline=10000, suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(frame_header=random_frame_header())
def test_frame_constructor_roundtrip(frame_constructor_instance, frame_header):
    rng = np.random.default_rng(42)
    payload_bits = rng.integers(0, 2, size=frame_header.length * 8)

    header_enc, payload_enc = frame_constructor_instance.encode(frame_header, payload_bits)
    assert frame_constructor_instance.decode_header(header_enc) == frame_header
    assert np.array_equal(frame_constructor_instance.decode_payload(frame_header, payload_enc), payload_bits)
