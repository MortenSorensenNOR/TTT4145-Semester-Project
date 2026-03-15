
from modules.frame_constructor import FrameConstructor, FrameHeader, FrameHeaderConfig, FrameHeaderConstructor


# --- Fixtures ---

@pytest.fixture
def module_instance():
    return <ClassName>(<args>)


# --- Tests ---

def test_<description>(module_instance):
    result = module_instance.<method>(<input>)
    assert result == <expected>

if __name__ == "__main__":
    config = FrameHeaderConfig()
    frame = FrameConstructor()
