import numpy as np
import pytest

from modules.pipeline import *

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
