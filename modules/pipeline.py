from dataclasses import dataclass
import numpy as np

from pulse_shaping import *
from modulation_schemes import *
from frame_constructor import *
from golay import *
from cfo_timing_and_detection import *
from costas_loop.costas import *
from ldpc.ldpc import *

@dataclass
class PipelineConfig:
    SPS: int = 8
    SPAN: int = 8
    RRC_ALPHA: float = 0.35
    MOD_SCHEME: ModulationSchemes = ModulationSchemes.QPSK
    CODING_RATE: CodeRates = CodeRates.NONE

    pulse_shaping: bool = True
    pilots: bool = False
    costas_loop: bool = False
    channnel_coding: bool = False
    interleaving: bool = False
    cfo_correction: bool = True

@dataclass
class Packet:
    """Packet of data to/from TAP/TUN"""

class TXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

    def send(self, packet: Packet) -> np.ndarray:
        return np.ndarray([])

@dataclass
class DetectionResult:
    """
    Result from detection loop:
    1. Timing offset of detection
    2. CFO estimate 
    3. Confidence
    """
    payload_start: int  = -1
    cfo_estimate: float = 0.0
    confidence: float   = 0.0

class RXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

    def detect(self, buffer: np.ndarray) -> DetectionResult:
        return DetectionResult()

    def decode(self, buffer: np.ndarray) -> Packet:
        return Packet()

if __name__ == "__main__":
    config = PipelineConfig()
    tx_pipe = TXPipeline(config)
    rx_pipe = RXPipeline(config)
