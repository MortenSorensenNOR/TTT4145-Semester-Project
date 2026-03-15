from dataclasses import dataclass
import numpy as np

from pulse_shaping import *
from modulation_schemes import *
from frame_constructor import *
from golay import *
from frame_sync import *
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
    src_mac: int
    dst_mac: int
    type: int
    seq_num: int
    length: int
    payload: np.ndarray

    valid: bool = False
    err_reason: str = ""

class TXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

        self.bpsk = BPSK()
        match (self.config.MOD_SCHEME):
            case ModulationSchemes.BPSK:
                self.payload_modulator = BPSK()
            case ModulationSchemes.QPSK:
                self.payload_modulator = QPSK()
            case ModulationSchemes.PSK8:
                self.payload_modulator = PSK8()

        self.num_taps = 2 * config.SPS * config.SPAN + 1
        self.rrc_taps = rrc_filter(config.SPS, config.RRC_ALPHA, self.num_taps)

    def transmit(self, packet: Packet) -> np.ndarray:
        # construct bits
        header = FrameHeader(
            length=packet.length,
            src=packet.src_mac,
            dst=packet.dst_mac,
            frame_type=packet.type,
            mod_scheme=self.config.MOD_SCHEME,
            sequence_number=packet.seq_num
        )
        (header_bits, payload_bits) = self.frame_constructor.encode(header, packet.payload)

        # modulate 
        header_syms = self.bpsk.bits2symbols(header_bits)
        payload_syms = self.payload_modulator.bits2symbols(payload_bits)

        # sync
        sync_syms = np.zeros(10) # TODO: Get the actual sequence

        # construct signal
        tx_syms = np.concat([sync_syms, header_syms, payload_syms])

        # upsample and filter
        tx_signal = upsample(tx_syms, self.config.SPS, self.rrc_taps)
        return tx_signal

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

    def decode(self, buffer: np.ndarray, detect_res: DetectionResult) -> Packet:
        # TODO: Actually do the signal processing
        header = self.frame_constructor.decode_header(...)
        payload = self.frame_constructor.decode_payload(header, ..., soft=False, channel_coding=self.config.channnel_coding, interleaving=self.config.interleaving)
        return Packet(
            src_mac=header.src, 
            dst_mac=header.dst,
            type=header.frame_type,
            seq_num=header.sequence_number,
            length=header.length,
            payload=payload,
            valid=header.crc_passed
        )

if __name__ == "__main__":
    config = PipelineConfig()
    tx_pipe = TXPipeline(config)
    rx_pipe = RXPipeline(config)
