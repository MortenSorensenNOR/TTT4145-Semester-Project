from dataclasses import dataclass, field
import numpy as np

from modules.pulse_shaping import *
from modules.modulators import *
from modules.frame_constructor import *
from modules.golay import *
from modules.frame_sync import *
from modules.costas_loop.costas import *
from modules.ldpc.ldpc import *

@dataclass
class PipelineConfig:
    SAMPLE_RATE: int = 5_336_000
    CENTER_FREQ: int = 2_400_000_000
    SPS: int = 8
    SPAN: int = 8
    RRC_ALPHA: float = 0.35
    MOD_SCHEME: ModulationSchemes = ModulationSchemes.QPSK
    CODING_RATE: CodeRates = CodeRates.NONE

    SYNC_CONFIG = SynchronizerConfig()
    COSTAS_CONFIG = CostasConfig()

    pulse_shaping: bool = True
    pilots: bool = False
    costas_loop: bool = False
    channnel_coding: bool = False
    interleaving: bool = False
    cfo_correction: bool = True

@dataclass
class Packet:
    """Packet of data to/from TAP/TUN"""
    src_mac: int = -1
    dst_mac: int = -1
    type: int = -1
    seq_num: int = -1
    length: int = -1
    payload: np.ndarray = field(default_factory=lambda: np.ndarray([]))

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

        # sync
        self.sync_syms = generate_preamble(self.config.SYNC_CONFIG)

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

        # construct signal
        tx_syms = np.concat([self.sync_syms, header_syms, payload_syms])

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

    valid: bool = False
    err_reason: str = ""

class RXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

        self.num_taps = 2 * config.SPS * config.SPAN + 1
        self.rrc_taps = rrc_filter(config.SPS, config.RRC_ALPHA, self.num_taps)

        self.bpsk = BPSK()
        self.qpsk = QPSK()
        self.psk8 = PSK8()

        # generate the known long preamble for matched filtering
        self.long_ref = build_long_ref(self.config.SYNC_CONFIG, self.config.SPS, self.rrc_taps)

    def receive(self, buffer: np.ndarray) -> np.ndarray:
        # TODO: Create a loop that runs receive or some shit
        detection_results = self.detect(buffer)

        num_of_detections = len(detection_results)

        packets = np.ndarray(num_of_detections, dtype=Packet)

        if num_of_detections < 1:
            return np.array([Packet(payload=np.array([0]))]) # have to return an empty packet
        
        for i in range(num_of_detections):
            #detection_results[i].payload_start = 243
            rx_syms = downsample(buffer[detection_results[i].payload_start:], self.config.SPS, self.rrc_taps)
            
            print(rx_syms.shape, detection_results[i])
            packets[i] = self.decode(rx_syms, detection_results[i])
        
        return packets

    def detect(self, buffer: np.ndarray) -> np.ndarray:
        cfg = self.config.SYNC_CONFIG
        sps = self.config.SPS

        # coarse timing + CFO
        try:
            coarse = coarse_sync(buffer, self.config.SAMPLE_RATE, sps, cfg)
        except ValueError as e:
            return np.array([])

        # fine timing via long preamble correlation
        try:
            fine = fine_timing(buffer, self.long_ref, coarse, self.config.SAMPLE_RATE, sps, cfg)
            fine_start = fine.sample_idx
        except ValueError as e:
            return np.array([])

        return np.array([DetectionResult(
            payload_start=(int(fine_start) + len(self.long_ref)),
            cfo_estimate=float(coarse.cfo_hat),
            confidence=float(coarse.m_peak),
            valid=True
        )])

    def decode(self, buffer: np.ndarray, detection_res: DetectionResult) -> Packet:
        header, payload_start, current_phase_estimate = self.header_decode(buffer, detection_res)
        payload = self.payload_decode(buffer, header, payload_start, current_phase_estimate)
        return Packet(
            src_mac=header.src, 
            dst_mac=header.dst,
            type=header.frame_type,
            seq_num=header.sequence_number,
            length=header.length,
            payload=payload,
            valid=header.crc_passed
        )

    def header_decode(self, buffer: np.ndarray, detection_res: DetectionResult) -> tuple[FrameHeader, int, float]:
        """Decode the header part of the packet. Assumes buffer input is already decimated."""
        header_syms = buffer[:2 * self.frame_constructor.header_config.header_total_size]

        # costas correction
        #header_syms, phase_est = apply_costas_loop(header_syms, self.config.COSTAS_CONFIG, ModulationSchemes.BPSK)
        phase_est = [0]
        print(header_syms.shape, detection_res.payload_start, detection_res.payload_start+2*self.frame_constructor.header_config.header_total_size)
        # demodulate header
        header_bits = self.bpsk.symbols2bits(header_syms)
        header = self.frame_constructor.decode_header(header_bits)
        return header, self.frame_constructor.header_config.header_total_size, phase_est[-1]

    def payload_decode(self, buffer: np.ndarray, header: FrameHeader, payload_start, phase_estimate: float) -> np.ndarray:
        rx_syms = buffer[payload_start:]

        # costas correction
        rx_syms, _ = apply_costas_loop(rx_syms, self.config.COSTAS_CONFIG, header.mod_scheme, phase_estimate)

        # demodulate
        match (header.mod_scheme):
            case ModulationSchemes.BPSK:
                payload_bits = self.bpsk.symbols2bits(rx_syms)
            case ModulationSchemes.QPSK:
                payload_bits = self.qpsk.symbols2bits(rx_syms)
            case ModulationSchemes.PSK8:
                payload_bits = self.psk8.symbols2bits(rx_syms)

        return payload_bits

if __name__ == "__main__":
    config = PipelineConfig()
    tx_pipe = TXPipeline(config)
    rx_pipe = RXPipeline(config)
