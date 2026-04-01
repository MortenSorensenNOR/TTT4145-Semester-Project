from dataclasses import dataclass
import numpy as np

from modules import gardner_ted
from modules.pulse_shaping import *
from modules.modulators import *
from modules.frame_constructor import *
from modules.golay import *
from modules.frame_sync import *
from modules.costas_loop.costas import *
# from modules.ldpc.ldpc import *
from modules.ldpc.channel_coding import *
from modules.gardner_ted.gardner import *

from utils.plotting import *

@dataclass
class PipelineConfig:
    SAMPLE_RATE: int = 5_336_000
    CENTER_FREQ: int = 2_400_000_000
    SPS: int = 8
    SPAN: int = 8
    RRC_ALPHA: float = 0.35
    MOD_SCHEME: ModulationSchemes = ModulationSchemes.QPSK
    CODING_RATE: CodeRates = CodeRates.NONE
    PRE_HEADER_GUARD_BITS: int = 0

    SYNC_CONFIG = SynchronizerConfig()
    COSTAS_CONFIG = CostasConfig(0.07) #Need to tune more

    pulse_shaping: bool = True
    pilots: bool = False
    costas_loop: bool = True
    gardner_ted: bool = False # Not working!
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

        self.guard_syms = np.zeros(500, dtype=complex)
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
            sequence_number=packet.seq_num,
        )
        (header_bits, payload_bits) = self.frame_constructor.encode(header, packet.payload)
        print("header_bits:", header_bits)
        # modulate
        header_syms = self.bpsk.bits2symbols(header_bits)

        # Adding extra known bits before header to figure out phase ambiguity of BPSK header
        if self.config.PRE_HEADER_GUARD_BITS > 0:
            known_sequence_syms = self.bpsk.bits2symbols(np.array([[0]*self.config.PRE_HEADER_GUARD_BITS]))
        else:
            known_sequence_syms = np.array([])

        payload_syms = self.payload_modulator.bits2symbols(payload_bits)
        # construct signal
        tx_syms = np.concat([self.guard_syms,self.sync_syms, known_sequence_syms, header_syms, payload_syms,self.guard_syms])

        # upsample and filter
        tx_signal = upsample(tx_syms, self.config.SPS, self.rrc_taps)
        return tx_signal

@dataclass
class DetectionResult:
    """Single frame detection result."""

    payload_start: int
    cfo_estimate: float
    phase_estimate: float
    confidence: float

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

    def receive(self, buffer: np.ndarray) -> list[Packet]:
        """Detect and decode all frames in buffer."""
        detections = self.detect(buffer)
        if not detections:
            return []

        packets = []
        filtered_buffer = match_filter(buffer, self.rrc_taps)

        for det in detections:
            rx_syms = filtered_buffer[det.payload_start:]
            print("\ndetection at index:", det.payload_start)
            try:
                decoded_packet = self.decode(rx_syms, det.cfo_estimate, det.phase_estimate)
                packets.append(decoded_packet)
            except Exception as e:
                print("DECODE ERROR:", e)

        return packets

    def detect(self, buffer: np.ndarray) -> list[DetectionResult]:
        cfg = self.config.SYNC_CONFIG
        sps = self.config.SPS

        try:
            coarse = coarse_sync(buffer, self.config.SAMPLE_RATE, sps, cfg)
        except Exception as e:
            print(e)
            return []

        if coarse.m_peaks.size == 0:
            print("no")
            return []

        try:
            fine = fine_timing(buffer, self.long_ref, coarse.d_hats, coarse.cfo_hats,
                               self.config.SAMPLE_RATE, sps, cfg)
        except Exception as e:
            print(e)
            return []

        payload_starts = fine.sample_idxs + len(self.long_ref) - (self.config.SPS*self.config.SPAN)# - self.config.SPS, #Should fix this in fine_timing
        return [
            DetectionResult(
                payload_start=int(payload_starts[i]),
                cfo_estimate=float(coarse.cfo_hats[i]),
                phase_estimate=float(fine.phase_estimates[i]),
                confidence=float(coarse.m_peaks[i]),
            )
            for i in range(len(payload_starts))
        ]

    def decode(self, buffer: np.ndarray, cfo: float, phase_estimate: float) -> Packet:
        cfo_rad_per_symbol = 2 * np.pi * cfo / self.config.SAMPLE_RATE * self.config.SPS
        
        header, payload_start, current_phase_estimate = self.header_decode(buffer, cfo_rad_per_symbol, phase_estimate)
        if header.length == 0:
            msg = "Payload length = 0"
            raise ValueError(msg)
        payload = self.payload_decode(buffer, header, payload_start, cfo_rad_per_symbol, current_phase_estimate)
        return Packet(
            src_mac=header.src,
            dst_mac=header.dst,
            type=header.frame_type,
            seq_num=header.sequence_number,
            length=header.length,
            payload=payload,
            valid=header.crc_passed,
        )

    def header_decode(self, buffer: np.ndarray, cfo:float, current_phase_estimate: float) -> tuple[FrameHeader, int, float]:
        """Decode the header part of the packet. Assumes buffer input is already decimated."""
        header_end = 2 * self.frame_constructor.header_config.header_total_size + self.config.PRE_HEADER_GUARD_BITS

        if header_end*self.config.SPS > len(buffer):
            msg = "header end is outside of buffer"
            raise IndexError(msg)

        # applying the phase estimate from preamble before gardner
        print("current_pahse_estimate:",current_phase_estimate)

        if self.config.gardner_ted:
            guard = self.config.SPS//2
            header_syms = apply_gardner_ted(buffer[:header_end*self.config.SPS+guard], self.config.SPS)
        else:
            header_syms = decimate(buffer[:header_end*self.config.SPS], self.config.SPS)

        # costas correction
        if self.config.costas_loop:
            header_syms_corr, phase_est = apply_costas_loop(header_syms[:header_end], self.config.COSTAS_CONFIG, ModulationSchemes.BPSK, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)
        else:
            header_syms_corr, phase_est = header_syms[:header_end], [current_phase_estimate]
        #print(header_syms_corr[:9])
        
        # checks known pre header bits and flips syms and rotates phase estimate
        if self.config.PRE_HEADER_GUARD_BITS > 0:
            if np.mean(np.real(header_syms_corr[:self.config.PRE_HEADER_GUARD_BITS])) > 0:
                header_syms_corr = -header_syms_corr
                phase_est[-1] -= np.pi

        # demodulate header
        header_bits = self.bpsk.symbols2bits(header_syms_corr[self.config.PRE_HEADER_GUARD_BITS:])
        print("header_bits:", header_bits.flatten())
        header = self.frame_constructor.decode_header(header_bits)

        return header, header_end, phase_est[-1]

    def payload_decode(self, buffer: np.ndarray, header: FrameHeader, payload_start, cfo:float, current_phase_estimate: float) -> np.ndarray:
        payload_end = payload_start + (header.length*8)//(header.mod_scheme.value+1) # header.mod_scheme.value+1 is same as bits per symbol of modulator
        
        if payload_end*self.config.SPS > len(buffer):
            msg = "payload end is outside of buffer"
            raise IndexError(msg)
        
        if self.config.gardner_ted:
            guard = self.config.SPS
            rx_syms = apply_gardner_ted(buffer[payload_start*self.config.SPS:payload_end*self.config.SPS+guard], self.config.SPS)
        else:
            rx_syms = decimate(buffer[payload_start*self.config.SPS:payload_end*self.config.SPS], self.config.SPS)
        
        #print("rx_syms_real:", len(rx_syms), "rx_syms_ideal:", payload_end-payload_start)
        print("rx_syms:", rx_syms[:8])
        print("current_phase_estimate:", current_phase_estimate)

        if self.config.costas_loop:
            rx_syms, _ = apply_costas_loop(rx_syms[:payload_end-payload_start], self.config.COSTAS_CONFIG, header.mod_scheme, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)
        else:
            rx_syms = rx_syms[:payload_end-payload_start]

        print("rx_syms costas:", rx_syms[:8])

        print("modulation:", header.mod_scheme)
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
