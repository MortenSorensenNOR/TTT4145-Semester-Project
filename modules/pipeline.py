from dataclasses import dataclass
from enum import IntEnum
from math import ceil
import numpy as np

from modules.pulse_shaping.pulse_shaping import *
from modules.modulators.modulators import *
from modules.frame_constructor.frame_constructor import *
from modules.frame_sync.frame_sync import *
from modules.costas_loop.costas import *
from modules.ldpc.ldpc import LDPCConfig, ldpc_encode_batch, ldpc_decode_batch
from modules.ldpc.channel_coding import *
from modules.nda_ted.nda_ted import *

from utils.plotting import *

import logging
logger = logging.getLogger(__name__)


_LDPC_BLOCK_PARAMS: dict[CodeRates, tuple[int, int]] = {
    CodeRates.TWO_THIRDS_RATE:   (1296, 1944),
    CodeRates.THREE_QUARTER_RATE:(1458, 1944),
    CodeRates.FIVE_SIXTH_RATE:   (1620, 1944),
}


def _on_air_payload_n_bits(pre_ldpc_n_bits: int, code_rate: CodeRates) -> tuple[int, int, int]:
    """For a given pre-LDPC payload length, return (n_codewords, k, n)."""
    if code_rate == CodeRates.NONE:
        return (0, 0, pre_ldpc_n_bits)
    k, n = _LDPC_BLOCK_PARAMS[code_rate]
    n_cw = (pre_ldpc_n_bits + k - 1) // k
    return (n_cw, k, n_cw * n)


# Pre-generated random LDPC padding
_LDPC_PAD_RNG  = np.random.default_rng(seed=0xA5A5A5A5)
_LDPC_PAD_BITS = _LDPC_PAD_RNG.integers(0, 2, max(_LDPC_BLOCK_PARAMS.values(), key=lambda kn: kn[0])[0], dtype=np.uint8)

# Bit-level whitener PRBS
_SCRAMBLE_RNG  = np.random.default_rng(seed=0xC3C3C3C3)
_SCRAMBLE_BITS = _SCRAMBLE_RNG.integers(0, 2, 1 << 16, dtype=np.uint8)


@dataclass
class PipelineConfig:
    SAMPLE_RATE: int = 6_000_000
    CENTER_FREQ: int = 2_410_000_000
    SPS: int = 4
    SPAN: int = 8
    RRC_ALPHA: np.float32 = np.float32(0.25)
    MOD_SCHEME: ModulationSchemes = ModulationSchemes.PSK8
    CODING_RATE: CodeRates = CodeRates.FIVE_SIXTH_RATE
    LDPC_MAX_ITER: int = 20
    GUARD_SYMS_LENGTH: int = 16

    SYNC_CONFIG = SynchronizerConfig()
    COSTAS_CONFIG = CostasConfig(
        loop_noise_bandwidth_normalized=0.01593,
        damping_factor=0.7071
    )

    NDA_BN_TS: float = 0.006489
    NDA_ZETA: float = 0.7071
    NDA_L: int = 19

    pulse_shaping: bool = True
    costas_loop: bool = True
    nda_ted: bool = True
    interleaving: bool = False
    cfo_correction: bool = True


class PacketType(IntEnum):
    """Frame types carried in the 2-bit `frame_type` header field."""
    DATA = 0  # carries a TUN payload
    ACK  = 1  # cumulative ACK; seq_num = last in-order DATA seq received
    NAK  = 2  # raw packets; no arq
    CTRL = 3  # reserved (link control / probe)


@dataclass
class Packet:
    """Packet of data to/from TAP/TUN"""
    src_mac: int = -1
    dst_mac: int = -1
    type:    int = -1
    seq_num: int = -1
    length:  int = -1
    payload: np.ndarray = field(default_factory=lambda: np.ndarray([]))
    mod_scheme: ModulationSchemes | None = None
    coding_rate: CodeRates | None = None

    valid:        bool = False
    err_reason:   str = ""
    sample_start: int = -1
    rx_symbols: np.ndarray | None = field(default=None)  # for diagnostics

class TXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

        self.bpsk  = BPSK()
        self.qpsk  = QPSK()
        self.psk8  = PSK8()
        self.psk16 = PSK16()
        self._modulators = {
            ModulationSchemes.BPSK:  self.bpsk,
            ModulationSchemes.QPSK:  self.qpsk,
            ModulationSchemes.PSK8:  self.psk8,
            ModulationSchemes.PSK16: self.psk16,
        }
        self.payload_modulator = self._modulators[self.config.MOD_SCHEME]

        self.num_taps = 2 * config.SPS * config.SPAN + 1
        self.rrc_taps = rrc_filter(config.SPS, config.RRC_ALPHA, self.num_taps)
        self.guard_syms = np.zeros(config.GUARD_SYMS_LENGTH, dtype=np.complex64)

        self.sync_syms = generate_preamble(self.config.SYNC_CONFIG)

    def _ldpc_encode(self, payload_bits: np.ndarray, code_rate: CodeRates) -> np.ndarray:
        """Pad payload to a multiple of k and emit n_cw concatenated codewords."""
        flat = payload_bits.ravel().astype(np.uint8)
        if code_rate == CodeRates.NONE:
            return flat

        # Pad to size
        n_cw, k, _ = _on_air_payload_n_bits(len(flat), code_rate)
        pad = n_cw * k - len(flat)
        msg = np.concatenate([flat, _LDPC_PAD_BITS[:pad]]) if pad else flat

        # Scramble
        msg = msg ^ _SCRAMBLE_BITS[:len(msg)]
        cfg = LDPCConfig(k=k, code_rate=code_rate)

        # Return encoded payload
        return ldpc_encode_batch(msg, cfg).ravel()

    def transmit(self, packet: Packet) -> np.ndarray:
        mod_scheme = packet.mod_scheme if packet.mod_scheme is not None else self.config.MOD_SCHEME
        if packet.coding_rate is not None:
            coding_rate = packet.coding_rate
        else:
            coding_rate = self.config.CODING_RATE if packet.length > 0 else CodeRates.NONE

        # construct bits
        header = FrameHeader(
            length=packet.length,
            src=packet.src_mac,
            dst=packet.dst_mac,
            frame_type=packet.type,
            mod_scheme=mod_scheme,
            sequence_number=packet.seq_num,
            coding_rate=coding_rate.value,
        )
        (header_bits, payload_bits) = self.frame_constructor.encode(header, packet.payload)

        # LDPC encode
        payload_for_mod = self._ldpc_encode(payload_bits, coding_rate)

        # modulate
        payload_modulator = self._modulators[mod_scheme]
        header_syms = self.bpsk.bits2symbols(header_bits)
        payload_syms = payload_modulator.bits2symbols(payload_for_mod.reshape(-1, mod_scheme.value+1))

        # construct signal
        tx_syms = np.concatenate([self.guard_syms,self.sync_syms, header_syms, payload_syms,self.guard_syms])
        tx_signal = upsample(tx_syms, self.config.SPS, self.rrc_taps)
        return tx_signal

@dataclass
class DetectionResult:
    """Single frame detection result."""

    payload_start:  int
    cfo_estimate:   np.float32
    phase_estimate: np.float32
    confidence:     np.float32

class RXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()

        self.num_taps = 2 * config.SPS * config.SPAN + 1
        self.rrc_taps = rrc_filter(config.SPS, config.RRC_ALPHA, self.num_taps)

        self.bpsk  = BPSK()
        self.qpsk  = QPSK()
        self.psk8  = PSK8()
        self.psk16 = PSK16()

        # generate the known preamble for matched filtering
        self.preamble_ref = build_preamble_ref(self.config.SYNC_CONFIG, self.config.SPS, self.rrc_taps)
        self.preamble_ref_rev = build_preamble_ref_rev(self.preamble_ref)

        # diagnostics
        self.last_payload_failures: int = 0
        self.last_tail_cutoffs:     int = 0

    def receive(self, buffer: np.ndarray, search_from: int = 0) -> tuple[list[Packet], int]:
        search_buf = buffer[search_from:]
        filtered_buffer = match_filter(search_buf, self.rrc_taps)
        detections = self.detect(filtered_buffer)
        if not detections:
            return [], search_from

        logger.debug(f"Detected {len(detections)} packets\n\t Cfo's: {[float(det.cfo_estimate) for det in detections]}'")

        packets = []
        n_payload_failures = 0 
        n_tail_cutoffs     = 0
        max_detection_sample = search_from
        for det in detections:
            abs_payload_start = search_from + det.payload_start
            rx_syms = filtered_buffer[det.payload_start:]
            try:
                decoded_packet = self.decode(rx_syms, det.cfo_estimate, det.phase_estimate)
                decoded_packet.sample_start = abs_payload_start
                packets.append(decoded_packet)
                max_detection_sample = max(max_detection_sample, abs_payload_start)
            except IndexError as e:
                n_tail_cutoffs += 1
                logger.debug(f"DECODE ERROR (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f}): {type(e).__name__}: {e}")
                break
            except Exception as e:
                max_detection_sample = max(max_detection_sample, abs_payload_start)
                n_payload_failures += 1
                logger.debug(f"DECODE ERROR (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f}): {type(e).__name__}: {e}")
            logger.debug(f"DECODE SUCCESS (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f})")

        n_decode_errors = n_payload_failures + n_tail_cutoffs
        if n_decode_errors:
            logger.info(f"{n_decode_errors}/{len(detections)} detections failed decode "
                        f"({n_payload_failures} payload, {n_tail_cutoffs} tail-cutoff)")

        self.last_payload_failures = n_payload_failures
        self.last_tail_cutoffs     = n_tail_cutoffs
        return packets, max_detection_sample

    def detect(self, filtered_buffer: np.ndarray) -> list[DetectionResult]:
        """Detect frames in a match-filtered buffer via full-buffer xcorr against the long-ZC reference."""
        cfg = self.config.SYNC_CONFIG

        try:
            fine, cfo_hats = full_buffer_xcorr_sync(
                filtered_buffer, self.preamble_ref, self.preamble_ref_rev,
                float(cfg.ncc_threshold), self.config.SAMPLE_RATE,
            )
            if fine.sample_idxs.size == 0:
                return []
        except Exception as e:
            logger.info(e)
            return []

        payload_starts = fine.sample_idxs + len(self.preamble_ref)
        return [
            DetectionResult(
                payload_start=int(payload_starts[i]),
                cfo_estimate=np.float32(cfo_hats[i]),
                phase_estimate=np.float32(fine.phase_estimates[i]),
                confidence=np.float32(fine.peak_ratios[i]),
            )
            for i in range(len(payload_starts))
        ]

    def decode(self, buffer: np.ndarray, cfo: np.float32, phase_estimate: np.float32) -> Packet:
        cfo_rad_per_symbol = np.float32(2 * np.pi * float(cfo) / self.config.SAMPLE_RATE * self.config.SPS)

        header, payload_start, current_phase_estimate = self.header_decode(buffer, cfo_rad_per_symbol, phase_estimate)

        if header.length == 0:
            payload = np.empty((0, 1), dtype=int)
            rx_symbols = np.empty(0, dtype=np.complex64)
        else:
            payload, rx_symbols = self.payload_decode(buffer, header, payload_start, cfo_rad_per_symbol, current_phase_estimate)

        return Packet(
            src_mac=header.src,
            dst_mac=header.dst,
            type=header.frame_type,
            seq_num=header.sequence_number,
            length=header.length,
            payload=payload,
            valid=header.crc_passed,
            rx_symbols=rx_symbols,
            mod_scheme=header.mod_scheme,
        )

    def header_decode(self, buffer: np.ndarray, cfo:np.float32, current_phase_estimate: np.float32) -> tuple[FrameHeader, int, np.float32]:
        """Decode the header part of the packet. Assumes buffer input is already decimated."""
        header_end = self.frame_constructor.header_encoded_n_bits
        if header_end*self.config.SPS > len(buffer):
            msg = "header end is outside of buffer"
            raise IndexError(msg)

        if self.config.nda_ted:
            guard = self.config.SPS * self.config.SPS
            nda_in = buffer[:header_end*self.config.SPS+guard]
            header_syms = apply_nda_ted(
                nda_in,
                self.config.SPS,
                BnTs=self.config.NDA_BN_TS,
                zeta=self.config.NDA_ZETA,
                L=self.config.NDA_L,
                prepend_first=True,
            )
        else:
            header_syms = decimate(buffer[:header_end*self.config.SPS], self.config.SPS)

        # costas correction on header symbols only
        header_only = header_syms[:header_end]
        header_syms_corr, phase_est = apply_costas_loop(header_only, self.config.COSTAS_CONFIG, ModulationSchemes.BPSK, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)

        # demodulate header
        header_bits = self.bpsk.symbols2bits(header_syms_corr)

        # try both polarities, just in case the first fails
        try:
            header = self.frame_constructor.decode_header(header_bits)
        except Exception:
            header = self.frame_constructor.decode_header(1 - header_bits)
            phase_est[-1] = np.float32(phase_est[-1]) - np.pi

        return header, header_end, np.float32(phase_est[-1] % (2 * np.pi))

    def payload_decode(self, buffer: np.ndarray, header: FrameHeader, payload_start, cfo:np.float32, current_phase_estimate: np.float32) -> tuple[np.ndarray, np.ndarray]:
        bps = header.mod_scheme.value + 1
        code_rate = CodeRates(header.coding_rate)

        pre_ldpc_n_bits = header.length * 8 + self.frame_constructor.PAYLOAD_CRC_BITS
        _n_cw, _k, n_air_bits = _on_air_payload_n_bits(pre_ldpc_n_bits, code_rate)
        payload_end = payload_start + ceil(n_air_bits / bps)

        if payload_end*self.config.SPS > len(buffer):
            msg = "payload end is outside of buffer"
            raise IndexError(msg)

        if self.config.nda_ted:
            guard = self.config.SPS * self.config.SPS
            nda_in = buffer[payload_start*self.config.SPS:payload_end*self.config.SPS+guard]
            rx_syms = apply_nda_ted(
                nda_in,
                self.config.SPS,
                BnTs=self.config.NDA_BN_TS,
                zeta=self.config.NDA_ZETA,
                L=self.config.NDA_L,
                prepend_first=True,
            )
        else:
            rx_syms = decimate(buffer[payload_start*self.config.SPS:payload_end*self.config.SPS], self.config.SPS)

        rx_syms, _ = apply_costas_loop(rx_syms[:payload_end-payload_start], self.config.COSTAS_CONFIG, header.mod_scheme, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)

        match header.mod_scheme:
            case ModulationSchemes.BPSK:  mod = self.bpsk
            case ModulationSchemes.QPSK:  mod = self.qpsk
            case ModulationSchemes.PSK8:  mod = self.psk8
            case ModulationSchemes.PSK16: mod = self.psk16
        logger.debug(f"Modulation scheme: {header.mod_scheme}")

        if code_rate == CodeRates.NONE:
            payload_bits_encoded = mod.symbols2bits(rx_syms).ravel()
            payload_bits = self.frame_constructor.decode_payload(header, payload_bits_encoded)
            return payload_bits.reshape(-1, 1), rx_syms

        # Soft demap → LDPC decode → strip LDPC pad → frame_constructor.decode_payload.
        llrs_per_sym = mod.symbols2llrs(rx_syms)
        llrs = llrs_per_sym.ravel()[:n_air_bits]
        n   = n_air_bits // _n_cw
        cfg = LDPCConfig(k=_k, code_rate=code_rate)
        decoded = ldpc_decode_batch(
            llrs.reshape(_n_cw, n), cfg,
            max_iterations=self.config.LDPC_MAX_ITER,
        ).ravel()

        # trim ldpc padding and descramble
        payload_bits_pre_ldpc = decoded[:pre_ldpc_n_bits] ^ _SCRAMBLE_BITS[:pre_ldpc_n_bits]
        payload_bits = self.frame_constructor.decode_payload(header, payload_bits_pre_ldpc)
        return payload_bits.reshape(-1, 1), rx_syms
