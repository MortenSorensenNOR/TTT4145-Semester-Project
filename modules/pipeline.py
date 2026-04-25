from dataclasses import dataclass
from enum import IntEnum
from math import ceil
import numpy as np

from modules.pulse_shaping.pulse_shaping import *
from modules.modulators import *
from modules.frame_constructor.frame_constructor import *
from modules.golay import *
from modules.frame_sync.frame_sync import *
from modules.costas_loop.costas import *
from modules.ldpc.ldpc import LDPCConfig, ldpc_encode, ldpc_decode
from modules.ldpc.channel_coding import *
from modules.gardner_ted.gardner import *

from utils.plotting import *


# LDPC block sizes per code rate. We fix n=1944 (best coding gain in the
# 802.11 set, divides cleanly by 1/2/3 bits-per-symbol). k follows from rate.
_LDPC_BLOCK_PARAMS: dict[CodeRates, tuple[int, int]] = {
    CodeRates.TWO_THIRDS_RATE:   (1296, 1944),
    CodeRates.THREE_QUARTER_RATE:(1458, 1944),
    CodeRates.FIVE_SIXTH_RATE:   (1620, 1944),
}


def _on_air_payload_n_bits(pre_ldpc_n_bits: int, code_rate: CodeRates) -> tuple[int, int, int]:
    """For a given pre-LDPC payload length, return (n_codewords, k, n).

    For NONE: returns (0, 0, pre_ldpc_n_bits) — caller treats this as passthrough.
    """
    if code_rate == CodeRates.NONE:
        return (0, 0, pre_ldpc_n_bits)
    k, n = _LDPC_BLOCK_PARAMS[code_rate]
    n_cw = (pre_ldpc_n_bits + k - 1) // k
    return (n_cw, k, n_cw * n)

import logging
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    SAMPLE_RATE: int = 4_000_000
    CENTER_FREQ: int = 2_410_000_000
    SPS: int = 4
    SPAN: int = 8
    RRC_ALPHA: np.float32 = np.float32(0.25)
    MOD_SCHEME: ModulationSchemes = ModulationSchemes.PSK8
    CODING_RATE: CodeRates = CodeRates.FIVE_SIXTH_RATE
    LDPC_MAX_ITER: int = 20
    PRE_HEADER_GUARD_BITS: int = 0
    GUARD_SYMS_LENGTH: int = 16

    SYNC_CONFIG = SynchronizerConfig()
    COSTAS_CONFIG = CostasConfig(0.008) # Bn=0.008 empirically optimal for PSK8 over coax
    # NDA Gardner (Rice 2009) — see modules/gardner_ted/gardner.py.
    # BnTs must stay narrow: with LDPC the systematic-bits region holds 1500+
    # consecutive constant-symbol BPSK pad, which gives the NDA TED nothing to
    # lock onto.  At BnTs >= 0.001 the loop wanders during this stretch and
    # corrupts the parity symbols that follow.
    GARDNER_BN_TS: float = 0.0005
    GARDNER_ZETA: float = 0.707
    GARDNER_L: int = 2             # TED smoothing half-length (window = 2L+1 symbols)

    pulse_shaping: bool = True
    pilots: bool = False
    costas_loop: bool = True
    gardner_ted: bool = True
    interleaving: bool = False
    cfo_correction: bool = True
    use_golay: bool = False
    # Two-stage sync: Schmidl-Cox coarse + long-ZC fine (default).
    # Set False to skip coarse_sync and detect frames via a single full-buffer
    # cross-correlation against the long ZC.  Cheaper preamble (no short reps),
    # ~9× more compute on x86, no CFO estimate (Costas must capture residual).
    # Only safe when CFO is small enough that the long-ZC peak stays sharp.
    two_stage_sync: bool = False
    # When True: TX skips software RRC convolution (just zero-inserts),
    # RX skips software match-filter — both assume the Pluto FPGA's
    # hardware RRC filter is active between the AD9363 and DMA.
    hardware_rrc: bool = False

class PacketType(IntEnum):
    """Frame types carried in the 2-bit `frame_type` header field.

    Values fit FrameHeaderConfig.frame_type_bits = 2 (range 0..3).
    """
    DATA = 0  # carries a TUN payload
    ACK  = 1  # cumulative ACK; seq_num = last in-order DATA seq received
    NAK  = 2  # reserved
    CTRL = 3  # reserved (link control / probe)


@dataclass
class Packet:
    """Packet of data to/from TAP/TUN"""
    src_mac: int = -1
    dst_mac: int = -1
    type: int = -1                                # one of PacketType
    seq_num: int = -1
    length: int = -1
    payload: np.ndarray = field(default_factory=lambda: np.ndarray([]))

    valid: bool = False
    err_reason: str = ""
    sample_start: int = -1   # payload_start within the buffer passed to receive()
    rx_symbols: np.ndarray | None = field(default=None)  # post-Costas PSK8 symbols (for diagnostics)

class TXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()
        self.frame_constructor.header_config.use_golay = config.use_golay

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

        self.guard_syms = np.zeros(config.GUARD_SYMS_LENGTH, dtype=np.complex64)
        # sync
        self.sync_syms = generate_preamble(self.config.SYNC_CONFIG)

        # Adding extra known bits before header to figure out phase ambiguity of BPSK header
        if config.PRE_HEADER_GUARD_BITS > 0:
            self.pre_header_guard_syms = self.bpsk.bits2symbols(np.array([[0]*config.PRE_HEADER_GUARD_BITS]))
        else:
            self.pre_header_guard_syms = np.array([])

    def _maybe_ldpc_encode(self, payload_bits: np.ndarray, code_rate: CodeRates) -> np.ndarray:
        """Pad payload to a multiple of k and emit n_cw concatenated codewords."""
        flat = payload_bits.ravel().astype(np.uint8)
        if code_rate == CodeRates.NONE:
            return flat
        n_cw, k, n_air = _on_air_payload_n_bits(len(flat), code_rate)
        n = n_air // n_cw
        pad = n_cw * k - len(flat)
        msg = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)]) if pad else flat
        cfg = LDPCConfig(k=k, code_rate=code_rate)
        coded = np.empty(n_cw * n, dtype=np.uint8)
        for i in range(n_cw):
            coded[i*n:(i+1)*n] = ldpc_encode(msg[i*k:(i+1)*k], cfg).astype(np.uint8)
        return coded

    def transmit(self, packet: Packet) -> np.ndarray:
        # Empty-payload control frames (e.g. ARQ ACKs) skip LDPC: there's
        # nothing but the 16-bit CRC to protect, so the codeword overhead
        # is wasteful — keep them uncoded by forcing CodeRates.NONE.
        coding_rate = (
            self.config.CODING_RATE if packet.length > 0 else CodeRates.NONE
        )

        # construct bits
        header = FrameHeader(
            length=packet.length,
            src=packet.src_mac,
            dst=packet.dst_mac,
            frame_type=packet.type,
            mod_scheme=self.config.MOD_SCHEME,
            sequence_number=packet.seq_num,
            coding_rate=coding_rate.value,
        )
        (header_bits, payload_bits) = self.frame_constructor.encode(header, packet.payload)

        # LDPC encode payload (data + CRC + pad-to-12) into n-bit codewords.
        payload_for_mod = self._maybe_ldpc_encode(payload_bits, coding_rate)

        # modulate
        header_syms = self.bpsk.bits2symbols(header_bits)
        payload_syms = self.payload_modulator.bits2symbols(payload_for_mod.reshape(-1, header.mod_scheme.value+1))

        # construct signal
        tx_syms = np.concatenate([self.guard_syms,self.sync_syms, self.pre_header_guard_syms, header_syms, payload_syms,self.guard_syms])

        # upsample and filter
        # hardware_rrc=True: FPGA does 4× polyphase interpolation + RRC shaping,
        # so we send raw baseband symbols (1 sample/symbol) with no upsampling.
        # The DMA will drain at 1 MHz (symbol rate) rather than 4 MHz; buffer
        # timing must be calculated in symbols, not samples.
        # hardware_rrc=False: do full software upsample + RRC convolution.
        if self.config.hardware_rrc:
            tx_signal = tx_syms
        else:
            tx_signal = upsample(tx_syms, self.config.SPS, self.rrc_taps)
        return tx_signal

@dataclass
class DetectionResult:
    """Single frame detection result."""

    payload_start: int
    cfo_estimate: np.float32
    phase_estimate: np.float32
    confidence: np.float32

class RXPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.frame_constructor = FrameConstructor()
        self.frame_constructor.header_config.use_golay = config.use_golay

        self.num_taps = 2 * config.SPS * config.SPAN + 1
        self.rrc_taps = rrc_filter(config.SPS, config.RRC_ALPHA, self.num_taps)

        self.bpsk = BPSK()
        self.qpsk = QPSK()
        self.psk8 = PSK8()

        # generate the known long preamble for matched filtering
        self.long_ref = build_long_ref(self.config.SYNC_CONFIG, self.config.SPS, self.rrc_taps)
        self.ref_f = build_fine_ref(self.long_ref, self.config.SYNC_CONFIG, self.config.SPS)

        # decimated (symbol-rate) reference for fine timing on the decimated buffer
        self.long_ref_dec = decimate(self.long_ref, self.config.SPS)
        self.ref_f_dec = build_fine_ref(self.long_ref_dec, self.config.SYNC_CONFIG, 1)

        # Time-reversed conjugate ref for the single-stage detector path.
        self.long_ref_rev = build_long_ref_rev(self.long_ref)

    def receive(self, buffer: np.ndarray, search_from: int = 0) -> tuple[list[Packet], int]:
        """Detect and decode all frames in buffer.

        search_from: sample offset into buffer where detection begins.  Samples
        before this index are ignored, which prevents re-detecting frames that
        were already decoded in a previous sliding-window iteration.

        Returns (packets, max_detection_sample) where max_detection_sample is the
        absolute position of the last detection attempted (success or failure).
        Callers should advance search_from to at least this position to avoid
        re-detecting packets that already failed decode.
        """
        search_buf = buffer[search_from:]
        # RX always uses software matched filter; FPGA no longer has an RRC
        # filter on the RX path (it was removed to save DSP resources).
        filtered_buffer = match_filter(search_buf, self.rrc_taps)
        detections = self.detect(filtered_buffer)
        if not detections:
            return [], search_from

        logger.debug(f"Detected {len(detections)} packets\n\t Cfo's: {[float(det.cfo_estimate) for det in detections]}'")

        packets = []
        n_decode_errors = 0
        max_detection_sample = search_from  # track furthest detection attempted
        for det in detections:
            abs_payload_start = search_from + det.payload_start
            rx_syms = filtered_buffer[det.payload_start:]
            try:
                decoded_packet = self.decode(rx_syms, det.cfo_estimate, det.phase_estimate)
                decoded_packet.sample_start = abs_payload_start
                packets.append(decoded_packet)
                max_detection_sample = max(max_detection_sample, abs_payload_start)
            except IndexError as e:
                # Tail cutoff: this frame extends beyond the buffer. Stop
                # processing remaining detections so they stay eligible for
                # retry on the next iteration (once more data is appended).
                # Without this break, a shorter frame detected at a higher
                # position could still decode and advance max_det past this
                # cutoff — permanently losing it. Matters for variable-length
                # payloads; with fixed lengths every later detection would
                # also cut off, so the break is a no-op.
                n_decode_errors += 1
                logger.debug(f"DECODE ERROR (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f}): {type(e).__name__}: {e}")
                break
            except Exception as e:
                max_detection_sample = max(max_detection_sample, abs_payload_start)
                n_decode_errors += 1
                logger.debug(f"DECODE ERROR (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f}): {type(e).__name__}: {e}")
            logger.debug(f"DECODE SUCCESS (cfo={det.cfo_estimate:.0f} Hz, ratio={det.confidence:.1f})")

        if n_decode_errors:
            logger.info(f"{n_decode_errors}/{len(detections)} detections failed decode")

        return packets, max_detection_sample

    def detect(self, filtered_buffer: np.ndarray) -> list[DetectionResult]:
        """Detect frames in a match-filtered buffer.

        two_stage_sync=True (default): Schmidl-Cox coarse + long-ZC fine.  Both
            stages run on the full-rate filtered buffer; fine timing preserves
            sub-symbol precision needed for correct decimation in decode().
        two_stage_sync=False: skip coarse, just convolve the buffer with the
            long-ZC reference and pick peaks.  No CFO estimate — Costas captures.
        """
        cfg = self.config.SYNC_CONFIG
        sps = self.config.SPS

        try:
            if self.config.two_stage_sync:
                coarse = coarse_sync(filtered_buffer, self.config.SAMPLE_RATE, sps, cfg)
                if coarse.m_peaks.size == 0:
                    return []
                fine = fine_timing(filtered_buffer, self.long_ref, coarse.d_hats, coarse.cfo_hats,
                                   self.config.SAMPLE_RATE, sps, cfg, self.ref_f)
                cfo_hats = coarse.cfo_hats
                # peak_ratio is a peak-to-mean ratio over a small fine_timing
                # window — gate via fine_peak_ratio_min (default 3.0).
                gate = lambda r: cfg.fine_peak_ratio_min <= 0 or r >= cfg.fine_peak_ratio_min
            else:
                fine, cfo_hats = full_buffer_xcorr_sync(
                    filtered_buffer, self.long_ref, self.long_ref_rev,
                    float(cfg.single_stage_ncc_threshold), self.config.SAMPLE_RATE,
                )
                if fine.sample_idxs.size == 0:
                    return []
                # peak_ratio is the NCC ∈ [0,1]; threshold is already applied
                # inside the helper, so no extra gate here.
                gate = lambda r: True
        except Exception as e:
            logger.info(e)
            return []

        payload_starts = fine.sample_idxs + len(self.long_ref)
        return [
            DetectionResult(
                payload_start=int(payload_starts[i]),
                cfo_estimate=np.float32(cfo_hats[i]),
                phase_estimate=np.float32(fine.phase_estimates[i]),
                confidence=np.float32(fine.peak_ratios[i]),
            )
            for i in range(len(payload_starts))
            if gate(fine.peak_ratios[i])
        ]

    def decode(self, buffer: np.ndarray, cfo: np.float32, phase_estimate: np.float32) -> Packet:
        cfo_rad_per_symbol = np.float32(2 * np.pi * float(cfo) / self.config.SAMPLE_RATE * self.config.SPS)

        header, payload_start, current_phase_estimate, current_timing_estimate = self.header_decode(buffer, cfo_rad_per_symbol, phase_estimate)

        # if header.crc_passed:
        #     logger.debug(f"HEADER: crc: {header.crc_passed}, header: length {header.length} bytes, coding rate: {header.coding_rate}, type: {header.frame_type}")

        logger.info(f"HEADER: crc: {header.crc_passed}, header: length {header.length} bytes, coding rate: {header.coding_rate}, type: {header.frame_type}")

        # length=0 is legitimate for control frames (e.g. ARQ ACKs). Skip
        # payload_decode for them — there is nothing but the 16-bit CRC to
        # verify, and payload_decode would return an empty array anyway.
        if header.length == 0:
            payload = np.empty((0, 1), dtype=int)
            rx_symbols = np.empty(0, dtype=np.complex64)
        else:
            payload, rx_symbols = self.payload_decode(buffer, header, payload_start, cfo_rad_per_symbol, current_phase_estimate, current_timing_estimate)

        return Packet(
            src_mac=header.src,
            dst_mac=header.dst,
            type=header.frame_type,
            seq_num=header.sequence_number,
            length=header.length,
            payload=payload,
            valid=header.crc_passed,
            rx_symbols=rx_symbols,
        )

    def header_decode(self, buffer: np.ndarray, cfo:np.float32, current_phase_estimate: np.float32) -> tuple[FrameHeader, int, np.float32, np.float32]:
        """Decode the header part of the packet. Assumes buffer input is already decimated."""
        # Use header_encoded_n_bits (rounded to even) instead of raw header_total_size
        # so payload_start matches what TX actually emits — the encoder pads odd-length
        # headers up by one bit.
        header_end = self.frame_constructor.header_encoded_n_bits + self.config.PRE_HEADER_GUARD_BITS

        if header_end*self.config.SPS > len(buffer):
            msg = "header end is outside of buffer"
            raise IndexError(msg)

        if self.config.gardner_ted:
            # New NDA gardner has two quirks vs the old one:
            #   1) intrinsic 1-sample bias — its first output is signal[1],
            #      not signal[0]. Prepend a duplicate of the first sample so
            #      the bias cancels and outputs land on the symbol peaks.
            #   2) output is (Ns-1) symbols short of naïve decimation —
            #      extend the trailing guard by sps*sps so downstream
            #      slicing has enough symbols.
            guard = self.config.SPS * self.config.SPS
            gardner_in = buffer[:header_end*self.config.SPS+guard]
            gardner_in = np.concatenate([gardner_in[:1], gardner_in])
            header_syms = apply_gardner_ted(
                gardner_in,
                self.config.SPS,
                BnTs=self.config.GARDNER_BN_TS,
                zeta=self.config.GARDNER_ZETA,
                L=self.config.GARDNER_L,
            )
            timing_est = [0.0]  # new NDA gardner does not expose state for handoff
        else:
            header_syms, timing_est = decimate(buffer[:header_end*self.config.SPS], self.config.SPS), [0.0]

        # If guard pilots are present, use them as known BPSK symbols (-1+0j) to compute
        # a noise-averaged ML phase estimate.  This replaces the Costas seed from fine_timing
        # and resolves BPSK π ambiguity before the header Costas even starts.
        # ML: angle( Σ r_k · conj(s_k) ) = angle( Σ r_k · (-1) ) = angle( -mean(guard) )
        n_guard = self.config.PRE_HEADER_GUARD_BITS
        if n_guard > 0:
            current_phase_estimate = np.float32(np.angle(np.mean(-header_syms[:n_guard])))

        # costas correction on header symbols only (guard already used for phase estimation)
        header_only = header_syms[n_guard:header_end]
        if self.config.costas_loop:
            header_syms_corr, phase_est = apply_costas_loop(header_only, self.config.COSTAS_CONFIG, ModulationSchemes.BPSK, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)
        else:
            header_syms_corr, phase_est = header_only * np.exp(-1j*current_phase_estimate), [current_phase_estimate]

        # demodulate header
        header_bits = self.bpsk.symbols2bits(header_syms_corr)

        # Resolve BPSK π phase ambiguity: try both polarities.
        # If normal polarity fails CRC, flip all bits (≡ +π phase) and retry.
        # False-pass probability with CRC-8 is only 1/256.
        try:
            header = self.frame_constructor.decode_header(header_bits)
        except Exception:
            header = self.frame_constructor.decode_header(1 - header_bits)
            # Costas locked π off — correct the phase estimate so payload
            # Costas loop gets the right starting point.
            phase_est[-1] = np.float32(phase_est[-1]) - np.pi

        return header, header_end, np.float32(phase_est[-1] % (2 * np.pi)), np.float32(timing_est[-1])

    def payload_decode(self, buffer: np.ndarray, header: FrameHeader, payload_start, cfo:np.float32, current_phase_estimate: np.float32, current_timing_estimate: np.float32) -> tuple[np.ndarray, np.ndarray]:
        bps = header.mod_scheme.value + 1
        code_rate = CodeRates(header.coding_rate)

        # Symbol count = on-air bit count / bits-per-symbol.  With LDPC the
        # on-air count is (codewords × n) instead of just (data + CRC).
        pre_ldpc_n_bits = header.length * 8 + self.frame_constructor.PAYLOAD_CRC_BITS
        _n_cw, _k, n_air_bits = _on_air_payload_n_bits(pre_ldpc_n_bits, code_rate)
        payload_end = payload_start + ceil(n_air_bits / bps)

        if payload_end*self.config.SPS > len(buffer):
            msg = "payload end is outside of buffer"
            raise IndexError(msg)

        if self.config.gardner_ted:
            # See header_decode for why we prepend a duplicate sample and use
            # a sps*sps trailing guard.
            guard = self.config.SPS * self.config.SPS
            gardner_in = buffer[payload_start*self.config.SPS:payload_end*self.config.SPS+guard]
            gardner_in = np.concatenate([gardner_in[:1], gardner_in])
            rx_syms = apply_gardner_ted(
                gardner_in,
                self.config.SPS,
                BnTs=self.config.GARDNER_BN_TS,
                zeta=self.config.GARDNER_ZETA,
                L=self.config.GARDNER_L,
            )
            timing_est = [0.0]  # new NDA gardner does not expose state for handoff
        else:
            rx_syms, timing_est = decimate(buffer[payload_start*self.config.SPS:payload_end*self.config.SPS], self.config.SPS), [0.0]

        if self.config.costas_loop:
            rx_syms, phase_est = apply_costas_loop(rx_syms[:payload_end-payload_start], self.config.COSTAS_CONFIG, header.mod_scheme, current_phase_estimate=current_phase_estimate, current_frequency_offset=cfo)
        else:
            rx_syms = rx_syms[:payload_end-payload_start]*np.exp(-1j*current_phase_estimate)

        match header.mod_scheme:
            case ModulationSchemes.BPSK: mod = self.bpsk
            case ModulationSchemes.QPSK: mod = self.qpsk
            case ModulationSchemes.PSK8: mod = self.psk8
        logger.debug(f"Modulation scheme: {header.mod_scheme}")

        if code_rate == CodeRates.NONE:
            # Hard-decision path (matches pre-LDPC behaviour).
            payload_bits_encoded = mod.symbols2bits(rx_syms).ravel()
            payload_bits = self.frame_constructor.decode_payload(header, payload_bits_encoded)
            return payload_bits.reshape(-1, 1), rx_syms

        # Soft demap → LDPC decode → strip LDPC pad → frame_constructor.decode_payload.
        llrs_per_sym = mod.symbols2llrs(rx_syms)              # (n_syms, bps)
        llrs = llrs_per_sym.ravel()[:n_air_bits]              # (n_air_bits,)
        n_cw = _n_cw
        k = _k
        n = n_air_bits // n_cw
        cfg = LDPCConfig(k=k, code_rate=code_rate)
        decoded = np.empty(n_cw * k, dtype=np.uint8)
        for i in range(n_cw):
            decoded[i*k:(i+1)*k] = ldpc_decode(
                llrs[i*n:(i+1)*n], cfg,
                max_iterations=self.config.LDPC_MAX_ITER,
            ).astype(np.uint8)

        # Trim the zero-pad we appended on TX to align with LDPC k.
        payload_bits_pre_ldpc = decoded[:pre_ldpc_n_bits]
        payload_bits = self.frame_constructor.decode_payload(header, payload_bits_pre_ldpc)
        return payload_bits.reshape(-1, 1), rx_syms

if __name__ == "__main__":
    config = PipelineConfig()
    tx_pipe = TXPipeline(config)
    rx_pipe = RXPipeline(config)
