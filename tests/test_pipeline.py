from os import pipe
from numpy import shape
import matplotlib.pyplot as plt
import pytest
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.strategies import composite

from modules import pipeline
from modules.frame_constructor import ModulationSchemes
from modules.pipeline import *
from modules.pulse_shaping import *
from modules.channel import *

from utils.plotting import *

MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]

@composite
def random_pipeline_config(draw):
    return PipelineConfig(
        MOD_SCHEME = draw(st.sampled_from(MOD_SCHEMES)),
        SPS = 8
    )

@composite
def random_packet_length(draw):
    return draw(st.integers(min_value=2**6, max_value=(2**9)))

@composite
def random_buffer_length(draw):
    return draw(st.integers(min_value=2**10, max_value=2**18))

@composite
def random_seed_rng(draw):
    return draw(st.integers(min_value=10, max_value=80))

# --- Tests ---

@given(pipeline_config = random_pipeline_config(), packet_length = random_packet_length(), buffer_length = random_buffer_length(), seed_rng = random_seed_rng())
def test_simple(pipeline_config, packet_length, buffer_length, seed_rng):
    run_pipeline(pipeline_config, packet_length, buffer_length, seed_rng)

def run_pipeline(pipeline_config, packet_length, buffer_length, seed_rng, plotting = False):

    snr = 15
    seed = 42
    actual_cfo = 2350
    actual_delay = 0#2034
    channel_config = ChannelConfig(
        sample_rate=pipeline_config.SAMPLE_RATE,
        snr_db=snr,
        seed=seed,
        cfo_hz=actual_cfo,
        initial_phase_rad=np.random.default_rng(seed).uniform(0, 2 * np.pi),
        delay_samples=actual_delay,
    )
    channel = ChannelModel(channel_config)
    
    tx = TXPipeline(pipeline_config)
    rx = RXPipeline(pipeline_config)
    
    rng   = np.random.default_rng(seed_rng)
    packet = Packet(
        src_mac=0,
        dst_mac=1,
        type=0,
        seq_num=0,
        length=packet_length,
        payload=rng.integers(0, 2, packet_length*8)
    )
    tx_signal_packet = tx.transmit(packet)

    max_packets = (2 * buffer_length // 3) // len(tx_signal_packet)
    num_of_packets = int(rng.integers(1, max_packets + 1)) if max_packets >= 1 else 1

    tx_signal_list = []
    detections_list = []
    prev_detection_end = actual_delay + tx.config.SPS * tx.config.SPAN
    sync_len = len(tx.sync_syms) * tx.config.SPS
    packet_len = len(tx_signal_packet)

    for i in range(num_of_packets):
        max_zeros_before = 2 * len(tx_signal_packet) // 3
        zeros_before_len = int(rng.integers(0, max_zeros_before + 1))
        zeros_after_len = packet_len - zeros_before_len

        tx_signal_list.append(np.concat([np.zeros(zeros_before_len, dtype=complex), tx_signal_packet, np.zeros(zeros_after_len, dtype=complex)]))
        current_detection = prev_detection_end + zeros_before_len + sync_len
        
        detections_list.append(current_detection)
        prev_detection_end = current_detection + packet_len + zeros_after_len

    tx_signal = np.concat(tx_signal_list)

    # Trim or pad to exactly buffer_length
    if len(tx_signal) >= buffer_length:
        tx_signal = tx_signal[:buffer_length]
    else:
        tx_signal = np.concat([tx_signal, np.zeros(buffer_length - len(tx_signal), dtype=complex)])
        # apply channel
    rx_signal = channel.apply(tx_signal)
    #rx_signal = np.concat([np.zeros(actual_delay, dtype=complex),tx_signal])

    print("num_of_packets:",num_of_packets, "\nbuffer_length:",buffer_length, "\nsignal length = buffer length?",len(rx_signal)==buffer_length, "\nlist of correct detection indexes",detections_list)
    
    # --- Plotting code that should not be run with pytest ---
    if plotting:
        header_len_samples = rx.frame_constructor.header_config.header_total_size*rx.config.SPS
        packet_len_samples = packet.length*rx.config.SPS

        fig, _ = plot_iq(rx_signal)
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal.png")
        
        for i in detections_list:
            #print(tx_signal.shape)
            fig, _ = plot_iq(rx_signal[i-sync_len:i])
            plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_{i}_tx_signal_sync.png")

            fig, _ = plot_iq(rx_signal[i:i+header_len_samples])
            plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_{i}_tx_signal_header.png")
            
            fig, _ = plot_iq(rx_signal[i+header_len_samples:i+header_len_samples+packet_len_samples])
            plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_{i}_tx_signal_packet.png")

    rx_packets = rx.receive(rx_signal)
    print(len(rx_packets))
    
    assert len(rx_packets) == num_of_packets-(detections_list[-1]+packet_len>buffer_length)
    for rx_packet in rx_packets:
        print(packet.payload.shape, rx_packet.payload.shape)

        print(packet.length, rx_packet.length)
        assert packet.length == rx_packet.length
        assert packet.payload.all() == rx_packet.payload.all()

if __name__ == "__main__":
    pipeline_config = PipelineConfig(MOD_SCHEME=ModulationSchemes.QPSK)
    run_pipeline(pipeline_config, 2**6, 2**14, 40, plotting=True)
