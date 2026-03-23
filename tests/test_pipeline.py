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
    print(buffer_length/(len(tx_signal_packet)))
    num_of_packets = rng.integers(0, (2*buffer_length//3)//(len(tx_signal_packet)))+1
    
    tx_signal_list = []
    detections_list = []
    prev_detection_end = actual_delay+tx.config.SPS*tx.config.SPAN
    sync_len = len(tx.sync_syms)*tx.config.SPS

    for i in range(num_of_packets):
        zeros_before = np.zeros(rng.integers(0, 2*len(tx_signal_packet)//3), dtype=complex)
        zeros_after = np.zeros(len(tx_signal_packet)-len(zeros_before), dtype=complex)

        tx_signal_list.append(np.concat([zeros_before, tx_signal_packet, zeros_after]))

        current_detection = prev_detection_end+len(zeros_before)+sync_len
        detections_list.append(current_detection)
        prev_detection_end = current_detection+len(tx_signal_packet)+len(zeros_after)
            
    tx_signal = np.concat(tx_signal_list)
    if len(tx_signal) < buffer_length:
        tx_signal = np.concat([tx_signal, np.zeros(buffer_length-len(tx_signal)-actual_delay, dtype=complex)])
    else:
        tx_signal = tx_signal[:buffer_length-actual_delay]

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
    
    assert len(rx_packets) == num_of_packets
    for rx_packet in rx_packets:
        print(packet.payload.shape, rx_packet.payload.shape)

        print(packet.length, rx_packet.length)
        assert packet.length == rx_packet.length
        assert packet.payload.all() == rx_packet.payload.all()

if __name__ == "__main__":
    pipeline_config = PipelineConfig(MOD_SCHEME=ModulationSchemes.QPSK)
    run_pipeline(pipeline_config, 2**8, 2**14, 42, plotting=True)
