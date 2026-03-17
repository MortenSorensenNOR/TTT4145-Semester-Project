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

from utils.plotting import *

PLOTTING = False
MOD_SCHEMES = [ModulationSchemes.BPSK, ModulationSchemes.QPSK, ModulationSchemes.PSK8]

@composite
def random_pipeline_config(draw):
    return PipelineConfig(
        MOD_SCHEME = draw(st.sampled_from(MOD_SCHEMES))
    )

@composite
def random_packet_length(draw):
    return draw(st.integers(min_value=2**0, max_value=(2**12)))

# --- Tests ---
@given(pipeline_config = random_pipeline_config(), packet_length = random_packet_length())
def test_simple(pipeline_config, packet_length):

    tx = TXPipeline(pipeline_config)
    rx = RXPipeline(pipeline_config)
    
    rng   = np.random.default_rng(42)
    packet = Packet(
        src_mac=0,
        dst_mac=1,
        type=0,
        seq_num=0,
        length=packet_length,
        payload=rng.integers(0, 2, packet_length*8)
    )

    tx_signal = tx.transmit(packet)

    # --- Plotting code that should not be run with pytest ---
    if PLOTTING:
        sync_len = len(tx.sync_syms)*tx.config.SPS
        header_len_samples = rx.frame_constructor.header_config.header_total_size*tx.config.SPS

        fig, _ = plot_iq(tx_signal)
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal.png")

        #print(tx_signal.shape)
        fig, _ = plot_iq(tx_signal[:sync_len])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_sync.png")

        fig, _ = plot_iq(tx_signal[sync_len:sync_len+header_len_samples])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_header.png")
        
        fig, _ = plot_iq(tx_signal[sync_len+header_len_samples:])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_packet.png")

    rx_packets = rx.receive(tx_signal)
    
    # --- Working manual detection with perfect channel ---
    #rx_downsampled = downsample(tx_signal, tx.config.SPS, tx.rrc_taps)
    #print(rx_downsampled.shape)
    #rx_packets = [rx.decode(rx_downsampled, DetectionResult(payload_start=sync_len//tx.config.SPS, valid=True))]
    
    assert packet.payload.all() == rx_packets[0].payload.all()

if __name__ == "__main__":
    pipeline_config = PipelineConfig(MOD_SCHEME=ModulationSchemes.QPSK)
    test_simple(pipeline_config, 2**10)
