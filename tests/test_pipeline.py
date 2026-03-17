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
    return draw(st.integers(min_value=2**0, max_value=(2**9)))

# --- Tests ---

@given(pipeline_config = random_pipeline_config(), packet_length = random_packet_length())
def test_simple(pipeline_config, packet_length):
    run_pipeline(pipeline_config, packet_length)

def run_pipeline(pipeline_config, packet_length, plotting = False):

    snr = 15
    seed = 42
    actual_cfo = 235
    actual_delay = 0
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
    # need to have at least 305 symbol delay
    tx_signal = np.concat([np.zeros(305, dtype=complex), tx_signal, np.zeros(500, dtype=complex), tx_signal, np.zeros(300, dtype=complex)])
    # apply channel
    rx_signal = channel.apply(tx_signal)

    # --- Plotting code that should not be run with pytest ---
    if plotting:
        sync_len = len(tx.sync_syms)*tx.config.SPS+actual_delay
        header_len_samples = rx.frame_constructor.header_config.header_total_size*tx.config.SPS

        fig, _ = plot_iq(tx_signal)
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal.png")

        #print(tx_signal.shape)
        fig, _ = plot_iq(tx_signal[actual_delay:sync_len])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_sync.png")

        fig, _ = plot_iq(rx_signal[sync_len:sync_len+header_len_samples])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_header.png")
        
        fig, _ = plot_iq(rx_signal[sync_len+header_len_samples:])
        plt.savefig(f"tests/plots/pipeline/mod{MOD_SCHEMES.index(pipeline_config.MOD_SCHEME)}_len{packet_length}_tx_signal_packet.png")

    rx_packets = rx.receive(rx_signal)
    print(len(rx_packets))
    
    assert len(rx_packets) > 0
    for rx_packet in rx_packets:
        print(packet.payload.all() == rx_packet.payload.all())
        assert packet.payload.all() == rx_packet.payload.all()

if __name__ == "__main__":
    pipeline_config = PipelineConfig(MOD_SCHEME=ModulationSchemes.QPSK)
    run_pipeline(pipeline_config, 2**7)
