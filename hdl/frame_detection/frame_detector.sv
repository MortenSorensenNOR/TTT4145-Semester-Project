`timescale 1ns/1ps

module frame_detector #(
    parameter ZC_LEN       = 13,      // length of one ZC copy
    parameter THRESHOLD    = 26214,   // 0.8 * 2^15, normalized detection threshold
    parameter DATA_WIDTH   = 16,
    parameter ACC_WIDTH    = 40,      // wide enough to accumulate ZC_LEN products
    parameter CFO_WIDTH    = 16
) (
    input  logic                  clk,
    input  logic                  rstn,

    // ADC data interface (from rx_cpack, same signals as go to DMA)
    input  logic [DATA_WIDTH-1:0] adc_data_i0,
    input  logic                  adc_enable_i0,
    input  logic                  adc_valid_i0,
    input  logic [DATA_WIDTH-1:0] adc_data_q0,
    input  logic                  adc_enable_q0,
    input  logic                  adc_valid_q0,

    // Interrupt to Core1 (active low, pulses low for one clock on detection)
    output logic                  frame_detected_nirq,

    // CFO estimate: signed fixed-point, units = Hz * 2^CFO_FRAC_BITS
    // Valid only when frame_detected_nirq pulses low
    output logic [CFO_WIDTH-1:0]  cfo_estimate,

    // Detection metadata for CPU readback via AXI-Lite (future)
    output logic                  cfo_valid,        // strobes high same cycle as nirq pulse
    output logic [31:0]           detection_count   // increments each detection, for debug
);

    // Local parameters
    localparam DELAY_DEPTH  = ZC_LEN;           // samples to delay for autocorrelation
    localparam PROD_WIDTH   = 2*DATA_WIDTH + 1; // complex multiply output width
    localparam NORM_WIDTH   = ACC_WIDTH;

    // process data when both I and Q are valid
    logic sample_valid;
    assign sample_valid = adc_valid_i0 & adc_enable_i0 & adc_valid_q0 & adc_enable_q0;

    // Delay line

endmodule
