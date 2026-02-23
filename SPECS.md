# Tech specs  
  
## Requirements  
**Numerical requirements:**  
Desired datarate sustained:             800 kbps  
Minimum required datarate:              400 kbps  
Max bandwidth usage:                    5 MHz       # minimize interference with 2.4 GHz WiFi  
Frequency band:                         2.4 GHz ISM  
Desired link specification:             100 meters outdoors  
Minimum required link spec:             10 meters indoors  
Desired latency end-to-end:             100 ms  
Max allowed latency end-to-end:         300 ms  
Desired latency end-to-end jitter:      10 ms  
Max allowed latency end-to-end jitter:  50 ms  
Required BER (pre ARQ):                 10^-5  
Max allowed PER (after ARQ):            1%  
Baseline modulation order:              QPSK  
Minimum coding gain:                    4 dB  
  
**Other requirements:**  
Should not interfere with existing 2.4 GHz WiFi deployments at the university  
Not required to account for moving nodes  
Two-way communication in full-duplex mode, with FDD  
Pluto presented as normal network interface to Linux  
Support video streaming at 480p (low bitrate)  
  
**Nice to haves**  
Dynamic modulation order and coding rate adjustments based on channel condition  
Multi-frame ARQ  
  
## System Realization  
Modulation:             QPSK  
Coding rate:            3/4  
Symbol rate:            667 ksym/s  
SPS:                    8  
SPAN:                   8  
RRC alpha:              0.35  
RRC taps:               2 * SPS * SPAN + 1 = 129  
Symbol period:          1.67us  
Occupied bandwidth:     900 kHz  
Throughput:             1 Mbps  
Channel coding:         LDPC with 802.11 base matrices (proven, existing implementation)  
Coding gain:            6 dB (TBD for 2/3 rate, but 8 dB at 1/2 rate)  
  
Link margin:                                  15.5 dB  
FDD channel separation (for initial testing): 50 MHz  
