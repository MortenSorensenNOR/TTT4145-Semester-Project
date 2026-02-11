#!/usr/bin/env python3
"""Demo script showing QPSK soft decision (LLR) visualization."""

import matplotlib.pyplot as plt
from modules.modulation import QPSK

qpsk = QPSK()

# Plot LLR heatmaps at different noise levels
fig1, axes1 = qpsk.plot_llr_heatmap(sigma_sq=0.1, grid_size=100)
fig1.suptitle('QPSK LLR Heatmap (σ² = 0.1 - moderate noise)', y=1.02)

fig2, axes2 = qpsk.plot_llr_heatmap(sigma_sq=0.02, grid_size=100)
fig2.suptitle('QPSK LLR Heatmap (σ² = 0.02 - low noise)', y=1.02)

fig3, axes3 = qpsk.plot_llr_heatmap(sigma_sq=0.5, grid_size=100)
fig3.suptitle('QPSK LLR Heatmap (σ² = 0.5 - high noise)', y=1.02)

plt.show()
