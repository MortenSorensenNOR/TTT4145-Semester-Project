"""Demo script showing QPSK soft decision (LLR) visualization."""

import matplotlib.pyplot as plt

from modules.modulation import QPSK
from modules.plotting import plot_llr_heatmap

qpsk = QPSK()

# Plot LLR heatmaps at different noise levels
fig1, axes1 = plot_llr_heatmap(qpsk, sigma_sq=0.1, grid_size=100)
fig1.suptitle("QPSK LLR Heatmap (σ² = 0.1 - moderate noise)", y=1.02)

fig2, axes2 = plot_llr_heatmap(qpsk, sigma_sq=0.02, grid_size=100)
fig2.suptitle("QPSK LLR Heatmap (σ² = 0.02 - low noise)", y=1.02)

fig3, axes3 = plot_llr_heatmap(qpsk, sigma_sq=0.5, grid_size=100)
fig3.suptitle("QPSK LLR Heatmap (σ² = 0.5 - high noise)", y=1.02)

plt.show()
