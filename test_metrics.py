import numpy as np
import matplotlib.pyplot as plt
from apdist.distances import AmplitudePhaseDistance

q = np.logspace(-3, -1, 200)

# Ground truth curve with a peak at q=0.02
S_gt = 100 * np.exp(-((q - 0.001) / 0.002)**2) + 10 * np.exp(-((q - 0.02) / 0.005)**2) + 1

# Sim 1 (analogous to iter 3): peak correctly aligned, but low-q is slightly off (lower amplitude)
S_sim1 = 80 * np.exp(-((q - 0.001) / 0.002)**2) + 10 * np.exp(-((q - 0.02) / 0.005)**2) + 1

# Sim 2 (analogous to iter 5): peak shifted incorrectly, low-q matches perfectly
S_sim2 = 100 * np.exp(-((q - 0.001) / 0.002)**2) + 10 * np.exp(-((q - 0.03) / 0.005)**2) + 1

# Compute standard MSE on log10 intensities
log_gt = np.log10(S_gt)
log_sim1 = np.log10(S_sim1)
log_sim2 = np.log10(S_sim2)

mse1 = np.mean((log_gt - log_sim1)**2)
mse2 = np.mean((log_gt - log_sim2)**2)

# Compute AP distance as implemented in metrics.py
q_AP = np.linspace(q[0], q[-1], len(q))
da1, dp1 = AmplitudePhaseDistance(q_AP, log_gt, log_sim1)
ap1 = da1 + dp1

da2, dp2 = AmplitudePhaseDistance(q_AP, log_gt, log_sim2)
ap2 = da2 + dp2

print("Standard log10 MSE:")
print(f"Sim 1 (correct peak, wrong low-q): {mse1:.4f}")
print(f"Sim 2 (shifted peak, right low-q): {mse2:.4f}")

print("\nAmplitude-Phase Distance (d_a + d_p):")
print(f"Sim 1 (correct peak, wrong low-q): {ap1:.4f}")
print(f"Sim 2 (shifted peak, right low-q): {ap2:.4f}")

