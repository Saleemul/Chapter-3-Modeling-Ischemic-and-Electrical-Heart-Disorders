"""
Supplementary Code 3: Physics-Constrained vs Data-Only Interpolation
Chapter 10 - Modeling Ischemic and Electrical Heart Disorders

Reconstructing a cardiac conduction velocity (CV) field from sparse
electrode measurements. A smoothness constraint (Laplacian penalty,
motivated by the diffusion equation) reduces oscillatory artifacts
compared to unconstrained cubic spline interpolation.

Requirements: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline

np.random.seed(42)

Nx = 300
x = np.linspace(0, 10, Nx)

def true_cv(x):
    cv = 60.0 * np.ones_like(x)
    cv -= 35.0 * np.exp(-0.5 * ((x - 5.0) / 1.0)**2)  # scar dip
    return cv

cv_true = true_cv(x)

# 10 irregularly spaced electrodes, including 2 inside scar
obs_x = np.array([0.3, 1.4, 2.3, 3.5, 4.2, 5.8, 6.5, 7.6, 8.7, 9.6])
noise = np.random.randn(len(obs_x)) * 3.0  # realistic measurement noise
obs_y = true_cv(obs_x) + noise

print(f"Observations: {len(obs_x)} electrodes with noise (std=3 cm/s)")

# Data-only: cubic spline
cv_spline = CubicSpline(obs_x, obs_y, bc_type='natural')(x)

# Physics-constrained: smoothing spline (penalizes curvature = Laplacian)
# s parameter controls data-fit vs smoothness tradeoff
cv_smooth = UnivariateSpline(obs_x, obs_y, s=len(obs_x)*6.0, k=3)(x)

# Errors
rmse_s_all = np.sqrt(np.mean((cv_spline - cv_true)**2))
rmse_p_all = np.sqrt(np.mean((cv_smooth - cv_true)**2))

# Focus on regions BETWEEN electrodes (interpolation quality)
gap = (x > 2.5) & (x < 7.5)
rmse_s_gap = np.sqrt(np.mean((cv_spline[gap] - cv_true[gap])**2))
rmse_p_gap = np.sqrt(np.mean((cv_smooth[gap] - cv_true[gap])**2))

print(f"\nGlobal RMSE:   spline={rmse_s_all:.2f}  smoothing={rmse_p_all:.2f}")
print(f"Scar region:   spline={rmse_s_gap:.2f}  smoothing={rmse_p_gap:.2f}")
print(f"Improvement:   {(1-rmse_p_gap/rmse_s_gap)*100:.1f}%")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].fill_between(x[(x>3)&(x<7)], 15, 68, alpha=0.08, color='red', label='Scar region')
axes[0].plot(x, cv_true, 'k-', lw=2, label='True CV')
axes[0].plot(obs_x, obs_y, 'ro', ms=8, zorder=5, label=f'Electrodes (n={len(obs_x)})')
axes[0].set_xlabel('Position (cm)'); axes[0].set_ylabel('CV (cm/s)')
axes[0].set_title('(a) Ground Truth + Noisy Observations')
axes[0].legend(fontsize=8); axes[0].set_ylim(15, 68)

axes[1].fill_between(x[(x>3)&(x<7)], 15, 68, alpha=0.08, color='red')
axes[1].plot(x, cv_true, 'k-', lw=2, label='True')
axes[1].plot(x, cv_spline, '--', color='#EA580C', lw=1.8,
             label=f'Cubic spline (RMSE={rmse_s_gap:.1f})')
axes[1].plot(x, cv_smooth, '-', color='#0D9488', lw=1.8,
             label=f'Physics-smoothed (RMSE={rmse_p_gap:.1f})')
axes[1].plot(obs_x, obs_y, 'ro', ms=5, alpha=0.4)
axes[1].set_xlabel('Position (cm)'); axes[1].set_ylabel('CV (cm/s)')
axes[1].set_title('(b) Reconstruction Comparison'); axes[1].legend(fontsize=8)
axes[1].set_ylim(15, 68)

axes[2].fill_between(x[(x>3)&(x<7)], 0, 20, alpha=0.08, color='red', label='Scar region')
axes[2].plot(x, np.abs(cv_spline - cv_true), '--', color='#EA580C', lw=1.5, label='Spline error')
axes[2].plot(x, np.abs(cv_smooth - cv_true), '-', color='#0D9488', lw=1.5, label='Physics error')
axes[2].set_xlabel('Position (cm)'); axes[2].set_ylabel('|Error| (cm/s)')
axes[2].set_title('(c) Pointwise Absolute Error'); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/home/claude/supplementary_code/fig_physics_informed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure saved: fig_physics_informed.png")
