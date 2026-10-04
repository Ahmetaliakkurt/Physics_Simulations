import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==========================================
# SETTINGS
# ==========================================
n_steps = 100
n_walks = 50000
n_plot = 100  # Number of paths to visualize

# Fixed Probabilities (Must sum to 1.0)
p = 0.40      # Probability of taking a step to the RIGHT
q = 0.40      # Probability of taking a step to the LEFT
n_prob = 0.20 # Probability of doing NOTHING (staying put)

assert np.isclose(p + q + n_prob, 1.0), "Probabilities must sum up to 1!"

print(f"Fixed Probabilities -> p (Right): {p:.2f}, q (Left): {q:.2f}, n (Stay): {n_prob:.2f}")
print("Simulating paths...")

# 1. Generate random directions: 1 (Right), -1 (Left), 0 (Stay)
directions = np.random.choice([1, -1, 0], size=(n_walks, n_steps), p=[p, q, n_prob])

# 2. Generate random step sizes between 0.0 and 1.0 meters
step_sizes = np.random.uniform(0.0, 1.0, size=(n_walks, n_steps))

# 3. Calculate actual steps (direction * size)
steps = directions * step_sizes

# 4. Calculate cumulative positions for the paths (Trajectories)
start_positions = np.zeros((n_walks, 1))
trajectories = np.hstack((start_positions, np.cumsum(steps, axis=1)))

# 5. Extract final positions for the histogram
final_positions = trajectories[:, -1]

# ==========================================
# THEORETICAL GAUSSIAN CALCULATION
# ==========================================
step_mean = 0.5 * (p - q)
step_var = ((p + q) / 3.0) - (step_mean ** 2)

total_mean = n_steps * step_mean
total_std = np.sqrt(n_steps * step_var)

# ==========================================
# PLOTTING
# ==========================================
# sharey=True aligns the Y-axis (Position) across both subplots perfectly
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]}, sharey=True)

# --- LEFT PLOT: Individual Trajectories ---
time_array = np.arange(n_steps + 1)

for i in range(n_plot):
    ax1.plot(time_array, trajectories[i], alpha=0.4, linewidth=1.5)

ax1.set_title(f"Individual Trajectories ({n_plot} Random Samples)")
ax1.set_xlabel("Step Number (Time)")
ax1.set_ylabel("Position (meters)")
ax1.axhline(0, color='black', linewidth=1.5, linestyle='--', label='Starting Point')
ax1.grid(True, alpha=0.3)
ax1.legend()

# --- RIGHT PLOT: Final Distribution Histogram (Axes Swapped) ---
# orientation='horizontal' swaps the histogram
ax2.hist(final_positions, bins=100, density=True, color='teal', alpha=0.7, 
         orientation='horizontal', label=f'Sim ({n_walks})')

x_values = np.linspace(total_mean - 4*total_std, total_mean + 4*total_std, 1000)
theoretical_pdf = stats.norm.pdf(x_values, total_mean, total_std)

# Swap theoretical_pdf and x_values for the plot
ax2.plot(theoretical_pdf, x_values, 'r-', lw=2.5, label='Theory (Gaussian)')

# Adjusted limits for swapped axes
ax2.set_ylim(total_mean - 4*total_std, total_mean + 4*total_std)
ax2.set_xlim(left=0) # Density always starts from 0

ax2.set_title(f"Final Distribution\np={p:.2f}, q={q:.2f}, n={n_prob:.2f}")
ax2.set_xlabel("Density")
# Y-axis label is shared with the left plot
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()