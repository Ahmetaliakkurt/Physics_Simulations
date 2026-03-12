import numpy as np
import matplotlib.pyplot as plt
import math

n_steps = 100
n_walks = 50000
p = 0.5

def binom_dist(n, k, p):
    combination = math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    return combination * (p**k) * ((1 - p)**(n - k))

steps = np.random.choice([-1, 1], size=(n_walks, n_steps), p=[1-p, p])
final_positions = np.sum(steps, axis=1)

plt.figure(figsize=(10, 6))
bins = np.arange(-n_steps - 1, n_steps + 2, 2)
plt.hist(final_positions, bins=bins, density=True, color='blue', alpha=0.7, label=f'Simulation ({n_walks} Sample)')

x_values = np.arange(-n_steps, n_steps + 1, 2)
theoretical_pmf = [binom_dist(n_steps, (x + n_steps) // 2, p) for x in x_values]

plt.plot(x_values, np.array(theoretical_pmf) / 2, 'r-', lw=2.5, label='Ideal Binomial Distribution')
plt.xlim(-1000,1000)
plt.title(f"Random Walk: Number of steps={n_steps}, Probability of right step={p}")
plt.xlabel("Final Position")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()