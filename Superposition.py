import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

rng = np.random.RandomState(123)
x = np.linspace(-15, 15, 1500)
k0 = 5.0
sigma_k = 0.5
sigma_A = 0.2
A_total = 1.0
wave_list = []

def ensure_waves(n):
    while len(wave_list) < n:
        k = rng.normal(k0, sigma_k)
        A = abs(rng.normal(1.0, sigma_A))
        wave_list.append((k, A))

def compute_total(n):
    n = int(n)
    ensure_waves(n)
    used = wave_list[:n]
    if not used: return np.zeros_like(x)
    
    ks = np.array([k for k, _ in used])
    As = np.array([A for _, A in used])
    
    sum_As = As.sum()
    if sum_As == 0: sum_As = 1.0
        
    As = As / sum_As * A_total
    
    psi = np.zeros_like(x, dtype=complex)
    for k, A in zip(ks, As):
        psi += A * np.exp(1j * k * x)
    return np.real(psi)

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

initial_n = 10
y_vals = compute_total(initial_n)
line, = ax.plot(x, y_vals, lw=2, color='blue')

ax.set_title("Superposition of the Plane Waves")
ax.set_ylim(-1.5, 1.5)
ax.grid(True, alpha=0.3)

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(
    ax=ax_slider,
    label='N waves',
    valmin=1,
    valmax=3000,
    valinit=initial_n,
    valstep=1
)

def update(val):
    n = slider.val
    y = compute_total(n)
    line.set_ydata(y)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()