import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

# --- Fiziksel Parametreler ---
G = 9.81    
L1, L2 = 1.0, 1.0  
M1, M2 = 1.0, 1.0  

def double_pendulum_derivs(t, state):
    state = state.reshape(-1, 4)
    t1, w1, t2, w2 = state[:,0], state[:,1], state[:,2], state[:,3]
    
    delta = t1 - t2
    den = (2*M1 + M2 - M2 * np.cos(2*t1 - 2*t2))
    
    dw1 = (-G * (2*M1 + M2) * np.sin(t1) 
           - M2 * G * np.sin(t1 - 2*t2) 
           - 2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))) / (L1 * den)
    
    dw2 = (2 * np.sin(delta) * (w1**2 * L1 * (M1 + M2) 
           + G * (M1 + M2) * np.cos(t1) 
           + w2**2 * L2 * M2 * np.cos(delta))) / (L2 * den)
    
    return np.stack([w1, dw1, w2, dw2], axis=1).flatten()

# --- Görselleştirme ---
fig, ax = plt.subplots(figsize=(9, 9))
plt.subplots_adjust(top=0.9, bottom=0.2)
ax.set_aspect('equal')
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
ax.grid(True, alpha=0.15, linestyle='--')

ani = None
lines = []
sol_y = None
t_vals = None
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontweight='bold', fontsize=11)

def start_sim(event):
    global ani, lines, sol_y, t_vals, time_text
    
    if ani:
        ani.event_source.stop()
    
    ax.clear()
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.grid(True, alpha=0.15)
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontweight='bold', fontsize=11)

    try:
        n = int(text_box.text)
    except ValueError:
        return

    # Başlangıç Koşulu: +y'nin (180 derece) 1 derece sağı (179 derece)
    # Her sarkaç arası delta_theta = 1 derece
    t_start_deg = 179 
    y0 = []
    for i in range(n):
        theta_i = np.radians(t_start_deg - i) # Saat yönünde 1'er derece fark
        y0.extend([theta_i, 0, theta_i, 0])   # Kollar dümdüz (theta1 = theta2)
    
    t_vals = np.linspace(0, 25, 1250)
    sol = solve_ivp(double_pendulum_derivs, (0, 25), y0, t_eval=t_vals, method='RK45')
    sol_y = sol.y
    
    colors = plt.cm.inferno(np.linspace(0.3, 0.9, n))
    lines = [ax.plot([], [], 'o-', lw=2, color=colors[i], alpha=0.6, markersize=4)[0] for i in range(n)]

    def update(frame):
        for i in range(n):
            idx = i * 4
            th1, th2 = sol_y[idx][frame], sol_y[idx+2][frame]
            x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
            x2, y2 = x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)
            lines[i].set_data([0, x1, x2], [0, y1, y2])
        
        time_text.set_text(f'Zaman: {t_vals[frame]:.1f}s')
        return lines + [time_text]

    ani = FuncAnimation(fig, update, frames=len(t_vals), interval=20, blit=True, cache_frame_data=False)
    plt.draw()

# --- Arayüz ---
ax_text = plt.axes([0.2, 0.05, 0.1, 0.05])
text_box = TextBox(ax_text, 'Sarkaç Sayısı: ', initial="1")

ax_start = plt.axes([0.45, 0.05, 0.25, 0.05])
btn_start = Button(ax_start, 'Başlat', color='#c0392b', hovercolor='#e74c3c')
btn_start.on_clicked(start_sim)

fig.suptitle("Katotik Multi Sarkaç Sistemi", fontsize=14, fontweight='bold')
plt.show()