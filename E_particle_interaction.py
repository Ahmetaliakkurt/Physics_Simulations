import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

c = 1.0
q = -1.0
m = 1.0
E0 = 9.0
omega = 10.0
k = omega / c

dt = 0.05
t_total = 60.0
steps = int(t_total / dt)

pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])

history = []

def get_fields(x, t):
    pulse_center = c * t - 10.0
    envelope = np.exp(-0.5 * ((x - pulse_center) / 3.0)**2)
    phase = k * x - omega * t
    
    Ey = E0 * np.cos(phase) * envelope
    Bz = (E0 / c) * np.cos(phase) * envelope
    
    return np.array([0, Ey, 0]), np.array([0, 0, Bz])

times = np.linspace(0, t_total, steps)
current_pos = pos.copy()
current_vel = vel.copy()

print("Veriler hesaplanıyor, lütfen bekleyin...")

for t in times:
    E_field, B_field = get_fields(current_pos[0], t)
    
    v_cross_B = np.cross(current_vel, B_field)
    Force = q * (E_field + v_cross_B)
    
    acc = Force / m
    current_vel += acc * dt
    current_pos += current_vel * dt
    
    history.append(current_pos.copy())

history = np.array(history)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(hspace=0.3)

ax1.set_xlim(-10, 20)
ax1.set_ylim(-10, 10)
ax1.set_title("Gelen Elektromanyetik Dalga (E-Alanı)")
ax1.grid(True, alpha=0.3)
wave_line, = ax1.plot([], [], 'g-', lw=2)
electron_marker, = ax1.plot([], [], 'ro')

ax2.set_xlim(-20, 20)
ax2.set_ylim(-1, 1)
ax2.set_title("Elektronun Hareketi")
ax2.set_xlabel("X (Sürüklenme)")
ax2.set_ylabel("Y (Salınım)")
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', lw=0.5)

traj_line, = ax2.plot([], [], 'b-', lw=1)
electron_dot, = ax2.plot([], [], 'bo', ms=8)
info_text = ax2.text(0.02, 0.9, '', transform=ax2.transAxes)

def init():
    wave_line.set_data([], [])
    electron_marker.set_data([], [])
    traj_line.set_data([], [])
    electron_dot.set_data([], [])
    return wave_line, electron_marker, traj_line, electron_dot

def update(frame):
    t = times[frame]
    elec_pos = history[frame]
    
    x_space = np.linspace(-10, 25, 400)
    pulse_center = c * t - 10.0
    envelope = np.exp(-0.5 * ((x_space - pulse_center) / 3.0)**2)
    phase = k * x_space - omega * t
    E_vals = E0 * np.cos(phase) * envelope
    
    wave_line.set_data(x_space, E_vals)
    electron_marker.set_data([elec_pos[0]], [0])
    
    start_idx = max(0, frame - 1000)
    traj_line.set_data(history[start_idx:frame, 0], history[start_idx:frame, 1])
    electron_dot.set_data([elec_pos[0]], [elec_pos[1]])
    
    info_text.set_text(f"t: {t:.2f}")
    
    return wave_line, electron_marker, traj_line, electron_dot

print("Animasyon penceresi açılıyor...")

ani = animation.FuncAnimation(fig, update, frames=range(0, len(times), 2), 
                              init_func=init, interval=20, blit=True)

plt.show()