import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import warnings

warnings.filterwarnings("ignore")

# ── 1. COMMON PHYSICAL PARAMETERS ───────────────────────────────────────────
m = 1.0           # Mass
F0 = 5.0          # Maximum Magnetic Field Gradient Force
v_y = 2.0         # Constant velocity in y-direction (forward)
y_B_start = 2.0   # The y-coordinate where the magnetic field starts
y_B_end = 5.0     # The y-coordinate where the magnetic field ends
y_screen = 7.0    # The y-coordinate of the screen (measurement)

dt = 0.02
frames = 500      

# ── 2. CLASSICAL SYSTEM (NEWTON-MAXWELL) ────────────────────────────────────
N_particles = 3000
# Random 3D orientations: Continuous Distribution
mu_z_c = np.random.uniform(-1.0, 1.0, N_particles)
a_z_c = F0 * mu_z_c / m

# ── 3. SEMI-CLASSICAL SYSTEM (AD-HOC QUANTIZATION) ──────────────────────────
# Point particles that can only take +1 or -1 values (Old Quantum Approach)
mu_z_sc = np.random.choice([-1.0, 1.0], N_particles)
a_z_sc = F0 * mu_z_sc / m

def get_z_traj(t, a_z_arr):
    y = v_y * t
    t_start = y_B_start / v_y
    t_end = y_B_end / v_y
    
    if t <= t_start:
        return np.zeros(N_particles)
    elif t <= t_end:
        dt_in = t - t_start
        return 0.5 * a_z_arr * (dt_in**2)
    else:
        dt_in = t_end - t_start
        z_exit = 0.5 * a_z_arr * (dt_in**2)
        v_z_exit = a_z_arr * dt_in
        dt_out = t - t_end
        return z_exit + v_z_exit * dt_out

# ── 4. QUANTUM SYSTEM (SCHRÖDINGER-PAULI SSFM) ──────────────────────────────
N_z = 2048
z_max = 25.0
z = np.linspace(-z_max, z_max, N_z)
dz = z[1] - z[0]
k = 2 * np.pi * np.fft.fftfreq(N_z, d=dz)

# Initial Wave Function (Gaussian Packet)
sigma = 0.6
psi_0 = (1.0 / (np.pi * sigma**2)**0.25) * np.exp(-z**2 / (2 * sigma**2))
psi_up = psi_0 / np.sqrt(2)
psi_down = psi_0 / np.sqrt(2)

# SSFM Operators
T_op = np.exp(-1j * (k**2 / (2 * m)) * dt)
V_up = -F0 * z   
V_down = F0 * z  
V_op_up = np.exp(-1j * V_up * dt)
V_op_down = np.exp(-1j * V_down * dt)

# Absorbing Boundary Mask
absorbing_mask = np.ones_like(z)
sponge_width = 5.0
right_idx = z > (z_max - sponge_width)
absorbing_mask[right_idx] = np.exp(-0.2 * (z[right_idx] - (z_max - sponge_width))**2)
left_idx = z < (-z_max + sponge_width)
absorbing_mask[left_idx] = np.exp(-0.2 * (z[left_idx] - (-z_max + sponge_width))**2)

# ── 5. VISUAL INTERFACE (3x2 GRID) ──────────────────────────────────────────
BG = '#0d0f1a'
GRID = '#1e2030'
WALL = '#eceff1'

fig = plt.figure(figsize=(14, 12), facecolor=BG)
gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.1)

ax_c_sim = fig.add_subplot(gs[0, 0])
ax_c_dst = fig.add_subplot(gs[0, 1], sharey=ax_c_sim)
ax_sc_sim = fig.add_subplot(gs[1, 0])
ax_sc_dst = fig.add_subplot(gs[1, 1], sharey=ax_sc_sim)
ax_q_sim = fig.add_subplot(gs[2, 0])
ax_q_dst = fig.add_subplot(gs[2, 1], sharey=ax_q_sim)

def setup_ax(ax, is_sim=True):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#607d8b')
    for sp in ax.spines.values(): sp.set_color(GRID)
    if is_sim:
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
        ax.axvspan(y_B_start, y_B_end, color='#37474f', alpha=0.3)
        ax.axvline(y_screen, color=WALL, lw=2, linestyle='--')
        ax.set_xlim(0, 10)
        ax.set_ylim(-20, 20)
        ax.set_ylabel("Deflection (z)", color='#90a4ae', fontsize=10)
    else:
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6, axis='x')
        ax.set_xlim(0, 0.4) 
        plt.setp(ax.get_yticklabels(), visible=False)

for ax in [ax_c_sim, ax_sc_sim, ax_q_sim]: setup_ax(ax, True)
for ax in [ax_c_dst, ax_sc_dst, ax_q_dst]: setup_ax(ax, False)

# Titles
ax_c_sim.set_title("1. CLASSICAL (Newton-Maxwell): Continuous Distribution", color='#ffb74d', fontsize=12, loc='left')
ax_sc_sim.set_title("2. SEMI-CLASSICAL (Old Quantum): Point Particle + Quantized Spin", color='#ffee58', fontsize=12, loc='left')
ax_q_sim.set_title("3. MODERN QUANTUM (Schrödinger-Pauli): Wave Packet Collapse", color='#4fc3f7', fontsize=12, loc='left')
ax_q_sim.set_xlabel("Direction of Propagation (y)", color='#90a4ae', fontsize=11)
ax_q_dst.set_xlabel("Probability / Density", color='#90a4ae', fontsize=11)

# Objects: Classical
scatter_c = ax_c_sim.scatter([], [], s=2, color='#ffb74d', alpha=0.5)
bins_c = np.linspace(-15, 15, 60)
_, _, bar_c = ax_c_dst.hist([], bins=bins_c, orientation='horizontal', color='#ffb74d', density=True, alpha=0.7)

# Objects: Semi-Classical
scatter_sc = ax_sc_sim.scatter([], [], s=2, color='#ffee58', alpha=0.5)
_, _, bar_sc = ax_sc_dst.hist([], bins=bins_c, orientation='horizontal', color='#ffee58', density=True, alpha=0.7)

# Objects: Quantum
fill_q = None
line_q_up, = ax_q_dst.plot([], [], color='#4fc3f7', lw=2, label=r'Spin $\uparrow$')
line_q_dn, = ax_q_dst.plot([], [], color='#ef5350', lw=2, label=r'Spin $\downarrow$')
line_q_tot, = ax_q_dst.plot([], [], color='white', lw=1.5, linestyle='--', alpha=0.6, label='Total')

time_text = ax_c_sim.text(0.02, 0.85, "", transform=ax_c_sim.transAxes, color='white', fontsize=11, fontweight='bold')

# ── 6. ANIMATION ENGINE ──────────────────────────────────────────────────────
def animate(frame):
    global psi_up, psi_down, fill_q
    t = frame * dt
    y_t = v_y * t
    
    if y_t >= y_screen:
        time_text.set_text(f"MEASUREMENT COMPLETE | Position: {y_screen:.2f}")
        time_text.set_color('#00e676') 
        ani.event_source.stop()
        return scatter_c, *bar_c.patches, scatter_sc, *bar_sc.patches, line_q_up, line_q_dn, line_q_tot, time_text

    # 1. Classical (Newton-Maxwell) Update
    z_c = get_z_traj(t, a_z_c)
    scatter_c.set_offsets(np.c_[np.full(N_particles, y_t), z_c])
    counts_c, _ = np.histogram(z_c, bins=bins_c, density=True)
    for count, rect in zip(counts_c, bar_c): rect.set_width(count)
        
    # 2. Semi-Classical Update
    z_sc = get_z_traj(t, a_z_sc)
    scatter_sc.set_offsets(np.c_[np.full(N_particles, y_t), z_sc])
    counts_sc, _ = np.histogram(z_sc, bins=bins_c, density=True)
    for count, rect in zip(counts_sc, bar_sc): rect.set_width(count)
        
    # 3. Quantum Update (SSFM)
    psi_up = np.fft.ifft(T_op * np.fft.fft(psi_up))
    psi_down = np.fft.ifft(T_op * np.fft.fft(psi_down))
    
    if y_B_start <= y_t <= y_B_end:
        psi_up = V_op_up * psi_up
        psi_down = V_op_down * psi_down
        
    psi_up *= absorbing_mask
    psi_down *= absorbing_mask
        
    dens_up = np.abs(psi_up)**2
    dens_down = np.abs(psi_down)**2
    dens_tot = dens_up + dens_down
    
    if fill_q is not None: fill_q.remove()
    scale = 3.0 
    fill_q = ax_q_sim.fill_betweenx(z, y_t - dens_tot*scale, y_t, color='#4fc3f7', alpha=0.8)
    
    line_q_up.set_data(dens_up, z)
    line_q_dn.set_data(dens_down, z)
    line_q_tot.set_data(dens_tot, z)
    
    time_text.set_text(f"Time: {t:.2f} s | Position: {y_t:.2f}")
    
    return scatter_c, *bar_c.patches, scatter_sc, *bar_sc.patches, line_q_up, line_q_dn, line_q_tot, time_text

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=20, blit=False)

plt.show()