import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.integrate import solve_ivp
import warnings

warnings.filterwarnings("ignore")

# ── 1. PARAMETRELERİ ──────────────────────────────────────────────────────────
L0     = 1.0
L_end  = 10.0
v_fast = 5.0
v_slow = 0.005
N_bas  = 12
hbar   = 1.0
m      = 1.0

T_fast = (L_end - L0) / v_fast
T_slow = (L_end - L0) / v_slow
frames = 500

t_fast = np.linspace(0, T_fast, frames)
t_slow = np.linspace(0, T_slow, frames)

# ── 2. BAĞLANMA MATRİSİ ───────────────────────────────────────────────────────
def M_nm(n, m, L):
    if n == m:            return 0.0
    if (n + m) % 2 == 0: return 0.0
    return -(2.0 * n * m) / (L * (n**2 - m**2))

def build_ODE(v_exp):
    def rhs(t, c):
        L  = L0 + v_exp * t
        dc = np.zeros(N_bas, dtype=complex)
        E  = np.array([(n+1)**2 * np.pi**2 * hbar**2 / (2*m*L**2)
                       for n in range(N_bas)])
        for i in range(N_bas):
            dc[i] = -1j * (E[i] / hbar) * c[i]
            for j in range(N_bas):
                if j != i:
                    Mij = M_nm(i+1, j+1, L)
                    if Mij != 0.0:
                        dc[i] -= v_exp * Mij * c[j]
        return dc
    return rhs

# ── 3. ODE ÇÖZÜMÜ ─────────────────────────────────────────────────────────────
c0    = np.zeros(N_bas, dtype=complex)
c0[0] = 1.0 + 0j

print("Simulation Begins...")
sol_fast = solve_ivp(build_ODE(v_fast), [0, T_fast], c0, t_eval=t_fast,
                     method='DOP853', atol=1e-9, rtol=1e-9)
sol_slow = solve_ivp(build_ODE(v_slow), [0, T_slow], c0, t_eval=t_slow,
                     method='DOP853', atol=1e-9, rtol=1e-9)

c_fast = sol_fast.y
c_slow = sol_slow.y

norm_f_end = np.sum(np.abs(c_fast[:, -1])**2)
norm_s_end = np.sum(np.abs(c_slow[:, -1])**2)
print(f"Son frame normları — Hızlı: {norm_f_end:.6f}, Yavaş: {norm_s_end:.6f}")

# ── 4. GÖRSEL KURULUM ─────────────────────────────────────────────────────────
x_res  = 600
n_show = 10

fig, axes = plt.subplots(
    3, 2, figsize=(16, 13),
    gridspec_kw={'height_ratios': [2.5, 1.5, 1.5]}
)
fig.patch.set_facecolor('#0d0f1a')
fig.suptitle(
    "Infinite Square Well Adiabatic Expansion",
    color='white', fontsize=14, fontweight='bold', y=0.99
)

ax_lev_s, ax_lev_f = axes[0]   # satır 0: enerji seviyeleri
ax_tot_s, ax_tot_f = axes[1]   # satır 1: toplam |ψ|²
ax_pop_s, ax_pop_f = axes[2]   # satır 2: |c_n|²

BG   = '#0d0f1a'
GRID = '#1e2030'
WALL = '#eceff1'

def level_color(k, n_total=n_show):
    t = k / max(n_total - 1, 1)
    r = int(0x4f + t * (0xff - 0x4f))
    g = int(0xc3 + t * (0x70 - 0xc3))
    b = int(0xf7 + t * (0x43 - 0xf7))
    return f'#{r:02x}{g:02x}{b:02x}'

LC = [level_color(k) for k in range(n_show)]

# ── Satır 0: SADECE enerji seviyeleri ─────────────────────────────────────────
def setup_lev(ax, title):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, L_end + 0.3)
    ax.set_ylim(-0.3, 22)
    ax.set_xlabel("Position  x", color='#90a4ae', fontsize=9)
    ax.set_ylabel("Energy  E",   color='#90a4ae', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, pad=5)
    ax.tick_params(colors='#607d8b', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.axvline(0, color=WALL, lw=3)

setup_lev(ax_lev_s, f"Quasistatic (v = {v_slow}) — Energy Levels")
setup_lev(ax_lev_f, f"Non-Quasistatic (v = {v_fast}) — Energy Levels")

# ── Satır 1: SADECE toplam olasılık yoğunluğu ─────────────────────────────────
def setup_tot(ax, title, col):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, L_end + 0.3)
    ax.set_ylim(-0.1, 3.5)
    ax.set_xlabel("Position  x",       color='#90a4ae', fontsize=9)
    ax.set_ylabel(r"Density $|\psi(x,t)|^2$",  color='#90a4ae', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, pad=5)
    ax.tick_params(colors='#607d8b', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.axvline(0, color=WALL, lw=3)

setup_tot(ax_tot_s, r"Probability Density  $|\psi(x,t)|^2$  —  Quasistatic",   '#4fc3f7')
setup_tot(ax_tot_f, r"Probability Density  $|\psi(x,t)|^2$  —  Non-Quasistatic", '#ff7043')

txt_area_s = ax_tot_s.text(0.98, 0.85, "", transform=ax_tot_s.transAxes, ha='right', color='#4fc3f7', fontsize=10, fontweight='bold')
txt_area_f = ax_tot_f.text(0.98, 0.85, "", transform=ax_tot_f.transAxes, ha='right', color='#ff7043', fontsize=10, fontweight='bold')

# ── Satır 2: popülasyon barları ───────────────────────────────────────────────
def setup_pop(ax, title):
    ax.set_facecolor(BG)
    ax.set_xlim(0.5, n_show + 0.5)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(range(1, n_show + 1))
    ax.tick_params(colors='#607d8b', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.4, axis='y')
    ax.set_xlabel("Energy level  n", color='#90a4ae', fontsize=9)
    ax.set_ylabel("$|c_n|^2$",       color='#90a4ae', fontsize=9)
    ax.set_title(title, color='white', fontsize=11, pad=5)

setup_pop(ax_pop_s, "Occupation Probabilities  —  Quasistatic")
setup_pop(ax_pop_f, "Occupation Probabilities  —  Non-Quasistatic")

# ── Dinamik objeler ────────────────────────────────────────────────────────────

# Satır 0 objeleri
wall_lev_s = ax_lev_s.axvline(L0, color=WALL,  lw=3)
wall_lev_f = ax_lev_f.axvline(L0, color=WALL,  lw=3)

E_lines_s = [ax_lev_s.plot([], [], '--', color=LC[k], lw=1.2, alpha=1.0)[0] for k in range(n_show)]
E_lines_f = [ax_lev_f.plot([], [], '--', color=LC[k], lw=1.2, alpha=1.0)[0] for k in range(n_show)]

# Satır 1 objeleri
wall_tot_s = ax_tot_s.axvline(L0, color=WALL,  lw=3)
wall_tot_f = ax_tot_f.axvline(L0, color=WALL,  lw=3)

fill_tot_s = None
fill_tot_f = None

# Satır 2 objeleri
bar_s = ax_pop_s.bar(range(1, n_show+1), np.zeros(n_show), color=LC, alpha=0.85, width=0.5)
bar_f = ax_pop_f.bar(range(1, n_show+1), np.zeros(n_show), color=LC, alpha=0.85, width=0.5)

txt_s = [ax_pop_s.text(i+1, 0, "", ha='center', va='bottom', color='white', fontsize=8) for i in range(n_show)]
txt_f = [ax_pop_f.text(i+1, 0, "", ha='center', va='bottom', color='white', fontsize=8) for i in range(n_show)]

txt_norm = ax_lev_f.text(0.98, 0.96, "", transform=ax_lev_f.transAxes, ha='right', va='top', color='#ef5350', fontsize=9)

is_playing  = [False]

# ── 5. ANİMASYON ──────────────────────────────────────────────────────────────
def animate(frame):
    global fill_tot_s, fill_tot_f

    if frame == 1 and not is_playing[0]:
        ani.pause()
        frame = 0

    # ── Reversible ──
    L_s   = L0 + v_slow * t_slow[frame]
    c_s   = c_slow[:, frame]
    x_s   = np.linspace(0, L_s, x_res)
    phi_s = np.array([np.sqrt(2/L_s) * np.sin((n+1)*np.pi*x_s/L_s) for n in range(N_bas)])
    E_s   = np.array([(n+1)**2 * np.pi**2 / (2*L_s**2) for n in range(N_bas)])
    pop_s = np.abs(c_s)**2
    
    psi_s     = np.einsum('n,nx->x', c_s, phi_s)
    density_s = np.abs(psi_s)**2
    area_s    = np.trapz(density_s, x_s)

    # ── Irreversible ──
    L_f   = L0 + v_fast * t_fast[frame]
    c_f   = c_fast[:, frame]
    x_f   = np.linspace(0, L_f, x_res)
    phi_f = np.array([np.sqrt(2/L_f) * np.sin((n+1)*np.pi*x_f/L_f) for n in range(N_bas)])
    E_f   = np.array([(n+1)**2 * np.pi**2 / (2*L_f**2) for n in range(N_bas)])
    pop_f = np.abs(c_f)**2
    
    psi_f     = np.einsum('n,nx->x', c_f, phi_f)
    density_f = np.abs(psi_f)**2
    area_f    = np.trapz(density_f, x_f)

    # ── Satır 0 Güncellemeleri (Sadece Çizgiler) ──
    wall_lev_s.set_xdata([L_s, L_s])
    wall_lev_f.set_xdata([L_f, L_f])

    for k in range(n_show):
        E_lines_s[k].set_data([0, L_s], [E_s[k], E_s[k]])
        E_lines_f[k].set_data([0, L_f], [E_f[k], E_f[k]])

    # ── Satır 1 Güncellemeleri (Toplam Yoğunluk) ──
    wall_tot_s.set_xdata([L_s, L_s])
    wall_tot_f.set_xdata([L_f, L_f])

    if fill_tot_s is not None: fill_tot_s.remove(); fill_tot_s = None
    if fill_tot_f is not None: fill_tot_f.remove(); fill_tot_f = None

    fill_tot_s = ax_tot_s.fill_between(x_s, 0, density_s, color='#4fc3f7', alpha=0.65, linewidth=0)
    fill_tot_f = ax_tot_f.fill_between(x_f, 0, density_f, color='#ff7043', alpha=0.65, linewidth=0)
    
    txt_area_s.set_text(rf"$\int |\psi|^2 dx = {area_s:.3f}$")
    txt_area_f.set_text(rf"$\int |\psi|^2 dx = {area_f:.3f}$")

    # ── Satır 2 Güncellemeleri (Barlar) ──
    for i in range(n_show):
        p_s = pop_s[i]
        bar_s[i].set_height(p_s)
        txt_s[i].set_text(f"{p_s:.2f}" if p_s > 0.01 else "")
        txt_s[i].set_y(p_s + 0.02)

        p_f = pop_f[i]
        bar_f[i].set_height(p_f)
        txt_f[i].set_text(f"{p_f:.2f}" if p_f > 0.01 else "")
        txt_f[i].set_y(p_f + 0.02)

    nf = float(np.sum(pop_f))
    txt_norm.set_text("" if abs(nf - 1.0) < 0.01 else f"Norm: {nf:.4f} !")

    return (wall_lev_s, wall_lev_f, wall_tot_s, wall_tot_f,
            *E_lines_s, *E_lines_f, txt_norm, txt_area_s, txt_area_f,
            *bar_s, *bar_f, *txt_s, *txt_f)

plt.tight_layout(rect=[0, 0.06, 1, 0.99])

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=40, blit=False)

# ── 6. KONTROL BUTONLARI ──────────────────────────────────────────────────────
ax_play   = plt.axes([0.42, 0.01, 0.07, 0.03])
ax_replay = plt.axes([0.51, 0.01, 0.07, 0.03])

btn_play   = Button(ax_play,   'Play',   color=GRID, hovercolor='#37474f')
btn_replay = Button(ax_replay, 'Replay', color=GRID, hovercolor='#37474f')
btn_play.label.set_color('white')
btn_replay.label.set_color('white')

def toggle_play(event):
    if is_playing[0]:
        ani.pause(); btn_play.label.set_text('Play')
    else:
        ani.resume(); btn_play.label.set_text('Pause')
    is_playing[0] = not is_playing[0]
    fig.canvas.draw_idle()

def restart_anim(event):
    ani.frame_seq = ani.new_frame_seq()
    if not is_playing[0]: animate(0)
    fig.canvas.draw_idle()

btn_play.on_clicked(toggle_play)
btn_replay.on_clicked(restart_anim)

plt.show()