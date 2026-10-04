import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from scipy.linalg import expm
import warnings

warnings.filterwarnings("ignore")

# ── 1. FİZİKSEL PARAMETRELER ──────────────────────────────────────────────────
L0     = 1.0
L_turn = 6.0   
L_end  = 8.0  
N_bas  = 20
hbar   = 1.0
m      = 1.0

v_fast_max = 5.0
v_slow_max = 0.05
frames = 600
sub_steps = 10 

# ── 2. ZAMAN EVRİM OPERATÖRÜ ÇÖZÜCÜSÜ ─────────────────────────────────────────
def solve_time_evolution(v_max, num_frames=600, substeps=10):
    t1 = (L_turn - L0) / v_max  
    t2 = np.pi * (L_end - L_turn) / (2 * v_max) 
    
    T_total = 2 * (t1 + t2)    
    t_eval = np.linspace(0, T_total, num_frames)
    dt_frame = T_total / (num_frames - 1)
    dt = dt_frame / substeps 
    
    c_history = np.zeros((N_bas, num_frames), dtype=complex)
    L_history = np.zeros(num_frames)
    v_history = np.zeros(num_frames)
    
    c = np.zeros(N_bas, dtype=complex)
    c[0] = 1.0 + 0j
    c_history[:, 0] = c
    
    def get_Lv(t_val):
        if t_val <= t1:
            return L0 + v_max * t_val, v_max
        elif t_val <= t1 + 2*t2:
            tau = t_val - t1
            L = L_turn + (L_end - L_turn) * np.sin(np.pi * tau / (2 * t2))
            v = v_max * np.cos(np.pi * tau / (2 * t2))
            return L, v
        else:
            tau = t_val - (t1 + 2*t2)
            return L_turn - v_max * tau, -v_max

    L_history[0], v_history[0] = get_Lv(0)

    print(f"Çözülüyor: v_max = {v_max} | Delta t = {dt:.6f} ...")
    
    # Hamiltoniyen inşasını hızlandırmak için sabit kısımları önceden hesapla
    n_q_arr = np.arange(1, N_bas + 1)
    const_diag = (np.pi**2 * hbar) / (2.0 * m)
    
    M0 = np.zeros((N_bas, N_bas))
    for m_idx in range(N_bas):
        n_q = m_idx + 1
        for n_idx in range(N_bas):
            m_q = n_idx + 1
            if (n_q + m_q) % 2 != 0:
                M0[m_idx, n_idx] = -(2.0 * n_q * m_q) / (n_q**2 - m_q**2)
    
    t_current = 0.0
    for f in range(1, num_frames):
        for _ in range(substeps):
            L_mid, v_mid = get_Lv(t_current + dt / 2.0)
            
            # Vektörel Hamiltoniyen/A matrisi inşası
            diag_vals = -1j * (n_q_arr**2 * const_diag) / (L_mid**2)
            A = np.diag(diag_vals) - (v_mid / L_mid) * M0
            
            U = expm(A * dt)
            c = U @ c  
            
            t_current += dt
            
        c_history[:, f] = c
        L_history[f], v_history[f] = get_Lv(t_current)
        
    return t_eval, L_history, v_history, c_history

print("Zaman Evrim Operatörü Devrede (U = exp(-iHt))...")
t_f, L_f_arr, v_f_arr, c_fast = solve_time_evolution(v_fast_max, frames, sub_steps)
t_s, L_s_arr, v_s_arr, c_slow = solve_time_evolution(v_slow_max, frames, sub_steps)
print("Simülasyon Bitti.")

# ── 3. ENERJİ (EXPECTATION VALUE) HESAPLAMASI VE GRAFİĞİ ──────────────────────
print("Enerji Beklenen Değerleri Hesaplanıyor <E> ...")

n_array = np.arange(1, N_bas + 1)[:, np.newaxis] 

# Her frame için anlık enerji seviyeleri E_n(t) matrisi
E_n_slow = (n_array**2 * np.pi**2 * hbar**2) / (2.0 * m * L_s_arr**2)
E_n_fast = (n_array**2 * np.pi**2 * hbar**2) / (2.0 * m * L_f_arr**2)

# Beklenen Enerji: <E> = Toplam( |c_n|^2 * E_n )
E_exp_slow = np.sum(np.abs(c_slow)**2 * E_n_slow, axis=0)
E_exp_fast = np.sum(np.abs(c_fast)**2 * E_n_fast, axis=0)

# DÜZELTME: Zamanları kendi periyotlarına bölerek (0 ile 1 arasına) normalize ediyoruz
t_s_norm = t_s / t_s[-1]
t_f_norm = t_f / t_f[-1]

plt.figure(figsize=(10, 6), facecolor='#0d0f1a')
ax_e = plt.axes()
ax_e.set_facecolor('#0d0f1a')

# X eksenine artık normalize zamanı veriyoruz
plt.plot(t_s_norm, E_exp_slow, color='#4fc3f7', lw=3, label='Reversible (Slow) Energy')
plt.plot(t_f_norm, E_exp_fast, color='#ff7043', lw=3, label='Irreversible (Fast) Energy')

# Maksimum genleşme anı tam olarak döngünün yarısıdır (0.5)
plt.axvline(0.5, color='#eceff1', linestyle='--', alpha=0.5, label='Maximum Expansion ($L=8$)')

plt.title("Expectation Value of Energy Over Normalized Cycle", color='white', fontsize=14)
plt.xlabel("Normalized Time ($t / T_{cycle}$)", color='#90a4ae', fontsize=12)
plt.ylabel("Energy $\\langle E \\rangle$", color='#90a4ae', fontsize=12)
plt.grid(True, color='#1e2030', lw=1, alpha=0.8)
ax_e.tick_params(colors='#607d8b')
for sp in ax_e.spines.values(): sp.set_color('#1e2030')

legend = plt.legend(facecolor='#0d0f1a', edgecolor='#1e2030')
for text in legend.get_texts(): text.set_color('white')

plt.tight_layout()
# plt.show(block=False)


# ── 3.5. ANİMASYON İÇİN ÖN-HESAPLAMA (PRECOMPUTATION) ─────────────────────────
print("Dalga Fonksiyonu Yoğunlukları ve Seviyeleri Ön-Hesaplanıyor...")
x_res  = 1000
n_show = 10
n_vals = np.arange(1, N_bas + 1)[:, np.newaxis]

# Reversible (Slow) Ön-Hesaplama
x_s_history = np.zeros((frames, x_res))
density_s_history = np.zeros((frames, x_res))
E_s_history = np.zeros((frames, N_bas))
area_s_history = np.zeros(frames)

for f in range(frames):
    L = L_s_arr[f]
    c = c_slow[:, f]
    x = np.linspace(0, L, x_res)
    phi = np.sqrt(2.0 / L) * np.sin(n_vals * (np.pi / L) * x)
    psi = c @ phi
    density = np.abs(psi)**2
    
    x_s_history[f, :] = x
    density_s_history[f, :] = density
    area_s_history[f] = np.trapz(density, x)
    E_s_history[f, :] = (np.arange(1, N_bas + 1)**2 * np.pi**2 * hbar**2) / (2.0 * m * L**2)

# Irreversible (Fast) Ön-Hesaplama
x_f_history = np.zeros((frames, x_res))
density_f_history = np.zeros((frames, x_res))
E_f_history = np.zeros((frames, N_bas))
area_f_history = np.zeros(frames)

for f in range(frames):
    L = L_f_arr[f]
    c = c_fast[:, f]
    x = np.linspace(0, L, x_res)
    phi = np.sqrt(2.0 / L) * np.sin(n_vals * (np.pi / L) * x)
    psi = c @ phi
    density = np.abs(psi)**2
    
    x_f_history[f, :] = x
    density_f_history[f, :] = density
    area_f_history[f] = np.trapz(density, x)
    E_f_history[f, :] = (np.arange(1, N_bas + 1)**2 * np.pi**2 * hbar**2) / (2.0 * m * L**2)

# ── 4. GÖRSEL KURULUM (ANİMASYON ARAYÜZÜ) ─────────────────────────────────────
print("Arayüz Hazırlanıyor...")


fig, axes = plt.subplots(
    3, 2, figsize=(16, 13),
    gridspec_kw={'height_ratios': [2.5, 1.5, 1.5]}
)
fig.patch.set_facecolor('#0d0f1a')
fig.suptitle(
    "Time-Dependent Infinite Square Well",
    color='white', fontsize=14, fontweight='bold', y=0.99
)

ax_lev_s, ax_lev_f = axes[0]
ax_tot_s, ax_tot_f = axes[1]
ax_pop_s, ax_pop_f = axes[2]

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

setup_lev(ax_lev_s, f"Reversible Cycle — Energy Levels (v={v_slow_max})")
setup_lev(ax_lev_f, f"Irreversible Cycle — Energy Levels (v={v_fast_max})")

txt_v_s = ax_lev_s.text(0.02, 0.96, "", transform=ax_lev_s.transAxes, ha='left', va='top', color='#4fc3f7', fontsize=11, fontweight='bold')
txt_v_f = ax_lev_f.text(0.02, 0.96, "", transform=ax_lev_f.transAxes, ha='left', va='top', color='#ff7043', fontsize=11, fontweight='bold')

def setup_tot(ax, title):
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, L_end + 0.3)
    ax.set_ylim(-0.1, 3.5) 
    ax.set_xlabel("Position  x", color='#90a4ae', fontsize=9)
    ax.set_ylabel(r"Density $|\psi(x,t)|^2$", color='#90a4ae', fontsize=9) 
    ax.set_title(title, color='white', fontsize=11, pad=5)
    ax.tick_params(colors='#607d8b', labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    ax.axvline(0, color=WALL, lw=3)

setup_tot(ax_tot_s, fr"Probability Density  $|\psi(x,t)|^2$  —  Reversible (v={v_slow_max})")
setup_tot(ax_tot_f, fr"Probability Density  $|\psi(x,t)|^2$  —  Irreversible (v={v_fast_max})")

txt_area_s = ax_tot_s.text(0.98, 0.85, "", transform=ax_tot_s.transAxes, ha='right', color='#4fc3f7', fontsize=10, fontweight='bold')
txt_area_f = ax_tot_f.text(0.98, 0.85, "", transform=ax_tot_f.transAxes, ha='right', color='#ff7043', fontsize=10, fontweight='bold')

# Periyot göstergeleri (toplam t süresi)
ax_tot_s.text(0.98, 0.70, rf"$T_{{cycle}} = {t_s[-1]:.2f}$ ", transform=ax_tot_s.transAxes, ha='right', color='#b0bec5', fontsize=10, fontweight='bold')
ax_tot_f.text(0.98, 0.70, rf"$T_{{cycle}} = {t_f[-1]:.2f}$ ", transform=ax_tot_f.transAxes, ha='right', color='#b0bec5', fontsize=10, fontweight='bold')

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

setup_pop(ax_pop_s, f"Occupation Probabilities  —  Reversible (v={v_slow_max})")
setup_pop(ax_pop_f, f"Occupation Probabilities  —  Irreversible (v={v_fast_max})")

wall_lev_s = ax_lev_s.axvline(L0, color=WALL,  lw=3)
wall_lev_f = ax_lev_f.axvline(L0, color=WALL,  lw=3)
E_lines_s = [ax_lev_s.plot([], [], '--', color=LC[k], lw=1.2, alpha=1.0)[0] for k in range(n_show)]
E_lines_f = [ax_lev_f.plot([], [], '--', color=LC[k], lw=1.2, alpha=1.0)[0] for k in range(n_show)]
wall_tot_s = ax_tot_s.axvline(L0, color=WALL,  lw=3)
wall_tot_f = ax_tot_f.axvline(L0, color=WALL,  lw=3)
x_init = np.linspace(0, L0, x_res)
y_init = np.zeros(x_res)
fill_tot_s = ax_tot_s.fill_between(x_init, 0, y_init, color='#4fc3f7', alpha=0.65, linewidth=0)
fill_tot_f = ax_tot_f.fill_between(x_init, 0, y_init, color='#ff7043', alpha=0.65, linewidth=0)
bar_s = ax_pop_s.bar(range(1, n_show+1), np.zeros(n_show), color=LC, alpha=0.85, width=0.5)
bar_f = ax_pop_f.bar(range(1, n_show+1), np.zeros(n_show), color=LC, alpha=0.85, width=0.5)
txt_s = [ax_pop_s.text(i+1, 0, "", ha='center', va='bottom', color='white', fontsize=8) for i in range(n_show)]
txt_f = [ax_pop_f.text(i+1, 0, "", ha='center', va='bottom', color='white', fontsize=8) for i in range(n_show)]
txt_norm = ax_lev_f.text(0.98, 0.96, "", transform=ax_lev_f.transAxes, ha='right', va='top', color='#ef5350', fontsize=9)

is_playing  = [False]

# ── 5. ANİMASYON MOTORU ───────────────────────────────────────────────────────
def animate(frame):
    if frame == 1 and not is_playing[0]:
        ani.pause()
        frame = 0

    # Önceden hesaplanmış verileri çek
    L_s = L_s_arr[frame]
    v_s = v_s_arr[frame]
    E_s = E_s_history[frame]
    x_s = x_s_history[frame]
    density_s = density_s_history[frame]
    area_s = area_s_history[frame]

    L_f = L_f_arr[frame]
    v_f = v_f_arr[frame]
    E_f = E_f_history[frame]
    x_f = x_f_history[frame]
    density_f = density_f_history[frame]
    area_f = area_f_history[frame]

    # Arayüz güncellemeleri
    wall_lev_s.set_xdata([L_s, L_s])
    wall_lev_f.set_xdata([L_f, L_f])
    for k in range(n_show):
        E_lines_s[k].set_data([0, L_s], [E_s[k], E_s[k]])
        E_lines_f[k].set_data([0, L_f], [E_f[k], E_f[k]])

    wall_tot_s.set_xdata([L_s, L_s])
    wall_tot_f.set_xdata([L_f, L_f])

    # Dalga fonksiyonu alanlarını set_verts kullanarak güncelle (Çok daha hızlı ve akıcı)
    verts_s = np.zeros((x_res + 2, 2))
    verts_s[0] = (x_s[0], 0)
    verts_s[1:-1, 0] = x_s
    verts_s[1:-1, 1] = density_s
    verts_s[-1] = (x_s[-1], 0)
    fill_tot_s.set_verts([verts_s])

    verts_f = np.zeros((x_res + 2, 2))
    verts_f[0] = (x_f[0], 0)
    verts_f[1:-1, 0] = x_f
    verts_f[1:-1, 1] = density_f
    verts_f[-1] = (x_f[-1], 0)
    fill_tot_f.set_verts([verts_f])
    
    txt_area_s.set_text(rf"$\int |\psi|^2 dx = {area_s:.3f}$")
    txt_area_f.set_text(rf"$\int |\psi|^2 dx = {area_f:.3f}$")
    
    pop_s = np.abs(c_slow[:, frame])**2
    pop_f = np.abs(c_fast[:, frame])**2
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
            txt_v_s, txt_v_f,
            *bar_s, *bar_f, *txt_s, *txt_f, fill_tot_s, fill_tot_f)

plt.tight_layout(rect=[0, 0.06, 1, 0.99])

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=45, blit=True)

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