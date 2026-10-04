import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, cdist
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. PARAMETRELER
# ==========================================
G = 9.81
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0

N_pendulums = 200
start_angle_deg = 179.0
noise_std_deg = 1e-6

T_sim = 20.0
fps = 30
frames = int(T_sim * fps)
t_eval = np.linspace(0, T_sim, frames)

N_show = 25  # Görsel netlik için sadece 25 sarkaç çizilecek

# ==========================================
# 2. ODE ÇÖZÜCÜ
# ==========================================
print("⏳ Kaotik Diferansiyel Denklemler Çözülüyor...")

def double_pendulum_derivs(t, state):
    state = state.reshape(-1, 4)
    t1, w1, t2, w2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    delta = t1 - t2
    den = (2 * M1 + M2 - M2 * np.cos(2 * t1 - 2 * t2))

    dw1 = (-G * (2 * M1 + M2) * np.sin(t1) - M2 * G * np.sin(t1 - 2 * t2)
           - 2 * np.sin(delta) * M2 * (w2**2 * L2 + w1**2 * L1 * np.cos(delta))) / (L1 * den)
    dw2 = (2 * np.sin(delta) * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(t1)
           + w2**2 * L2 * M2 * np.cos(delta))) / (L2 * den)
    return np.stack([w1, dw1, w2, dw2], axis=1).flatten()

np.random.seed(42)
theta1_initials = np.random.normal(
    loc=np.radians(start_angle_deg),
    scale=np.radians(noise_std_deg),
    size=N_pendulums
)
y0 = []
for th1 in theta1_initials:
    y0.extend([th1, 0.0, th1, 0.0])

sol = solve_ivp(double_pendulum_derivs, (0, T_sim), y0, t_eval=t_eval, method='RK45')
sol_y = sol.y

# ==========================================
# 3. VERİ ÇIKARIMI
# ==========================================
print("📊 Veriler İşleniyor...")

theta1_all = sol_y[0::4, :]  # (N, frames)
omega1_all = sol_y[1::4, :]
theta2_all = sol_y[2::4, :]
omega2_all = sol_y[3::4, :]

x1_all = L1 * np.sin(theta1_all)
y1_all = -L1 * np.cos(theta1_all)
x2_all = x1_all + L2 * np.sin(theta2_all)
y2_all = y1_all - L2 * np.cos(theta2_all)

# ==========================================
# 4. İSTATİSTİKSEL HESAPLAMALAR
# ==========================================

# --- (a) Dairesel Standart Sapma ---
print("   → Dairesel Std Sapma Hesaplanıyor...")
mean_sin = np.mean(np.sin(theta2_all), axis=0)
mean_cos = np.mean(np.cos(theta2_all), axis=0)
R_bar = np.sqrt(mean_sin**2 + mean_cos**2)  # Dairesel ortalama vektör büyüklüğü
circular_std_deg = np.degrees(np.sqrt(-2 * np.log(np.clip(R_bar, 1e-15, 1.0))))

# --- (b) KDE Tabanlı Diferansiyel Entropi (doğrudan θ₂ üzerinden) ---
print("   → KDE Entropisi Hesaplanıyor...")
# θ₂ açılarını [-π, π] aralığına sar
theta2_wrapped = np.arctan2(np.sin(theta2_all), np.cos(theta2_all))  # (N, frames)

kde_grid = np.linspace(-np.pi, np.pi, 300)
kde_curves = np.zeros((frames, len(kde_grid)))
kde_entropy = np.zeros(frames)

# Uniform dağılım entropisi: H_max = log2(2π) ≈ 2.65 bit
H_uniform = np.log2(2 * np.pi)

for i in range(frames):
    data = theta2_wrapped[:, i]
    spread = np.std(data)
    if spread < 1e-10:
        # Çok dar dağılım — entropi minimum
        kde_entropy[i] = 0.0
        continue
    kde = gaussian_kde(data, bw_method='silverman')
    density = kde(kde_grid)
    kde_curves[i] = density
    mask = density > 1e-15
    h_raw = -np.trapz(density[mask] * np.log2(density[mask]), kde_grid[mask])
    # Normalize: 0 = delta fonksiyonu, 1 = uniform dağılım
    kde_entropy[i] = np.clip(h_raw / H_uniform, 0.0, 1.0)

# --- (c) Ortalama Faz Uzayı Uzaklaşması ---
print("   → Faz Uzayı Uzaklaşması Hesaplanıyor...")
mean_divergence = np.zeros(frames)
for i in range(frames):
    tips = np.column_stack([x2_all[:, i], y2_all[:, i]])
    mean_divergence[i] = np.mean(pdist(tips))

# --- (d) Lyapunov Üsteli Tahmini ---
print("   → Lyapunov Üsteli Hesaplanıyor...")
state0 = np.column_stack([
    np.sin(theta1_all[:, 0]), np.cos(theta1_all[:, 0]), omega1_all[:, 0],
    np.sin(theta2_all[:, 0]), np.cos(theta2_all[:, 0]), omega2_all[:, 0]
])
dist_matrix_0 = cdist(state0, state0)
np.fill_diagonal(dist_matrix_0, np.inf)
nn_idx = np.argmin(dist_matrix_0, axis=1)
d0 = dist_matrix_0[np.arange(N_pendulums), nn_idx]

lyapunov = np.full(frames, np.nan)
for i in range(1, frames):
    state_i = np.column_stack([
        np.sin(theta1_all[:, i]), np.cos(theta1_all[:, i]), omega1_all[:, i],
        np.sin(theta2_all[:, i]), np.cos(theta2_all[:, i]), omega2_all[:, i]
    ])
    d_t = np.linalg.norm(state_i - state_i[nn_idx], axis=1)
    valid = (d_t > 1e-15) & (d0 > 1e-15)
    if np.any(valid):
        lyapunov[i] = np.mean(np.log(d_t[valid] / d0[valid])) / t_eval[i]

# --- (e) Toplam Enerji (Korunum Kontrolü) ---
print("   → Enerji Hesaplanıyor...")
def compute_energy(th1, w1, th2, w2):
    """Çift sarkaç toplam enerjisi (kinetik + potansiyel)"""
    KE = (0.5 * (M1 + M2) * L1**2 * w1**2
          + 0.5 * M2 * L2**2 * w2**2
          + M2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
    PE = (-(M1 + M2) * G * L1 * np.cos(th1)
          - M2 * G * L2 * np.cos(th2))
    return KE + PE

energy_all = compute_energy(theta1_all, omega1_all, theta2_all, omega2_all)  # (N, frames)
energy_mean = np.mean(energy_all, axis=0)
energy_std = np.std(energy_all, axis=0)
energy_init = energy_all[:, 0]
energy_drift = np.mean(np.abs(energy_all - energy_init[:, None]) / np.abs(energy_init[:, None] + 1e-15), axis=0) * 100

# --- (f) Korelasyon Boyutu Tahmini (Grassberger-Procaccia) ---
print("   → Korelasyon Boyutu Hesaplanıyor...")
corr_dim = np.full(frames, np.nan)
for i in range(5, frames, 3):  # Her 3 frame'de bir, hesaplama yükünü azalt
    tips = np.column_stack([x2_all[:, i], y2_all[:, i]])
    dists = pdist(tips)
    if len(dists) == 0 or np.max(dists) < 1e-10:
        continue
    r_values = np.logspace(np.log10(max(np.min(dists[dists > 0]), 1e-8)),
                           np.log10(np.max(dists)), 15)
    C_r = np.array([np.mean(dists < r) for r in r_values])
    valid_cr = (C_r > 0.01) & (C_r < 0.95)
    if np.sum(valid_cr) >= 4:
        log_r = np.log(r_values[valid_cr])
        log_C = np.log(C_r[valid_cr])
        coeffs = np.polyfit(log_r, log_C, 1)
        corr_dim[i] = coeffs[0]

# Eksik frame'leri interpole et
valid_cd = ~np.isnan(corr_dim)
if np.sum(valid_cd) > 2:
    corr_dim_interp = np.interp(t_eval, t_eval[valid_cd], corr_dim[valid_cd])
else:
    corr_dim_interp = np.zeros(frames)

print("✅ Tüm hesaplamalar tamamlandı!\n")

# ==========================================
# 5. GÖRSELLEŞTİRME (3×2 Grid)
# ==========================================
C_MAG   = '#ff2d9b'
C_CYAN  = '#00e5ff'
C_LIME  = '#76ff03'
C_AMBER = '#ffab00'
C_ORANGE = '#ff6d00'
C_VIOLET = '#b388ff'
C_WHITE = '#e0e0e0'
C_BG    = '#0d0d0d'

def style_ax(ax, title, xlabel='', ylabel='', ycolor=C_WHITE):
    ax.set_facecolor(C_BG)
    ax.set_title(title, color=C_WHITE, fontsize=11, fontweight='bold', pad=8)
    if xlabel: ax.set_xlabel(xlabel, color='#888888', fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=ycolor, fontsize=9)
    ax.tick_params(colors='#666666', labelsize=8)
    ax.grid(True, color='#1a1a1a', alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color('#333333')

fig = plt.figure(figsize=(18, 14))
fig.canvas.manager.set_window_title('Kaotik Çift Sarkaç — İstatistiksel Analiz')
fig.patch.set_facecolor('#0a0a0a')
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.38)

# ── Panel 1: Sarkaç Animasyonu ──
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, f'Çift Sarkaç Simülasyonu  (N={N_pendulums})')
ax1.set_xlim(-2.5, 2.5); ax1.set_ylim(-2.5, 2.5); ax1.set_aspect('equal')
ax1.grid(False)

show_idx = np.linspace(0, N_pendulums - 1, N_show, dtype=int)
pend_lines = []
for idx in show_idx:
    c = plt.cm.plasma(idx / N_pendulums)
    line, = ax1.plot([], [], 'o-', lw=1.2, color=c, alpha=0.35, markersize=2)
    pend_lines.append(line)

# FIX: scatter'ı başlangıç verileriyle oluştur, boş array ile değil
tip_scatter = ax1.scatter(
    x2_all[:, 0], y2_all[:, 0],
    s=10, c=np.arange(N_pendulums), cmap='plasma',
    alpha=0.75, zorder=5, edgecolors='none'
)

time_text = ax1.text(0.03, 0.95, '', transform=ax1.transAxes, color=C_WHITE,
                     fontsize=11, fontweight='bold', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# ── Panel 2: KDE Yoğunluk Dağılımı (θ₂ doğrudan) ──
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'θ₂ Açısal Dağılımı (KDE)', xlabel='θ₂ (rad)', ylabel='Olasılık Yoğunluğu', ycolor=C_CYAN)
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(0, 1)
ax2.tick_params(axis='y', colors=C_CYAN, labelcolor=C_CYAN)
# Uniform dağılım referans çizgisi: p = 1/(2π)
ax2.axhline(y=1.0/(2*np.pi), color='#444444', ls='--', lw=0.8, alpha=0.6)
ax2.text(np.pi*0.95, 1.0/(2*np.pi)+0.01, 'uniform', color='#666666', fontsize=7, ha='right')
kde_line, = ax2.plot(kde_grid, np.zeros_like(kde_grid), color=C_CYAN, lw=2)
kde_fill_coll = [ax2.fill_between(kde_grid, 0, 0, color=C_CYAN, alpha=0.25)]
kde_info = ax2.text(0.97, 0.95, '', transform=ax2.transAxes, color=C_WHITE, fontsize=9,
                    ha='right', va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# ── Panel 3: Uzaklaşma + Lyapunov ──
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 'Faz Uzayı Uzaklaşması & Lyapunov Üsteli', xlabel='Zaman (s)',
         ylabel='Ortalama Uzaklık (m)', ycolor=C_LIME)
ax3.set_xlim(0, T_sim); ax3.set_ylim(0, np.max(mean_divergence) * 1.15)
ax3.tick_params(axis='y', colors=C_LIME, labelcolor=C_LIME)
line_div, = ax3.plot([], [], color=C_LIME, lw=2, label='⟨d(t)⟩ Ort. Uzaklık')

ax3b = ax3.twinx()
ax3b.set_ylabel('λ  Lyapunov (1/s)', color=C_AMBER, fontsize=9)
ax3b.tick_params(axis='y', colors=C_AMBER, labelcolor=C_AMBER)
lyap_valid = lyapunov[~np.isnan(lyapunov)]
if len(lyap_valid) > 10:
    ax3b.set_ylim(min(0, np.percentile(lyap_valid, 2)), np.percentile(lyap_valid, 98) * 1.4)
else:
    ax3b.set_ylim(-1, 5)
for sp in ax3b.spines.values(): sp.set_color('#333333')
line_lyap, = ax3b.plot([], [], color=C_AMBER, lw=2, linestyle='--', label='λ Lyapunov')

h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax3b.get_legend_handles_labels()
ax3.legend(h1 + h2, l1 + l2, loc='upper left', facecolor='black', labelcolor=C_WHITE, fontsize=8)

# ── Panel 4: Normalize Entropi + Std Sapma ──
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'Normalize Entropi & Dairesel Standart Sapma', xlabel='Zaman (s)',
         ylabel='H / H_uniform', ycolor=C_CYAN)
ax4.set_xlim(0, T_sim)
ax4.set_ylim(-0.05, 1.1)
ax4.axhline(y=1.0, color='#444444', ls='--', lw=0.8, alpha=0.5)
ax4.text(T_sim * 0.98, 1.03, 'H = H_uniform (tam kaos)', color='#666666', fontsize=7, ha='right')
ax4.tick_params(axis='y', colors=C_CYAN, labelcolor=C_CYAN)
line_ent, = ax4.plot([], [], color=C_CYAN, lw=2, label='Norm. Entropi (H/H_max)')

ax4b = ax4.twinx()
ax4b.set_ylabel('Dairesel σ (°)', color=C_MAG, fontsize=9)
ax4b.tick_params(axis='y', colors=C_MAG, labelcolor=C_MAG)
ax4b.set_ylim(-5, max(circular_std_deg) * 1.15)
for sp in ax4b.spines.values(): sp.set_color('#333333')
line_std, = ax4b.plot([], [], color=C_MAG, lw=2, linestyle='--', label='σ Dairesel Std.')

h3, l3 = ax4.get_legend_handles_labels()
h4, l4 = ax4b.get_legend_handles_labels()
ax4.legend(h3 + h4, l3 + l4, loc='center right', facecolor='black', labelcolor=C_WHITE, fontsize=8)

# ── Panel 5: Enerji Korunumu ──
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, 'Toplam Enerji Korunumu', xlabel='Zaman (s)',
         ylabel='⟨E⟩ Enerji (J)', ycolor=C_ORANGE)
ax5.set_xlim(0, T_sim)
ax5.set_ylim(np.min(energy_mean - energy_std) * 1.05, np.max(energy_mean + energy_std) * 0.95)
ax5.tick_params(axis='y', colors=C_ORANGE, labelcolor=C_ORANGE)
line_energy, = ax5.plot([], [], color=C_ORANGE, lw=2, label='⟨E(t)⟩ Ortalama')
energy_band = [ax5.fill_between([], [], [], color=C_ORANGE, alpha=0.15)]

ax5b = ax5.twinx()
ax5b.set_ylabel('ΔE/E₀ (%)', color=C_VIOLET, fontsize=9)
ax5b.tick_params(axis='y', colors=C_VIOLET, labelcolor=C_VIOLET)
max_drift = max(np.max(energy_drift), 0.01)
ax5b.set_ylim(-0.005, max_drift * 1.3)
for sp in ax5b.spines.values(): sp.set_color('#333333')
line_drift, = ax5b.plot([], [], color=C_VIOLET, lw=2, linestyle='--', label='ΔE/E₀ Sapma')

h5, l5 = ax5.get_legend_handles_labels()
h6, l6 = ax5b.get_legend_handles_labels()
ax5.legend(h5 + h6, l5 + l6, loc='upper left', facecolor='black', labelcolor=C_WHITE, fontsize=8)

energy_info = ax5.text(0.97, 0.95, '', transform=ax5.transAxes, color=C_WHITE, fontsize=9,
                       ha='right', va='top', family='monospace',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# ── Panel 6: Korelasyon Boyutu ──
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, 'Korelasyon Boyutu (Grassberger-Procaccia)', xlabel='Zaman (s)',
         ylabel='D₂ Boyut', ycolor=C_LIME)
ax6.set_xlim(0, T_sim)
cd_valid_vals = corr_dim_interp[corr_dim_interp > 0]
if len(cd_valid_vals) > 0:
    ax6.set_ylim(0, max(np.max(cd_valid_vals) * 1.3, 3.0))
else:
    ax6.set_ylim(0, 3.0)
ax6.tick_params(axis='y', colors=C_LIME, labelcolor=C_LIME)
line_corrdim, = ax6.plot([], [], color=C_LIME, lw=2, label='D₂ Korelasyon Boyutu')
# Referans çizgileri
ax6.axhline(y=1.0, color='#444444', ls='--', lw=0.8, alpha=0.5)
ax6.axhline(y=2.0, color='#444444', ls='--', lw=0.8, alpha=0.5)
ax6.text(T_sim * 0.98, 1.05, 'D=1 (çizgi)', color='#666666', fontsize=7, ha='right')
ax6.text(T_sim * 0.98, 2.05, 'D=2 (yüzey)', color='#666666', fontsize=7, ha='right')
ax6.legend(loc='upper left', facecolor='black', labelcolor=C_WHITE, fontsize=8)

corrdim_info = ax6.text(0.97, 0.95, '', transform=ax6.transAxes, color=C_WHITE, fontsize=9,
                        ha='right', va='top', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

# Dikey zaman çizgileri
vline3 = ax3.axvline(0, color=C_WHITE, ls=':', lw=0.7, alpha=0.4)
vline4 = ax4.axvline(0, color=C_WHITE, ls=':', lw=0.7, alpha=0.4)
vline5 = ax5.axvline(0, color=C_WHITE, ls=':', lw=0.7, alpha=0.4)
vline6 = ax6.axvline(0, color=C_WHITE, ls=':', lw=0.7, alpha=0.4)

# ==========================================
# 6. ANİMASYON DÖNGÜSÜ
# ==========================================
def update(frame):
    artists = []

    # Panel 1 — Sarkaçlar
    for j, idx in enumerate(show_idx):
        th1 = theta1_all[idx, frame]
        th2 = theta2_all[idx, frame]
        x1 = L1 * np.sin(th1); y1 = -L1 * np.cos(th1)
        x2 = x1 + L2 * np.sin(th2); y2 = y1 - L2 * np.cos(th2)
        pend_lines[j].set_data([0, x1, x2], [0, y1, y2])
    tip_scatter.set_offsets(np.column_stack([x2_all[:, frame], y2_all[:, frame]]))
    time_text.set_text(f't = {t_eval[frame]:.2f} s')
    artists += pend_lines + [tip_scatter, time_text]

    # Panel 2 — KDE
    density = kde_curves[frame]
    kde_line.set_ydata(density)
    max_d = max(np.max(density), 0.1)
    ax2.set_ylim(0, max_d * 1.2)
    kde_fill_coll[0].remove()
    kde_fill_coll[0] = ax2.fill_between(kde_grid, 0, density, color=C_CYAN, alpha=0.25)
    kde_info.set_text(f'H/H_max = {kde_entropy[frame]:.3f}\nσ = {circular_std_deg[frame]:.1f}°')
    artists += [kde_line, kde_fill_coll[0], kde_info]

    # Panel 3 — Uzaklaşma + Lyapunov
    f = frame + 1
    line_div.set_data(t_eval[:f], mean_divergence[:f])
    lyap_plot = lyapunov[:f].copy()
    lyap_plot[np.isnan(lyap_plot)] = 0
    line_lyap.set_data(t_eval[:f], lyap_plot)
    vline3.set_xdata([t_eval[frame], t_eval[frame]])
    artists += [line_div, line_lyap, vline3]

    # Panel 4 — Entropi + Std
    line_ent.set_data(t_eval[:f], kde_entropy[:f])
    line_std.set_data(t_eval[:f], circular_std_deg[:f])
    vline4.set_xdata([t_eval[frame], t_eval[frame]])
    artists += [line_ent, line_std, vline4]

    # Panel 5 — Enerji
    line_energy.set_data(t_eval[:f], energy_mean[:f])
    energy_band[0].remove()
    energy_band[0] = ax5.fill_between(t_eval[:f],
                                       energy_mean[:f] - energy_std[:f],
                                       energy_mean[:f] + energy_std[:f],
                                       color=C_ORANGE, alpha=0.15)
    line_drift.set_data(t_eval[:f], energy_drift[:f])
    vline5.set_xdata([t_eval[frame], t_eval[frame]])
    energy_info.set_text(f'⟨E⟩ = {energy_mean[frame]:.3f} J\nΔE = {energy_drift[frame]:.4f}%')
    artists += [line_energy, energy_band[0], line_drift, vline5, energy_info]

    # Panel 6 — Korelasyon Boyutu
    line_corrdim.set_data(t_eval[:f], corr_dim_interp[:f])
    vline6.set_xdata([t_eval[frame], t_eval[frame]])
    corrdim_info.set_text(f'D₂ = {corr_dim_interp[frame]:.2f}')
    artists += [line_corrdim, vline6, corrdim_info]

    return artists

ani = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
plt.tight_layout()
plt.show()