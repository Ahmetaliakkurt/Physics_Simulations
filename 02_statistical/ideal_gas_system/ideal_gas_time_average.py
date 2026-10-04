import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. SİMÜLASYON PARAMETRELERİ
# ==========================================
N = 300           # Parçacık Sayısı
L = 10.0          # Kutu Boyutu (L x L)
m = 1.0           # Parçacık Kütlesi
v_max = 5.0       # Başlangıç Maksimum Hızı
dt = 0.005        # Zaman Adımı
radius = 0.001     # Çarpışma Yarıçapı

# Hücre tabanlı çarpışma için grid
CELL_SIZE = 2 * radius * 3  # Her hücre en az 1 çarpışmayı kapsar

TARGET_T = 300.0
TARGET_KE = N * TARGET_T  # 2D: E_k = N * k_B * T (k_B = 1)

# ==========================================
# 2. BAŞLANGIÇ KOŞULLARI
# ==========================================
# Grid üzerine yerleştir — üst üste binmeyi önle
cols = int(np.sqrt(N)) + 1
rows = int(np.ceil(N / cols))
spacing = (L - 2 * radius) / max(cols, rows)
positions = []
for i in range(rows):
    for j in range(cols):
        if len(positions) < N:
            x = radius + spacing * (j + 0.5) + np.random.uniform(-spacing * 0.1, spacing * 0.1)
            y = radius + spacing * (i + 0.5) + np.random.uniform(-spacing * 0.1, spacing * 0.1)
            x = np.clip(x, radius, L - radius)
            y = np.clip(y, radius, L - radius)
            positions.append([x, y])

pos = np.array(positions[:N], dtype=float)

# Rastgele hızlar
angles = np.random.uniform(0, 2 * np.pi, N)
speeds = np.random.uniform(1.0, v_max, N)
vel = np.zeros((N, 2))
vel[:, 0] = speeds * np.cos(angles)
vel[:, 1] = speeds * np.sin(angles)

# --- KRİTİK: VEKTÖREL MOMENTUM SIFIRLA ---
# Toplam momentum = 0 olacak şekilde merkez kütle hızını çıkar
vel -= vel.mean(axis=0)

# Enerji ölçeklendir
current_ke = np.sum(0.5 * m * (vel[:, 0]**2 + vel[:, 1]**2))
vel *= np.sqrt(TARGET_KE / current_ke)

# Momentum sıfırlama tekrar (ölçekleme sonrası kontrol)
vel -= vel.mean(axis=0)

initial_ke = TARGET_KE
P_theo = initial_ke / (L**2)

# ==========================================
# 3. VERİ LİSTELERİ
# ==========================================
time_data = []
pressure_data = []
instant_pressure_data = []
ke_data = []
te_data = []
error_data = []
px_data = []  # Toplam x-momentumu
py_data = []  # Toplam y-momentumu

accumulated_momentum = 0.0
x_max = 50.0

# ==========================================
# 4. ÇARPIŞMA FONKSİYONU (Hücre Tabanlı)
# ==========================================
def handle_particle_collisions(pos, vel, radius, L, cell_size):
    """
    Elastic particle-particle collision with cell-based neighbor search.
    Sadece yakın parçacıkları kontrol eder → O(N) yerine O(N*k) karmaşıklık.
    """
    n_cells_x = max(1, int(L / cell_size))
    n_cells_y = max(1, int(L / cell_size))
    cs_x = L / n_cells_x
    cs_y = L / n_cells_y

    cell_idx_x = np.floor(pos[:, 0] / cs_x).astype(int).clip(0, n_cells_x - 1)
    cell_idx_y = np.floor(pos[:, 1] / cs_y).astype(int).clip(0, n_cells_y - 1)
    cell_id = cell_idx_x * n_cells_y + cell_idx_y

    # Her hücreden parçacık listesi oluştur
    order = np.argsort(cell_id)
    sorted_cells = cell_id[order]

    # Komşu hücre çiftlerini kontrol et
    diameter_sq = (2 * radius) ** 2

    # Basit yaklaşım: her parçacık için aynı ve komşu hücrelerdeki parçacıklara bak
    cell_dict = {}
    for idx in range(N):
        c = cell_id[idx]
        if c not in cell_dict:
            cell_dict[c] = []
        cell_dict[c].append(idx)

    for cx in range(n_cells_x):
        for cy in range(n_cells_y):
            cid = cx * n_cells_y + cy
            if cid not in cell_dict:
                continue
            particles_here = cell_dict[cid]
            # Komşu hücreler (dahil kendisi)
            neighbors = []
            for dcx in [-1, 0, 1]:
                for dcy in [-1, 0, 1]:
                    ncx = cx + dcx
                    ncy = cy + dcy
                    if 0 <= ncx < n_cells_x and 0 <= ncy < n_cells_y:
                        ncid = ncx * n_cells_y + ncy
                        if ncid in cell_dict:
                            neighbors.extend(cell_dict[ncid])

            for i in particles_here:
                for j in neighbors:
                    if j <= i:
                        continue
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < diameter_sq and dist_sq > 1e-10:
                        dist = np.sqrt(dist_sq)
                        # Birim normal vektör
                        nx = dx / dist
                        ny = dy / dist
                        # Göreceli hız
                        dvx = vel[j, 0] - vel[i, 0]
                        dvy = vel[j, 1] - vel[i, 1]
                        # Yaklaşıyor mu?
                        dot = dvx * nx + dvy * ny
                        if dot < 0:
                            # Elastik çarpışma impülsü (eşit kütle)
                            impulse = dot  # (2*m1*m2/(m1+m2)) / m = 1 eşit kütlede
                            vel[i, 0] += impulse * nx
                            vel[i, 1] += impulse * ny
                            vel[j, 0] -= impulse * nx
                            vel[j, 1] -= impulse * ny
                            # Örtüşmeyi çöz
                            overlap = 2 * radius - dist
                            pos[i, 0] -= overlap * nx * 0.5
                            pos[i, 1] -= overlap * ny * 0.5
                            pos[j, 0] += overlap * nx * 0.5
                            pos[j, 1] += overlap * ny * 0.5

    return pos, vel


# ==========================================
# 5. GRAFİK KURULUMU
# ==========================================
fig = plt.figure(figsize=(20, 10), facecolor='#0d0d0d')
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1.3, 0.9],
                      hspace=0.45, wspace=0.35)

DARK_BG  = '#0d0d0d'
PANEL_BG = '#141414'
GRID_COL = '#2a2a2a'
TEXT_COL = '#e0e0e0'
ACC_BLUE = '#00aaff'
ACC_RED  = '#ff4455'
ACC_GRN  = '#00e676'
ACC_YLW  = '#ffcc00'
ACC_MAG  = '#ff44cc'
ACC_ORG  = '#ff8800'

def style_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=9, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, color='#888', fontsize=8)
    ax.set_ylabel(ylabel, color='#888', fontsize=8)
    ax.tick_params(colors='#666', labelsize=7)
    ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# --- Sol: Gaz Kutusu ---
ax_box = fig.add_subplot(gs[0:2, 0])
ax_box.set_facecolor('#050510')
ax_box.set_xlim(0, L)
ax_box.set_ylim(0, L)
ax_box.set_aspect('equal')
ax_box.set_xticks([])
ax_box.set_yticks([])
ax_box.set_title("İzole İdeal Gaz Sistemi", color=TEXT_COL, fontweight='bold', fontsize=10)
for spine in ax_box.spines.values():
    spine.set_edgecolor('#00aaff')
    spine.set_linewidth(2.5)

speeds_now = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
scatter = ax_box.scatter(pos[:, 0], pos[:, 1], s=12,
                         c=speeds_now, cmap='plasma',
                         alpha=0.85, vmin=0, vmax=v_max * 1.5)

# HUD
info_text_str = (f"N  = {N}\n"
                 f"L  = {L} m\n"
                 f"T  = {TARGET_T:.0f} K\n"
                 f"P₀ = {P_theo:.3f}")
ax_box.text(0.03, 0.97, info_text_str, transform=ax_box.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            color=ACC_GRN,
            bbox=dict(boxstyle='round', facecolor='#000020', alpha=0.85, edgecolor=ACC_BLUE))

# --- Sol alt: Hata ---
ax_error = fig.add_subplot(gs[2, 0])
style_ax(ax_error, "Basınç Hatası (Teorik'e Göre %)", "Zaman", "Hata (%)")
ax_error.set_xlim(0, x_max)
ax_error.set_ylim(0, 100)
line_error, = ax_error.plot([], [], lw=1.5, color=ACC_MAG)
ax_error.axhline(0, color='#444', lw=1)
error_text = ax_error.text(0.04, 0.88, "", transform=ax_error.transAxes,
                            fontsize=9, fontweight='bold', color=ACC_MAG)

# --- Orta üst: Basınç ---
ax_press = fig.add_subplot(gs[0, 1])
style_ax(ax_press, f"Zaman-Ortalamalı Basınç  (P₀={P_theo:.3f})", "Zaman", "P")
ax_press.set_xlim(0, x_max)
ax_press.set_ylim(0, P_theo * 2)
line_press, = ax_press.plot([], [], lw=1.5, color=ACC_RED, label='Ölçülen')
ax_press.axhline(P_theo, color=ACC_BLUE, linestyle='--', lw=1.2, label='Teorik')
ax_press.legend(fontsize=7, labelcolor=TEXT_COL, facecolor=PANEL_BG)

# --- Orta orta: Enerji ---
ax_energy = fig.add_subplot(gs[1, 1])
style_ax(ax_energy, "Sistem Enerjisi (Parçacık Çarpışmaları Dahil)", "Zaman", "E (J)")
ax_energy.set_xlim(0, x_max)
ax_energy.set_ylim(initial_ke * 0.85, initial_ke * 1.15)
line_ke, = ax_energy.plot([], [], lw=1.2, color=ACC_GRN, label='KE')
line_te, = ax_energy.plot([], [], lw=1.5, linestyle=':', color='white', alpha=0.7, label='TE')
ax_energy.axhline(initial_ke, color=ACC_YLW, linestyle='--', lw=1, alpha=0.6, label='E₀')
ax_energy.legend(fontsize=7, labelcolor=TEXT_COL, facecolor=PANEL_BG)

# --- Orta alt: Basınç dağılımı ---
ax_hist = fig.add_subplot(gs[2, 1])
style_ax(ax_hist, "Basınç Olasılık Dağılımı", "P", "Yoğunluk")

# --- Sağ üst: Momentum Px ---
ax_px = fig.add_subplot(gs[0, 2])
style_ax(ax_px, "Toplam Momentum Pₓ(t)", "Zaman", "Pₓ")
ax_px.set_xlim(0, x_max)
ax_px.set_ylim(-20, 20)
ax_px.axhline(0, color='#555', lw=1.2)
line_px, = ax_px.plot([], [], lw=1.2, color=ACC_BLUE)
px_val_text = ax_px.text(0.04, 0.88, "", transform=ax_px.transAxes,
                          fontsize=8, color=ACC_BLUE, family='monospace')

# --- Sağ orta: Momentum Py ---
ax_py = fig.add_subplot(gs[1, 2])
style_ax(ax_py, "Toplam Momentum Pᵧ(t)", "Zaman", "Pᵧ")
ax_py.set_xlim(0, x_max)
ax_py.set_ylim(-20, 20)
ax_py.axhline(0, color='#555', lw=1.2)
line_py, = ax_py.plot([], [], lw=1.2, color=ACC_ORG)
py_val_text = ax_py.text(0.04, 0.88, "", transform=ax_py.transAxes,
                          fontsize=8, color=ACC_ORG, family='monospace')

# --- Sağ alt: Makroskobik tablo ---
ax_dash = fig.add_subplot(gs[2, 2])
ax_dash.set_facecolor(PANEL_BG)
ax_dash.axis('off')
for spine in ax_dash.spines.values():
    spine.set_edgecolor('#333')
dash_text = ax_dash.text(0.08, 0.92, "", transform=ax_dash.transAxes,
                          fontsize=10.5, verticalalignment='top', family='monospace',
                          color=ACC_GRN,
                          bbox=dict(boxstyle='round,pad=0.8',
                                    facecolor='#0a0a0a',
                                    edgecolor=ACC_BLUE, alpha=0.95))

fig.patch.set_facecolor(DARK_BG)

# ==========================================
# 6. ANİMASYON FONKSİYONU
# ==========================================
COLLISION_EVERY = 2   # Her kaç frame'de bir çarpışma hesabı
frame_count = [0]

def animate(frame):
    global pos, vel, accumulated_momentum, x_max
    frame_count[0] += 1

    current_time = (frame + 1) * dt
    time_data.append(current_time)

    # --- Hareket ---
    pos += vel * dt

    # --- Duvar çarpışmaları ---
    hit_left   = pos[:, 0] <= radius
    hit_right  = pos[:, 0] >= L - radius
    hit_bottom = pos[:, 1] <= radius
    hit_top    = pos[:, 1] >= L - radius

    dp_instant = (np.sum(2 * m * np.abs(vel[hit_left,   0])) +
                  np.sum(2 * m * np.abs(vel[hit_right,  0])) +
                  np.sum(2 * m * np.abs(vel[hit_bottom, 1])) +
                  np.sum(2 * m * np.abs(vel[hit_top,    1])))

    accumulated_momentum += dp_instant

    vel[hit_left,   0] =  np.abs(vel[hit_left,   0])
    vel[hit_right,  0] = -np.abs(vel[hit_right,  0])
    vel[hit_bottom, 1] =  np.abs(vel[hit_bottom, 1])
    vel[hit_top,    1] = -np.abs(vel[hit_top,    1])

    # Sınır içine al
    pos[:, 0] = np.clip(pos[:, 0], radius, L - radius)
    pos[:, 1] = np.clip(pos[:, 1], radius, L - radius)

    # --- Parçacık-Parçacık Elastik Çarpışmalar ---
    # Her frame'de çalıştır; atlama yapmak örtüşme biriktirir
    pos, vel = handle_particle_collisions(pos, vel, radius, L, CELL_SIZE)

    # --- ENERJİ KORUNUMU: Rescale (adyabatik zorlama) ---
    # Momentum düzeltmesi (vel.mean çıkarma) enerji çalar — yapma.
    # Bunun yerine: elastik çarpışmalar zaten momentum+enerji korur.
    # Sadece sayısal birikim için enerjiyi TARGET'a rescale et,
    # ama MERKEZİ KÜTLE HIZINI ÇIKARMA.
    current_ke_raw = np.sum(0.5 * m * (vel[:, 0]**2 + vel[:, 1]**2))
    if current_ke_raw > 1e-10:
        vel *= np.sqrt(initial_ke / current_ke_raw)  # enerji koru, momentum yönlerini bozmaz

    # --- Fiziksel Büyüklükler ---
    current_pressure = accumulated_momentum / (current_time * 4 * L)
    pressure_data.append(current_pressure)

    current_error = (abs(current_pressure - P_theo) / P_theo) * 100.0
    error_data.append(current_error)

    instant_pressure = dp_instant / (dt * 4 * L)
    if instant_pressure > 0:
        instant_pressure_data.append(instant_pressure)

    current_ke = np.sum(0.5 * m * (vel[:, 0]**2 + vel[:, 1]**2))
    ke_data.append(current_ke)
    te_data.append(current_ke)  # PE=0, TE=KE (ideal gaz)

    # --- Toplam Vektörel Momentum ---
    total_px = np.sum(m * vel[:, 0])
    total_py = np.sum(m * vel[:, 1])
    px_data.append(total_px)
    py_data.append(total_py)

    current_temp = current_ke / N
    current_vol  = L ** 2

    # --- Eksen Güncelleme ---
    if current_time >= x_max:
        x_max += 50.0
        for ax in [ax_press, ax_energy, ax_error, ax_px, ax_py]:
            ax.set_xlim(0, x_max)

    if frame > 10:
        p_recent = pressure_data[-50:]
        if max(p_recent) > ax_press.get_ylim()[1] * 0.9:
            ax_press.set_ylim(0, max(p_recent) * 1.3)
        err_recent = error_data[-50:]
        ax_error.set_ylim(0, max(5.0, max(err_recent) * 1.2))

        # Momentum eksenlerini dinamik ayarla
        all_p = px_data + py_data
        p_amp = max(10.0, np.max(np.abs(all_p[-200:])) * 1.5) if all_p else 20.0
        ax_px.set_ylim(-p_amp, p_amp)
        ax_py.set_ylim(-p_amp, p_amp)

        # Enerji ekseni: küçük sayısal sapmaları göster
        ke_recent = ke_data[-200:]
        ke_mean = np.mean(ke_recent)
        ke_std  = np.std(ke_recent)
        margin  = max(ke_std * 6, initial_ke * 0.005)
        ax_energy.set_ylim(ke_mean - margin, ke_mean + margin)

    # --- Scatter Renk (hıza göre) ---
    speeds_now = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
    scatter.set_offsets(pos)
    scatter.set_array(speeds_now)

    # --- Çizgiler ---
    line_press.set_data(time_data, pressure_data)
    line_ke.set_data(time_data, ke_data)
    line_te.set_data(time_data, te_data)
    line_error.set_data(time_data, error_data)
    line_px.set_data(time_data, px_data)
    line_py.set_data(time_data, py_data)

    error_text.set_text(f"Hata: %{current_error:.2f}")
    px_val_text.set_text(f"Pₓ = {total_px:+.4f}")
    py_val_text.set_text(f"Pᵧ = {total_py:+.4f}")

    # --- Dashboard ---
    dash_str = (
        f"MAKROSKOBİK TABLO\n"
        f"─────────────────\n"
        f" N   = {N}\n"
        f" V   = {current_vol:.1f} m²\n"
        f" T   = {current_temp:.2f} K\n"
        f"─────────────────\n"
        f" Pₓ  = {total_px:+.3f}\n"
        f" Pᵧ  = {total_py:+.3f}\n"
        f"─────────────────\n"
        f" KE  = {current_ke:.1f} J\n"
        f" ΔE  = {current_ke - initial_ke:+.1f} J\n"
    )
    dash_text.set_text(dash_str)

    # --- Histogram (her 15 frame'de bir) ---
    if frame % 15 == 0 and len(instant_pressure_data) > 10:
        ax_hist.clear()
        style_ax(ax_hist, "Basınç Olasılık Dağılımı", "P", "Yoğunluk")
        ax_hist.hist(instant_pressure_data, bins=30, density=True,
                     color='#7c4dff', alpha=0.75, edgecolor='#333')
        mean_p = np.mean(instant_pressure_data)
        ax_hist.axvline(mean_p, color=ACC_RED, linestyle='--', lw=1.8,
                        label=f'μ={mean_p:.1f}')
        ax_hist.axvline(P_theo, color=ACC_BLUE, linestyle=':', lw=1.5,
                        label=f'P₀={P_theo:.1f}')
        ax_hist.legend(fontsize=7, labelcolor=TEXT_COL, facecolor=PANEL_BG)

    return (scatter, line_press, line_ke, line_te,
            line_error, line_px, line_py,
            error_text, px_val_text, py_val_text, dash_text)


ani = animation.FuncAnimation(
    fig, animate,
    frames=itertools.count(),
    interval=30,
    blit=False
)

plt.show()