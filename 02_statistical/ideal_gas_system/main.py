import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import fast_ensemble  # C++ motoru

warnings.filterwarnings("ignore")


# 1. ANSAMBLE PARAMETRELERİ
M = 100000        # Paralel Evren Sayısı
N = 500           # Her evrendeki parçacık sayısı
L = 10.0          # Kutu Boyutu
m = 1.0           # Kütle
dt = 0.005        # Zaman Adımı
radius = 0.001     # Parçacık Yarıçapı
STEPS = 200       # Toplam Simülasyon Adımı

K_PAIRS = N // 2  
TARGET_T = 300.0
TARGET_KE = N * TARGET_T   
P_theo = TARGET_KE / (L**2)

print("=" * 50)
print("C++ / OPENMP ANSAMBLE SİMÜLASYONU")
print("=" * 50)
print(f"Başlangıç matrisleri hazırlanıyor... ({M} Evren, Toplam {M*N} Atom)")


# 2. BAŞLANGIÇ MATRİSLERİNİ HAZIRLAMA

pos = np.random.uniform(radius, L - radius, size=(M, N, 2))
angles = np.random.uniform(0, 2 * np.pi, size=(M, N))
speeds = np.random.uniform(1.0, 5.0, size=(M, N))

vel = np.zeros((M, N, 2))
vel[:, :, 0] = speeds * np.cos(angles)
vel[:, :, 1] = speeds * np.sin(angles)

# Vektörel momentumu sıfırla (Her evren için ayrı ayrı)
cm_vel = vel.mean(axis=1, keepdims=True)
vel -= cm_vel

# Enerjiyi ölçekle
current_ke = np.sum(0.5 * m * (vel[:, :, 0]**2 + vel[:, :, 1]**2), axis=1)
scale = np.sqrt(TARGET_KE / current_ke)
vel[:, :, 0] *= scale[:, np.newaxis]
vel[:, :, 1] *= scale[:, np.newaxis]

# Son bir kez momentum sıfırlama (Ölçekleme kaymalarını önlemek için)
vel -= vel.mean(axis=1, keepdims=True)

print("Fizik Motoru Tetiklendi! C++ çekirdekleri devrede...")
start_time = time.time()

# 3. İŞİ C++ MOTORUNA DEVRETME

# 12 Çekirdeğin tamamı burada %100 yüke çıkacak
results = fast_ensemble.run_simulation(
    pos, vel, M, N, L, m, dt, radius, STEPS, K_PAIRS, TARGET_KE
)

calc_time = time.time() - start_time
print(f"Hesaplama Bitti! Toplam C++ süresi: {calc_time:.4f} saniye.")


# 4. VERİLERİ PARÇALAMA VE HATA HESABI

time_data = np.array(results["time_data"])
pressure_data = np.array(results["ensemble_pressure"])
ke_data = np.array(results["ke_data"])
px_data = np.array(results["mean_abs_px"])
py_data = np.array(results["mean_abs_py"])

error_data = (np.abs(pressure_data - P_theo) / P_theo) * 100.0

overall_mean_p = np.mean(pressure_data)
final_error = error_data[-1]

# 5. GRAFİK KURULUMU 

print("Grafikler oluşturuluyor...")

fig = plt.figure(figsize=(20, 10), facecolor='#0d0d0d')
fig.canvas.manager.set_window_title('C++ HPC - Ansamble Analiz Raporu')

# 2 Satır, 3 Sütunluk harika bir grid tasarımı
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)

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
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#888', fontsize=9)
    ax.set_ylabel(ylabel, color='#888', fontsize=9)
    ax.tick_params(colors='#666', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, color=GRID_COL)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

# --- Panel 1: Ansamble Basınç ---
ax_press = fig.add_subplot(gs[0, 0])
style_ax(ax_press, f"100.000 Evrenin Ort. Basıncı (P₀={P_theo:.3f})", "Zaman (s)", "Basınç (P)")
ax_press.plot(time_data, pressure_data, lw=2, color=ACC_RED, label='Ölçülen P')
ax_press.axhline(P_theo, color=ACC_BLUE, linestyle='--', lw=1.5, label='Teorik P')
ax_press.legend(fontsize=9, labelcolor=TEXT_COL, facecolor=PANEL_BG)

# --- Panel 2: Enerji (KE) ---
ax_energy = fig.add_subplot(gs[0, 1])
style_ax(ax_energy, "Sistem Enerjisi (Rescaling Testi)", "Zaman (s)", "KE (J)")
# Y eksenini milimetrik sapmaları görecek kadar daraltalım
margin = TARGET_KE * 0.001 
ax_energy.set_ylim(TARGET_KE - margin, TARGET_KE + margin)
ax_energy.plot(time_data, ke_data, lw=2, color=ACC_GRN, label='Kinetik Enerji')
ax_energy.axhline(TARGET_KE, color=ACC_YLW, linestyle='--', lw=1.5, alpha=0.6, label='Hedef E₀')
ax_energy.legend(fontsize=9, labelcolor=TEXT_COL, facecolor=PANEL_BG)

# --- Panel 3: Bağıl Hata ---
ax_error = fig.add_subplot(gs[1, 0])
style_ax(ax_error, "Basınç Hatası (Teorik'e Göre %)", "Zaman (s)", "Hata (%)")
ax_error.plot(time_data, error_data, lw=2, color=ACC_MAG)
ax_error.axhline(0, color='#444', lw=1.5)

# --- Panel 4: Momentum Px ---
ax_px = fig.add_subplot(gs[1, 1])
style_ax(ax_px, "Evren Başına Ort. |Pₓ| (≈0 Olmalı)", "Zaman (s)", "|Pₓ|")
ax_px.plot(time_data, px_data, lw=2, color=ACC_BLUE)
ax_px.axhline(0, color='#555', lw=1.5)

# --- Panel 5: Momentum Py ---
ax_py = fig.add_subplot(gs[1, 2])
style_ax(ax_py, "Evren Başına Ort. |Pᵧ| (≈0 Olmalı)", "Zaman (s)", "|Pᵧ|")
ax_py.plot(time_data, py_data, lw=2, color=ACC_ORG)
ax_py.axhline(0, color='#555', lw=1.5)

# --- Panel 6: Makroskobik Dashboard ---
ax_dash = fig.add_subplot(gs[0, 2])
ax_dash.set_facecolor(PANEL_BG)
ax_dash.axis('off')
for spine in ax_dash.spines.values():
    spine.set_edgecolor('#333')

dash_str = (
    f" MAKROSKOBİK TABLO\n"
    f" ─────────────────\n"
    f" Evren(M) = {M}\n"
    f" Atom(N)  = {N}\n"
    f" Kutu(V)  = {L**2:.1f} m²\n"
    f" Hedef T  = {TARGET_T:.1f} K\n"
    f" ─────────────────\n"
    f" Teorik P = {P_theo:.4f}\n"
    f" ÖlçülenP = {overall_mean_p:.4f}\n"
    f" Fin.Hata = % {final_error:.4f}\n"
    f" ─────────────────\n"
    f" Ort |Pₓ| = {px_data[-1]:.5f}\n"
    f" Ort |Pᵧ| = {py_data[-1]:.5f}\n"
    f" C++ Süre = {calc_time:.2f} s\n"
)

ax_dash.text(0.5, 0.5, dash_str, transform=ax_dash.transAxes,
             fontsize=11.5, verticalalignment='center', horizontalalignment='center', 
             family='monospace', color=ACC_GRN,
             bbox=dict(boxstyle='round,pad=1', facecolor='#0a0a0a', edgecolor=ACC_BLUE, alpha=0.95))

fig.patch.set_facecolor(DARK_BG)
plt.show()