import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. ANSAMBLE PARAMETRELERİ 
# ==========================================
M = 100000        # PARALEL EVREN SAYISI (100 Bin Evren -> ~1.6 GB RAM)
N = 500           # Her evrendeki parçacık sayısı
L = 10.0          # Kutu Boyutu
m = 1.0           # Kütle
dt = 0.005         # Zaman Adımı
radius = 0.15     # Parçacık Yarıçapı
STEPS = 200       # Toplam Simülasyon Adımı (Zaman çözünürlüğü)

# --- TERMODİNAMİK HEDEF (Sıcaklık Girdisi) ---
TARGET_T = 300.0             # Hedef Sıcaklık (Kelvin)
TARGET_KE = N * TARGET_T   # 2D için E_k = N * k_B * T (k_B = 1 alındı)
P_theo = TARGET_KE / (L**2)

print("=" * 60)
print("🚀 ANSAMBLE SİMÜLASYONU BAŞLIYOR")
print("=" * 60)
print(f"[{time.strftime('%X')}] 1. Bellek tahsis ediliyor... ({M} Sistem, Toplam {M*N} Atom)")

# ==========================================
# 2. TENSÖR İNŞASI (RAM'E YÜKLEME)
# ==========================================
pos = np.random.uniform(radius, L - radius, size=(M, N, 2))
angles = np.random.uniform(0, 2 * np.pi, size=(M, N))
speeds = np.random.uniform(1.0, 5.0, size=(M, N))

vel = np.zeros((M, N, 2))
vel[:, :, 0] = speeds * np.cos(angles)
vel[:, :, 1] = speeds * np.sin(angles)

# Başlangıç enerjilerini hedefe sabitleme (Sıcaklığa göre ölçekleme)
current_ke_per_univ = np.sum(0.5 * m * (vel[:, :, 0]**2 + vel[:, :, 1]**2), axis=1)
scaling_factor = np.sqrt(TARGET_KE / current_ke_per_univ)

vel[:, :, 0] *= scaling_factor[:, np.newaxis]
vel[:, :, 1] *= scaling_factor[:, np.newaxis]

# Veri kaydetme dizileri
time_data = np.zeros(STEPS)
ensemble_pressure_data = np.zeros(STEPS)
ke_data = np.zeros(STEPS)
pe_data = np.zeros(STEPS)

print(f"[{time.strftime('%X')}] 2. Simülasyon döngüsü başladı. Animasyon kapalı, CPU saf fizik hesaplıyor...")
start_time = time.time()

# ==========================================
# 3. SESSİZ HESAPLAMA DÖNGÜSÜ (CPU TAM YÜK)
# ==========================================
for step in range(STEPS):
    current_time = (step + 1) * dt
    time_data[step] = current_time
    
    # Kinetik hareket
    pos += vel * dt
    
    # Çarpışma Maskeleri (Duvar tespiti)
    hit_left = pos[:, :, 0] <= radius
    hit_right = pos[:, :, 0] >= L - radius
    hit_bottom = pos[:, :, 1] <= radius
    hit_top = pos[:, :, 1] >= L - radius
    
    # Evren başına momentum aktarımı
    dp_left = np.sum(2 * m * np.abs(vel[:, :, 0]) * hit_left, axis=1)
    dp_right = np.sum(2 * m * np.abs(vel[:, :, 0]) * hit_right, axis=1)
    dp_bottom = np.sum(2 * m * np.abs(vel[:, :, 1]) * hit_bottom, axis=1)
    dp_top = np.sum(2 * m * np.abs(vel[:, :, 1]) * hit_top, axis=1)
    
    # 100 Bin evrenin o anki basınç listesi
    P_instant_per_universe = (dp_left + dp_right + dp_bottom + dp_top) / (dt * 4 * L)
    
    # Ansamble Ortalaması (Kayıt)
    ensemble_pressure_data[step] = np.mean(P_instant_per_universe)
    
    # Esnek Yansıma Algoritması
    v_x = vel[:, :, 0]
    p_x = pos[:, :, 0]
    vel[:, :, 0] = np.where(p_x <= radius, np.abs(v_x), v_x)
    vel[:, :, 0] = np.where(p_x >= L - radius, -np.abs(v_x), vel[:, :, 0])
    
    v_y = vel[:, :, 1]
    p_y = pos[:, :, 1]
    vel[:, :, 1] = np.where(p_y <= radius, np.abs(v_y), v_y)
    vel[:, :, 1] = np.where(p_y >= L - radius, -np.abs(v_y), vel[:, :, 1])

    # Enerji Kontrolü
    ke_data[step] = np.mean(np.sum(0.5 * m * (vel[:, :, 0]**2 + vel[:, :, 1]**2), axis=1))
    pe_data[step] = 0.0

    # Terminale ilerleme durumu bas (Her 50 adımda bir)
    if (step + 1) % 50 == 0:
        print(f"  -> Adım {step + 1}/{STEPS} işlendi...")

calc_time = time.time() - start_time
print(f"[{time.strftime('%X')}] 3. Hesaplama bitti! Toplam süre: {calc_time:.2f} saniye.")

# ==========================================
# 4. FİNAL İSTATİSTİKSEL HESAPLAMALAR VE TERMİNAL ÇIKTISI
# ==========================================
overall_mean_p = np.mean(ensemble_pressure_data)
relative_error = np.abs(overall_mean_p - P_theo) / P_theo * 100

print("\n" + "=" * 60)
print("📊 FİNAL İSTATİSTİKSEL SONUÇLAR")
print("=" * 60)
print(f"Hedeflenen Sıcaklık    : {TARGET_T:.2f} K")
print(f"Teorik Beklenen Basınç : {P_theo:.6f} Pa")
print(f"Ölçülen Ansamble Ort.  : {overall_mean_p:.6f} Pa")
print(f"Bağıl (Relative) Hata  : % {relative_error:.6f}")
print("=" * 60 + "\n")

print(f"[{time.strftime('%X')}] 4. İstatistiksel Rapor hazırlanıyor...")

# ==========================================
# 5. FİNAL STATİK ÇİZİM (MATPLOTLIB) VE TABLO
# ==========================================
# 4 Panelli geniş bir pencere (Grafikler + Tablo)
fig = plt.figure(figsize=(22, 5))
fig.canvas.manager.set_window_title('İzole İdeal Gaz - Ansamble Analiz Raporu')
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.6])

# Grafik 1: Basınç vs Zaman
ax_press = fig.add_subplot(gs[0])
ax_press.plot(time_data, ensemble_pressure_data, color='darkred', lw=2, label='Ansamble P(t)')
ax_press.axhline(P_theo, color='blue', linestyle='--', lw=2, label=f'Teorik P ({P_theo:.1f})')
ax_press.set_title("Zamana Bağlı Ansamble Basıncı", fontweight='bold')
ax_press.set_xlabel("Zaman (s)")
ax_press.set_ylabel("Ortalama Basınç")
ax_press.grid(True, linestyle='--', alpha=0.6)
ax_press.legend()

# Grafik 2: Enerji vs Zaman
ax_energy = fig.add_subplot(gs[1])
ax_energy.plot(time_data, ke_data, color='green', lw=2, label='Ortalama Kinetik Enerji')
ax_energy.plot(time_data, pe_data, color='orange', lw=2, label='Ortalama Potansiyel Enerji')
ax_energy.axhline(TARGET_KE, color='black', linestyle=':', lw=2, label='Hedef Kinetik Enerji')
ax_energy.set_title("100 Bin Evrenin Ortalama Enerjisi", fontweight='bold')
ax_energy.set_xlabel("Zaman (s)")
ax_energy.set_ylabel("Enerji (J)")
ax_energy.set_ylim(-10, TARGET_KE + 50)
ax_energy.grid(True, linestyle='--', alpha=0.6)
ax_energy.legend()

# Grafik 3: Son Adımdaki Basınç Dağılımı (Histogram)
ax_hist = fig.add_subplot(gs[2])
ax_hist.hist(P_instant_per_universe, bins=50, density=True, color='purple', alpha=0.7, edgecolor='black')
ax_hist.axvline(P_theo, color='blue', linestyle='--', linewidth=2, label=f'Teorik P ({P_theo:.1f})')
final_mean_p = np.mean(P_instant_per_universe)
ax_hist.axvline(final_mean_p, color='red', linestyle='-', linewidth=2, label=f'Ansamble Ort. ({final_mean_p:.1f})')
ax_hist.set_title(f"Final Adımı: {M} Evrenin Basınç Dağılımı", fontweight='bold')
ax_hist.set_xlabel("Anlık Basınç (P)")
ax_hist.set_ylabel("Olasılık Yoğunluğu")
ax_hist.grid(True, linestyle='--', alpha=0.6)
ax_hist.legend()

# Tablo Paneli 4: Makroskobik Parametreler (V, N, T)
final_vol = L**2

ax_dash = fig.add_subplot(gs[3])
ax_dash.axis('off')
dashboard_str = (
    f"MAKROSKOBİK TABLO\n"
    f"+-------+----------+\n"
    f"| Param | Değer    |\n"
    f"+-------+----------+\n"
    f"|   N   | {N:<8} |\n"
    f"|   V   | {final_vol:<4.1f} m²  |\n"
    f"|   T   | {TARGET_T:<5.2f} K  |\n"
    f"+-------+----------+\n"
)
ax_dash.text(0.1, 0.8, dashboard_str, transform=ax_dash.transAxes, 
             fontsize=14, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#1e1e1e', edgecolor='#444444', alpha=0.9),
             color='#00ff00')

plt.tight_layout()
plt.show()