import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import warnings

warnings.filterwarnings("ignore")

# ── 1. SABİT PARAMETRELER ─────────────────────────────────────────────────────
L0    = 1.0
L_end = 10.0
N_bas = 20
hbar  = 1.0
m     = 1.0

# ── 2. BAĞLANMA MATRİSİ ───────────────────────────────────────────────────────
def M_nm(n, m, L):
    if n == m:            return 0.0
    if (n + m) % 2 == 0: return 0.0
    return -(2.0 * n * m) / (L * (n**2 - m**2))

# ── 3. PARAMETRE TARAMASI (Zaman Evrim Operatörü İle) ─────────────────────────
v_values = np.logspace(np.log10(0.005), np.log10(100.0), 40)

# Hassasiyet ayarı: Duvarın L=1'den L=10'a gidişi kaç adımda bölünecek?
# 2000 adım demek, her döngüde duvarın sadece 0.0045 birim ilerlemesi demektir.
N_steps = 2000  

p_ground_list = []   
p_excited_list = []

print("Hesaplama Başlıyor")

# Hızlandırmak için enerji çarpanlarını önceden hesaplıyoruz (n^2 * pi^2 * hbar^2 / 2m)
n_array = np.arange(1, N_bas + 1)
E_factors = (n_array**2 * np.pi**2 * hbar**2) / (2.0 * m)

for v in v_values:
    # Her hız için geçecek toplam süreye göre adım aralığı (Delta t)
    dt = (L_end - L0) / (v * N_steps)
    
    c = np.zeros(N_bas, dtype=complex)
    c[0] = 1.0 + 0j
    
    # Kuantum durumu N_steps boyunca geleceğe taşı (Propagator Loop)
    for step in range(N_steps):
        # Orta nokta entegrasyonu (Midpoint Rule) ile hassasiyeti artırıyoruz
        t_mid = (step + 0.5) * dt
        L_mid = L0 + v * t_mid
        
        # Etkin Hamiltoniyen Matrisinin (A = -i * H_eff / hbar) İnşası
        A = np.zeros((N_bas, N_bas), dtype=complex)
        
        for i in range(N_bas):
            # Köşegen (Enerji) Elemanları
            E_i = E_factors[i] / (L_mid**2)
            A[i, i] = -1j * E_i / hbar
            
            # Çapraz (Bağlanma - Kinetik Şok) Elemanları
            for j in range(N_bas):
                if i != j:
                    M_val = M_nm(i + 1, j + 1, L_mid)
                    if M_val != 0.0:
                        A[i, j] = -v * M_val
        
        # Zaman Evrim Operatörü: U(t, t+dt) = exp(A * dt)
        U = expm(A * dt)
        
        # Dalga fonksiyonunu bir adım ileri ışınla
        c = U @ c
        
    # Simülasyon bittiğinde (L=10'a ulaşıldığında) temel durumdaki olasılığı hesapla
    p1 = np.abs(c[0])**2
    
    p_ground_list.append(p1)
    p_excited_list.append(1.0 - p1)
print("\nHesaplama Tamamlandı.")

# ── 4. LİMİT GRAFİĞİNİN ÇİZİMİ ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d0f1a')
ax.set_facecolor('#0d0f1a')

ax.plot(v_values, p_excited_list, marker='o', color='#ff7043', 
        linewidth=2, markersize=5, label='Total Excitation Probability ($1 - |c_1|^2$)')

ax.axvspan(0.001, 0.1, color='#4fc3f7', alpha=0.1, label='Adiabatic Regime')
ax.axvspan(0.1, 10.0, color='#ffee58', alpha=0.1, label='Transition Regime')
ax.axvspan(10.0, 200.0, color='#ef5350', alpha=0.1, label='Sudden (Non-Quasistatic) Regime')

ax.set_xscale('log')
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(min(v_values), max(v_values))

ax.set_title("Quantum Adiabaticity Boundary: Velocity vs Excitation (expm)", color='white', fontsize=14, pad=15)
ax.set_xlabel("Expansion Velocity ($v$) — Log Scale", color='#90a4ae', fontsize=12)
ax.set_ylabel("Excitation Probability (Non-Quasistaticity)", color='#90a4ae', fontsize=12)

ax.tick_params(colors='#607d8b', labelsize=10)
for sp in ax.spines.values(): sp.set_color('#1e2030')
ax.grid(True, color='#1e2030', lw=1, alpha=0.8, which="both", ls="--")

legend = ax.legend(loc='center right', facecolor='#0d0f1a', edgecolor='#1e2030', fontsize=10)
for text in legend.get_texts(): text.set_color('white')

plt.tight_layout()
plt.show()