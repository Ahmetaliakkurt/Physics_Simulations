import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# --- 1. ORTAK PARAMETRELER ---
N = 1000            
L = 6.0             
x = np.linspace(-L, L, N)
dx = x[1] - x[0]

hbar = 1.0
m = 1.0
V0 = 10.0          # BURAYI DEĞİŞTİRDİĞİNDE GRAFİK OTOMATİK UYARLANACAK
width = 2.5        

# --- ÇÖZÜCÜ VE ÇİZDİRİCİ FONKSİYON ---
def solve_and_plot(V_array, title_prefix):
    """
    Verilen V_array potansiyeli için Schrödinger denklemini çözer 
    ve ölçeklemeyi V0'a göre dinamik yapar.
    """
    # Hamiltonyen Matrisi Kurulumu
    diag_kin = (hbar**2) / (m * dx**2)
    off_diag_kin = - (hbar**2) / (2 * m * dx**2)

    main_diag = V_array + diag_kin
    off_diag = off_diag_kin * np.ones(N - 1)

    # Çözüm
    energies, wavefunctions = eigh_tridiagonal(main_diag, off_diag)

    # --- GRAFİK BAŞLANGICI ---
    plt.figure(figsize=(10, 7))
    
    # Potansiyeli çiz
    plt.plot(x, V_array, color='black', linewidth=2, alpha=0.4, label="Potential $V(x)$")
    plt.fill_between(x, V_array, -V0*1.15, color='gray', alpha=0.1)

    # --- DİNAMİK ÖLÇEKLEME AYARLARI (ÖNEMLİ KISIM) ---
    num_states = 3
    
    # Scale faktörünü V0'a göre ayarla (Kuyu derinliğinin %20'si kadar büyüklükte çiz)
    # Eğer V0 çok küçükse (örn 0.1), scale çok küçülmesin diye min 1.0 sınırı koyabiliriz.
    scale = max(V0 * 0.20, 0.5)  

    max_energy_plotted = -V0 # Başlangıç değeri

    for i in range(num_states):
        # Eğer enerji 0'dan büyükse (bağlı durum değilse) döngüyü kırabilir veya çizebilirsin.
        # Genellikle V0 küçükse bound state sayısı azalır.
        if energies[i] > 0 and i > 0: 
            # Sadece pozitif enerjili ilk state'i gösterelim, diğerlerini göstermeyelim (opsiyonel)
            pass

        psi = wavefunctions[:, i]
        E = energies[i]
        max_energy_plotted = E 
        
        # Normalizasyon
        norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
        psi = psi / norm
        prob = np.abs(psi)**2
        
        # 1. Dalga Fonksiyonu
        plt.plot(x, scale * psi + E, '--', color=f'C{i}', linewidth=2, alpha=0.9, 
                 label=f"$\psi_{i}$")
        
        # 2. Olasılık Yoğunluğu
        plt.fill_between(x, E, scale * prob + E, color=f'C{i}', alpha=0.4, 
                         label=f"$|\psi_{i}|^2$")
        
        # Enerji Referans Çizgisi
        plt.hlines(E, -L, L, color=f'C{i}', linestyle='-', linewidth=0.5, alpha=0.5)

    # --- GRAFİK AYARLARI ---
    plt.title(f"{title_prefix} (V0={V0}) - Energy Levels", fontsize=14)
    plt.xlabel("Position (x)", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    
    plt.xticks([])
    plt.yticks([])

    # --- DİNAMİK LİMİT AYARLARI ---
    # Alt limit: Kuyunun biraz altı
    bottom_limit = -V0 * 1.1 
    # Üst limit: Çizilen en yüksek enerji + dalganın boyu (scale) + biraz boşluk
    top_limit = max_energy_plotted + (scale * 1.5)

    plt.ylim(bottom_limit, top_limit)
    plt.xlim(-L, L)
    
    plt.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

# --- 2. ÇALIŞTIRMA ---

# 1. SMOOTH KUYU
power = 8
V_smooth = -V0 * np.exp(-(x / width)**power)
solve_and_plot(V_smooth, "SMOOTH WELL")

# 2. KARE KUYU
V_square = np.zeros_like(x)
mask = (x > -width) & (x < width)
V_square[mask] = -V0
solve_and_plot(V_square, "SQUARE WELL")