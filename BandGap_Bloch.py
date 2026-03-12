import numpy as np
import matplotlib.pyplot as plt

def plot_ultra_smooth_bands_with_two_gaps(ax, U_potential, N_basis=15):
    """
    Daha fazla G vektörü kullanarak (N_basis) pürüzsüz bantlar elde eder
    ve 1. ve 2. Yasak Enerji Aralıklarını (Gap) grafik üzerine yazar.
    """

    # 1. Kristal Parametreleri
    a = 1.0
    G = 2 * np.pi / a

    # Smoothluk için k nokta sayısını artırdık
    k_range = np.linspace(-np.pi/a, np.pi/a, 600)

    # Bant listeleri
    band1, band2, band3 = [], [], []

    # 2. Dinamik G Vektörleri Listesi Oluşturma
    max_n = N_basis // 2
    n_values = np.arange(-max_n, max_n + 1)
    Gs = n_values * G

    # 3. Hesaplama Döngüsü
    for k in k_range:
        lambdas = (k - Gs)**2
        H = np.zeros((N_basis, N_basis))

        for i in range(N_basis):
            for j in range(N_basis):
                if i == j: H[i, j] = lambdas[i]
                else:      H[i, j] = U_potential

        eigenvalues = np.linalg.eigvalsh(H)
        band1.append(eigenvalues[0])
        band2.append(eigenvalues[1])
        band3.append(eigenvalues[2])

    # 4. Çizim
    ax.plot(k_range, band1, color='black', linewidth=2.5, label='1. Bant')
    ax.plot(k_range, band2, color='black', linewidth=2.5, label='2. Bant')
    ax.plot(k_range, band3, color='black', linewidth=2.5, label='3. Bant')

    # --- Taralı Alanlar (Gaps) ---
    ax.fill_between(k_range, np.max(band1), np.min(band2),
                    color='gray', alpha=0.3, hatch='///', edgecolor='none')

    ax.fill_between(k_range, np.max(band2), np.min(band3),
                    color='gray', alpha=0.3, hatch='///', edgecolor='none')

    # --- GAP HESAPLAMA VE YAZDIRMA ---

    # 1. Gap Hesabı (Bant 1 Tepesi ile Bant 2 Dibi Arası)
    # Bu gap genellikle Brillouin Bölgesi sınırında (pi/a) açılır.
    gap1_val = np.min(band2) - np.max(band1)
    gap1_y_pos = (np.min(band2) + np.max(band1)) / 2

    # 2. Gap Hesabı (Bant 2 Tepesi ile Bant 3 Dibi Arası)
    # Bu gap genellikle Merkezde (k=0) açılır.
    gap2_val = np.min(band3) - np.max(band2)
    gap2_y_pos = (np.min(band3) + np.max(band2)) / 2

    # Yazı Stili
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5)

    # --- 1. GAP ETİKETİ (Sağ Kenara Yakın) ---
    ax.annotate(f'1. GAP\n{gap1_val:.3f} eV',
                xy=(np.pi/a, gap1_y_pos), xycoords='data',
                xytext=(np.pi/a * 0.6, gap1_y_pos), textcoords='data',
                arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                ha='center', va='center', color='red', fontweight='bold', bbox=bbox_props)

    # --- 2. GAP ETİKETİ (Merkeze Yakın) ---
    ax.annotate(f'2. GAP\n{gap2_val:.3f} eV',
                xy=(0, gap2_y_pos), xycoords='data',
                xytext=(np.pi/a * 0.5, gap2_y_pos + 1.5), textcoords='data', # Biraz yukarı alalım karışmasın
                arrowprops=dict(arrowstyle="->", color='darkblue', lw=1.5),
                ha='center', va='center', color='darkblue', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="darkblue", lw=1.5))


    # Etiketler ve Sınırlar
    ax.set_xlim(-np.pi/a, np.pi/a)
    ax.set_ylim(np.min(band1)-1, np.max(band3)+2)

    ax.set_xticks([-np.pi/a, 0, np.pi/a])
    ax.set_xticklabels(['$-\pi/a$', '0', '$\pi/a$'], fontsize=12)

    ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Dalga Vektörü ($k$)', fontsize=12)
    ax.set_ylabel('Enerji ($E$)', fontsize=12)
    ax.set_title(f"Bant Yapısı ve Gap Değerleri\n(U={U_potential} eV)", fontsize=12)

# --- Çalıştırma ---
fig, ax = plt.subplots(figsize=(8, 6))

# Lityum için U=0.9
plot_ultra_smooth_bands_with_two_gaps(ax, U_potential=2.75, N_basis=15)

plt.tight_layout()
plt.show()