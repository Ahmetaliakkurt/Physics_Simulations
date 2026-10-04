import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import dstn, idstn
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. PARAMETRELER 
# ==========================================
L = 20.0           # Büyük Kutu
N = 400            # Yüksek Grid Çözünürlüğü
hbar = 1.0
mass = 3.0         # DAĞILMAYI AZALTMAK İÇİN KÜTLE ARTIRILDI!

# Dalga Paketi
x0, y0 = 10.0, 10.0  
sigma = 1.0        # Belirsizlik ilkesi dengesi için optimize edildi
kx0, ky0 = 18.0, 14.0 # Başlangıç momentumu

x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

# ==========================================
# 2. İLK DURUM HAZIRLIĞI
# ==========================================
psi_0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) * \
        np.exp(1j * (kx0 * X + ky0 * Y))

# Sınır koşulları (Duvarlarda dalga 0 olmalı)
psi_0[[0, -1], :] = 0
psi_0[:, [0, -1]] = 0

# Normalizasyon
psi_0 /= np.sqrt(np.sum(np.abs(psi_0)**2) * (L/N)**2)

# Sadece iç noktaları al (Sınırlar her zaman 0 olduğu için DST'ye girmez)
psi_inner = psi_0[1:-1, 1:-1]

print("FFT (DST) Başlangıç durumu hesaplanıyor...")

# ==========================================
# 3. FFT (DST) TABANLI ZAMAN EVRİMİ
# ==========================================
# Başlangıç durumunu Spektral (Momentum) Uzayına geçir (Type 1 DST tam uyar)
# norm='ortho' dalganın toplam olasılığını korur
C = dstn(psi_inner, type=1, norm='ortho')

# Enerji Seviyelerini (E_nm) oluştur
n_vals = np.arange(1, N - 1)
n_X, n_Y = np.meshgrid(n_vals, n_vals) # X sütunlar, Y satırlar

E_scale = (np.pi**2 * hbar**2) / (2 * mass * L**2)
E_nm = (n_X**2 + n_Y**2) * E_scale

def get_psi_at_t(t):
    """Zaman evrimi sadece spektral katsayıların fazını döndürmektir."""
    # 1. Faz kaydırmayı uygula: exp(-iEt/hbar)
    C_t = C * np.exp(-1j * (E_nm / hbar) * t)
    
    # 2. Ters DST ile tekrar Konum Uzayına dön
    psi_t_inner = idstn(C_t, type=1, norm='ortho')
    
    # 3. Duvarları (0'ları) geri ekleyerek tam grid'i oluştur
    psi_t = np.zeros((N, N), dtype=complex)
    psi_t[1:-1, 1:-1] = psi_t_inner
    return psi_t

# ==========================================
# 4. PÜRÜZSÜZ GÖRSELLEŞTİRME
# ==========================================
fig, ax = plt.subplots(figsize=(9, 9))
fig.patch.set_facecolor('#050505')
ax.set_facecolor('#050505')

psi_init = get_psi_at_t(0)
prob_init = np.abs(psi_init)**2
prob_init_max = np.max(prob_init) # Başlangıç maksimum değerini hafızaya al

# 'bicubic' yüksek frekanslı girişim desenlerinde spline36'dan daha yumuşak çalışır
im = ax.imshow(prob_init, extent=[0, L, 0, L], origin='lower',
               cmap='inferno', interpolation='bicubic', vmax=prob_init_max)

ax.axis('off')

# Neon Duvar Çizgileri
rect = plt.Rectangle((0, 0), L, L, linewidth=3, edgecolor='#00ffcc', facecolor='none')
ax.add_patch(rect)

dt = 0.06 

def update(frame):
    t = frame * dt
    psi_t = get_psi_at_t(t)
    prob = np.abs(psi_t)**2
    
    im.set_array(prob)
    
    # KUSURSUZ KONTRAST AYARI:
    # Çarpışma anındaki yapıcı girişim parlamalarını filtreliyoruz.
    # Böylece arka plandaki sayısal pikseller (noise) görünmez oluyor.
    current_max = np.max(prob)
    limit = min(current_max * 0.85, prob_init_max * 1.5)
    im.set_clim(0, limit)
    
    return [im]

ani = animation.FuncAnimation(fig, update, frames=800, interval=25, blit=True)

print("FFT Simülasyonu Başladı! Daha derli toplu ve hızlı sekme hareketi izleniyor.")
plt.show()