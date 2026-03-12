import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Ayarlar ve Fiziksel Parametreler ---
L = 100.0          # Uzay genişliği
N = 512            # Grid nokta sayısı
dx = L / N
x = np.linspace(-L/2, L/2, N)

# Zaman parametreleri
dt = 0.05          # Zaman adımı
t_max = 30.0       # Toplam simülasyon süresi
frames = int(t_max / dt)

# --- Potansiyel: Çift Lorentzian ---
x1, x2 = 5.0, 15.0   # Bariyer merkezleri
width = 1.0          # Genişlik
V0 = 2.0             # Yükseklik
V = V0 / (1 + ((x - x1)/width)**2) + V0 / (1 + ((x - x2)/width)**2)

# --- Başlangıç Dalga Fonksiyonu ---
x0 = -20.0
k0 = 2.5
sigma = 2.0
psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)

# --- Split-Step Operatörleri ---
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
exp_T = np.exp(-1j * (k**2 / 2) * dt)
exp_V_half = np.exp(-1j * V * dt / 2)

# --- Görselleştirme ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plt.subplots_adjust(hspace=0.3)

# 1. Grafik: Reel Kısım
ax1.set_xlim(-L/2, L/2)
ax1.set_ylim(-1, 1)
ax1.set_title(r"Reel Kısım: Re($\psi$)")
ax1.set_ylabel("Genlik")
# Potansiyeli ekle (Yüksekliği 0.8'e ölçeklendi)
ax1.fill_between(x, V * (0.8/V0), color='gray', alpha=0.2, label='Potansiyel')
ax1.legend(loc='upper right', fontsize='small')

# 2. Grafik: İmajiner Kısım
ax2.set_xlim(-L/2, L/2)
ax2.set_ylim(-1, 1)
ax2.set_title(r"İmajiner Kısım: Im($\psi$)")
ax2.set_ylabel("Genlik")
# Potansiyeli ekle (Yüksekliği 0.8'e ölçeklendi)
ax2.fill_between(x, V * (0.8/V0), color='gray', alpha=0.2, label='Potansiyel')
ax2.legend(loc='upper right', fontsize='small')

# 3. Grafik: Olasılık Yoğunluğu
ax3.set_xlim(-L/2, L/2)
ax3.set_ylim(0, 1.0)
ax3.set_title(r"Olasılık Yoğunluğu: $|\psi|^2$")
ax3.set_xlabel("Konum (x)")
ax3.set_ylabel("Olasılık")
# Potansiyeli ekle (Yüksekliği 0.2'ye ölçeklendi)
ax3.fill_between(x, V * (0.2/V0), color='gray', alpha=0.3, label='Potansiyel')
ax3.legend(loc='upper right', fontsize='small')

# Çizgi nesneleri
line_re, = ax1.plot([], [], 'b-', lw=1)
line_im, = ax2.plot([], [], 'r-', lw=1)
line_prob, = ax3.plot([], [], 'k-', lw=2)

psi_curr = psi.copy()

def init():
    line_re.set_data([], [])
    line_im.set_data([], [])
    line_prob.set_data([], [])
    return line_re, line_im, line_prob

def animate(frame):
    global psi_curr
    
    # Split-Step Adımları
    psi_curr *= exp_V_half
    psi_fft = np.fft.fft(psi_curr)
    psi_fft *= exp_T
    psi_curr = np.fft.ifft(psi_fft)
    psi_curr *= exp_V_half
    
    # Güncelleme
    line_re.set_data(x, np.real(psi_curr))
    line_im.set_data(x, np.imag(psi_curr))
    line_prob.set_data(x, np.abs(psi_curr)**2)
    
    return line_re, line_im, line_prob

anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)
plt.show()