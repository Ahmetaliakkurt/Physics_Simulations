# Double-slit görselleştirmesi — örnek estetik (Colab / Jupyter)
# Gereksinimler: numpy, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# --------------------------
# Fiziksel / sayısal parametreler
# --------------------------
hbar = 1.0
m = 1.0

# grid (Colab için başlangıçta düşük çözünürlük önerilir)
Nx, Ny = 180, 180        # 180x180 iyi denge; eğer yavaşsa 128x128 deneyin
Lx, Ly = 30.0, 30.0      # uzay boyutu (örn. Å biriminde düşün)
dx = Lx / Nx
dy = Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# zaman parametreleri
dt = 0.015
n_steps = 800
frames_interval = 4      # her frame için atlanan iç adım sayısı
n_frames = n_steps // frames_interval

# Başlangıç: sola doğru gelen gauss dalga paketi
x0 = -10.0
y0 = 0.0
sigma_x = 1.2
sigma_y = 1.2
k0 = 6.0                 # merkez momentumu (büyüt -> daha küçük dalga boyu)

def gaussian_packet(X, Y):
    envelope = np.exp(-((X - x0)**2)/(4*sigma_x**2) - ((Y - y0)**2)/(4*sigma_y**2))
    phase = np.exp(1j * k0 * X)
    psi = envelope * phase
    # normalize (diskret)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dx * dy)
    return psi / norm

psi = gaussian_packet(X, Y)

# --------------------------
# Bariyer: yatay doğrultuda ince barier içinde iki yarık
# --------------------------
V0 = 1e6                       # çok büyük -> neredeyse sonsuz duvar
barrier_x = 0.0
barrier_thickness = 0.6        # bariyer kalınlığı (x yönünde)
slit_half_height = 1.2         # yarık yarı yüksekliği (y yönünde)
slit_separation = 4.0          # iki yarık merkezleri arası mesafe

V = np.zeros_like(X)
barrier_mask = np.abs(X - barrier_x) < (barrier_thickness/2)
y_c1 = -slit_separation/2
y_c2 = +slit_separation/2
V[barrier_mask] = V0
V[barrier_mask & (np.abs(Y - y_c1) < slit_half_height)] = 0.0
V[barrier_mask & (np.abs(Y - y_c2) < slit_half_height)] = 0.0

# --------------------------
# Kenar emilimi (absorbing) — yansımaları azaltmak için
# --------------------------
absorb_frac = 0.10
absorb_width_x = int(absorb_frac * Nx)
absorb_width_y = int(absorb_frac * Ny)
absorb_mask = np.ones_like(X)
absorb_coeff = 0.045
# left-right
for i in range(absorb_width_x):
    fac = np.exp(-((absorb_coeff * (absorb_width_x - i))**2))
    absorb_mask[i, :] *= fac
    absorb_mask[-1-i, :] *= fac
# top-bottom
for j in range(absorb_width_y):
    fac = np.exp(-((absorb_coeff * (absorb_width_y - j))**2))
    absorb_mask[:, j] *= fac
    absorb_mask[:, -1-j] *= fac

# --------------------------
# Önişlem: k-space faktörü
# --------------------------
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
# split-step: exp(-i * (k^2)/(2m) * dt)
kinetic = np.exp(-1j * (KX**2 + KY**2) * (dt / (2*m)))

# potansiyel yarım adım propagatorü
Vprop_half = np.exp(-1j * V * dt / (2 * hbar))

# --------------------------
# Görselleştirme hazırlığı (estetik)
# --------------------------
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6.5,6.5))
extent = [x.min(), x.max(), y.min(), y.max()]

prob = np.abs(psi)**2
# renk paleti ve görüntü limiti — interferans çizgilerini parlak göstermek için ayar
im = ax.imshow(prob.T, origin='lower', extent=extent, cmap='nipy_spectral',
               vmin=0, vmax=prob.max()*0.6, interpolation='bilinear', aspect='equal')
# bariyeri beyaz çizgiyle öne çıkar
ax.contour(X.T, Y.T, (V>0).T, levels=[0.5], colors='white', linewidths=1.1)
ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
title = ax.set_title('t = 0.00 fs', color='white')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('|ψ(x,y,t)|²', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

# --------------------------
# split-step adımı
# --------------------------
def step_ssfm(psi):
    psi = Vprop_half * psi
    psi_k = np.fft.fft2(psi)
    psi_k = psi_k * kinetic
    psi = np.fft.ifft2(psi_k)
    psi = Vprop_half * psi
    psi = psi * absorb_mask
    return psi

# --------------------------
# animasyon güncellemesi
# --------------------------
def update(frame):
    global psi
    for _ in range(frames_interval):
        psi = step_ssfm(psi)
    t = frame * dt * frames_interval
    prob = np.abs(psi)**2
    # kontrastı dinamik ayarla (görsel etki için)
    im.set_data(prob.T)
    im.set_clim(0, prob.max()*0.6)
    title.set_text(f't = {t*1e3:.3f} fs')  # örnekte fs ölçeğinde gösteriyoruz, birim ölçekledik
    return [im, title]

# --------------------------
# animasyon oluşturma ve inline gösterim
# --------------------------
anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
plt.tight_layout()
HTML(anim.to_jshtml())
