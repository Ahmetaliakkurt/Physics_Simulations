import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib import animation

print("Simülasyon verileri hesaplanıyor, lütfen bekleyiniz...")

hbar = 1.0
m = 1.0
Nx, Ny = 160, 160
Lx, Ly = 30.0, 30.0 
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

dt = 0.015
n_steps = 700
frames_interval = 4
total_frames = n_steps // frames_interval

x0, y0 = -10.0, 0.0
k0 = 6.0
sigma = 1.2
psi = np.exp(-((X-x0)**2 + (Y-y0)**2)/(4*sigma**2)) * np.exp(1j*k0*X)
norm = np.sqrt(np.sum(np.abs(psi)**2)*dx*dy)
psi /= norm

V0 = 1e6
V = np.zeros_like(X)
barrier_mask = np.abs(X) < 0.3
slit_separation = 4.0
slit_width = 1.6
V[barrier_mask] = V0
mask_slit1 = (np.abs(Y - slit_separation/2) < slit_width/2)
mask_slit2 = (np.abs(Y + slit_separation/2) < slit_width/2)
V[barrier_mask & (mask_slit1 | mask_slit2)] = 0.0

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
kinetic_prop = np.exp(-1j * (KX**2 + KY**2) * dt / (2*m))
potential_prop = np.exp(-1j * V * dt / (2*hbar))

absorb = np.ones_like(X)
mask_edge = (np.abs(X) > Lx/2*0.9) | (np.abs(Y) > Ly/2*0.9)
absorb[mask_edge] = np.exp(-0.1) 

history = []
current_psi = psi.copy()
history.append(np.abs(current_psi)**2)

for i in range(total_frames):
    for _ in range(frames_interval):
        current_psi *= potential_prop
        psi_k = np.fft.fft2(current_psi)
        psi_k *= kinetic_prop
        current_psi = np.fft.ifft2(psi_k)
        current_psi *= potential_prop
        current_psi *= absorb
    
    history.append(np.abs(current_psi)**2)

print(f"Hesaplama tamamlandı. {len(history)} kare yüklendi.")

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(9, 7))
plt.subplots_adjust(bottom=0.25)

extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
im = ax.imshow(history[0].T, origin='lower', extent=extent, cmap='inferno', animated=True)
ax.contour(X.T, Y.T, (V>0).T, levels=[0.5], colors='white', alpha=0.3)
ax.set_title("Quantum Double Slit")
ax.set_xlabel("x"); ax.set_ylabel("y")
time_txt = ax.text(0.05, 0.95, "Frame: 0", transform=ax.transAxes, color='white')

ax_slider = plt.axes([0.2, 0.1, 0.60, 0.03], facecolor='gray')
slider = Slider(
    ax=ax_slider,
    label='Time',
    valmin=0,
    valmax=len(history)-1,
    valinit=0,
    valstep=1,
    color='orange'
)

ax_play = plt.axes([0.35, 0.025, 0.1, 0.04])
ax_stop = plt.axes([0.55, 0.025, 0.1, 0.04])
btn_play = Button(ax_play, 'Play', color='0.2', hovercolor='0.3')
btn_stop = Button(ax_stop, 'Stop', color='0.2', hovercolor='0.3')

is_playing = False

def update_image(val):
    idx = int(val)
    data = history[idx]
    im.set_data(data.T)
    vmax = np.max(data)
    if vmax > 1e-6:
        im.set_clim(0, vmax)
    
    time_txt.set_text(f"Frame: {idx}")
    fig.canvas.draw_idle()

def slider_on_change(val):
    update_image(val)

slider.on_changed(slider_on_change)

def play(event):
    global is_playing
    is_playing = True

def stop(event):
    global is_playing
    is_playing = False

btn_play.on_clicked(play)
btn_stop.on_clicked(stop)

def animate(i):
    if is_playing:
        current_val = slider.val
        next_val = current_val + 1
        
        if next_val >= len(history):
            next_val = 0
            
        slider.set_val(next_val)
    return im,

ani = animation.FuncAnimation(fig, animate, interval=50, blit=False)

plt.show()