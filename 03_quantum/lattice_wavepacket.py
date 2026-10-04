import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, RadioButtons, Slider
import matplotlib.colors as mcolors

# ==========================================
# 1. Simulation Parameters (AKICI KARE EVREN)
# ==========================================
Nx = 384             
Ny = 384             
Lx = 60.0            
Ly = 60.0            
dt = 0.05            # Fiziğin daha hassas ve yavaş evrilmesi için küçültüldü
steps_per_frame = 2  # Animasyonun "zıplamaması" ve pürüzsüz akması için 4'ten 2'ye düşürüldü

# ==========================================
# 2. Setup the Spatial and Momentum Grids
# ==========================================
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky)

T = 0.5 * (KX**2 + KY**2)  

kx_plot = np.fft.fftshift(kx)
ky_plot = np.fft.fftshift(ky)

# ==========================================
# 3. Lattice Generation Function
# ==========================================
V0 = 80.0          
spacing = 4.0      
radius = 0.8       

def generate_lattice(lattice_type):
    V_temp = np.zeros_like(X)
    
    if lattice_type == 'SC':
        for i in np.arange(0, Lx/2, spacing):
            for j in np.arange(-Ly/2, Ly/2 + spacing, spacing):
                V_temp += V0 * np.exp(-((X - i)**2 + (Y - j)**2) / (radius**2))
                
    elif lattice_type == 'BCC':
        for i in np.arange(0, Lx/2, spacing):
            for j in np.arange(-Ly/2, Ly/2 + spacing, spacing):
                V_temp += V0 * np.exp(-((X - i)**2 + (Y - j)**2) / (radius**2))
        for i in np.arange(spacing/2, Lx/2, spacing):
            for j in np.arange(-Ly/2 + spacing/2, Ly/2 + spacing, spacing):
                V_temp += V0 * np.exp(-((X - i)**2 + (Y - j)**2) / (radius**2))
                
    elif lattice_type == 'FCC':
        row = 0
        dx_fcc = spacing * np.sqrt(3)/2 
        for i in np.arange(0, Lx/2, dx_fcc):
            y_shift = (spacing / 2.0) if (row % 2 == 1) else 0.0
            for j in np.arange(-Ly/2 - spacing, Ly/2 + spacing, spacing):
                V_temp += V0 * np.exp(-((X - i)**2 + (Y - (j + y_shift))**2) / (radius**2))
            row += 1
    return V_temp

current_lattice = 'SC'
V = generate_lattice(current_lattice)
U_V = np.exp(-1j * V * dt / 2)  
U_T = np.exp(-1j * T * dt)      

# ==========================================
# 4. Wave Packet Initialization & Sponge
# ==========================================
current_theta = 0.0 
k0_mag = 10.0       
current_sigma = spacing / 4.0  

def create_wave_packet():
    theta_rad = np.radians(current_theta)
    k_x = k0_mag * np.cos(theta_rad)
    k_y = k0_mag * np.sin(theta_rad)
    
    R = 15.0 
    x0 = -R * np.cos(theta_rad)
    y0 = -R * np.sin(theta_rad)
            
    sigma = current_sigma            
    psi_temp = np.exp(-((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2)) * np.exp(1j * (k_x * X + k_y * Y))
    psi_temp /= np.sqrt(np.sum(np.abs(psi_temp)**2) * (Lx/Nx) * (Ly/Ny))
    return psi_temp

psi = create_wave_packet()

sponge = np.ones_like(X)
thickness = 6.0  
right_zone = X > (Lx/2 - thickness)
sponge[right_zone] *= np.exp(-0.05 * (X[right_zone] - (Lx/2 - thickness))**2)
left_zone = X < (-Lx/2 + thickness)
sponge[left_zone] *= np.exp(-0.05 * (np.abs(X[left_zone]) - (Lx/2 - thickness))**2)
top_zone = Y > (Ly/2 - thickness)
sponge[top_zone] *= np.exp(-0.05 * (Y[top_zone] - (Ly/2 - thickness))**2)
bottom_zone = Y < (-Ly/2 + thickness)
sponge[bottom_zone] *= np.exp(-0.05 * (np.abs(Y[bottom_zone]) - (Ly/2 - thickness))**2)

# ==========================================
# 5. Visualization setup with UI Layout
# ==========================================
fig = plt.figure(figsize=(12, 7)) 
plt.subplots_adjust(bottom=0.25, wspace=0.3) 
fig.patch.set_facecolor('black')

ax_real = fig.add_subplot(121)
ax_real.set_facecolor('black')
density = np.abs(psi)**2
im_real = ax_real.imshow(density, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], 
                         origin='lower', cmap='turbo', interpolation='bilinear', 
                         vmax=np.max(density)*0.4, zorder=1) 

colors = [(1, 1, 1, 0), (1, 1, 1, 0.4)] 
white_cmap = mcolors.LinearSegmentedColormap.from_list('white_alpha', colors)
lattice_im = ax_real.imshow(V, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2], 
                            origin='lower', cmap=white_cmap, vmax=V0, zorder=2)
ax_real.axvline(x=0, color='lightblue', linestyle='--', alpha=0.3, zorder=3)
ax_real.set_title("Real Space (Scattering)", color='white')
ax_real.tick_params(colors='white')

ax_k = fig.add_subplot(122)
ax_k.set_facecolor('black')
psi_k = np.fft.fft2(psi)
density_k = np.fft.fftshift(np.abs(psi_k)**2)
k_extent = [kx_plot.min(), kx_plot.max(), ky_plot.min(), ky_plot.max()]

gamma = 0.70
density_k_display = density_k ** gamma

im_k = ax_k.imshow(density_k_display, extent=k_extent, origin='lower', 
                   cmap='inferno', vmax=np.max(density_k_display)*0.6)

circle = plt.Circle((0, 0), k0_mag, color='red', fill=False, linestyle='--', linewidth=1.5, alpha=0.7)
ax_k.add_patch(circle)

ax_k.set_title("Momentum Space (Laue / Bragg Peaks)", color='white')
ax_k.set_xlim(-20, 15)
ax_k.set_ylim(-15, 15)
ax_k.axhline(0, color='gray', linestyle=':', alpha=0.5)
ax_k.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax_k.tick_params(colors='white')

# ==========================================
# 6. Interactive Widgets
# ==========================================
is_running = True

ax_button = plt.axes([0.45, 0.10, 0.1, 0.05])
btn_play_pause = Button(ax_button, 'Pause')

def toggle_pause(event):
    global is_running
    is_running = not is_running
    btn_play_pause.label.set_text('Pause' if is_running else 'Play')

btn_play_pause.on_clicked(toggle_pause)

ax_radio = plt.axes([0.05, 0.05, 0.12, 0.12], facecolor='lightgray')
radio = RadioButtons(ax_radio, ('SC', 'BCC', 'FCC'))

def change_lattice(label):
    global V, U_V, psi
    V = generate_lattice(label)
    U_V = np.exp(-1j * V * dt / 2)
    psi = create_wave_packet() 
    lattice_im.set_data(V) 
    im_real.set_data(np.abs(psi)**2)
    
    new_density_k = np.fft.fftshift(np.abs(np.fft.fft2(psi))**2)
    im_k.set_data(new_density_k ** gamma)
    fig.canvas.draw_idle()

radio.on_clicked(change_lattice)

ax_angle = plt.axes([0.22, 0.05, 0.25, 0.03], facecolor='lightgray')
slider_angle = Slider(ax_angle, 'Angle', -45.0, 45.0, valinit=0.0)

def update_angle(val):
    global current_theta, psi
    current_theta = val
    psi = create_wave_packet() 
    im_real.set_data(np.abs(psi)**2)
    
    new_density_k = np.fft.fftshift(np.abs(np.fft.fft2(psi))**2)
    im_k.set_data(new_density_k ** gamma)
    fig.canvas.draw_idle()

slider_angle.on_changed(update_angle)

ax_sigma = plt.axes([0.65, 0.05, 0.18, 0.12], facecolor='lightgray')
radio_sigma = RadioButtons(ax_sigma, ('$a/4$', '$a/2$', '$a$'))

def change_sigma(label):
    global current_sigma, psi
    if label == '$a/4$':
        current_sigma = spacing / 4.0
    elif label == '$a/2$':
        current_sigma = spacing / 2.0
    elif label == '$a$':
        current_sigma = spacing
        
    psi = create_wave_packet() 
    im_real.set_data(np.abs(psi)**2)
    
    new_density_k = np.fft.fftshift(np.abs(np.fft.fft2(psi))**2)
    im_k.set_data(new_density_k ** gamma)
    fig.canvas.draw_idle()

radio_sigma.on_clicked(change_sigma)

# ==========================================
# 7. Animation Loop
# ==========================================
def update(frame):
    global psi
    
    if is_running:
        for _ in range(steps_per_frame):
            psi *= U_V                   
            psi_k_temp = np.fft.fft2(psi)     
            psi_k_temp *= U_T                 
            psi = np.fft.ifft2(psi_k_temp)    
            psi *= U_V 
            psi *= sponge
            
        im_real.set_data(np.abs(psi)**2)
        
        psi_k_out = np.fft.fft2(psi)
        density_k_out = np.fft.fftshift(np.abs(psi_k_out)**2)
        im_k.set_data(density_k_out ** gamma)
        
    return [im_real, im_k]

ani = animation.FuncAnimation(fig, update, frames=1500, interval=1, blit=False)
plt.show()