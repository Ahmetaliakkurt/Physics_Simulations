import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.special import sph_harm, genlaguerre, factorial

pio.renderers.default = 'browser'

n = int(input("Enter the principal quantum number n (n ≥ 1): "))
l = int(input(f"Enter the azimuthal quantum number l (0 ≤ l < {n}): "))
m = int(input(f"Enter the magnetic quantum number m (−{l} ≤ m ≤ {l}): "))

if not (n >= 1 and 0 <= l < n and -l <= m <= l):
    raise ValueError("Invalid quantum numbers: ensure n ≥ 1, 0 ≤ l < n, and |m| ≤ l.")

def radial_wavefunction(n, l, r):
    rho = 2 * r / n
    norm = np.sqrt((2.0 / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    L = genlaguerre(n - l - 1, 2*l + 1)
    return norm * rho**l * np.exp(-rho/2) * L(rho)

def hydrogen_wavefunction(n, l, m, r, theta, phi):
    R = radial_wavefunction(n, l, r)
    Y = sph_harm(m, l, phi, theta)
    return R * Y

grid_pts = 60
axis = np.linspace(-30, 30, grid_pts)
X, Y, Z = np.meshgrid(axis, axis, axis, indexing='ij')

r = np.sqrt(X**2 + Y**2 + Z**2)
theta = np.arccos(np.divide(Z, r, out=np.zeros_like(r), where=r!=0))
phi = np.arctan2(Y, X)

psi_vals = hydrogen_wavefunction(n, l, m, r, theta, phi)
prob = np.abs(psi_vals)**2

fig = go.Figure(data=go.Volume(
    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
    value=prob.flatten(),
    isomin=prob.max()*0.03,
    isomax=prob.max()*0.5,
    opacity=0.15,
    surface_count=20,
    colorscale='Viridis',
    showscale=True,
    colorbar=dict(title='|ψ|²')
))
# --- go.update_layout Kısmını Güncelle ---
fig.update_layout(
    template='plotly_white',  # Temayı beyaza çekip arka planları elle gri yapacağız
    title=f'3D Probability Density of Hydrogen Atom State (n={n}, l={l}, m={m})',
    scene=dict(
        xaxis_title='x (a₀)',
        yaxis_title='y (a₀)',
        zaxis_title='z (a₀)',
        aspectmode='cube',
        # --- ARKA PLAN RENKLERİNİ BURADAN AYARLA ---
        # rgb(120, 120, 120) civarı orta gri bir tondur, morları gösterir.
        bgcolor='rgb(130, 130, 130)', 
        
        # Grid çizgilerini de belirginleştirelim
        xaxis=dict(gridcolor='rgb(180, 180, 180)', showbackground=False),
        yaxis=dict(gridcolor='rgb(180, 180, 180)', showbackground=False),
        zaxis=dict(gridcolor='rgb(180, 180, 180)', showbackground=False)
    ),
    # Dış arka planı çizim alanıyla aynı ya da biraz daha açık yapabilirsin
    paper_bgcolor='rgb(140, 140, 140)', 
    font=dict(color='black') # Arka plan açıldığı için yazılar siyah olmalı
)
fig.show()
