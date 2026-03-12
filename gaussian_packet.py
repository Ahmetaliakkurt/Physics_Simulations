import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

hbar   = 1.0
mass   = 1.0
k0     = 3.0
sigma  = 2.0
alpha  = 1 / (4 * sigma**2)

x = np.linspace(-20, 100, 600)
time_values = np.linspace(0, 30, 150)

def psi_xt(x, t):
    denom     = 1 + 2j * hbar * alpha * t / mass
    prefactor = (1/(2 * np.pi * sigma**2))**0.25 / np.sqrt(denom)
    phase     = np.exp(1j * (k0 * x - hbar * k0**2 * t / (2 * mass)))
    envelope  = np.exp(-alpha * (x - hbar * k0 * t / mass)**2 / denom)
    return prefactor * phase * envelope

psi0 = psi_xt(x, 0)
prob0 = np.abs(psi0)**2

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Wave Function (Real & Imag)', 'Probability Density |ψ|²'),
    vertical_spacing=0.15
)

fig.add_trace(go.Scatter(x=x, y=np.real(psi0), name='Re(ψ)', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=np.imag(psi0), name='Im(ψ)', line=dict(color='red', dash='dot')), row=1, col=1)

fig.add_trace(go.Scatter(x=x, y=prob0, name='|ψ|²', line=dict(color='green', width=2), fill='tozeroy'), row=2, col=1)

frames = []
for t in time_values:
    psi = psi_xt(x, t)
    prob = np.abs(psi)**2
    
    frames.append(go.Frame(
        data=[
            go.Scatter(y=np.real(psi)),
            go.Scatter(y=np.imag(psi)),
            go.Scatter(y=prob)
        ],
        name=f'{t:.2f}'
    ))

fig.frames = frames

fig.update_layout(
    title="Quantum Wave Packet Evolution",
    template="plotly_white",
    height=800,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        y=1.15, x=1.1,
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, dict(frame=dict(duration=30, redraw=True), 
                                       fromcurrent=True, mode='immediate')]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                         mode='immediate', transition=dict(duration=0))])]
    )],
    sliders=[dict(
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        steps=[dict(method='animate', 
                    args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                    label=f.name) for f in frames]
    )]
)

fig.update_yaxes(range=[-0.8, 0.8], row=1, col=1, title="Amplitude")
fig.update_yaxes(range=[0, 0.5], row=2, col=1, title="Probability")
fig.update_xaxes(title="Position (x)", row=2, col=1)

print("Animasyon dosyası oluşturuluyor...")
file_name = "quantum_simulation.html"
fig.write_html(file_name, auto_open=True)
print(f"İşlem tamam! '{file_name}' dosyasının tarayıcınızda açılmış olması lazım.")