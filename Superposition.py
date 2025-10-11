# Plotly interactive version
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
rng = np.random.RandomState(123)

x = np.linspace(-15,15,1500)
k0 = 5.0
sigma_k = 0.5
sigma_A = 0.2
A_total = 1.0
wave_list = []

def ensure_waves(n):
    while len(wave_list) < n:
        k = rng.normal(k0, sigma_k)
        A = abs(rng.normal(1.0, sigma_A))
        wave_list.append((k,A))

def compute_total(n):
    ensure_waves(n)
    used = wave_list[:n]
    ks = np.array([k for k,_ in used])
    As = np.array([A for _,A in used])
    As = As / As.sum() * A_total
    psi = np.zeros_like(x, dtype=complex)
    for k,A in zip(ks,As):
        psi += A * np.exp(1j*k*x)
    return np.real(psi)

slider = widgets.IntSlider(value=10, min=1, max=3000, description='N waves')

fig = go.FigureWidget()
fig.add_scatter(x=x, y=np.zeros_like(x), mode='lines')
fig.update_layout(height=400, title='Wave packet (Plotly)')

def update_plot(change):
    n = change['new']
    y = compute_total(n)
    fig.data[0].y = y

slider.observe(update_plot, names='value')
display(slider, fig)
# initial
update_plot({'new': slider.value})
