import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import math
from scipy.special import genlaguerre, sph_harm

# --- FİZİK MOTORU ---
def get_radial_wf(n, l, r, Z=1):
    rho = 2 * Z * r / n
    num = (2 * Z / n)**3 * math.factorial(n - l - 1)
    den = 2 * n * math.factorial(n + l)
    norm = np.sqrt(num / den)
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * np.exp(-rho / 2) * (rho**l) * laguerre

ATOM_DATA = {
    'H': 1.0, 'He': 1.69, 'Li': 1.28, 'Be': 1.91, 'B': 2.42, 'C': 3.14, 'N': 3.83, 'O': 4.45, 'F': 5.10, 'Ne': 5.85,
    'Na': 2.51, 'Mg': 3.31, 'Al': 4.07, 'Si': 4.29, 'P': 4.89, 'S': 5.48, 'Cl': 6.12, 'Ar': 6.76,
    'K': 3.50, 'Ca': 4.40, 'Sc': 4.63, 'Ti': 4.82, 'V': 5.12, 'Cr': 5.13, 'Mn': 5.43, 'Fe': 5.73,
    'Co': 6.03, 'Ni': 6.33, 'Cu': 6.63, 'Zn': 6.93, 'Br': 9.03, 'Kr': 9.73, 'Ag': 10.5, 'Au': 12.0
}

class QuantumApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Core - Interaction & Insight")
        self.root.geometry("1300x950")
        self.root.configure(bg='#000000')

        # --- Sol Kontrol Paneli ---
        side_panel = tk.Frame(root, bg='#050505', padx=20, pady=20)
        side_panel.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(side_panel, text="QUANTUM LAB", font=('sans-serif', 16, 'bold'), bg='#050505', fg='#00ffcc').pack(pady=20)
        
        tk.Label(side_panel, text="Select Element:", bg='#050505', fg='white').pack()
        self.atom_var = tk.StringVar(value="H")
        self.atom_combo = ttk.Combobox(side_panel, textvariable=self.atom_var, values=list(ATOM_DATA.keys()), width=12)
        self.atom_combo.pack(pady=10)

        self.n_slider = self.create_slider(side_panel, "n (Energy Layer)", 1, 7, 1)
        self.l_slider = self.create_slider(side_panel, "l (Orbital Type)", 0, 6, 0)
        self.m_slider = self.create_slider(side_panel, "m (Orientation)", -6, 6, 0)

        tk.Label(side_panel, text="\nRender Resolution:", bg='#050505', fg='#666').pack()
        self.res_slider = tk.Scale(side_panel, from_=20, to=80, orient=tk.HORIZONTAL, bg='#050505', fg='#666', highlightthickness=0)
        self.res_slider.set(45)
        self.res_slider.pack()

        self.btn = tk.Button(side_panel, text="CALCULATE", command=self.draw, 
                             bg='#00ffcc', fg='#000', font=('sans-serif', 11, 'bold'), 
                             relief='flat', padx=20, pady=15)
        self.btn.pack(pady=30)

        # --- Veri Paneli ---
        data_frame = tk.LabelFrame(side_panel, text="Physical Data", bg='#050505', fg='#00ffcc', padx=10, pady=10)
        data_frame.pack(fill=tk.X, pady=10)
        self.rmp_label = tk.Label(data_frame, text="r_mp: --- a0", bg='#050505', fg='white', font=('sans-serif', 10))
        self.rmp_label.pack()

        tk.Label(side_panel, text="\nScroll: Zoom In/Out\nClick & Drag: Rotate", bg='#050505', fg='#444', font=('sans-serif', 8)).pack(side=tk.BOTTOM)

        # --- Sağ Grafik Alanı ---
        self.fig = plt.Figure(figsize=(10, 10), facecolor='#000000')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Fare tekerleği olayını (scroll) bağla
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        self.ax2 = None # 3D eksen referansı

    def create_slider(self, parent, label, f, t, start):
        tk.Label(parent, text=f"\n{label}:", bg='#050505', fg='white').pack()
        s = tk.Scale(parent, from_=f, to=t, orient=tk.HORIZONTAL, bg='#050505', fg='#00ffcc', highlightthickness=0, length=200)
        s.set(start)
        s.pack()
        return s

    def on_scroll(self, event):
        """Scroll ile Zoom İn/Out mantığı"""
        if self.ax2 is None or event.inaxes != self.ax2:
            return
        
        # Mevcut limitleri al
        cur_xlim = self.ax2.get_xlim()
        cur_ylim = self.ax2.get_ylim()
        cur_zlim = self.ax2.get_zlim()
        
        # Zoom oranı (%15)
        base_scale = 1.15
        if event.button == 'up': # Yakınlaştır
            scale_factor = 1 / base_scale
        elif event.button == 'down': # Uzaklaştır
            scale_factor = base_scale
        else:
            scale_factor = 1

        # Yeni limitleri hesapla ve uygula
        self.ax2.set_xlim([cur_xlim[0]*scale_factor, cur_xlim[1]*scale_factor])
        self.ax2.set_ylim([cur_ylim[0]*scale_factor, cur_ylim[1]*scale_factor])
        self.ax2.set_zlim([cur_zlim[0]*scale_factor, cur_zlim[1]*scale_factor])
        
        self.canvas.draw_idle()

    def draw(self):
        n, l, m = self.n_slider.get(), self.l_slider.get(), self.m_slider.get()
        res = self.res_slider.get()
        atom = self.atom_var.get()
        z_eff = ATOM_DATA.get(atom, 1.0)

        if l >= n:
            messagebox.showwarning("Selection Error", f"n={n} için l en fazla {n-1} olabilir.")
            return
        if abs(m) > l:
            messagebox.showwarning("Selection Error", f"l={l} için m, -{l} ile +{l} arasında olmalıdır.")
            return

        self.fig.clear()
        dynamic_limit = (n**2 / z_eff) * 5.0
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.4)
        
        # --- 1. RADYAL DAĞILIM ---
        ax1 = self.fig.add_subplot(gs[0], facecolor='#000000')
        r_max = dynamic_limit * 1.5
        r_vals = np.linspace(0, r_max, 2000)
        R = get_radial_wf(n, l, r_vals, z_eff)
        radial_dist = r_vals**2 * R**2
        
        max_idx = np.argmax(radial_dist)
        r_mp = r_vals[max_idx]
        self.rmp_label.config(text=f"r_mp: {r_mp:.3f} a0")

        ax1.fill_between(r_vals, radial_dist, color='#00ffcc', alpha=0.15)
        ax1.plot(r_vals, radial_dist, color='#00ffcc', lw=1.2)
        ax1.axvline(x=r_mp, color='white', linestyle='--', alpha=0.6, lw=1)
        ax1.text(r_mp, radial_dist[max_idx], f'  $r_{{mp}} = {r_mp:.2f} a_0$', color='white', verticalalignment='bottom', fontsize=8)
        
        ax1.set_title("Radial Probability Profile ($r^2|R_{nl}|^2$)", color='#00ffcc', fontsize=10, pad=10)
        ax1.set_xlabel("Distance ($a_0$)", color='#00ffcc', fontsize=9, labelpad=8)
        ax1.spines['bottom'].set_color('#00ffcc')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.tick_params(axis='x', colors='#00ffcc', labelsize=8)
        ax1.tick_params(axis='y', left=False, labelleft=False)
        ax1.set_xlim(0, r_max)

        # --- 2. 3D ORBİTAL ---
        self.ax2 = self.fig.add_subplot(gs[1], projection='3d', facecolor='#000000')
        x = np.linspace(-dynamic_limit, dynamic_limit, res)
        X, Y, Z = np.meshgrid(x, x, x)
        r_3d = np.sqrt(X**2 + Y**2 + Z**2)
        r_safe = np.where(r_3d == 0, 1e-10, r_3d)
        theta = np.arccos(Z / r_safe)
        phi = np.arctan2(Y, X)

        psi = get_radial_wf(n, l, r_3d, z_eff) * sph_harm(m, l, phi, theta)
        prob = np.abs(psi)**2

        mask = prob > (prob.max() * 0.015)
        cmaps = ['Wistia', 'spring', 'winter', 'plasma', 'summer', 'autumn', 'cool']
        selected_cmap = cmaps[l % len(cmaps)]
        
        self.ax2.scatter(X[mask], Y[mask], Z[mask], 
                   c=prob[mask], cmap=selected_cmap, 
                   s=15 if n < 4 else 8, alpha=0.4, edgecolors='none', antialiased=False)

        self.ax2.set_axis_off()
        self.ax2.set_title(f"ORBITAL: {n}{'spdfgh'[l]} | Atom: {atom}", color='white', fontsize=15, y=0.92)
        self.ax2.view_init(elev=22, azim=45)
        
        self.fig.patch.set_facecolor('#000000')
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumApp(root)
    root.mainloop()