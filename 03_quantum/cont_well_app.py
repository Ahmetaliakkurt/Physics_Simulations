import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.linalg import eigh_tridiagonal

class QuantumWellApp:
    def __init__(self, root):
        self.root = root
        self.root.title("1D Quantum Well Solver (Natural Units)")
        self.root.geometry("1100x820")

        input_frame = ttk.LabelFrame(root, text="Parameters", padding=(10, 10))
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.var_pot_type = tk.StringVar(value="smooth")
        self.var_v0 = tk.DoubleVar(value=10.0)
        self.var_width = tk.DoubleVar(value=2.0)
        self.var_L = tk.DoubleVar(value=6.0)
        self.var_mass = tk.DoubleVar(value=1.0)
        self.var_power = tk.DoubleVar(value=8.0)
        self.var_barrier = tk.DoubleVar(value=0.5) 
        self.var_n_states = tk.IntVar(value=4)

        self.var_show_energy = tk.BooleanVar(value=True) 
        self.var_show_wave = tk.BooleanVar(value=False)
        self.var_show_prob = tk.BooleanVar(value=False)

        self.cached_data = None

        type_frame = ttk.LabelFrame(input_frame, text="Well Type Selection", padding=(5, 5))
        type_frame.pack(fill=tk.X, pady=10)

        rb1 = ttk.Radiobutton(type_frame, text="Smooth", variable=self.var_pot_type,
                              value="smooth", command=self.toggle_inputs)
        rb1.pack(anchor=tk.W, padx=5)

        rb2 = ttk.Radiobutton(type_frame, text="Square", variable=self.var_pot_type,
                              value="square", command=self.toggle_inputs)
        rb2.pack(anchor=tk.W, padx=5)

        rb3 = ttk.Radiobutton(type_frame, text="Double Well (Tunneling)", variable=self.var_pot_type,
                              value="double_well", command=self.toggle_inputs)
        rb3.pack(anchor=tk.W, padx=5)

        # 4. RADYO BUTONU: TRIPLE WELL
        rb4 = ttk.Radiobutton(type_frame, text="Triple Well (Splitting)", variable=self.var_pot_type,
                              value="triple_well", command=self.toggle_inputs)
        rb4.pack(anchor=tk.W, padx=5)

        self.create_input(input_frame, "Potential Depth V0:", self.var_v0)
        self.create_input(input_frame, "Width of EACH Well:", self.var_width)
        self.create_input(input_frame, "Calculation Domain ±L:", self.var_L)
        self.create_input(input_frame, "Mass (m):", self.var_mass)

        self.power_frame, self.power_entry = self.create_input(input_frame, "Softness Exponent (Power):", self.var_power, return_widgets=True)
        self.barrier_frame, self.barrier_entry = self.create_input(input_frame, "Barrier Width Between Wells:", self.var_barrier, return_widgets=True)

        self.create_input(input_frame, "Number of States:", self.var_n_states)

        check_frame = ttk.LabelFrame(input_frame, text="Plot Options", padding=(5, 5))
        check_frame.pack(fill=tk.X, pady=10)

        ttk.Checkbutton(check_frame, text="Energy Levels (E)", variable=self.var_show_energy,
                        command=self.draw_plot).pack(anchor=tk.W)
        ttk.Checkbutton(check_frame, text="Wavefunctions (ψ)", variable=self.var_show_wave,
                        command=self.draw_plot).pack(anchor=tk.W)
        ttk.Checkbutton(check_frame, text="Probabilities (|ψ|²)", variable=self.var_show_prob,
                        command=self.draw_plot).pack(anchor=tk.W)

        solve_btn = ttk.Button(input_frame, text="SOLVE AND PLOT", command=self.solve_and_plot)
        solve_btn.pack(pady=20, fill=tk.X)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toggle_inputs()

        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Select Well type, enter parameters and press 'SOLVE AND PLOT'",
                     horizontalalignment='center', verticalalignment='center',
                     transform=self.ax.transAxes, fontsize=12, color='gray')
        self.canvas.draw()

    def create_input(self, parent, label_text, variable, return_widgets=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        lbl = ttk.Label(frame, text=label_text)
        lbl.pack(anchor=tk.W)
        entry = ttk.Entry(frame, textvariable=variable)
        entry.pack(fill=tk.X)
        if return_widgets:
            return frame, entry
        return frame

    def toggle_inputs(self):
        selection = self.var_pot_type.get()
        if selection == "smooth":
            self.power_entry.config(state='normal')
            self.barrier_entry.config(state='disabled')
        elif selection == "square":
            self.power_entry.config(state='disabled')
            self.barrier_entry.config(state='disabled')
        elif selection == "double_well":
            self.power_entry.config(state='disabled')
            self.barrier_entry.config(state='normal')
        elif selection == "triple_well":
            self.power_entry.config(state='disabled')
            self.barrier_entry.config(state='normal')

    def solve_schrodinger(self, V_array, x_array, dx, mass):
        N = len(V_array)
        coeff = -1.0 / (2.0 * mass * dx**2)
        kinetic_diag = -2 * coeff
        kinetic_off_diag = coeff

        main_diag = V_array + kinetic_diag
        off_diag = kinetic_off_diag * np.ones(N - 1)

        energies, wavefunctions = eigh_tridiagonal(main_diag, off_diag)
        return energies, wavefunctions

    def solve_and_plot(self):
        try:
            pot_type = self.var_pot_type.get()
            V0 = self.var_v0.get()
            width = self.var_width.get()
            L = self.var_L.get()
            mass = self.var_mass.get()
            num_states = self.var_n_states.get()

            N = 1200  # Çözünürlüğü 3 kuyu için biraz artırdık
            x = np.linspace(-L, L, N)
            dx = x[1] - x[0]

            if pot_type == "smooth":
                w_param = width / 2.0
                power = self.var_power.get()
                V = -V0 * np.exp(-(x / w_param)**power)
                title = "SMOOTH WELL"
            elif pot_type == "square":
                w_param = width / 2.0
                V = np.zeros_like(x)
                mask = (x > -w_param) & (x < w_param)
                V[mask] = -V0
                title = "FINITE SQUARE WELL"
            elif pot_type == "double_well":
                barrier_w = self.var_barrier.get()
                V = np.zeros_like(x)
                left_mask = (x > -barrier_w/2.0 - width) & (x < -barrier_w/2.0)
                right_mask = (x > barrier_w/2.0) & (x < barrier_w/2.0 + width)
                V[left_mask] = -V0
                V[right_mask] = -V0
                title = "SYMMETRIC DOUBLE QUANTUM WELL"
            elif pot_type == "triple_well":
                barrier_w = self.var_barrier.get()
                V = np.zeros_like(x)
                
                # Merkez Kuyu
                center_mask = (x > -width/2.0) & (x < width/2.0)
                # Sağ Kuyu
                right_start = width/2.0 + barrier_w
                right_mask = (x > right_start) & (x < right_start + width)
                # Sol Kuyu
                left_end = -width/2.0 - barrier_w
                left_mask = (x > left_end - width) & (x < left_end)
                
                V[center_mask] = -V0
                V[right_mask] = -V0
                V[left_mask] = -V0
                title = "SYMMETRIC TRIPLE QUANTUM WELL"

            E, psi = self.solve_schrodinger(V, x, dx, mass)

            self.cached_data = {
                'x': x,
                'V': V,
                'E': E,
                'psi': psi,
                'V0': V0,
                'title': title,
                'dx': dx,
                'L': L,
                'num_states': num_states
            }

            self.draw_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Calculation Error:\n{str(e)}")

    def draw_plot(self):
        if self.cached_data is None:
            return  

        x = self.cached_data['x']
        V = self.cached_data['V']
        E_vals = self.cached_data['E']
        psi = self.cached_data['psi']
        V0 = self.cached_data['V0']
        title = self.cached_data['title']
        dx = self.cached_data['dx']
        L = self.cached_data['L']
        num_states = self.cached_data['num_states']

        show_energy = self.var_show_energy.get()
        show_wave = self.var_show_wave.get()
        show_prob = self.var_show_prob.get()

        self.ax.clear()
        self.ax.axis('on')

        y_min = -V0 * 1.5

        self.ax.plot(x, V, color='black', linewidth=2, alpha=0.5, label="Potential V(x)")
        self.ax.fill_between(x, V, y_min, color='gray', alpha=0.2)

        scale = max(V0 * 0.15, 0.5)
        highest_plotted_energy = -V0
        limit_states = min(num_states, len(E_vals))

        for i in range(limit_states):
            E = E_vals[i]
            psi_state = psi[:, i]
            highest_plotted_energy = max(highest_plotted_energy, E)

            norm = np.sqrt(np.sum(np.abs(psi_state)**2) * dx)
            psi_norm = psi_state / norm
            
            max_amp = np.max(np.abs(psi_norm))
            if max_amp > 0:
                visual_scale_factor = (scale * 0.6) / max_amp
            else:
                visual_scale_factor = 1.0

            psi_visual = psi_norm * visual_scale_factor
            prob_visual = (np.abs(psi_norm)**2 / (max_amp**2)) * (scale * 0.6)

            color = f'C{i}'
            
            if show_energy:
                self.ax.hlines(E, -L, L, color=color, linestyle='-', linewidth=2.0, alpha=0.8, label=f"State {i} (E={E:.2f})" if not (show_wave or show_prob) else "")

            if show_wave:
                self.ax.plot(x, psi_visual + E, '--', color=color, linewidth=1.5, label=f"State {i} (ψ)")

            if show_prob:
                self.ax.fill_between(x, E, prob_visual + E, color=color, alpha=0.3, label=f"State {i} (|ψ|²)" if not show_wave else "")

        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Position (x)", fontsize=12)
        self.ax.set_ylabel("Energy (E)", fontsize=12)

        wave_peak_clearance = scale * 1.5 if (show_wave or show_prob) else V0 * 0.2
        required_top = highest_plotted_energy + wave_peak_clearance
        minimal_top = V0 * 0.1
        y_max = max(required_top, minimal_top)

        self.ax.set_ylim(y_min, y_max)
        self.ax.set_xlim(-L, L)
        
        if show_energy or show_wave or show_prob:
            self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumWellApp(root)
    root.mainloop()