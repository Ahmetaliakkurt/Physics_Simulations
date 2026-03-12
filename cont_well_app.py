import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.linalg import eigh_tridiagonal

HBAR = 1.0545718e-34
M_E = 9.10938356e-31
EV_TO_J = 1.60217663e-19
NM_TO_M = 1e-9

class QuantumWellApp:
    def __init__(self, root):
        self.root = root
        self.root.title("1D Quantum Well Solver ")
        self.root.geometry("1100x750")

        input_frame = ttk.LabelFrame(root, text="Parameters", padding=(10, 10))
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.var_pot_type = tk.StringVar(value="smooth")
        self.var_v0 = tk.DoubleVar(value=10.0)
        self.var_width = tk.DoubleVar(value=2.0)
        self.var_L = tk.DoubleVar(value=4.0)
        self.var_mass = tk.DoubleVar(value=1.0)
        self.var_power = tk.DoubleVar(value=8.0)
        self.var_n_states = tk.IntVar(value=3)

        type_frame = ttk.LabelFrame(input_frame, text="Well Type Selection", padding=(5, 5))
        type_frame.pack(fill=tk.X, pady=10)

        rb1 = ttk.Radiobutton(type_frame, text="Smooth", variable=self.var_pot_type,
                              value="smooth", command=self.toggle_power_entry)
        rb1.pack(anchor=tk.W, padx=5)

        rb2 = ttk.Radiobutton(type_frame, text="Square", variable=self.var_pot_type,
                              value="square", command=self.toggle_power_entry)
        rb2.pack(anchor=tk.W, padx=5)

        self.create_input(input_frame, "Potential Depth V0 (eV):", self.var_v0)
        self.create_input(input_frame, "Well Width (nm):", self.var_width)
        self.create_input(input_frame, "Calculation Domain ±L (nm):", self.var_L)
        self.create_input(input_frame, "Mass (x Electron Mass):", self.var_mass)

        self.power_frame, self.power_entry = self.create_input(input_frame, "Softness Exponent (Power):", self.var_power, return_widgets=True)

        self.create_input(input_frame, "Number of States:", self.var_n_states)

        solve_btn = ttk.Button(input_frame, text="SOLVE AND PLOT", command=self.solve_and_plot)
        solve_btn.pack(pady=20, fill=tk.X)

        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toggle_power_entry()

        self.ax.axis('off')
        self.ax.text(0.5, 0.5, "Enter parameters and press 'SOLVE AND PLOT'",
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

    def toggle_power_entry(self):
        selection = self.var_pot_type.get()
        if selection == "square":
            self.power_entry.config(state='disabled')
        else:
            self.power_entry.config(state='normal')

    def solve_schrodinger(self, V_array, x_array, dx, mass_kg):
        N = len(V_array)
        coeff = - (HBAR**2) / (2 * mass_kg * dx**2)
        kinetic_diag = -2 * coeff
        kinetic_off_diag = coeff

        main_diag = V_array + kinetic_diag
        off_diag = kinetic_off_diag * np.ones(N - 1)

        energies_J, wavefunctions = eigh_tridiagonal(main_diag, off_diag)
        return energies_J, wavefunctions

    def solve_and_plot(self):
        try:
            pot_type = self.var_pot_type.get()
            V0_eV = self.var_v0.get()
            width_nm = self.var_width.get()
            L_nm = self.var_L.get()
            mass_kg = self.var_mass.get() * M_E
            num_states = self.var_n_states.get()

            N = 1000
            x_nm = np.linspace(-L_nm, L_nm, N)
            x_m = x_nm * NM_TO_M
            dx_m = x_m[1] - x_m[0]
            w_param = width_nm / 2.0

            if pot_type == "smooth":
                power = self.var_power.get()
                V_eV = -V0_eV * np.exp(-(x_nm / w_param)**power)
                title = "SMOOTH WELL"
            else:
                V_eV = np.zeros_like(x_nm)
                mask = (x_nm > -w_param) & (x_nm < w_param)
                V_eV[mask] = -V0_eV
                title = "FINITE SQUARE WELL"

            V_J = V_eV * EV_TO_J

            E_J, psi = self.solve_schrodinger(V_J, x_m, dx_m, mass_kg)
            E_eV = E_J / EV_TO_J

            self.ax.clear()
            self.ax.axis('on')

            y_min = -V0_eV * 2.0

            self.ax.plot(x_nm, V_eV, color='black', linewidth=2, alpha=0.5, label="Potential V(x)")

            self.ax.fill_between(x_nm, V_eV, y_min, color='gray', alpha=0.2)

            scale = max(V0_eV * 0.15, 0.5)
            highest_plotted_energy = -V0_eV
            limit_states = min(num_states, len(E_eV))

            for i in range(limit_states):
                E = E_eV[i]
                psi_state = psi[:, i]
                highest_plotted_energy = E

                norm = np.sqrt(np.sum(np.abs(psi_state)**2) * dx_m)
                psi_norm = psi_state / norm
                psi_visual = psi_norm * (scale * np.sqrt(NM_TO_M)) * 2
                prob_visual = np.abs(psi_visual)**2 / scale

                self.ax.plot(x_nm, scale * psi_visual + E, '--', color=f'C{i}', linewidth=1.5, label=f"State {i}")
                self.ax.fill_between(x_nm, E, scale * prob_visual + E, color=f'C{i}', alpha=0.3)
                self.ax.hlines(E, -L_nm, L_nm, color=f'C{i}', linestyle='-', linewidth=0.5, alpha=0.5)

            self.ax.set_title(title, fontsize=14, fontweight='bold')
            self.ax.set_xlabel("Position (nm)", fontsize=12)
            self.ax.set_ylabel("Energy (eV)", fontsize=12)

            wave_peak_clearance = scale * 1.5
            required_top = highest_plotted_energy + wave_peak_clearance
            minimal_top = V0_eV * 0.1
            y_max = max(required_top, minimal_top)

            self.ax.set_ylim(y_min, y_max)
            self.ax.set_xlim(-L_nm, L_nm)
            self.ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Calculation Error:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumWellApp(root)
    root.mainloop()