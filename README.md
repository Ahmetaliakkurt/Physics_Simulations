# Physics Simulations

Physics simulations and numerical projects from my physics education and personal research — from Python scripts to an interactive browser lab with 32 real-time simulations.

---

## Quick Start

### Interactive app (no installation)
The app runs entirely in the browser — no Python, no Node.js, no internet connection needed.

- **Windows:** double-click `05_physics_lab_app/run.bat`
- **Any platform:** open `05_physics_lab_app/index.html` in a modern browser (Chrome, Edge, Firefox, Safari)

```bash
git clone https://github.com/Ahmetaliakkurt/Physics_Simulations.git
cd Physics_Simulations/05_physics_lab_app
open index.html        # macOS   (Linux: xdg-open index.html)
```

### Python scripts
```bash
pip install -r requirements.txt
python 01_classical/kaotik_pendulum.py
```

---

## Repository Structure

```text
Physics_Simulations/
├── 01_classical/           Classical mechanics & optics scripts
├── 02_statistical/         Statistical physics models & distributions
│   └── ideal_gas_system/   Ideal gas with a C++ (pybind11 + OpenMP) engine
├── 03_quantum/             Schrödinger-equation solvers & atomic physics
├── 04_special_projects/    Algorithms & games (Sudoku solver, pursuit–evasion)
├── 05_physics_lab_app/     Interactive web app (32 simulations, zero dependencies)
├── requirements.txt        Python dependencies for folders 01–04
└── README.md
```

---

## 01_classical

### Mechanics
- `kaotik_pendulum.py` — chaotic double pendulum, sensitivity to initial conditions
- `E_particle_interaction.py` — electron in an electromagnetic pulse

### Optics
- `laser_focus.py` — thin/thick lens design and dispersion

### Mathematical methods
- `manifold.py` — surface and tangent plane
- `taylor_exps.py` — Taylor series approximations

---

## 02_statistical

### Distributions
- `binomial_dist.py` — random walk vs binomial distribution
- `central_limit_theorem.py` — central limit theorem
- `dist_functions.py` — Fermi–Dirac, Bose–Einstein, Maxwell–Boltzmann

### Many-body systems
- `kaotik_pendulum_statistics.py` — ensemble statistics of chaotic pendulums
- `central_force_field_atom.py` — hydrogen-like orbitals with effective nuclear charge
- `ideal_gas_system/` — ensemble vs time average of a 2D ideal gas

---

## 03_quantum

### Potential wells
- `adiabatic_inf_well_exp_comp.py`, `adiabatic_inf_well_expension.py` — expanding infinite well
- `phase_transition_map_quasistatic.py` — adiabatic vs sudden regime
- `cont_well_app.py`, `continuos_well.py` — finite wells, bound states

### Wave packets & scattering
- `gaussian_packet.py` — free Gaussian wave packet
- `TDSEvsBarrier.py` — tunnelling through a barrier
- `doublle slit.py` — 2D double-slit interference
- `lattice_wavepacket.py` — scattering from a lattice
- `quantum_billard.py` — wave packet in a square box
- `Superposition.py` — Fourier synthesis of a wave packet

### Atoms & spin
- `hydrogen_atom.py`, `H_atom_full_solution.py` — hydrogen orbitals and energy levels
- `stern_gerlach.py` — Stern–Gerlach experiment

---

## 04_special_projects

- `sudoku_solver_graph.py`, `sudoku_solver_graph_pygame.py` — Sudoku as graph colouring
- `f35_escape.py`, `track_destroy.py` — pursuit–evasion games (pygame)

---

## 05_physics_lab_app

### What it is
A browser app that brings the projects above together — 32 simulations computed live at 60 FPS, each with a write-up of the equations solved and the numerical method used.

### Contents
- **Classical Physics** — projectile motion with drag, double pendulum, Kepler orbits, coupled oscillators, electric & magnetic fields, charged particles in E/B fields, Faraday induction, RLC circuits, lens design, Taylor series
- **Statistical Physics** — random walk, central limit theorem, quantum statistics, ideal gas, atomic orbitals
- **Quantum Mechanics** — potential wells, wave packets, tunnelling, double slit, hydrogen atom, Stern–Gerlach
- **Special Projects** — Sudoku solver, missile evasion

### Structure
```text
05_physics_lab_app/
├── index.html        Entry point
├── run.bat           Windows launcher
└── assets/
    ├── css/          Dark theme
    ├── js/core/      Numerics (FFT, RK4, eigensolver) & rendering
    ├── js/sims/      One file per simulation
    └── vendor/katex/ Offline math rendering
```

---

## Requirements

### Web app
None — any modern browser.

### Python scripts
Python 3.10+ and the packages in `requirements.txt` (NumPy, SciPy, Matplotlib, Plotly, SymPy, pygame-ce, OpenCV).

### C++ ideal-gas engine (optional)
`02_statistical/ideal_gas_system/main.py` needs the compiled `fast_ensemble` module. The included build is for Linux / Python 3.10; to rebuild it you need CMake, a C++17 compiler with OpenMP and `pybind11`:

```bash
cd 02_statistical/ideal_gas_system
pip install pybind11
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build
```
The other two ideal-gas scripts are pure Python and need no compilation.
