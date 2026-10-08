# Physics Simulation Lab

33 interactive simulations in classical, statistical and quantum physics, computed live in the browser at 60 fps.

## How to open
Double-click `index.html` (or `run.bat`). No Python, no installation and no internet connection are needed.
A recent browser (Chrome, Edge or Firefox) is recommended.

## Contents
- **Classical Physics**
  - Mechanics: projectile motion with air resistance, chaotic double pendulum (trajectories + ensemble statistics), Kepler orbits, coupled oscillators.
  - Electromagnetism: electric fields of point charges, magnetic fields (Biot–Savart), charged particles in E/B fields, an electron in a magnetic field with a field-free window, Faraday induction, RLC circuits, electron in a laser pulse.
  - Optics: lens design.
  - Mathematical methods: tangent planes, Taylor series.
- **Statistical Physics:** random walk, central limit theorem, quantum statistics, ideal gas (ensemble vs time average), atomic orbitals.
- **Quantum Mechanics:** adiabatic well, adiabatic–sudden map, finite wells, Gaussian packet, hydrogen levels and orbitals, lattice scattering, square box, Stern–Gerlach, superposition, tunnelling, double slit.
- **Special Projects:** Sudoku solver (graph colouring), missile evasion.

Every page ends with a detailed section, "The Physics — what this simulation solves", covering:
- the physical system;
- the equations being solved;
- the numerical method;
- experiments to try;
- limitations.

## Structure
```
index.html                 loads everything (classic scripts, works from file://)
run.bat                    opens index.html in the default browser
assets/css/style.css       dark theme
assets/js/core/math.js     numerics: FFT, RK4, tridiagonal eigensolver, special functions, RNG
assets/js/core/plot.js     Canvas 2D plotting + 3D view
assets/js/core/app.js      menu, routing, control panel, 60 fps loop
assets/js/sims/*.js        one file per simulation (App.register)
assets/vendor/katex/       offline formula rendering
```

## Adding a simulation
1. Create a new file in `assets/js/sims/` that calls `App.register({...})`. See `gaussian-wavepacket.js` for a compact example.
2. Add a `<script>` line for it in `index.html`.
