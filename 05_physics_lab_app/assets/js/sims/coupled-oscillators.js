/* Coupled oscillators — a chain of N identical masses joined by springs, with fixed or free ends.
 * The normal modes are found exactly from the stiffness matrix with a cyclic Jacobi eigenvalue routine,
 * and the motion is evaluated in closed form as a superposition of (optionally damped) normal modes,
 * so there is no time-stepping error at all. */
(function () {
  "use strict";

  const NMAX = 12;
  const SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
  const sub = (n) => String(n).split("").map((c) => SUB[+c]).join("");
  const INT = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : ""); // integer-only tick labels
  const NTR = 400; // samples per trace in the x_i(t) window

  /**
   * Cyclic Jacobi eigenvalue routine for a symmetric n×n matrix (row-major Float64Array, destroyed).
   * Returns eigenvalues (ascending) and orthonormal eigenvectors vec[k] (Float64Array of length n).
   */
  function jacobiEigen(A, n) {
    const V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;
    let norm = 0;
    for (let i = 0; i < n * n; i++) norm += A[i] * A[i];
    norm = Math.sqrt(norm) || 1;
    let sweeps = 0;
    for (; sweeps < 60; sweeps++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (Math.sqrt(2 * off) < 1e-15 * norm) break;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const app = A[p * n + p], aqq = A[q * n + q];
          const theta = (aqq - app) / (2 * apq);
          const tt = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
          for (let k = 0; k < n; k++) { // columns p, q
            const akp = A[k * n + p], akq = A[k * n + q];
            A[k * n + p] = c * akp - s * akq; A[k * n + q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) { // rows p, q
            const apk = A[p * n + k], aqk = A[q * n + k];
            A[p * n + k] = c * apk - s * aqk; A[q * n + k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k * n + p], vkq = V[k * n + q];
            V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[a * n + a] - A[b * n + b]);
    const values = new Float64Array(n), vectors = [];
    idx.forEach((j, k) => {
      values[k] = A[j * n + j];
      const v = new Float64Array(n);
      for (let i = 0; i < n; i++) v[i] = V[i * n + j];
      // sign convention: the first clearly non-zero component is positive
      for (let i = 0; i < n; i++) if (Math.abs(v[i]) > 1e-8) { if (v[i] < 0) for (let m = 0; m < n; m++) v[m] = -v[m]; break; }
      vectors.push(v);
    });
    return { values, vectors, sweeps };
  }

  App.register({
    id: "coupled-oscillators",
    category: "classical",
    group: "Mechanics",
    order: 13,
    title: "Coupled Oscillators & Normal Modes",
    icon: "〰️",
    subtitle: "A chain of 1–12 masses connected by springs between two walls: the motion is decomposed exactly into normal modes, " +
      "and you can watch energy slosh between the masses (beats) while every normal mode keeps its own energy.",
    notes: [
      { type: "info", html: "The chain at the top moves along its length (longitudinal oscillation); underneath, the same displacements are " +
        "drawn sideways as a <b>displacement profile</b> so the mode shapes are easy to see. With the default <b>N = 2</b> and weak coupling, " +
        "the energy given to mass 1 migrates to mass 2 and back — the beat period is $2\\pi/(\\omega_2-\\omega_1)$. " +
        "Press <b>Uniform chain</b> and raise N to compare the exact frequencies with the dispersion relation $\\omega_n=2\\sqrt{k/m}\\,|\\sin(n\\pi/2(N+1))|$." },
    ],
    animated: true,
    speed: { min: 0.1, max: 5, value: 1, step: 0.1 },
    controls: [
      { id: "N", type: "slider", label: "Number of masses $N$", min: 1, max: NMAX, step: 1, value: 2 },
      { id: "leftEnd", type: "select", label: "Left end", value: "fixed", options: [
        { value: "fixed", label: "Fixed — spring k to the wall" }, { value: "free", label: "Free — no wall spring" }] },
      { id: "rightEnd", type: "select", label: "Right end", value: "fixed", options: [
        { value: "fixed", label: "Fixed — spring k to the wall" }, { value: "free", label: "Free — no wall spring" }] },
      { type: "section", label: "Masses and springs" },
      { id: "m", type: "slider", label: "Mass $m$ (each)", min: 0.1, max: 5, step: 0.05, value: 1, unit: "kg" },
      { id: "k", type: "slider", label: "Wall springs $k$", min: 0.5, max: 50, step: 0.5, value: 10, unit: "N/m",
        help: "Springs that connect the end masses to the walls (only at fixed ends)." },
      { id: "kc", type: "slider", label: "Coupling springs $k_c$", min: 0, max: 50, step: 0.1, value: 1, unit: "N/m",
        help: "Springs between neighbouring masses. Weak coupling ($k_c \\ll k$) gives slow, complete beats for N = 2." },
      { id: "uniform", type: "button", label: "Uniform chain (set k_c = k)" },
      { id: "gamma", type: "slider", label: "Damping rate $\\gamma$", min: 0, max: 2, step: 0.01, value: 0, unit: "1/s",
        help: "Friction force $-\\gamma m\\dot x_i$ on every mass; each mode's amplitude decays as $e^{-\\gamma t/2}$." },
      { type: "section", label: "Initial condition (masses start at rest)" },
      { id: "init", type: "select", label: "Initial displacement", value: "mass1", options: [
        { value: "mass1", label: "Displace mass 1 only" },
        { value: "mode", label: "Pure normal mode n" },
        { value: "random", label: "Random displacements" },
        { value: "pluck", label: "Pluck (triangular profile)" },
      ] },
      { id: "modeN", type: "slider", label: "Mode number $n$", min: 1, max: 2, step: 1, value: 1,
        visibleIf: (p) => p.init === "mode" && p.N > 1, help: "Modes are numbered by increasing frequency." },
      { id: "pluckAt", type: "slider", label: "Pluck position (mass)", min: 1, max: 2, step: 1, value: 1,
        visibleIf: (p) => p.init === "pluck" && p.N > 1 },
      { id: "reroll", type: "button", label: "New random displacements", visibleIf: (p) => p.init === "random" },
      { id: "A", type: "slider", label: "Largest initial displacement $A$", min: 0.05, max: 0.45, step: 0.01, value: 0.3, unit: "m",
        help: "Equilibrium spacing of the masses is 1 m." },
      { type: "section", label: "Display" },
      { id: "win", type: "slider", label: "Trace window", min: 5, max: 120, step: 1, value: 30, unit: "s", live: true },
      { id: "profile", type: "checkbox", label: "Displacement profile under the chain", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>$N$ identical point masses $m$ (kg), $1\\le N\\le 12$, slide without friction along a straight line. Their equilibrium
      positions are 1 m apart and $x_i(t)$ (m) is the displacement of mass $i$ from equilibrium. Neighbouring masses are joined by
      ideal massless <b>coupling springs</b> of stiffness $k_c$ (N/m); at a <b>fixed end</b> the outermost mass is tied to a rigid wall by a
      spring of stiffness $k$ (N/m), at a <b>free end</b> it has no wall spring. An optional viscous friction force $-\\gamma m\\dot x_i$
      ($\\gamma$ in s⁻¹) acts on every mass. Springs are linear (Hooke's law) at all displacements, and the masses start at rest from a
      chosen displacement pattern. For $N=2$ with both ends fixed this is the textbook pair of pendula/carts coupled by a weak spring;
      for $k_c=k$ and large $N$ it is a discrete model of a string or of a 1-D crystal lattice.</p>

      <h4>Equations being solved</h4>
      <p>The Lagrangian $\\mathcal L=\\tfrac12 m\\sum_i\\dot x_i^2-V$ with
      $V=\\tfrac12 k\\,x_1^2\\,[\\text{left fixed}]+\\tfrac12k_c\\sum_{i=1}^{N-1}(x_{i+1}-x_i)^2+\\tfrac12k\\,x_N^2\\,[\\text{right fixed}]$
      gives $N$ coupled linear equations, written compactly with the symmetric tridiagonal <b>stiffness matrix</b> $\\mathsf K$:</p>
      $$m\\,\\ddot x_i=-k_c(x_i-x_{i-1})-k_c(x_i-x_{i+1})-\\gamma m\\,\\dot x_i\\quad\\Longleftrightarrow\\quad
        m\\,\\ddot{\\mathbf x}+\\gamma m\\,\\dot{\\mathbf x}+\\mathsf K\\,\\mathbf x=0,$$
      <p>where a wall spring replaces the missing neighbour at a fixed end ($K_{11}$ and $K_{NN}$ contain $k$ instead of one $k_c$),
      $K_{i,i\\pm1}=-k_c$. Seeking solutions $\\mathbf x=\\mathbf v\\,e^{i\\omega t}$ of the undamped problem gives the eigenvalue problem</p>
      <div class="callout">$$\\mathsf K\\,\\mathbf v_n=m\\,\\omega_n^2\\,\\mathbf v_n,\\qquad
        \\mathbf x(t)=\\sum_{n=1}^{N}\\mathbf v_n\\,q_n(t),\\qquad \\ddot q_n+\\gamma\\dot q_n+\\omega_n^2q_n=0 .$$</div>
      <p>Because $\\mathsf K$ is symmetric, the eigenvectors (mode shapes) are orthonormal, $\\mathbf v_n\\cdot\\mathbf v_{n'}=\\delta_{nn'}$, and
      the normal coordinates $q_n=\\mathbf v_n\\cdot\\mathbf x$ decouple into independent damped oscillators. For a <b>uniform</b> chain
      ($k=k_c$) the mode shapes are sine waves, $v_{n,i}\\propto\\sin\\big(n\\pi i/(N+1)\\big)$, and the frequencies follow the
      <b>dispersion relation</b></p>
      $$\\begin{aligned}
        \\omega_n&=2\\sqrt{k/m}\\,\\Big|\\sin\\frac{n\\pi}{2(N+1)}\\Big| &&\\text{(both ends fixed)},\\\\
        \\omega_n&=2\\sqrt{k/m}\\,\\sin\\frac{(n-1)\\pi}{2N} &&\\text{(both ends free; } n=1 \\text{ is the zero mode)},\\\\
        \\omega_n&=2\\sqrt{k/m}\\,\\sin\\frac{(2n-1)\\pi}{2(2N+1)} &&\\text{(one fixed and one free end)}, \\qquad n=1,\\dots,N.
      \\end{aligned}$$
      <p>For $N=2$ with fixed ends and any $k,k_c$: $\\omega_1=\\sqrt{k/m}$ (in phase, coupling spring unstretched) and
      $\\omega_2=\\sqrt{(k+2k_c)/m}$ (out of phase). Starting with only mass 1 displaced, $x_{1,2}=\\tfrac A2(\\cos\\omega_1t\\pm\\cos\\omega_2t)$:
      a fast oscillation at $\\bar\\omega=(\\omega_1+\\omega_2)/2$ inside an envelope that transfers all the energy to mass 2 after half a
      <b>beat period</b> $T_b=2\\pi/(\\omega_2-\\omega_1)$. The energy $E=\\tfrac12m|\\dot{\\mathbf x}|^2+\\tfrac12\\mathbf x^{\\mathsf T}\\mathsf K\\mathbf x
      =\\sum_n\\tfrac12m(\\dot q_n^2+\\omega_n^2q_n^2)$ splits into independent mode energies, each conserved when $\\gamma=0$.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Normal modes.</b> On every restart the $N\\times N$ matrix $\\mathsf K/m$ is built and diagonalised with the cyclic
        <b>Jacobi method</b>: plane rotations with $\\tan2\\theta=2A_{pq}/(A_{qq}-A_{pp})$ annihilate each off-diagonal element in turn,
        sweeping until the off-diagonal norm is below $10^{-15}$ of the matrix norm (typically 5–8 sweeps). The accumulated rotations are
        the orthonormal eigenvectors; eigenvalues $\\omega_n^2$ are sorted in ascending order (round-off negatives are clipped to 0).</li>
        <li><b>Exact time evolution — no integrator.</b> The initial normal coordinates are $q_n(0)=\\mathbf v_n\\cdot\\mathbf x(0)$,
        $\\dot q_n(0)=0$. Each one is advanced in closed form, $q_n(t)=e^{-\\gamma t/2}\\big[q_n(0)\\,C(t)+\\tfrac{\\gamma}{2}q_n(0)\\,S(t)\\big]$
        with $\\Omega_n^2=\\omega_n^2-\\gamma^2/4$ and $C=\\cos\\Omega_nt$, $S=\\sin(\\Omega_nt)/\\Omega_n$ (underdamped),
        $C=\\cosh\\kappa t$, $S=\\sinh(\\kappa t)/\\kappa$ with $\\kappa=\\sqrt{-\\Omega_n^2}$ (overdamped; evaluated as separate
        exponentials so nothing overflows), or $C=1$, $S=t$ (critical / zero mode). The displayed time only advances a clock; positions and
        velocities are recomputed from these formulas every frame, so there is no step size and no accumulated error.</li>
        <li><b>Plots.</b> The traces $x_i(t)$ in the scrolling window are evaluated from the same closed form at 400 instants (each trace
        offset vertically). The energy of mass $i$ is $\\tfrac12m\\dot x_i^2$ plus half of the energy of each coupling spring it touches and
        the whole energy of its wall spring, so the bars add up to $E$. The mode panel shows the envelope
        $|a_n|=\\sqrt{q_n^2+\\dot q_n^2/\\omega_n^2}$ (faint) and the instantaneous $q_n(t)$ (solid). The dispersion panel compares the
        Jacobi eigenfrequencies (dots) with the analytic formula above (curve) when $k=k_c$.</li>
        <li><b>Accuracy checks.</b> The metric "energy check" compares $E$ computed from positions and velocities with
        $\\sum_n\\tfrac12m(\\dot q_n^2+\\omega_n^2q_n^2)$ — they agree only if the eigenvectors are orthonormal and complete; the
        "eigen residual" is $\\max_n\\lVert\\mathsf K\\mathbf v_n-m\\omega_n^2\\mathbf v_n\\rVert$. Both should sit at the $10^{-15}$
        round-off level. With $\\gamma=0$, $E(t)/E_0$ stays at 100 % by construction.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Beats.</b> Defaults ($N=2$, $m=1$ kg, $k=10$, $k_c=1$ N/m, mass 1 displaced): $\\omega_1=3.162$, $\\omega_2=3.464$ rad/s,
        beat period $T_b=2\\pi/0.302\\approx20.8$ s. All energy moves to mass 2 after 10.4 s and back; the mode bars do not change.
        Raise $k_c$ to 5: the beats become faster ($T_b=2\\pi/(\\sqrt{20}-\\sqrt{10})\\approx4.8$ s).</li>
        <li><b>Pure modes.</b> Choose "Pure normal mode n": every mass oscillates at the single frequency $\\omega_n$ with a fixed shape,
        and only one mode bar is non-zero. For $N=2$, mode 1 is in phase, mode 2 out of phase.</li>
        <li><b>The dispersion relation.</b> Press <em>Uniform chain</em>, set $N=12$: the 12 dots lie exactly on
        $\\omega_n=2\\sqrt{k/m}\\sin(n\\pi/26)$; the highest mode ($\\omega\\to2\\sqrt{k/m}$) has neighbouring masses moving almost opposite.</li>
        <li><b>Zero mode.</b> Make both ends free: the lowest frequency becomes $\\omega_1=0$ — a rigid translation of the whole chain
        that costs no spring energy.</li>
        <li><b>Pluck and damping.</b> Pluck a 12-mass chain: many modes are excited, and the profile travels as a pulse that reflects at the
        walls (inverted at a fixed end, upright at a free end). Add $\\gamma=0.2$ s⁻¹: $E(t)$ decays roughly as $e^{-\\gamma t}$.</li>
        <li><b>Overdamping.</b> With $\\gamma\\gt 2\\omega_1$ the slow modes creep back without oscillating, while the fast modes still ring.</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>Linear springs and small, purely longitudinal motion; identical masses; damping proportional to mass so that the modes stay
      decoupled (general damping would mix them). Real lattices add anharmonicity and quantisation (phonons). See J. R. Taylor,
      <em>Classical Mechanics</em>, ch. 11; H. Goldstein, <em>Classical Mechanics</em>, ch. 6; A. P. French, <em>Vibrations and Waves</em>,
      ch. 5–6; Kittel, <em>Introduction to Solid State Physics</em>, ch. 4 (lattice dispersion).</p>`,

    mount(api) {
      const P = api.params;
      const plots = api.plots([
        { id: "chain", title: "Mass–spring chain (top) and displacement profile $x_i$ drawn sideways (bottom)", span: 2, aspect: 0.3, minHeight: 220,
          equal: true, axes: false, xlim: [-0.3, 3.3], ylim: [-1.6, 0.62] },
        { id: "tr", title: "Displacements $x_i(t)$ — one trace per mass, offset vertically", span: 2, aspect: 0.4, xlabel: "time t (s)", ylabel: "xᵢ (traces offset)",
          ytickFormat: () => "", margin: { l: 34 } },
        { id: "en", xtickFormat: INT, title: "Energy of each mass (kinetic + share of its springs), % of $E_0$", aspect: 0.62, xlabel: "mass i", ylabel: "energy (% of E₀)" },
        { id: "mo", xtickFormat: INT, title: "Normal-mode amplitudes: envelope $|a_n|$ (faint) and $q_n(t)$ (solid)", aspect: 0.62, xlabel: "mode n", ylabel: "amplitude (m)" },
        { id: "disp", xtickFormat: INT, title: "Dispersion: exact $\\omega_n$ (dots) vs analytic uniform-chain formula", aspect: 0.62, xlabel: "mode number n", ylabel: "ω (rad/s)" },
        { id: "shape", xtickFormat: INT, ytickFormat: INT, title: "Mode shapes $\\mathbf v_n$ (offset by n, frequency on the right)", aspect: 0.62, xlabel: "mass i", ylabel: "mode n" },
      ]);
      const M = api.metrics([
        { id: "w", label: "Lowest / highest $\\omega$ (rad/s)" },
        { id: "tb", label: "Beat period $2\\pi/(\\omega_2-\\omega_1)$" },
        { id: "E", label: "Energy $E(t)/E_0$" },
        { id: "chk", label: "Energy check: masses vs modes" },
        { id: "res", label: "Eigen residual (Jacobi sweeps)" },
      ]);

      // ---------------------------------------------------------------- state (allocated once for NMAX)
      let N = 2, m = 1, k = 10, kc = 1, gam = 0, leftFixed = true, rightFixed = true, t = 0, E0 = 0, seed = 1;
      let w2 = new Float64Array(NMAX), om = new Float64Array(NMAX), vec = [], q0 = new Float64Array(NMAX), env0 = new Float64Array(NMAX);
      const x = new Float64Array(NMAX), v = new Float64Array(NMAX), q = new Float64Array(NMAX), qd = new Float64Array(NMAX);
      const Emass = new Float64Array(NMAX), Emode = new Float64Array(NMAX), env = new Float64Array(NMAX);
      const IDX = new Float64Array(NMAX), TT = new Float64Array(NTR), TX = [];
      for (let i = 0; i < NMAX; i++) TX.push(new Float64Array(NTR));
      const ZX = new Float64Array(64), ZY = new Float64Array(64);
      const PX = new Float64Array(NMAX + 2), PY = new Float64Array(NMAX + 2);
      const AX = new Float64Array(200), AY = new Float64Array(200);
      const MSX = new Float64Array(NMAX + 2), MSY = new Float64Array(NMAX + 2);
      const shScale = new Float64Array(NMAX), PCT = new Float64Array(NMAX), COLS = [];
      for (let i = 0; i < NMAX; i++) COLS.push(PlotCycle[i % PlotCycle.length]);
      let resid = 0, sweeps = 0, eMaxSeen = 1, xBound = 0.3, ampMax = 0.3, analytic = null, analyticName = "";

      /** Normal coordinate q(t) and its derivative for one mode (ω², γ, q(0), q̇(0) = 0). */
      function modeEval(wsq, q0n, tt, out) {
        const Om2 = wsq - 0.25 * gam * gam, B = 0.5 * gam * q0n; // B = q̇(0) + γ q(0)/2
        let C, S, Cd; // C(t), S(t), dC/dt; dS/dt = C
        if (Om2 > 1e-12) { const W = Math.sqrt(Om2); C = Math.cos(W * tt); S = Math.sin(W * tt) / W; Cd = -W * W * S; }
        else if (Om2 < -1e-12) {
          // e^{-γt/2}cosh κt etc. are evaluated as separate decaying exponentials (no overflow)
          const K = Math.sqrt(-Om2), ep = Math.exp((K - 0.5 * gam) * tt), em = Math.exp((-K - 0.5 * gam) * tt);
          const Ce = 0.5 * (ep + em), Se = 0.5 * (ep - em) / K, Cde = 0.5 * K * (ep - em);
          out[0] = q0n * Ce + B * Se;
          out[1] = -0.5 * gam * out[0] + q0n * Cde + B * Ce;
          return;
        } else { C = 1; S = tt; Cd = 0; }
        const e = Math.exp(-0.5 * gam * tt);
        const val = q0n * C + B * S;
        out[0] = e * val;
        out[1] = e * (-0.5 * gam * val + q0n * Cd + B * C);
      }
      const QO = new Float64Array(2);
      function stateAt(tt, X, Vv) {
        for (let i = 0; i < N; i++) { X[i] = 0; if (Vv) Vv[i] = 0; }
        for (let n = 0; n < N; n++) {
          modeEval(w2[n], q0[n], tt, QO);
          if (Vv) { q[n] = QO[0]; qd[n] = QO[1]; }
          const vn = vec[n];
          for (let i = 0; i < N; i++) { X[i] += vn[i] * QO[0]; if (Vv) Vv[i] += vn[i] * QO[1]; }
        }
      }
      function springPE(X) {
        let U = 0;
        if (leftFixed) U += 0.5 * k * X[0] * X[0];
        if (rightFixed) U += 0.5 * k * X[N - 1] * X[N - 1];
        for (let i = 0; i < N - 1; i++) U += 0.5 * kc * (X[i + 1] - X[i]) ** 2;
        return U;
      }

      function reset() {
        N = Math.round(P.N); m = P.m; k = P.k; kc = P.kc; gam = P.gamma;
        leftFixed = P.leftEnd === "fixed"; rightFixed = P.rightEnd === "fixed";
        // slider ranges that depend on N (never min == max: hidden when N = 1)
        api.setControl("modeN", { max: Math.max(N, 2), value: Math.min(P.modeN, Math.max(N, 2)) });
        api.setControl("pluckAt", { max: Math.max(N, 2), value: Math.min(P.pluckAt, Math.max(N, 2)) });

        // stiffness matrix / m
        const A = new Float64Array(N * N), K = new Float64Array(N * N);
        for (let i = 0; i < N; i++) {
          let d = 0;
          if (i > 0) { d += kc; K[i * N + i - 1] = -kc; }
          if (i < N - 1) { d += kc; K[i * N + i + 1] = -kc; }
          if (i === 0 && leftFixed) d += k;
          if (i === N - 1 && rightFixed) d += k;
          K[i * N + i] = d;
        }
        for (let i = 0; i < N * N; i++) A[i] = K[i] / m;
        const eig = jacobiEigen(A, N);
        sweeps = eig.sweeps; vec = eig.vectors;
        resid = 0;
        for (let n = 0; n < N; n++) {
          w2[n] = eig.values[n] < 0 && eig.values[n] > -1e-9 * (2 * (k + 2 * kc) / m) ? 0 : eig.values[n];
          om[n] = Math.sqrt(Math.max(w2[n], 0));
          let r2 = 0;
          for (let i = 0; i < N; i++) {
            let s = 0; for (let j = 0; j < N; j++) s += K[i * N + j] * vec[n][j];
            r2 += (s - m * eig.values[n] * vec[n][i]) ** 2;
          }
          resid = Math.max(resid, Math.sqrt(r2));
        }

        // initial displacement pattern
        const x0 = new Float64Array(N), Amp = P.A;
        if (P.init === "mass1") x0[0] = 1;
        else if (P.init === "mode") { const n = Math.min(Math.round(P.modeN), N) - 1; for (let i = 0; i < N; i++) x0[i] = vec[n][i]; }
        else if (P.init === "random") { const rng = new PM.RNG(9001 + 7919 * seed); for (let i = 0; i < N; i++) x0[i] = rng.uniform(-1, 1); }
        else { // pluck: triangle peaked at the chosen mass, zero just beyond the ends
          const pk = Math.min(Math.round(P.pluckAt), N);
          for (let i = 1; i <= N; i++) x0[i - 1] = i <= pk ? i / pk : (N + 1 - i) / (N + 1 - pk);
        }
        let mx = 0; for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(x0[i]));
        for (let i = 0; i < N; i++) x0[i] *= Amp / (mx || 1);
        for (let n = 0; n < N; n++) { let s = 0; for (let i = 0; i < N; i++) s += vec[n][i] * x0[i]; q0[n] = Math.abs(s) < 1e-14 * Amp ? 0 : s; }
        E0 = springPE(x0);
        // envelope at t = 0 and bound on |x_i|
        xBound = 0; ampMax = 0;
        for (let n = 0; n < N; n++) { env0[n] = Math.abs(q0[n]); ampMax = Math.max(ampMax, env0[n]); }
        for (let i = 0; i < N; i++) { let s = 0; for (let n = 0; n < N; n++) s += Math.abs(vec[n][i]) * env0[n]; xBound = Math.max(xBound, s); }
        for (let n = 0; n < N; n++) { let mv = 0; for (let i = 0; i < N; i++) mv = Math.max(mv, Math.abs(vec[n][i])); shScale[n] = 0.42 / (mv || 1); }
        xBound = Math.max(xBound, 1e-6); ampMax = Math.max(ampMax, 1e-6);
        t = 0; eMaxSeen = 5;

        // analytic dispersion (uniform chain only, or N = 2 with fixed ends)
        analytic = null; analyticName = "";
        const w0 = 2 * Math.sqrt(kc / m);
        if (Math.abs(k - kc) < 1e-9) {
          if (leftFixed && rightFixed) { analytic = (n) => w0 * Math.abs(Math.sin((n * Math.PI) / (2 * (N + 1)))); analyticName = "2√(k/m)|sin(nπ/2(N+1))|"; }
          else if (!leftFixed && !rightFixed) { analytic = (n) => w0 * Math.abs(Math.sin(((n - 1) * Math.PI) / (2 * N))); analyticName = "2√(k/m) sin((n−1)π/2N)"; }
          else { analytic = (n) => w0 * Math.abs(Math.sin(((2 * n - 1) * Math.PI) / (2 * (2 * N + 1)))); analyticName = "2√(k/m) sin((2n−1)π/2(2N+1))"; }
        }

        // plot limits
        plots.chain.setLimits([-0.35, N + 1.35], [-1.6, 0.62]);
        plots.en.setLimits([0.4, N + 0.6], [0, 105]);
        plots.mo.setLimits([0.4, N + 0.6], [-1.2 * ampMax, 1.2 * ampMax]);
        let wTop = om[N - 1];
        if (analytic) for (let n = 1; n <= N; n++) wTop = Math.max(wTop, analytic(n));
        plots.disp.setLimits([0.4, N + 0.6], [0, Math.max(wTop * 1.15, 0.1)]);
        plots.shape.setLimits([leftFixed ? -0.2 : 0.6, (rightFixed ? N + 1 : N) + 0.25 + 0.17 * (N + 1)], [0.2, N + 0.9]);
        const D = 2.3 * xBound;
        plots.tr.setLimits(null, [-(N - 0.5) * D, 0.6 * D]);
        for (let i = 0; i < N; i++) IDX[i] = i + 1;

        M.set("w", `${PM.fmt(om[0], 3)} / ${PM.fmt(om[N - 1], 3)}`);
        M.set("tb", N >= 2 && om[1] - om[0] > 1e-9 ? PM.fmt((2 * Math.PI) / (om[1] - om[0]), 3) + " s" : "—");
        M.set("res", `${PM.fmt(resid, 2)} (${sweeps})`);
        update();
      }

      function update() {
        stateAt(t, x, v);
        let KE = 0;
        for (let i = 0; i < N; i++) { KE += 0.5 * m * v[i] * v[i]; Emass[i] = 0.5 * m * v[i] * v[i]; }
        if (leftFixed) Emass[0] += 0.5 * k * x[0] * x[0];
        if (rightFixed) Emass[N - 1] += 0.5 * k * x[N - 1] * x[N - 1];
        for (let i = 0; i < N - 1; i++) { const u = 0.25 * kc * (x[i + 1] - x[i]) ** 2; Emass[i] += u; Emass[i + 1] += u; }
        const E = KE + springPE(x);
        let Em = 0;
        for (let n = 0; n < N; n++) {
          Emode[n] = 0.5 * m * (qd[n] * qd[n] + w2[n] * q[n] * q[n]); Em += Emode[n];
          env[n] = om[n] > 1e-9 ? Math.sqrt(q[n] * q[n] + (qd[n] * qd[n]) / w2[n]) : Math.abs(q[n]);
        }
        M.set("E", E0 > 1e-15 ? (100 * E / E0).toFixed(2) + " %" : "E₀ = 0 (static)");
        M.set("chk", E0 > 1e-15 ? PM.fmt(Math.abs(E - Em) / E0, 2) : "—");
        return E;
      }

      function step(dt) { t += dt; }

      // ---------------------------------------------------------------- drawing helpers
      function spring(p, xa, xb, y, col) {
        const L = xb - xa, nZ = 12, lead = Math.min(0.12, 0.2 * Math.abs(L)), amp = 0.075;
        let n = 0;
        ZX[n] = xa; ZY[n] = y; n++;
        ZX[n] = xa + lead; ZY[n] = y; n++;
        for (let j = 1; j < nZ; j++) { ZX[n] = xa + lead + ((L - 2 * lead) * j) / nZ; ZY[n] = y + (j % 2 ? amp : -amp); n++; }
        ZX[n] = xb - lead; ZY[n] = y; n++;
        ZX[n] = xb; ZY[n] = y; n++;
        p.line(ZX.subarray(0, n), ZY.subarray(0, n), { color: col, width: 1.6 });
      }
      function wall(p, xw, side) {
        p.rect(xw, -0.32, xw + side * 0.12, 0.32, { color: "#3a4656" });
        for (let j = 0; j < 7; j++) { const yy = -0.3 + j * 0.1; p.segment(xw + side * 0.12, yy, xw + side * 0.02, yy + 0.08, { color: "#8b98a8", width: 1 }); }
        p.segment(xw, -0.32, xw, 0.32, { color: "#c9d1d9", width: 2 });
      }

      function render() {
        const E = update();
        const p = plots.chain, yC = 0.12, bw = 0.17, bh = 0.17;
        p.clear();
        // track
        p.segment(leftFixed ? 0 : 0.3, yC - bh - 0.01, rightFixed ? N + 1 : N + 0.7, yC - bh - 0.01, { color: "#2b3848", width: 2 });
        if (leftFixed) wall(p, 0, -1);
        if (rightFixed) wall(p, N + 1, 1);
        for (let i = 0; i < N; i++) p.segment(i + 1, yC - bh - 0.06, i + 1, yC - bh - 0.01, { color: PlotColors.muted, width: 1 });
        // springs
        if (leftFixed) spring(p, 0, 1 + x[0] - bw, yC, PlotColors.accent3);
        if (rightFixed) spring(p, N + x[N - 1] + bw, N + 1, yC, PlotColors.accent3);
        for (let i = 0; i < N - 1; i++) spring(p, i + 1 + x[i] + bw, i + 2 + x[i + 1] - bw, yC, kc > 0 ? PlotColors.accent : "rgba(139,152,168,0.25)");
        // blocks coloured by their energy share
        for (let i = 0; i < N; i++) {
          const X = i + 1 + x[i], f = E0 > 1e-15 ? Math.min(1, Emass[i] / E0) : 0;
          p.rect(X - bw, yC - bh, X + bw, yC + bh, { color: colormap("plasma", 0.15 + 0.85 * f), stroke: "#ffffff", strokeWidth: 1 });
          p.text(i + 1, yC - bh - 0.14, String(i + 1), { align: "center", size: 10.5, color: PlotColors.muted });
        }
        // displacement profile drawn sideways
        if (P.profile) {
          const yb = -0.95, sc = 0.5 / xBound;
          p.segment(leftFixed ? 0 : 0.6, yb, rightFixed ? N + 1 : N + 0.4, yb, { color: "#2b3848", width: 1.2 });
          let n = 0;
          if (leftFixed) { PX[n] = 0; PY[n] = yb; n++; }
          for (let i = 0; i < N; i++) {
            const yy = yb + x[i] * sc;
            p.segment(i + 1, yb, i + 1, yy, { color: PlotCycle[i % PlotCycle.length], width: 2 });
            PX[n] = i + 1; PY[n] = yy; n++;
          }
          if (rightFixed) { PX[n] = N + 1; PY[n] = yb; n++; }
          p.line(PX.subarray(0, n), PY.subarray(0, n), { color: "#ffffff", width: 1.2, alpha: 0.6 });
          for (let i = 0; i < N; i++) p.circle(i + 1, yb + x[i] * sc, 4, { px: true, color: PlotCycle[i % PlotCycle.length] });
          p.textPx(p.m.l + 6, p.Y(yb - 0.5), "profile ×" + PM.fmt(sc, 2), { size: 10.5, color: PlotColors.muted, baseline: "bottom" });
        }
        p.label([`t = ${PM.fmt(t, 2)} s`], "tl", { size: 11.5 });

        // traces x_i(t)
        const pt = plots.tr, W = P.win, t1 = Math.max(t, W), t0 = t1 - W, D = 2.3 * xBound;
        pt.setLimits([t0, t1], null);
        pt.clear();
        const tEnd = Math.min(t, t1);
        const nS = Math.max(2, Math.min(NTR, Math.round(NTR * (tEnd - t0) / W) + 2));
        for (let s = 0; s < nS; s++) {
          const ts = t0 + ((tEnd - t0) * s) / (nS - 1);
          TT[s] = ts;
          if (ts < 0) { for (let i = 0; i < N; i++) TX[i][s] = NaN; continue; }
          stateAt(ts, x, null);
          for (let i = 0; i < N; i++) TX[i][s] = x[i] - i * D;
        }
        stateAt(t, x, v); // restore the current state (also q, qd)
        for (let i = 0; i < N; i++) {
          pt.hline(-i * D, { color: PlotColors.muted, alpha: 0.25, width: 1 });
          pt.line(TT.subarray(0, nS), TX[i].subarray(0, nS), { color: PlotCycle[i % PlotCycle.length], width: 1.5 });
          pt.text(t0, -i * D + 0.3 * D, "x" + sub(i + 1), { dx: 6, size: 11, color: PlotCycle[i % PlotCycle.length], bold: true });
        }
        if (N >= 2 && om[1] - om[0] > 1e-9 && P.init === "mass1") {
          pt.label([`beat period 2π/(ω₂−ω₁) = ${PM.fmt((2 * Math.PI) / (om[1] - om[0]), 2)} s`], "tr", { size: 11 });
        }
        pt.label([`baseline spacing ${PM.fmt(D, 3)} m`], "br", { size: 10.5, color: PlotColors.muted });

        // energy per mass
        const pe = plots.en, pct = PCT.subarray(0, N);
        let mxp = 0;
        for (let i = 0; i < N; i++) { pct[i] = E0 > 1e-15 ? (100 * Emass[i]) / E0 : 0; mxp = Math.max(mxp, pct[i]); }
        eMaxSeen = Math.max(eMaxSeen, mxp);
        pe.setLimits(null, [0, Math.min(105, eMaxSeen * 1.12)]);
        pe.clear();
        pe.bars(IDX.subarray(0, N), pct, 0.7, { colors: COLS, alpha: 0.85 });
        pe.label([`total E = ${PM.fmt(E, 3)} J  (${E0 > 1e-15 ? ((100 * E) / E0).toFixed(1) : "—"} % of E₀)`], "tr", { size: 11 });

        // mode amplitudes
        const pm = plots.mo;
        pm.clear();
        pm.hline(0, { color: PlotColors.muted, alpha: 0.5 });
        for (let n = 0; n < N; n++) {
          pm.rect(n + 1 - 0.35, -env[n], n + 1 + 0.35, env[n], { color: PlotColors.accent2, alpha: 0.22 });
          pm.rect(n + 1 - 0.22, 0, n + 1 + 0.22, q[n], { color: PlotColors.accent2, alpha: 0.9 });
        }
        let Etot = 0; for (let n = 0; n < N; n++) Etot += Emode[n];
        if (Etot > 0) {
          let best = 0; for (let n = 1; n < N; n++) if (Emode[n] > Emode[best]) best = n;
          pm.label([`largest mode energy: n = ${best + 1} (${((100 * Emode[best]) / Etot).toFixed(1)} %)`], "tr", { size: 11 });
        }

        // dispersion
        const pd = plots.disp;
        pd.clear();
        if (analytic) {
          let nA = 0;
          for (let j = 0; j < AX.length; j++) { const nn = 0.4 + ((N + 0.2) * j) / (AX.length - 1); AX[j] = nn; AY[j] = analytic(nn); nA++; }
          pd.line(AX.subarray(0, nA), AY.subarray(0, nA), { color: PlotColors.accent3, width: 1.6, dash: [6, 4] });
        } else if (N === 2 && leftFixed && rightFixed) {
          pd.points([1, 2], [Math.sqrt(k / m), Math.sqrt((k + 2 * kc) / m)], { color: PlotColors.accent3, size: 7, alpha: 0.5 });
        }
        if (leftFixed && rightFixed && analytic) pd.hline(2 * Math.sqrt(kc / m), { color: PlotColors.muted, dash: [3, 4], alpha: 0.6 });
        for (let n = 0; n < N; n++) MSX[n] = n + 1;
        pd.points(MSX.subarray(0, N), om.subarray(0, N), { color: PlotColors.accent, size: 4.5, stroke: "#0f151c" });
        if (P.init === "mode") { const n = Math.min(Math.round(P.modeN), N); pd.circle(n, om[n - 1], 8, { px: true, fill: false, stroke: "#ffffff" }); }
        const leg = [{ label: "Jacobi eigenfrequencies", color: PlotColors.accent, type: "dot" }];
        if (analytic) leg.push({ label: "ω = " + analyticName, color: PlotColors.accent3, dash: [6, 4] });
        else if (N === 2 && leftFixed && rightFixed) leg.push({ label: "√(k/m), √((k+2k_c)/m)", color: PlotColors.accent3, type: "dot" });
        pd.legend(leg, "br");
        if (!analytic && !(N === 2 && leftFixed && rightFixed)) pd.label(["analytic curve: press “Uniform chain” (k = k_c)"], "tl", { size: 10.5, color: PlotColors.muted });

        // mode shapes
        const ps = plots.shape;
        ps.clear();
        for (let n = 0; n < N; n++) {
          let c = 0;
          if (leftFixed) { MSX[c] = 0; MSY[c] = n + 1; c++; }
          for (let i = 0; i < N; i++) { MSX[c] = i + 1; MSY[c] = n + 1 + shScale[n] * vec[n][i]; c++; }
          if (rightFixed) { MSX[c] = N + 1; MSY[c] = n + 1; c++; }
          const sel = P.init === "mode" && Math.min(Math.round(P.modeN), N) === n + 1;
          const col = sel ? "#ffffff" : PlotCycle[n % PlotCycle.length];
          ps.hline(n + 1, { color: PlotColors.muted, alpha: 0.18, width: 1 });
          ps.line(MSX.subarray(0, c), MSY.subarray(0, c), { color: col, width: sel ? 2.4 : 1.5 });
          ps.points(MSX.subarray(leftFixed ? 1 : 0, leftFixed ? N + 1 : N), MSY.subarray(leftFixed ? 1 : 0, leftFixed ? N + 1 : N), { color: col, size: 2.6 });
        }
        if (N <= 12) {
          for (let n = 0; n < N; n++) ps.textPx(ps.m.l + ps._v.pw - 4, ps.Y(n + 1), "ω=" + PM.fmt(om[n], 2), { align: "right", size: 9.5, color: PlotColors.muted, bg: "#0f151c" });
        }
        api.setTime(`t = ${PM.fmt(t, 2)} s`);
      }

      return {
        reset, step, render,
        onAction(id) {
          if (id === "uniform") { api.setControl("kc", { value: P.k }); reset(); }
          else if (id === "reroll") { seed++; reset(); }
          api.invalidate();
        },
      };
    },
  });
})();
