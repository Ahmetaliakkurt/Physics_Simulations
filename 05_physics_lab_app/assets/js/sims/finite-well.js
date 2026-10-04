/* Finite potential well solver — finite differences + tridiagonal eigenvalue problem (PM.tridiagLowest).
 * Re-solved instantly when a parameter changes; a two-state superposition is played live. */
App.register({
  id: "finite-well",
  category: "quantum",
  order: 31,
  title: "Finite Potential Well Solver",
  icon: "🧪",
  subtitle: "Bound states of smooth, square, double and triple wells from a diagonalised finite-difference Hamiltonian, plus the live tunnelling oscillation of a two-state superposition.",
  notes: [
    { type: "info", html: "The calculation is done inside a finite box $[-L,L]$ with $\\psi(\\pm L)=0$. Besides the true bound states ($E\\lt 0$) the eigenvalue list therefore also contains $E\\ge0$ states quantised by the box walls; they become the continuum as $L\\to\\infty$ and are not bound states. They are drawn grey and dotted when enabled." },
  ],
  animated: true,
  speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
  controls: [
    { id: "type", type: "select", label: "Well type", value: "smooth", live: true,
      options: [{ value: "smooth", label: "Smooth well (super-Gaussian)" }, { value: "square", label: "Finite square well" },
        { value: "double_well", label: "Symmetric double well (tunnelling)" }, { value: "triple_well", label: "Symmetric triple well (splitting)" }] },
    { id: "V0", type: "slider", label: "Well depth $V_0$", min: 1, max: 30, step: 0.5, value: 10, live: true },
    { id: "width", type: "slider", label: "Width of each well $w$", min: 0.5, max: 6, step: 0.1, value: 2, live: true },
    { id: "power", type: "slider", label: "Edge sharpness exponent $p$", min: 2, max: 20, step: 1, value: 8, live: true, visibleIf: (p) => p.type === "smooth" },
    { id: "barrier", type: "slider", label: "Barrier width between wells $b$", min: 0.1, max: 5, step: 0.1, value: 0.5, live: true,
      visibleIf: (p) => p.type === "double_well" || p.type === "triple_well" },
    { id: "L", type: "slider", label: "Computational domain $\\pm L$", min: 4, max: 20, step: 0.5, value: 6, live: true },
    { id: "mass", type: "slider", label: "Mass $m$", min: 0.2, max: 3, step: 0.1, value: 1, live: true },
    { id: "nst", type: "slider", label: "Number of states shown", min: 1, max: 12, step: 1, value: 4, live: true },
    { type: "section", label: "Display" },
    { id: "showE", type: "checkbox", label: "Energy levels", value: true, live: true },
    { id: "showW", type: "checkbox", label: "Wave functions ψₙ", value: true, live: true },
    { id: "showP", type: "checkbox", label: "Probability densities |ψₙ|²", value: false, live: true },
    { id: "showArt", type: "checkbox", label: "Also show box states ($E\\ge0$)", value: false, live: true },
    { type: "section", label: "Superposition (time evolution)" },
    { id: "sup", type: "checkbox", label: "Play a two-state superposition", value: true, live: true },
    { id: "pair", type: "select", label: "State pair", value: "0,1", live: true, visibleIf: (p) => p.sup,
      options: [{ value: "0,1", label: "ψ₀ + ψ₁" }, { value: "0,2", label: "ψ₀ + ψ₂" }, { value: "1,2", label: "ψ₁ + ψ₂" }, { value: "2,3", label: "ψ₂ + ψ₃" }] },
    { id: "tper", type: "slider", label: "Playback time of one period", min: 2, max: 12, step: 0.5, value: 4, unit: "s", live: true, visibleIf: (p) => p.sup },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A particle of mass $m$ moves in one dimension in a static potential $V(x)\\le0$ that vanishes far from the origin. Units are $\\hbar=1$ with
    $m$ adjustable (default 1); lengths are in an arbitrary unit $\\ell$ and energies in $\\hbar^2/m_0\\ell^2$. Four potentials are available, each well
    having depth $V_0$ and width $w$:</p>
    <ul>
      <li><b>smooth well</b> $V=-V_0\\,\\exp[-|2x/w|^{p}]$ — a Gaussian for $p=2$, approaching a square well as $p\\to\\infty$;</li>
      <li><b>square well</b> $V=-V_0$ for $|x|\\lt w/2$, otherwise 0;</li>
      <li><b>double well</b>: two square wells of width $w$ separated by a barrier of width $b$ (height $V_0$ above the well floor);</li>
      <li><b>triple well</b>: three such wells separated by two barriers of width $b$.</li>
    </ul>
    <p>To make the problem finite the particle is enclosed in a hard-walled box $[-L,L]$ much larger than the wells.</p>

    <h4>Equations being solved</h4>
    <p>The stationary states are the solutions of the time-independent Schrödinger equation</p>
    $$-\\frac{\\hbar^2}{2m}\\frac{d^2\\psi}{dx^2}+V(x)\\,\\psi(x)=E\\,\\psi(x),\\qquad \\psi(-L)=\\psi(L)=0 .$$
    <p>States with $E\\lt 0$ decay as $e^{-\\kappa|x|}$, $\\kappa=\\sqrt{2m|E|}/\\hbar$, outside the wells and are the bound states. For the square well
    they satisfy $\\tan z=\\sqrt{z_0^2/z^2-1}$ (even) or $-\\cot z=\\sqrt{z_0^2/z^2-1}$ (odd), $z_0=\\tfrac w2\\sqrt{2mV_0}/\\hbar$, and their number is
    $\\lceil 2z_0/\\pi\\rceil$.</p>
    <p><b>Double-well tunnelling.</b> Each single-well level splits into a symmetric state $\\psi_0$ and an antisymmetric state $\\psi_1$ with
    $\\Delta E=E_1-E_0\\propto e^{-\\kappa b}$. The superposition $\\Psi=(\\psi_0e^{-iE_0t/\\hbar}+\\psi_1e^{-iE_1t/\\hbar})/\\sqrt2$ starts localised in the left well and
    its density</p>
    <div class="callout">
    $$|\\Psi(x,t)|^2=\\tfrac12\\Big[\\psi_0^2+\\psi_1^2+2\\psi_0\\psi_1\\cos\\frac{\\Delta E\\,t}{\\hbar}\\Big],\\qquad T=\\frac{2\\pi\\hbar}{\\Delta E}$$
    </div>
    <p>oscillates between the wells with period $T$: the particle is in the right well after $T/2=\\pi\\hbar/\\Delta E$ and back after $T$. In a triple well every
    level splits into a triplet.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Discretisation.</b> $N_g=1200$ grid points $x_i=-L+i\\Delta x$, $\\Delta x=2L/(N_g-1)$ (0.01 for $L=6$). The second derivative becomes
      $\\psi''_i\\approx(\\psi_{i+1}-2\\psi_i+\\psi_{i-1})/\\Delta x^2$, giving a symmetric tridiagonal matrix with
      $H_{ii}=\\hbar^2/(m\\Delta x^2)+V(x_i)$ and $H_{i,i\\pm1}=-\\hbar^2/(2m\\Delta x^2)$. Dropping the end points imposes $\\psi(\\pm L)=0$.</li>
      <li><b>Eigenvalues.</b> The lowest 12 eigenvalues are found by <em>Sturm-sequence bisection</em>: the number of negative pivots of $H-E\\,I$ in an
      $LDL^T$ factorisation equals the number of eigenvalues below $E$, so each eigenvalue is bracketed to a relative accuracy of $10^{-13}$.
      The same count at $E=0$ gives the number of bound states directly (metric).</li>
      <li><b>Eigenvectors.</b> Four sweeps of <em>inverse iteration</em>, solving $(H-\\tilde E I)v=v_{old}$ with the Thomas algorithm, with re-orthogonalisation
      for nearly degenerate pairs (deep double wells). The vectors are rescaled by $1/\\sqrt{\\Delta x}$ so that $\\sum_i\\psi_i^2\\Delta x=1$; the sign is fixed so the leftmost significant amplitude is positive.</li>
      <li><b>Time evolution.</b> No time stepping is needed: the superposition is evaluated exactly from the eigenpairs with the phase $\\Delta E\\,t/\\hbar$.
      One period $T$ is mapped onto the chosen playback time; the physical time $t$ is shown in the toolbar. $P_{left}=\\int_{x\\lt 0}|\\Psi|^2dx$ is a rectangle-rule sum.</li>
      <li><b>Accuracy.</b> The finite-difference error is $O(\\Delta x^2)$: for the default square well the lowest level is $-9.181$, close to the
      exact transcendental-equation value. The bound-state energies should not change when you enlarge $L$ — a good check that the box is big enough.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Square well</b> with $V_0=10$, $w=2$, $m=1$: $z_0=\\sqrt{20}\\approx4.47$, so there are $\\lceil 2.85\\rceil=3$ bound states, at $E\\approx-9.18,-6.78,-3.06$.</li>
      <li><b>Double well, barrier width.</b> With $V_0=10$, $w=2$: $b=0.5$ gives $\\Delta E\\approx0.067$ ($T\\approx94$); $b=1.0$ gives $\\Delta E\\approx0.0078$
      ($T\\approx800$). Each extra 0.5 of barrier reduces $\\Delta E$ by ≈ 8.5 ≈ $e^{\\kappa\\cdot0.5}$ with $\\kappa=\\sqrt{2\\cdot9.18}\\approx4.3$.</li>
      <li><b>Watch the tunnelling.</b> With "ψ₀ + ψ₁" the density starts in the left well; $P_{left}$ follows $\\tfrac12(1+\\cos\\Delta E t/\\hbar)$. Try the pair ψ₂ + ψ₃:
      the excited doublet is split more, so its period is much shorter in physical time.</li>
      <li><b>Heavier particle.</b> Increase $m$ to 2: more bound states appear ($\\propto\\sqrt m$) and $\\Delta E$ drops because $\\kappa\\propto\\sqrt m$.</li>
      <li><b>Smooth well, $p=2$.</b> A Gaussian well: near the bottom it is harmonic with $\\hbar\\omega=\\hbar\\sqrt{8V_0/mw^2}\\approx4.5$, but the computed spacing
      ($E_1-E_0\\approx3.7$) is smaller because the Gaussian softens away from the minimum. Increase $p$ to 20 and the levels approach the square-well values.</li>
      <li><b>Box states.</b> Tick "box states" and change $L$: the $E\\ge0$ levels move like $(n\\pi\\hbar/2L)^2/2m$, the bound levels stay put.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>One dimension, a non-relativistic single particle, piecewise-constant wells with sharp edges (resolved to $\\Delta x$), and a hard box standing in for
    open space, so the continuum appears as discrete box states. See D. J. Griffiths &amp; D. F. Schroeter, <em>Introduction to Quantum Mechanics</em>, §2.6
    (finite square well); L. D. Landau &amp; E. M. Lifshitz, <em>Quantum Mechanics</em>, §50 (double-well splitting); W. H. Press et al., <em>Numerical Recipes</em>,
    ch. 11 (tridiagonal eigenproblems).</p>`,

  mount(api) {
    const P = api.params, hbar = 1, NG = 1200, KMAX = 12, WIN = 3; // WIN: periods shown in the P_left plot
    const plots = api.plots([
      { id: "main", title: "Potential, energy levels and wave functions", span: 2, aspect: 0.46, xlim: [-6, 6], ylim: [-15, 3], xlabel: "position x", ylabel: "energy E" },
      { id: "sup", title: "Superposition |Ψ(x,t)|²", aspect: 0.62, xlim: [-6, 6], ylim: [0, 1], xlabel: "position x", ylabel: "|Ψ|²" },
      { id: "pl", title: "Probability of the left half P_left(t)", aspect: 0.62, xlim: [0, WIN], ylim: [0, 1.02], xlabel: "t / T", ylabel: "P(x<0)" },
    ]);
    const Mt = api.metrics([
      { id: "nb", label: "Bound states ($E\\lt 0$)" },
      { id: "e0", label: "$E_0$" },
      { id: "e1", label: "$E_1$" },
      { id: "de", label: "Splitting $\\Delta E=E_b-E_a$" },
      { id: "tt", label: "Period $T=2\\pi\\hbar/\\Delta E$" },
    ]);
    const x = new Float64Array(NG), V = new Float64Array(NG), rho = new Float64Array(NG), yb = new Float64Array(NG), buf = new Float64Array(NG);
    let dx, E = [], psi = [], nBound = 0, phase = 0, hist = [], histT = [];

    function potential() {
      const { type, V0, width, barrier: b, L, power } = P;
      dx = (2 * L) / (NG - 1);
      for (let i = 0; i < NG; i++) {
        const xi = -L + i * dx; x[i] = xi; let v = 0;
        if (type === "smooth") v = -V0 * Math.exp(-Math.pow(Math.abs(xi / (width / 2)), power));
        else if (type === "square") v = xi > -width / 2 && xi < width / 2 ? -V0 : 0;
        else if (type === "double_well") v = (xi > -b / 2 - width && xi < -b / 2) || (xi > b / 2 && xi < b / 2 + width) ? -V0 : 0;
        else {
          const rs = width / 2 + b, le = -width / 2 - b;
          v = (xi > -width / 2 && xi < width / 2) || (xi > rs && xi < rs + width) || (xi > le - width && xi < le) ? -V0 : 0;
        }
        V[i] = v;
      }
    }
    function solve() {
      potential();
      const c = -hbar * hbar / (2 * P.mass * dx * dx);
      const d = new Float64Array(NG), e = new Float64Array(NG - 1).fill(c);
      for (let i = 0; i < NG; i++) d[i] = V[i] - 2 * c;
      // Sturm count: number of eigenvalues below 0 = number of bound states
      let cnt = 0, q = d[0];
      if (q < 0) cnt++;
      for (let i = 1; i < NG; i++) { q = d[i] - (c * c) / (q === 0 ? 1e-300 : q); if (q < 0) cnt++; }
      nBound = cnt;
      const r = PM.tridiagLowest(d, e, KMAX);
      E = Array.from(r.values);
      const s = 1 / Math.sqrt(dx);
      psi = r.vectors.map((v) => { const o = new Float64Array(NG); for (let i = 0; i < NG; i++) o[i] = v[i] * s; return o; });
      hist = []; histT = [];
    }
    function pairIdx() { const [a, b] = P.pair.split(",").map(Number); return [a, b]; }
    function superpose() {
      const [a, b] = pairIdx(), pa = psi[a], pb = psi[b], cph = Math.cos(phase);
      let left = 0, mx = 0;
      for (let i = 0; i < NG; i++) {
        const r = 0.5 * (pa[i] * pa[i] + pb[i] * pb[i] + 2 * pa[i] * pb[i] * cph);
        rho[i] = r; if (r > mx) mx = r;
        if (x[i] < 0) left += r * dx; else if (x[i] === 0) left += 0.5 * r * dx;
      }
      return { left, mx };
    }

    return {
      reset() { solve(); phase = 0; },
      onParam(id) {
        if (["type", "V0", "width", "power", "barrier", "L", "mass"].includes(id)) solve();
        if (id === "pair" || id === "sup") { phase = 0; hist = []; histT = []; }
        if (id === "L") { plots.main.setLimits([-P.L, P.L]); plots.sup.setLimits([-P.L, P.L]); }
      },
      step(dt) {
        if (!P.sup) return;
        phase += (2 * Math.PI * dt) / P.tper; // phase = ΔE t / ħ
        const { left } = superpose();
        hist.push(left); histT.push(phase / (2 * Math.PI)); // t / T
        while (histT.length > 1 && histT[histT.length - 1] - histT[0] > WIN) { hist.shift(); histT.shift(); }
      },
      render() {
        const { V0, L, nst } = P;
        const pm = plots.main;
        const scale = Math.max(V0 * 0.15, 0.5);
        const lim = Math.min(nst, E.length);
        let hi = -V0;
        for (let i = 0; i < lim; i++) if (E[i] < 0 || P.showArt) hi = Math.max(hi, E[i]);
        const clear = P.showW || P.showP ? scale * 1.5 : V0 * 0.2;
        pm.setLimits([-L, L], [-V0 * 1.5, Math.max(hi + clear, V0 * 0.1)]);
        pm.clear();
        pm.fill(x, V, -V0 * 2, { color: PlotColors.muted, alpha: 0.15 });
        pm.line(x, V, { color: PlotColors.muted, width: 2 });
        pm.hline(0, { color: PlotColors.muted, dash: [2, 4], alpha: 0.6 });
        let lastY = Infinity;
        for (let i = 0; i < lim; i++) {
          const art = E[i] >= 0;
          if (art && !P.showArt) continue;
          const col = art ? PlotColors.muted : PlotCycle[i % PlotCycle.length];
          const p = psi[i];
          let ma = 0; for (let j = 0; j < NG; j++) ma = Math.max(ma, Math.abs(p[j]));
          const f = (scale * 0.6) / (ma || 1);
          if (P.showP) {
            for (let j = 0; j < NG; j++) yb[j] = E[i] + (p[j] * p[j] * f * f) / (scale * 0.6);
            pm.fill(x, yb, E[i], { color: col, alpha: art ? 0.15 : 0.3 });
          }
          if (P.showE) pm.hline(E[i], { color: col, width: art ? 1.2 : 1.8, dash: art ? [2, 4] : [], alpha: 0.85 });
          if (P.showW) {
            for (let j = 0; j < NG; j++) yb[j] = E[i] + p[j] * f;
            pm.line(x, yb, { color: col, width: 1.6, dash: art ? [2, 4] : [], alpha: art ? 0.7 : 1 });
          }
          const Y = pm.Y(E[i]);
          if (lastY - Y > 15) { pm.text(L, E[i], `n=${i}  E=${PM.fmt(E[i], 3)}${art ? " (box)" : ""}`, { align: "right", dx: -6, dy: -8, size: 10.5, color: col, bg: "#0f151c" }); lastY = Y; }
        }
        const hidden = E.slice(0, lim).filter((e) => e >= 0).length;
        if (hidden && !P.showArt) pm.label(`${hidden} state(s) with E ≥ 0 (box states, hidden)`, "tl", { color: PlotColors.muted, size: 11 });

        // superposition
        const [a, b] = pairIdx(), dE = E[b] - E[a];
        const ps = plots.sup, pl = plots.pl;
        ps.setLimits([-L, L]);
        if (P.sup) {
          const { left, mx } = superpose();
          let vmax = 0; for (let i = 0; i < NG; i++) { const pa = psi[a][i], pb = psi[b][i]; vmax = Math.max(vmax, 0.5 * (Math.abs(pa) + Math.abs(pb)) ** 2); }
          ps.setLimits(null, [0, vmax * 1.1 || 1]);
          ps.clear();
          for (let i = 0; i < NG; i++) buf[i] = vmax * 1.05 * (V[i] / V0 + 1);
          ps.fill(x, buf, 0, { color: PlotColors.muted, alpha: 0.12 });
          ps.fill(x, rho, 0, { color: PlotColors.accent, alpha: 0.4 });
          ps.line(x, rho, { color: PlotColors.accent, width: 2 });
          ps.vline(0, { color: PlotColors.muted, dash: [3, 4] });
          ps.label([`P_left = ${PM.fmt(left, 3)}`, `max |Ψ|² = ${PM.fmt(mx, 3)}`], "tr");
          const t0 = histT.length ? histT[0] : 0;
          pl.setLimits([t0, t0 + WIN]);
          pl.clear();
          for (let k = Math.ceil(t0 * 2) / 2; k <= t0 + WIN; k += 0.5) pl.vline(k, { color: PlotColors.muted, dash: [2, 4], alpha: Number.isInteger(k) ? 0.6 : 0.25 });
          if (hist.length > 1) pl.line(histT, hist, { color: PlotColors.accent3, width: 2 });
          pl.label(`ψ${a} + ψ${b}`, "tr");
          api.setTime(`t = ${PM.fmt(phase / (dE || 1), 3)}`);
        } else {
          ps.clear(); ps.label("Superposition off", "tl");
          pl.clear(); pl.label("Superposition off", "tl");
          api.setTime("");
        }

        Mt.set("nb", String(nBound));
        Mt.set("e0", PM.fmt(E[0], 4));
        Mt.set("e1", PM.fmt(E[1], 4));
        Mt.set("de", PM.fmt(dE, 4));
        Mt.set("tt", dE > 0 ? PM.fmt((2 * Math.PI * hbar) / dE, 3) : "—");
      },
    };
  },
});
