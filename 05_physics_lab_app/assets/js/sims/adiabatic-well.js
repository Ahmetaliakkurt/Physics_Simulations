/* Infinite square well with a moving wall — the coefficients in the instantaneous eigenbasis are
 * evolved live and unitarily every frame through an expansion → turn-around → compression cycle. */
App.register({
  id: "adiabatic-well",
  category: "quantum",
  order: 30,
  title: "Adiabatically Expanding Well",
  icon: "📦",
  subtitle: "An infinite square well whose right wall moves out and back: watch the ground state survive a slow (adiabatic) cycle and break up into many levels in a fast (sudden) one.",
  notes: [
    { type: "info", html: "The right wall first opens at constant speed, turns around smoothly at $L_{end}$ and closes again at the same speed. Because the instantaneous eigenstates $\\phi_n(x;L(t))$ change with time, the particle need not stay in one level. Change the wall speed to compare the <b>adiabatic</b> regime (the particle stays in the ground state and ⟨E⟩ returns to its initial value) with the <b>sudden</b> regime (levels mix and ⟨E⟩ ends up irreversibly higher)." },
  ],
  animated: true,
  speed: { min: 0.2, max: 4, value: 1, step: 0.1 },
  controls: [
    { id: "logv", type: "slider", label: "Wall speed $v_{max}$", min: -1.5, max: 0.7, step: 0.01, value: 0,
      fmt: (v) => { const x = Math.pow(10, v); return x < 0.1 ? x.toFixed(3) : x.toFixed(2); },
      help: "Logarithmic scale: 0.03 (very slow) … 5 (very fast)." },
    { id: "L0", type: "slider", label: "Initial width $L_0$", min: 0.5, max: 3, step: 0.1, value: 1 },
    { id: "Lturn", type: "slider", label: "Start of turn-around $L_{turn}$", min: 1.5, max: 15, step: 0.5, value: 6 },
    { id: "Lend", type: "slider", label: "Maximum width $L_{end}$", min: 2, max: 20, step: 0.5, value: 8 },
    { id: "N", type: "slider", label: "Basis size $N$ (number of eigenstates)", min: 8, max: 30, step: 1, value: 20 },
    { type: "section", label: "Display" },
    { id: "tplay", type: "slider", label: "Playback time of one cycle", min: 5, max: 40, step: 1, value: 14, unit: "s", live: true,
      help: "Every speed plays the whole cycle in this time (the physical duration $T_{\\text{cycle}}$ is shown in the status line)." },
    { id: "nshow", type: "slider", label: "Number of levels shown", min: 3, max: 12, step: 1, value: 8, live: true },
    { id: "showRef", type: "checkbox", label: "Adiabatic reference $E_1(L(t))$", value: true, live: true },
    { id: "showGhost", type: "checkbox", label: "Show ⟨E⟩ curve of the previous cycle", value: true, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A single particle of mass $m$ is confined to a one-dimensional infinite square well $0\\le x\\le L(t)$. The left wall is
    fixed at $x=0$; the right wall moves according to a prescribed protocol $L(t)$, so the boundary itself is time dependent.
    Units are $\\hbar=m=1$: lengths are in an arbitrary unit $\\ell$, energies in $\\hbar^2/m\\ell^2$ and times in $m\\ell^2/\\hbar$.
    The particle starts in the ground state of the well of width $L_0$. The wall protocol has three stages:</p>
    <ul>
      <li>expansion at constant speed $v$: $L=L_0+vt$ for $0\\lt t\\lt t_1$, $t_1=(L_{turn}-L_0)/v$;</li>
      <li>smooth turn-around $L=L_{turn}+(L_{end}-L_{turn})\\sin(\\pi\\tau/2t_2)$ for $0\\lt \\tau\\lt 2t_2$, with $t_2=\\pi(L_{end}-L_{turn})/2v$, so that $\\dot L$ is continuous and $|\\dot L|\\le v$;</li>
      <li>compression at speed $v$: $L=L_{turn}-v\\tau$ back to $L_0$. The total cycle time is $T_{\\text{cycle}}=2(t_1+t_2)$.</li>
    </ul>

    <h4>Equations being solved</h4>
    <p>Inside the well the time-dependent Schrödinger equation is $i\\hbar\\,\\partial_t\\psi=-\\frac{\\hbar^2}{2m}\\partial_x^2\\psi$ with the
    moving boundary conditions $\\psi(0,t)=\\psi(L(t),t)=0$. At every instant the Hamiltonian has the eigenstates</p>
    $$\\phi_n(x;L)=\\sqrt{\\frac{2}{L}}\\sin\\frac{n\\pi x}{L},\\qquad E_n(L)=\\frac{n^2\\pi^2\\hbar^2}{2mL^2},\\qquad n=1,2,\\dots$$
    <p>We expand $\\psi(x,t)=\\sum_m c_m(t)\\,\\phi_m(x;L(t))$; every term satisfies the boundary conditions automatically. Inserting this
    into the Schrödinger equation and projecting onto $\\phi_n$ gives</p>
    $$\\dot c_n=-\\frac{i}{\\hbar}E_n c_n-\\dot L\\sum_m\\langle\\phi_n|\\partial_L\\phi_m\\rangle\\,c_m .$$
    <p><b>The coupling.</b> Differentiating, $\\partial_L\\phi_m=-\\frac{1}{2L}\\phi_m-\\sqrt{\\tfrac{2}{L}}\\,\\frac{m\\pi x}{L^2}\\cos\\frac{m\\pi x}{L}$.
    With $u=\\pi x/L$ the overlap reduces to $I=\\int_0^\\pi u\\sin(nu)\\cos(mu)\\,du$. Using $\\sin nu\\cos mu=\\tfrac12[\\sin(n+m)u+\\sin(n-m)u]$ and
    $\\int_0^\\pi u\\sin ku\\,du=\\pi(-1)^{k+1}/k$ one finds $I=\\pi(-1)^{n+m+1}\\,n/(n^2-m^2)$ for $n\\ne m$, and for $n=m$ the two terms cancel
    exactly. Hence</p>
    <div class="callout">
    $$\\dot c_n=-\\frac{i}{\\hbar}E_n(L)\\,c_n-\\frac{\\dot L}{L}\\sum_{m\\ne n}M_{nm}\\,c_m,\\qquad
      M_{nm}=L\\,\\langle\\phi_n|\\partial_L\\phi_m\\rangle=(-1)^{n+m}\\,\\frac{2nm}{n^2-m^2}$$
    </div>
    <p>The coupling connects <em>every</em> pair $n\\ne m$ (not only neighbours), with sign $(-1)^{n+m}$, because a single wall moves and
    the problem has no reflection symmetry about the well centre. $M$ is real and antisymmetric, so the generator
    $A=-\\tfrac{i}{\\hbar}\\mathrm{diag}(E_n)-\\tfrac{\\dot L}{L}M$ is anti-Hermitian and $\\sum_n|c_n|^2=1$ is conserved exactly.</p>
    <p><b>Adiabatic theorem.</b> If the wall moves slowly compared with the internal frequencies,
    $\\hbar\\,|\\dot L/L|\\,|M_{nm}|\\ll|E_n-E_m|$, transitions are suppressed: $|c_n|^2$ stays constant and each coefficient only picks up the
    dynamical phase $e^{-i\\int E_n dt/\\hbar}$. A particle starting in $\\phi_1$ remains in $\\phi_1(L(t))$, so
    $\\langle E\\rangle=E_1(L(t))\\propto 1/L^2$ and the cycle is reversible. For the dominant $1\\to2$ coupling ($|M_{21}|=4/3$, $E_2-E_1=3\\pi^2\\hbar^2/2mL^2$)
    the adiabaticity parameter is</p>
    $$\\varepsilon(L)=\\frac{\\hbar\\,\\tfrac43|\\dot L|/L}{E_2-E_1}=\\frac{8\\,m\\,|\\dot L|\\,L}{9\\pi^2\\hbar}.$$
    <p>It grows with $L$, so the widest point of the cycle is the most dangerous; the metric shows $\\varepsilon_{max}=\\varepsilon(L_{end})$ at $|\\dot L|=v$.</p>
    <p><b>Sudden limit.</b> For $\\varepsilon\\gg1$ the wave function has no time to react: an instantaneous expansion leaves $\\psi$ unchanged,
    so $c_n=\\langle\\phi_n(L')|\\phi_1(L_0)\\rangle$ and $\\langle E\\rangle$ stays at $E_1(L_0)$ (the kinetic energy of the frozen packet is unchanged).
    A fast compression, on the other hand, pushes on the particle and does work on it, so after a fast cycle the energy is much larger than at the start.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Basis truncation.</b> The coefficient vector is kept for $n=1\\dots N$ ($N=8$–$30$); $M$ is the full $N\\times N$ matrix above.</li>
      <li><b>Integrator.</b> Each sub-step of length $h$ is a symmetric (Strang) splitting: half a phase step $c_n\\to e^{-iE_n(L_{mid})h/2\\hbar}c_n$ (exact),
      a full coupling step, and another half phase step, with $L$ and $\\dot L$ evaluated at the mid-point of the sub-step.</li>
      <li><b>Coupling step.</b> $c\\to e^{aM}c$ with $a=-\\dot L h/L$ is applied by the Cayley transform
      $c\\to(I-\\tfrac a2M)^{-1}(I+\\tfrac a2M)\\,c$, which is <em>exactly</em> orthogonal for antisymmetric $M$. The linear system is solved by Gaussian
      elimination; no pivoting is needed because the symmetric part of $I-\\tfrac a2M$ is the identity.</li>
      <li><b>Time step.</b> $h=\\min\\big(T_{\\text{cycle}}/4000,\\;0.0015\\,L^2/(1+|\\dot L|L)\\big)$, i.e. finer when the well is narrow (fast phases) or the
      wall is fast. Each animation frame integrates the physical time $\\Delta t\\,T_{\\text{cycle}}/t_{play}$ with as many sub-steps as needed
      (frame budget ≈ 6 ms), so every speed plays the whole cycle in the chosen playback time.</li>
      <li><b>Plotted quantities.</b> $|\\psi(x)|^2$ is evaluated on 400 points from $\\psi=\\sqrt{2/L}\\sum c_n\\sin(n\\theta)$, $\\theta=\\pi x/L$, using the
      recurrence $\\sin(n+1)\\theta=2\\cos\\theta\\sin n\\theta-\\sin(n-1)\\theta$. The level diagram draws $E_n(L)$ with line thickness $\\propto\\sqrt{|c_n|^2}$.
      $\\langle E\\rangle=\\sum_n|c_n|^2E_n(L)$ is recorded against $t/T_{\\text{cycle}}$, together with the reference curve $E_1(L(t))$.</li>
      <li><b>Accuracy checks.</b> The norm $\\sum_n|c_n|^2$ (metric) stays at 1 to machine precision because every step is unitary; the trapezoidal
      integral $\\int|\\psi|^2dx$ in the density panel should also read 1 (it falls slightly below only if the state leaks into $n>N$).</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Adiabatic cycle.</b> Set $v_{max}\\approx0.03$ ($\\varepsilon_{max}\\approx0.02$). $|c_1|^2$ stays at 1.000, ⟨E⟩ lies on the dashed $E_1(L(t))$ curve
      (from $E_1(L_0)=\\pi^2/2\\approx4.93$ down to $E_1(L_{end})\\approx0.077$) and at the end of the cycle $\\langle E\\rangle/E_1(L_0)=1.000$.</li>
      <li><b>Intermediate regime.</b> At $v_{max}=0.3$ ($\\varepsilon\\approx0.2$) a few per cent leaks into $n=2,3$ and the cycle ends with
      $\\langle E\\rangle/E_1(L_0)\\approx1.08$. At the default $v=1$ ($\\varepsilon\\approx0.7$) only $|c_1|^2\\approx0.13$ survives and the ratio is ≈ 6.</li>
      <li><b>Sudden regime.</b> At $v_{max}=5$ ($\\varepsilon\\approx3.6$) the density stays squeezed near $x\\lt L_0$ while the wall runs away and ⟨E⟩ remains close to
      $E_1(L_0)$ during the expansion; the compression then pumps the energy up by two orders of magnitude.</li>
      <li><b>Compare cycles.</b> Keep "previous cycle" on and change only the speed: the purple ghost curve lets you compare two runs directly.</li>
      <li><b>Basis size.</b> In the sudden regime change $N$ from 8 to 30: the final energy changes, showing that a narrow, sharply cut wave function needs many
      basis states. The norm stays at 1.000000000 in all cases.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The walls are perfectly hard and the model is one-dimensional, non-relativistic and single-particle; the basis is truncated at $N$ states, which
    matters only for very fast walls. The coupled-coefficient method and the adiabatic theorem are covered in D. J. Griffiths &amp; D. F. Schroeter,
    <em>Introduction to Quantum Mechanics</em> (3rd ed.), ch. 11; J. J. Sakurai, <em>Modern Quantum Mechanics</em>, §5.6; and the moving-wall well in
    S. W. Doescher &amp; M. H. Rice, <em>Am. J. Phys.</em> 37, 1246 (1969).</p>`,

  mount(api) {
    const P = api.params, hbar = 1, m = 1, NX = 400;
    const plots = api.plots([
      { id: "dens", title: "Probability density |ψ(x,t)|² and the moving wall", aspect: 0.62, xlim: [-0.3, 8.3], ylim: [0, 3], xlabel: "position x", ylabel: "|ψ|²" },
      { id: "lev", title: "Instantaneous levels Eₙ(L) and ⟨E⟩ — log scale", aspect: 0.62, xlim: [-0.3, 8.3], ylim: [0.05, 400], ylog: true, xlabel: "position x", ylabel: "energy E" },
      { id: "pop", title: "Occupation probabilities |cₙ(t)|²", aspect: 0.62, xlim: [0.4, 8.6], ylim: [0, 1.12], xlabel: "level n", ylabel: "|cₙ|²",
        xtickFormat: (v) => (Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : "") },
      { id: "en", title: "Expectation energy ⟨E⟩(t) over the cycle — log scale", aspect: 0.62, xlim: [0, 1], ylim: [0.05, 50], ylog: true, xlabel: "normalised time t / T_cycle", ylabel: "⟨E⟩" },
    ]);
    const Mt = api.metrics([
      { id: "L", label: "Wall position $L(t)$" },
      { id: "E", label: "Expectation energy $\\langle E\\rangle$" },
      { id: "c1", label: "Ground state $|c_1|^2$" },
      { id: "eps", label: "Adiabaticity $\\varepsilon_{max}$" },
      { id: "norm", label: "Norm $\\sum_n|c_n|^2$" },
    ]);

    // ---- state
    let N, vmax, L0, Lt, Le, t1, t2, T, tSim, done, hold;
    let re, im, Mc, Aw, br, bi, pop, histT, histE, ghost = null, ghostV = 0, Emax, dMax, curV = 0;
    const xs = new Float64Array(NX), dens = new Float64Array(NX);
    const refT = new Float64Array(401), refE = new Float64Array(401);

    function wall(t) { // [L, dL/dt]
      if (t <= t1) return [L0 + vmax * t, vmax];
      if (t <= t1 + 2 * t2) { const tau = t - t1; return [Lt + (Le - Lt) * Math.sin((Math.PI * tau) / (2 * t2)), vmax * Math.cos((Math.PI * tau) / (2 * t2))]; }
      const tau = t - (t1 + 2 * t2);
      return [Lt - vmax * tau, -vmax];
    }
    const En = (n, L) => (n * n * Math.PI * Math.PI * hbar * hbar) / (2 * m * L * L);

    function buildCoupling() {
      // M[i,j] = (−1)^{n+m}·2nm/(n²−m²) with n=i+1, m=j+1 — full coupling for a single moving wall (all n≠m pairs)
      Mc = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const n = i + 1, q = j + 1;
        if (n !== q) Mc[i * N + j] = ((n + q) % 2 !== 0 ? 1 : -1) * (-(2 * n * q) / (n * n - q * q));
      }
      Aw = new Float64Array(N * N); br = new Float64Array(N); bi = new Float64Array(N); pop = new Float64Array(N);
    }
    // c_n ← e^{-i E_n h/ħ} c_n
    function phase(L, h) {
      const f = (Math.PI * Math.PI * hbar) / (2 * m * L * L) * h;
      for (let n = 0; n < N; n++) {
        const w = (n + 1) * (n + 1) * f, c = Math.cos(w), s = Math.sin(w), r = re[n], q = im[n];
        re[n] = r * c + q * s; im[n] = q * c - r * s;
      }
    }
    // c ← (I - a/2·M)^{-1} (I + a/2·M) c   (a = -v h / L) — orthogonal Cayley transform
    function cayley(a) {
      const hh = a / 2;
      for (let i = 0; i < N; i++) {
        let sr = re[i], si = im[i]; const row = i * N;
        for (let j = 0; j < N; j++) { const mm = Mc[row + j]; if (mm !== 0) { sr += hh * mm * re[j]; si += hh * mm * im[j]; } }
        br[i] = sr; bi[i] = si;
      }
      for (let k = 0; k < N * N; k++) Aw[k] = -hh * Mc[k];
      for (let i = 0; i < N; i++) Aw[i * N + i] = 1;
      // identity + antisymmetric ⇒ Gaussian elimination without pivoting is stable
      for (let k = 0; k < N; k++) {
        const p = Aw[k * N + k];
        for (let i = k + 1; i < N; i++) {
          const f = Aw[i * N + k] / p; if (f === 0) continue;
          for (let j = k; j < N; j++) Aw[i * N + j] -= f * Aw[k * N + j];
          br[i] -= f * br[k]; bi[i] -= f * bi[k];
        }
      }
      for (let i = N - 1; i >= 0; i--) {
        let sr = br[i], si = bi[i];
        for (let j = i + 1; j < N; j++) { const a2 = Aw[i * N + j]; sr -= a2 * re[j]; si -= a2 * im[j]; }
        re[i] = sr / Aw[i * N + i]; im[i] = si / Aw[i * N + i];
      }
    }
    function expectE(L) { let e = 0; for (let n = 0; n < N; n++) e += (re[n] * re[n] + im[n] * im[n]) * En(n + 1, L); return e; }

    function startCycle() {
      if (histT && histT.length > 5) { ghost = { t: histT, e: histE }; ghostV = vmax; }
      re = new Float64Array(N); im = new Float64Array(N); re[0] = 1;
      tSim = 0; done = false; hold = 0;
      histT = [0]; histE = [En(1, L0)];
      Emax = En(1, L0); dMax = (2 / L0) * 1.3;
      let gmax = 0; if (ghost && P.showGhost) for (const e of ghost.e) gmax = Math.max(gmax, e);
      plots.en.setLimits(null, [0.5 * En(1, Le), Math.max(3 * En(1, L0), 2 * En(2, L0), 1.5 * gmax)]);
      plots.dens.setLimits(null, [0, Math.max(2.4, dMax)]);
    }

    function reset() {
      N = P.N; vmax = Math.pow(10, P.logv); L0 = P.L0; Lt = P.Lturn; Le = P.Lend;
      if (Lt < L0 + 0.5) Lt = L0 + 0.5;
      if (Le < Lt + 0.5) Le = Lt + 0.5;
      t1 = (Lt - L0) / vmax; t2 = (Math.PI * (Le - Lt)) / (2 * vmax); T = 2 * (t1 + t2);
      buildCoupling();
      startCycle();
      for (let k = 0; k <= 400; k++) { refT[k] = k / 400; refE[k] = En(1, wall((k / 400) * T)[0]); }
      plots.dens.setLimits([-0.3, Le + 0.3]);
      plots.lev.setLimits([-0.3, Le + 0.3]);
      const eps = (8 * m * vmax * Le) / (9 * Math.PI * Math.PI * hbar);
      Mt.set("eps", PM.fmt(eps, 3));
      const reg = eps < 0.1 ? "🐢 <b>Adiabatic regime</b>: the particle follows the ground state and the cycle is reversible."
        : eps > 1 ? "⚡ <b>Sudden (non-adiabatic) regime</b>: the wall outruns the wave function; levels mix and the cycle is irreversible."
          : "〰️ <b>Transition regime</b>: partial excitation.";
      const tx = (x) => PM.fmt(x, 2);
      api.status(`${reg} &nbsp; $T_{\\text{cycle}}=${tx(T)}$, $t_1=${tx(t1)}$, $t_2=${tx(t2)}$ (ħ = m = 1).`);
    }

    function advance(target) {
      const clock = performance.now();
      let k = 0;
      while (tSim < target - 1e-12) {
        const [L, v] = wall(tSim);
        let h = Math.min(T / 4000, (0.0015 * L * L) / (1 + Math.abs(v) * L));
        if (tSim + h > target) h = target - tSim;
        const [Lm, vm] = wall(tSim + h / 2);
        phase(Lm, h / 2); cayley((-vm / Lm) * h); phase(Lm, h / 2);
        tSim += h;
        if ((++k & 15) === 0 && performance.now() - clock > 6) break; // per-frame time budget
      }
    }

    function fillDensity(L) {
      // ψ(x) = Σ c_n √(2/L) sin(nπx/L) — sin(nθ) via the three-term recurrence
      const A = Math.sqrt(2 / L);
      for (let j = 0; j < NX; j++) {
        const x = (L * j) / (NX - 1), th = (Math.PI * x) / L, c2 = 2 * Math.cos(th);
        let s0 = 0, s1 = Math.sin(th), pr = 0, pi = 0;
        for (let n = 0; n < N; n++) { pr += re[n] * s1; pi += im[n] * s1; const s2 = c2 * s1 - s0; s0 = s1; s1 = s2; }
        xs[j] = x; dens[j] = A * A * (pr * pr + pi * pi);
      }
    }

    function lerpCol(k, n) {
      const a = [79, 209, 197], b = [245, 158, 11], t = n > 1 ? k / (n - 1) : 0;
      return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(a[1] + (b[1] - a[1]) * t)},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
    }

    api.setControl("Lturn", { min: P.L0 + 0.5 });
    api.setControl("Lend", { min: P.Lturn + 0.5 });

    return {
      reset,
      onParam(id, v) {
        if (id === "L0") {
          api.setControl("Lturn", { min: v + 0.5, value: Math.max(P.Lturn, v + 0.5) });
          api.setControl("Lend", { min: P.Lturn + 0.5, value: Math.max(P.Lend, P.Lturn + 0.5) });
        } else if (id === "Lturn") {
          api.setControl("Lend", { min: v + 0.5, value: Math.max(P.Lend, v + 0.5) });
        }
      },
      step(dt) {
        if (done) { hold += dt; if (hold > 2.5) startCycle(); return; }
        advance(Math.min(tSim + (dt * T) / P.tplay, T));
        const [L, v] = wall(tSim); curV = v;
        const e = expectE(L);
        histT.push(tSim / T); histE.push(e);
        if (tSim >= T - 1e-9) done = true;
        if (e > Emax) {
          Emax = e;
          const yl = plots.en.ylim;
          if (e > yl[1] * 0.8) plots.en.setLimits(null, [yl[0], e * 2]);
        }
      },
      render() {
        const [L, v0] = wall(tSim), v = tSim > 0 ? curV : v0;
        fillDensity(L);
        const nshow = Math.min(P.nshow, N);
        let nrm = 0; for (let n = 0; n < N; n++) { pop[n] = re[n] * re[n] + im[n] * im[n]; nrm += pop[n]; }
        const eNow = expectE(L);
        const W = PlotColors.white;

        // --- density
        const pd = plots.dens;
        const dm = PM.max(dens);
        if (dm > pd.ylim[1] * 0.97) { pd.setLimits(null, [0, dm * 1.25]); }
        pd.clear();
        const top = pd.ylim[1];
        pd.rect(L, 0, Le + 1, top * 2, { color: PlotColors.muted, alpha: 0.12 });
        pd.rect(-1, 0, 0, top * 2, { color: PlotColors.muted, alpha: 0.12 });
        pd.fill(xs, dens, 0, { color: PlotColors.accent, alpha: 0.45 });
        pd.line(xs, dens, { color: PlotColors.accent, width: 2 });
        pd.segment(0, 0, 0, top * 2, { color: W, width: 4 });
        pd.segment(L, 0, L, top * 2, { color: W, width: 4 });
        if (Math.abs(v) > 1e-9) {
          const len = (0.06 + 0.1 * Math.abs(v) / vmax) * (Le + 0.6) * Math.sign(v);
          pd.arrow(L, top * 0.9, L + len, top * 0.9, { color: PlotColors.accent3, width: 2.2 });
        }
        let area = 0; const dxs = L / (NX - 1);
        for (let j = 1; j < NX; j++) area += 0.5 * (dens[j] + dens[j - 1]) * dxs;
        pd.label([`L = ${PM.fmt(L, 3)},  dL/dt = ${PM.fmt(v, 3)}`, `∫|ψ|²dx = ${PM.fmt(area, 4)}`], "tr");

        // --- levels
        const pl = plots.lev;
        pl.setLimits(null, [0.6 * En(1, Le), 1.6 * En(nshow, L0)]);
        const ytop = pl.ylim[1], ybot = pl.ylim[0];
        pl.clear();
        pl.rect(L, ybot, Le + 1, ytop * 2, { color: PlotColors.muted, alpha: 0.12 });
        for (let n = nshow; n >= 1; n--) {
          const E = En(n, L); if (E > ytop * 1.05) continue;
          const p = pop[n - 1], col = lerpCol(n - 1, nshow);
          pl.segment(0, E, L, E, { color: col, width: 1 + 5 * Math.sqrt(p), alpha: 0.35 + 0.65 * Math.sqrt(p), dash: p < 0.01 ? [5, 4] : [] });
          if (p >= 0.01) pl.text(L, E, `n=${n}: ${PM.fmt(p, 2)} `, { color: col, size: 10.5, align: "right", dx: -6, dy: -8 });
        }
        if (eNow <= ytop) {
          pl.segment(0, eNow, L, eNow, { color: W, width: 1.8, dash: [7, 4] });
          pl.text(0.05, eNow, "⟨E⟩", { dy: -9, color: W, size: 11.5, bold: true });
        } else pl.label(`⟨E⟩ = ${PM.fmt(eNow, 2)} ↑ (off scale)`, "tl", { color: PlotColors.accent3 });
        pl.segment(0, ybot, 0, ytop * 2, { color: W, width: 4 });
        pl.segment(L, ybot, L, ytop * 2, { color: W, width: 4 });

        // --- populations
        const pp = plots.pop;
        pp.setLimits([0.4, nshow + 0.6]);
        pp.clear();
        const ns = [], hs = [], cols = [];
        for (let n = 1; n <= nshow; n++) { ns.push(n); hs.push(pop[n - 1]); cols.push(lerpCol(n - 1, nshow)); }
        pp.bars(ns, hs, 0.55, { colors: cols, alpha: 0.85 });
        for (let n = 1; n <= nshow; n++) if (pop[n - 1] > 0.01) pp.text(n, pop[n - 1], PM.fmt(pop[n - 1], 2), { dy: -8, align: "center", size: 10 });
        let rest = 0; for (let n = nshow; n < N; n++) rest += pop[n];
        pp.label(`n > ${nshow}: ${PM.fmt(rest, 3)}`, "tr", { color: PlotColors.muted, size: 11 });

        // --- ⟨E⟩(t)
        const pe = plots.en;
        pe.clear();
        pe.vline(0.5, { color: PlotColors.muted, dash: [3, 4] });
        if (P.showRef) pe.line(refT, refE, { color: PlotColors.muted, width: 1.4, dash: [6, 4] });
        if (P.showGhost && ghost) pe.line(ghost.t, ghost.e, { color: PlotColors.accent2, width: 1.6, alpha: 0.75 });
        pe.line(histT, histE, { color: PlotColors.accent3, width: 2.4 });
        pe.points([tSim / T], [eNow], { color: PlotColors.accent3, size: 4.5 });
        const leg = [{ label: `v = ${PM.fmt(vmax, 3)} (current)`, color: PlotColors.accent3 }];
        if (P.showGhost && ghost) leg.push({ label: `v = ${PM.fmt(ghostV, 3)} (previous cycle)`, color: PlotColors.accent2 });
        if (P.showRef) leg.push({ label: "adiabatic: E₁(L(t))", color: PlotColors.muted, dash: [6, 4] });
        pe.legend(leg, "tr");
        if (done) pe.label(`End of cycle: ⟨E⟩/E₁(L₀) = ${PM.fmt(eNow / En(1, L0), 3)}`, "bl", { color: PlotColors.accent3 });

        Mt.set("L", PM.fmt(L, 3));
        Mt.set("E", PM.fmt(eNow, 4));
        Mt.set("c1", PM.fmt(pop[0], 4));
        Mt.set("norm", nrm.toFixed(9));
        api.setTime(`t/T = ${PM.fmt(tSim / T, 3)}  (t = ${PM.fmt(tSim, 2)})`);
      },
    };
  },
});
