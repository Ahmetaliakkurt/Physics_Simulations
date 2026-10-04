/* Adiabatic ↔ sudden map: excitation probability out of the ground state of an expanding infinite well
 * as a function of wall speed. The speed sweep is integrated live in small chunks per frame (the UI never freezes). */
App.register({
  id: "adiabatic-sudden-map",
  category: "quantum",
  order: 36,
  title: "Adiabatic vs Sudden Map",
  icon: "🔀",
  subtitle: "Excitation probability out of the ground state of an expanding infinite well versus wall speed: the crossover from the adiabatic regime (P ∝ v²) to the sudden regime.",
  notes: [
    { type: "info", html: "The wall of the box opens from $L_0$ to $L_{end}$ at a constant speed $v$. For every speed the coefficients are integrated from start to finish and the final $P_{exc}=1-|c_1|^2$ is recorded. The sweep runs from the fast end to the slow end and the curve is drawn while you watch. As $v\\to0$ the adiabatic theorem keeps the system in the ground state; at large $v$ the wave function cannot keep up with the wall." },
  ],
  animated: true,
  speed: { min: 0.5, max: 2, value: 1, step: 0.1 },
  controls: [
    { id: "N", type: "slider", label: "Number of basis states $N$", min: 5, max: 30, step: 1, value: 20 },
    { id: "L0", type: "slider", label: "Initial width $L_0$", min: 0.5, max: 5, step: 0.1, value: 1 },
    { id: "Lend", type: "slider", label: "Final width $L_{end}$", min: 2, max: 20, step: 0.5, value: 10 },
    { type: "section", label: "Speed sweep" },
    { id: "nv", type: "slider", label: "Number of speeds", min: 10, max: 60, step: 5, value: 30 },
    { id: "lvmin", type: "slider", label: "Slowest speed $v_{min}$", min: -3, max: -0.5, step: 0.05, value: Math.log10(0.005),
      fmt: (v) => Math.pow(10, v).toPrecision(2) },
    { id: "lvmax", type: "slider", label: "Fastest speed $v_{max}$", min: 0, max: 2.7, step: 0.05, value: 2,
      fmt: (v) => Math.pow(10, v).toPrecision(3) },
    { id: "acc", type: "select", label: "Time-step accuracy", value: "0.002",
      options: [{ value: "0.004", label: "Normal (fast)" }, { value: "0.002", label: "High" }, { value: "0.001", label: "Very high (slow)" }] },
    { type: "section", label: "Display" },
    { id: "logy", type: "checkbox", label: "Logarithmic vertical axis", value: false, live: true,
      help: "Makes the tiny excitations of the adiabatic regime (∝ v²) visible." },
    { id: "showTr", type: "checkbox", label: "Show the traces $P(L)$ of all speeds", value: true, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A particle of mass $m$ sits in the ground state of a one-dimensional infinite square well $0\\le x\\le L$ of initial width $L_0$.
    At $t=0$ the right wall starts moving outward at constant speed $v$, $L(t)=L_0+vt$, and stops at $L_{end}$ after $T=(L_{end}-L_0)/v$.
    The question is simple: what is the probability $P_{exc}=1-|c_1|^2$ that the particle is <em>not</em> in the ground state of the final well?
    The simulation answers it for $N_v$ logarithmically spaced speeds between $v_{min}$ and $v_{max}$. Units: $\\hbar=m=1$, lengths in an arbitrary unit $\\ell$,
    speeds in $\\hbar/m\\ell$.</p>

    <h4>Equations being solved</h4>
    <p>Expanding $\\psi(x,t)=\\sum_n c_n(t)\\,\\phi_n(x;L(t))$ in the instantaneous eigenstates
    $\\phi_n=\\sqrt{2/L}\\,\\sin(n\\pi x/L)$, $E_n(L)=n^2\\pi^2\\hbar^2/2mL^2$, and projecting the Schrödinger equation onto $\\phi_n$ gives</p>
    <div class="callout">
    $$\\dot c_n=-\\frac{i}{\\hbar}E_n(L)\\,c_n-\\frac{v}{L}\\sum_{m\\ne n}M_{nm}\\,c_m,\\qquad M_{nm}=(-1)^{n+m}\\,\\frac{2nm}{n^2-m^2}$$
    </div>
    <p>Here $M_{nm}/L=\\langle\\phi_n|\\partial_L\\phi_m\\rangle$, obtained from $\\int_0^\\pi u\\sin(nu)\\cos(mu)\\,du=\\pi(-1)^{n+m+1}n/(n^2-m^2)$; the diagonal
    elements vanish. For a single moving wall <em>all</em> pairs $n\\ne m$ are coupled. $M$ is real antisymmetric, so the evolution is unitary.</p>
    <p><b>Adiabatic regime: $P_{exc}\\propto v^2$.</b> First-order adiabatic perturbation theory gives
    $c_n(T)\\approx-\\int_0^T\\frac{v}{L}M_{n1}\\,e^{i\\int_0^t\\omega_{n1}dt'}dt$ with $\\omega_{n1}=(E_n-E_1)/\\hbar$. Integrating by parts, the contributions come from
    the moments when the wall starts and stops abruptly:</p>
    $$|c_2(T)|^2\\approx\\big|\\varepsilon(L_0)-\\varepsilon(L_{end})\\,e^{i\\Phi}\\big|^2,\\qquad \\varepsilon(L)=\\frac{\\hbar\\,v|M_{21}|/L}{E_2-E_1}=\\frac{8\\,m\\,v\\,L}{9\\pi^2\\hbar},$$
    <p>with $\\Phi$ the accumulated dynamical phase. Hence $P_{exc}\\approx\\varepsilon(L_0)^2+\\varepsilon(L_{end})^2-2\\varepsilon(L_0)\\varepsilon(L_{end})\\cos\\Phi\\propto v^2$:
    a straight line of slope 2 on a log–log plot (with small interference wiggles). The crossover $P_{exc}\\approx\\tfrac12$ happens when $\\varepsilon(L_{end})\\sim1$.</p>
    <p><b>Sudden regime.</b> For $v\\to\\infty$ the wave function is frozen while the wall jumps, so</p>
    $$P_{exc}\\to1-\\big|\\langle\\phi_1(L_{end})|\\phi_1(L_0)\\rangle\\big|^2,$$
    <p>which is 0.996 for $L_0=1$, $L_{end}=10$ and 0.904 for $L_0=3$, $L_{end}=10$.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Basis.</b> The coefficient vector is truncated to $n=1\\dots N$ and the full $N\\times N$ coupling matrix is built once.</li>
      <li><b>Integrator.</b> Symmetric (Strang) splitting per step $h$: exact half phase step $e^{-iE_n(L_{mid})h/2\\hbar}$, then the coupling step
      $c\\to(I-\\tfrac a2M)^{-1}(I+\\tfrac a2M)c$ with $a=-vh/L_{mid}$ (Cayley transform, exactly orthogonal, solved by Gaussian elimination), then another half phase step.</li>
      <li><b>Step size.</b> $h=\\min\\big(T/400,\\;f\\,L^2/(1+vL)\\big)$ with $f$ set by the accuracy selector (0.004 / 0.002 / 0.001). Because $E_n\\propto1/L^2$, this keeps the
      phase error per step roughly constant as the box grows.</li>
      <li><b>Scheduling.</b> Each animation frame spends about 5 ms integrating, then draws. Speeds are processed from fastest (short $T$) to slowest
      (long $T$), so the sudden half of the map appears almost at once and the adiabatic tail fills in afterwards.</li>
      <li><b>Plotted quantities.</b> The top panel shows $P_{exc}(v)$ at $L=L_{end}$; the background bands mark the slowest factor of 20,
      $[v_{min},20v_{min}]$ (adiabatic, teal), the fastest $[v_{max}/20,v_{max}]$ (sudden, red) and the transition in between. The left panel shows
      $1-|c_1(L)|^2$ along each run (colour = speed), the right panel the final $|c_n|^2$ of the most recently finished speed.</li>
      <li><b>Metrics.</b> $v_{1/2}$ is found by log-linear interpolation where $P_{exc}$ crosses 0.5. The adiabatic slope is a least-squares fit of
      $\\ln P$ versus $\\ln v$ over all points with $10^{-9}\\lt P\\lt 0.05$; it should be close to 2.</li>
      <li><b>Accuracy check.</b> The Cayley/phase steps are unitary, so $\\sum|c_n|^2=1$ to round-off; refining the accuracy selector should not move the curve.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Default sweep</b> ($L_0=1$, $L_{end}=10$, $N=20$): the slope metric comes out ≈ 2, $v_{1/2}\\approx0.8$–$1$, and at the slowest speed $v=0.005$
      $P_{exc}\\approx2\\times10^{-5}$, consistent with $\\varepsilon(L_{end})^2=(8\\cdot0.005\\cdot10/9\\pi^2)^2\\approx2\\times10^{-5}$.</li>
      <li><b>Log axis.</b> Tick "logarithmic vertical axis" to see the straight $v^2$ line and the small oscillations from the $\\cos\\Phi$ interference term.</li>
      <li><b>Sudden plateau.</b> Set $L_0=3$: at high speed the curve saturates at $1-|\\langle\\phi_1(10)|\\phi_1(3)\\rangle|^2\\approx0.90$ instead of 0.996.</li>
      <li><b>Finite-basis artefact.</b> Reduce $N$ to 5 and push $v_{max}$ to 500: the curve <em>falls</em> again at very high speed (to ≈ 0.4).
      This is not physics: the frozen wave function, squeezed into $x\\lt L_0$ of a box ten times wider, needs many basis states; with $N=20$–30 the
      plateau stays at ≈ 0.99.</li>
      <li><b>Scaling with size.</b> Doubling $L_{end}$ roughly halves $v_{1/2}$, because the adiabaticity parameter grows as $vL$.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The model is 1D with ideal hard walls and an abrupt start and stop of the wall; a smooth velocity ramp would suppress the boundary terms and
    make $P_{exc}$ fall much faster than $v^2$. The basis truncation limits the sudden regime. See D. J. Griffiths &amp; D. F. Schroeter,
    <em>Introduction to Quantum Mechanics</em> (3rd ed.), ch. 11; A. Messiah, <em>Quantum Mechanics</em> vol. II, ch. XVII (adiabatic and sudden approximations).</p>`,

  mount(api) {
    const P = api.params, hbar = 1, m = 1, NTR = 120;
    const plots = api.plots([
      { id: "map", title: "Excitation probability 1 − |c₁|² versus expansion speed", span: 2, aspect: 0.4, xlog: true, xlim: [0.005, 100], ylim: [-0.05, 1.05],
        xlabel: "expansion speed v (log scale)", ylabel: "excitation probability" },
      { id: "tr", title: "1 − |c₁(L)|² during the expansion (colour: slow → fast)", aspect: 0.62, xlim: [1, 10], ylim: [0, 1.05], xlabel: "box width L", ylabel: "1 − |c₁|²" },
      { id: "pop", title: "|cₙ|² at L_end (most recent speed)", aspect: 0.62, xlim: [0.4, 10.6], ylim: [0, 1.08], xlabel: "level n", ylabel: "|cₙ|²" },
    ]);
    const Mt = api.metrics([
      { id: "prog", label: "Progress" },
      { id: "vnow", label: "Speed being computed $v$" },
      { id: "vhalf", label: "Crossover speed $v_{1/2}$" },
      { id: "slope", label: "Adiabatic slope $d\\ln P/d\\ln v$" },
      { id: "ms", label: "Compute time" },
    ]);

    let N, L0, Le, fac, vs, res, traces, order, oi, run, Mc, Aw, br, bi, re, im, lastPop, elapsed, doneAll;

    function build() {
      // full coupling for a single moving wall: all n≠m pairs, sign (−1)^{n+m}
      Mc = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const n = i + 1, q = j + 1;
        if (n !== q) Mc[i * N + j] = ((n + q) % 2 !== 0 ? 1 : -1) * (-(2 * n * q) / (n * n - q * q));
      }
      Aw = new Float64Array(N * N); br = new Float64Array(N); bi = new Float64Array(N);
      re = new Float64Array(N); im = new Float64Array(N);
    }
    function phase(L, h) {
      const f = ((Math.PI * Math.PI * hbar) / (2 * m * L * L)) * h;
      for (let n = 0; n < N; n++) {
        const w = (n + 1) * (n + 1) * f, c = Math.cos(w), s = Math.sin(w), r = re[n], q = im[n];
        re[n] = r * c + q * s; im[n] = q * c - r * s;
      }
    }
    function cayley(a) { // c ← (I − a/2 M)⁻¹ (I + a/2 M) c
      const hh = a / 2;
      for (let i = 0; i < N; i++) {
        let sr = re[i], si = im[i]; const row = i * N;
        for (let j = 0; j < N; j++) { const mm = Mc[row + j]; if (mm !== 0) { sr += hh * mm * re[j]; si += hh * mm * im[j]; } }
        br[i] = sr; bi[i] = si;
      }
      for (let k = 0; k < N * N; k++) Aw[k] = -hh * Mc[k];
      for (let i = 0; i < N; i++) Aw[i * N + i] = 1;
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
    function startRun(k) {
      re.fill(0); im.fill(0); re[0] = 1;
      const v = vs[k];
      run = { k, v, t: 0, T: (Le - L0) / v, trL: [L0], trP: [0], nextL: L0 + (Le - L0) / NTR };
    }
    function advanceRun(budgetEnd) {
      const r = run, v = r.v; let cnt = 0;
      while (r.t < r.T - 1e-12) {
        const L = L0 + v * r.t;
        let h = Math.min(r.T / 400, (fac * L * L) / (1 + v * L));
        if (r.t + h > r.T) h = r.T - r.t;
        const Lm = L0 + v * (r.t + h / 2);
        phase(Lm, h / 2); cayley((-v / Lm) * h); phase(Lm, h / 2);
        r.t += h;
        const Ln = L0 + v * r.t;
        if (Ln >= r.nextL - 1e-12 || r.t >= r.T - 1e-12) { r.trL.push(Ln); r.trP.push(Math.max(0, 1 - (re[0] * re[0] + im[0] * im[0]))); r.nextL += (Le - L0) / NTR; }
        if ((++cnt & 15) === 0 && performance.now() > budgetEnd) return false;
      }
      return true;
    }

    function reset() {
      N = P.N; L0 = P.L0; Le = Math.max(P.Lend, L0 + 1); fac = parseFloat(P.acc);
      const a = P.lvmin, b = Math.max(P.lvmax, a + 0.5), n = P.nv;
      vs = []; for (let k = 0; k < n; k++) vs.push(Math.pow(10, a + ((b - a) * k) / (n - 1)));
      res = new Float64Array(n).fill(NaN); traces = new Array(n).fill(null);
      order = []; for (let k = n - 1; k >= 0; k--) order.push(k); // fast → slow
      oi = 0; elapsed = 0; doneAll = false; lastPop = null;
      build(); startRun(order[0]);
      plots.map.setLimits([vs[0], vs[n - 1]]);
      plots.tr.setLimits([L0, Le]);
      plots.pop.setLimits([0.4, Math.min(N, 10) + 0.6]);
      api.play();
    }

    function slopeFit() { // least-squares slope of log P vs log v in the adiabatic region
      let sx = 0, sy = 0, sxx = 0, sxy = 0, c = 0;
      for (let k = 0; k < vs.length; k++) {
        const p = res[k];
        if (!(p > 1e-9 && p < 0.05)) continue;
        const x = Math.log(vs[k]), y = Math.log(p);
        sx += x; sy += y; sxx += x * x; sxy += x * y; c++;
      }
      return c >= 3 ? (c * sxy - sx * sy) / (c * sxx - sx * sx) : NaN;
    }
    function vHalf() {
      for (let k = 1; k < vs.length; k++) {
        const a = res[k - 1], b = res[k];
        if (isFinite(a) && isFinite(b) && a < 0.5 && b >= 0.5) {
          const f = (0.5 - a) / (b - a);
          return Math.exp(Math.log(vs[k - 1]) + f * (Math.log(vs[k]) - Math.log(vs[k - 1])));
        }
      }
      return NaN;
    }

    return {
      reset,
      onParam(id, v) {
        if (id === "L0") api.setControl("Lend", { min: v + 1, value: Math.max(P.Lend, v + 1) });
        if (id === "lvmin") api.setControl("lvmax", { min: Math.max(0, v + 0.5) });
      },
      step() {
        if (doneAll) return;
        const t0 = performance.now(), end = t0 + 5; // ≈ 5 ms of integration per frame
        while (performance.now() < end && !doneAll) {
          if (advanceRun(end)) {
            const k = run.k;
            res[k] = Math.max(0, 1 - (re[0] * re[0] + im[0] * im[0]));
            traces[k] = { L: Float64Array.from(run.trL), p: Float64Array.from(run.trP) };
            lastPop = { v: run.v, p: Array.from({ length: N }, (_, n) => re[n] * re[n] + im[n] * im[n]) };
            oi++;
            if (oi >= order.length) doneAll = true; else startRun(order[oi]);
          }
        }
        elapsed += performance.now() - t0;
      },
      render() {
        const n = vs.length, pm = plots.map;
        const logy = !!P.logy;
        if (pm.opts.ylog !== logy) { pm.opts.ylog = logy; pm.setLimits(null, logy ? [1e-7, 1.5] : [-0.05, 1.05]); }
        pm.clear();
        const y0 = logy ? 1e-7 : -1, y1 = logy ? 10 : 2;
        const a = vs[0], b = vs[n - 1], s1 = Math.min(a * 20, b), s2 = Math.max(b / 20, s1);
        pm.rect(a, y0, s1, y1, { color: PlotColors.accent, alpha: 0.1 });
        pm.rect(s1, y0, s2, y1, { color: "#f5d742", alpha: 0.08 });
        pm.rect(s2, y0, b, y1, { color: PlotColors.bad, alpha: 0.1 });
        const xs = [], ys = [];
        for (let k = 0; k < n; k++) if (isFinite(res[k])) { xs.push(vs[k]); ys.push(logy ? Math.max(res[k], 1e-12) : res[k]); }
        if (xs.length) {
          pm.line(xs, ys, { color: PlotColors.accent3, width: 2 });
          pm.points(xs, ys, { color: PlotColors.accent3, size: 3.5, stroke: "#0f151c" });
        }
        if (!doneAll) {
          pm.vline(run.v, { color: PlotColors.text, dash: [4, 4], alpha: 0.7 });
          const pc = Math.max(0, 1 - (re[0] * re[0] + im[0] * im[0]));
          pm.points([run.v], [logy ? Math.max(pc, 1e-12) : pc], { color: PlotColors.text, size: 4 });
        }
        const vh = vHalf();
        if (isFinite(vh)) pm.vline(vh, { color: PlotColors.accent2, dash: [6, 4], width: 1.4 });
        pm.legend([
          { label: "1 − |c₁|² (L = L_end)", color: PlotColors.accent3 },
          { label: "adiabatic regime", color: PlotColors.accent, type: "box" },
          { label: "transition", color: "#f5d742", type: "box" },
          { label: "sudden regime", color: PlotColors.bad, type: "box" },
        ].concat(isFinite(vh) ? [{ label: "v₁/₂", color: PlotColors.accent2, dash: [6, 4] }] : []), logy ? "br" : "tl");

        // traces
        const pt = plots.tr;
        pt.clear();
        const la = Math.log(a), lb = Math.log(b);
        if (P.showTr) for (let k = 0; k < n; k++) if (traces[k]) {
          pt.line(traces[k].L, traces[k].p, { color: colormap("viridis", 0.1 + 0.85 * (Math.log(vs[k]) - la) / (lb - la || 1)), width: 1.2, alpha: 0.75 });
        }
        if (!doneAll && run.trL.length > 1) pt.line(run.trL, run.trP, { color: PlotColors.text, width: 2.4 });
        pt.label(doneAll ? "Sweep complete" : `computing: v = ${PM.fmt(run.v, 3)}`, "tl");

        // populations
        const pp = plots.pop;
        pp.clear();
        if (lastPop) {
          const nn = Math.min(N, 10), X = [], H = [];
          for (let q = 0; q < nn; q++) { X.push(q + 1); H.push(lastPop.p[q]); }
          pp.bars(X, H, 0.55, { color: PlotColors.accent2, alpha: 0.85 });
          for (let q = 0; q < nn; q++) if (H[q] > 0.01) pp.text(q + 1, H[q], PM.fmt(H[q], 2), { dy: -8, align: "center", size: 10 });
          pp.label(`v = ${PM.fmt(lastPop.v, 3)}`, "tr");
        }

        const done = res.reduce((s, x) => s + (isFinite(x) ? 1 : 0), 0);
        Mt.set("prog", `${done} / ${n}` + (doneAll ? " ✓" : ""));
        Mt.set("vnow", doneAll ? "—" : PM.fmt(run.v, 3));
        Mt.set("vhalf", isFinite(vh) ? PM.fmt(vh, 3) : "—");
        const sl = slopeFit();
        Mt.set("slope", isFinite(sl) ? PM.fmt(sl, 2) : "—");
        Mt.set("ms", PM.fmt(elapsed / 1000, 2) + " s");
        api.setTime(doneAll ? "complete" : `computing… ${Math.round((100 * done) / n)}%`);
      },
    };
  },
});
