/* Superposition of plane waves — Fourier synthesis of N plane waves with random k and amplitudes.
 * The sum is kept incrementally (phasor recursion), so the N slider updates instantly even for thousands of waves. */
App.register({
  id: "wave-superposition",
  category: "quantum",
  order: 39,
  title: "Superposition of Plane Waves",
  icon: "➕",
  subtitle: "Fourier synthesis of many plane waves with random wavenumbers and amplitudes: watch a localised wave packet emerge from interference, then move and spread.",
  notes: [
    { type: "info", html: "A wave packet is a sum of many plane waves whose wavenumbers are spread by $\\sigma_k$ around $k_0$. As more waves are added, destructive interference cancels the wave almost everywhere and only the constructive region around $x\\approx0$ survives. Use <b>Add waves one by one</b> to watch the localisation form, and <b>Time evolution</b> to see the packet travel at the group velocity and spread." },
  ],
  animated: true,
  speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
  controls: [
    { id: "mode", type: "select", label: "Mode", value: "build", live: true,
      options: [{ value: "static", label: "Static: N waves from the slider" }, { value: "build", label: "Add waves one by one (animation)" }, { value: "evolve", label: "Time evolution ψ(x,t)" }] },
    { id: "N", type: "slider", label: "Number of plane waves $N$", min: 1, max: 3000, step: 1, value: 3000, live: true,
      help: "In the build-up mode this is the target number of waves." },
    { id: "k0", type: "slider", label: "Central wavenumber $k_0$", min: 1, max: 15, step: 0.5, value: 5, live: true },
    { id: "sk", type: "slider", label: "Wavenumber spread $\\sigma_k$", min: 0.05, max: 3, step: 0.05, value: 0.5, live: true },
    { id: "sA", type: "slider", label: "Amplitude spread $\\sigma_A$", min: 0, max: 1, step: 0.05, value: 0.2, live: true },
    { id: "seed", type: "number", label: "Random seed", min: 0, max: 9999, step: 1, value: 123, live: true },
    { id: "rate", type: "slider", label: "Build-up speed", min: 0.2, max: 3, step: 0.1, value: 1, live: true, visibleIf: (p) => p.mode === "build",
      help: "N(t) ≈ exp(0.6 · speed · t): the first waves are added one at a time, later ones ever faster." },
    { type: "section", label: "Display" },
    { id: "env", type: "checkbox", label: "Show the $N\\to\\infty$ limit envelope", value: true, live: true },
    { id: "im", type: "checkbox", label: "Also show Im ψ", value: false, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A free non-relativistic particle in one dimension, described not by a closed formula but by explicitly adding up $N$ plane waves
    $e^{i(k_jx-\\omega_jt)}$ — the Fourier picture of a wave packet. The wavenumbers $k_j$ are random numbers from a normal distribution with mean
    $k_0$ and standard deviation $\\sigma_k$; the amplitudes $A_j=|a_j|$ with $a_j$ normal of mean 1 and spread $\\sigma_A$. All waves start in phase
    at $x=0$. Units: $\\hbar=m=1$, so $\\omega=k^2/2$; the window is $-15\\le x\\le15$. The random numbers come from a seeded generator,
    so a given seed always gives the same set of waves.</p>

    <h4>Equations being solved</h4>
    $$\\psi_N(x,t) = \\frac{1}{\\sum_j A_j}\\sum_{j=1}^{N} A_j\\, e^{i(k_j x-\\omega_j t)},\\qquad
      k_j \\sim \\mathcal{N}(k_0, \\sigma_k^2),\\quad A_j=|\\mathcal N(1,\\sigma_A^2)|,\\quad \\omega_j=\\frac{\\hbar k_j^2}{2m}.$$
    <p>Normalising by $\\sum A_j$ makes $\\psi(0,0)=1$ (all waves in phase at the origin). For $N\\to\\infty$ the sum becomes an average over the
    $k$ distribution — its characteristic function — which for a Gaussian is again a Gaussian; including the dispersion $\\omega=k^2/2$:</p>
    <div class="callout">$$\\psi_\\infty(x,0)= e^{ik_0x}\\,e^{-\\sigma_k^2x^2/2},\\qquad
      |\\psi_\\infty(x,t)|=\\frac{1}{(1+\\sigma_k^4t^2)^{1/4}}\\exp\\!\\left[-\\frac{\\sigma_k^2(x-k_0t)^2}{2(1+\\sigma_k^4t^2)}\\right],\\qquad
      \\Delta x\\,\\Delta k\\ge\\tfrac12 .$$</div>
    <p>Here $\\Delta x=1/(\\sqrt2\\,\\sigma_k)$ is the standard deviation of $|\\psi|^2$ and $\\Delta k=\\sigma_k/\\sqrt2$ that of $|\\phi(k)|^2$ (the
    amplitude distribution has width $\\sigma_k$, its square $\\sigma_k/\\sqrt2$), so the Gaussian saturates the bound $\\Delta x\\,\\Delta k=1/2$ at $t=0$ —
    the Fourier-transform uncertainty relation that becomes Heisenberg's $\\Delta x\\,\\Delta p\\ge\\hbar/2$ with $p=\\hbar k$. Narrow in $k$ means wide in
    $x$ and vice versa. The centre moves at the group velocity $d\\omega/dk=k_0$ and the width grows as $\\Delta x(t)=\\Delta x(0)\\sqrt{1+\\sigma_k^4t^2}$.</p>
    <p><b>Random phases and the noise floor.</b> Away from the packet the waves have effectively random phases, so their sum is a 2D random walk:
    its typical size is $\\sqrt{\\sum A_j^2}$ instead of $\\sum A_j$. After normalisation a residual "noise floor"
    $|\\psi|^2\\sim\\sum A_j^2/(\\sum A_j)^2\\approx(1+\\sigma_A^2)/N$ remains everywhere — finite-$N$ synthesis never localises perfectly.
    In addition, a finite set of discrete $k_j$ is almost periodic: with only a few waves the pattern repeats (beats) instead of decaying.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Direct summation</b> on 1500 grid points ($\\Delta x=0.02$, ≥ 20 points per wavelength up to $k=15$). Each wave is generated with the
      phasor recursion $e^{ik_jx_{i+1}}=e^{ik_jx_i}\\,e^{ik_j\\Delta x}$, i.e. one complex multiplication per point instead of a sine and cosine.</li>
      <li><b>Static and build-up modes ($t=0$):</b> the running sum is kept incrementally; changing $N$ only adds or subtracts the waves that changed,
      so even 3000 waves update instantly. In build-up mode $N(t)=\\lfloor e^{0.6\\,r\\,t}\\rfloor$ grows exponentially, then holds for 2.5 s and restarts.</li>
      <li><b>Time evolution:</b> every frame the full sum is recomputed with phases $k_jx-\\tfrac12k_j^2t$ (four waves at a time for speed), on a grid
      thinned to ≈ 0.3 rad per point at the largest $k$. Time runs from $-t_e$ to $t_e$ with $t_e=17.25/k_0$, so the packet crosses the window,
      and you see it converge before $t=0$ and spread after.</li>
      <li><b>Plots and metrics:</b> top, $\\mathrm{Re}\\,\\psi$, $\\pm|\\psi|$ and the analytic envelope; left, the amplitude-weighted histogram of the
      actual $k_j$ with the Gaussian it is drawn from; right, $|\\psi|^2$ with the noise floor $\\sum A_j^2/(\\sum A_j)^2$ (red dashed). "Measured $\\Delta x$"
      is the standard deviation of the computed $|\\psi|^2$ over the window (it includes the noise floor, so it exceeds the limit value at small $N$);
      "Limit $\\Delta x$" is $\\sqrt{1+\\sigma_k^4t^2}/(\\sqrt2\\,\\sigma_k)$; "Mean $k$" is $\\sum A_jk_j/\\sum A_j$, which approaches $k_0$ like $\\sigma_k/\\sqrt N$.
      The convergence of these numbers to their limits as $N$ grows is the accuracy check of the synthesis.</li>
    </ul>

    <h4>What to try</h4>
    <ul>
      <li><b>Watch localisation.</b> In build-up mode: with 1 wave $|\\psi|^2=1$ everywhere; with 2 waves you get beats; by $N\\approx100$ a single bump
      at $x=0$ dominates, and the noise floor falls like $1/N$ (≈ 3×10⁻⁴ at $N=3000$).</li>
      <li><b>Uncertainty relation.</b> Static mode, $N=3000$: with $\\sigma_k=0.5$ the measured $\\Delta x\\approx1.5$ (limit 1.41; the excess is the
      noise floor spread over the window). Halve $\\sigma_k$ to 0.25 and $\\Delta x$ roughly doubles (≈ 2.7, limit 2.83) — the product
      $\\Delta x\\,\\Delta k$ stays close to 1/2.</li>
      <li><b>Few waves never localise.</b> Set $N=10$: the bump at $x=0$ is there, but large revivals of the pattern appear elsewhere
      (random-phase sum of only 10 terms).</li>
      <li><b>Dispersion.</b> Time-evolution mode with $\\sigma_k=1.5$: the packet broadens quickly, $\\Delta x(t)=\\Delta x(0)\\sqrt{1+\\sigma_k^4t^2}$; with
      $\\sigma_k=0.2$ it barely changes while crossing.</li>
      <li><b>Equal amplitudes.</b> $\\sigma_A=0$ gives all $A_j=1$ and the lowest noise floor, exactly $1/N$.</li>
    </ul>

    <h4>Limitations &amp; further reading</h4>
    <p>A finite random sample of $k_j$ is not a continuous spectrum, so the finite-$N$ wave function is not normalisable over all $x$ and the
    window shows only the central region. Free particle only. Reading: D. J. Griffiths &amp; D. F. Schroeter, <i>Introduction to Quantum Mechanics</i>,
    §2.4 and §3.5; E. Hecht, <i>Optics</i>, ch. 7 (superposition, Fourier synthesis, group velocity); A. P. French, <i>Vibrations and Waves</i>, ch. 7.</p>`,

  mount(api) {
    const P = api.params, NX = 1500, X0 = -15, X1 = 15, dx = (X1 - X0) / (NX - 1), NMAX = 3000, NB = 60;
    const x = PM.linspace(X0, X1, NX);
    const sr = new Float64Array(NX), si = new Float64Array(NX);
    const yre = new Float64Array(NX), yim = new Float64Array(NX), yab = new Float64Array(NX), yneg = new Float64Array(NX), prob = new Float64Array(NX);
    const eup = new Float64Array(NX), edn = new Float64Array(NX), eprob = new Float64Array(NX);
    const ks = new Float64Array(NMAX), As = new Float64Array(NMAX), cnt = new Float64Array(NB), ctr = new Float64Array(NB);
    let xv = x, xvStride = 1;
    let nCur = 0, sumA = 0, sumA2 = 0, nShow = 1, tb = 0, t = 0, hold = 0;
    const plots = api.plots([
      { id: "wave", title: "Sum of plane waves: Re ψ(x, t)", span: 2, aspect: 0.34, xlim: [X0, X1], ylim: [-1.5, 1.5], xlabel: "position x", ylabel: "amplitude" },
      { id: "spec", title: "Wavenumbers used (amplitude-weighted)", aspect: 0.62, xlim: [0, 10], ylim: [0, 1], xlabel: "wavenumber k", ylabel: "weight / bin" },
      { id: "prob", title: "Probability density |ψ(x)|²", aspect: 0.62, xlim: [X0, X1], ylim: [0, 1.1], xlabel: "position x", ylabel: "|ψ|²" },
    ]);
    const Mt = api.metrics([
      { id: "n", label: "Number of waves $N$" },
      { id: "km", label: "Mean $k$ (weighted)" },
      { id: "dx", label: "Measured $\\Delta x$ from $|\\psi|^2$" },
      { id: "dxt", label: "Limit $\\Delta x$ ($N\\to\\infty$)" },
      { id: "noise", label: "Noise amplitude $\\sqrt{\\sum A^2}/\\sum A$" },
    ]);

    function generate() {
      const rng = new PM.RNG((P.seed | 0) + 1);
      for (let j = 0; j < NMAX; j++) { ks[j] = rng.gauss(P.k0, P.sk); As[j] = Math.abs(rng.gauss(1, P.sA)); }
      sr.fill(0); si.fill(0); nCur = 0; sumA = 0; sumA2 = 0;
      plots.spec.setLimits([Math.max(0, P.k0 - 4 * P.sk - 0.5), P.k0 + 4 * P.sk + 0.5]);
    }
    // add wave j with sign sgn to the running sum (at time tt)
    function addWave(j, sgn, tt, st = 1, M = NX) {
      const k = ks[j], A = As[j] * sgn, ph = k * X0 - 0.5 * k * k * tt;
      let zr = A * Math.cos(ph), zi = A * Math.sin(ph);
      const wr = Math.cos(k * dx * st), wi = Math.sin(k * dx * st);
      for (let i = 0; i < M; i++) {
        sr[i] += zr; si[i] += zi;
        const q = zr * wr - zi * wi; zi = zr * wi + zi * wr; zr = q;
      }
      sumA += A; sumA2 += sgn * As[j] * As[j];
    }
    function setN(n) { // incremental, t = 0 only
      n = Math.max(1, Math.min(NMAX, n | 0));
      if (n === nCur) return;
      if (n > nCur) { for (let j = nCur; j < n; j++) addWave(j, 1, 0); }
      else if (nCur - n < n) { for (let j = n; j < nCur; j++) addWave(j, -1, 0); }
      else { sr.fill(0); si.fill(0); sumA = 0; sumA2 = 0; for (let j = 0; j < n; j++) addWave(j, 1, 0); }
      nCur = n;
    }
    function evolveAt(tt, n, st, M) { // full recomputation (time evolution) — four waves at a time (less memory traffic)
      sr.fill(0); si.fill(0); sumA = 0; sumA2 = 0;
      let j = 0;
      for (; j + 3 < n; j += 4) {
        const z = [], w = [];
        for (let q = 0; q < 4; q++) {
          const k = ks[j + q], A = As[j + q], ph = k * X0 - 0.5 * k * k * tt;
          z.push(A * Math.cos(ph), A * Math.sin(ph)); w.push(Math.cos(k * dx * st), Math.sin(k * dx * st));
          sumA += A; sumA2 += A * A;
        }
        let ar = z[0], ai = z[1], br_ = z[2], bi_ = z[3], cr = z[4], ci = z[5], dr = z[6], di = z[7];
        const awr = w[0], awi = w[1], bwr = w[2], bwi = w[3], cwr = w[4], cwi = w[5], dwr = w[6], dwi = w[7];
        for (let i = 0; i < M; i++) {
          sr[i] += ar + br_ + cr + dr; si[i] += ai + bi_ + ci + di;
          let q = ar * awr - ai * awi; ai = ar * awi + ai * awr; ar = q;
          q = br_ * bwr - bi_ * bwi; bi_ = br_ * bwi + bi_ * bwr; br_ = q;
          q = cr * cwr - ci * cwi; ci = cr * cwi + ci * cwr; cr = q;
          q = dr * dwr - di * dwi; di = dr * dwi + di * dwr; dr = q;
        }
      }
      for (; j < n; j++) addWave(j, 1, tt, st, M);
      nCur = -1; // incremental state is now invalid
    }
    const tEdge = () => (1.15 * X1) / P.k0;

    function restart() {
      generate(); t = 0; hold = 0; tb = 0;
      if (P.mode === "build") nShow = 1; else nShow = P.N;
      if (P.mode === "evolve") t = -tEdge();
    }

    return {
      reset: restart,
      onParam(id) {
        if (["k0", "sk", "sA", "seed", "mode"].includes(id)) restart();
        else if (id === "N" && P.mode !== "build") nShow = P.N;
      },
      step(dt) {
        if (P.mode === "build") {
          if (nShow >= P.N) { hold += dt; if (hold > 2.5) { hold = 0; tb = 0; nShow = 1; } return; }
          tb += dt;
          nShow = Math.min(P.N, Math.max(1, Math.floor(Math.exp(0.6 * P.rate * tb))));
        } else if (P.mode === "evolve") {
          t += (dt * 2 * tEdge()) / 8;
          if (t > tEdge()) t = -tEdge();
        }
      },
      render() {
        const n = P.mode === "static" ? P.N : nShow;
        const tt = P.mode === "evolve" ? t : 0;
        // in time evolution the grid is thinned to ~0.3 rad per point at the largest k
        let st = 1;
        if (P.mode === "evolve") st = Math.max(1, Math.floor(0.3 / ((P.k0 + 4 * P.sk) * dx)));
        const M = Math.floor((NX - 1) / st) + 1;
        if (st !== xvStride) { xvStride = st; xv = new Float64Array(M); for (let i = 0; i < M; i++) xv[i] = X0 + i * st * dx; }
        if (P.mode === "evolve") evolveAt(tt, n, st, M); else { if (nCur < 0) { sr.fill(0); si.fill(0); sumA = 0; sumA2 = 0; nCur = 0; } setN(n); }
        const inv = 1 / (sumA || 1);
        let m0 = 0, m1 = 0, m2 = 0;
        for (let i = 0; i < M; i++) {
          const a = sr[i] * inv, b = si[i] * inv;
          yre[i] = a; yim[i] = b; const p = a * a + b * b;
          prob[i] = p; yab[i] = Math.sqrt(p); yneg[i] = -yab[i];
          m0 += p; m1 += p * xv[i]; m2 += p * xv[i] * xv[i];
        }
        const xm = m1 / m0, dxm = Math.sqrt(Math.max(0, m2 / m0 - xm * xm));
        const s2 = P.sk * P.sk, g = 1 + s2 * s2 * tt * tt, amp = Math.pow(g, -0.25);
        for (let i = 0; i < M; i++) {
          const u = xv[i] - P.k0 * tt, e = amp * Math.exp((-s2 * u * u) / (2 * g));
          eup[i] = e; edn[i] = -e; eprob[i] = e * e;
        }

        const pw = plots.wave;
        pw.clear();
        pw.line(xv, yab, { color: PlotColors.muted, width: 1, alpha: 0.6 });
        pw.line(xv, yneg, { color: PlotColors.muted, width: 1, alpha: 0.6 });
        if (P.im) pw.line(xv, yim, { color: PlotColors.bad, width: 1.2, alpha: 0.8 });
        pw.line(xv, yre, { color: PlotColors.accent, width: 1.6 });
        if (P.env) { pw.line(xv, eup, { color: PlotColors.accent3, width: 1.4, dash: [6, 4] }); pw.line(xv, edn, { color: PlotColors.accent3, width: 1.4, dash: [6, 4] }); }
        const leg = [{ label: "Re ψ", color: PlotColors.accent }, { label: "±|ψ|", color: PlotColors.muted }];
        if (P.im) leg.push({ label: "Im ψ", color: PlotColors.bad });
        if (P.env) leg.push({ label: "N → ∞ envelope", color: PlotColors.accent3, dash: [6, 4] });
        pw.legend(leg, "tr");
        pw.label(P.mode === "evolve" ? `N = ${n},  t = ${PM.fmt(tt, 2)}` : `N = ${n}`, "tl");

        // k spectrum (amplitude-weighted histogram)
        const ps = plots.spec, [ka, kb] = ps.xlim, bw = (kb - ka) / NB;
        cnt.fill(0);
        let ksum = 0;
        for (let j = 0; j < n; j++) { const b = Math.floor((ks[j] - ka) / bw); if (b >= 0 && b < NB) cnt[b] += As[j] * inv; ksum += As[j] * ks[j]; }
        for (let b = 0; b < NB; b++) ctr[b] = ka + (b + 0.5) * bw;
        const gpk = bw / (Math.sqrt(2 * Math.PI) * P.sk);
        ps.setLimits(null, [0, Math.max(gpk * 1.6, PM.max(cnt) * 1.1)]);
        ps.clear();
        ps.bars(ctr, cnt, bw * 0.9, { color: PlotColors.accent2, alpha: 0.75 });
        ps.fn((k) => gpk * Math.exp((-(k - P.k0) * (k - P.k0)) / (2 * s2)), { color: PlotColors.accent3, width: 1.8, dash: [6, 4] });
        ps.vline(P.k0, { color: PlotColors.muted, dash: [3, 4] });

        const pp = plots.prob;
        pp.clear();
        pp.fill(xv, prob, 0, { color: PlotColors.accent, alpha: 0.35 });
        pp.line(xv, prob, { color: PlotColors.accent, width: 1.6 });
        if (P.env) pp.line(xv, eprob, { color: PlotColors.accent3, width: 1.4, dash: [6, 4] });
        const nf = Math.sqrt(Math.max(sumA2, 0)) * inv;
        pp.hline(nf * nf, { color: PlotColors.bad, dash: [3, 4], alpha: 0.8 });
        pp.label(`noise floor ≈ ${PM.fmt(nf * nf, 2)}`, "tr", { color: PlotColors.bad, size: 11 });

        Mt.set("n", String(n));
        Mt.set("km", PM.fmt(ksum * inv, 3));
        Mt.set("dx", PM.fmt(dxm, 3));
        Mt.set("dxt", PM.fmt(Math.sqrt(g) / (Math.SQRT2 * P.sk), 3));
        Mt.set("noise", PM.fmt(nf, 3));
        api.setTime(P.mode === "evolve" ? `t = ${PM.fmt(tt, 2)}` : `N = ${n}`);
      },
    };
  },
});
