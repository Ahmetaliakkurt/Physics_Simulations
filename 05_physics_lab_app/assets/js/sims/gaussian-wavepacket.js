/* Free-particle Gaussian wave packet — closed-form analytic solution, evaluated live every frame. */
App.register({
  id: "gaussian-wavepacket",
  category: "quantum",
  order: 32,
  title: "Gaussian Wave Packet",
  icon: "🌊",
  subtitle: "A free Gaussian wave packet moves at the group velocity ħk₀/m while its width grows in time (dispersion): Re ψ, Im ψ and |ψ|² from the exact solution.",
  notes: [{
    type: "info",
    html: "The upper panel shows the oscillating real and imaginary parts under the envelope $\\pm|\\psi|$; the lower panel shows the probability density with the " +
      "analytic $\\pm\\sigma(t)$ band. Increase $m$ or $\\sigma_0$ to slow down the spreading, set $k_0=0$ to watch a packet that spreads without moving.",
  }],
  animated: true,
  speed: { min: 0.2, max: 5, value: 1.5, step: 0.1 },
  controls: [
    { id: "k0", type: "slider", label: "Central wavenumber $k_0$", min: 0, max: 8, step: 0.1, value: 3 },
    { id: "sigma", type: "slider", label: "Initial width $\\sigma_0$", min: 0.5, max: 5, step: 0.1, value: 2 },
    { id: "m", type: "slider", label: "Mass $m$", min: 0.2, max: 3, step: 0.1, value: 1 },
    { id: "tmax", type: "slider", label: "Loop duration $t_{max}$", min: 5, max: 60, step: 1, value: 30, live: true,
      help: "The animation restarts from $t=0$ after $t_{max}$." },
    { type: "section", label: "Display" },
    { id: "showRe", type: "checkbox", label: "Show Re ψ and Im ψ", value: true, live: true },
    { id: "showBand", type: "checkbox", label: "Show the ±σ(t) band", value: true, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A single non-relativistic particle of mass $m$ moves freely ($V=0$) along the $x$ axis. At $t=0$ it is prepared in a
    minimum-uncertainty Gaussian state centred at $x=0$ with mean wavenumber $k_0$ (mean momentum $p_0=\\hbar k_0$) and
    position spread $\\sigma_0$. Units: $\\hbar=1$; lengths, times and masses are dimensionless, so $k_0$ is an inverse length,
    $\\sigma_0$ a length, and velocities are $\\hbar k/m$. The plotted window is $-20\\le x\\le100$.</p>

    <h4>Equations being solved</h4>
    <p>The free time-dependent Schrödinger equation with the Gaussian initial condition</p>
    $$ i\\hbar\\frac{\\partial\\psi}{\\partial t} = -\\frac{\\hbar^2}{2m}\\frac{\\partial^2\\psi}{\\partial x^2},\\qquad
       \\psi(x,0)=(2\\pi\\sigma_0^2)^{-1/4}\\,e^{ik_0x}\\,e^{-x^2/4\\sigma_0^2}. $$
    <p>Because every plane wave $e^{i(kx-\\omega t)}$ with $\\omega=\\hbar k^2/2m$ solves the equation, one Fourier-transforms the
    initial state (again a Gaussian, centred at $k_0$ with width $1/2\\sigma_0$), attaches the phase $e^{-i\\hbar k^2t/2m}$ to each
    component and transforms back. The remaining Gaussian integral gives, with $\\alpha=1/4\\sigma_0^2$,</p>
    $$ \\psi(x,t)=\\frac{(2\\pi\\sigma_0^2)^{-1/4}}{\\sqrt{1+\\frac{2i\\hbar\\alpha t}{m}}}\\;
       \\exp\\!\\Big[i\\Big(k_0x-\\frac{\\hbar k_0^2}{2m}t\\Big)\\Big]\\;
       \\exp\\!\\left[-\\frac{\\alpha\\,(x-\\hbar k_0 t/m)^2}{1+\\frac{2i\\hbar\\alpha t}{m}}\\right]. $$
    <p>Its modulus squared is a normalised Gaussian whose centre moves at the group velocity $v_g=d\\omega/dk|_{k_0}=\\hbar k_0/m$
    (twice the phase velocity $\\omega/k=\\hbar k_0/2m$ of the carrier) and whose width grows:</p>
    <div class="callout">$$ |\\psi(x,t)|^2=\\frac{1}{\\sqrt{2\\pi}\\,\\sigma(t)}\\exp\\!\\left[-\\frac{(x-\\hbar k_0t/m)^2}{2\\sigma(t)^2}\\right],\\qquad
      \\sigma(t)=\\sigma_0\\sqrt{1+\\left(\\frac{\\hbar t}{2m\\sigma_0^2}\\right)^2}. $$</div>
    <p>The momentum distribution never changes (free particle), $\\Delta p=\\hbar/2\\sigma_0$, so at $t=0$ the product
    $\\Delta x\\,\\Delta p=\\hbar/2$ saturates the Heisenberg bound; afterwards position–momentum correlations build up and
    $\\Delta x$ grows, asymptotically linearly as $\\Delta x\\approx \\Delta p\\,t/m$ — the faster components simply run ahead.
    The characteristic spreading time is $\\tau=2m\\sigma_0^2/\\hbar$.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li>No time stepping is used: on every frame the closed-form expression above is evaluated on 900 equally spaced points in
      $[-20,100]$ ($\\Delta x\\approx0.13$, i.e. ≥ 6 points per carrier wavelength $2\\pi/k_0$ up to $k_0=8$). The complex square root is
      written as $(1+ib)^{-1/2}=r^{-1/2}e^{-i\\varphi/2}$ with $b=2\\hbar\\alpha t/m$, $r=\\sqrt{1+b^2}$, $\\varphi=\\arctan b$, so there is
      no branch ambiguity. Because the solution is exact, no numerical error accumulates, however long the run.</li>
      <li>Upper panel: $\\mathrm{Re}\\,\\psi$ (blue), $\\mathrm{Im}\\,\\psi$ (red, dashed) and the envelope $\\pm|\\psi|$.
      Lower panel: $|\\psi|^2$, the centre $\\langle x\\rangle=\\hbar k_0t/m$ (dashed line) and the band
      $\\langle x\\rangle\\pm\\sigma(t)$.</li>
      <li>Metrics: $\\langle x\\rangle$, $\\sigma(t)$ and $v_g$ from the formulas above; "Measured $\\Delta x$" is the standard deviation
      computed numerically from the sampled $|\\psi|^2$, and "Probability in window" is $\\int|\\psi|^2dx$ by the trapezoidal rule.
      The norm is the conserved quantity: it stays at 1.000 as long as the packet is inside the window, and the measured
      $\\Delta x$ agrees with the analytic $\\sigma(t)$ — two independent checks of the evaluation.</li>
    </ul>

    <h4>What to try</h4>
    <ul>
      <li><b>Spreading time.</b> With $\\sigma_0=2$, $m=1$ the spreading time is $\\tau=2m\\sigma_0^2/\\hbar=8$: at $t=8$ the width is
      $\\sqrt2\\,\\sigma_0\\approx2.83$ and the peak of $|\\psi|^2$ has dropped by $1/\\sqrt2$.</li>
      <li><b>Narrow packets spread fast.</b> Set $\\sigma_0=0.5$: $\\tau=0.5$, and by $t=10$ the width is already ≈ 10 — a sharply
      localised particle has a broad momentum distribution ($\\Delta p=\\hbar/2\\sigma_0=1$).</li>
      <li><b>Heavy particles.</b> Raise $m$ to 3: both the group velocity $\\hbar k_0/m$ and the spreading slow down, as $\\tau\\propto m$.
      In the classical limit (large $m$) the packet behaves like a point particle.</li>
      <li><b>Phase vs group velocity.</b> Watch the wiggles of $\\mathrm{Re}\\,\\psi$: they slide through the envelope at half the
      envelope's speed, and the carrier wavelength becomes shorter at the front of the packet than at its back (a "chirp").</li>
      <li><b>$k_0=0$.</b> The packet stays centred at $x=0$ and only spreads; the probability density stays symmetric.</li>
    </ul>

    <h4>Limitations &amp; further reading</h4>
    <p>The packet is free and one-dimensional; there are no boundaries, and the plotted window is just a viewport (the norm in the
    window drops once the packet leaves it). Spin and relativistic effects are ignored. See D. J. Griffiths &amp; D. F. Schroeter,
    <i>Introduction to Quantum Mechanics</i>, §2.4 and Problem 2.21; J. J. Sakurai, <i>Modern Quantum Mechanics</i>, §2.4;
    C. Cohen-Tannoudji et al., <i>Quantum Mechanics</i> Vol. 1, Complement G<sub>I</sub>.</p>`,

  mount(api) {
    const P = api.params, hbar = 1, N = 900, XA = -20, XB = 100, dxg = (XB - XA) / (N - 1);
    const x = PM.linspace(XA, XB, N);
    const re = new Float64Array(N), im = new Float64Array(N), prob = new Float64Array(N);
    const env = new Float64Array(N), nenv = new Float64Array(N);
    const M = api.metrics([
      { id: "t", label: "Time $t$" },
      { id: "xc", label: "Packet centre $\\langle x\\rangle=\\hbar k_0t/m$" },
      { id: "sig", label: "Width $\\sigma(t)$ (analytic)" },
      { id: "dxm", label: "Measured $\\Delta x$ from $|\\psi|^2$" },
      { id: "vg", label: "Group velocity $v_g=\\hbar k_0/m$" },
      { id: "norm", label: "Probability in window" },
    ]);
    const plots = api.plots([
      { id: "wave", title: "Wave function: Re ψ, Im ψ and envelope ±|ψ|", span: 2, aspect: 0.32, xlim: [XA, XB], ylim: [-0.55, 0.55], ylabel: "amplitude" },
      { id: "prob", title: "Probability density |ψ(x,t)|²", span: 2, aspect: 0.32, xlim: [XA, XB], ylim: [0, 0.3], xlabel: "position x", ylabel: "|ψ|²" },
    ]);
    let t = 0, normNow = 1, dxMeas = 0;

    function compute() {
      const { k0, sigma, m } = P;
      const a = 1 / (4 * sigma * sigma);
      const pre = Math.pow(2 * Math.PI * sigma * sigma, -0.25);
      // D = 1 + i·b,  b = 2ħαt/m
      const b = (2 * hbar * a * t) / m, D2 = 1 + b * b;
      // 1/sqrt(D) = r^(-1/2) e^{-iφ/2}
      const r = Math.sqrt(D2), phi = Math.atan2(b, 1);
      const sr = pre / Math.sqrt(r), sc = Math.cos(-phi / 2), ss = Math.sin(-phi / 2);
      const xc = (hbar * k0 * t) / m, w = (hbar * k0 * k0 * t) / (2 * m);
      let s0 = 0, s1 = 0, s2 = 0;
      for (let i = 0; i < N; i++) {
        const u = x[i] - xc;
        // -α u²/(1+ib) = -α u² (1-ib)/D2
        const eRe = (-a * u * u) / D2, eIm = (a * u * u * b) / D2;
        const mag = sr * Math.exp(eRe);
        const ang = k0 * x[i] - w + eIm;
        const cr = Math.cos(ang), ci = Math.sin(ang);
        re[i] = mag * (cr * sc - ci * ss);
        im[i] = mag * (cr * ss + ci * sc);
        const p = re[i] * re[i] + im[i] * im[i];
        prob[i] = p; env[i] = Math.sqrt(p); nenv[i] = -env[i];
        const wq = i === 0 || i === N - 1 ? 0.5 : 1; // trapezoidal weights
        s0 += wq * p; s1 += wq * p * x[i]; s2 += wq * p * x[i] * x[i];
      }
      normNow = s0 * dxg;
      const mu = s1 / (s0 || 1);
      dxMeas = Math.sqrt(Math.max(s2 / (s0 || 1) - mu * mu, 0));
    }

    return {
      reset() { t = 0; },
      step(dt) { t += dt; if (t > P.tmax) t = 0; },
      render() {
        compute();
        const { k0, sigma, m } = P;
        const xc = (hbar * k0 * t) / m;
        const st = sigma * Math.sqrt(1 + Math.pow((hbar * t) / (2 * m * sigma * sigma), 2));
        const pw = plots.wave, pp = plots.prob;
        const amp0 = Math.pow(2 * Math.PI * sigma * sigma, -0.25);
        pw.setLimits(null, [-1.2 * amp0, 1.2 * amp0]);
        pw.clear();
        if (P.showRe) {
          pw.line(x, re, { color: PlotColors.blue, width: 1.5 });
          pw.line(x, im, { color: PlotColors.bad, width: 1.3, dash: [5, 4], alpha: 0.9 });
        }
        pw.line(x, env, { color: PlotColors.muted, width: 1, alpha: 0.7 });
        pw.line(x, nenv, { color: PlotColors.muted, width: 1, alpha: 0.7 });
        const leg = [{ label: "±|ψ|", color: PlotColors.muted }];
        if (P.showRe) leg.unshift({ label: "Re ψ", color: PlotColors.blue }, { label: "Im ψ", color: PlotColors.bad, dash: [5, 4] });
        pw.legend(leg);

        const peak = Math.pow(2 * Math.PI * sigma * sigma, -0.5);
        pp.setLimits(null, [0, peak * 1.15]);
        pp.clear();
        if (P.showBand) pp.rect(xc - st, 0, xc + st, peak * 2, { color: PlotColors.accent3, alpha: 0.1 });
        pp.fill(x, prob, 0, { color: PlotColors.accent, alpha: 0.35 });
        pp.line(x, prob, { color: PlotColors.accent, width: 2 });
        pp.vline(xc, { color: PlotColors.accent3, dash: [6, 4], width: 1.4 });
        pp.label(`t = ${PM.fmt(t, 2)}`, "tl");
        if (P.showBand) pp.legend([{ label: "⟨x⟩ ± σ(t)", color: PlotColors.accent3, type: "box" }, { label: "|ψ|²", color: PlotColors.accent }]);

        M.set("t", PM.fmt(t, 2));
        M.set("xc", PM.fmt(xc, 2));
        M.set("sig", PM.fmt(st, 3));
        M.set("dxm", normNow > 0.99 ? PM.fmt(dxMeas, 3) : "— (leaving window)");
        M.set("vg", PM.fmt((hbar * k0) / m, 2));
        M.set("norm", PM.fmt(normNow, 4));
        api.setTime(`t = ${PM.fmt(t, 2)}`);
      },
    };
  },
});
