/* Quantum tunnelling through a double Lorentzian barrier — 1D time-dependent Schrödinger equation,
 * symmetric (Strang) split-step Fourier, live every frame; complex absorbing potential at the edges.
 * The stationary transmission T(k) of the same potential (Numerov, scattering boundary conditions)
 * is computed once per reset as an independent prediction of the long-time T. */
App.register({
  id: "quantum-tunneling",
  category: "quantum",
  order: 40,
  title: "Quantum Tunnelling Through a Barrier",
  icon: "🚧",
  subtitle: "A Gaussian wave packet hits a double Lorentzian barrier: transmitted and reflected probabilities computed live with the split-step Fourier method and compared with the stationary transmission T(k).",
  notes: [
    { type: "info", html: "A classical particle with less energy than the barrier is always reflected; quantum mechanically part of the wave function <b>leaks through</b> the barrier. The part trapped between the two barriers bounces back and forth and leaks out slowly (resonances). The shaded strips at the edges are <b>absorbing boundaries</b> that swallow outgoing waves without reflection; the absorbed probability is added to T or R." },
  ],
  animated: true,
  speed: { min: 0.2, max: 4, value: 1, step: 0.1 },
  controls: [
    { id: "V0", type: "slider", label: "Barrier height $V_0$", min: 0.5, max: 6, step: 0.1, value: 2 },
    { id: "w", type: "slider", label: "Barrier width $w$", min: 0.3, max: 3, step: 0.1, value: 1 },
    { id: "sep", type: "slider", label: "Barrier separation $d$", min: 2, max: 20, step: 0.5, value: 10 },
    { id: "k0", type: "slider", label: "Initial wavenumber $k_0$", min: 0.5, max: 6, step: 0.1, value: 2.5 },
    { id: "sigma", type: "slider", label: "Packet width $\\sigma$", min: 0.5, max: 5, step: 0.1, value: 2 },
    { type: "section", label: "Numerics" },
    { id: "cap", type: "checkbox", label: "Absorbing boundary (CAP)", value: true,
      help: "When off, the periodic boundary of the FFT lets the wave leave through one edge and re-enter through the other." },
    { id: "rate", type: "slider", label: "Simulation rate (time units / s)", min: 1, max: 10, step: 0.5, value: 4, live: true },
    { type: "section", label: "Display" },
    { id: "showRI", type: "checkbox", label: "Show Re ψ and Im ψ", value: true, live: true },
    { id: "showTk", type: "checkbox", label: "Overlay the stationary $T(k)$ on the momentum plot", value: true, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A particle of mass $m$ moves on a line through two identical smooth barriers of height $V_0$ and half-width $w$, centred at
    $x_{1,2}=\\mp d/2$. It starts at $x_0=-30$ as a Gaussian packet with mean wavenumber $k_0>0$ (moving right) and width $\\sigma$.
    Units: $\\hbar=m=1$, so the kinetic energy of a plane wave is $E=k^2/2$ and the mean energy of the packet is
    $\\langle E\\rangle=k_0^2/2+1/4\\sigma^2$. The computational box is $-60\\le x\\le60$.</p>

    <h4>Equations being solved</h4>
    $$ i\\hbar \\frac{\\partial \\psi}{\\partial t} = \\left[-\\frac{\\hbar^2}{2m}\\frac{\\partial^2}{\\partial x^2} + V(x) - iW(x)\\right]\\psi,\\qquad
       V(x) = \\frac{V_0}{1+\\big((x-x_1)/w\\big)^2} + \\frac{V_0}{1+\\big((x-x_2)/w\\big)^2}, $$
    $$ \\psi(x,0)=(\\pi\\sigma^2)^{-1/4}\\,e^{-(x-x_0)^2/2\\sigma^2}\\,e^{ik_0x}. $$
    <p>$W(x)\\ge0$ is the complex absorbing potential (CAP), non-zero only near the edges. A packet is a superposition of plane waves with
    momentum distribution $|\\phi_0(k)|^2=\\frac{\\sigma}{\\sqrt\\pi}e^{-\\sigma^2(k-k_0)^2}$; long after the collision each component $k$ has been
    transmitted with the stationary probability $T(k)$, so the final transmission is</p>
    <div class="callout">$$ T_\\infty=\\int_0^\\infty T(k)\\,|\\phi_0(k)|^2\\,dk,\\qquad
      T_{\\rm rect}(E)=\\left[1+\\frac{V_0^2\\sinh^2(\\kappa a)}{4E(V_0-E)}\\right]^{-1},\\ \\ \\kappa=\\frac{\\sqrt{2m(V_0-E)}}{\\hbar}. $$</div>
    <p>$T_{\\rm rect}$ is the textbook result for a single rectangular barrier of height $V_0$ and width $a$ (for $E>V_0$ replace
    $\\sinh\\to\\sin$ and $V_0-E\\to E-V_0$). It is shown as a metric with $a=2w$ (the full width at half maximum of one Lorentzian) for intuition:
    tunnelling is exponentially sensitive, $T\\approx16\\frac{E}{V_0}(1-\\frac{E}{V_0})e^{-2\\kappa a}$ for thick barriers. If the two barriers
    acted incoherently, the total would be $T_1T_2/(1-R_1R_2)=T_1/(2-T_1)$; coherently, waves bouncing between them interfere, and at the
    resonance energies (quasi-bound states of the well between the barriers, spaced roughly by $\\Delta k\\approx\\pi/d$) the double barrier becomes
    fully transparent, $T(k)=1$, even when one barrier alone transmits little — resonant tunnelling.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Grid:</b> $N=2048$ points on $[-60,60)$, $\\Delta x\\approx0.059$, so $|k|\\le\\pi/\\Delta x\\approx54$, far above any $k$ in the packet.</li>
      <li><b>Integrator:</b> symmetric (Strang) split-step Fourier with $\\Delta t=0.02$,
      $$ \\psi(t+\\Delta t) = e^{-(iV+W)\\Delta t/2\\hbar}\\, \\mathcal{F}^{-1}\\!\\left[e^{-i\\hbar k^2 \\Delta t/2m}\\, \\mathcal{F}\\left[e^{-(iV+W)\\Delta t/2\\hbar}\\,\\psi(t)\\right]\\right]. $$
      The kinetic step is exact in Fourier space (FFT), the potential step is exact in real space; the splitting error is $\\mathcal O(\\Delta t^3)$
      per step and the scheme is unitary apart from the deliberate absorption. The number of steps per frame is set by the "simulation rate".</li>
      <li><b>Absorbing boundary:</b> for $|x|>x_a=46$, $W=W_0\\big[(|x|-x_a)/(60-x_a)\\big]^2$ with $W_0=4$ — a smooth 14-unit ramp, long compared with the
      wavelength, so it absorbs instead of reflecting. The probability removed in each half-step, $|\\psi|^2(1-e^{-W\\Delta t})\\Delta x$, is booked to the
      right or left side.</li>
      <li><b>T and R:</b> $T(t)=\\int_{x>x_2+3w}|\\psi|^2dx+P_{\\rm abs,right}$, $R(t)=\\int_{x\\lt x_1-3w}|\\psi|^2dx+P_{\\rm abs,left}$,
      evaluated as Riemann sums on the grid; "between barriers" is the rest. The conserved quantity monitored is the total $T+R+P_{\\rm between}=\\int|\\psi|^2dx+P_{\\rm abs}$,
      which must stay 1 — any drift would reveal a non-unitary error.</li>
      <li><b>Stationary prediction:</b> at each reset the time-independent equation $\\psi''=2(V-E)\\psi$ is integrated with the Numerov method on the same grid,
      from a purely outgoing wave $e^{ikx}$ on the right to the left edge, where the solution is split into incident and reflected parts
      $Ae^{ikx}+Be^{-ikx}$; then $T(k)=1/|A|^2$. This is done for 500 values of $k$, plotted (scaled to the plot height) over the momentum distribution, and
      averaged with $|\\phi_0(k)|^2$ to give the metric $T_\\infty$, which the time-dependent $T(t)$ should approach.</li>
      <li><b>Momentum plot:</b> $|\\phi(k,t)|^2=|\\tilde\\psi(k)|^2\\Delta x^2/2\\pi$ from an FFT of the current $\\psi$; the dashed line marks $k=\\sqrt{2V_0}$, the wavenumber whose
      energy equals the barrier top.</li>
    </ul>

    <h4>What to try</h4>
    <ul>
      <li><b>Tunnelling below the top.</b> Defaults: $\\langle E\\rangle=3.19\\gt V_0=2$ — most of the packet passes. Raise $V_0$ to 4.5: now $\\langle E\\rangle\\lt V_0$, the
      components below $k=\\sqrt{2V_0}=3$ must tunnel, and $T$ drops sharply; $T(t)$ settles at the predicted $T_\\infty$.</li>
      <li><b>Exponential sensitivity.</b> With $V_0=4.5$ increase the width $w$ from 1 to 2: $T_{\\rm rect}(k_0)$ falls by orders of magnitude ($\\propto e^{-2\\kappa a}$).</li>
      <li><b>Resonant tunnelling.</b> Make the packet narrow in $k$ (σ = 5) and set $V_0=4.5$, $w=0.6$: the $T(k)$ overlay shows sharp resonance peaks reaching 1;
      move $k_0$ onto a peak and the transmission is far above $T_1/(2-T_1)$, while the probability between the barriers decays slowly (a long-lived
      quasi-bound state).</li>
      <li><b>Separation.</b> Increase $d$: the resonances get closer ($\\Delta k\\sim\\pi/d$) and narrower, and the trapped probability rings longer.</li>
      <li><b>Turn off the CAP.</b> The transmitted wave reappears from the left edge (periodic FFT box) and $T,R$ become meaningless — why absorbing boundaries are needed.</li>
    </ul>

    <h4>Limitations &amp; further reading</h4>
    <p>One dimension, spinless, non-relativistic; the Lorentzian tails extend to the box edges and are truncated there. The Numerov transmission uses a
    second-derivative finite-difference model while the time evolution uses the exact spectral kinetic energy — they agree to well under a per cent
    at these resolutions. Reading: D. J. Griffiths &amp; D. F. Schroeter, <i>Introduction to Quantum Mechanics</i>, §2.5–2.6 (scattering states, rectangular barrier problems);
    C. Cohen-Tannoudji et al., <i>Quantum Mechanics</i> Vol. 1, Complement H<sub>I</sub>; D. J. Tannor, <i>Introduction to Quantum Mechanics: A Time-Dependent Perspective</i>
    (split-operator method, absorbing potentials).</p>`,

  mount(api) {
    const P = api.params, N = 2048, LB = 120, dx = LB / N, DT = 0.02, XA = 46, W0 = 4, X0 = -30, NK = 500, HCAP = 20000;
    const x = new Float64Array(N), V = new Float64Array(N), re = new Float64Array(N), im = new Float64Array(N);
    const vr = new Float64Array(N), vi = new Float64Array(N), dmp = new Float64Array(N), tr = new Float64Array(N), ti = new Float64Array(N);
    const prob = new Float64Array(N), vplot = new Float64Array(N), absY = new Float64Array(N), absN = new Float64Array(N);
    const kk = PM.fftk(N, dx), ksh = new Float64Array(N), pk = new Float64Array(N), fr = new Float64Array(N), fi = new Float64Array(N);
    const kT = new Float64Array(NK), Tk = new Float64Array(NK), TkPlot = new Float64Array(NK);
    const hTt = new Float64Array(HCAP), hT = new Float64Array(HCAP), hR = new Float64Array(HCAP), hB = new Float64Array(HCAP);
    for (let i = 0; i < N; i++) { x[i] = -LB / 2 + i * dx; ksh[i] = kk[(i + N / 2) % N]; }
    const plots = api.plots([
      { id: "wave", title: "Wave function Re ψ, Im ψ and the potential (scaled)", span: 2, aspect: 0.3, xlim: [-60, 60], ylim: [-0.7, 0.7], ylabel: "amplitude" },
      { id: "prob", title: "Probability density |ψ(x,t)|²", span: 2, aspect: 0.3, xlim: [-60, 60], ylim: [0, 0.35], xlabel: "position x", ylabel: "|ψ|²" },
      { id: "tr", title: "Transmitted T(t) and reflected R(t) probability", aspect: 0.62, xlim: [0, 30], ylim: [0, 1.05], xlabel: "time t", ylabel: "probability" },
      { id: "mom", title: "Momentum distribution |φ(k,t)|² and stationary T(k)", aspect: 0.62, xlim: [-6, 6], ylim: [0, 1], xlabel: "wavenumber k", ylabel: "|φ(k)|²" },
    ]);
    const Mt = api.metrics([
      { id: "T", label: "Transmitted $T$ (right)" },
      { id: "R", label: "Reflected $R$ (left)" },
      { id: "B", label: "Between barriers" },
      { id: "Tinf", label: "Predicted $T_\\infty$ (stationary)" },
      { id: "Trect", label: "One rect. barrier $T_{\\rm rect}(k_0)$, $a=2w$" },
      { id: "Tinc", label: "Incoherent pair $T_1/(2-T_1)$" },
      { id: "E", label: "$\\langle E\\rangle / V_0$" },
      { id: "nrm", label: "Total probability (check)" },
    ]);
    let t, absL, absR, amp0, hn, Emean, xT, xR, Tinf, Trect, Tinc;

    function measure() {
      let pL = 0, pR = 0, tot = 0;
      for (let i = 0; i < N; i++) {
        const p = re[i] * re[i] + im[i] * im[i]; prob[i] = p; tot += p;
        if (x[i] > xT) pR += p; else if (x[i] < xR) pL += p;
      }
      tot *= dx;
      return { T: pR * dx + absR, R: pL * dx + absL, B: tot - (pR + pL) * dx, tot: tot + absL + absR };
    }
    function record() {
      const m = measure();
      if (hn >= HCAP) { // keep every second sample when full
        for (let i = 0; i < HCAP / 2; i++) { hTt[i] = hTt[2 * i]; hT[i] = hT[2 * i]; hR[i] = hR[2 * i]; hB[i] = hB[2 * i]; }
        hn = HCAP / 2;
      }
      hTt[hn] = t; hT[hn] = m.T; hR[hn] = m.R; hB[hn] = m.B; hn++;
    }
    function halfV() { // e^{-iVΔt/2} · e^{-WΔt/2} plus bookkeeping of the absorbed probability
      for (let i = 0; i < N; i++) {
        const r = re[i], q = im[i];
        if (dmp[i] < 1) {
          const lost = (r * r + q * q) * (1 - dmp[i] * dmp[i]) * dx;
          if (x[i] < 0) absL += lost; else absR += lost;
        }
        re[i] = r * vr[i] - q * vi[i]; im[i] = r * vi[i] + q * vr[i];
      }
    }
    function stepOnce() {
      halfV();
      PM.fft(re, im, false);
      for (let i = 0; i < N; i++) { const r = re[i], q = im[i]; re[i] = r * tr[i] - q * ti[i]; im[i] = r * ti[i] + q * tr[i]; }
      PM.fft(re, im, true);
      halfV();
      t += DT;
    }
    /** Stationary transmission T(k) by Numerov integration from the right (outgoing wave e^{ikx}) to the left. */
    function transmission(k) {
      const E = 0.5 * k * k, h12 = (dx * dx) / 12;
      let r1 = Math.cos(k * x[N - 1]), i1 = Math.sin(k * x[N - 1]);   // ψ_{i+1}
      let r0 = Math.cos(k * x[N - 2]), i0 = Math.sin(k * x[N - 2]);   // ψ_i
      let f1 = 2 * (V[N - 1] - E), f0 = 2 * (V[N - 2] - E);
      for (let i = N - 2; i >= 1; i--) {
        const fm = 2 * (V[i - 1] - E);
        const c0 = 2 * (1 + 5 * h12 * f0), c1 = 1 - h12 * f1, cm = 1 - h12 * fm;
        const rm = (c0 * r0 - c1 * r1) / cm, im_ = (c0 * i0 - c1 * i1) / cm;
        r1 = r0; i1 = i0; r0 = rm; i0 = im_; f1 = f0; f0 = fm;
      }
      // now r0,i0 = ψ(x0), r1,i1 = ψ(x1): solve ψ = A e^{ikx} + B e^{-ikx}
      const a0r = Math.cos(k * x[0]), a0i = Math.sin(k * x[0]), a1r = Math.cos(k * x[1]), a1i = Math.sin(k * x[1]);
      // b = conj(a);  A = (ψ0 b1 − ψ1 b0) / (a0 b1 − a1 b0)
      const nr = (r0 * a1r + i0 * a1i) - (r1 * a0r + i1 * a0i), ni = (i0 * a1r - r0 * a1i) - (i1 * a0r - r1 * a0i);
      const dr = (a0r * a1r + a0i * a1i) - (a1r * a0r + a1i * a0i), di = (a0i * a1r - a0r * a1i) - (a1i * a0r - a1r * a0i);
      const A2 = (nr * nr + ni * ni) / (dr * dr + di * di);
      return A2 > 0 ? Math.min(1, 1 / A2) : 0;
    }
    function rectT(k, V0, a) {
      const E = 0.5 * k * k;
      if (Math.abs(E - V0) < 1e-9) return 1 / (1 + V0 * a * a / 2);
      if (E < V0) { const ka = Math.sqrt(2 * (V0 - E)) * a, s = Math.sinh(ka); return 1 / (1 + (V0 * V0 * s * s) / (4 * E * (V0 - E))); }
      const qa = Math.sqrt(2 * (E - V0)) * a, s = Math.sin(qa); return 1 / (1 + (V0 * V0 * s * s) / (4 * E * (E - V0)));
    }

    function reset() {
      const { V0, w, sep, k0, sigma } = P;
      const x1 = -sep / 2, x2 = sep / 2;
      xT = x2 + 3 * w; xR = x1 - 3 * w;
      let s = 0;
      for (let i = 0; i < N; i++) {
        const xi = x[i];
        V[i] = V0 / (1 + ((xi - x1) / w) ** 2) + V0 / (1 + ((xi - x2) / w) ** 2);
        const a = Math.abs(xi), W = P.cap && a > XA ? W0 * ((a - XA) / (LB / 2 - XA)) ** 2 : 0;
        dmp[i] = Math.exp((-W * DT) / 2);
        vr[i] = dmp[i] * Math.cos((-V[i] * DT) / 2); vi[i] = dmp[i] * Math.sin((-V[i] * DT) / 2);
        const g = Math.exp(-((xi - X0) ** 2) / (2 * sigma * sigma));
        re[i] = g * Math.cos(k0 * xi); im[i] = g * Math.sin(k0 * xi); s += g * g * dx;
        tr[i] = Math.cos((-kk[i] * kk[i] * DT) / 2); ti[i] = Math.sin((-kk[i] * kk[i] * DT) / 2);
      }
      s = 1 / Math.sqrt(s);
      for (let i = 0; i < N; i++) { re[i] *= s; im[i] *= s; }
      t = 0; absL = 0; absR = 0; hn = 0;
      amp0 = Math.pow(Math.PI * sigma * sigma, -0.25);
      Emean = (k0 * k0) / 2 + 1 / (4 * sigma * sigma);
      plots.wave.setLimits(null, [-1.15 * amp0, 1.15 * amp0]);
      plots.prob.setLimits(null, [0, 1.3 * amp0 * amp0]);
      plots.tr.setLimits([0, Math.max(20, (1.3 * 60) / Math.max(k0, 0.5))]);
      const km = k0 + 4 / sigma + 0.5;
      plots.mom.setLimits([-km, km]);
      // stationary transmission on k ∈ (0, km], packet average over k > 0
      let num = 0;
      const dkT = km / NK, c = sigma / Math.sqrt(Math.PI);
      for (let j = 0; j < NK; j++) {
        const k = (j + 1) * dkT;
        kT[j] = k; Tk[j] = transmission(k);
        num += Tk[j] * c * Math.exp(-sigma * sigma * (k - k0) * (k - k0)) * dkT;
      }
      Tinf = num;
      Trect = rectT(k0, V0, 2 * w);
      Tinc = Trect / (2 - Trect);
      record();
    }

    function potPlot(top, vref) { for (let i = 0; i < N; i++) vplot[i] = (V[i] / vref) * 0.8 * top; }
    function capZones(p, y0, y1) {
      if (!P.cap) return;
      p.rect(-60, y0, -XA, y1, { color: PlotColors.bad, alpha: 0.08 });
      p.rect(XA, y0, 60, y1, { color: PlotColors.bad, alpha: 0.08 });
    }

    return {
      reset,
      step(dt) {
        const n = Math.max(1, Math.round((dt * P.rate) / DT));
        for (let s = 0; s < n; s++) stepOnce();
        record();
        const xl = plots.tr.xlim;
        if (t > xl[1]) plots.tr.setLimits([0, t * 1.5]);
      },
      render() {
        const m = measure();
        // wave function
        const pw = plots.wave, top = pw.ylim[1];
        pw.clear();
        capZones(pw, -10, 10);
        potPlot(top, P.V0);
        pw.fill(x, vplot, 0, { color: PlotColors.muted, alpha: 0.25 });
        if (P.showRI) {
          pw.line(x, re, { color: PlotColors.accent, width: 1.2 });
          pw.line(x, im, { color: PlotColors.bad, width: 1.1, alpha: 0.85 });
        }
        for (let i = 0; i < N; i++) { absY[i] = Math.sqrt(prob[i]); absN[i] = -absY[i]; }
        pw.line(x, absY, { color: PlotColors.text, width: 1, alpha: 0.6 });
        pw.line(x, absN, { color: PlotColors.text, width: 1, alpha: 0.6 });
        const lw = [{ label: "±|ψ|", color: PlotColors.text }, { label: "V(x) (scaled)", color: PlotColors.muted, type: "box" }];
        if (P.showRI) lw.unshift({ label: "Re ψ", color: PlotColors.accent }, { label: "Im ψ", color: PlotColors.bad });
        pw.legend(lw, "tr");

        // probability density
        const pp = plots.prob, pt = pp.ylim[1];
        pp.clear();
        capZones(pp, 0, 10);
        pp.rect(xT, 0, P.cap ? XA : 60, pt * 2, { color: PlotColors.good, alpha: 0.06 });
        pp.rect(P.cap ? -XA : -60, 0, xR, pt * 2, { color: PlotColors.blue, alpha: 0.06 });
        const vref = Math.max(P.V0, Emean * 1.08);
        potPlot(pt, vref);
        pp.fill(x, vplot, 0, { color: PlotColors.muted, alpha: 0.25 });
        pp.line(x, vplot, { color: PlotColors.muted, width: 1 });
        const eY = (Emean / vref) * 0.8 * pt;
        pp.hline(eY, { color: PlotColors.accent3, dash: [6, 4], width: 1.3 });
        pp.text(-58, eY, "⟨E⟩", { dy: -8, color: PlotColors.accent3, size: 11 });
        pp.fill(x, prob, 0, { color: PlotColors.accent2, alpha: 0.35 });
        pp.line(x, prob, { color: PlotColors.accent2, width: 1.8 });
        pp.label(`t = ${PM.fmt(t, 2)}`, "tl");
        pp.legend([{ label: "R region", color: PlotColors.blue, type: "box" }, { label: "T region", color: PlotColors.good, type: "box" }], "tr");

        // T, R history
        const ptr = plots.tr;
        ptr.clear();
        ptr.hline(Tinf, { color: PlotColors.good, dash: [2, 4], alpha: 0.8 });
        const a = hTt.subarray(0, hn);
        ptr.line(a, hT.subarray(0, hn), { color: PlotColors.good, width: 2 });
        ptr.line(a, hR.subarray(0, hn), { color: PlotColors.blue, width: 2 });
        ptr.line(a, hB.subarray(0, hn), { color: PlotColors.accent3, width: 1.4, dash: [5, 4] });
        ptr.legend([{ label: "T (transmitted)", color: PlotColors.good }, { label: "R (reflected)", color: PlotColors.blue },
          { label: "between barriers", color: PlotColors.accent3, dash: [5, 4] }, { label: "predicted T∞", color: PlotColors.good, dash: [2, 4] }], "tr");

        // momentum distribution
        fr.set(re); fi.set(im);
        PM.fft(fr, fi, false);
        const c = (dx * dx) / (2 * Math.PI);
        let pmax = 0;
        for (let i = 0; i < N; i++) { const j = (i + N / 2) % N; pk[i] = (fr[j] * fr[j] + fi[j] * fi[j]) * c; if (pk[i] > pmax) pmax = pk[i]; }
        const pm = plots.mom, p0 = (P.sigma / Math.sqrt(Math.PI)), ytop = 1.15 * Math.max(p0, pmax);
        pm.setLimits(null, [0, ytop]);
        pm.clear();
        if (P.showTk) {
          for (let j = 0; j < NK; j++) TkPlot[j] = Tk[j] * ytop * 0.97;
          pm.line(kT, TkPlot, { color: PlotColors.good, width: 1.4, dash: [4, 3], alpha: 0.9 });
        }
        pm.fill(ksh, pk, 0, { color: PlotColors.pink, alpha: 0.3 });
        pm.line(ksh, pk, { color: PlotColors.pink, width: 1.6 });
        pm.vline(0, { color: PlotColors.muted, dash: [3, 4] });
        pm.vline(Math.sqrt(2 * P.V0), { color: PlotColors.accent3, dash: [6, 4], alpha: 0.8 });
        const lm = [{ label: "|φ(k,t)|²", color: PlotColors.pink }, { label: "k = √(2V₀)", color: PlotColors.accent3, dash: [6, 4] }];
        if (P.showTk) lm.push({ label: "T(k), 0…1 full height", color: PlotColors.good, dash: [4, 3] });
        pm.legend(lm, "tl");

        Mt.set("T", PM.fmt(m.T, 4));
        Mt.set("R", PM.fmt(m.R, 4));
        Mt.set("B", PM.fmt(m.B, 4));
        Mt.set("Tinf", PM.fmt(Tinf, 4));
        Mt.set("Trect", PM.fmt(Trect, 4));
        Mt.set("Tinc", PM.fmt(Tinc, 4));
        Mt.set("E", PM.fmt(Emean / P.V0, 3));
        Mt.set("nrm", PM.fmt(m.tot, 6));
        api.setTime(`t = ${PM.fmt(t, 2)}`);
      },
    };
  },
});
