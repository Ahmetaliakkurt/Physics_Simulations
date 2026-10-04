/* Fermi–Dirac, Bose–Einstein and Maxwell–Boltzmann occupation numbers — analytic curves updated instantly,
 * with an optional temperature-sweep animation. */
App.register({
  id: "quantum-statistics",
  category: "statistical",
  order: 22,
  title: "Fermi–Dirac, Bose–Einstein & Maxwell–Boltzmann Statistics",
  icon: "📊",
  subtitle: "Mean occupation number of a single-particle state versus energy for fermions, bosons and classical particles: watch the Fermi step soften and all three curves merge into the Boltzmann tail as the temperature rises.",
  notes: [{ type: "info", html: "All three laws follow from the same grand-canonical calculation; they differ only in how many particles one quantum state may hold. The energy axis is in units of the Fermi energy $E_f$; Fermi–Dirac uses $\\mu=E_f$, Bose–Einstein uses $\\mu=0$ (as for photons and phonons) and the Maxwell–Boltzmann curve is normalised to its largest value. Tick <b>Temperature sweep</b> to animate $T$." }],
  animated: false,
  controls: [
    { id: "T", type: "slider", label: "Temperature $T$", min: 10, max: 3000, step: 10, value: 300, unit: "K", live: true },
    { id: "Ef", type: "slider", label: "Fermi energy $E_f$", min: 0.1, max: 5, step: 0.1, value: 1, unit: "eV", live: true },
    { id: "emax", type: "slider", label: "Energy-axis maximum $E/E_f$", min: 1, max: 5, step: 0.1, value: 2.5, live: true },
    { id: "ymax", type: "slider", label: "Occupation-axis maximum", min: 1, max: 3, step: 0.1, value: 1.2, live: true },
    { type: "section", label: "Curves" },
    { id: "showFD", type: "checkbox", label: "Fermi–Dirac", value: true, live: true },
    { id: "showBE", type: "checkbox", label: "Bose–Einstein", value: true, live: true },
    { id: "showMB", type: "checkbox", label: "Maxwell–Boltzmann", value: true, live: true },
    { id: "showWin", type: "checkbox", label: "Show the thermal window $E_f \\pm 2k_BT$", value: true, live: true },
    { type: "section", label: "Animation" },
    { id: "sweep", type: "checkbox", label: "▶ Temperature sweep (10 K ↔ 3000 K)", value: false, live: true,
      help: "The temperature is swept back and forth on a logarithmic scale; moving the slider by hand makes the sweep continue from that value." },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>Consider an ideal (non-interacting) gas of identical particles in thermal and diffusive equilibrium with a reservoir at
    temperature $T$ and chemical potential $\\mu$. Each single-particle quantum state of energy $E$ is occupied by some number of
    particles $n$; the question is the <b>mean occupation number</b> $\\langle n(E)\\rangle$. Three cases are compared:</p>
    <ul>
      <li><b>Fermions</b> (electrons, protons, ³He): the Pauli principle allows $n\\in\\{0,1\\}$ per state. The curve uses
      $\\mu=E_f$, the Fermi energy of the slider (typical metals: $E_f\\approx2$–$10$ eV).</li>
      <li><b>Bosons</b> (photons, phonons, ⁴He): any $n=0,1,2,\\dots$. The curve uses $\\mu=0$, appropriate for particles whose number
      is not conserved (black-body photons, phonons).</li>
      <li><b>Classical, distinguishable particles</b>: the Maxwell–Boltzmann limit, valid when every state is sparsely occupied.</li>
    </ul>
    <p>Units: $E$ and $E_f$ in eV, $T$ in kelvin, $k_B=8.617333262\\times10^{-5}$ eV/K (so $k_BT\\approx25.9$ meV at 300 K).</p>

    <h4>Equations being solved</h4>
    <p>In the grand canonical ensemble a single state of energy $E$ is itself a small system that can exchange particles and energy
    with the reservoir. The probability of finding it with $n$ particles is $\\propto e^{-\\beta n(E-\\mu)}$, $\\beta=1/k_BT$, so its
    grand partition function and mean occupation are</p>
    $$\\mathcal Z=\\sum_{n}e^{-\\beta n(E-\\mu)},\\qquad \\langle n\\rangle=\\frac{\\sum_n n\\,e^{-\\beta n(E-\\mu)}}{\\mathcal Z}=-\\frac{1}{\\beta}\\frac{\\partial\\ln\\mathcal Z}{\\partial E}.$$
    <p><b>Fermions</b>, $n\\in\\{0,1\\}$: $\\mathcal Z=1+e^{-\\beta(E-\\mu)}$. <b>Bosons</b>, $n=0,1,2,\\dots$: a geometric series,
    $\\mathcal Z=1/(1-e^{-\\beta(E-\\mu)})$, which converges only if $E&gt;\\mu$. Differentiating gives the two quantum distributions:</p>
    <div class="callout">$$f_{FD}(E)=\\frac{1}{e^{(E-\\mu)/k_BT}+1},\\qquad f_{BE}(E)=\\frac{1}{e^{(E-\\mu)/k_BT}-1}\\ \\ (E&gt;\\mu),\\qquad
      f_{MB}(E)=e^{-(E-\\mu)/k_BT}.$$</div>
    <p>When $e^{(E-\\mu)/k_BT}\\gg1$ the $\\pm1$ in the denominators is negligible and both quantum laws reduce to the Boltzmann factor:
    occupancies are so small that the chance of two particles competing for one state — the only place where the statistics
    matter — vanishes. In terms of $x=(E-\\mu)/k_BT$ the three curves are universal functions $1/(e^x\\pm1)$ and $e^{-x}$,
    satisfying $f_{BE}&gt;f_{MB}&gt;f_{FD}$ for every $x&gt;0$: bosons "bunch", fermions "avoid each other".</p>
    <p>Useful properties: $f_{FD}(\\mu)=\\tfrac12$ exactly; $f_{FD}(\\mu+\\epsilon)=1-f_{FD}(\\mu-\\epsilon)$ (particle–hole symmetry);
    the Fermi step falls from 0.9 to 0.1 over $\\Delta E=2\\ln9\\,k_BT\\approx4.39\\,k_BT$; and for $\\mu=0$,
    $f_{BE}\\approx k_BT/E$ as $E\\to0$ (the classical equipartition limit of a field mode).</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li>The formulas above are evaluated analytically (no sampling) at 800 energies from 0.001 eV to $E_{\\max}=(E/E_f)_{\\max}\\,E_f$,
      whenever a parameter changes. $f_{BE}$ is computed as <code>1/expm1(x)</code> to stay accurate for small $x$; it diverges at
      $E\\to\\mu=0$ and is clipped by the axis.</li>
      <li><b>Main plot:</b> $f_{FD}$ with $\\mu=E_f$, $f_{BE}$ with $\\mu=0$, and $f_{MB}\\propto e^{-E/k_BT}$ divided by its value at the
      first grid point (so it starts at 1 and only its shape matters — the absolute MB normalisation depends on $\\mu$, which is
      different for each physical system). The shaded band is $E_f\\pm2k_BT$, where the Fermi function changes.</li>
      <li><b>Lower left:</b> $f_{FD}$ at 10, 100, 300, 1000 and 3000 K for the current $E_f$, with the current $T$ highlighted.</li>
      <li><b>Lower right:</b> the universal forms versus $x=(E-\\mu)/k_BT$ on a logarithmic axis; for $x\\gtrsim3$ the three lines
      coincide within 5%.</li>
      <li><b>Sweep:</b> $\\ln T$ moves linearly between $\\ln10$ and $\\ln3000$, 7 s per direction.</li>
      <li><b>Metrics:</b> $k_BT$; the degeneracy parameter $E_f/k_BT$ (≫1: degenerate quantum gas, ≲1: classical); the 90%–10% width
      of the Fermi step in units of $E_f$, $4.39\\,k_BT/E_f$; and $f_{BE}(E_f)=1/(e^{E_f/k_BT}-1)$.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Room-temperature metal:</b> $E_f=5$ eV, $T=300$ K: $E_f/k_BT\\approx194$, the Fermi function is an almost perfect step,
      and only electrons within $\\sim0.1$ eV of $E_f$ can be thermally excited — the reason the electronic heat capacity of metals
      is so small ($\\propto T/T_F$).</li>
      <li><b>Cold limit:</b> $T=10$ K: $k_BT=0.86$ meV; the step width is $3.8$ meV, invisible on the plot.</li>
      <li><b>Classical limit:</b> $E_f=0.1$ eV, $T=3000$ K: $E_f/k_BT\\approx0.39$, the Fermi "step" is a smooth decay and above
      $E_f$ it is practically the Boltzmann exponential.</li>
      <li><b>Bose enhancement:</b> with $\\mu=0$, $f_{BE}$ exceeds 1 for $E&lt;k_BT\\ln2$; raise $T$ and watch the low-energy occupancy
      grow — the seed of Bose–Einstein condensation and of the Rayleigh–Jeans divergence $\\propto k_BT/E$.</li>
      <li>In the universal plot, read off $x=3$: $f_{FD}=0.047$, $f_{MB}=0.050$, $f_{BE}=0.052$ — the statistics barely matter once
      a state is less than ~5% occupied.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The chemical potentials are held fixed. For a real 3D electron gas $\\mu$ itself drifts with temperature,
    $\\mu(T)\\approx E_f\\big[1-\\tfrac{\\pi^2}{12}(k_BT/E_f)^2\\big]$ (Sommerfeld), and for a conserved-number Bose gas $\\mu&lt;0$ rises
    towards 0 as $T\\to T_c$. Interactions and the density of states $g(E)$ (needed for actual particle numbers $N=\\int g f\\,dE$) are
    not included. References: C. Kittel &amp; H. Kroemer, <i>Thermal Physics</i>, ch. 6–7; D. V. Schroeder, <i>An Introduction to
    Thermal Physics</i>, ch. 7; R. K. Pathria &amp; P. D. Beale, <i>Statistical Mechanics</i>, ch. 6–8; N. W. Ashcroft &amp;
    N. D. Mermin, <i>Solid State Physics</i>, ch. 2.</p>`,

  mount(api) {
    const P = api.params, KB = 8.617333262e-5, NE = 800;
    const M = api.metrics([
      { id: "kt", label: "$k_B T$" },
      { id: "ratio", label: "Degeneracy $E_f / k_B T$" },
      { id: "width", label: "Fermi-step width (90%→10%) / $E_f$" },
      { id: "be", label: "Bose–Einstein $f(E=E_f)$" },
    ]);
    const plots = api.plots([
      { id: "main", title: "Comparison of the three statistics", span: 2, aspect: 0.42, xlabel: "E / E_f", ylabel: "mean occupation ⟨n⟩" },
      { id: "fdT", title: "Softening of the Fermi–Dirac step with temperature", aspect: 0.62, xlim: [0, 2], ylim: [0, 1.05], xlabel: "E / E_f", ylabel: "f_FD" },
      { id: "univ", title: "Universal form vs. x = (E−μ)/k_BT (log scale)", aspect: 0.62, xlim: [-4, 10], ylim: [1e-4, 30], ylog: true, xlabel: "x = (E − μ) / k_BT", ylabel: "f(x)" },
    ]);
    const E = new Float64Array(NE), X = new Float64Array(NE), fd = new Float64Array(NE), be = new Float64Array(NE), mb = new Float64Array(NE);
    const Ts = [10, 100, 300, 1000, 3000], xs = PM.linspace(0, 2, 400), ys = new Float64Array(400);
    const LT0 = Math.log(10), LT1 = Math.log(3000);
    let T = P.T, sweepU = (Math.log(T) - LT0) / (LT1 - LT0), sweepDir = 1, lastSlider = 0;

    const fermi = (e, mu, T) => 1 / (Math.exp((e - mu) / (KB * T)) + 1);
    const bose = (e, mu, T) => (e <= mu ? Infinity : 1 / Math.expm1((e - mu) / (KB * T)));

    function compute() {
      const Ef = P.Ef, e0 = 0.001, e1 = P.emax * Ef;
      for (let i = 0; i < NE; i++) {
        const e = e0 + ((e1 - e0) * i) / (NE - 1);
        E[i] = e; X[i] = e / Ef;
        fd[i] = fermi(e, Ef, T);
        be[i] = bose(e, 0, T);
        mb[i] = Math.exp(-(e - e0) / (KB * T)); // exp(−E/kT) / max  (the maximum is at the first point)
      }
    }

    return {
      reset() { T = P.T; },
      onParam(id, v) {
        if (id === "T") { T = v; sweepU = (Math.log(T) - LT0) / (LT1 - LT0); lastSlider = v; }
        if (id === "sweep") { if (v) api.play(); else api.pause(); }
      },
      step(dt) {
        if (!P.sweep) { api.pause(); return; }
        sweepU += sweepDir * dt / 7; // one direction ≈ 7 s
        if (sweepU >= 1) { sweepU = 1; sweepDir = -1; } else if (sweepU <= 0) { sweepU = 0; sweepDir = 1; }
        T = Math.exp(LT0 + sweepU * (LT1 - LT0));
        const ts = Math.round(T / 10) * 10;
        if (ts !== lastSlider) { lastSlider = ts; api.setControl("T", { value: ts }); }
      },
      render() {
        compute();
        const Ef = P.Ef, kT = KB * T;
        const pm = plots.main;
        pm.setLimits([0, P.emax], [0, P.ymax]);
        pm.clear();
        if (P.showWin) {
          pm.rect(Math.max(0, 1 - (2 * kT) / Ef), 0, 1 + (2 * kT) / Ef, P.ymax, { color: PlotColors.accent, alpha: 0.07 });
          pm.vline(1, { color: PlotColors.muted, dash: [4, 4], width: 1 });
        }
        pm.hline(0.5, { color: PlotColors.muted, dash: [2, 5], width: 0.8, alpha: 0.6 });
        const leg = [];
        if (P.showFD) { pm.line(X, fd, { color: PlotColors.accent, width: 2.4 }); leg.push({ label: "Fermi–Dirac (μ = E_f)", color: PlotColors.accent }); }
        if (P.showBE) { pm.line(X, be, { color: PlotColors.accent2, width: 2.4, dash: [8, 5] }); leg.push({ label: "Bose–Einstein (μ = 0)", color: PlotColors.accent2, dash: [8, 5] }); }
        if (P.showMB) { pm.line(X, mb, { color: PlotColors.accent3, width: 2.4, dash: [10, 4, 2, 4] }); leg.push({ label: "Maxwell–Boltzmann (normalised)", color: PlotColors.accent3, dash: [10, 4, 2, 4] }); }
        if (leg.length) pm.legend(leg, "tr");
        pm.label(`T = ${Math.round(T)} K · k_BT = ${PM.fmt(kT * 1000, 2)} meV`, "tl", { size: 13 });

        // --- FD step at several temperatures (fixed E_f)
        const pf = plots.fdT;
        pf.clear();
        Ts.forEach((Tk, k) => {
          for (let i = 0; i < 400; i++) ys[i] = fermi(xs[i] * Ef, Ef, Tk);
          pf.line(xs, ys, { color: colormap("plasma", 0.15 + (0.75 * k) / (Ts.length - 1)), width: 1.3, alpha: 0.55 });
        });
        for (let i = 0; i < 400; i++) ys[i] = fermi(xs[i] * Ef, Ef, T);
        pf.line(xs, ys, { color: PlotColors.accent, width: 2.8 });
        pf.legend([{ label: `current T = ${Math.round(T)} K`, color: PlotColors.accent }].concat(Ts.map((Tk, k) => ({ label: `${Tk} K`, color: colormap("plasma", 0.15 + (0.75 * k) / (Ts.length - 1)) }))), "tr");

        // --- universal form
        const pu = plots.univ;
        pu.clear();
        pu.fn((x) => 1 / (Math.exp(x) + 1), { color: PlotColors.accent, width: 2.2, samples: 400 });
        pu.fn((x) => (x > 0 ? 1 / Math.expm1(x) : NaN), { color: PlotColors.accent2, width: 2.2, dash: [8, 5], samples: 400 });
        pu.fn((x) => Math.exp(-x), { color: PlotColors.accent3, width: 2.2, dash: [10, 4, 2, 4], samples: 400 });
        pu.vline(0, { color: PlotColors.muted, dash: [4, 4], width: 1 });
        pu.legend([{ label: "FD: 1/(eˣ+1)", color: PlotColors.accent }, { label: "BE: 1/(eˣ−1)", color: PlotColors.accent2, dash: [8, 5] }, { label: "MB: e⁻ˣ", color: PlotColors.accent3, dash: [10, 4, 2, 4] }], "tr");

        M.set("kt", PM.fmt(kT * 1000, 3) + " meV");
        M.set("ratio", PM.fmt(Ef / kT, 2));
        M.set("width", PM.fmt((2 * Math.log(9) * kT) / Ef, 4));
        M.set("be", PM.fmt(bose(Ef, 0, T), 3));
      },
    };
  },
});
