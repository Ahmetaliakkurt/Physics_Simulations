/* Electron in an intense laser pulse — live RK4 integration of the relativistic equation of motion (momentum form). */
App.register({
  id: "electron-laser",
  category: "classical",
  group: "Electromagnetism",
  order: 25,
  title: "Electron in an Intense Laser Pulse",
  icon: "⚡",
  subtitle: "A free electron hit by a plane electromagnetic pulse with a Gaussian envelope: it oscillates sideways, is pushed forward, its speed saturates below $c$ — and after the pulse it is left at rest, but displaced.",
  notes: [{ type: "info", html: "Top: the incoming field $E_y(x,t)$ and the electron (red dot). Middle: the electron's trajectory (figure-of-eight-like oscillation plus forward drift). Bottom: speed and Lorentz factor; the dashed red curve is the non-relativistic prediction, which exceeds the speed of light for $a_0&gt;1$." }],
  animated: true,
  speed: { min: 0.25, max: 4, value: 1, step: 0.05 },
  controls: [
    { id: "E0", type: "slider", label: "Field amplitude $E_0$", min: 1, max: 30, step: 0.5, value: 9,
      fmt: (v) => `${v.toFixed(1)}  (a₀ = ${(v / 10).toFixed(2)})`,
      help: "Normalised amplitude $a_0 = |q|E_0/(m\\omega c)$. For $a_0 \\gtrsim 1$ the motion becomes strongly relativistic." },
    { type: "section", label: "Display" },
    { id: "newton", type: "checkbox", label: "Show the non-relativistic (Newtonian) prediction", value: true, live: true },
    { id: "env", type: "checkbox", label: "Show the pulse envelope", value: true, live: true },
    { type: "info", html: "Units: $c = m = |q| = 1$, $\\omega = 10$, $k = \\omega/c$. At $t=0$ the pulse centre is 10 length units to the left of the electron. The run lasts $t = 60$ and then restarts." },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A free electron (charge $q=-1$, mass $m=1$), initially at rest at the origin, is overtaken by a linearly polarised plane
    electromagnetic wave packet travelling in the $+x$ direction — an idealised model of a short, intense laser pulse.
    Natural units are used: $c=m=|q|=1$, so velocities are in units of $c$, momenta in units of $mc$ and energies in units of $mc^2$.
    The carrier has angular frequency $\\omega=10$ (wavenumber $k=\\omega/c=10$, wavelength $2\\pi/10\\approx0.63$), the Gaussian envelope
    has an rms length of $\\sigma=3$ length units (≈ 5 wavelengths) and its centre starts 10 units behind the electron.
    The only control is the field amplitude $E_0$; the dimensionless strength parameter is</p>
    $$a_0=\\frac{|q|E_0}{m\\omega c}=\\frac{E_0}{10},$$
    <p>the peak transverse momentum (in units of $mc$) the field can give the electron. $a_0\\ll1$ is the classical, non-relativistic regime;
    $a_0\\gtrsim1$ (reached by real lasers at intensities above $\\sim10^{18}$ W/cm² for 1 μm light) is the relativistic regime.
    Radiation reaction, space charge and the transverse profile of a real focused beam are neglected.</p>

    <h4>Equations being solved</h4>
    <p>The fields of the pulse (vacuum plane wave, so $|\\mathbf B|=|\\mathbf E|/c$ and $\\mathbf E\\perp\\mathbf B\\perp\\hat x$):</p>
    $$E_y(x,t)=E_0\\cos(kx-\\omega t)\\,\\exp\\!\\Big[-\\frac{(x-ct+10)^2}{2\\sigma^2}\\Big],\\qquad B_z=\\frac{E_y}{c}.$$
    <p>The relativistic Lorentz-force law, with the momentum as the dynamical variable:</p>
    <div class="callout">$$\\frac{d\\mathbf p}{dt}=q\\big(\\mathbf E+\\mathbf v\\times\\mathbf B\\big),\\qquad
      \\mathbf v=\\frac{\\mathbf p}{m\\gamma},\\qquad \\gamma=\\sqrt{1+\\frac{p^2}{m^2c^2}} .$$</div>
    <p>In components ($\\mathbf v\\times\\mathbf B=(v_yB_z,\\,-v_xB_z,\\,0)$): $\\dot p_x=q\\,v_yB_z$ — the magnetic force pushes the electron
    forward —, and $\\dot p_y=q(E_y-v_xB_z)$. Because $v=p/(m\\gamma)$, the speed tends to $c$ as $p\\to\\infty$ but never exceeds it.
    The Newtonian comparison uses $m\\,d\\mathbf v/dt=q(\\mathbf E+\\mathbf v\\times\\mathbf B)$, which has no speed limit; its peak transverse
    speed is $v_{\\max}\\approx|q|E_0/(m\\omega)=a_0c$.</p>
    <p><b>Exact invariants.</b> The fields depend on $x$ and $t$ only through $\\xi=x-ct$. With the vector potential $A_y(\\xi)$
    ($E_y=-\\partial_tA_y$), the canonical transverse momentum and the "light-front" combination are conserved
    (electron initially at rest):</p>
    $$p_y+qA_y(\\xi)=0,\\qquad \\gamma-\\frac{p_x}{mc}=1 .$$
    <p>Combining them with $\\gamma^2=1+p_x^2+p_y^2$ gives $p_x=p_y^2/2mc$ and $\\gamma=1+p_y^2/2m^2c^2$. With $|p_y|_{\\max}\\approx a_0mc$:</p>
    $$\\gamma_{\\max}\\approx1+\\frac{a_0^2}{2},\\qquad p_{x,\\max}\\approx\\frac{a_0^2}{2}mc .$$
    <p>The transverse quiver grows like $a_0$, the forward push like $a_0^2$ (ponderomotive drift). When the pulse has passed, $A_y\\to0$,
    so $p_y\\to0$, $p_x\\to0$: the electron is left at rest — no net energy gain from a plane wave in vacuum (Lawson–Woodward theorem) —
    but displaced forward.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li>Two independent integrations run side by side: the relativistic one with state $(x,y,p_x,p_y)$ and the Newtonian one with
      state $(x,y,v_x,v_y)$. Both use the classical fourth-order Runge–Kutta method with a fixed step $\\Delta t=0.01$
      (63 steps per optical period, $\\omega\\Delta t=0.1$), so the integration error is far below plotting resolution.</li>
      <li>Playback: 4 time units per real second at speed ×1; every 2nd step is recorded for the plots. The run lasts $t=60$
      (the pulse has fully passed by then), pauses briefly and restarts.</li>
      <li>Before each run a fast pre-pass integrates the whole run once to fix the axis ranges, so the plots do not rescale while
      the electron moves; the animation itself is computed live.</li>
      <li>Plotted: the field $E_y(x,t)$ on $-10\\le x\\le25$ (1400 points) with its envelope; the trajectory $(x,y)$; speed $|\\mathbf v|/c$ for
      both models with the line $v=c$; $\\gamma(t)$ with the estimate $1+a_0^2/2$.</li>
      <li>Accuracy check: the invariant $\\gamma-p_x/mc$ is shown as a metric; it must stay equal to 1 (to $\\lesssim10^{-6}$ for moderate fields, ~$10^{-4}$ at the largest amplitude, where the electron's motion is fastest).
      The metric "$\\gamma_{\\max}$ sim / theory" compares the measured peak with $1+a_0^2/2$.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li>$E_0=3$ ($a_0=0.3$): nearly non-relativistic — the relativistic and Newtonian speeds almost coincide, $\\gamma_{\\max}\\approx1.045$,
      the forward drift is tiny and the trajectory is a thin figure-of-eight smeared along $x$.</li>
      <li>Default $E_0=9$ ($a_0=0.9$): the Newtonian speed peaks near $0.9c$ while the relativistic one stays lower; $\\gamma_{\\max}\\approx1.4$.</li>
      <li>$E_0=20$ ($a_0=2$): the Newtonian speed reaches $\\approx2c$ — clearly unphysical — while the true speed saturates just below $c$;
      $\\gamma_{\\max}$ approaches $1+a_0^2/2=3$ (a little below it, because the electron surfs ahead of the pulse peak) and the forward drift now dominates the motion.</li>
      <li>Watch the electron surf: while it moves forward at nearly $c$ it sees the wave with a Doppler-reduced frequency
      $\\omega(1-v_x/c)$, so it stays in the pulse much longer than the pulse duration — the oscillations in the trajectory become longer.</li>
      <li>After the pulse: speed back to zero, $\\gamma=1$, but $x$ has advanced — the net displacement grows roughly like $a_0^2$.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>Plane-wave pulse (no focusing, so no transverse ponderomotive force and no net acceleration), no radiation reaction (relevant only for
    $a_0\\gtrsim100$), a single test electron. Reading: J. D. Jackson, <i>Classical Electrodynamics</i>, ch. 12 (relativistic dynamics in fields);
    Landau &amp; Lifshitz, <i>The Classical Theory of Fields</i>, §47 problem 2 (motion in a plane wave); E. Esarey, P. Sprangle &amp; J. Krall,
    Phys. Rev. E 52, 5443 (1995); P. Gibbon, <i>Short Pulse Laser Interactions with Matter</i>, ch. 3.</p>`,

  mount(api) {
    const P = api.params;
    const C = 1, Q = -1, MASS = 1, OMEGA = 10, K = OMEGA / C, SIG = 3, T_TOTAL = 60, H = 0.01;
    const RATE = 4; // simulation time per real second (× speed)
    const REC = 2;  // record every REC internal steps
    const NREC = Math.ceil(T_TOTAL / H / REC) + 2;
    const XW0 = -10, XW1 = 25, NW = 1400;

    const M = api.metrics([
      { id: "t", label: "Time $t$" },
      { id: "v", label: "Speed $v/c$" },
      { id: "g", label: "Lorentz factor $\\gamma$" },
      { id: "vmax", label: "Max $v/c$ so far" },
      { id: "gmax", label: "$\\gamma_{\\max}$: sim / $1+a_0^2/2$" },
      { id: "inv", label: "$\\gamma - p_x/mc$ (must be 1)" },
    ]);
    const plots = api.plots([
      { id: "wave", title: "Incoming electromagnetic pulse $E_y(x,t)$ and the electron position", span: 2, aspect: 0.27, xlim: [XW0, XW1], ylim: [-10, 10], xlabel: "x", ylabel: "Eᵧ", minHeight: 200 },
      { id: "traj", title: "Electron trajectory (transverse quiver + forward drift)", span: 2, aspect: 0.25, xlim: [0, 1], ylim: [-1, 1], xlabel: "x (drift direction)", ylabel: "y (quiver)", minHeight: 190 },
      { id: "vel", title: "Speed $v/c$", aspect: 0.62, xlim: [0, T_TOTAL], ylim: [0, 1.1], xlabel: "t", ylabel: "v / c" },
      { id: "gam", title: "Lorentz factor $\\gamma$", aspect: 0.62, xlim: [0, T_TOTAL], ylim: [1, 2], xlabel: "t", ylabel: "γ" },
    ]);

    // field E_y(x, t)
    function field(x, t) {
      const u = (x - (C * t - 10)) / SIG;
      return P.E0 * Math.cos(K * x - OMEGA * t) * Math.exp(-0.5 * u * u);
    }
    // relativistic: s = [x, y, px, py]
    function dRel(t, s, o) {
      const px = s[2], py = s[3];
      const g = Math.sqrt(1 + (px * px + py * py) / (MASS * C) ** 2);
      const vx = px / (MASS * g), vy = py / (MASS * g);
      const Ey = field(s[0], t), Bz = Ey / C;
      o[0] = vx; o[1] = vy;
      o[2] = Q * (vy * Bz);          // (v×B)_x = vy·Bz
      o[3] = Q * (Ey - vx * Bz);     // (v×B)_y = −vx·Bz
    }
    // Newtonian: s = [x, y, vx, vy]
    function dNew(t, s, o) {
      const vx = s[2], vy = s[3];
      const Ey = field(s[0], t), Bz = Ey / C;
      o[0] = vx; o[1] = vy;
      o[2] = (Q / MASS) * (vy * Bz);
      o[3] = (Q / MASS) * (Ey - vx * Bz);
    }

    const sR = new Float64Array(4), sN = new Float64Array(4);
    let wsR = null, wsN = null, t = 0, nstep = 0, nrec = 0, finished = false, holdT = 0;
    const tr = new Float64Array(NREC), xr = new Float64Array(NREC), yr = new Float64Array(NREC);
    const vr = new Float64Array(NREC), gr = new Float64Array(NREC), vn = new Float64Array(NREC);
    const xw = PM.linspace(XW0, XW1, NW), ew = new Float64Array(NW), envP = new Float64Array(NW), envM = new Float64Array(NW);
    let bounds = null, vmaxSeen = 0, gmaxSeen = 1;

    const gammaOf = (s) => Math.sqrt(1 + (s[2] * s[2] + s[3] * s[3]) / (MASS * C) ** 2);

    // Fast pre-pass to fix the axis ranges (bounds only; the animation itself is computed live)
    function prepass() {
      const a = new Float64Array(4), b = new Float64Array(4);
      let w1 = null, w2 = null, tt = 0;
      const B = { x0: 0, x1: 0, y0: 0, y1: 0, vR: 0, vN: 0, g: 1 };
      const n = Math.round(T_TOTAL / H);
      for (let i = 0; i < n; i++) {
        w1 = PM.rk4(dRel, tt, a, H, w1); w2 = PM.rk4(dNew, tt, b, H, w2); tt += H;
        B.x0 = Math.min(B.x0, a[0]); B.x1 = Math.max(B.x1, a[0]);
        B.y0 = Math.min(B.y0, a[1]); B.y1 = Math.max(B.y1, a[1]);
        const g = gammaOf(a);
        B.g = Math.max(B.g, g);
        B.vR = Math.max(B.vR, Math.hypot(a[2], a[3]) / (MASS * g) / C);
        B.vN = Math.max(B.vN, Math.hypot(b[2], b[3]) / C);
      }
      return B;
    }

    function record() {
      if (nrec >= NREC) return;
      const g = gammaOf(sR);
      tr[nrec] = t; xr[nrec] = sR[0]; yr[nrec] = sR[1];
      vr[nrec] = Math.hypot(sR[2], sR[3]) / (MASS * g) / C;
      gr[nrec] = g;
      vn[nrec] = Math.hypot(sN[2], sN[3]) / C;
      vmaxSeen = Math.max(vmaxSeen, vr[nrec]);
      gmaxSeen = Math.max(gmaxSeen, g);
      nrec++;
    }

    function setupAxes() {
      const B = bounds;
      plots.wave.setLimits(null, [-1.18 * P.E0, 1.18 * P.E0]);
      const dx = Math.max(B.x1 - B.x0, 1e-3), dy = Math.max(B.y1 - B.y0, 1e-3);
      plots.traj.setLimits([B.x0 - 0.04 * dx, B.x1 + 0.04 * dx], [B.y0 - 0.12 * dy, B.y1 + 0.12 * dy]);
      applyVelAxis();
      const gp = Math.max(0.05 * (B.g - 1), 0.002);
      plots.gam.setLimits(null, [1 - gp, B.g + gp]);
    }
    function applyVelAxis() {
      const top = Math.max(1.1, P.newton ? bounds.vN * 1.06 : 0, bounds.vR * 1.06);
      plots.vel.setLimits(null, [0, top]);
    }

    return {
      reset() {
        sR.fill(0); sN.fill(0); wsR = wsN = null;
        t = 0; nstep = 0; nrec = 0; finished = false; holdT = 0; vmaxSeen = 0; gmaxSeen = 1;
        bounds = prepass();
        setupAxes();
        record();
      },
      onParam(id) { if (id === "newton" && bounds) applyVelAxis(); },
      step(dt) {
        if (finished) { // short hold, then start over
          holdT += dt;
          if (holdT > 1.5) this.reset();
          return;
        }
        const n = Math.max(1, Math.round((dt * RATE) / H));
        for (let i = 0; i < n && !finished; i++) {
          wsR = PM.rk4(dRel, t, sR, H, wsR);
          wsN = PM.rk4(dNew, t, sN, H, wsN);
          t += H; nstep++;
          if (nstep % REC === 0) record();
          if (t >= T_TOTAL - 1e-9) finished = true;
        }
      },
      render() {
        const g = gammaOf(sR);
        const v = Math.hypot(sR[2], sR[3]) / (MASS * g) / C;
        const a0 = P.E0 / OMEGA, gth = 1 + (a0 * a0) / 2;

        // --- pulse
        const pw = plots.wave;
        pw.clear();
        for (let i = 0; i < NW; i++) {
          const u = (xw[i] - (C * t - 10)) / SIG, e = P.E0 * Math.exp(-0.5 * u * u);
          envP[i] = e; envM[i] = -e;
          ew[i] = e * Math.cos(K * xw[i] - OMEGA * t);
        }
        if (P.env) {
          pw.fill(xw, envP, envM, { color: PlotColors.accent, alpha: 0.07 });
          pw.line(xw, envP, { color: PlotColors.muted, width: 1, dash: [5, 4], alpha: 0.7 });
          pw.line(xw, envM, { color: PlotColors.muted, width: 1, dash: [5, 4], alpha: 0.7 });
        }
        pw.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
        pw.line(xw, ew, { color: PlotColors.accent, width: 1.5 });
        pw.vline(sR[0], { color: PlotColors.bad, dash: [3, 4], width: 1, alpha: 0.6 });
        pw.circle(sR[0], 0, 6, { px: true, color: PlotColors.bad, stroke: "#0f151c" });
        pw.label(`t = ${PM.fmt(t, 2)}   ·   a₀ = ${PM.fmt(a0, 2)}   ·   pulse moves → at c`, "tl");
        pw.legend([{ label: "Eᵧ", color: PlotColors.accent }, { label: "electron", color: PlotColors.bad, type: "dot" }], "tr");

        // --- trajectory
        const pt = plots.traj;
        pt.clear();
        pt.hline(0, { color: PlotColors.muted, alpha: 0.5, width: 1 });
        if (nrec > 1) pt.line(xr.subarray(0, nrec), yr.subarray(0, nrec), { color: PlotColors.accent2, width: 1.3 });
        pt.circle(sR[0], sR[1], 6, { px: true, color: PlotColors.accent2, stroke: PlotColors.text });
        pt.label(`drift x = ${PM.fmt(sR[0], 3)}`, "tl", { size: 11 });

        // --- speed
        const pv = plots.vel;
        pv.clear();
        pv.hline(1, { color: PlotColors.text, dash: [2, 4], width: 1.2, alpha: 0.8 });
        if (nrec > 1) {
          const T = tr.subarray(0, nrec);
          if (P.newton) pv.line(T, vn.subarray(0, nrec), { color: PlotColors.bad, width: 1.2, dash: [6, 4], alpha: 0.85 });
          pv.line(T, vr.subarray(0, nrec), { color: PlotColors.good, width: 1.7 });
        }
        pv.vline(t, { color: PlotColors.accent, dash: [4, 4], width: 1, alpha: 0.6 });
        const leg = [{ label: "relativistic", color: PlotColors.good }];
        if (P.newton) leg.push({ label: "Newtonian (F = ma)", color: PlotColors.bad, dash: [6, 4] });
        leg.push({ label: "c", color: PlotColors.text, dash: [2, 4] });
        pv.legend(leg, "tr");

        // --- gamma
        const pg = plots.gam;
        pg.clear();
        pg.hline(gth, { color: PlotColors.muted, dash: [4, 4], width: 1 });
        if (nrec > 1) pg.line(tr.subarray(0, nrec), gr.subarray(0, nrec), { color: PlotColors.accent3, width: 1.7 });
        pg.vline(t, { color: PlotColors.accent, dash: [4, 4], width: 1, alpha: 0.6 });
        pg.label(`γₘₐₓ ≈ 1 + a₀²/2 = ${PM.fmt(gth, 3)}`, "tr", { size: 11 });

        M.set("t", PM.fmt(t, 2));
        M.set("v", PM.fmt(v, 4));
        M.set("g", PM.fmt(g, 3));
        M.set("vmax", PM.fmt(vmaxSeen, 4));
        M.set("gmax", `${PM.fmt(gmaxSeen, 3)} / ${PM.fmt(gth, 3)}`);
        M.set("inv", PM.fmt(g - sR[2] / (MASS * C), 6));
        api.setTime(`t = ${PM.fmt(t, 1)} / ${T_TOTAL}`);
      },
    };
  },
});
