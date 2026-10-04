/* Faraday's law & electromagnetic induction — AC generator, magnet falling through a coil, sliding rod on rails.
 * Generator: exact closed-form current of a series R–L load. Magnet and rod: RK4 (sub-stepped) with the
 * induced-current back-reaction (Lenz braking), plus energy bookkeeping as an accuracy check. SI units throughout. */
(function () {
  "use strict";
  const MU0 = 4e-7 * Math.PI, G = 9.81;
  const C = () => PlotColors;
  const COPPER = "#d9904a", IND = "#ffd166", BCOL = "#58a6ff", FLUXP = "79,209,197", FLUXN = "139,92,246";

  // ------------------------------------------------------------------ helpers
  /** Playback factor (simulated seconds per real second) as text. */
  const slowTxt = (f) => (f < 1 ? `slow motion ×1/${PM.fmt(1 / f, 0)}` : `time-lapse ×${PM.fmt(f, 1)}`);
  /** Time-series buffer. mode "roll": drop the oldest half when full; "grow": keep every 2nd sample when full. */
  function Series(nmax, names, mode) {
    const s = { n: 0, decim: 1, t: new Float64Array(nmax), d: {} };
    for (const k of names) s.d[k] = new Float64Array(nmax);
    s.push = function (t, vals) {
      if (s.n >= nmax) {
        const h = nmax >> 1;
        if (mode === "roll") { s.t.copyWithin(0, h); for (const k of names) s.d[k].copyWithin(0, h); s.n = nmax - h; }
        else { for (let i = 0; i < h; i++) { s.t[i] = s.t[2 * i]; for (const k of names) s.d[k][i] = s.d[k][2 * i]; } s.n = h; s.decim *= 2; }
      }
      s.t[s.n] = t; for (const k of names) s.d[k][s.n] = vals[k]; s.n++;
    };
    s.T = () => s.t.subarray(0, s.n);
    s.Y = (k) => s.d[k].subarray(0, s.n);
    s.clear = () => { s.n = 0; s.decim = 1; };
    return s;
  }
  function maxAbs(a, n, from) { let m = 0; for (let i = from || 0; i < n; i++) { const v = Math.abs(a[i]); if (v > m && isFinite(v)) m = v; } return m; }
  /** Axis limit with hysteresis: grows immediately, shrinks only when the data use < 45 % of the range. */
  function Scaler() {
    let lim = 0;
    return { fit(m, floor) { const t = Math.max(m * 1.15, floor || 1e-12); if (t > lim || t < 0.45 * lim) lim = t; return lim; }, reset() { lim = 0; } };
  }
  /** First index with t >= t0 (time arrays are increasing). */
  function firstIdx(T, n, t0) { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (T[m] < t0) lo = m + 1; else hi = m; } return lo; }
  function arrowPx(c, x0, y0, x1, y1, col, w, head) {
    const a = Math.atan2(y1 - y0, x1 - x0), h = head || 8;
    c.strokeStyle = c.fillStyle = col; c.lineWidth = w || 2;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    c.beginPath(); c.moveTo(x1, y1);
    c.lineTo(x1 - h * Math.cos(a - 0.42), y1 - h * Math.sin(a - 0.42));
    c.lineTo(x1 - h * Math.cos(a + 0.42), y1 - h * Math.sin(a + 0.42)); c.closePath(); c.fill();
  }
  function chevron(c, x, y, ang, col, s) {
    s = s || 6; c.fillStyle = col; c.beginPath();
    c.moveTo(x + s * Math.cos(ang), y + s * Math.sin(ang));
    c.lineTo(x + s * Math.cos(ang + 2.5), y + s * Math.sin(ang + 2.5));
    c.lineTo(x + s * Math.cos(ang - 2.5), y + s * Math.sin(ang - 2.5)); c.closePath(); c.fill();
  }
  function txt(c, s, x, y, o) {
    o = o || {};
    c.font = (o.bold ? "600 " : "") + (o.size || 12) + 'px "Segoe UI", system-ui, sans-serif';
    c.textAlign = o.align || "left"; c.textBaseline = o.base || "middle";
    if (o.bg) { const w = c.measureText(s).width, h = (o.size || 12) + 6, bx = x - (c.textAlign === "center" ? w / 2 : c.textAlign === "right" ? w : 0) - 4;
      c.globalAlpha = 0.85; c.fillStyle = o.bg; c.fillRect(bx, y - h / 2, w + 8, h); c.globalAlpha = 1; }
    c.fillStyle = o.color || "#e6edf3"; c.fillText(s, x, y);
  }
  function resistorPx(c, x0, y0, x1, y1, glow, col) {
    // zig-zag resistor between two points (pixel coordinates)
    const L = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / L, uy = (y1 - y0) / L, nx = -uy, ny = ux, z = 7, k = 7;
    if (glow > 0.01) { c.save(); c.shadowColor = "rgba(255,120,60,0.9)"; c.shadowBlur = 22 * glow; c.strokeStyle = `rgba(255,140,80,${0.35 + 0.6 * glow})`; c.lineWidth = 6; c.beginPath(); c.moveTo(x0 + ux * L * 0.15, y0 + uy * L * 0.15); c.lineTo(x0 + ux * L * 0.85, y0 + uy * L * 0.85); c.stroke(); c.restore(); }
    c.strokeStyle = col || "#e6edf3"; c.lineWidth = 2; c.beginPath(); c.moveTo(x0, y0);
    const a = 0.15 * L, b = 0.85 * L; c.lineTo(x0 + ux * a, y0 + uy * a);
    for (let i = 1; i <= 2 * k; i++) { const s = a + ((b - a) * (i - 0.5)) / (2 * k), sg = i % 2 ? 1 : -1; c.lineTo(x0 + ux * s + nx * z * sg, y0 + uy * s + ny * z * sg); }
    c.lineTo(x0 + ux * b, y0 + uy * b); c.lineTo(x1, y1); c.stroke();
  }

  // ------------------------------------------------------------------ theory (scenario-dependent)
  const TH_COMMON_LIMITS = `
    <h4>Limitations &amp; further reading</h4>
    <p>All three set-ups are quasi-static: radiation, displacement current and the finite propagation speed of the fields are
    neglected (fine for coil sizes ≪ wavelength). Wires are ideal except for the stated resistances, and the
    magnet is a point dipole. Reading: D. J. Griffiths, <i>Introduction to Electrodynamics</i>, ch. 7 (Faraday's law, motional emf,
    inductance, energy in magnetic fields); E. M. Purcell &amp; D. J. Morin, <i>Electricity and Magnetism</i>, ch. 7;
    Feynman Lectures vol. II, ch. 16–17 (including the "exceptions to the flux rule").</p>`;

  function theory(P) {
    const scn = P.scn || "gen";
    const intro = `
    <h4>The physical system</h4>
    <p>Faraday's law says that a changing magnetic flux through a circuit induces an electromotive force (emf) around it, and
    Lenz's law fixes the sign: the induced current always flows so that its own magnetic field <em>opposes the change</em> of
    flux — which is why induction always costs mechanical work and produces braking forces. This page shows the three classic
    ways of changing the flux $\\Phi=\\int\\mathbf B\\cdot d\\mathbf A$: rotating the circuit in a fixed field (generator),
    moving the source of the field (falling magnet), and changing the area of the circuit (sliding rod). SI units are used
    throughout: $B$ in tesla, $\\Phi$ in weber (Wb = T·m²), emf in volts, $R$ in ohms, $L$ in henry, forces in newtons.</p>`;
    let sys = "", eq = "", how = "", tryit = "";
    if (scn === "gen") {
      sys = `
    <p><b>Current scenario — AC generator.</b> A rectangular coil of $N$ turns, side lengths $a$ (width) and $b$ (height), area
    $A=ab$, rotates at constant angular velocity $\\omega=2\\pi f$ about a vertical axis perpendicular to a uniform horizontal
    field $\\mathbf B$ between the poles of a magnet. Slip rings and brushes connect it to a load resistor $R_L$; the coil's own
    self-inductance $L$ can be included (its resistance is lumped into $R_L$). The motor driving the coil supplies whatever torque is
    needed to keep $\\omega$ constant.</p>`;
      eq = `
    <h4>Equations being solved</h4>
    <p>With the coil normal $\\hat n=(\\cos\\theta,\\sin\\theta,0)$, $\\theta=\\omega t$, and $\\mathbf B=B\\hat x$, the flux linkage is</p>
    $$N\\Phi(t)=NBA\\cos\\omega t .$$
    <div class="callout">$$\\varepsilon(t)=-\\frac{d(N\\Phi)}{dt}=NBA\\,\\omega\\,\\sin\\omega t\\equiv\\varepsilon_0\\sin\\omega t ,\\qquad \\varepsilon_0=NBA\\omega .$$</div>
    <p>The same result follows from the motional emf $\\oint(\\mathbf v\\times\\mathbf B)\\cdot d\\boldsymbol\\ell$ on the two vertical sides
    (speed $\\omega a/2$, length $b$, $N$ turns). The circuit equation (Kirchhoff's voltage law) is</p>
    $$L\\frac{dI}{dt}+R_L I=\\varepsilon_0\\sin\\omega t ,\\qquad I(0)=0,$$
    <p>with the exact solution (impedance $|Z|=\\sqrt{R_L^2+\\omega^2L^2}$, phase lag $\\varphi=\\arctan(\\omega L/R_L)$, $\\tau=L/R_L$)</p>
    $$I(t)=\\frac{\\varepsilon_0}{|Z|}\\Big[\\sin(\\omega t-\\varphi)+\\sin\\varphi\\,e^{-t/\\tau}\\Big]\\;\\xrightarrow{L\\to0}\\;\\frac{\\varepsilon_0}{R_L}\\sin\\omega t .$$
    <p>Energy: the driving torque $\\tau_{\\rm mech}=NIAB\\sin\\theta$ delivers $P_{\\rm mech}=\\varepsilon I=I^2R_L+\\tfrac{d}{dt}\\big(\\tfrac12LI^2\\big)$.
    The steady-state mean power in the load is</p>
    $$\\langle P\\rangle=\\frac{\\varepsilon_0^2R_L}{2|Z|^2},\\qquad I_{\\rm rms}=\\frac{\\varepsilon_0}{\\sqrt2\\,|Z|}.$$`;
      how = `
    <h4>How the simulation solves them</h4>
    <ul>
      <li>No numerical integration is needed: $\\theta=\\omega t$ is prescribed and $I(t)$ is evaluated from the exact formula above at
      every sub-step (the $L=0$ case is handled separately, so there is no division by zero).</li>
      <li>Time runs in slow motion: one revolution takes about 2.5 s of real time at speed ×1 (the toolbar shows the slow-motion
      factor). Each frame is split into sub-steps of $T/400$ ($T=1/f$).</li>
      <li>The Joule energy $\\int I^2R_L\\,dt$ and the mechanical work $\\int\\varepsilon I\\,dt$ are accumulated with the midpoint rule;
      the measured mean power is the Joule energy of the last complete revolution divided by $T$ and is compared with
      $\\varepsilon_0^2R_L/2|Z|^2$ (they agree once the start-up transient $e^{-t/\\tau}$ has died away).</li>
      <li>Plots: flux linkage $N\\Phi$, emf $\\varepsilon$ with the resistor voltage $IR_L$ (dashed — it lags behind when $L&gt;0$), current $I$,
      and instantaneous load power $I^2R_L$ with its running mean. The last three revolutions are shown.</li>
      <li>Animation: perspective drawing of the coil. The coloured fill of the coil shows the flux through it (teal: $\\Phi&gt;0$, violet: $\\Phi&lt;0$,
      opacity ∝ $|\\cos\\theta|$); yellow dots move along the winding with speed ∝ $I$, and the pink arrow is the field of the induced current,
      $\\mathbf B_{\\rm ind}\\propto I\\hat n$.</li>
    </ul>`;
      tryit = `
    <h4>What to try</h4>
    <ol>
      <li>Watch Lenz's law: when $|\\Phi|$ is decreasing the pink $\\mathbf B_{\\rm ind}$ arrow points <em>along</em> $\\mathbf B$ (it tries to keep the flux),
      when $|\\Phi|$ grows it points against it. The emf is maximal when the coil plane is parallel to $\\mathbf B$ ($\\Phi=0$).</li>
      <li>Defaults ($N=50$, $B=0.2$ T, $10\\times8$ cm, 50 Hz): $\\varepsilon_0=50\\cdot0.2\\cdot0.008\\cdot2\\pi\\cdot50\\approx25.1$ V.
      Double $f$ or $N$ and the peak emf doubles.</li>
      <li>Set $L=0$: emf and $IR_L$ coincide. Raise $L$ to 200 mH: $\\omega L\\approx63\\ \\Omega$, the current lags by
      $\\varphi=\\arctan(63/10)\\approx81^\\circ$ and the mean power drops by $R_L^2/|Z|^2$.</li>
      <li>Load matching: with $L&gt;0$ the power is largest for $R_L=\\omega L$ (try $L=31.8$ mH, $R_L=10\\ \\Omega$ at 50 Hz).</li>
      <li>The power curve $I^2R_L$ oscillates at $2\\omega$ — the "flicker" of an AC lamp.</li>
    </ol>`;
    } else if (scn === "magnet") {
      sys = `
    <p><b>Current scenario — magnet falling through a coil.</b> A small bar magnet (point dipole, moment $m$ in A·m², mass $M$)
    is released from rest at height $z_0$ on the axis of a horizontal circular coil of radius $a$ with $N$ turns and total
    resistance $R$ (self-inductance neglected). The magnet falls under gravity $g=9.81$ m/s²; optionally the induced current
    acts back on it ("Lenz braking").</p>`;
      eq = `
    <h4>Equations being solved</h4>
    <p>The vector potential of a dipole $\\mathbf m=m\\hat z$ is $\\mathbf A=\\frac{\\mu_0}{4\\pi}\\frac{\\mathbf m\\times\\mathbf r}{r^3}$, i.e.
    $A_\\phi=\\frac{\\mu_0 m}{4\\pi}\\frac{\\rho}{r^3}$. By Stokes' theorem the flux through a circle of radius $a$ at axial distance $z$ is
    $\\Phi=\\oint\\mathbf A\\cdot d\\boldsymbol\\ell=2\\pi a A_\\phi$:</p>
    <div class="callout">$$\\Phi(z)=\\frac{\\mu_0\\,m\\,a^2}{2\\,(a^2+z^2)^{3/2}},\\qquad
      \\varepsilon=-N\\frac{d\\Phi}{dt}=-N\\Phi'(z)\\,\\dot z,\\qquad \\Phi'(z)=-\\frac{3\\mu_0 m a^2 z}{2\\,(a^2+z^2)^{5/2}} .$$</div>
    <p>Because $\\Phi'(z)$ is odd in $z$, the emf is a two-lobed pulse: one sign while the magnet approaches, the opposite sign
    while it leaves, and zero when it is exactly in the plane of the coil (maximum flux). The induced current is $I=\\varepsilon/R$.
    The force of the coil on the magnet follows from energy conservation ($F\\dot z=-\\varepsilon I$):</p>
    $$F_z=-\\frac{\\big(N\\Phi'(z)\\big)^2}{R}\\,\\dot z ,\\qquad M\\ddot z=-Mg+F_z .$$
    <p>$F_z$ always opposes the velocity (Lenz) and is proportional to it, like viscous drag, but only acts within a few coil radii.
    Energy bookkeeping: $Mg\\,(z_0-z)=\\tfrac12M\\dot z^2+\\int I^2R\\,dt$.</p>`;
      how = `
    <h4>How the simulation solves them</h4>
    <ul>
      <li>State $(z,\\dot z,Q_J)$ with $Q_J=\\int\\varepsilon^2/R\\,dt$ is advanced with classical RK4. The braking term is stiff when $N$ is
      large and $R$ small (rate $\\lambda=(N\\Phi'_{\\max})^2/(MR)$, with $\\Phi'_{\\max}$ at $z=a/2$), so the step is $h=\\min(0.4/\\lambda,\\,2\\times10^{-4}\\,\\text{s})$.</li>
      <li>Slow motion: the free-fall time $\\sqrt{4z_0/g}$ is stretched to about 4 s of real time. The run stops when the magnet
      reaches $z=-z_0$; press Play or Restart to drop it again.</li>
      <li>Plots: flux linkage $N\\Phi(t)$, emf $\\varepsilon(t)$ and speed, each with the free-fall prediction without braking (dashed,
      $z=z_0-\\tfrac12gt^2$), and the energy budget (gravitational energy released, kinetic energy, Joule heat). The
      residual $Mg(z_0-z)-K-Q_J$ is shown as a metric — it measures the integration error (with braking off, $Q_J$ is not taken from the magnet and is left out of the balance).</li>
      <li>The panel on the right of the animation shows the static profiles $\\Phi(z)$ and $\\Phi'(z)$ at the same heights as the
      scene; dots on the coil move with speed ∝ $I$ and the pink arrow is the coil's induced magnetic moment.</li>
    </ul>`;
      tryit = `
    <h4>What to try</h4>
    <ol>
      <li>Switch braking off: the two emf lobes have opposite sign and the second (leaving) lobe is larger, because the magnet is faster.
      The areas of the two lobes are equal and opposite: $\\int\\varepsilon\\,dt=-N\\Delta\\Phi=0$ when the magnet ends far below.</li>
      <li>Defaults ($N=200$, $m=1$ A·m², $a=1.5$ cm, $R=2\\ \\Omega$, $M=10$ g): braking is strong — the peak force exceeds $Mg$ and the
      magnet nearly reaches a local "terminal" speed $v\\approx MgR/(N\\Phi')^2$ inside the coil.</li>
      <li>Lower $R$ to 0.1 Ω: the magnet almost stops in the coil (copper-tube effect); raise $R$ to 100 Ω: the motion is
      practically free fall.</li>
      <li>Flip the magnet: every sign of $\\Phi$, $\\varepsilon$ and $I$ reverses, the braking force does not.</li>
      <li>Check the bookkeeping: the Joule heat is exactly the kinetic energy missing compared with free fall.</li>
    </ol>`;
    } else {
      sys = `
    <p><b>Current scenario — sliding rod on rails.</b> A conducting rod of length $L$ and mass $M$ slides without friction on two
    parallel rails joined at one end by a resistor $R$ (rails and rod have no resistance). A uniform field $B$ is perpendicular to
    the plane of the rails (into the screen). The rod is driven by a constant applied force $F$, by gravity on rails inclined at
    angle $\\theta$ ($F=Mg\\sin\\theta$, $\\mathbf B$ still perpendicular to the rail plane), or just given an initial push $v_0$.</p>`;
      eq = `
    <h4>Equations being solved</h4>
    <p>The circuit encloses the area $Lx$, so $\\Phi=BLx$ and the emf is $\\varepsilon=-d\\Phi/dt=-BLv$ — equivalently the motional emf
    $(\\mathbf v\\times\\mathbf B)\\cdot\\mathbf L$ on the free charges of the rod. The current $I=BLv/R$ flows so that its field opposes the
    growth of flux (counter-clockwise in the picture), and the magnetic force $I\\mathbf L\\times\\mathbf B$ on the rod points backwards:</p>
    $$\\varepsilon=BLv,\\qquad I=\\frac{BLv}{R},\\qquad F_B=-\\frac{B^2L^2}{R}\\,v .$$
    <div class="callout">$$M\\frac{dv}{dt}=F-\\frac{B^2L^2}{R}v\\quad\\Longrightarrow\\quad v(t)=v_t+(v_0-v_t)\\,e^{-t/\\tau},\\qquad
      v_t=\\frac{FR}{B^2L^2},\\quad \\tau=\\frac{MR}{B^2L^2}.$$</div>
    <p>Energy bookkeeping: the work done by the driving force is shared between kinetic energy and Joule heat,</p>
    $$\\int_0^t Fv\\,dt'=\\Delta\\big(\\tfrac12Mv^2\\big)+\\int_0^t I^2R\\,dt' ,$$
    <p>and at terminal velocity all the input power $Fv_t$ is dissipated: $Fv_t=I^2R=F^2R/(B^2L^2)$. With $F=0$ the rod coasts to a stop
    after the finite distance $x_\\infty=Mv_0R/(B^2L^2)=v_0\\tau$.</p>`;
      how = `
    <h4>How the simulation solves them</h4>
    <ul>
      <li>State $(x,v,W,Q_J)$ — position, velocity, work done by $F$ and Joule heat — integrated with classical RK4,
      step $h=\\tau/60$ (accurate to ~$10^{-9}$ relative). The analytic $v(t)$ is overlaid as a dashed curve.</li>
      <li>One run lasts $6\\tau$ and plays in about 8 s of real time; the toolbar shows the slow-motion factor.</li>
      <li>Plots: speed (with $v_t$), current, powers ($Fv$, $I^2R$ and $dK/dt=Mv\\dot v$, which add up: $Fv=I^2R+dK/dt$), and the
      energy budget $W$, $\\Delta K$, $Q_J$ with $\\Delta K+Q_J$ dashed on top of $W$. The residual $(W-\\Delta K-Q_J)/W$ is a metric.</li>
      <li>Animation: top view of the rails; the shaded area is the flux-carrying region; yellow dots show the induced current
      (speed ∝ $I$); green and red arrows are the driving and magnetic forces. The view follows the rod.</li>
    </ul>`;
      tryit = `
    <h4>What to try</h4>
    <ol>
      <li>Defaults ($B=1$ T, $L=0.5$ m, $R=1\\ \\Omega$, $M=0.2$ kg, $F=1$ N): $v_t=4$ m/s, $\\tau=0.8$ s; after $\\tau$ the speed is
      $63\\%$ of $v_t$.</li>
      <li>Double $B$: $v_t$ and $\\tau$ both fall by a factor 4 — magnetic braking is quadratic in $B$.</li>
      <li>Watch the energy plot: early on most of the work goes into kinetic energy, later all of it is turned into heat.</li>
      <li>Inclined rails at $\\theta=30^\\circ$: $F=Mg/2=0.98$ N, so $v_t\\approx3.92$ m/s for the defaults.</li>
      <li>Initial push only, $v_0=5$ m/s: exponential decay, stopping distance $v_0\\tau=4$ m; all of $\\tfrac12Mv_0^2=2.5$ J ends up as heat.</li>
    </ol>`;
    }
    const others = `
    <p>Other scenarios (select above): <b>AC generator</b> — rotating coil, $\\varepsilon=NBA\\omega\\sin\\omega t$; <b>falling magnet</b> —
    dipole flux and the two-lobed emf pulse; <b>sliding rod</b> — motional emf $BLv$ and terminal velocity.</p>`;
    return intro + sys + others + eq + how + tryit + TH_COMMON_LIMITS;
  }

  App.register({
    id: "faraday-induction",
    category: "classical",
    group: "Electromagnetism",
    order: 23,
    title: "Faraday's Law & Electromagnetic Induction",
    icon: "🔄",
    subtitle: "Changing magnetic flux induces an emf $\\varepsilon=-d\\Phi/dt$: a rotating generator coil, a magnet falling through a coil and a rod sliding on rails, with the induced currents, Lenz braking forces and energy budget computed live.",
    notes: [{ type: "info", html: "Choose a scenario. Yellow dots show the induced current (speed ∝ $I$, direction by Lenz's law), the pink arrow the magnetic field / moment of that current, which always opposes the change of flux. Time runs in slow motion; the toolbar shows the factor." }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "scn", type: "select", label: "Scenario", value: "gen", rebuild: true, options: [
        { value: "gen", label: "AC generator (rotating coil)" },
        { value: "magnet", label: "Magnet falling through a coil" },
        { value: "rod", label: "Sliding rod on rails" },
      ] },
      // --- generator
      { id: "sec1", type: "section", label: "Coil and field", visibleIf: (p) => p.scn === "gen" },
      { id: "gN", type: "slider", label: "Turns $N$", min: 1, max: 200, step: 1, value: 50, visibleIf: (p) => p.scn === "gen" },
      { id: "gB", type: "slider", label: "Field $B$", min: 0.01, max: 1, step: 0.01, value: 0.2, unit: "T", visibleIf: (p) => p.scn === "gen" },
      { id: "ga", type: "slider", label: "Coil width $a$", min: 2, max: 30, step: 1, value: 10, unit: "cm", visibleIf: (p) => p.scn === "gen" },
      { id: "gb", type: "slider", label: "Coil height $b$", min: 2, max: 30, step: 1, value: 8, unit: "cm", visibleIf: (p) => p.scn === "gen" },
      { id: "gf", type: "slider", label: "Rotation frequency $f$", min: 0.5, max: 60, step: 0.5, value: 50, unit: "Hz", visibleIf: (p) => p.scn === "gen" },
      { id: "sec2", type: "section", label: "Load", visibleIf: (p) => p.scn === "gen" },
      { id: "gR", type: "slider", label: "Load resistance $R_L$", min: 1, max: 200, step: 1, value: 10, unit: "Ω", visibleIf: (p) => p.scn === "gen" },
      { id: "gL", type: "slider", label: "Coil inductance $L$", min: 0, max: 200, step: 1, value: 20, unit: "mH", visibleIf: (p) => p.scn === "gen",
        help: "Self-inductance makes the current lag the emf by $\\varphi=\\arctan(\\omega L/R_L)$. $L=0$: purely resistive." },
      // --- magnet
      { id: "sec3", type: "section", label: "Magnet", visibleIf: (p) => p.scn === "magnet" },
      { id: "mm", type: "slider", label: "Dipole moment $m$", min: 0.1, max: 3, step: 0.1, value: 1, unit: "A·m²", visibleIf: (p) => p.scn === "magnet",
        help: "A 1 cm³ neodymium magnet has $m\\approx1$ A·m²." },
      { id: "mM", type: "slider", label: "Magnet mass $M$", min: 2, max: 50, step: 1, value: 10, unit: "g", visibleIf: (p) => p.scn === "magnet" },
      { id: "mz0", type: "slider", label: "Release height $z_0$", min: 3, max: 30, step: 1, value: 10, unit: "cm", visibleIf: (p) => p.scn === "magnet" },
      { id: "mflip", type: "checkbox", label: "Flip magnet (south pole leading)", value: false, visibleIf: (p) => p.scn === "magnet" },
      { id: "sec4", type: "section", label: "Coil", visibleIf: (p) => p.scn === "magnet" },
      { id: "mN", type: "slider", label: "Turns $N$", min: 1, max: 1000, step: 1, value: 200, visibleIf: (p) => p.scn === "magnet" },
      { id: "ma", type: "slider", label: "Coil radius $a$", min: 0.5, max: 5, step: 0.1, value: 1.5, unit: "cm", visibleIf: (p) => p.scn === "magnet" },
      { id: "mR", type: "slider", label: "Coil resistance $R$", min: 0.1, max: 100, step: 0.1, value: 2, unit: "Ω", visibleIf: (p) => p.scn === "magnet" },
      { id: "mbrake", type: "checkbox", label: "Lenz braking (induced current acts back on the magnet)", value: true, visibleIf: (p) => p.scn === "magnet" },
      // --- rod
      { id: "sec5", type: "section", label: "Rails and rod", visibleIf: (p) => p.scn === "rod" },
      { id: "rB", type: "slider", label: "Field $B$ (⊥ rail plane)", min: 0.1, max: 2, step: 0.05, value: 1, unit: "T", visibleIf: (p) => p.scn === "rod" },
      { id: "rL", type: "slider", label: "Rod length $L$", min: 0.1, max: 1, step: 0.05, value: 0.5, unit: "m", visibleIf: (p) => p.scn === "rod" },
      { id: "rR", type: "slider", label: "Resistance $R$", min: 0.1, max: 10, step: 0.1, value: 1, unit: "Ω", visibleIf: (p) => p.scn === "rod" },
      { id: "rM", type: "slider", label: "Rod mass $M$", min: 0.05, max: 2, step: 0.05, value: 0.2, unit: "kg", visibleIf: (p) => p.scn === "rod" },
      { id: "sec6", type: "section", label: "Drive", visibleIf: (p) => p.scn === "rod" },
      { id: "rdrive", type: "select", label: "Driving force", value: "force", visibleIf: (p) => p.scn === "rod", options: [
        { value: "force", label: "Constant applied force F" },
        { value: "incline", label: "Gravity on inclined rails (F = Mg sin θ)" },
        { value: "push", label: "Initial push only (F = 0)" },
      ] },
      { id: "rF", type: "slider", label: "Applied force $F$", min: 0.1, max: 5, step: 0.1, value: 1, unit: "N", visibleIf: (p) => p.scn === "rod" && p.rdrive === "force" },
      { id: "rth", type: "slider", label: "Incline angle $\\theta$", min: 5, max: 60, step: 1, value: 30, unit: "°", visibleIf: (p) => p.scn === "rod" && p.rdrive === "incline" },
      { id: "rv0", type: "slider", label: "Initial speed $v_0$", min: 0, max: 10, step: 0.1, value: 0, unit: "m/s", visibleIf: (p) => p.scn === "rod",
        help: "With “initial push only” a zero $v_0$ is replaced by 5 m/s." },
    ],
    theory,

    mount(api) {
      const P = api.params, scn = P.scn || "gen";
      if (scn === "gen") return mountGenerator(api, P);
      if (scn === "magnet") return mountMagnet(api, P);
      return mountRod(api, P);
    },
  });

  // ================================================================== (a) AC generator
  function mountGenerator(api, P) {
    const M = api.metrics([
      { id: "th", label: "Angle $\\theta=\\omega t$" },
      { id: "eps", label: "emf $\\varepsilon$" },
      { id: "e0", label: "Peak emf $NBA\\omega$" },
      { id: "Ipk", label: "Peak current: sim / theory" },
      { id: "P", label: "Mean power $\\langle I^2R_L\\rangle$: sim / theory" },
      { id: "phi", label: "Current lag $\\varphi$" },
    ]);
    const plots = api.plots([
      { id: "anim", title: "Rotating coil in a uniform field (perspective view)", span: 2, aspect: 0.42, axes: false, minHeight: 300, maxHeight: 460 },
      { id: "flux", title: "Flux linkage $N\\Phi(t)=NBA\\cos\\omega t$", aspect: 0.5, xlabel: "t (s)", ylabel: "NΦ (mWb)" },
      { id: "emf", title: "emf $\\varepsilon(t)$ and resistor voltage $IR_L$", aspect: 0.5, xlabel: "t (s)", ylabel: "voltage (V)" },
      { id: "cur", title: "Current $I(t)$", aspect: 0.5, xlabel: "t (s)", ylabel: "I (A)" },
      { id: "pow", title: "Power in the load $I^2R_L$", aspect: 0.5, xlabel: "t (s)", ylabel: "P (W)" },
    ]);
    const S = Series(1400, ["phi", "eps", "vr", "I", "P"], "roll");
    const sc = { f: Scaler(), e: Scaler(), i: Scaler(), p: Scaler() };
    let t = 0, ER = 0, EM = 0, ERprev = 0, lastRev = 0, meanP = NaN, Ipk = 0, IpkPrev = NaN, dotOff = 0, nextRec = 0;
    let N, B, A, w, T, R, L, e0, Z, phi, tau, PI_TH, dtRec, slow;

    function params() {
      N = P.gN; B = P.gB; A = (P.ga / 100) * (P.gb / 100); w = 2 * Math.PI * P.gf; T = 1 / P.gf;
      R = P.gR; L = P.gL / 1000; e0 = N * B * A * w;
      Z = Math.hypot(R, w * L); phi = Math.atan2(w * L, R); tau = L > 0 ? L / R : 0;
      PI_TH = (e0 * e0 * R) / (2 * Z * Z); dtRec = T / 160; slow = T / 2.5;
    }
    const emf = (tt) => e0 * Math.sin(w * tt);
    const cur = (tt) => (L > 0 ? (e0 / Z) * (Math.sin(w * tt - phi) + Math.sin(phi) * Math.exp(-tt / tau)) : (e0 / R) * Math.sin(w * tt));
    function record() {
      const I = cur(t);
      S.push(t, { phi: N * B * A * Math.cos(w * t) * 1000, eps: emf(t), vr: I * R, I, P: I * I * R });
    }

    // ---------------------------------------------------- drawing
    const PITCH = 0.36, D = 5.5;
    function drawScene() {
      const p = plots.anim; p.clear();
      const I = cur(t), Imax = e0 / Z, th = w * t, ct = Math.cos(th), st = Math.sin(th);
      const hw = 1.1 * Math.pow(P.ga / 30, 0.6), hh = 1.0 * Math.pow(P.gb / 30, 0.6); // half-sizes (compressed scale)
      p.custom((c, pl) => {
        const W = pl.W, H = pl.H;
        const cx = W * 0.4, cy = H * 0.44, Sc = Math.min(W * 0.16, H * 0.3);
        const cp = Math.cos(PITCH), sp = Math.sin(PITCH);
        const pr = (x, y, z) => { const up = z * cp + y * sp, dep = y * cp - z * sp, k = D / (D + dep); return [cx + Sc * x * k, cy - Sc * up * k, dep]; };
        const u = [-st, ct, 0], n = [ct, st, 0];
        const pt = (su, sz) => pr(su * u[0], su * u[1], sz);
        // --- magnet poles (boxes)
        function box(x0, x1, col, colTop, colIn, label, inner) {
          const y0 = -0.95, y1 = 0.95, z0 = -1.15, z1 = 1.15;
          const face = (pts, fill) => { c.beginPath(); pts.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); c.fillStyle = fill; c.fill(); c.strokeStyle = "rgba(0,0,0,0.35)"; c.lineWidth = 1; c.stroke(); };
          face([pr(x0, y1, z1), pr(x1, y1, z1), pr(x1, y0, z1), pr(x0, y0, z1)], colTop);
          face([pr(inner, y0, z0), pr(inner, y1, z0), pr(inner, y1, z1), pr(inner, y0, z1)], colIn);
          face([pr(x0, y0, z0), pr(x1, y0, z0), pr(x1, y0, z1), pr(x0, y0, z1)], col);
          const q = pr((x0 + x1) / 2, y0, 0);
          txt(c, label, q[0], q[1], { size: 26, bold: true, align: "center", color: "rgba(255,255,255,0.9)" });
        }
        box(-2.1, -1.4, "#a8423a", "#c95b50", "#7d2f29", "N", -1.4);
        box(1.4, 2.1, "#2f5f9e", "#4a7cc0", "#22477a", "S", 1.4);
        // --- field lines (behind the coil first)
        const lines = [];
        for (const y of [0.6, 0, -0.6]) for (const z of [-0.85, -0.3, 0.3, 0.85]) lines.push([y, z]);
        function fieldLine(y, z, a) {
          const q0 = pr(-1.4, y, z), q1 = pr(1.4, y, z), qa = pr(0.95, y, z), qb = pr(1.15, y, z);
          c.globalAlpha = a; c.strokeStyle = BCOL; c.lineWidth = 1.2; c.setLineDash([6, 5]);
          c.beginPath(); c.moveTo(q0[0], q0[1]); c.lineTo(q1[0], q1[1]); c.stroke(); c.setLineDash([]);
          arrowPx(c, qa[0], qa[1], qb[0], qb[1], BCOL, 1.2, 7); c.globalAlpha = 1;
        }
        for (const [y, z] of lines) if (y >= 0) fieldLine(y, z, y > 0 ? 0.35 : 0.55);
        // --- axle and slip rings
        const zb = -hh - 0.35;
        const a0 = pr(0, 0, -hh - 0.62), a1 = pr(0, 0, hh + 0.3);
        c.strokeStyle = "#8b98a8"; c.lineWidth = 3; c.beginPath(); c.moveTo(a0[0], a0[1]); c.lineTo(a1[0], a1[1]); c.stroke();
        const ringPts = [];
        for (const zr of [zb, zb - 0.17]) {
          c.strokeStyle = "#c9a227"; c.lineWidth = 2.5; c.beginPath();
          for (let k = 0; k <= 32; k++) { const ph = (2 * Math.PI * k) / 32, q = pr(0.13 * Math.cos(ph), 0.13 * Math.sin(ph), zr); k ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]); }
          c.stroke(); ringPts.push(pr(0.15, 0, zr));
        }
        // --- coil: flux fill
        const cs = [pt(hw, -hh), pt(hw, hh), pt(-hw, hh), pt(-hw, -hh)];
        c.beginPath(); cs.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath();
        c.fillStyle = `rgba(${ct >= 0 ? FLUXP : FLUXN},${0.06 + 0.3 * Math.abs(ct)})`; c.fill();
        // edges sorted by depth (painter)
        const edges = [[0, 1], [1, 2], [2, 3], [3, 0]].map(([i, j]) => [i, j, (cs[i][2] + cs[j][2]) / 2]).sort((e1, e2) => e2[2] - e1[2]);
        for (const [i, j] of edges) {
          for (const [wd, col] of [[7, "#7a4a1d"], [4.5, COPPER], [1.2, "#ffd9a8"]]) {
            c.strokeStyle = col; c.lineWidth = wd; c.lineCap = "round"; c.beginPath(); c.moveTo(cs[i][0], cs[i][1]); c.lineTo(cs[j][0], cs[j][1]); c.stroke();
          }
        }
        // leads from the coil to the slip rings
        const lb = pt(0.05, -hh);
        c.strokeStyle = COPPER; c.lineWidth = 1.5; c.beginPath(); c.moveTo(lb[0], lb[1]); c.lineTo(ringPts[0][0] - 6, ringPts[0][1]); c.stroke();
        // --- induced current: dots + edge chevrons
        const per = 4 * (hw + hh), rel = Imax > 0 ? I / Imax : 0;
        function loopPt(s) {
          s = ((s % per) + per) % per;
          if (s < 2 * hh) return pt(hw, -hh + s); s -= 2 * hh;
          if (s < 2 * hw) return pt(hw - s, hh); s -= 2 * hw;
          if (s < 2 * hh) return pt(-hw, hh - s); s -= 2 * hh;
          return pt(-hw + s, -hh);
        }
        const nd = 22;
        c.fillStyle = IND;
        for (let k = 0; k < nd; k++) { const q = loopPt(dotOff * per + (k * per) / nd); c.globalAlpha = 0.25 + 0.75 * Math.min(1, Math.abs(rel) * 3); c.beginPath(); c.arc(q[0], q[1], 3, 0, 2 * Math.PI); c.fill(); }
        c.globalAlpha = 1;
        if (Math.abs(rel) > 0.04) {
          for (let e = 0; e < 4; e++) {
            const qa = cs[e], qb = cs[(e + 1) % 4], mx = (qa[0] + qb[0]) / 2, my = (qa[1] + qb[1]) / 2;
            let ang = Math.atan2(qb[1] - qa[1], qb[0] - qa[0]); if (I < 0) ang += Math.PI;
            if (Math.hypot(qb[0] - qa[0], qb[1] - qa[1]) > 14) chevron(c, mx, my, ang, IND, 7);
          }
        }
        // --- normal and induced field
        const o = pr(0, 0, 0), nn = pr(0.75 * n[0], 0.75 * n[1], 0);
        c.setLineDash([3, 3]); arrowPx(c, o[0], o[1], nn[0], nn[1], "#b9c4d0", 1.3, 7); c.setLineDash([]);
        txt(c, "n̂", nn[0] + 6, nn[1] - 8, { size: 12, color: "#b9c4d0" });
        if (Math.abs(rel) > 0.03) {
          const s = Math.sign(I) * Math.min(1, Math.abs(rel)) * 1.0, bi = pr(s * n[0], s * n[1], 0.02);
          arrowPx(c, o[0], o[1], bi[0], bi[1], PlotColors.pink, 3, 10);
          txt(c, "B_ind", bi[0] + (bi[0] > o[0] ? 8 : -46), bi[1] - 12, { size: 12, bold: true, color: PlotColors.pink });
        }
        // front field lines
        for (const [y, z] of lines) if (y < 0) fieldLine(y, z, 0.6);
        const bl = pr(-1.25, -0.6, 0.85); txt(c, "B", bl[0] + 4, bl[1] - 12, { size: 14, bold: true, color: BCOL });
        // --- brushes, wires and load resistor
        const XR = W * 0.86, Rtop = H * 0.2, Rbot = H * 0.52, glow = PI_TH > 0 ? Math.min(1, (I * I * R) / (2 * PI_TH + 1e-30)) : 0;
        const b1 = ringPts[0], b2 = ringPts[1];
        c.fillStyle = "#555f6b"; for (const b of [b1, b2]) c.fillRect(b[0], b[1] - 4, 12, 8);
        c.strokeStyle = "#c7ced6"; c.lineWidth = 1.8;
        c.beginPath(); c.moveTo(b1[0] + 12, b1[1]); c.lineTo(XR, b1[1]); c.lineTo(XR, Rbot); c.stroke();
        c.beginPath(); c.moveTo(b2[0] + 12, b2[1]); c.lineTo(XR + 34, b2[1]); c.lineTo(XR + 34, Rtop - 14); c.lineTo(XR, Rtop - 14); c.lineTo(XR, Rtop); c.stroke();
        resistorPx(c, XR, Rtop, XR, Rbot, glow, "#e6edf3");
        txt(c, `R_L = ${PM.fmt(R, 0)} Ω`, XR - 14, (Rtop + Rbot) / 2, { size: 12, align: "right", color: "#e6edf3", bg: "#0f151c" });
        if (Math.abs(rel) > 0.04) { const yy = (Rtop + Rbot) / 2 + (I > 0 ? -1 : 1) * 30; chevron(c, XR, yy, I > 0 ? -Math.PI / 2 : Math.PI / 2, IND, 8); }
        txt(c, `I = ${PM.fmt(I, 3)} A`, XR - 14, Rbot + 18, { size: 12, align: "right", color: IND });
        // --- labels
        const deg = ((th * 180) / Math.PI) % 360;
        txt(c, `θ = ωt = ${deg.toFixed(0)}°`, 14, 18, { size: 13, bold: true });
        txt(c, `NΦ = NBA cos θ = ${PM.fmt(N * B * A * ct * 1000, 2)} mWb`, 14, 38, { size: 12, color: ct >= 0 ? "#4fd1c5" : "#b69cff" });
        txt(c, `ε = NBAω sin θ = ${PM.fmt(e0 * st, 2)} V`, 14, 57, { size: 12, color: PlotColors.accent3 });
        txt(c, "Lenz: B_ind opposes the change of Φ", 14, H - 16, { size: 11.5, color: "#c9d1d9" });
      });
    }

    function timePlot(p, key, scaler, floor, extra) {
      const n = S.n, T0 = S.T(), x0 = Math.max(0, t - 3 * T), x1 = Math.max(t, 3 * T);
      const i0 = firstIdx(T0, n, x0);
      let m = 0; for (const k of key) m = Math.max(m, maxAbs(S.d[k], n, i0));
      const lim = scaler.fit(m, floor);
      p.setLimits([x0, x1], extra && extra.positive ? [0, lim] : [-lim, lim]);
      p.clear(); p.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
      return { T: T0, i0 };
    }

    return {
      reset() {
        params();
        t = 0; ER = EM = ERprev = 0; lastRev = 0; meanP = NaN; Ipk = 0; IpkPrev = NaN; dotOff = 0; nextRec = 0;
        S.clear(); for (const k in sc) sc[k].reset();
        record(); nextRec = dtRec;
      },
      step(dt) {
        const simDt = dt * slow, h = T / 400, n = Math.max(1, Math.ceil(simDt / h)), hh = simDt / n;
        for (let i = 0; i < n; i++) {
          const tm = t + hh / 2, Im = cur(tm);
          ER += Im * Im * R * hh; EM += emf(tm) * Im * hh;
          t += hh;
          Ipk = Math.max(Ipk, Math.abs(cur(t)));
          const rev = Math.floor(t / T);
          if (rev > lastRev) { meanP = (ER - ERprev) / (T * (rev - lastRev)); ERprev = ER; lastRev = rev; IpkPrev = Ipk; Ipk = 0; }
          if (t >= nextRec) { record(); nextRec += dtRec; }
        }
        dotOff += (dt * (cur(t) / (e0 / Z || 1))) * 0.35;
      },
      render() {
        drawScene();
        const I = cur(t);
        // flux
        let pp = plots.flux, r = timePlot(pp, ["phi"], sc.f, 1e-6);
        pp.line(r.T, S.Y("phi"), { color: PlotColors.accent, width: 1.8 });
        // emf
        pp = plots.emf; r = timePlot(pp, ["eps", "vr"], sc.e, 1e-6);
        pp.line(r.T, S.Y("eps"), { color: PlotColors.accent3, width: 1.8 });
        pp.line(r.T, S.Y("vr"), { color: PlotColors.pink, width: 1.4, dash: [6, 4] });
        pp.legend([{ label: "ε", color: PlotColors.accent3 }, { label: "I·R_L", color: PlotColors.pink, dash: [6, 4] }], "tr");
        // current
        pp = plots.cur; r = timePlot(pp, ["I"], sc.i, 1e-9);
        pp.hline(e0 / Z, { color: PlotColors.muted, dash: [3, 4], width: 1 }); pp.hline(-e0 / Z, { color: PlotColors.muted, dash: [3, 4], width: 1 });
        pp.line(r.T, S.Y("I"), { color: IND, width: 1.8 });
        pp.label(`steady amplitude ε₀/|Z| = ${PM.fmt(e0 / Z, 3)} A`, "bl", { size: 11 });
        // power
        pp = plots.pow; r = timePlot(pp, ["P"], sc.p, 1e-9, { positive: true });
        pp.line(r.T, S.Y("P"), { color: PlotColors.bad, width: 1.6 });
        pp.hline(PI_TH, { color: PlotColors.text, dash: [5, 4], width: 1.2 });
        pp.legend([{ label: "I²R_L", color: PlotColors.bad }, { label: "⟨P⟩ theory", color: PlotColors.text, dash: [5, 4] }], "tr");

        M.set("th", ((((w * t) * 180) / Math.PI) % 360).toFixed(1) + "°");
        M.set("eps", PM.fmt(emf(t), 3) + " V");
        M.set("e0", PM.fmt(e0, 3) + " V");
        M.set("Ipk", (isFinite(IpkPrev) ? PM.fmt(IpkPrev, 3) : "…") + " / " + PM.fmt(e0 / Z, 3) + " A");
        M.set("P", (isFinite(meanP) ? PM.fmt(meanP, 3) : "…") + " / " + PM.fmt(PI_TH, 3) + " W");
        M.set("phi", PM.fmt((phi * 180) / Math.PI, 1) + "°");
        api.setTime(`t = ${PM.fmt(t * 1000, 1)} ms · ${slowTxt(slow)}`);
        void I;
      },
    };
  }

  // ================================================================== (b) magnet through a coil
  function mountMagnet(api, P) {
    const M = api.metrics([
      { id: "z", label: "Height $z$ / speed $v$" },
      { id: "eps", label: "emf $\\varepsilon$" },
      { id: "epk", label: "Peak emf (+ / −)" },
      { id: "F", label: "Braking force / $Mg$" },
      { id: "Q", label: "Joule heat $Q_J$" },
      { id: "bal", label: "$Mg\\Delta h-K-Q_J$ (≈ 0)" },
    ]);
    const plots = api.plots([
      { id: "anim", title: "Magnet falling through a coil — side view (left) and flux profile Φ(z) (right)", span: 2, aspect: 0.42, axes: false, minHeight: 320, maxHeight: 480 },
      { id: "flux", title: "Flux linkage $N\\Phi(t)$", aspect: 0.5, xlabel: "t (s)", ylabel: "NΦ (mWb)" },
      { id: "emf", title: "Induced emf $\\varepsilon=-N\\,d\\Phi/dt$", aspect: 0.5, xlabel: "t (s)", ylabel: "ε (V)" },
      { id: "vel", title: "Speed of the magnet", aspect: 0.5, xlabel: "t (s)", ylabel: "|v| (m/s)" },
      { id: "en", title: "Energy budget", aspect: 0.5, xlabel: "t (s)", ylabel: "energy (mJ)" },
    ]);
    const S = Series(2400, ["phi", "eps", "v", "Ed", "K", "Q", "phiF", "epsF", "vF"], "grow");
    const sc = { f: Scaler(), e: Scaler(), v: Scaler(), en: Scaler() };
    const y = new Float64Array(3);
    let ws = null, t = 0, done = false, nextRec = 0, epP = 0, epN = 0, Fnow = 0, dotOff = 0;
    let N, mz, a, R, Mk, z0, k0, brake, slow, h, dtRec, Tff, Imax;
    const PROF = 160, profZ = new Float64Array(PROF), profF = new Float64Array(PROF), profD = new Float64Array(PROF);

    function params() {
      N = P.mN; mz = (P.mflip ? 1 : -1) * P.mm; // north pole down ⇒ m points down
      a = P.ma / 100; R = P.mR; Mk = P.mM / 1000; z0 = P.mz0 / 100; brake = !!P.mbrake;
      k0 = (MU0 * mz * a * a) / 2;
      const dmax = 0.429 * MU0 * Math.abs(mz) / (a * a) * N; // N·|Φ'| at z = a/2
      const lam = (dmax * dmax) / (Mk * R);
      Tff = Math.sqrt((4 * z0) / G); slow = Tff / 4;
      h = Math.min(brake ? 0.4 / lam : 1, 2e-4, Tff / 2000);
      dtRec = Tff / 1200;
      Imax = (dmax * Math.sqrt(2 * G * z0)) / R;
      for (let i = 0; i < PROF; i++) { const z = -z0 - 0.02 + ((2 * z0 + 0.04) * i) / (PROF - 1); profZ[i] = z; profF[i] = phiOf(z); profD[i] = dphiOf(z); }
    }
    const phiOf = (z) => k0 / Math.pow(a * a + z * z, 1.5);                 // flux per turn
    const dphiOf = (z) => (-3 * k0 * z) / Math.pow(a * a + z * z, 2.5);     // dΦ/dz per turn
    function deriv(_t, s, o) {
      const z = s[0], v = s[1], g1 = N * dphiOf(z), eps = -g1 * v;
      o[0] = v; o[1] = -G + (brake ? (-(g1 * g1) / R) * v / Mk : 0); o[2] = (eps * eps) / R;
    }
    function record() {
      const z = y[0], v = y[1], eps = -N * dphiOf(z) * v;
      const zf = z0 - 0.5 * G * t * t, vf = -G * t;
      S.push(t, {
        phi: N * phiOf(z) * 1000, eps, v: Math.abs(v), Ed: Mk * G * (z0 - z) * 1000, K: 0.5 * Mk * v * v * 1000, Q: y[2] * 1000,
        phiF: zf > -z0 ? N * phiOf(zf) * 1000 : NaN, epsF: zf > -z0 ? -N * dphiOf(zf) * vf : NaN, vF: zf > -z0 ? Math.abs(vf) : NaN,
      });
    }

    function drawScene() {
      const p = plots.anim; p.clear();
      const z = y[0], v = y[1], g1 = N * dphiOf(z), eps = -g1 * v, I = eps / R, rel = Imax > 0 ? I / Imax : 0;
      p.custom((c, pl) => {
        const W = pl.W, H = pl.H, top = 18, bot = H - 18;
        const zT = z0 + 0.03, zB = -z0 - 0.03, sc = (bot - top) / (zT - zB);
        const Y = (zz) => top + (zT - zz) * sc, cx = W * 0.3;
        const hx = PM.clamp((0.1 * W) / (a * sc), 1, 4); // horizontal exaggeration so that the coil stays visible
        const rx = a * sc * hx, ry = rx * 0.32;
        // field lines of the dipole (meridian plane), drawn around the magnet
        c.strokeStyle = "rgba(88,166,255,0.28)"; c.lineWidth = 1;
        for (const r0 of [0.012, 0.022, 0.036, 0.055, 0.08]) {
          for (const sgn of [-1, 1]) {
            c.beginPath();
            for (let k = 0; k <= 60; k++) { const th = 0.06 + ((Math.PI - 0.12) * k) / 60, r = r0 * Math.sin(th) ** 2; const X = cx + sgn * r * Math.sin(th) * sc * hx, YY = Y(z + r * Math.cos(th)); k ? c.lineTo(X, YY) : c.moveTo(X, YY); }
            c.stroke();
          }
        }
        // coil: back half
        const Yc = Y(0), turns = 5;
        const ring = (from, to) => {
          for (let k = 0; k < turns; k++) {
            const off = (k - (turns - 1) / 2) * 3;
            c.strokeStyle = k % 2 ? COPPER : "#b8763a"; c.lineWidth = 3; c.beginPath();
            c.ellipse(cx, Yc + off, rx, ry, 0, from, to); c.stroke();
          }
        };
        ring(Math.PI, 2 * Math.PI);
        // magnet
        const mw = Math.max(Math.min(0.6 * a, 0.007) * sc * hx, 5), ml = Math.max(0.025 * sc, 18), Ym = Y(z);
        const downCol = mz < 0 ? "#c0392b" : "#2f6fbf", upCol = mz < 0 ? "#2f6fbf" : "#c0392b";
        c.fillStyle = upCol; c.fillRect(cx - mw, Ym - ml / 2, 2 * mw, ml / 2);
        c.fillStyle = downCol; c.fillRect(cx - mw, Ym, 2 * mw, ml / 2);
        c.fillStyle = "rgba(255,255,255,0.18)"; c.beginPath(); c.ellipse(cx, Ym - ml / 2, mw, mw * 0.32, 0, 0, 2 * Math.PI); c.fill();
        c.strokeStyle = "rgba(0,0,0,0.5)"; c.lineWidth = 1; c.strokeRect(cx - mw, Ym - ml / 2, 2 * mw, ml);
        txt(c, mz < 0 ? "S" : "N", cx, Ym - ml / 4, { size: 11, bold: true, align: "center" });
        txt(c, mz < 0 ? "N" : "S", cx, Ym + ml / 4, { size: 11, bold: true, align: "center" });
        // coil: front half
        ring(0, Math.PI);
        // induced current dots on the front arc + chevron
        c.fillStyle = IND;
        for (let k = 0; k < 14; k++) {
          const ph = dotOff + (2 * Math.PI * k) / 14, sn = Math.sin(ph);
          if (sn > 0.05) continue; // only the front half (sin φ < 0) is visible
          c.globalAlpha = 0.3 + 0.7 * Math.min(1, Math.abs(rel) * 3);
          c.beginPath(); c.arc(cx + rx * Math.cos(ph), Yc - ry * sn, 3, 0, 2 * Math.PI); c.fill();
        }
        c.globalAlpha = 1;
        if (Math.abs(rel) > 0.03) chevron(c, cx, Yc + ry + 1, I > 0 ? 0 : Math.PI, IND, 8);
        // induced moment of the coil
        if (Math.abs(rel) > 0.03) {
          const L = Math.sign(I) * Math.min(1, Math.abs(rel)) * Math.max(40, rx * 1.2);
          arrowPx(c, cx + rx + 26, Yc, cx + rx + 26, Yc - L, PlotColors.pink, 3, 9);
          txt(c, "m_ind", cx + rx + 34, Yc - L / 2, { size: 11.5, bold: true, color: PlotColors.pink });
        }
        // forces on the magnet
        const Fg = Mk * G, Lg = 42;
        arrowPx(c, cx - mw - 16, Ym, cx - mw - 16, Ym + Lg, "#8b98a8", 2.2, 8);
        txt(c, "Mg", cx - mw - 22, Ym + Lg, { size: 11, align: "right", color: "#8b98a8" });
        if (brake && Math.abs(Fnow) > 0.01 * Fg) {
          const Lf = Math.min(Lg * Math.abs(Fnow) / Fg, 140) * Math.sign(Fnow);
          arrowPx(c, cx + mw + 16, Ym, cx + mw + 16, Ym - Lf, PlotColors.bad, 2.6, 9);
          txt(c, "F_Lenz", cx + mw + 22, Ym - Lf, { size: 11, color: PlotColors.bad });
        }
        txt(c, `coil: N = ${N}, a = ${PM.fmt(a * 100, 1)} cm`, cx - rx - 12, Yc, { size: 11, align: "right", color: COPPER, bg: "#0f151c" });
        txt(c, `ε = ${PM.fmt(eps, 3)} V   I = ${PM.fmt(I * 1000, 1)} mA`, 12, 16, { size: 12.5, bold: true, color: PlotColors.accent3 });
        // ---- profile panel Φ(z), Φ'(z)
        const x0 = W * 0.6, x1 = W * 0.96, xm = (x0 + x1) / 2, hwp = (x1 - x0) / 2;
        c.strokeStyle = "#2b3848"; c.lineWidth = 1; c.strokeRect(x0, top, x1 - x0, bot - top);
        c.beginPath(); c.moveTo(xm, top); c.lineTo(xm, bot); c.moveTo(x0, Yc); c.lineTo(x1, Yc); c.stroke();
        const fm = Math.abs(k0) / (a * a * a), dm = Math.abs(3 * k0 * 0.5 * a) / Math.pow(1.25 * a * a, 2.5);
        for (const [arr, col, mx] of [[profF, PlotColors.accent, fm], [profD, PlotColors.accent3, dm]]) {
          c.strokeStyle = col; c.lineWidth = 1.8; c.beginPath();
          for (let i = 0; i < PROF; i++) { const X = xm + (0.92 * hwp * arr[i]) / mx, YY = Y(profZ[i]); i ? c.lineTo(X, YY) : c.moveTo(X, YY); }
          c.stroke();
        }
        c.strokeStyle = "rgba(230,237,243,0.6)"; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(x0, Ym); c.lineTo(x1, Ym); c.stroke(); c.setLineDash([]);
        c.fillStyle = PlotColors.accent; c.beginPath(); c.arc(xm + (0.92 * hwp * phiOf(z)) / fm, Ym, 4.5, 0, 2 * Math.PI); c.fill();
        c.fillStyle = PlotColors.accent3; c.beginPath(); c.arc(xm + (0.92 * hwp * dphiOf(z)) / dm, Ym, 4.5, 0, 2 * Math.PI); c.fill();
        txt(c, "Φ(z)", x0 + 6, top + 12, { size: 11.5, bold: true, color: PlotColors.accent });
        txt(c, "dΦ/dz", x0 + 6, top + 28, { size: 11.5, bold: true, color: PlotColors.accent3 });
        txt(c, "coil plane z = 0", x1 - 6, Yc - 9, { size: 10.5, align: "right", color: "#8b98a8" });
        txt(c, `z = ${PM.fmt(z * 100, 1)} cm`, x1 - 6, top + 12, { size: 11.5, align: "right" });
        if (hx > 1.05) txt(c, `widths drawn ×${hx.toFixed(1)}`, 12, 34, { size: 10.5, color: "#8b98a8" });
        txt(c, "Lenz: the coil repels the approaching pole and attracts the receding one", 12, H - 12, { size: 11, color: "#c9d1d9" });
      });
    }

    function tplot(p, keys, scaler, floor, positive) {
      const n = S.n, x1 = Math.max(Tff * 1.05, t * 1.02);
      let m = 0; for (const k of keys) m = Math.max(m, maxAbs(S.d[k], n));
      const lim = scaler.fit(m, floor);
      p.setLimits([0, x1], positive ? [0, lim] : [-lim, lim]);
      p.clear(); if (!positive) p.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
      return S.T();
    }

    return {
      reset() {
        params();
        y[0] = z0; y[1] = 0; y[2] = 0; ws = null; t = 0; done = false; epP = epN = 0; Fnow = 0; dotOff = 0;
        S.clear(); for (const k in sc) sc[k].reset();
        record(); nextRec = dtRec;
      },
      step(dt) {
        if (done) { this.reset(); return; }
        const simDt = dt * slow;
        let n = Math.ceil(simDt / h); if (n > 6000) n = 6000;
        const hh = Math.min(h, simDt / n);
        for (let i = 0; i < n && !done; i++) {
          ws = PM.rk4(deriv, t, y, hh, ws); t += hh;
          const eps = -N * dphiOf(y[0]) * y[1];
          if (eps > epP) epP = eps; if (eps < epN) epN = eps;
          if (t >= nextRec) { record(); nextRec += dtRec * S.decim; }
          if (y[0] <= -z0) { done = true; record(); api.pause(); }
        }
        const g1 = N * dphiOf(y[0]);
        Fnow = brake ? -(g1 * g1 / R) * y[1] : 0;
        dotOff += dt * (-(g1 * y[1]) / R / (Imax || 1)) * 2.2;
      },
      render() {
        drawScene();
        const T = S.T();
        let p = plots.flux; tplot(p, ["phi", "phiF"], sc.f, 1e-6);
        p.line(T, S.Y("phiF"), { color: PlotColors.muted, width: 1.2, dash: [5, 4] });
        p.line(T, S.Y("phi"), { color: PlotColors.accent, width: 1.9 });
        p.legend([{ label: "N Φ", color: PlotColors.accent }, { label: "free fall", color: PlotColors.muted, dash: [5, 4] }], "tr");
        p = plots.emf; tplot(p, ["eps", "epsF"], sc.e, 1e-6);
        p.line(T, S.Y("epsF"), { color: PlotColors.muted, width: 1.2, dash: [5, 4] });
        p.line(T, S.Y("eps"), { color: PlotColors.accent3, width: 1.9 });
        p.legend([{ label: "ε (with braking setting)", color: PlotColors.accent3 }, { label: "free fall", color: PlotColors.muted, dash: [5, 4] }], "tl");
        p = plots.vel; tplot(p, ["v", "vF"], sc.v, 1e-3, true);
        p.line(T, S.Y("vF"), { color: PlotColors.muted, width: 1.2, dash: [5, 4] });
        p.line(T, S.Y("v"), { color: PlotColors.blue, width: 1.9 });
        p.legend([{ label: "|v|", color: PlotColors.blue }, { label: "free fall gt", color: PlotColors.muted, dash: [5, 4] }], "tl");
        p = plots.en; tplot(p, ["Ed"], sc.en, 1e-6, true);
        p.line(T, S.Y("Ed"), { color: PlotColors.text, width: 1.6 });
        p.line(T, S.Y("K"), { color: PlotColors.blue, width: 1.6 });
        p.line(T, S.Y("Q"), { color: PlotColors.bad, width: 1.6 });
        p.legend([{ label: "Mg(z₀ − z)", color: PlotColors.text }, { label: "kinetic K", color: PlotColors.blue }, { label: "Joule heat Q_J", color: PlotColors.bad }], "tl");

        const z = y[0], v = y[1], eps = -N * dphiOf(z) * v;
        const bal = Mk * G * (z0 - z) - 0.5 * Mk * v * v - (brake ? y[2] : 0); // without braking the Joule heat is not taken from the magnet
        M.set("z", `${PM.fmt(z * 100, 1)} cm / ${PM.fmt(Math.abs(v), 3)} m/s`);
        M.set("eps", PM.fmt(eps, 3) + " V");
        M.set("epk", `+${PM.fmt(epP, 3)} / ${PM.fmt(epN, 3)} V`);
        M.set("F", PM.fmt(Fnow / (Mk * G), 3));
        M.set("Q", PM.fmt(y[2] * 1000, 3) + " mJ");
        M.set("bal", PM.fmt(bal * 1000, 2) + " mJ");
        api.setTime(`t = ${PM.fmt(t * 1000, 1)} ms · ${slowTxt(slow)}${done ? " · finished" : ""}`);
      },
    };
  }

  // ================================================================== (c) sliding rod
  function mountRod(api, P) {
    const M = api.metrics([
      { id: "v", label: "Speed $v$" },
      { id: "vt", label: "Terminal speed $v_t=FR/B^2L^2$" },
      { id: "tau", label: "Time constant $\\tau=MR/B^2L^2$" },
      { id: "eps", label: "emf $BLv$ / current $I$" },
      { id: "F", label: "Magnetic force $B^2L^2v/R$" },
      { id: "bal", label: "$(W-\\Delta K-Q_J)/W$" },
    ]);
    const plots = api.plots([
      { id: "anim", title: "Rod on rails in a field perpendicular to the rail plane (top view)", span: 2, aspect: 0.38, axes: false, minHeight: 280, maxHeight: 440 },
      { id: "vel", title: "Speed $v(t)$: RK4 (solid) and analytic (dashed)", aspect: 0.5, xlabel: "t (s)", ylabel: "v (m/s)" },
      { id: "cur", title: "Induced current $I=BLv/R$", aspect: 0.5, xlabel: "t (s)", ylabel: "I (A)" },
      { id: "pow", title: "Power balance $Fv = I^2R + dK/dt$", aspect: 0.5, xlabel: "t (s)", ylabel: "P (W)" },
      { id: "en", title: "Energy budget", aspect: 0.5, xlabel: "t (s)", ylabel: "energy (J)" },
    ]);
    const S = Series(1600, ["v", "I", "Pin", "PJ", "dK", "W", "dKc", "Q", "KQ"], "grow");
    const sc = { v: Scaler(), i: Scaler(), p: Scaler(), e: Scaler() };
    const y = new Float64Array(4);
    let ws = null, t = 0, done = false, nextRec = 0, dotOff = 0;
    let B, L, R, Mr, F, v0, k, tau, vt, Tend, h, slow, dtRec, Iref, thRad;
    const NA = 300, anT = new Float64Array(NA), anV = new Float64Array(NA);

    function params() {
      B = P.rB; L = P.rL; R = P.rR; Mr = P.rM; thRad = (P.rth * Math.PI) / 180;
      F = P.rdrive === "force" ? P.rF : P.rdrive === "incline" ? Mr * G * Math.sin(thRad) : 0;
      v0 = P.rv0; if (P.rdrive === "push" && v0 <= 0) { v0 = 5; api.setControl("rv0", { value: 5 }); }
      k = (B * B * L * L) / R; tau = Mr / k; vt = F / k;
      Tend = 6 * tau; h = tau / 60; slow = Tend / 8; dtRec = Tend / 1400;
      Iref = Math.max(Math.abs(vt), Math.abs(v0)) * B * L / R || 1;
      for (let i = 0; i < NA; i++) { anT[i] = (Tend * i) / (NA - 1); anV[i] = vt + (v0 - vt) * Math.exp(-anT[i] / tau); }
    }
    function deriv(_t, s, o) { const v = s[1]; o[0] = v; o[1] = (F - k * v) / Mr; o[2] = F * v; o[3] = k * v * v; }
    function record() {
      const v = y[1], I = (B * L * v) / R, a = (F - k * v) / Mr, dK = 0.5 * Mr * (v * v - v0 * v0);
      S.push(t, { v, I, Pin: F * v, PJ: I * I * R, dK: Mr * v * a, W: y[2], dKc: dK, Q: y[3], KQ: dK + y[3] });
    }

    function drawScene() {
      const p = plots.anim; p.clear();
      const x = y[0], v = y[1], I = (B * L * v) / R, rel = I / Iref;
      p.custom((c, pl) => {
        const W = pl.W, H = pl.H, ppm = (0.5 * H) / L, Wv = W / ppm;
        const xv = Math.max(-0.12 * Wv, x - 0.62 * Wv);
        const X = (xx) => (xx - xv) * ppm, yT = H * 0.25, yB = H * 0.75;
        // B field symbols (into the screen), fixed to the world so they scroll
        const sp = Math.max(L / 3, 1e-3) * 1.0, kx0 = Math.floor(xv / sp);
        c.strokeStyle = "rgba(88,166,255,0.45)"; c.lineWidth = 1.3;
        for (let kx = kx0; kx * sp < xv + Wv + sp; kx++) {
          for (let j = -1; j <= 4; j++) {
            const px = X(kx * sp), py = yT + (j + 0.5) * (yB - yT) / 3.5 - 0.1 * (yB - yT);
            if (py < 8 || py > H - 8) continue;
            c.beginPath(); c.moveTo(px - 4, py - 4); c.lineTo(px + 4, py + 4); c.moveTo(px + 4, py - 4); c.lineTo(px - 4, py + 4); c.stroke();
          }
        }
        // flux area
        const xa = Math.max(X(0), -10), xr = X(x);
        c.fillStyle = "rgba(79,209,197,0.14)"; c.fillRect(xa, yT, Math.max(xr - xa, 0), yB - yT);
        // rails
        c.strokeStyle = "#a9b4c0"; c.lineWidth = 4;
        c.beginPath(); c.moveTo(Math.max(X(0), 0), yT); c.lineTo(W, yT); c.moveTo(Math.max(X(0), 0), yB); c.lineTo(W, yB); c.stroke();
        // distance ticks
        const tick = (() => { const raw = Wv / 6, m = Math.pow(10, Math.floor(Math.log10(raw))), q = raw / m; return (q < 1.5 ? 1 : q < 3.5 ? 2 : 5) * m; })();
        for (let xx = Math.ceil(xv / tick) * tick; xx < xv + Wv; xx += tick) {
          if (xx < 0) continue;
          const px = X(xx); c.strokeStyle = "#8b98a8"; c.lineWidth = 1; c.beginPath(); c.moveTo(px, yB + 4); c.lineTo(px, yB + 10); c.stroke();
          txt(c, `${PM.fmt(xx, tick < 1 ? (tick < 0.1 ? 2 : 1) : 0)} m`, px, yB + 20, { size: 10.5, align: "center", color: "#8b98a8" });
        }
        // resistor at x = 0
        const glow = Math.min(1, Math.abs(rel));
        if (X(0) > -20) { resistorPx(c, X(0), yT, X(0), yB, glow * glow, "#e6edf3"); txt(c, `R = ${PM.fmt(R, 1)} Ω`, X(0) + 14, (yT + yB) / 2, { size: 12, color: "#e6edf3", bg: "#0f151c" }); }
        else txt(c, `← resistor R = ${PM.fmt(R, 1)} Ω at x = 0`, 10, (yT + yB) / 2, { size: 11.5, color: "#c9d1d9", bg: "#0f151c" });
        // induced current dots around the loop (counter-clockwise for v > 0)
        const x0p = Math.max(X(0), -30), Lx = xr - x0p, Ly = yB - yT, per = 2 * (Lx + Ly);
        if (Lx > 2) {
          const loop = (s) => { s = ((s % per) + per) % per; if (s < Ly) return [xr, yB - s]; s -= Ly; if (s < Lx) return [xr - s, yT]; s -= Lx; if (s < Ly) return [x0p, yT + s]; s -= Ly; return [x0p + s, yB]; };
          const nd = Math.max(8, Math.round(per / 34));
          c.fillStyle = IND; c.globalAlpha = 0.3 + 0.7 * Math.min(1, Math.abs(rel) * 3);
          for (let i = 0; i < nd; i++) { const q = loop(dotOff * 34 + (i * per) / nd); if (q[0] < -5) continue; c.beginPath(); c.arc(q[0], q[1], 3, 0, 2 * Math.PI); c.fill(); }
          c.globalAlpha = 1;
        }
        // rod
        c.fillStyle = "#d9904a"; c.fillRect(xr - 5, yT - 14, 10, yB - yT + 28);
        c.strokeStyle = "#7a4a1d"; c.lineWidth = 1; c.strokeRect(xr - 5, yT - 14, 10, yB - yT + 28);
        if (Math.abs(rel) > 0.02) chevron(c, xr, (yT + yB) / 2, I > 0 ? -Math.PI / 2 : Math.PI / 2, "#3a2410", 7);
        // forces
        const Fm = k * v, Fs = Math.max(Math.abs(F), Math.abs(k * v0), 1e-12), Lmax = 0.22 * W, ym = (yT + yB) / 2;
        if (Math.abs(F) > 0) { const Lf = (Lmax * F) / Fs; arrowPx(c, xr + 8, ym - 16, xr + 8 + Lf, ym - 16, PlotColors.good, 3, 10); txt(c, P.rdrive === "incline" ? "Mg sin θ" : "F", xr + 14 + Lf, ym - 16, { size: 12, bold: true, color: PlotColors.good }); }
        if (Math.abs(Fm) > 0.005 * Fs) { const Lf = (Lmax * Fm) / Fs; arrowPx(c, xr - 8, ym + 16, xr - 8 - Lf, ym + 16, PlotColors.bad, 3, 10); txt(c, "F_B = −B²L²v/R", xr - 14 - Lf, ym + 16, { size: 12, bold: true, color: PlotColors.bad, align: "right", bg: "#0f151c" }); }
        // velocity arrow and labels
        txt(c, `v = ${PM.fmt(v, 3)} m/s`, xr, yT - 24, { size: 12, bold: true, align: "center", color: PlotColors.blue, bg: "#0f151c" });
        txt(c, "B into the screen (×)", W - 10, 22, { size: 11.5, align: "right", color: BCOL });
        txt(c, `Φ = BLx = ${PM.fmt(B * L * x, 3)} Wb    ε = BLv = ${PM.fmt(B * L * v, 3)} V    I = ${PM.fmt(I, 3)} A`, 10, 22, { size: 12, color: "#e6edf3", bg: "#0f151c" });
        txt(c, "Lenz: the induced current (counter-clockwise) opposes the growth of Φ, so the force on the rod brakes it", 10, H - 12, { size: 11, color: "#c9d1d9" });
        // incline inset
        if (P.rdrive === "incline") {
          const bx = W - 150, by = H - 34, bw = 120, bh = bw * Math.tan(thRad) * 0.6;
          c.fillStyle = "rgba(139,152,168,0.18)"; c.strokeStyle = "#8b98a8"; c.lineWidth = 1.2;
          c.beginPath(); c.moveTo(bx, by - bh); c.lineTo(bx + bw, by); c.lineTo(bx, by); c.closePath(); c.fill(); c.stroke();
          txt(c, `θ = ${P.rth}°  (side view)`, bx + bw, by - bh - 4, { size: 10.5, align: "right", color: "#c9d1d9" });
        }
      });
    }

    function tplot(p, keys, scaler, floor, positive) {
      const n = S.n;
      let m = 0; for (const kk of keys) m = Math.max(m, maxAbs(S.d[kk], n));
      if (keys.includes("v")) m = Math.max(m, Math.abs(vt), Math.abs(v0));
      const lim = scaler.fit(m, floor);
      let lo = -lim;
      if (positive) lo = 0;
      else { let mn = 0; for (const kk of keys) for (let i = 0; i < n; i++) mn = Math.min(mn, S.d[kk][i]); if (mn > -0.02 * lim) lo = -0.04 * lim; }
      p.setLimits([0, Tend], [lo, lim]);
      p.clear(); p.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
      return S.T();
    }

    return {
      reset() {
        params();
        y[0] = 0; y[1] = v0; y[2] = 0; y[3] = 0; ws = null; t = 0; done = false; dotOff = 0;
        S.clear(); for (const kk in sc) sc[kk].reset();
        record(); nextRec = dtRec;
      },
      step(dt) {
        if (done) { this.reset(); return; }
        const simDt = dt * slow, n = Math.max(1, Math.ceil(simDt / h)), hh = simDt / n;
        for (let i = 0; i < n && !done; i++) {
          ws = PM.rk4(deriv, t, y, hh, ws); t += hh;
          if (t >= nextRec) { record(); nextRec += dtRec * S.decim; }
          if (t >= Tend) { done = true; record(); api.pause(); }
        }
        dotOff += dt * ((B * L * y[1]) / R / Iref) * 2;
      },
      render() {
        drawScene();
        const T = S.T();
        let p = plots.vel; tplot(p, ["v"], sc.v, 1e-6);
        if (Math.abs(F) > 0) p.hline(vt, { color: PlotColors.text, dash: [3, 4], width: 1 });
        p.line(anT, anV, { color: PlotColors.text, width: 1.2, dash: [6, 4], alpha: 0.8 });
        p.line(T, S.Y("v"), { color: PlotColors.blue, width: 2 });
        p.vline(tau, { color: PlotColors.accent2, dash: [2, 4], width: 1 });
        p.legend([{ label: "v (RK4)", color: PlotColors.blue }, { label: "analytic", color: PlotColors.text, dash: [6, 4] }, { label: "t = τ", color: PlotColors.accent2, dash: [2, 4] }], "br");
        p = plots.cur; tplot(p, ["I"], sc.i, 1e-9);
        p.line(T, S.Y("I"), { color: IND, width: 1.9 });
        p = plots.pow; tplot(p, ["Pin", "PJ", "dK"], sc.p, 1e-9);
        p.line(T, S.Y("Pin"), { color: PlotColors.good, width: 1.8 });
        p.line(T, S.Y("PJ"), { color: PlotColors.bad, width: 1.6 });
        p.line(T, S.Y("dK"), { color: PlotColors.blue, width: 1.6 });
        p.legend([{ label: "F·v (input)", color: PlotColors.good }, { label: "I²R (heat)", color: PlotColors.bad }, { label: "dK/dt", color: PlotColors.blue }], "tr");
        p = plots.en; tplot(p, ["W", "dKc", "Q", "KQ"], sc.e, 1e-9);
        p.line(T, S.Y("W"), { color: PlotColors.good, width: 2.4 });
        p.line(T, S.Y("dKc"), { color: PlotColors.blue, width: 1.6 });
        p.line(T, S.Y("Q"), { color: PlotColors.bad, width: 1.6 });
        p.line(T, S.Y("KQ"), { color: PlotColors.text, width: 1.3, dash: [5, 4] });
        p.legend([{ label: "work W = ∫Fv dt", color: PlotColors.good }, { label: "ΔK", color: PlotColors.blue }, { label: "Joule heat Q_J", color: PlotColors.bad }, { label: "ΔK + Q_J", color: PlotColors.text, dash: [5, 4] }], "tl");

        const v = y[1], I = (B * L * v) / R, W = y[2], dK = 0.5 * Mr * (v * v - v0 * v0);
        const ref = Math.max(Math.abs(W), y[3], 1e-30);
        M.set("v", PM.fmt(v, 4) + " m/s");
        M.set("vt", Math.abs(F) > 0 ? PM.fmt(vt, 4) + " m/s" : "0 (no drive)");
        M.set("tau", PM.fmt(tau, 4) + " s");
        M.set("eps", `${PM.fmt(B * L * v, 3)} V / ${PM.fmt(I, 3)} A`);
        M.set("F", PM.fmt(k * v, 4) + " N");
        M.set("bal", t > 0 ? PM.fmt((W - dK - y[3]) / ref, 2) : "—");
        api.setTime(`t = ${PM.fmt(t, 3)} s / ${PM.fmt(Tend, 3)} s · ${slowTxt(slow)}${done ? " · finished" : ""}`);
      },
    };
  }
})();
