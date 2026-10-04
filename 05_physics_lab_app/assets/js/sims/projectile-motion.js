/* Projectile motion with air resistance — three force models (no drag, linear Stokes drag, quadratic drag
 * with wind) integrated side by side with a fixed-step RK4 integrator; ground impact found by cubic Hermite
 * interpolation inside the last step. Exact analytic solutions for the no-drag and linear-drag cases are
 * overlaid as an accuracy check. */
(function () {
  "use strict";

  const MU_AIR = 1.81e-5; // dynamic viscosity of air at ~15 °C (Pa·s)
  const PRESETS = {
    baseball: { m: 0.145, d: 0.074, Cd: 0.35, v0: 40, angle: 35, h0: 1 },
    golf: { m: 0.0459, d: 0.0427, Cd: 0.25, v0: 70, angle: 14, h0: 0 },
    pingpong: { m: 0.0027, d: 0.04, Cd: 0.5, v0: 20, angle: 30, h0: 1 },
    cannon: { m: 5.4, d: 0.117, Cd: 0.47, v0: 150, angle: 30, h0: 2 },
    raindrop: { m: 4.19e-6, d: 0.002, Cd: 0.5, v0: 5, angle: 0, h0: 100 },
  };
  const fmtMass = (lm) => {
    const m = Math.pow(10, lm);
    if (m >= 1) return m.toFixed(2) + " kg";
    if (m >= 1e-3) return (m * 1e3).toPrecision(3) + " g";
    return (m * 1e6).toPrecision(3) + " mg";
  };
  const fmtLen = (ld) => {
    const d = Math.pow(10, ld);
    if (d >= 0.01) return (d * 100).toPrecision(3) + " cm";
    return (d * 1000).toPrecision(3) + " mm";
  };

  App.register({
    id: "projectile-motion",
    category: "classical",
    group: "Mechanics",
    order: 10,
    title: "Projectile Motion with Air Resistance",
    icon: "🎯",
    subtitle: "The same projectile is fired three times at once — in vacuum, with linear (Stokes) drag and with quadratic drag — " +
      "and the three trajectories, speeds, energies and ranges are integrated live with RK4 and compared with the exact solutions.",
    notes: [
      { type: "info", html: "Pick a <b>preset</b> (baseball, golf ball, ping-pong ball, cannonball, raindrop) or set the projectile by hand. " +
        "The faint curves are the full precomputed flights, the bright trails the live projectiles; the white dashed curves are the " +
        "<b>exact</b> analytic solutions for no drag and for linear drag, which the numerical curves must cover. The table lists range, " +
        "maximum height, flight time, impact speed, terminal speed and the optimum launch angle of every model." },
    ],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "preset", type: "select", label: "Projectile preset", value: "baseball", options: [
        { value: "baseball", label: "Baseball" },
        { value: "golf", label: "Golf ball" },
        { value: "pingpong", label: "Ping-pong ball" },
        { value: "cannon", label: "Cannonball" },
        { value: "raindrop", label: "Raindrop (from 100 m)" },
        { value: "custom", label: "Custom" },
      ], help: "Baseball 145 g, 7.4 cm, C<sub>d</sub> 0.35 · golf ball 45.9 g, 4.27 cm, 0.25 · ping-pong 2.7 g, 4 cm, 0.5 · cannonball 5.4 kg, 11.7 cm, 0.47 · raindrop 2 mm, 4.2 mg, 0.5. A preset also sets a typical launch; changing mass, diameter or C<sub>d</sub> switches to Custom." },
      { type: "section", label: "Launch" },
      { id: "v0", type: "slider", label: "Launch speed $v_0$", min: 0, max: 200, step: 0.5, value: 40, unit: "m/s" },
      { id: "angle", type: "slider", label: "Launch angle $\\theta$", min: -30, max: 90, step: 0.5, value: 35, fmt: (v) => v.toFixed(1) + "°" },
      { id: "h0", type: "slider", label: "Launch height $h_0$", min: 0, max: 100, step: 0.5, value: 1, unit: "m" },
      { type: "section", label: "Projectile (sphere)" },
      { id: "lm", type: "slider", label: "Mass $m$ (log scale)", min: -6, max: 1, step: 0.01, value: Math.log10(0.145), fmt: fmtMass },
      { id: "ld", type: "slider", label: "Diameter $d$ (log scale)", min: -3, max: -0.5, step: 0.01, value: Math.log10(0.074), fmt: fmtLen },
      { id: "Cd", type: "slider", label: "Drag coefficient $C_d$", min: 0.05, max: 1.5, step: 0.01, value: 0.35 },
      { type: "section", label: "Air and gravity" },
      { id: "rho", type: "slider", label: "Air density $\\rho$", min: 0, max: 1.5, step: 0.01, value: 1.2, unit: "kg/m³",
        help: "Sea level ≈ 1.2 kg/m³, Denver ≈ 1.0, 10 km altitude ≈ 0.41. Zero switches the quadratic drag off." },
      { id: "wind", type: "slider", label: "Horizontal wind $w$", min: -20, max: 20, step: 0.5, value: 0, unit: "m/s",
        help: "Positive = tailwind (blowing towards +x). Drag acts on the velocity relative to the air, $\\mathbf v-w\\hat x$." },
      { id: "g", type: "slider", label: "Gravitational acceleration $g$", min: 1, max: 25, step: 0.01, value: 9.81, unit: "m/s²",
        help: "Moon 1.62, Mars 3.71, Earth 9.81, Jupiter 24.8." },
      { id: "bmode", type: "select", label: "Linear-drag coefficient $b$", value: "match", options: [
        { value: "match", label: "Matched: same terminal speed as quadratic drag" },
        { value: "stokes", label: "Physical Stokes law b = 3πμd" },
      ], help: "Stokes' law is only valid at Reynolds numbers Re ≲ 1 (tiny droplets, dust); for a ball it is negligibly small. " +
        "The matched choice gives a linear model with the same terminal speed, for a fair comparison of the two drag laws." },
      { type: "section", label: "Display" },
      { id: "showNone", type: "checkbox", label: "No drag (vacuum)", value: true },
      { id: "showLin", type: "checkbox", label: "Linear drag $-b\\mathbf v$", value: true },
      { id: "showQuad", type: "checkbox", label: "Quadratic drag $-\\tfrac12\\rho C_dA|\\mathbf v|\\mathbf v$", value: true },
      { id: "showVel", type: "checkbox", label: "Velocity vectors", value: true, live: true },
      { id: "showDrag", type: "checkbox", label: "Drag-force and gravity vectors (per unit mass)", value: false, live: true },
      { id: "showExact", type: "checkbox", label: "Overlay exact analytic solutions", value: true, live: true },
      { id: "loop", type: "checkbox", label: "Relaunch automatically", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A rigid, non-spinning sphere of mass $m$ (kg) and diameter $d$ (m), cross-section $A=\\pi d^2/4$, is launched from height
      $h_0$ (m) above flat ground with speed $v_0$ (m/s) at angle $\\theta$ above the horizontal, in a uniform gravitational field $g$
      (m/s²). The air has density $\\rho$ (kg/m³), dynamic viscosity $\\mu=1.81\\times10^{-5}$ Pa·s, and may move horizontally with a
      uniform wind speed $w$ (m/s). Motion is restricted to the vertical $x$–$y$ plane; the Earth is flat and non-rotating; there is
      no lift (Magnus force) and the drag coefficient $C_d$ is constant. The <em>same</em> launch is integrated with three different
      air-resistance laws so that their effects can be compared directly.</p>

      <h4>Equations being solved</h4>
      <p>Newton's second law with gravity and a drag force that depends on the velocity relative to the air,
      $\\mathbf u=\\mathbf v-w\\,\\hat{\\mathbf x}$:</p>
      <div class="callout">$$m\\,\\frac{d\\mathbf v}{dt}=-mg\\,\\hat{\\mathbf y}+\\mathbf F_d(\\mathbf u),\\qquad
        \\mathbf F_d=\\begin{cases}\\mathbf 0 & \\text{no drag}\\\\[2pt] -b\\,\\mathbf u & \\text{linear (Stokes)}\\\\[2pt]
        -\\tfrac12\\rho C_d A\\,|\\mathbf u|\\,\\mathbf u & \\text{quadratic}\\end{cases}$$</div>
      <p>with $d\\mathbf r/dt=\\mathbf v$. Stokes' law gives $b=3\\pi\\mu d$; it describes creeping flow, Reynolds number
      $\\mathrm{Re}=\\rho v d/\\mu\\lesssim1$. A thrown ball has $\\mathrm{Re}\\sim10^4$–$10^5$, where the quadratic law with
      $C_d\\approx0.2$–$0.5$ is appropriate. Setting $\\dot{\\mathbf v}=0$ in still air gives the <b>terminal speeds</b></p>
      $$v_t^{\\text{lin}}=\\frac{mg}{b},\\qquad v_t^{\\text{quad}}=\\sqrt{\\frac{2mg}{\\rho C_d A}} .$$
      <p><b>Exact solutions.</b> Without drag, $x=v_{0x}t$, $y=h_0+v_{0y}t-\\tfrac12gt^2$: a parabola with range
      $R=v_0^2\\sin2\\theta/g$ for $h_0=0$. Linear drag decouples the components; with $\\tau=m/b$ and wind $w$,</p>
      $$x(t)=wt+(v_{0x}-w)\\,\\tau\\left(1-e^{-t/\\tau}\\right),\\qquad
        y(t)=h_0+\\tau\\,(v_{0y}+g\\tau)\\left(1-e^{-t/\\tau}\\right)-g\\tau\\,t ,$$
      <p>so the velocity relaxes exponentially to $(w,-g\\tau)$. Quadratic drag couples $x$ and $y$ through $|\\mathbf u|$ and has no
      closed-form solution — it must be integrated numerically. The mechanical energy per unit mass,
      $\\varepsilon=\\tfrac12|\\mathbf v|^2+gy$, obeys $d\\varepsilon/dt=\\mathbf F_d\\cdot\\mathbf v/m$, i.e. it can only decrease in still
      air, while a tailwind can feed energy in.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Integrator.</b> The state $(x,y,v_x,v_y)$ of each model is advanced with the classical fourth-order Runge–Kutta method
        with the step $h=\\min\\big(T_0/2000,\\ 0.05/\\lambda\\big)$, where $T_0$ is the vacuum flight time and $\\lambda$ the
        instantaneous drag rate of the model ($b/m$ for linear drag, $k|\\mathbf u|$ with $k=\\rho C_dA/2m$ for quadratic drag), re-evaluated
        every step. The rule is deterministic, so the live and the pre-integrated flights take exactly the same steps. Extremely
        drag-dominated falls (a feather-like sphere) are cut off after $6\\times10^4$ steps and reported as "not landed". The animation integrates the projectiles live
        frame by frame, taking as many sub-steps of size $h$ as fit into the frame's time; the playback is scaled so that the
        longest flight lasts about four seconds at speed ×1.</li>
        <li><b>Ground impact.</b> When a step ends below $y=0$, the crossing is located inside that step by bisection on the cubic
        Hermite interpolant built from $y$ and $v_y$ at both ends; $x$, $v$ and $t$ are interpolated to that instant, so the range and
        flight time are accurate to $O(h^4)$ rather than $O(h)$. The apex is found the same way from the sign change of $v_y$.</li>
        <li><b>Metrics.</b> On every restart the full flights are pre-integrated (same step), giving range $R=x_{\\rm impact}$,
        maximum height, flight time, impact speed $|\\mathbf v_{\\rm impact}|$ and the dissipated fraction
        $1-\\varepsilon_{\\rm impact}/\\varepsilon_0$. The <b>range-versus-angle</b> curves repeat this for 61 angles between 0° and 90°
        (vacuum: exact formula $R=v_0\\cos\\theta\\,(v_0\\sin\\theta+\\sqrt{v_0^2\\sin^2\\theta+2gh_0})/g$); the optimum angle is refined
        by a parabola through the three best points.</li>
        <li><b>Accuracy check.</b> The numerical no-drag and linear-drag trajectories are compared with the exact formulas above at every
        stored sample; the maximum deviation $\\max|\\mathbf r_{\\rm RK4}-\\mathbf r_{\\rm exact}|$ is shown as a metric and should be
        far below a millimetre.</li>
        <li><b>Plots.</b> Speed $|\\mathbf v|(t)$ with the terminal speeds dotted; $\\varepsilon(t)/\\varepsilon_0$ in per cent; and the
        <b>hodograph</b> $(v_x,v_y)$: a vertical line in vacuum, a straight line towards $(w,-v_t)$ for linear drag (because
        $\\dot{\\mathbf v}+\\mathbf v/\\tau$ is constant), a curved path for quadratic drag.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>The vacuum benchmark.</b> Set $h_0=0$, $\\theta=45°$ and look at the blue curve: $R=v_0^2/g$ (163.1 m for 40 m/s), maximum
        height $v_0^2/4g$, flight time $\\sqrt2\\,v_0/g$. The range-versus-angle curve is $\\propto\\sin2\\theta$, symmetric about 45°.</li>
        <li><b>Drag lowers the optimum angle.</b> With the baseball preset the quadratic-drag range peaks clearly below 45° (about 41°),
        and the descending branch is steeper than the ascending one — the trajectory is no longer symmetric.</li>
        <li><b>Terminal velocity.</b> Choose the raindrop: it reaches $v_t\\approx6.6$ m/s within about a second and then falls at constant
        speed; the hodograph ends at $(w,-v_t)$. With 100 m to fall, the vacuum drop would hit at 44 m/s.</li>
        <li><b>Linear vs quadratic.</b> With matched $b$ both models share the same $v_t$, but quadratic drag is stronger at high speed
        and weaker at low speed: for a fast cannonball (150 m/s) the quadratic range is much shorter. Switch to the physical Stokes
        $b$ to see how negligible viscous drag is for a ball (the amber curve then hides under the blue one).</li>
        <li><b>Wind.</b> Give the ping-pong ball a 10 m/s headwind ($w=-10$): it can be blown back behind the launcher, and its energy
        plot shows strong dissipation. A tailwind raises the range and can make $\\varepsilon/\\varepsilon_0$ exceed 100 %.</li>
        <li><b>Other planets.</b> Lower $g$ to 1.62 (Moon) with $\\rho=0$: ranges grow by $9.81/1.62\\approx6$.</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>Real balls have a speed-dependent $C_d(\\mathrm{Re})$ (the "drag crisis" near $\\mathrm{Re}\\approx3\\times10^5$), spin-induced lift
      (Magnus force, essential for golf drives), and air density decreasing with altitude; raindrops flatten and their $C_d$ grows. See
      J. R. Taylor, <em>Classical Mechanics</em>, ch. 2; D. Morin, <em>Introduction to Classical Mechanics</em>, ch. 3;
      Marion &amp; Thornton, <em>Classical Dynamics</em>, ch. 2; and R. K. Adair, <em>The Physics of Baseball</em>.</p>`,

    mount(api) {
      const P = api.params;
      const NM = 3, NAMES = ["No drag", "Linear drag", "Quadratic drag"];
      const COL = [PlotColors.blue, PlotColors.accent3, PlotColors.accent];
      const SHOW = ["showNone", "showLin", "showQuad"];
      const NA = 61, ANG = PM.linspace(0, 90, NA), RANGE = [new Float64Array(NA), new Float64Array(NA), new Float64Array(NA)];
      const CAP = 1024;

      const plots = api.plots([
        { id: "traj", title: "Trajectories — bright: live projectiles, faint: full flights, white dashed: exact solutions", span: 2, aspect: 0.42, equal: true,
          xlabel: "horizontal distance x (m)", ylabel: "height y (m)" },
        { id: "ra", title: "Range vs launch angle (current angle: vertical line)", aspect: 0.62, xlim: [0, 90], xlabel: "launch angle θ (°)", ylabel: "range R (m)" },
        { id: "sp", title: "Speed $|\\mathbf v|(t)$ — dotted: terminal speeds", aspect: 0.62, xlabel: "time t (s)", ylabel: "speed (m/s)" },
        { id: "en", title: "Mechanical energy per unit mass $\\varepsilon(t)/\\varepsilon_0$, with $\\varepsilon=\\tfrac12v^2+gy$", aspect: 0.62, xlabel: "time t (s)", ylabel: "ε / ε₀ (%)" },
        { id: "hod", title: "Hodograph $(v_x, v_y)$ — × marks the terminal velocity", aspect: 0.62, equal: true, xlabel: "vx (m/s)", ylabel: "vy (m/s)" },
      ]);
      const M = api.metrics([
        { id: "re", label: "Reynolds number at launch $\\mathrm{Re}=\\rho v_0 d/\\mu$" },
        { id: "fw", label: "Quadratic drag / weight at launch" },
        { id: "b", label: "Linear coefficient $b$ (kg/s)" },
        { id: "e0", label: "RK4 vs exact, no drag: max $|\\Delta\\mathbf r|$" },
        { id: "e1", label: "RK4 vs exact, linear drag: max $|\\Delta\\mathbf r|$" },
      ]);

      // results table (per model), placed between the metric cards and the plots
      const tbl = document.createElement("div");
      tbl.style.cssText = "overflow-x:auto;margin:0 0 12px;background:var(--panel);border:1px solid var(--border);border-radius:8px;";
      const plotsBox = api.stage.querySelector(".plots");
      api.stage.insertBefore(tbl, plotsBox);

      // ---------------------------------------------------------------- dynamics
      let g = 9.81, w = 0, kl = 0, kq = 0, b = 0, h0 = 0, vx0 = 0, vy0 = 0, mass = 1, diam = 0.1;
      const hStep = [1e-3, 1e-3, 1e-3];
      const K1 = new Float64Array(4), K2 = new Float64Array(4), K3 = new Float64Array(4), K4 = new Float64Array(4);
      const TMP = new Float64Array(4), PREV = new Float64Array(4), D = new Float64Array(4);

      function deriv(mo, s, o) {
        const ux = s[2] - w, uy = s[3];
        o[0] = s[2]; o[1] = s[3];
        if (mo === 0) { o[2] = 0; o[3] = -g; }
        else if (mo === 1) { o[2] = -kl * ux; o[3] = -g - kl * uy; }
        else { const sp = Math.sqrt(ux * ux + uy * uy); o[2] = -kq * sp * ux; o[3] = -g - kq * sp * uy; }
      }
      function rk4(mo, s, h) {
        deriv(mo, s, K1);
        for (let i = 0; i < 4; i++) TMP[i] = s[i] + 0.5 * h * K1[i];
        deriv(mo, TMP, K2);
        for (let i = 0; i < 4; i++) TMP[i] = s[i] + 0.5 * h * K2[i];
        deriv(mo, TMP, K3);
        for (let i = 0; i < 4; i++) TMP[i] = s[i] + h * K3[i];
        deriv(mo, TMP, K4);
        for (let i = 0; i < 4; i++) s[i] += (h / 6) * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
      }
      const herm = (p0, p1, m0, m1, f) => {
        const f2 = f * f, f3 = f2 * f;
        return (2 * f3 - 3 * f2 + 1) * p0 + (f3 - 2 * f2 + f) * m0 + (-2 * f3 + 3 * f2) * p1 + (f3 - f2) * m1;
      };
      function newState() { return { s: new Float64Array(4), t: 0, landed: false, H: 0, xH: 0, apex: false }; }
      function initState(st, ux0, uy0) {
        st.s[0] = 0; st.s[1] = h0; st.s[2] = ux0; st.s[3] = uy0;
        st.t = 0; st.landed = false; st.H = h0; st.xH = 0; st.apex = uy0 <= 0;
      }
      /** One RK4 step with apex and ground-impact detection (Hermite interpolation inside the step). */
      function stepImpact(mo, st, h) {
        const s = st.s;
        PREV.set(s);
        rk4(mo, s, h);
        if (PREV[3] > 0 && s[3] <= 0) {
          const f = PREV[3] / (PREV[3] - s[3]);
          const ya = herm(PREV[1], s[1], PREV[3] * h, s[3] * h, f);
          if (ya > st.H) { st.H = ya; st.xH = herm(PREV[0], s[0], PREV[2] * h, s[2] * h, f); }
          st.apex = true;
        }
        if (s[1] > st.H) { st.H = s[1]; st.xH = s[0]; }
        if (s[1] < 0) {
          const y0 = PREV[1], y1 = s[1], d0 = PREV[3] * h, d1 = s[3] * h;
          let a = 0, c = 1;
          for (let it = 0; it < 50; it++) { const mid = 0.5 * (a + c); if (herm(y0, y1, d0, d1, mid) > 0) a = mid; else c = mid; }
          const f = 0.5 * (a + c);
          s[0] = herm(PREV[0], s[0], PREV[2] * h, s[2] * h, f);
          s[1] = 0;
          s[2] = PREV[2] + f * (s[2] - PREV[2]);
          s[3] = PREV[3] + f * (s[3] - PREV[3]);
          st.t += f * h; st.landed = true; st.apex = true;
          return true;
        }
        st.t += h;
        return false;
      }
      /** Base step: a fraction of the vacuum flight time (raised if needed so that the slowest fall fits into maxSteps). */
      function baseStep(ux0, uy0, mo, nPer, maxSteps) {
        const T0 = (uy0 + Math.sqrt(Math.max(uy0 * uy0 + 2 * g * h0, 0))) / g;
        const H0 = h0 + Math.max(uy0, 0) ** 2 / (2 * g);
        let vterm = Infinity;
        if (mo === 1 && kl > 0) vterm = g / kl;
        if (mo === 2 && kq > 0) vterm = Math.sqrt(g / kq);
        const Tup = T0 + (isFinite(vterm) ? H0 / vterm : 0);
        return Math.max(Math.max(T0, 1e-3) / nPer, Tup / maxSteps);
      }
      /** Step actually taken: the base step, limited to frac/λ with λ the instantaneous drag rate (b/m or k|u|). */
      function hFor(mo, s, base, frac) {
        const lam = mo === 1 ? kl : mo === 2 ? kq * Math.hypot(s[2] - w, s[3]) : 0;
        return lam > 0 ? Math.min(base, frac / lam) : base;
      }

      // decimating sample recorder (keeps at most CAP points, halving when full)
      function makeRec() { return { t: new Float64Array(CAP), x: new Float64Array(CAP), y: new Float64Array(CAP), vx: new Float64Array(CAP), vy: new Float64Array(CAP), n: 0, stride: 1, cnt: 0 }; }
      function recClear(r) { r.n = 0; r.stride = 1; r.cnt = 0; }
      function recPush(r, t, s, force) {
        if (!force && r.cnt++ % r.stride !== 0) return;
        if (r.n >= CAP) {
          let j = 0;
          for (let i = 0; i < r.n; i += 2, j++) { r.t[j] = r.t[i]; r.x[j] = r.x[i]; r.y[j] = r.y[i]; r.vx[j] = r.vx[i]; r.vy[j] = r.vy[i]; }
          r.n = j; r.stride *= 2;
        }
        const k = r.n++;
        r.t[k] = t; r.x[k] = s[0]; r.y[k] = s[1]; r.vx[k] = s[2]; r.vy[k] = s[3];
      }

      const pre = [newState(), newState(), newState()], live = [newState(), newState(), newState()];
      const preRec = [makeRec(), makeRec(), makeRec()], trail = [makeRec(), makeRec(), makeRec()];
      const tmpSt = newState();
      const scrX = new Float64Array(CAP), scrY = new Float64Array(CAP);
      const exX = new Float64Array(300), exY = new Float64Array(300);
      let res = [], tSim = 0, rate = 1, hold = 0, allDone = false, Tmax = 1, E0 = 1, shown = [true, true, true], optAng = [NaN, NaN, NaN];
      let vt = { lin: Infinity, quad: Infinity }, maxStepsPre = 60000;

      function exactPos(mo, t, out) {
        if (mo === 0 || kl <= 0) { out[0] = vx0 * t; out[1] = h0 + vy0 * t - 0.5 * g * t * t; return; }
        const tau = 1 / kl, e = -Math.expm1(-t * kl);
        out[0] = w * t + (vx0 - w) * tau * e;
        out[1] = h0 + tau * (vy0 + g * tau) * e - g * tau * t;
      }
      const EX = new Float64Array(2);
      function exactError(mo) {
        const r = preRec[mo];
        let mx = 0;
        for (let i = 0; i < r.n; i++) {
          exactPos(mo, r.t[i], EX);
          mx = Math.max(mx, Math.hypot(r.x[i] - EX[0], r.y[i] - EX[1]));
        }
        return mx;
      }
      const energy = (vx, vy, y) => 0.5 * (vx * vx + vy * vy) + g * y;
      const fmtLenM = (x) => (Math.abs(x) >= 1000 ? (x / 1000).toFixed(2) + " km" : PM.fmt(x, Math.abs(x) >= 100 ? 1 : 2) + " m");

      function launchLive() {
        for (let mo = 0; mo < NM; mo++) {
          initState(live[mo], vx0, vy0);
          recClear(trail[mo]);
          recPush(trail[mo], 0, live[mo].s, true);
        }
        tSim = 0; hold = 0; allDone = false;
      }

      function reset() {
        g = P.g; w = P.wind; h0 = P.h0;
        mass = Math.pow(10, P.lm); diam = Math.pow(10, P.ld);
        const A = Math.PI * diam * diam / 4;
        kq = 0.5 * P.rho * P.Cd * A / mass;
        vt.quad = kq > 0 ? Math.sqrt(g / kq) : Infinity;
        if (P.bmode === "stokes") b = 3 * Math.PI * MU_AIR * diam;
        else b = kq > 0 ? mass * g / vt.quad : 0;
        kl = b / mass;
        vt.lin = kl > 0 ? g / kl : Infinity;
        const th = P.angle * Math.PI / 180;
        vx0 = P.v0 * Math.cos(th); vy0 = P.v0 * Math.sin(th);
        if (Math.abs(vx0) < 1e-12) vx0 = 0;
        shown = SHOW.map((k) => !!P[k]);

        // --- full flights (metrics, framing)
        E0 = energy(vx0, vy0, h0);
        res = [];
        for (let mo = 0; mo < NM; mo++) {
          const st = pre[mo], r = preRec[mo];
          hStep[mo] = baseStep(vx0, vy0, mo, 2000, maxStepsPre);
          initState(st, vx0, vy0); recClear(r); recPush(r, 0, st.s, true);
          for (let i = 0; i < maxStepsPre && !st.landed; i++) { stepImpact(mo, st, hFor(mo, st.s, hStep[mo], 0.05)); recPush(r, st.t, st.s, st.landed); }
          if (!st.landed) recPush(r, st.t, st.s, true);
          const s = st.s, Ei = energy(s[2], s[3], s[1]);
          const cl = (v) => (Math.abs(v) < 1e-9 ? 0 : v);
          res.push({ landed: st.landed, R: cl(s[0]), H: cl(st.H), T: cl(st.t), vImp: cl(Math.hypot(s[2], s[3])), loss: E0 > 1e-12 ? cl(1 - Ei / E0) : NaN });
        }

        // --- range vs launch angle
        for (let k = 0; k < NA; k++) {
          const a = ANG[k] * Math.PI / 180, ux = P.v0 * Math.cos(a), uy = P.v0 * Math.sin(a);
          RANGE[0][k] = ux * (uy + Math.sqrt(uy * uy + 2 * g * h0)) / g;
          for (let mo = 1; mo < NM; mo++) {
            const hA = baseStep(ux, uy, mo, 150, 4000);
            initState(tmpSt, ux, uy);
            let i = 0;
            while (!tmpSt.landed && i++ < 4000) stepImpact(mo, tmpSt, hFor(mo, tmpSt.s, hA, 0.2));
            RANGE[mo][k] = tmpSt.landed ? tmpSt.s[0] : NaN;
          }
        }
        for (let mo = 0; mo < NM; mo++) {
          let best = -1, bv = -Infinity;
          for (let k = 0; k < NA; k++) if (RANGE[mo][k] > bv) { bv = RANGE[mo][k]; best = k; }
          let opt = best >= 0 ? ANG[best] : NaN;
          if (best > 0 && best < NA - 1) {
            const y0 = RANGE[mo][best - 1], y1 = RANGE[mo][best], y2 = RANGE[mo][best + 1], den = y0 - 2 * y1 + y2;
            if (den < 0) opt = ANG[best] + 0.5 * (y0 - y2) / den * (ANG[1] - ANG[0]);
          }
          optAng[mo] = P.v0 > 0 ? opt : NaN;
        }

        // --- framing
        let x0 = 0, x1 = 1e-9, y1 = Math.max(h0, 1e-9), tM = 0, vM = 1e-9, eMax = 100, hx0 = Infinity, hx1 = -Infinity, hy0 = Infinity, hy1 = -Infinity;
        const any = shown.some(Boolean);
        for (let mo = 0; mo < NM; mo++) {
          if (any && !shown[mo]) continue;
          const r = preRec[mo];
          for (let i = 0; i < r.n; i++) {
            x0 = Math.min(x0, r.x[i]); x1 = Math.max(x1, r.x[i]); y1 = Math.max(y1, r.y[i]);
            const sp = Math.hypot(r.vx[i], r.vy[i]); vM = Math.max(vM, sp);
            if (E0 > 0) eMax = Math.max(eMax, 100 * energy(r.vx[i], r.vy[i], r.y[i]) / E0);
            hx0 = Math.min(hx0, r.vx[i]); hx1 = Math.max(hx1, r.vx[i]); hy0 = Math.min(hy0, r.vy[i]); hy1 = Math.max(hy1, r.vy[i]);
          }
          tM = Math.max(tM, r.t[r.n - 1]);
        }
        const span = Math.max(x1 - x0, y1, 1e-3);
        plots.traj.setLimits([x0 - 0.04 * span, x1 + 0.04 * span], [-0.05 * span, y1 + 0.1 * span]);
        Tmax = Math.max(tM, 1e-3);
        plots.sp.setLimits([0, Tmax * 1.02], [0, vM * 1.12]);
        plots.en.setLimits([0, Tmax * 1.02], [0, eMax * 1.06]);
        const hs = Math.max(hx1 - hx0, hy1 - hy0, 1e-3);
        plots.hod.setLimits([Math.min(hx0, w, 0) - 0.08 * hs, Math.max(hx1, w, 0) + 0.08 * hs], [hy0 - 0.08 * hs, Math.max(hy1, 0) + 0.08 * hs]);
        let rMin = 0, rMax = 1e-9;
        for (let mo = 0; mo < NM; mo++) {
          if (any && !shown[mo]) continue;
          for (let k = 0; k < NA; k++) if (isFinite(RANGE[mo][k])) { rMin = Math.min(rMin, RANGE[mo][k]); rMax = Math.max(rMax, RANGE[mo][k]); }
        }
        plots.ra.setLimits(null, [rMin - 0.05 * (rMax - rMin), rMax + 0.1 * (rMax - rMin)]);
        rate = Math.max(Tmax / 4, 1e-3);

        // --- metrics and table
        const Re = P.rho * Math.hypot(vx0 - w, vy0) * diam / MU_AIR;
        M.set("re", PM.fmt(Re, 0));
        M.set("fw", PM.fmt(kq * ((vx0 - w) ** 2 + vy0 ** 2) / g, 3));
        M.set("b", PM.fmt(b, 3));
        M.set("e0", PM.fmt(exactError(0), 2) + " m");
        M.set("e1", PM.fmt(exactError(1), 2) + " m");
        buildTable();
        launchLive();
      }

      function buildTable() {
        const vts = [Infinity, vt.lin, vt.quad];
        const cell = "padding:6px 10px;border-bottom:1px solid var(--border);white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums;";
        const head = "padding:6px 10px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600;font-size:12px;text-align:right;white-space:nowrap;";
        let h = `<table style="border-collapse:collapse;width:100%;font-size:13.5px"><tr><th style="${head}text-align:left">Model</th>` +
          ["Range R", "Max height", "Flight time", "Impact speed", "Terminal v<sub>t</sub>", "Optimum θ*", "Energy lost"].map((s) => `<th style="${head}">${s}</th>`).join("") + "</tr>";
        for (let mo = 0; mo < NM; mo++) {
          const r = res[mo], op = shown[mo] ? "1" : "0.4";
          const nl = r.landed ? "" : " (not landed)";
          h += `<tr style="opacity:${op}"><td style="${cell}text-align:left"><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${COL[mo]};margin-right:8px"></span>${NAMES[mo]}</td>` +
            `<td style="${cell}">${fmtLenM(r.R)}${nl}</td><td style="${cell}">${fmtLenM(r.H)}</td><td style="${cell}">${PM.fmt(r.T, 2)} s</td>` +
            `<td style="${cell}">${PM.fmt(r.vImp, 2)} m/s</td><td style="${cell}">${isFinite(vts[mo]) ? PM.fmt(vts[mo], 2) + " m/s" : "∞"}</td>` +
            `<td style="${cell}">${isFinite(optAng[mo]) ? optAng[mo].toFixed(1) + "°" : "—"}</td><td style="${cell}">${isFinite(r.loss) ? (Math.abs(100 * r.loss) < 0.05 ? "0.0" : (100 * r.loss).toFixed(1)) + " %" : "—"}</td></tr>`;
        }
        tbl.innerHTML = h + "</table>";
      }

      function step(dt) {
        if (allDone) {
          hold += dt;
          if (P.loop && hold > 1.2) launchLive();
          return;
        }
        tSim += dt * rate;
        let done = true;
        for (let mo = 0; mo < NM; mo++) {
          if (!shown[mo]) continue;
          const L = live[mo];
          if (L.landed) continue;
          const tEnd = pre[mo].t;
          let guard = 0;
          let hm = hFor(mo, L.s, hStep[mo], 0.05);
          while (!L.landed && L.t + hm <= tSim + 1e-12 && L.t < tEnd && guard++ < maxStepsPre) { stepImpact(mo, L, hm); hm = hFor(mo, L.s, hStep[mo], 0.05); }
          if (L.t >= tEnd - 1e-12 && !L.landed) L.landed = true; // flight cut at the pre-integration limit
          recPush(trail[mo], L.t, L.s, L.landed);
          if (!L.landed) done = false;
        }
        allDone = done;
      }

      // ---------------------------------------------------------------- drawing
      function drawExact(p) {
        if (!P.showExact) return;
        for (const mo of [0, 1]) {
          if (!shown[mo]) continue;
          const T = pre[mo].t, n = exX.length;
          for (let i = 0; i < n; i++) { exactPos(mo, (T * i) / (n - 1), EX); exX[i] = EX[0]; exY[i] = Math.max(EX[1], -1e-9); }
          p.line(exX, exY, { color: "#ffffff", width: 1.2, dash: [5, 5], alpha: 0.75 });
        }
      }
      function render() {
        const p = plots.traj;
        const [X0, X1] = p.visibleXlim, [Y0, Y1] = p.visibleYlim, span = Math.max(X1 - X0, Y1 - Y0);
        p.clear();
        p.rect(X0, Y0, X1, 0, { color: "#2a2418", alpha: 0.9 });
        p.hline(0, { color: "#7a6640", width: 1.5 });
        if (h0 > 0) p.rect(-0.006 * span, 0, 0, h0, { color: PlotColors.muted, alpha: 0.25 });
        for (let mo = 0; mo < NM; mo++) {
          if (!shown[mo]) continue;
          const r = preRec[mo];
          p.line(r.x.subarray(0, r.n), r.y.subarray(0, r.n), { color: COL[mo], width: 1.2, alpha: 0.2 });
        }
        drawExact(p);
        const sV = (0.11 * span) / Math.max(P.v0, 1), sA = (0.08 * span) / g;
        for (let mo = 0; mo < NM; mo++) {
          if (!shown[mo]) continue;
          const tr = trail[mo], L = live[mo], s = L.s;
          p.line(tr.x.subarray(0, tr.n), tr.y.subarray(0, tr.n), { color: COL[mo], width: 2.2 });
          if (L.apex && L.H > h0 + 1e-9) p.points([L.xH], [L.H], { color: COL[mo], size: 3.5, stroke: "#0f151c" });
          if (L.landed && pre[mo].landed) {
            p.segment(s[0], -0.02 * span, s[0], 0.02 * span, { color: COL[mo], width: 2.5 });
            p.text(s[0], 0, `R = ${fmtLenM(s[0])}`, { dy: -16 - 17 * mo, align: "center", size: 11, color: COL[mo], bg: "#0f151c" });
          }
          if (P.showDrag && !L.landed) {
            deriv(mo, s, D);
            p.arrow(s[0], s[1], s[0], s[1] - g * sA, { color: PlotColors.muted, width: 1.4, head: 7 });
            if (mo > 0) p.arrow(s[0], s[1], s[0] + D[2] * sA, s[1] + (D[3] + g) * sA, { color: PlotColors.bad, width: 1.8, head: 7 });
          }
          if (P.showVel && !L.landed) p.arrow(s[0], s[1], s[0] + s[2] * sV, s[1] + s[3] * sV, { color: COL[mo], width: 2, head: 8 });
          p.circle(s[0], s[1], 5.5, { px: true, color: COL[mo], stroke: "#ffffff", strokeWidth: 1.2 });
        }
        const lab = [`t = ${PM.fmt(tSim, 2)} s   (playback ×${PM.fmt(rate * (App.current ? App.current.speed : 1), 2)})`];
        if (w !== 0) lab.push(`wind ${w > 0 ? "→ +" : "← "}${PM.fmt(w, 1)} m/s`);
        p.label(lab, "tl", { size: 11.5 });
        const leg = [];
        for (let mo = 0; mo < NM; mo++) if (shown[mo]) leg.push({ label: NAMES[mo], color: COL[mo] });
        if (P.showExact && (shown[0] || shown[1])) leg.push({ label: "exact (analytic)", color: "#ffffff", dash: [5, 5] });
        if (P.showDrag) leg.push({ label: "drag / m", color: PlotColors.bad }, { label: "gravity g", color: PlotColors.muted });
        if (leg.length) p.legend(leg, "tr");

        // range vs angle
        const pr = plots.ra;
        pr.clear();
        pr.vline(45, { color: PlotColors.muted, dash: [3, 4], alpha: 0.6 });
        if (P.angle >= 0) pr.vline(P.angle, { color: "#ffffff", alpha: 0.7 });
        let lk = 0;
        for (let mo = 0; mo < NM; mo++) {
          if (!shown[mo]) continue;
          pr.line(ANG, RANGE[mo], { color: COL[mo], width: 2 });
          if (isFinite(optAng[mo])) {
            let ri = 0; // range at the optimum (linear interpolation in the table)
            const kk = Math.min(NA - 2, Math.floor(optAng[mo] / (ANG[1] - ANG[0])));
            const f = optAng[mo] / (ANG[1] - ANG[0]) - kk;
            ri = RANGE[mo][kk] * (1 - f) + RANGE[mo][kk + 1] * f;
            pr.points([optAng[mo]], [ri], { color: COL[mo], size: 4, stroke: "#0f151c" });
            pr.textPx(pr.m.l + pr._v.pw - 8, pr.m.t + 14 + 17 * lk, `θ* = ${optAng[mo].toFixed(1)}°  (${NAMES[mo].toLowerCase()})`, { align: "right", color: COL[mo], size: 11, bg: "#0f151c" });
            lk++;
          }
        }

        // speed and energy vs time, hodograph
        const ps = plots.sp, pe = plots.en, ph = plots.hod;
        ps.clear(); pe.clear(); ph.clear();
        ph.hline(0, { color: PlotColors.muted, alpha: 0.4 }); ph.vline(0, { color: PlotColors.muted, alpha: 0.4 });
        pe.hline(100, { color: PlotColors.muted, dash: [3, 4], alpha: 0.6 });
        const vts = [Infinity, vt.lin, vt.quad];
        for (let mo = 0; mo < NM; mo++) {
          if (!shown[mo]) continue;
          if (isFinite(vts[mo])) ps.hline(vts[mo], { color: COL[mo], dash: [2, 4], width: 1.4 });
          for (const [r, a] of [[preRec[mo], 0.2], [trail[mo], 1]]) {
            for (let i = 0; i < r.n; i++) { scrX[i] = Math.hypot(r.vx[i], r.vy[i]); scrY[i] = E0 > 0 ? 100 * energy(r.vx[i], r.vy[i], r.y[i]) / E0 : NaN; }
            const tt = r.t.subarray(0, r.n);
            ps.line(tt, scrX.subarray(0, r.n), { color: COL[mo], width: a < 1 ? 1.2 : 2, alpha: a });
            pe.line(tt, scrY.subarray(0, r.n), { color: COL[mo], width: a < 1 ? 1.2 : 2, alpha: a });
            ph.line(r.vx.subarray(0, r.n), r.vy.subarray(0, r.n), { color: COL[mo], width: a < 1 ? 1.2 : 2, alpha: a });
          }
          const L = live[mo];
          ph.points([L.s[2]], [L.s[3]], { color: COL[mo], size: 4 });
          if (mo > 0 && isFinite(vts[mo])) {
            ph.text(w, -vts[mo], "×", { align: "center", size: 16, color: COL[mo], bold: true });
          }
        }
        pe.label(res.map((r, mo) => (shown[mo] && isFinite(r.loss) ? `${NAMES[mo]}: ${Math.abs(100 * r.loss) < 0.05 ? "0.0" : (100 * r.loss).toFixed(1)} % dissipated` : null)).filter(Boolean), "bl", { size: 11 });
        api.setTime(`t = ${PM.fmt(tSim, 2)} s`);
      }

      return {
        reset,
        step,
        render,
        onParam(id) {
          if (id === "preset" && P.preset !== "custom") {
            const q = PRESETS[P.preset];
            api.setControl("lm", { value: Math.log10(q.m) });
            api.setControl("ld", { value: Math.log10(q.d) });
            api.setControl("Cd", { value: q.Cd });
            api.setControl("v0", { value: q.v0 });
            api.setControl("angle", { value: q.angle });
            api.setControl("h0", { value: q.h0 });
          } else if (id === "lm" || id === "ld" || id === "Cd") {
            api.setControl("preset", { value: "custom" });
          }
        },
      };
    },
  });
})();
