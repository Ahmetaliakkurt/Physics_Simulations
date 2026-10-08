/* Electron in a rectangular magnetic-field region with a field-free hole in the middle.
   B = B ẑ inside the outer rectangle, B = 0 inside the inner rectangle (the "window") and outside.
   Relativistic motion, integrated with an exact-rotation leapfrog step (speed conserved exactly). */
(function () {
  "use strict";

  // ------------------------------------------------------------------ constants (SI)
  const E_CH = 1.602176634e-19;   // elementary charge [C]
  const ME = 9.1093837015e-31;    // electron mass [kg]
  const C = 299792458;            // speed of light [m/s]
  const MEC2_EV = 510998.95;      // electron rest energy [eV]
  const CM = 0.01;

  const PARTICLES = {
    electron: { label: "Electron (e⁻)", q: -1, m: 1 },
    positron: { label: "Positron (e⁺)", q: 1, m: 1 },
    proton: { label: "Proton (p⁺)", q: 1, m: 1836.15267 },
    alpha: { label: "Alpha particle (He²⁺)", q: 2, m: 7294.29954 },
    custom: { label: "Custom q and m", q: null, m: null },
  };

  const MAX_STEPS = 400000;  // safety cap per trajectory
  const MAX_PTS = 5000;      // points kept for drawing one trajectory
  const MARGIN = 3;          // cm outside the region before a trajectory is stopped

  const fmtE = (eV) => (eV >= 1e6 ? (eV / 1e6).toFixed(2) + " MeV" : eV >= 1e3 ? (eV / 1e3).toFixed(eV >= 1e4 ? 1 : 2) + " keV" : eV.toFixed(eV >= 100 ? 0 : 1) + " eV");
  const fmtB = (T) => (T >= 1e-1 ? T.toFixed(2) + " T" : (T * 1e3).toFixed(T >= 1e-2 ? 1 : T >= 1e-3 ? 2 : 3) + " mT");
  const deg = (r) => (r * 180) / Math.PI;
  const wrap = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };

  App.register({
    id: "field-window",
    category: "classical",
    group: "Electromagnetism",
    order: 22.5,
    title: "Electron in a Magnetic Field with a Field-Free Window",
    icon: "🔲",
    subtitle: "An electron enters a rectangular region of uniform magnetic field $\\mathbf B = B\\hat{\\mathbf z}$ that has a field-free rectangular hole in its centre. In the field it moves on circular arcs, in the hole on straight lines — choose the field, the particle, its energy and its entry angle and watch where it comes out.",
    notes: [{
      type: "info",
      html: "The motion is in the $xy$-plane; the field points along $z$ (⊙ out of the screen, ⊗ into it). The shaded frame is the field region, the dark rectangle in the middle is field-free. " +
        "The dashed line is the complete trajectory computed instantly when you change a parameter; the particle then travels along it. " +
        "The <b>red arrow</b> is the magnetic force $\\mathbf F = q\\,\\mathbf v\\times\\mathbf B$, the <b>teal arrow</b> the velocity.",
    }],
    animated: true,
    speed: { min: 0.1, max: 5, value: 1, step: 0.1 },
    controls: [
      { type: "section", label: "Particle & force" },
      { id: "part", type: "select", label: "Particle", value: "electron",
        options: Object.keys(PARTICLES).map((k) => ({ value: k, label: PARTICLES[k].label })) },
      { id: "qc", type: "slider", label: "Charge $q$ (in units of $e$)", min: -3, max: 3, step: 0.5, value: -1,
        visibleIf: (p) => p.part === "custom", help: "Negative = electron-like. $q = 0$ is replaced by a tiny value (the particle then flies straight)." },
      { id: "mc", type: "slider", label: "Mass $m$ (log scale, in units of $m_e$)", min: 0, max: 4, step: 0.05, value: 0,
        fmt: (v) => Math.pow(10, v).toPrecision(3) + " mₑ", visibleIf: (p) => p.part === "custom" },
      { id: "lgB", type: "slider", label: "Magnetic field strength $|\\mathbf B|$", min: -4, max: -1, step: 0.01, value: -3,
        fmt: (v) => fmtB(Math.pow(10, v)), help: "Logarithmic slider, 0.1 mT … 100 mT." },
      { id: "Bdir", type: "select", label: "Field direction", value: "1",
        options: [{ value: "1", label: "+z (out of the screen ⊙)" }, { value: "-1", label: "−z (into the screen ⊗)" }] },
      { id: "lgK", type: "slider", label: "Kinetic energy $K$", min: 1, max: 6, step: 0.01, value: Math.log10(2000),
        fmt: (v) => fmtE(Math.pow(10, v)), help: "Logarithmic slider, 10 eV … 1 MeV. The motion is treated relativistically." },
      { type: "section", label: "Entry into the field" },
      { id: "theta", type: "slider", label: "Entry angle $\\theta_{in}$ (from the $+x$ axis)", min: -80, max: 80, step: 0.5, value: 0, unit: "°" },
      { id: "y0", type: "slider", label: "Entry height $y_0$", min: -10, max: 10, step: 0.1, value: -3, unit: "cm",
        help: "The particle starts 2 cm to the left of the field region." },
      { type: "section", label: "Field region (cm)" },
      { id: "W", type: "slider", label: "Region width $W$", min: 4, max: 40, step: 0.5, value: 20, unit: "cm" },
      { id: "H", type: "slider", label: "Region height $H$", min: 4, max: 30, step: 0.5, value: 14, unit: "cm" },
      { id: "w", type: "slider", label: "Hole width $w$", min: 0, max: 36, step: 0.5, value: 8, unit: "cm",
        help: "Set to 0 for a region without a hole." },
      { id: "h", type: "slider", label: "Hole height $h$", min: 0, max: 26, step: 0.5, value: 6, unit: "cm" },
      { type: "section", label: "Display" },
      { id: "fan", type: "slider", label: "Extra trajectories (angle fan)", min: 0, max: 12, step: 1, value: 0, live: true,
        help: "Faint paths at entry angles spread ±15° around $\\theta_{in}$." },
      { id: "ghost", type: "checkbox", label: "Compare with the same region <i>without</i> a hole", value: true, live: true },
      { id: "circ", type: "checkbox", label: "Show the full circle of the current arc", value: true, live: true },
      { id: "vec", type: "checkbox", label: "Show velocity and force arrows", value: true, live: true },
    ],

    theory: () => `
      <h4>The physical system</h4>
      <p>A single charged particle — by default an electron ($q=-e$, $m=m_e$) — moves in the $xy$-plane. A static, uniform magnetic
      field $\\mathbf B = B_z\\hat{\\mathbf z}$ fills a rectangle of width $W$ and height $H$ centred on the origin, <b>except</b> for a
      smaller centred rectangle (width $w$, height $h$) in which the field is exactly zero — a field-free "window". Outside the outer rectangle
      the field is also zero. The particle starts $2\\,\\text{cm}$ to the left of the region at height $y_0$, with kinetic energy $K$ and
      direction $\\theta_{in}$ measured from the $+x$ axis.</p>
      <ul>
        <li>The field edges are ideal (no fringe fields) and there is no electric field, no gravity, no radiation (radiated power is
        negligible at these energies) and no interaction with matter — the particle moves in vacuum.</li>
        <li>SI units throughout: $B$ in tesla (slider in mT), lengths in cm, energy in eV, time in ns.</li>
        <li>The motion is treated <b>relativistically</b>, so energies up to 1 MeV (where an electron moves at $0.94\\,c$) are correct.</li>
      </ul>

      <h4>Equations being solved</h4>
      <p>The only force is the magnetic part of the Lorentz force. With the relativistic momentum $\\mathbf p = \\gamma m\\mathbf v$:</p>
      $$\\frac{d\\mathbf p}{dt} = q\\,\\mathbf v\\times\\mathbf B(\\mathbf r),\\qquad \\frac{d\\mathbf r}{dt} = \\mathbf v,\\qquad
        \\mathbf B(\\mathbf r)=\\begin{cases}B_z\\hat{\\mathbf z} & \\mathbf r \\text{ in the frame}\\\\ \\mathbf 0 & \\text{in the hole or outside}\\end{cases}$$
      <p>Because $\\mathbf F\\perp\\mathbf v$, the magnetic force does no work: $|\\mathbf v|$, $\\gamma$ and $K$ never change. Written in
      components ($B=B_z$):</p>
      $$\\dot v_x = \\frac{qB}{\\gamma m}\\,v_y,\\qquad \\dot v_y = -\\frac{qB}{\\gamma m}\\,v_x$$
      <p>so inside the field the velocity simply rotates at the constant <b>cyclotron angular frequency</b> and the path is a circular arc of
      the <b>Larmor (gyro) radius</b>:</p>
      <div class="callout">$$\\omega_c = \\frac{|q|B}{\\gamma m},\\qquad r_L=\\frac{p}{|q|B}=\\frac{\\gamma m v}{|q|B},\\qquad T_c=\\frac{2\\pi}{\\omega_c},
        \\qquad p=\\frac{\\sqrt{K^2+2Kmc^2}}{c}$$</div>
      <p>The sense of rotation follows from $\\mathbf F = q\\mathbf v\\times\\mathbf B$: for an <b>electron</b> in a field pointing <b>out</b>
      of the screen the path curves <b>counter-clockwise</b> (to the left); a positive charge, or reversing $\\mathbf B$, curves it clockwise.
      In the field-free window $\\mathbf F=0$, so the particle crosses it on a <b>straight line</b> — the hole "pauses" the rotation.</p>

      <h5>An exact prediction for the exit angle</h5>
      <p>The $y$-component of the equation of motion can be integrated exactly, whatever the shape of the field region:</p>
      $$\\frac{dp_y}{dt} = -qB_z(\\mathbf r)\\,v_x \\;\\Longrightarrow\\; \\Delta p_y = -qB_z\\,\\Delta x_{\\text{field}},
        \\qquad \\Delta x_{\\text{field}}=\\int_{\\text{in field}} v_x\\,dt$$
      <p>where $\\Delta x_{\\text{field}}$ is the net $x$-distance travelled <i>while inside the field</i> (hole segments do not count). Since
      $p_y = p\\sin\\theta$ and $p$ is constant,</p>
      <div class="callout">$$\\sin\\theta_{out} = \\sin\\theta_{in} - \\frac{qB_z}{p}\\,\\Delta x_{\\text{field}}
        \\;=\\; \\sin\\theta_{in} + \\frac{\\Delta x_{\\text{field}}}{r_L}\\quad(\\text{electron},\\;B_z>0)$$</div>
      <p>(similarly $\\Delta p_x = qB_z\\,\\Delta y_{\\text{field}}$). For a region <i>without</i> a hole, crossed from left to right,
      $\\Delta x_{\\text{field}} = W$, giving the textbook result $\\sin\\theta_{out}=\\sin\\theta_{in}+W/r_L$: the particle is transmitted if
      the right-hand side stays below 1, otherwise it is turned back. A hole crossed horizontally removes its width from
      $\\Delta x_{\\text{field}}$, so the same particle is deflected <b>less</b> — the metrics panel shows this prediction next to the
      simulated exit angle.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Integrator:</b> an exact-rotation leapfrog (the Boris scheme with the exact rotation angle). Each step of length $\\Delta t$:
        drift half a step $\\mathbf r \\to \\mathbf r + \\mathbf v\\,\\Delta t/2$; look up $B$ at that midpoint; rotate $\\mathbf v$ by the angle
        $-\\frac{qB}{\\gamma m}\\Delta t$ about $z$; drift the second half step. A rotation cannot change $|\\mathbf v|$, so kinetic energy is
        conserved to machine precision, and the scheme is second-order accurate in $\\Delta t$.</li>
        <li><b>Step size:</b> the path length per step is $\\Delta s = v\\Delta t \\le \\min(r_L, W, H, w, h)/400$ (at least $10\\,\\mu\\text{m}$),
        so arcs are resolved by hundreds of steps and the field edges are located to better than $0.25\\,\\%$ of the smallest length.</li>
        <li><b>Termination:</b> the trajectory is integrated until the particle is $3\\,\\text{cm}$ outside the region (it then moves on a
        straight line forever), or until $4\\times10^{5}$ steps (reported as "still circling").</li>
        <li>The whole path is computed instantly whenever a parameter changes and drawn dashed; the animation then moves the particle
        along it (playback speed in cm of path per second; the toolbar shows the true physical time in ns).</li>
        <li><b>Metrics:</b> $r_L$, $T_c$, $v/c$ from the formulas above; time in field and $\\Delta x_{\\text{field}}$ are summed during the
        integration; $\\theta_{out}$ is the direction of the final velocity; the "prediction" is the boxed formula evaluated with the
        simulated $\\Delta x_{\\text{field}}$; the angle-fan plot repeats the whole calculation for 121 entry angles.</li>
        <li><b>Accuracy checks:</b> $|\\mathbf v|/v_0-1$ (should be $\\sim10^{-15}$) and $\\sin\\theta_{out}$ simulated vs predicted.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Hole off vs on.</b> With the defaults (2 keV electron, 1 mT, $y_0=-3\\,\\text{cm}$) the electron crosses the window and is transmitted at about $53^\\circ$: only $\\Delta x_{\\text{field}}\\approx 12\\,\\text{cm}$ of the $20\\,\\text{cm}$ is spent in the field and $12/r_L\\approx0.8\\lt 1$. Without the hole ($w=0$) the full $20\\,\\text{cm}$ would require $\\sin\\theta_{out}=20/15.1\\gt 1$ — impossible, so it is turned back or leaves through the top. The dashed grey "no hole" ghost path shows the difference.</li>
        <li><b>Critical field.</b> With $\\theta_{in}=0$, $y_0$ outside the hole's height and no hole in the way, the particle just fails
        to cross when $r_L = W$. Tune $B$ until the exit angle approaches $90^\\circ$: at that point
        $B_{crit}=p/(|q|W)$. For a 2 keV electron and $W=20\\,\\text{cm}$, $B_{crit}\\approx 0.75\\,\\text{mT}$.</li>
        <li><b>Flip the field or the charge.</b> Switch $\\mathbf B$ to $-z$ or choose a positron: the trajectory is mirrored about the
        entry direction.</li>
        <li><b>Mass matters.</b> Switch to a proton at the same energy: $r_L\\propto\\sqrt{m}$ (non-relativistically), so it is about
        $43\\times$ stiffer and barely bends — you need ~40× more field for the same path.</li>
        <li><b>Energy scan.</b> Raise $K$ towards 1 MeV and watch $v/c\\to1$ while $r_L$ keeps growing as $p$, not $v$: the relativistic
        momentum, not the speed, sets the radius.</li>
        <li><b>Angle fan.</b> Add extra trajectories: some pass through the window, some only through the field — the lower plot shows
        the exit angle for every entry angle, with jumps where the path starts or stops crossing the hole.</li>
      </ol>

      <h4>Limitations & further reading</h4>
      <p>Real magnets have fringe fields near their edges, which smear out the sharp corners of the trajectory; the hole would in practice
      need magnetic shielding (e.g. mu-metal) to be truly field-free. Electric fields, space charge and collisions with residual gas
      are ignored. See D. J. Griffiths, <i>Introduction to Electrodynamics</i>, §5.1 (Lorentz force, cyclotron motion);
      E. M. Purcell & D. J. Morin, <i>Electricity and Magnetism</i>, Ch. 6; J. D. Jackson, <i>Classical Electrodynamics</i>, §12.2
      (relativistic motion in a uniform magnetic field); and C. K. Birdsall & A. B. Langdon, <i>Plasma Physics via Computer Simulation</i>
      (the Boris integrator).</p>`,

    mount(api) {
      const P = api.params;
      const plots = api.plots([
        { id: "xy", title: "Motion in the $xy$-plane — field region (shaded), field-free window (dark), trajectory (dashed)", span: 2,
          aspect: 0.56, equal: true, xlim: [-14, 14], ylim: [-10, 10], xlabel: "x (cm)", ylabel: "y (cm)", maxHeight: 640 },
        { id: "ang", title: "Direction of motion $\\varphi(t)$ — rotates in the field, constant in the window", aspect: 0.55,
          xlim: [0, 1], ylim: [-180, 180], xlabel: "t (ns)", ylabel: "φ (°)" },
        { id: "fanp", title: "Exit angle vs entry angle (121 trajectories)", aspect: 0.55,
          xlim: [-80, 80], ylim: [-180, 180], xlabel: "θ_in (°)", ylabel: "θ_out (°)" },
      ]);
      const M = api.metrics([
        { id: "rl", label: "Larmor radius $r_L$" },
        { id: "tc", label: "Cyclotron period $T_c$" },
        { id: "v", label: "Speed $v/c$" },
        { id: "out", label: "Outcome" },
        { id: "th", label: "Exit angle $\\theta_{out}$ / deflection" },
        { id: "pred", label: "$\\sin\\theta_{out}$: simulated / predicted" },
        { id: "tf", label: "Time in field / in window" },
        { id: "chk", label: "Speed drift $|v/v_0-1|$" },
      ]);

      let st = null;        // current physical setup
      let main = null;      // main trajectory
      let ghost = null;     // same, without hole
      let fanPaths = [];    // extra trajectories
      let sweep = null;     // exit-angle sweep
      let sPlay = 0;        // path length travelled in the animation [m]
      let finished = false;

      // -------------------------------------------------------------- physics setup
      function setup() {
        let qE, mMe;
        if (P.part === "custom") { qE = P.qc === 0 ? 1e-6 : P.qc; mMe = Math.pow(10, P.mc); }
        else { qE = PARTICLES[P.part].q; mMe = PARTICLES[P.part].m; }
        const q = qE * E_CH, m = mMe * ME;
        const K = Math.pow(10, P.lgK);                 // eV
        const mc2 = mMe * MEC2_EV;                     // eV
        const gamma = 1 + K / mc2;
        const beta = Math.sqrt(1 - 1 / (gamma * gamma));
        const v = beta * C;
        const p = gamma * m * v;
        const Bz = Math.pow(10, P.lgB) * (P.Bdir === "-1" ? -1 : 1);
        const W = P.W * CM, H = P.H * CM;
        // the hole must fit inside the frame
        const w = Math.min(P.w, P.W - 0.5) * CM, h = Math.min(P.h, P.H - 0.5) * CM;
        const rL = p / (Math.abs(q) * Math.abs(Bz));
        const Om = -(q * Bz) / (gamma * m);            // signed rotation rate of v (counter-clockwise > 0)
        return { q, m, qE, mMe, K, gamma, beta, v, p, Bz, W, H, w, h, rL, Om, Tc: (2 * Math.PI) / Math.abs(Om) };
      }

      const inField = (s, x, y, hole) => {
        if (Math.abs(x) > s.W / 2 || Math.abs(y) > s.H / 2) return false;
        if (hole && s.w > 0 && s.h > 0 && Math.abs(x) < s.w / 2 && Math.abs(y) < s.h / 2) return false;
        return true;
      };

      /** Integrate one trajectory. Returns path samples and diagnostics. */
      function trace(s, thetaDeg, y0, hole, coarse) {
        const th = (thetaDeg * Math.PI) / 180;
        let x = -s.W / 2 - 2 * CM, y = y0 * CM, vx = s.v * Math.cos(th), vy = s.v * Math.sin(th);
        const scales = [s.rL, s.W, s.H];
        if (hole && s.w > 0) scales.push(s.w);
        if (hole && s.h > 0) scales.push(s.h);
        const ds = Math.max(Math.min(...scales) / (coarse ? 150 : 400), 1e-5);
        const dt = ds / s.v;
        const xmax = s.W / 2 + MARGIN * CM, ymax = s.H / 2 + MARGIN * CM;
        const xs = [x], ys = [y], ts = [0], ss = [0], fs = [0], phis = [deg(Math.atan2(vy, vx))];
        let t = 0, sPath = 0, tField = 0, tHole = 0, dxField = 0, entered = false, n = 0, everIn = false;
        const stride = coarse ? Infinity : 1;
        let keepEvery = 1;
        const cap = coarse ? 60000 : MAX_STEPS;
        while (n < cap) {
          // half drift
          let xm = x + vx * dt / 2, ym = y + vy * dt / 2;
          const f = inField(s, xm, ym, hole);
          if (f) {
            const a = s.Om * dt, ca = Math.cos(a), sa = Math.sin(a);
            const nvx = vx * ca - vy * sa, nvy = vx * sa + vy * ca;
            dxField += 0.5 * (vx + nvx) * dt; // exact-enough average of v_x over the rotation step
            vx = nvx; vy = nvy;
            tField += dt; everIn = true;
          } else if (Math.abs(xm) < s.W / 2 && Math.abs(ym) < s.H / 2) {
            tHole += dt;
          }
          x = xm + vx * dt / 2; y = ym + vy * dt / 2;
          t += dt; sPath += ds; n++;
          if (Math.abs(x) <= s.W / 2 && Math.abs(y) <= s.H / 2) entered = true;
          if (stride !== Infinity && n % keepEvery === 0) {
            xs.push(x); ys.push(y); ts.push(t); ss.push(sPath); fs.push(f ? 1 : 0); phis.push(deg(Math.atan2(vy, vx)));
            if (xs.length > MAX_PTS) { // thin out: keep every second point
              const thin = (a) => a.filter((_, k) => k % 2 === 0);
              const nx = thin(xs), ny = thin(ys), nt = thin(ts), nss = thin(ss), nf = thin(fs), np = thin(phis);
              xs.length = 0; xs.push(...nx); ys.length = 0; ys.push(...ny); ts.length = 0; ts.push(...nt);
              ss.length = 0; ss.push(...nss); fs.length = 0; fs.push(...nf); phis.length = 0; phis.push(...np);
              keepEvery *= 2;
            }
          }
          if (Math.abs(x) > xmax || Math.abs(y) > ymax) break;
          if (!entered && n > 4 * (s.W + 4 * CM) / ds) break; // aimed away and never reached the region
        }
        if (stride !== Infinity && xs[xs.length - 1] !== x) { xs.push(x); ys.push(y); ts.push(t); ss.push(sPath); fs.push(0); phis.push(deg(Math.atan2(vy, vx))); }
        let outcome;
        if (n >= cap) outcome = "still circling (step limit)";
        else if (!everIn && !entered) outcome = "missed the region";
        else if (x > s.W / 2) outcome = "transmitted → right";
        else if (x < -s.W / 2) outcome = everIn ? "turned back ← left" : "passed by";
        else outcome = y > 0 ? "exits through the top ↑" : "exits through the bottom ↓";
        const thOut = Math.atan2(vy, vx);
        return { xs, ys, ts, ss, fs, phis, n, t, sPath, tField, tHole, dxField, thOut, th, outcome, trapped: n >= cap,
          vEnd: Math.hypot(vx, vy) };
      }

      function computeSweep(s) {
        const th = [], out = [], col = [];
        for (let k = 0; k <= 120; k++) {
          const a = -80 + (160 * k) / 120;
          const r = trace(s, a, P.y0, true, true);
          th.push(a); out.push(r.trapped ? NaN : deg(r.thOut));
          col.push(r.outcome.startsWith("transmitted") ? PlotColors.accent : r.outcome.startsWith("turned") ? PlotColors.accent3 : PlotColors.accent2);
        }
        return { th, out, col };
      }

      function frameLimits(s) {
        const W = s.W / CM, H = s.H / CM;
        let x0 = -W / 2 - MARGIN - 0.5, x1 = W / 2 + MARGIN + 0.5, y0 = -H / 2 - MARGIN - 0.5, y1 = H / 2 + MARGIN + 0.5;
        plots.xy.setLimits([x0, x1], [y0, y1]);
      }

      // -------------------------------------------------------------- drawing helpers
      function drawRegion(p, s) {
        const W = s.W / CM, H = s.H / CM, w = s.w / CM, h = s.h / CM;
        const fieldCol = s.Bz > 0 ? "#2c5f8a" : "#7a3b5c";
        p.rect(-W / 2, -H / 2, W / 2, H / 2, { color: fieldCol, alpha: 0.32, stroke: fieldCol, strokeWidth: 1.5 });
        if (w > 0 && h > 0) p.rect(-w / 2, -h / 2, w / 2, h / 2, { color: "#0f151c", alpha: 1, stroke: "#5f6e80", strokeWidth: 1.2, dash: [5, 4] });
        // ⊙ / ⊗ symbols on a grid inside the field region
        const step = Math.max(W, H) / 12;
        p.custom((c) => {
          c.strokeStyle = "rgba(200,220,240,0.55)"; c.fillStyle = "rgba(200,220,240,0.55)"; c.lineWidth = 1;
          const R = 4;
          for (let gx = -W / 2 + step / 2; gx < W / 2; gx += step) {
            for (let gy = -H / 2 + step / 2; gy < H / 2; gy += step) {
              if (w > 0 && h > 0 && Math.abs(gx) < w / 2 + 0.3 && Math.abs(gy) < h / 2 + 0.3) continue;
              const X = p.X(gx), Y = p.Y(gy);
              c.beginPath(); c.arc(X, Y, R, 0, 2 * Math.PI); c.stroke();
              if (s.Bz > 0) { c.beginPath(); c.arc(X, Y, 1.4, 0, 2 * Math.PI); c.fill(); }
              else { c.beginPath(); c.moveTo(X - 2.6, Y - 2.6); c.lineTo(X + 2.6, Y + 2.6); c.moveTo(X + 2.6, Y - 2.6); c.lineTo(X - 2.6, Y + 2.6); c.stroke(); }
            }
          }
        });
        if (w > 0 && h > 0) p.text(0, 0, "B = 0", { align: "center", color: PlotColors.muted, size: 12 });
        p.text(-W / 2 + 0.3, H / 2 - 0.6, `B = ${fmtB(Math.abs(s.Bz))} ${s.Bz > 0 ? "⊙" : "⊗"}`, { color: "#cfe3f5", size: 12 });
      }
      const toCm = (a) => a.map((v) => v / CM);

      // -------------------------------------------------------------- simulation hooks
      return {
        reset() {
          // keep the dependent sliders consistent (hole smaller than frame, entry height inside view)
          if (P.w > P.W - 0.5) api.setControl("w", { value: Math.max(0, P.W - 0.5) });
          if (P.h > P.H - 0.5) api.setControl("h", { value: Math.max(0, P.H - 0.5) });
          api.setControl("w", { max: Math.max(0.5, P.W - 0.5) });
          api.setControl("h", { max: Math.max(0.5, P.H - 0.5) });
          const ylim = P.H / 2 + MARGIN - 0.5;
          api.setControl("y0", { min: -ylim, max: ylim, value: PM.clamp(P.y0, -ylim, ylim) });

          st = setup();
          main = trace(st, P.theta, P.y0, true, false);
          ghost = trace(st, P.theta, P.y0, false, false);
          fanPaths = [];
          sweep = computeSweep(st);
          frameLimits(st);
          // y-range of the direction plot: the actual range of φ (with padding), within ±180°
          let pmin = Math.min(...main.phis), pmax = Math.max(...main.phis);
          const pad = Math.max(10, 0.12 * (pmax - pmin));
          plots.ang.setLimits([0, Math.max(main.t * 1e9, 1e-6)], [Math.max(-185, pmin - pad), Math.min(185, pmax + pad)]);
          sPlay = 0; finished = false;
          this.rebuildFan();
        },
        rebuildFan() {
          fanPaths = [];
          const n = P.fan | 0;
          for (let k = 0; k < n; k++) {
            const a = P.theta + (n === 1 ? 15 : -15 + (30 * k) / (n - 1));
            if (Math.abs(a) <= 89) fanPaths.push(trace(st, a, P.y0, true, false));
          }
        },
        onParam(id) { if (id === "fan" && st) this.rebuildFan(); },
        step(dt) {
          if (!main) return;
          if (finished) { sPlay = 0; finished = false; } // Play pressed after the end: run again
          // playback: 6 cm of path per second at speed ×1
          sPlay += 0.06 * dt;
          if (sPlay >= main.sPath) { sPlay = main.sPath; finished = true; api.pause(); }
        },
        render() {
          if (!main) return;
          const s = st, p = plots.xy;
          p.clear();
          drawRegion(p, s);

          // fan + ghost + full path
          fanPaths.forEach((f) => p.line(toCm(f.xs), toCm(f.ys), { color: PlotColors.accent2, width: 1, alpha: 0.35 }));
          if (P.ghost && s.w > 0 && s.h > 0) p.line(toCm(ghost.xs), toCm(ghost.ys), { color: PlotColors.muted, width: 1.3, dash: [3, 4], alpha: 0.75 });
          p.line(toCm(main.xs), toCm(main.ys), { color: PlotColors.accent, width: 1.2, dash: [6, 5], alpha: 0.55 });

          // current position along the path
          const ss = main.ss;
          let k = 0, lo = 0, hi = ss.length - 1;
          while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (ss[mid] <= sPlay) lo = mid; else hi = mid; }
          k = lo;
          const f = ss[hi] > ss[lo] ? (sPlay - ss[lo]) / (ss[hi] - ss[lo]) : 0;
          const cx = (main.xs[lo] + f * (main.xs[hi] - main.xs[lo])) / CM, cy = (main.ys[lo] + f * (main.ys[hi] - main.ys[lo])) / CM;
          const tNow = main.ts[lo] + f * (main.ts[hi] - main.ts[lo]);
          const phi = (Math.atan2(main.ys[hi] - main.ys[lo], main.xs[hi] - main.xs[lo]));
          const inF = inField(s, cx * CM, cy * CM, true);

          // travelled part, solid
          p.line(toCm(main.xs.slice(0, k + 1)).concat([cx]), toCm(main.ys.slice(0, k + 1)).concat([cy]), { color: PlotColors.accent, width: 2.4 });

          // full circle of the current arc
          if (P.circ && inF) {
            const sg = Math.sign(s.Om); // +1 counter-clockwise
            const rc = s.rL / CM;
            const ccx = cx - sg * rc * Math.sin(phi), ccy = cy + sg * rc * Math.cos(phi);
            p.circle(ccx, ccy, rc, { fill: false, stroke: "rgba(245,158,11,0.35)", strokeWidth: 1 });
            p.circle(ccx, ccy, 2.5, { px: true, color: "rgba(245,158,11,0.6)" });
          }
          // particle + arrows
          const span = (p.visibleXlim[1] - p.visibleXlim[0]) * 0.07;
          if (P.vec) {
            p.arrow(cx, cy, cx + span * Math.cos(phi), cy + span * Math.sin(phi), { color: PlotColors.accent, width: 2.2 });
            if (inF) {
              // F = q v × B is perpendicular to v, towards the centre of curvature
              const sg = Math.sign(s.Om);
              p.arrow(cx, cy, cx - sg * span * Math.sin(phi), cy + sg * span * Math.cos(phi), { color: PlotColors.bad, width: 2.2 });
            }
          }
          p.circle(cx, cy, 6, { px: true, color: s.qE < 0 ? "#79c0ff" : "#ff7b72", stroke: "#0f151c", strokeWidth: 1.5 });
          p.text(cx, cy, s.qE < 0 ? "−" : "+", { align: "center", color: "#0f151c", bold: true, size: 11 });
          // entry marker
          p.arrow(-s.W / 2 / CM - 2.8, P.y0 - 0.8 * Math.tan((P.theta * Math.PI) / 180), -s.W / 2 / CM - 2, P.y0, { color: PlotColors.muted, width: 1.2 });
          const lg = [{ label: "trajectory", color: PlotColors.accent, dash: [6, 5] }];
          if (P.ghost && s.w > 0 && s.h > 0) lg.push({ label: "same, no hole", color: PlotColors.muted, dash: [3, 4] });
          if (P.vec) { lg.push({ label: "velocity v", color: PlotColors.accent }); lg.push({ label: "force qv×B", color: PlotColors.bad }); }
          if (fanPaths.length) lg.push({ label: "angle fan", color: PlotColors.accent2 });
          p.legend(lg, "tr");
          p.label([`t = ${(tNow * 1e9).toFixed(2)} ns`, inF ? "in the field: circular arc" : "B = 0: straight line"], "tl");

          // direction-of-motion plot
          const pa = plots.ang;
          pa.clear();
          const tns = main.ts.map((v) => v * 1e9);
          // shade field intervals
          pa.custom((c) => {
            c.fillStyle = "rgba(44,95,138,0.28)";
            let a = -1;
            for (let i = 0; i < tns.length; i++) {
              if (main.fs[i] && a < 0) a = i;
              if ((!main.fs[i] || i === tns.length - 1) && a >= 0) { const X0 = pa.X(tns[a]), X1 = pa.X(tns[i]); c.fillRect(X0, pa.m.t, Math.max(X1 - X0, 1), pa._v.ph); a = -1; }
            }
          });
          // unwrap-free display: break the line at ±180° jumps
          const ph = main.phis.slice();
          for (let i = 1; i < ph.length; i++) if (Math.abs(ph[i] - ph[i - 1]) > 180) ph[i - 1] = NaN;
          pa.line(tns, ph, { color: PlotColors.accent, width: 1.8 });
          pa.vline(tNow * 1e9, { color: PlotColors.accent3, dash: [4, 3] });
          pa.legend([{ label: "φ(t)", color: PlotColors.accent }, { label: "in the field", color: "#2c5f8a", type: "box" }], "tr");

          // sweep plot
          const pf = plots.fanp;
          pf.clear();
          pf.hline(0, { color: "rgba(139,152,168,0.4)" });
          pf.points(sweep.th, sweep.out, { colors: sweep.col, size: 2.8 });
          pf.circle(P.theta, deg(main.thOut), 6, { px: true, fill: false, stroke: PlotColors.text, strokeWidth: 2 });
          pf.legend([{ label: "transmitted →", color: PlotColors.accent, type: "dot" }, { label: "turned back ←", color: PlotColors.accent3, type: "dot" }, { label: "top / bottom", color: PlotColors.accent2, type: "dot" }], "br");

          // metrics
          const rl = s.rL / CM;
          M.set("rl", rl >= 100 ? (rl / 100).toFixed(2) + " m" : rl.toFixed(2) + " cm");
          M.set("tc", PM.fmt(s.Tc * 1e9, 2) + " ns");
          M.set("v", s.beta.toFixed(4) + "  (" + PM.fmt(s.v, 3) + " m/s)");
          M.set("out", main.outcome);
          const defl = deg(wrap(main.thOut - main.th));
          M.set("th", main.trapped ? "—" : `${deg(main.thOut).toFixed(1)}° / ${defl.toFixed(1)}°`);
          const sinPred = Math.sin(main.th) - (s.q * s.Bz / (s.gamma * s.m * s.v)) * main.dxField;
          M.set("pred", main.trapped ? "—" : `${Math.sin(main.thOut).toFixed(4)} / ${sinPred.toFixed(4)}`);
          M.set("tf", `${(main.tField * 1e9).toFixed(2)} / ${(main.tHole * 1e9).toFixed(2)} ns`);
          M.set("chk", PM.fmt(Math.abs(main.vEnd / s.v - 1), 2));
          api.setTime(`t = ${(tNow * 1e9).toFixed(2)} ns`);
        },
      };
    },
  });
})();
