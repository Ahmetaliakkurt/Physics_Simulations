/* Magnetic fields from steady currents — Biot–Savart law: straight wires (exact superposition) and circular loops
   (numerical Biot–Savart quadrature), |B| map, evenly spaced field lines, compass needles and on-axis checks. */
(function () {
  "use strict";

  const NX = 240, NY = 150;            // field grid (spans the visible plot area, cm)
  const MPHI = 128;                    // Biot–Savart quadrature points around a loop
  const DSL = 0.04;                    // field-line RK4 step (cm)
  const MAXSTEP = 4000;
  const LINE_CAP = 700000, ARW_CAP = 4000, MAXW = 24;
  const MU0 = 4e-7 * Math.PI;

  const WIRE_CFG = ["wire1", "parallel", "anti", "ring", "coax"];
  const isWire = (c) => WIRE_CFG.indexOf(c) >= 0;

  // magma-like colour map (dark → purple → orange → pale yellow)
  const STOPS = [[15, 21, 28], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]];
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (STOPS.length - 1), k = Math.min(Math.floor(t), STOPS.length - 2), f = t - k;
    for (let c = 0; c < 3; c++) LUT[i * 3 + c] = STOPS[k][c] + (STOPS[k + 1][c] - STOPS[k][c]) * f;
  }

  function wirePreset(cfg, I) {
    switch (cfg) {
      case "wire1": return [{ x: 0, y: 0, I }];
      case "parallel": return [{ x: -2, y: 0, I }, { x: 2, y: 0, I }];
      case "anti": return [{ x: -2, y: 0, I }, { x: 2, y: 0, I: -I }];
      case "ring": {
        const a = [];
        for (let k = 0; k < 12; k++) { const th = (2 * Math.PI * k) / 12; a.push({ x: 4 * Math.cos(th), y: 4 * Math.sin(th), I }); }
        return a;
      }
      default: { // coax: centre conductor + 12 return wires
        const a = [{ x: 0, y: 0, I }];
        for (let k = 0; k < 12; k++) { const th = (2 * Math.PI * (k + 0.5)) / 12; a.push({ x: 4.5 * Math.cos(th), y: 4.5 * Math.sin(th), I: -I / 12 }); }
        return a;
      }
    }
  }

  App.register({
    id: "magnetic-field",
    category: "classical",
    group: "Electromagnetism",
    order: 21,
    title: "Magnetic Fields from Currents (Biot–Savart)",
    icon: "🧲",
    subtitle: "The magnetic field of straight wires, a current loop, Helmholtz coils and a solenoid, computed live from the Biot–Savart law: |B| map, field lines, swinging compass needles and quantitative checks against Ampère's law and the textbook formulas.",
    notes: [{
      type: "info",
      html: "Wires run perpendicular to the screen: <b>⊙</b> = current out of the screen, <b>⊗</b> = into the screen. <b>Drag</b> a wire to move it, click it to select it and edit its current. " +
        "For loops and coils the screen is a cut through the symmetry axis (horizontal): each turn appears as a ⊙/⊗ pair. Colour = $|\\mathbf B|$ on a logarithmic scale, " +
        "lines = field lines (moving dots run along $\\mathbf B$), needles = compasses (red tip = north pole, points along $\\mathbf B$).",
    }],
    animated: true,
    speed: { min: 0.1, max: 3, value: 1, step: 0.1 },
    controls: [
      { id: "cfg", type: "select", label: "Current configuration", value: "parallel", rebuild: true,
        options: [
          { value: "wire1", label: "Single straight wire" },
          { value: "parallel", label: "Two parallel wires (same direction)" },
          { value: "anti", label: "Two antiparallel wires" },
          { value: "ring", label: "Ring of 12 wires (cylindrical current sheet)" },
          { value: "coax", label: "Coaxial cable (centre wire + return ring)" },
          { value: "loop", label: "Circular current loop" },
          { value: "helmholtz", label: "Helmholtz coils (two loops)" },
          { value: "solenoid", label: "Finite solenoid" },
        ] },
      { id: "I", type: "slider", label: "Current $I$", min: 1, max: 50, step: 1, value: 10, unit: "A",
        help: "Changing it reloads the configuration." },
      { id: "secW", type: "section", label: "Wires", visibleIf: (p) => isWire(p.cfg) },
      { id: "Isel", type: "slider", label: "Current of the selected wire", min: -50, max: 50, step: 1, value: 10, unit: "A", live: true,
        visibleIf: (p) => isWire(p.cfg), help: "Positive = out of the screen (⊙)." },
      { id: "addOut", type: "button", label: "＋ Add wire ⊙ (out of screen)", visibleIf: (p) => isWire(p.cfg) },
      { id: "addIn", type: "button", label: "＋ Add wire ⊗ (into screen)", visibleIf: (p) => isWire(p.cfg) },
      { id: "remove", type: "button", label: "✕ Remove selected wire", visibleIf: (p) => isWire(p.cfg) },
      { id: "acen", type: "select", label: "Ampère loop centred on", value: "origin", live: true, visibleIf: (p) => isWire(p.cfg),
        options: [{ value: "origin", label: "the origin (0, 0)" }, { value: "sel", label: "the selected wire" }] },
      { id: "RA", type: "slider", label: "Ampère loop radius $r_A$", min: 0.5, max: 9, step: 0.1, value: 3, unit: "cm", live: true, visibleIf: (p) => isWire(p.cfg) },
      { id: "secL", type: "section", label: "Loops and coils", visibleIf: (p) => !isWire(p.cfg) },
      { id: "R", type: "slider", label: "Loop radius $R$", min: 1, max: 5, step: 0.1, value: 3, unit: "cm", visibleIf: (p) => !isWire(p.cfg) },
      { id: "dR", type: "slider", label: "Coil spacing $d/R$", min: 0.3, max: 2, step: 0.05, value: 1, visibleIf: (p) => p.cfg === "helmholtz",
        help: "$d = R$ is the Helmholtz condition (maximally flat field)." },
      { id: "N", type: "slider", label: "Number of turns $N$", min: 2, max: 30, step: 1, value: 16, visibleIf: (p) => p.cfg === "solenoid" },
      { id: "L", type: "slider", label: "Solenoid length $L$", min: 2, max: 16, step: 0.5, value: 10, unit: "cm", visibleIf: (p) => p.cfg === "solenoid" },
      { id: "rc", type: "slider", label: "Off-axis line $\\rho = f\\,R$, $f$", min: 0.1, max: 0.9, step: 0.05, value: 0.5, live: true, visibleIf: (p) => !isWire(p.cfg),
        help: "A second numerical $B_z(z)$ curve is computed along this line parallel to the axis." },
      { type: "section", label: "Display" },
      { id: "showHeat", type: "checkbox", label: "$|\\mathbf B|$ map (log scale)", value: true, live: true },
      { id: "showLines", type: "checkbox", label: "Field lines", value: true, live: true },
      { id: "dsep", type: "slider", label: "Field-line spacing", min: 0.3, max: 1.5, step: 0.05, value: 0.6, unit: "cm", live: true },
      { id: "flow", type: "checkbox", label: "Animate the field direction", value: true, live: true },
      { id: "compass", type: "checkbox", label: "Compass needles", value: true, live: true },
      { id: "force", type: "checkbox", label: "Force-per-length arrows on wires", value: true, live: true, visibleIf: (p) => isWire(p.cfg) },
    ],
    theory: (p) => `
      <h4>The physical system</h4>
      <p>Steady (time-independent) currents in thin conductors in vacuum. Two families are simulated:
      <b>infinitely long straight wires</b> perpendicular to the screen (the screen is the $xy$-plane, current $I_i$ along $\\pm\\hat{\\mathbf z}$;
      positive = out of the screen), and <b>coaxial circular loops</b> of radius $R$ whose common symmetry axis is the horizontal
      axis $z$ of the picture (the screen is the meridional plane $y=0$, which contains the axis; the vertical coordinate is $x$).
      Units are SI: lengths in cm, currents in A, fields in μT ($1\\,\\mu\\text{T}=10^{-6}$ T; the Earth's field is about 50 μT),
      forces per length in mN/m, $\\mu_0=4\\pi\\times10^{-7}$ T·m/A. The conductors are idealised as filaments of zero thickness.
      ${isWire(p.cfg) ? "<i>Current view: straight wires.</i>" : "<i>Current view: circular loops (" + p.cfg + ").</i>"}</p>

      <h4>Equations being solved</h4>
      <p>Magnetostatics: $\\nabla\\cdot\\mathbf B=0$, $\\nabla\\times\\mathbf B=\\mu_0\\mathbf J$. For filamentary currents its solution is the
      Biot–Savart law</p>
      <div class="callout">$$ \\mathbf B(\\mathbf r)=\\frac{\\mu_0}{4\\pi}\\sum_{\\text{wires}} I\\oint\\frac{d\\boldsymbol\\ell'\\times(\\mathbf r-\\mathbf r')}{|\\mathbf r-\\mathbf r'|^{3}} $$</div>
      <p><b>Straight wire</b> (integral done analytically): $\\mathbf B=\\dfrac{\\mu_0 I}{2\\pi r}\\,\\hat{\\boldsymbol\\varphi}$, i.e.
      $B_x=-\\dfrac{\\mu_0I}{2\\pi}\\dfrac{y-y_i}{r^2}$, $B_y=\\dfrac{\\mu_0I}{2\\pi}\\dfrac{x-x_i}{r^2}$, superposed over all wires.
      <b>Ampère's law</b> $\\oint\\mathbf B\\cdot d\\boldsymbol\\ell=\\mu_0 I_{\\text{enc}}$ holds for every closed path. The force per unit length on wire $a$
      is $\\mathbf f_a=I_a\\,\\hat{\\mathbf z}\\times\\mathbf B_{\\text{others}}(\\mathbf r_a)$; for two wires a distance $d$ apart
      $|\\mathbf f|=\\mu_0 I_1I_2/2\\pi d$ — attractive for parallel, repulsive for antiparallel currents.</p>
      <p><b>Circular loop</b> at $z=z_0$: with $\\mathbf r'=(R\\cos\\varphi,R\\sin\\varphi,z_0)$ and a field point $(\\rho,0,z)$ in the screen plane,</p>
      $$ B_\\rho=\\frac{\\mu_0 I R}{4\\pi}\\int_0^{2\\pi}\\frac{(z-z_0)\\cos\\varphi\\,d\\varphi}{D^{3}},\\quad
      B_z=\\frac{\\mu_0 I R}{4\\pi}\\int_0^{2\\pi}\\frac{(R-\\rho\\cos\\varphi)\\,d\\varphi}{D^{3}},\\quad
      D^2=\\rho^2+R^2-2\\rho R\\cos\\varphi+(z-z_0)^2 $$
      <p>($B_y=0$ by symmetry). On the axis this reduces to the textbook results</p>
      $$ B_{\\text{loop}}(z)=\\frac{\\mu_0 I R^2}{2\\left(R^2+z^2\\right)^{3/2}},\\qquad
      B_{\\text{sol}}(z)=\\frac{\\mu_0 nI}{2}\\left[\\frac{z+L/2}{\\sqrt{(z+L/2)^2+R^2}}-\\frac{z-L/2}{\\sqrt{(z-L/2)^2+R^2}}\\right]\\xrightarrow{L\\gg R}\\mu_0 nI , $$
      <p>with $n=N/L$ turns per length. Two coaxial loops a distance $d$ apart have $\\partial_z^2B=0$ at the centre when $d=R$
      (<b>Helmholtz condition</b>); the first non-zero correction is then $\\propto(z/R)^4$.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Grid.</b> $\\mathbf B$ is sampled on a $240\\times150$ grid covering the visible area (spacing ≈ 0.09 cm).</li>
        <li><b>Wires:</b> exact superposition of the analytic wire fields — recomputed whenever a wire is dragged or edited.</li>
        <li><b>Loops:</b> the Biot–Savart integrals above are evaluated numerically with the midpoint rule, $M=128$ points per turn.
        For a smooth periodic integrand this rule converges exponentially fast, except within about one segment length
        ($2\\pi R/128$) of the wire itself. Because every turn is identical up to a shift along $z$, a kernel table
        $B_{\\rho,z}(\\rho,\\Delta z)$ is computed once per radius and all $N$ turns are superposed by linear interpolation in $\\Delta z$.</li>
        <li><b>Colour</b> is $\\log_{10}|\\mathbf B|$ between the 3rd and the 98.5th percentile of the grid values (1–3 decades; the field diverges at the wires).</li>
        <li><b>Field lines</b> are integrated with RK4 along $\\hat{\\mathbf B}=\\mathbf B/|\\mathbf B|$ (step 0.04 cm; bilinear interpolation of the grid
        for loops) in both directions from seed points, using an evenly-spaced-streamline algorithm (Jobard–Lefer type): an occupancy grid
        rejects seeds closer than the chosen spacing to an existing line and stops a line when it approaches another one. A line also
        stops at a conductor, at the edge, or when it closes on itself (closed lines are the rule, since $\\nabla\\cdot\\mathbf B=0$).
        Line <i>spacing</i> is therefore uniform by design; the field strength is shown by the colour.</li>
        <li><b>Compass needles</b> obey $\\ddot\\theta=\\kappa\\,s\\,\\sin(\\theta_B-\\theta)-\\gamma\\dot\\theta$ (damped torque $\\boldsymbol\\mu\\times\\mathbf B$,
        $s$ = local $|\\mathbf B|$ relative to the median, clipped), integrated with semi-implicit Euler.</li>
        <li><b>Checks.</b> Wires: the side plot shows the circle-averaged azimuthal field $\\langle B_\\varphi\\rangle(r)$ (160-point average
        around a circle) against Ampère's prediction $\\mu_0 I_{\\text{enc}}(r)/2\\pi r$, and the metric gives $\\oint\\mathbf B\\cdot d\\boldsymbol\\ell/\\mu_0$
        on the drawn Ampère circle. Loops: numerical $B_z$ on the axis and on an off-axis line versus the analytic formulas.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Single wire:</b> the side plot follows $B=\\mu_0I/2\\pi r$ exactly: at $I=10$ A and $r=1$ cm, $B=200\\ \\mu$T (four Earth fields).</li>
        <li><b>Parallel vs antiparallel:</b> with $I=10$ A and $d=4$ cm the force is $\\mu_0I^2/2\\pi d=0.5$ mN/m; parallel currents attract, antiparallel repel
        (red arrows). Between parallel wires there is a null point where the needles become sluggish.</li>
        <li><b>Ampère's law:</b> drag wires in and out of the Ampère circle — $\\oint\\mathbf B\\cdot d\\boldsymbol\\ell/\\mu_0$ jumps exactly by the current of the wire
        that crosses it, while wires outside do not contribute at all, however close they are.</li>
        <li><b>Shielding:</b> in the coaxial cable the field outside the return ring is (almost) zero because $I_{\\text{enc}}=0$; inside the ring of 12 parallel wires
        the field almost vanishes, outside it looks like one wire carrying $12I$.</li>
        <li><b>Helmholtz:</b> at $d/R=1$ the on-axis curve is flat ($|B(R/4)-B(0)|/B(0)\\approx0.4\\%$); try $d/R=0.5$ (peaked) and $1.5$ (dip in the middle).</li>
        <li><b>Solenoid:</b> increase $N$ and $L/R$: the interior field becomes uniform and approaches $\\mu_0nI$, the field outside becomes weak,
        and at the ends it drops to about one half of the centre value.</li>
      </ol>

      <h4>Limitations & further reading</h4>
      <p>Filamentary conductors (the field diverges at the wire instead of falling to zero inside a real conductor), magnetostatics only (no induction,
      no magnetic materials), and for loops only the meridional cut is drawn — the field is rotationally symmetric about the axis. Each solenoid turn
      is a separate closed loop (no helical pitch). See D. J. Griffiths, <i>Introduction to Electrodynamics</i>, ch. 5; E. M. Purcell &amp; D. J. Morin,
      <i>Electricity and Magnetism</i>, ch. 6; J. D. Jackson, <i>Classical Electrodynamics</i>, §5.5; B. Jobard &amp; W. Lefer, "Creating evenly-spaced streamlines" (1997).</p>`,

    mount(api) {
      const P = api.params;
      const WIRE = isWire(P.cfg);
      const rng = new PM.RNG(77);

      // ---- state
      let wires = [], sel = 0, loops = [];  // loops: z positions (cm)
      let gx0 = 0, gy0 = 0, hx = 1, hy = 1;
      const GU = new Float32Array(NX * NY), GV = new Float32Array(NX * NY), GM = new Float32Array(NX * NY);
      let fieldDirty = true, linesDirty = true, layerDirty = true, flowPhase = 0, bMed = 1;
      const off = document.createElement("canvas"); off.width = NX; off.height = NY;
      const offCtx = off.getContext("2d"), img = offCtx.createImageData(NX, NY);
      const layer = document.createElement("canvas"), layerCtx = layer.getContext("2d");
      const lineBuf = new Float32Array(LINE_CAP), arcBuf = new Float32Array(LINE_CAP / 2);
      const lineOff = new Int32Array(6000), lineLen = new Int32Array(6000); let nLines = 0;
      const arw = new Float32Array(ARW_CAP * 3); let nArw = 0;
      const fwd = new Float32Array(2 * MAXSTEP + 8), bwd = new Float32Array(2 * MAXSTEP + 8);
      let occ = new Int32Array(1), ocW = 1, ocH = 1, ocC = 1;
      // compass needles
      const NCX = 10, NCY = 6;
      const ndX = new Float32Array(NCX * NCY), ndY = new Float32Array(NCX * NCY), ndT = new Float32Array(NCX * NCY), ndW = new Float32Array(NCX * NCY);
      const ndOn = new Uint8Array(NCX * NCY), ndTB = new Float32Array(NCX * NCY), ndS = new Float32Array(NCX * NCY);
      let cursor = null, drag = -1;
      // side-plot buffers
      const NS = 220;
      const sr = new Float64Array(NS), sA = new Float64Array(NS), sAmp = new Float64Array(NS), sRay = new Float64Array(NS);
      const sz = new Float64Array(NS), sNum = new Float64Array(NS), sAna = new Float64Array(NS), sOff = new Float64Array(NS), sSheet = new Float64Array(NS);
      const sNumX = new Float64Array(28), sNumY = new Float64Array(28);
      let sideDirty = true, ampere = { circ: 0, enc: 0 };
      // loop kernel
      let Kr = null, Kz = null, nK = 0, hK = 1, kRows = 0;
      const cosP = new Float64Array(MPHI / 2);
      for (let k = 0; k < MPHI / 2; k++) cosP[k] = Math.cos((2 * Math.PI * (k + 0.5)) / MPHI);
      const dPhi = (2 * Math.PI) / MPHI;

      const metricDefs = WIRE ? [
        { id: "B", label: "$|\\mathbf B|$ at the cursor" },
        { id: "amp", label: "Ampère: $\\oint\\mathbf B\\cdot d\\boldsymbol\\ell/\\mu_0$ | $I_{\\rm enc}$" },
        { id: "F", label: "Force per length on the selected wire" },
        { id: "F2", label: "Two-wire formula $\\mu_0I_1I_2/2\\pi d$" },
        { id: "Itot", label: "Total current $\\sum I$" },
      ] : [
        { id: "B", label: "$|\\mathbf B|$ at the cursor" },
        { id: "B0", label: "$B_z$ at the centre: numerical | analytic" },
        { id: "x1", label: P.cfg === "solenoid" ? "$B(0)\\,/\\,\\mu_0 nI$" : P.cfg === "helmholtz" ? "Flatness $|B(R/4)-B(0)|/B(0)$" : "$B(z{=}R)/B(0)$ (theory $2^{-3/2}=0.354$)" },
        { id: "x2", label: P.cfg === "solenoid" ? "Ideal $\\mu_0 nI$" : "Analytic $B$ at the centre" },
        { id: "m", label: "Magnetic moment $m=NI\\pi R^2$" },
      ];
      const M = api.metrics(metricDefs);
      const plots = api.plots([
        { id: "f", title: WIRE ? "$|\\mathbf B|$ (colour), field lines and compasses in the plane ⟂ to the wires — drag the wires" : "$|\\mathbf B|$ (colour), field lines and compasses in a plane through the coil axis",
          span: 2, aspect: 0.56, xlim: [-10, 10], ylim: [-5.8, 5.8], equal: true, xlabel: WIRE ? "x (cm)" : "z, along the axis (cm)", ylabel: WIRE ? "y (cm)" : "x (cm)", grid: false, maxHeight: 640 },
        WIRE
          ? { id: "s", title: "Ampère check: circle-averaged $\\langle B_\\varphi\\rangle(r)$ vs $\\mu_0 I_{\\rm enc}/2\\pi r$", span: 2, aspect: 0.26, minHeight: 200, maxHeight: 300, xlim: [0, 9.5], ylim: [0, 1], xlabel: "distance r from the Ampère-loop centre (cm)", ylabel: "B (μT)" }
          : { id: "s", title: "$B_z$ along the axis: numerical Biot–Savart vs analytic formula", span: 2, aspect: 0.26, minHeight: 200, maxHeight: 300, xlim: [-10, 10], ylim: [0, 1], xlabel: "z (cm)", ylabel: "B_z (μT)" },
      ]);
      const pl = plots.f, cv = pl.canvas;
      const prevResize = pl.onResize;
      pl.onResize = () => { fieldDirty = true; layerDirty = true; if (prevResize) prevResize(); };
      cv.style.touchAction = "none";
      cv.style.cursor = WIRE ? "crosshair" : "default";

      // ------------------------------------------------------------ configuration
      function loadConfig() {
        if (WIRE) {
          wires = wirePreset(P.cfg, P.I);
          sel = 0;
          api.setControl("Isel", { value: wires[0].I });
        } else {
          const R = P.R;
          if (P.cfg === "loop") loops = [0];
          else if (P.cfg === "helmholtz") loops = [-0.5 * P.dR * R, 0.5 * P.dR * R];
          else { loops = []; for (let k = 0; k < P.N; k++) loops.push(-P.L / 2 + (P.L * (k + 0.5)) / P.N); }
          Kr = null;
        }
        fieldDirty = true; sideDirty = true; layerDirty = true;
        for (let k = 0; k < ndT.length; k++) { ndT[k] = rng.uniform(-Math.PI, Math.PI); ndW[k] = 0; }
      }

      // ------------------------------------------------------------ fields
      let fu = 0, fv = 0;
      function wireB(x, y) { // μT
        let bx = 0, by = 0;
        for (let k = 0; k < wires.length; k++) {
          const w = wires[k], dx = x - w.x, dy = y - w.y, r2 = dx * dx + dy * dy + 1e-9, f = (20 * w.I) / r2;
          bx -= f * dy; by += f * dx;
        }
        fu = bx; fv = by;
      }
      function gridB(u, v) {
        let fi = (u - gx0) / hx, fj = (v - gy0) / hy;
        if (fi < 0) fi = 0; if (fj < 0) fj = 0; if (fi > NX - 1.001) fi = NX - 1.001; if (fj > NY - 1.001) fj = NY - 1.001;
        const i = fi | 0, j = fj | 0, a = fi - i, b = fj - j, p = j * NX + i;
        fu = (1 - b) * ((1 - a) * GU[p] + a * GU[p + 1]) + b * ((1 - a) * GU[p + NX] + a * GU[p + NX + 1]);
        fv = (1 - b) * ((1 - a) * GV[p] + a * GV[p + 1]) + b * ((1 - a) * GV[p + NX] + a * GV[p + NX + 1]);
      }
      const fieldAt = WIRE ? wireB : gridB;

      /** Biot–Savart for one loop (radius R, unit current) at (rho, dz): returns S (cm⁻¹); B(μT) = 10·I·S. */
      let kSr = 0, kSz = 0;
      function loopKernel(rho, dz, R) {
        let sr = 0, szz = 0;
        const a = rho * rho + R * R + dz * dz, b = 2 * rho * R;
        for (let k = 0; k < MPHI / 2; k++) {
          const c = cosP[k], D2 = a - b * c, iD3 = 1 / (D2 * Math.sqrt(D2));
          sr += c * iD3; szz += (R - rho * c) * iD3;
        }
        kSr = 2 * R * dz * sr * dPhi; kSz = 2 * R * szz * dPhi;
      }
      function computeKernel() {
        const R = P.R;
        kRows = NY / 2;
        const zmax = Math.max(...loops.map(Math.abs));
        const [x0, x1] = pl.visibleXlim;
        const Dmax = Math.max(Math.abs(x0), Math.abs(x1)) + zmax + 2 * hx;
        hK = hx / 2; nK = Math.ceil(Dmax / hK) + 2;
        Kr = new Float32Array(kRows * nK); Kz = new Float32Array(kRows * nK);
        for (let jr = 0; jr < kRows; jr++) {
          const rho = gy0 + (kRows + jr) * hy; // upper-half row (rho > 0)
          for (let m = 0; m < nK; m++) {
            loopKernel(rho, m * hK, R);
            Kr[jr * nK + m] = kSr; Kz[jr * nK + m] = kSz;
          }
        }
      }
      function computeGrid() {
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        gx0 = x0; gy0 = y0; hx = (x1 - x0) / (NX - 1); hy = (y1 - y0) / (NY - 1);
        if (WIRE) {
          for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
            wireB(gx0 + i * hx, gy0 + j * hy);
            const p = j * NX + i; GU[p] = fu; GV[p] = fv;
          }
        } else {
          computeKernel();
          GU.fill(0); GV.fill(0);
          const s = 10 * P.I;
          for (const z0 of loops) {
            for (let i = 0; i < NX; i++) {
              const dz = gx0 + i * hx - z0, sg = dz < 0 ? -1 : 1, a = Math.abs(dz) / hK, m = Math.min(a | 0, nK - 2), f = a - m;
              for (let jr = 0; jr < kRows; jr++) {
                const q = jr * nK + m;
                const br = sg * ((1 - f) * Kr[q] + f * Kr[q + 1]) * s, bz = ((1 - f) * Kz[q] + f * Kz[q + 1]) * s;
                const pu = (kRows + jr) * NX + i, pd = (kRows - 1 - jr) * NX + i;
                GU[pu] += bz; GV[pu] += br;
                GU[pd] += bz; GV[pd] -= br;
              }
            }
          }
        }
        // magnitude + colour
        const tmp = new Float32Array(NX * NY);
        for (let p = 0; p < NX * NY; p++) { GM[p] = Math.hypot(GU[p], GV[p]); tmp[p] = GM[p]; }
        tmp.sort();
        const vmax = Math.log10(Math.max(tmp[Math.floor(0.985 * (tmp.length - 1))], 1e-6)), vmin = Math.max(vmax - 3, Math.min(vmax - 1, Math.log10(Math.max(tmp[Math.floor(0.03 * tmp.length)], 1e-9))));
        bMed = Math.max(tmp[Math.floor(0.5 * tmp.length)], 1e-9);
        const d = img.data;
        for (let j = 0; j < NY; j++) {
          const row = (NY - 1 - j) * NX;
          for (let i = 0; i < NX; i++) {
            const t = (Math.log10(GM[j * NX + i] + 1e-12) - vmin) / (vmax - vmin);
            const k = (t <= 0 ? 0 : t >= 1 ? 255 : (t * 255) | 0) * 3, p = (row + i) * 4;
            d[p] = LUT[k]; d[p + 1] = LUT[k + 1]; d[p + 2] = LUT[k + 2]; d[p + 3] = 255;
          }
        }
        offCtx.putImageData(img, 0, 0);
        colorRange = [vmin, vmax];
        // compass needle sites
        for (let j = 0; j < NCY; j++) for (let i = 0; i < NCX; i++) {
          const k = j * NCX + i, x = x0 + ((i + 0.5) / NCX) * (x1 - x0), y = y0 + ((j + 0.5) / NCY) * (y1 - y0);
          ndX[k] = x; ndY[k] = y;
          ndOn[k] = nearSource(x, y, 0.75) ? 0 : 1;
          fieldAt(x, y);
          ndTB[k] = Math.atan2(fv, fu);
          ndS[k] = PM.clamp(Math.hypot(fu, fv) / bMed, 0.08, 4);
        }
        linesDirty = true; sideDirty = true;
      }
      let colorRange = [0, 1];

      function nearSource(u, v, r) {
        const r2 = r * r;
        if (WIRE) { for (const w of wires) { const dx = u - w.x, dy = v - w.y; if (dx * dx + dy * dy < r2) return true; } return false; }
        for (const z0 of loops) { const dz = u - z0, dr = Math.abs(v) - P.R; if (dz * dz + dr * dr < r2) return true; }
        return false;
      }

      // ------------------------------------------------------------ evenly spaced field lines
      let dX = 0, dY = 0;
      function dirAt(u, v, sgn) {
        fieldAt(u, v);
        const m = Math.hypot(fu, fv);
        if (m < 1e-9) { dX = 0; dY = 0; return 0; }
        dX = (sgn * fu) / m; dY = (sgn * fv) / m; return m;
      }
      function cellOf(u, v) {
        const ci = Math.floor((u - gx0) / ocC), cj = Math.floor((v - gy0) / ocC);
        if (ci < 0 || cj < 0 || ci >= ocW || cj >= ocH) return -1;
        return cj * ocW + ci;
      }
      let trN = 0, trClosed = false;
      function traceDir(u0, v0, sgn, id, buf) {
        let u = u0, v = v0, n = 0, len = 0;
        buf[n++] = u; buf[n++] = v; trClosed = false;
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        const rStop = WIRE ? 0.28 : 0.22;
        for (let s = 0; s < MAXSTEP; s++) {
          if (dirAt(u, v, sgn) === 0) break;
          const a1x = dX, a1y = dY;
          dirAt(u + 0.5 * DSL * a1x, v + 0.5 * DSL * a1y, sgn); const a2x = dX, a2y = dY;
          dirAt(u + 0.5 * DSL * a2x, v + 0.5 * DSL * a2y, sgn); const a3x = dX, a3y = dY;
          dirAt(u + DSL * a3x, v + DSL * a3y, sgn); const a4x = dX, a4y = dY;
          u += (DSL / 6) * (a1x + 2 * a2x + 2 * a3x + a4x); v += (DSL / 6) * (a1y + 2 * a2y + 2 * a3y + a4y);
          len += DSL;
          if (u < x0 || u > x1 || v < y0 || v > y1) { buf[n++] = u; buf[n++] = v; break; }
          if (nearSource(u, v, rStop)) break;
          if (len > 1 && (u - u0) * (u - u0) + (v - v0) * (v - v0) < 0.36 * DSL * DSL * 4) { buf[n++] = u0; buf[n++] = v0; trClosed = true; break; }
          const c = cellOf(u, v);
          if (c >= 0) { if (occ[c] && occ[c] !== id) break; occ[c] = id; }
          buf[n++] = u; buf[n++] = v;
        }
        trN = n;
      }
      function computeLines() {
        nLines = 0; nArw = 0;
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim, dsep = P.dsep;
        ocC = dsep * 0.5; ocW = Math.ceil((x1 - x0) / ocC) + 1; ocH = Math.ceil((y1 - y0) / ocC) + 1;
        if (occ.length < ocW * ocH) occ = new Int32Array(ocW * ocH); else occ.fill(0);
        const seeds = [];
        if (WIRE) {
          for (const w of wires) for (let r = 0.45; r < 3.2; r += dsep) seeds.push([w.x + r, w.y]);
        } else {
          for (let v = -P.R + 0.35; v < P.R - 0.3; v += dsep) seeds.push([0, v]);
          for (let v = P.R + 0.4; v < y1; v += dsep) { seeds.push([0, v]); seeds.push([0, -v]); }
        }
        const cand = [];
        for (let v = y0 + dsep / 2; v < y1; v += dsep) for (let u = x0 + dsep / 2; u < x1; u += dsep) cand.push([u + 0.2 * dsep * (rng.next() - 0.5), v + 0.2 * dsep * (rng.next() - 0.5)]);
        for (let k = cand.length - 1; k > 0; k--) { const j = Math.floor(rng.next() * (k + 1)); const t = cand[k]; cand[k] = cand[j]; cand[j] = t; }
        const all = seeds.concat(cand);
        let o = 0, id = 1;
        for (const [su, sv] of all) {
          if (nLines >= lineOff.length - 1 || o > LINE_CAP - 4 * MAXSTEP - 16) break;
          if (su < x0 || su > x1 || sv < y0 || sv > y1 || nearSource(su, sv, 0.3)) continue;
          // seed must be at least ~dsep away from every existing line
          const c0 = cellOf(su, sv);
          if (c0 < 0) continue;
          const ci = c0 % ocW, cj = (c0 / ocW) | 0;
          let free = true;
          for (let dj = -1; dj <= 1 && free; dj++) for (let di = -1; di <= 1; di++) {
            const ii = ci + di, jj = cj + dj;
            if (ii >= 0 && jj >= 0 && ii < ocW && jj < ocH && occ[jj * ocW + ii]) { free = false; break; }
          }
          if (!free) continue;
          id++;
          occ[c0] = id;
          traceDir(su, sv, 1, id, fwd); const nf = trN, closed = trClosed;
          let nb = 0;
          if (!closed) { traceDir(su, sv, -1, id, bwd); nb = trN; }
          const npts = nb / 2 + nf / 2 - (nb ? 1 : 0);
          if (npts < 12) continue;
          // backward part reversed, then forward part (so the line runs along +B)
          let n = 0;
          for (let q = nb / 2 - 1; q >= 1; q--) { lineBuf[o + n++] = bwd[2 * q]; lineBuf[o + n++] = bwd[2 * q + 1]; }
          for (let q = 0; q < nf / 2; q++) { lineBuf[o + n++] = fwd[2 * q]; lineBuf[o + n++] = fwd[2 * q + 1]; }
          lineOff[nLines] = o; lineLen[nLines] = n / 2;
          const ao = o >> 1; arcBuf[ao] = 0;
          for (let q = 1; q < n / 2; q++) arcBuf[ao + q] = arcBuf[ao + q - 1] + Math.hypot(lineBuf[o + 2 * q] - lineBuf[o + 2 * q - 2], lineBuf[o + 2 * q + 1] - lineBuf[o + 2 * q - 1]);
          const Ltot = arcBuf[ao + n / 2 - 1];
          // arrowheads
          const gap = 3.2, first = Math.min(1.2, Ltot / 2);
          let next = first;
          for (let q = 1; q < n / 2 && nArw < ARW_CAP; q++) {
            if (arcBuf[ao + q] >= next) {
              arw[3 * nArw] = lineBuf[o + 2 * q]; arw[3 * nArw + 1] = lineBuf[o + 2 * q + 1];
              arw[3 * nArw + 2] = Math.atan2(lineBuf[o + 2 * q + 1] - lineBuf[o + 2 * q - 1], lineBuf[o + 2 * q] - lineBuf[o + 2 * q - 2]);
              nArw++; next += gap;
            }
          }
          nLines++; o += n;
        }
      }

      // ------------------------------------------------------------ side plot data
      function ampereCentre() {
        if (P.acen === "sel" && wires[sel]) return { x: wires[sel].x, y: wires[sel].y };
        return { x: 0, y: 0 };
      }
      function circleAvg(cx, cy, r) {
        const n = 160; let s = 0;
        for (let k = 0; k < n; k++) {
          const th = (2 * Math.PI * (k + 0.5)) / n, c = Math.cos(th), sn = Math.sin(th);
          wireB(cx + r * c, cy + r * sn);
          s += -fu * sn + fv * c; // B·φ̂
        }
        return s / n;
      }
      function iEnc(cx, cy, r) { let s = 0; for (const w of wires) if (Math.hypot(w.x - cx, w.y - cy) < r) s += w.I; return s; }
      function computeSide() {
        if (WIRE) {
          const c = ampereCentre();
          for (let k = 0; k < NS; k++) {
            const r = 0.05 + (9.5 * (k + 0.5)) / NS;
            sr[k] = r; sA[k] = circleAvg(c.x, c.y, r); sAmp[k] = (20 * iEnc(c.x, c.y, r)) / r;
            wireB(c.x + r, c.y); sRay[k] = Math.hypot(fu, fv);
          }
          const circ = circleAvg(c.x, c.y, P.RA) * 2 * Math.PI * P.RA * 1e-8 / MU0; // μT·cm → T·m
          ampere = { circ, enc: iEnc(c.x, c.y, P.RA) };
        } else {
          const [x0, x1] = pl.visibleXlim, R = P.R, rho = P.rc * R, s = 10 * P.I;
          for (let k = 0; k < NS; k++) {
            const z = x0 + ((x1 - x0) * (k + 0.5)) / NS;
            sz[k] = z;
            let num = 0, ana = 0, offv = 0;
            for (const z0 of loops) {
              loopKernel(0, z - z0, R); num += kSz * s;
              ana += (20 * Math.PI * P.I * R * R) / Math.pow(R * R + (z - z0) * (z - z0), 1.5);
              loopKernel(rho, z - z0, R); offv += kSz * s;
            }
            sNum[k] = num; sAna[k] = ana; sOff[k] = offv;
            if (P.cfg === "solenoid") {
              const nI = (P.N / P.L) * P.I, a = z + P.L / 2, b = z - P.L / 2;
              sSheet[k] = 20 * Math.PI * nI * (a / Math.sqrt(a * a + R * R) - b / Math.sqrt(b * b + R * R));
            }
          }
          for (let k = 0; k < sNumX.length; k++) { const idx = Math.floor(((k + 0.5) / sNumX.length) * NS); sNumX[k] = sz[idx]; sNumY[k] = sNum[idx]; }
        }
        sideDirty = false;
      }

      const clean = (x, scale) => (Math.abs(x) < 1e-9 * Math.max(1, scale) ? 0 : x);
      // ------------------------------------------------------------ interaction
      function toData(e) {
        const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
        return { px, py, x: pl.invX(px), y: pl.invY(py) };
      }
      function pick(px, py) {
        let best = -1, bd = 16;
        wires.forEach((w, k) => { const d = Math.hypot(pl.X(w.x) - px, pl.Y(w.y) - py); if (d < bd) { bd = d; best = k; } });
        return best;
      }
      function select(k) { sel = k; if (wires[k]) api.setControl("Isel", { value: wires[k].I }); sideDirty = true; layerDirty = true; }
      const onDown = (e) => {
        const d = toData(e); cursor = d;
        if (WIRE) {
          const k = pick(d.px, d.py);
          if (k >= 0) { select(k); drag = k; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; }
        }
        api.invalidate();
      };
      const onMove = (e) => {
        const d = toData(e); cursor = d;
        if (drag >= 0 && wires[drag]) {
          const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
          wires[drag].x = PM.clamp(d.x, x0 + 0.2, x1 - 0.2); wires[drag].y = PM.clamp(d.y, y0 + 0.2, y1 - 0.2);
          fieldDirty = true;
        } else if (WIRE) cv.style.cursor = pick(d.px, d.py) >= 0 ? "grab" : "crosshair";
        api.invalidate();
      };
      const onUp = () => { drag = -1; if (WIRE) cv.style.cursor = "crosshair"; };
      const onLeave = () => { if (drag < 0) { cursor = null; api.invalidate(); } };
      cv.addEventListener("pointerdown", onDown);
      cv.addEventListener("pointermove", onMove);
      cv.addEventListener("pointerup", onUp);
      cv.addEventListener("pointercancel", onUp);
      cv.addEventListener("pointerleave", onLeave);

      function addWire(sign) {
        if (wires.length >= MAXW) { api.status("At most " + MAXW + " wires."); return; }
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        let w = null;
        for (let t = 0; t < 200 && !w; t++) {
          const x = rng.uniform(x0 + 1, x1 - 1), y = rng.uniform(y0 + 1, y1 - 1);
          if (!wires.some((o) => Math.hypot(o.x - x, o.y - y) < 1.2)) w = { x, y, I: sign * P.I };
        }
        wires.push(w || { x: 0, y: 3, I: sign * P.I });
        select(wires.length - 1);
        fieldDirty = true;
      }

      // ------------------------------------------------------------ drawing
      function drawSources(ctx, p) {
        const symbol = (X, Y, out, r, col) => {
          ctx.fillStyle = "rgba(15,21,28,0.85)"; ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.fill();
          ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
          ctx.fillStyle = col; ctx.strokeStyle = col;
          if (out) { ctx.beginPath(); ctx.arc(X, Y, r * 0.3, 0, 2 * Math.PI); ctx.fill(); }
          else { const a = r * 0.55; ctx.beginPath(); ctx.moveTo(X - a, Y - a); ctx.lineTo(X + a, Y + a); ctx.moveTo(X + a, Y - a); ctx.lineTo(X - a, Y + a); ctx.stroke(); }
        };
        if (WIRE) {
          wires.forEach((w, k) => {
            const X = p.X(w.x), Y = p.Y(w.y), r = 7 + 2.2 * Math.sqrt(Math.abs(w.I) / 10);
            const col = w.I > 0 ? "#ffa657" : w.I < 0 ? "#58d6ff" : "#8b98a8";
            const g = ctx.createRadialGradient(X, Y, r * 0.5, X, Y, r * 2.3);
            g.addColorStop(0, w.I >= 0 ? "rgba(255,166,87,0.5)" : "rgba(88,214,255,0.5)"); g.addColorStop(1, "rgba(0,0,0,0)");
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X, Y, r * 2.3, 0, 2 * Math.PI); ctx.fill();
            symbol(X, Y, w.I >= 0, r, col);
            if (k === sel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(X, Y, r + 5, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]); }
          });
        } else {
          const R = P.R;
          if (P.cfg === "solenoid") {
            ctx.fillStyle = "rgba(200,140,60,0.10)";
            ctx.fillRect(p.X(-P.L / 2), p.Y(R + 0.25), p.X(P.L / 2) - p.X(-P.L / 2), p.Y(R - 0.25) - p.Y(R + 0.25));
            ctx.fillRect(p.X(-P.L / 2), p.Y(-R + 0.25), p.X(P.L / 2) - p.X(-P.L / 2), p.Y(-R - 0.25) - p.Y(-R + 0.25));
          }
          ctx.strokeStyle = "rgba(230,237,243,0.25)"; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
          ctx.beginPath();
          for (const z0 of loops) { ctx.moveTo(p.X(z0), p.Y(R)); ctx.lineTo(p.X(z0), p.Y(-R)); }
          ctx.stroke(); ctx.setLineDash([]);
          const r = Math.max(3.5, Math.min(7, 0.42 * (P.cfg === "solenoid" ? P.L / P.N : 2) * p.sx));
          for (const z0 of loops) { symbol(p.X(z0), p.Y(R), P.I > 0, r, "#ffa657"); symbol(p.X(z0), p.Y(-R), P.I < 0, r, "#58d6ff"); }
          // axis
          ctx.strokeStyle = "rgba(230,237,243,0.35)"; ctx.setLineDash([8, 5]); ctx.beginPath(); ctx.moveTo(p.X(-100), p.Y(0)); ctx.lineTo(p.X(100), p.Y(0)); ctx.stroke(); ctx.setLineDash([]);
        }
      }
      function drawStatic() {
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        pl.clear();
        if (P.showHeat) pl.custom((ctx, p) => {
          ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
          ctx.drawImage(off, p.X(gx0 - hx / 2), p.Y(gy0 + (NY - 0.5) * hy), NX * hx * p.sx, NY * hy * p.sy);
        });
        if (P.showLines) pl.custom((ctx, p) => {
          ctx.lineJoin = "round"; ctx.lineCap = "round";
          ctx.beginPath();
          for (let l = 0; l < nLines; l++) {
            const o = lineOff[l], n = lineLen[l];
            ctx.moveTo(p.X(lineBuf[o]), p.Y(lineBuf[o + 1]));
            for (let q = 1; q < n; q++) ctx.lineTo(p.X(lineBuf[o + 2 * q]), p.Y(lineBuf[o + 2 * q + 1]));
          }
          ctx.strokeStyle = "rgba(200,225,255,0.5)"; ctx.lineWidth = 1.2; ctx.stroke();
          ctx.fillStyle = "rgba(210,232,255,0.85)"; ctx.beginPath();
          for (let a = 0; a < nArw; a++) {
            const X = p.X(arw[3 * a]), Y = p.Y(arw[3 * a + 1]), th = -arw[3 * a + 2];
            ctx.moveTo(X + 5.5 * Math.cos(th), Y + 5.5 * Math.sin(th));
            ctx.lineTo(X + 5.5 * Math.cos(th + 2.5), Y + 5.5 * Math.sin(th + 2.5));
            ctx.lineTo(X + 5.5 * Math.cos(th - 2.5), Y + 5.5 * Math.sin(th - 2.5)); ctx.closePath();
          }
          ctx.fill();
        });
        if (WIRE) {
          const c = ampereCentre(), ok = Math.abs(ampere.circ - ampere.enc) < 1e-3 * Math.max(1, Math.abs(ampere.enc));
          pl.circle(c.x, c.y, P.RA, { fill: false, stroke: ok ? "rgba(63,185,80,0.9)" : "rgba(245,158,11,0.9)", strokeWidth: 1.5 });
          pl.text(c.x + P.RA * 0.71, c.y + P.RA * 0.71, "Ampère loop", { color: ok ? "#3fb950" : "#f59e0b", size: 11, dx: 4, dy: -6 });
        }
        pl.custom(drawSources);
        layer.width = cv.width; layer.height = cv.height;
        layerCtx.drawImage(cv, 0, 0);
        layerDirty = false;
      }
      function forceOn(k) { // N/m
        const w = wires[k]; let bx = 0, by = 0;
        for (let j = 0; j < wires.length; j++) {
          if (j === k) continue;
          const o = wires[j], dx = w.x - o.x, dy = w.y - o.y, r2 = dx * dx + dy * dy + 1e-9, f = (20 * o.I) / r2;
          bx -= f * dy; by += f * dx;
        }
        // f = I ẑ × B  (B in T)
        return { fx: -w.I * by * 1e-6, fy: w.I * bx * 1e-6 };
      }

      return {
        reset() {
          loadConfig();
        },
        onParam(id, v) {
          layerDirty = true;
          if (id === "Isel" && wires[sel]) { wires[sel].I = v; fieldDirty = true; }
          if (id === "dsep") linesDirty = true;
          if (id === "RA" || id === "acen" || id === "rc") sideDirty = true;
        },
        onAction(id) {
          layerDirty = true;
          if (id === "addOut") addWire(1);
          else if (id === "addIn") addWire(-1);
          else if (id === "remove") {
            if (wires.length > 1) { wires.splice(sel, 1); select(Math.min(sel, wires.length - 1)); fieldDirty = true; }
            else api.status("At least one wire must remain.");
          }
        },
        step(dt) {
          flowPhase += dt;
          // compass needles: damped torque toward the local field direction
          const h = Math.min(dt, 1 / 30), kap = 70, gam = 5;
          for (let k = 0; k < ndT.length; k++) {
            if (!ndOn[k]) continue;
            let d = ndTB[k] - ndT[k];
            d = Math.atan2(Math.sin(d), Math.cos(d));
            ndW[k] += h * (kap * Math.sqrt(ndS[k]) * Math.sin(d) - gam * ndW[k]);
            ndT[k] += h * ndW[k];
          }
        },
        render() {
          if (fieldDirty) { computeGrid(); fieldDirty = false; layerDirty = true; }
          if (linesDirty) { computeLines(); linesDirty = false; layerDirty = true; }
          if (sideDirty) { computeSide(); layerDirty = true; }
          if (layerDirty) drawStatic();
          else {
            pl.clear();
            pl.custom((ctx) => { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(layer, 0, 0); ctx.restore(); });
          }
          // flowing dots along B
          if (P.showLines && P.flow) pl.custom((ctx, p) => {
            ctx.fillStyle = "rgba(235,245,255,0.9)"; ctx.beginPath();
            const sp = 0.6, ph = (flowPhase * 0.9) % sp;
            for (let l = 0; l < nLines; l++) {
              const o = lineOff[l], n = lineLen[l], ao = o >> 1;
              let next = ph;
              for (let q = 1; q < n; q++) {
                const s1 = arcBuf[ao + q];
                while (next <= s1) {
                  const s0 = arcBuf[ao + q - 1], f = (next - s0) / (s1 - s0 || 1);
                  const X = p.X(lineBuf[o + 2 * q - 2] + f * (lineBuf[o + 2 * q] - lineBuf[o + 2 * q - 2]));
                  const Y = p.Y(lineBuf[o + 2 * q - 1] + f * (lineBuf[o + 2 * q + 1] - lineBuf[o + 2 * q - 1]));
                  ctx.moveTo(X + 1.7, Y); ctx.arc(X, Y, 1.7, 0, 2 * Math.PI);
                  next += sp;
                }
              }
            }
            ctx.fill();
          });
          // compass needles
          if (P.compass) pl.custom((ctx, p) => {
            const Lh = 0.36 * p.sx, Wd = 0.09 * p.sx;
            for (let k = 0; k < ndT.length; k++) {
              if (!ndOn[k]) continue;
              const X = p.X(ndX[k]), Y = p.Y(ndY[k]), c = Math.cos(ndT[k]), s = -Math.sin(ndT[k]);
              ctx.fillStyle = "rgba(15,21,28,0.35)"; ctx.beginPath(); ctx.arc(X, Y, Lh + 2, 0, 2 * Math.PI); ctx.fill();
              ctx.strokeStyle = "rgba(200,210,220,0.35)"; ctx.lineWidth = 1; ctx.stroke();
              ctx.fillStyle = "#f85149";
              ctx.beginPath(); ctx.moveTo(X + c * Lh, Y + s * Lh); ctx.lineTo(X - s * Wd, Y + c * Wd); ctx.lineTo(X + s * Wd, Y - c * Wd); ctx.closePath(); ctx.fill();
              ctx.fillStyle = "#e6edf3";
              ctx.beginPath(); ctx.moveTo(X - c * Lh, Y - s * Lh); ctx.lineTo(X - s * Wd, Y + c * Wd); ctx.lineTo(X + s * Wd, Y - c * Wd); ctx.closePath(); ctx.fill();
            }
          });
          // colour bar (log10 |B|)
          if (P.showHeat) pl.custom((ctx, p) => {
            const W = 130, H = 9, X = p.m.l + 10, Y = p.m.t + p._v.ph - 30;
            ctx.fillStyle = "rgba(15,21,28,0.85)"; ctx.fillRect(X - 6, Y - 16, W + 12, H + 34);
            for (let i = 0; i < W; i++) { const k = Math.floor((i / (W - 1)) * 255) * 3; ctx.fillStyle = `rgb(${LUT[k]},${LUT[k + 1]},${LUT[k + 2]})`; ctx.fillRect(X + i, Y, 1.5, H); }
            ctx.fillStyle = "#c9d1d9"; ctx.font = "11px system-ui, sans-serif"; ctx.textBaseline = "alphabetic";
            ctx.textAlign = "left"; ctx.fillText("|B| (μT), log scale", X, Y - 4);
            ctx.textBaseline = "top";
            ctx.fillText(PM.fmt(Math.pow(10, colorRange[0]), 1), X, Y + H + 3);
            ctx.textAlign = "right"; ctx.fillText(PM.fmt(Math.pow(10, colorRange[1]), 0), X + W, Y + H + 3);
          });
          // forces on wires
          if (WIRE && P.force && wires.length > 1) {
            let fmax = 1e-12; const F = wires.map((w, k) => forceOn(k));
            for (const f of F) fmax = Math.max(fmax, Math.hypot(f.fx, f.fy));
            wires.forEach((w, k) => {
              const f = F[k], m = Math.hypot(f.fx, f.fy);
              if (m < 1e-6 * fmax) return;
              const L = 0.4 + 1.6 * Math.sqrt(m / fmax);
              pl.arrow(w.x, w.y, w.x + (L * f.fx) / m, w.y + (L * f.fy) / m, { color: "#ff6b6b", width: 2.4, head: 9 });
            });
          }
          // cursor probe
          let Bc = null;
          if (cursor) {
            fieldAt(cursor.x, cursor.y); Bc = Math.hypot(fu, fv);
            pl.circle(cursor.x, cursor.y, 5, { px: true, fill: false, stroke: "rgba(255,255,255,0.85)", strokeWidth: 1 });
          }
          M.set("B", Bc === null ? "— (hover the plot)" : `${PM.fmt(Bc, 3)} μT`);

          const sp = plots.s;
          if (WIRE) {
            M.set("amp", `${PM.fmt(clean(ampere.circ, P.I), 3)} | ${PM.fmt(clean(ampere.enc, P.I), 2)} A`);
            if (wires[sel]) {
              const f = forceOn(sel), m = Math.hypot(f.fx, f.fy) * 1e3;
              M.set("F", wires.length > 1 ? `${PM.fmt(clean(m, 1e-3), 3)} mN/m` : "— (single wire)");
            }
            if (wires.length === 2) {
              const d = Math.hypot(wires[0].x - wires[1].x, wires[0].y - wires[1].y) / 100;
              const f = (MU0 * wires[0].I * wires[1].I) / (2 * Math.PI * d) * 1e3;
              M.set("F2", `${PM.fmt(Math.abs(f), 3)} mN/m, ${f > 0 ? "attractive" : f < 0 ? "repulsive" : "zero"}`);
            } else M.set("F2", "— (needs exactly 2 wires)");
            let It = 0; for (const w of wires) It += w.I;
            M.set("Itot", `${PM.fmt(clean(It, P.I), 2)} A (${wires.length} wire${wires.length > 1 ? "s" : ""})`);
            // side plot
            let ymax = 1, ymin = 0;
            for (let k = 0; k < NS; k++) { ymax = Math.max(ymax, Math.min(sRay[k], 4 * Math.abs(sAmp[k]) + 1), sA[k]); ymin = Math.min(ymin, sA[k], sAmp[k]); }
            const yr = Math.min(ymax, 2500);
            sp.setLimits([0, 9.5], [Math.max(ymin * 1.15, -yr), yr * 1.12]);
            sp.clear();
            sp.hline(0, { color: "rgba(139,152,168,0.5)" });
            sp.line(sr, sRay, { color: "rgba(139,152,168,0.9)", width: 1.3, dash: [4, 3] });
            sp.line(sr, sAmp, { color: PlotColors.accent3, width: 2.4 });
            sp.line(sr, sA, { color: PlotColors.accent, width: 1.8 });
            sp.vline(P.RA, { color: "rgba(63,185,80,0.8)", dash: [5, 4] });
            sp.legend([
              { label: "⟨B_φ⟩(r), numerical circle average", color: PlotColors.accent },
              { label: "μ₀ I_enc(r) / 2πr (Ampère)", color: PlotColors.accent3 },
              { label: "|B| along the +x ray", color: "rgba(139,152,168,0.9)", dash: [4, 3] },
            ], "tr");
          } else {
            const R = P.R, mid = Math.floor(NS / 2);
            const iz0 = sz.reduce((b, z, k) => (Math.abs(z) < Math.abs(sz[b]) ? k : b), 0);
            let num0 = 0, ana0 = 0;
            for (const z0 of loops) { loopKernel(0, -z0, R); num0 += kSz * 10 * P.I; ana0 += (20 * Math.PI * P.I * R * R) / Math.pow(R * R + z0 * z0, 1.5); }
            M.set("B0", `${PM.fmt(num0, 2)} | ${PM.fmt(ana0, 2)} μT`);
            if (P.cfg === "solenoid") {
              const ideal = (40 * Math.PI * P.N * P.I) / P.L;
              M.set("x1", PM.fmt(num0 / ideal, 4)); M.set("x2", `${PM.fmt(ideal, 1)} μT`);
            } else if (P.cfg === "helmholtz") {
              let b4 = 0; for (const z0 of loops) { loopKernel(0, R / 4 - z0, R); b4 += kSz * 10 * P.I; }
              M.set("x1", PM.fmt(Math.abs(b4 - num0) / num0, 2)); M.set("x2", `${PM.fmt(ana0, 2)} μT`);
            } else {
              loopKernel(0, R, R); const bR = kSz * 10 * P.I;
              M.set("x1", PM.fmt(bR / num0, 4)); M.set("x2", `${PM.fmt(ana0, 2)} μT`);
            }
            const mm = loops.length * P.I * Math.PI * R * R * 1e-4;
            M.set("m", `${PM.fmt(mm, 3)} A·m²`);
            let ymax = 1; for (let k = 0; k < NS; k++) ymax = Math.max(ymax, sNum[k], sOff[k]);
            const [x0, x1] = pl.visibleXlim;
            sp.setLimits([x0, x1], [Math.min(0, ...sOff) * 1.1, ymax * 1.15]);
            sp.clear();
            sp.hline(0, { color: "rgba(139,152,168,0.5)" });
            for (const z0 of loops) sp.vline(z0, { color: "rgba(255,166,87,0.22)", width: 1 });
            const items = [];
            if (P.cfg === "solenoid") {
              const ideal = (40 * Math.PI * P.N * P.I) / P.L;
              sp.line([-P.L / 2, P.L / 2], [ideal, ideal], { color: "rgba(230,237,243,0.6)", dash: [6, 4], width: 1.4 });
              sp.line(sz, sSheet, { color: PlotColors.accent2, width: 1.6, dash: [3, 3] });
              items.push({ label: "μ₀nI (ideal solenoid)", color: "rgba(230,237,243,0.6)", dash: [6, 4] });
              items.push({ label: "current-sheet formula", color: PlotColors.accent2, dash: [3, 3] });
            }
            sp.line(sz, sAna, { color: PlotColors.accent3, width: 2.4 });
            sp.points(sNumX, sNumY, { color: PlotColors.accent, size: 3.6 });
            sp.line(sz, sOff, { color: PlotColors.pink, width: 1.6 });
            items.unshift({ label: `numerical B_z off axis, ρ = ${PM.fmt(P.rc * R, 2)} cm`, color: PlotColors.pink });
            items.unshift({ label: "analytic Σ μ₀IR²/2(R²+z²)^{3/2}", color: PlotColors.accent3 });
            items.unshift({ label: "numerical Biot–Savart on axis", color: PlotColors.accent, type: "dot" });
            sp.legend(items, "tr");
            void mid; void iz0;
          }
          api.setTime("");
        },
        destroy() {
          cv.removeEventListener("pointerdown", onDown);
          cv.removeEventListener("pointermove", onMove);
          cv.removeEventListener("pointerup", onUp);
          cv.removeEventListener("pointercancel", onUp);
          cv.removeEventListener("pointerleave", onLeave);
        },
      };
    },
  });
})();
