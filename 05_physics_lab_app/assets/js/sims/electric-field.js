/* Electric fields and potentials of point charges — Coulomb superposition on a grid, equipotentials by
   marching squares, RK4 field lines, Gauss's-law flux check and a test charge moving under the Coulomb force. */
(function () {
  "use strict";

  const NX = 300, NY = 190;      // potential grid (spans the visible plot area)
  const V0 = 0.04;               // asinh scale of the potential colouring
  const VCLIP = 10;              // |V| beyond this value is saturated in the colour map
  const EPS = 0.05;              // softening length of the test-charge dynamics
  const R0 = 0.05;               // field lines start/stop this close to a charge
  const DS = 0.02;               // arc-length step of the field-line RK4 integrator
  const MAXSTEP = 1400;          // max RK4 steps per field line
  const MAXREAL = 40, MAXSRC = 2 * MAXREAL;
  const LINE_CAP = 900000, SEG_CAP = 800000, ARW_CAP = 6000, TRAIL = 2500;
  const YP = -1.5;               // position of the grounded plane (preset "ground")
  const NLEV = 8;                // equipotential levels per sign
  const GL_N = 32, PHI_N = 64;   // Gauss-sphere quadrature

  /** Gauss–Legendre nodes and weights on [-1, 1]. */
  function gaussLegendre(n) {
    const x = new Float64Array(n), w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), pp = 1;
      for (let it = 0; it < 100; it++) {
        let p1 = 1, p2 = 0;
        for (let j = 1; j <= n; j++) { const p3 = p2; p2 = p1; p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j; }
        pp = (n * (z * p1 - p2)) / (z * z - 1);
        const z1 = z; z = z1 - p1 / pp;
        if (Math.abs(z - z1) < 1e-15) break;
      }
      x[i] = z; w[i] = 2 / ((1 - z * z) * pp * pp);
    }
    return { x, w };
  }

  /** Diverging colour map with a dark centre: blue (V<0) — background — red/yellow (V>0). */
  function buildLUT() {
    const neg = [[15, 21, 28], [22, 42, 92], [35, 90, 190], [80, 160, 245], [190, 230, 255]];
    const pos = [[15, 21, 28], [92, 24, 30], [190, 52, 40], [245, 130, 50], [255, 225, 140]];
    const L = new Uint8ClampedArray(512 * 3);
    for (let k = 0; k < 512; k++) {
      const t = (k - 255.5) / 255.5, st = t < 0 ? neg : pos, a = Math.min(Math.abs(t), 1) * (st.length - 1);
      const i0 = Math.min(Math.floor(a), st.length - 2), f = a - i0;
      for (let c = 0; c < 3; c++) L[k * 3 + c] = st[i0][c] + (st[i0 + 1][c] - st[i0][c]) * f;
    }
    return L;
  }

  function presetCharges(name, rng) {
    switch (name) {
      case "single": return [{ x: 0, y: 0, q: 1 }];
      case "dipole": return [{ x: -1.2, y: 0, q: 1 }, { x: 1.2, y: 0, q: -1 }];
      case "like": return [{ x: -1.2, y: 0, q: 1 }, { x: 1.2, y: 0, q: 1 }];
      case "unequal": return [{ x: -0.6, y: 0, q: 2 }, { x: 0.6, y: 0, q: -1 }];
      case "quad": return [{ x: -1, y: 1, q: 1 }, { x: 1, y: 1, q: -1 }, { x: 1, y: -1, q: 1 }, { x: -1, y: -1, q: -1 }];
      case "cap": {
        const a = [];
        for (let k = 0; k < 13; k++) { const x = -3 + k * 0.5; a.push({ x, y: 0.9, q: 0.5 }); a.push({ x, y: -0.9, q: -0.5 }); }
        return a;
      }
      case "ground": return [{ x: 0, y: 0, q: 1 }];
      default: { // random
        const n = 3 + rng.int(0, 3), a = [];
        let guard = 0;
        while (a.length < n && guard++ < 500) {
          const x = rng.uniform(-4, 4), y = rng.uniform(-2.4, 2.4);
          if (a.some((c) => Math.hypot(c.x - x, c.y - y) < 1.1)) continue;
          const q = (rng.next() < 0.5 ? -1 : 1) * (0.5 + 0.5 * rng.int(0, 3));
          a.push({ x, y, q });
        }
        return a;
      }
    }
  }

  App.register({
    id: "electric-field",
    category: "classical",
    group: "Electromagnetism",
    order: 20,
    title: "Electric Fields & Potentials of Point Charges",
    icon: "⚡",
    subtitle: "Drag point charges around and watch the potential map, the equipotentials and the field lines recompute instantly; release a test charge and follow it through the Coulomb field.",
    notes: [{
      type: "info",
      html: "<b>Drag</b> a charge to move it (click it to select it and edit its charge). <b>Click on empty space</b> to move the launch point (✛) of the test charge, " +
        "then press <b>Release test charge</b>. Colour = electric potential $V$ (red positive, blue negative, compressed logarithmically near the charges), " +
        "thin lines = equipotentials, bright lines with arrows = field lines (the flowing dots move along $\\mathbf E$).",
    }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "preset", type: "select", label: "Charge configuration", value: "dipole",
        options: [
          { value: "single", label: "Single charge" },
          { value: "dipole", label: "Electric dipole (+q, −q)" },
          { value: "like", label: "Two like charges (+q, +q)" },
          { value: "unequal", label: "Unequal pair (+2q, −q)" },
          { value: "quad", label: "Quadrupole" },
          { value: "cap", label: "Parallel-plate capacitor (two rows)" },
          { value: "ground", label: "Charge above a grounded plane (image charge)" },
          { value: "random", label: "Random charges" },
        ] },
      { id: "reload", type: "button", label: "↺ Reload configuration" },
      { type: "section", label: "Selected charge" },
      { id: "qsel", type: "slider", label: "Charge $q$ of the selected charge", min: -5, max: 5, step: 0.25, value: 1, live: true,
        help: "Click a charge to select it (white ring). Units: $k = 1/4\\pi\\varepsilon_0 = 1$." },
      { id: "addPos", type: "button", label: "＋ Add positive charge" },
      { id: "addNeg", type: "button", label: "－ Add negative charge" },
      { id: "remove", type: "button", label: "✕ Remove selected charge" },
      { id: "gaussR", type: "slider", label: "Gauss sphere radius $R$", min: 0.2, max: 3, step: 0.05, value: 0.8, live: true,
        help: "A sphere of radius $R$ centred on the selected charge (its equator is the dashed circle). The flux through it is integrated numerically." },
      { type: "section", label: "Display" },
      { id: "showHeat", type: "checkbox", label: "Potential map $V(x,y)$", value: true, live: true },
      { id: "showEq", type: "checkbox", label: "Equipotential lines", value: true, live: true },
      { id: "showLines", type: "checkbox", label: "Field lines", value: true, live: true },
      { id: "density", type: "slider", label: "Field lines per unit charge", min: 4, max: 24, step: 1, value: 12, live: true },
      { id: "flow", type: "checkbox", label: "Animate the field-line direction", value: true, live: true },
      { id: "showArrows", type: "checkbox", label: "Arrow grid (direction of $\\mathbf E$)", value: false, live: true },
      { type: "section", label: "Test charge" },
      { id: "qt", type: "slider", label: "Test charge $q_t$ (mass $m = 1$)", min: -1, max: 1, step: 0.05, value: 0.3, live: true },
      { id: "v0", type: "slider", label: "Launch speed $v_0$", min: 0, max: 3, step: 0.05, value: 0, live: true },
      { id: "ang", type: "slider", label: "Launch direction", min: -180, max: 180, step: 5, value: 90, live: true, fmt: (v) => v + "°" },
      { id: "release", type: "button", label: "▶ Release test charge", primary: true },
      { id: "clear", type: "button", label: "Remove test charge" },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A set of static point charges $q_i$ at positions $\\mathbf r_i$ in the plane $z = 0$ of ordinary three-dimensional space.
      The page shows the slice $z = 0$ through the 3D field, so the charges obey the true $1/r^2$ Coulomb law (they are not
      infinite line charges). Units: lengths are in arbitrary units $a$, charges in units $q_0$, and the Coulomb constant is set to
      $k = 1/4\\pi\\varepsilon_0 = 1$ (so $\\varepsilon_0 = 1/4\\pi$); potentials are then in $q_0/a$, fields in $q_0/a^2$.
      The source charges are held fixed (electrostatics). A light <i>test charge</i> $q_t$ of mass $m = 1$ can be released; it feels
      the field but does not disturb the sources. In the "grounded plane" configuration the half-space $y < -1.5$ is an ideal
      conductor held at $V = 0$.</p>

      <h4>Equations being solved</h4>
      <p>Coulomb's law plus the superposition principle give the potential and the field of $N$ point charges:</p>
      <div class="callout">$$ V(\\mathbf r)=\\sum_{i=1}^{N}\\frac{k\\,q_i}{|\\mathbf r-\\mathbf r_i|},\\qquad
      \\mathbf E(\\mathbf r)=-\\nabla V=\\sum_{i=1}^{N}\\frac{k\\,q_i\\,(\\mathbf r-\\mathbf r_i)}{|\\mathbf r-\\mathbf r_i|^{3}} $$</div>
      <p>These satisfy Gauss's law $\\nabla\\cdot\\mathbf E=\\rho/\\varepsilon_0$ and $\\nabla\\times\\mathbf E=0$. In integral form, for any
      closed surface $S$,</p>
      $$ \\Phi_E=\\oint_S \\mathbf E\\cdot d\\mathbf A=\\frac{Q_{\\text{enc}}}{\\varepsilon_0}=4\\pi k\\,Q_{\\text{enc}} . $$
      <p>A <b>field line</b> is a curve $\\mathbf r(s)$ everywhere tangent to $\\mathbf E$; parametrised by arc length $s$ it solves
      $d\\mathbf r/ds=\\mathbf E/|\\mathbf E|$. <b>Equipotentials</b> are the level curves $V(\\mathbf r)=\\text{const}$ and cross the field lines
      at right angles. For the <b>grounded plane</b> $y=y_p$ the method of images replaces the conductor by a charge $-q$ at the mirror
      point $(x,\\,2y_p-y)$; the combination satisfies $V=0$ on the plane, so by the uniqueness theorem it is the true field above it.
      The induced surface charge is $\\sigma=\\varepsilon_0E_y(x,y_p^+)$. The <b>test charge</b> obeys Newton's law</p>
      $$ m\\,\\ddot{\\mathbf r}=q_t\\,\\mathbf E_\\varepsilon(\\mathbf r),\\qquad
      \\mathbf E_\\varepsilon=\\sum_i\\frac{k q_i(\\mathbf r-\\mathbf r_i)}{\\left(|\\mathbf r-\\mathbf r_i|^2+\\varepsilon^2\\right)^{3/2}},$$
      <p>with total energy $\\mathcal E=\\tfrac12 m v^2+q_t\\sum_i k q_i/\\sqrt{|\\mathbf r-\\mathbf r_i|^2+\\varepsilon^2}$ conserved.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Potential map:</b> $V$ is evaluated exactly (direct sum) on a $300\\times190$ grid spanning the visible area every time a
        charge moves. The colour is $\\operatorname{asinh}(V/V_0)$ with $V_0=0.04$ and $|V|$ clipped at $10$ — a symmetric logarithmic
        scale, so both the singular neighbourhood of each charge and the weak far field are visible.</li>
        <li><b>Equipotentials:</b> marching squares on that grid (linear interpolation along cell edges, saddle cells resolved by the
        cell average) at 17 levels $V=\\pm V_0\\sinh(k\\Delta)$, which double from one level to the next; $V=0$ is drawn dashed.</li>
        <li><b>Field lines:</b> started on a small circle ($r=0.05$) around every positive charge, $n_i=\\operatorname{round}(\\lambda|q_i|)$
        seeds evenly spaced in angle ($\\lambda$ = "lines per unit charge"), and integrated with classical RK4 in arc length,
        $\\Delta s=0.02$. A line stops when it reaches another charge, the conductor or leaves the view. Lines are also traced backwards
        from negative charges and kept only if they come from infinity (or from the conductor), so a net negative charge still
        gets its lines. Arrowheads are placed every 1.6 length units; the moving dots travel along $+\\mathbf E$ (their speed is purely decorative).</li>
        <li><b>Gauss check:</b> the flux through a sphere of radius $R$ about the selected charge is computed with a
        $32$-point Gauss–Legendre rule in $\\cos\\theta$ times a $64$-point rule in $\\varphi$:
        $\\Phi\\approx R^2\\sum_{a,b}w_a\\,\\tfrac{2\\pi}{64}\\,\\mathbf E\\cdot\\hat{\\mathbf n}$. The metric shows $\\Phi/4\\pi k$, to be compared
        with the enclosed charge (image charges count as the induced charge on the conductor).</li>
        <li><b>Test charge:</b> RK4 for $(x,y,v_x,v_y)$ with an adaptive step $h\\propto\\min\\!\\left(r_{\\min}/v,\\;r_{\\min}^{3/2}\\right)$
        ($10^{-5}\\le h\\le 4\\times10^{-3}$), where $r_{\\min}$ is the distance to the nearest charge, and a softening length
        $\\varepsilon=0.05$. The relative drift of the energy $\\mathcal E$ is displayed as the accuracy monitor (re-zeroed whenever a source charge is
        moved or edited, because the potential then changes in time); it typically stays
        below $10^{-6}$. The particle is stopped when it hits an attracting charge, the conductor, or leaves the region.</li>
        <li><b>Profile plot:</b> $V$ and $E_x$ evaluated exactly along the dashed horizontal line through the probe point.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Gauss's law.</b> Select a charge and grow the Gauss sphere: $\\Phi/4\\pi k$ equals the enclosed charge to
        4–6 digits and jumps by exactly $q_j$ when the sphere swallows another charge $q_j$ — no matter where the outside charges are.</li>
        <li><b>Dipole far field.</b> In the dipole preset move the probe far away along the axis and perpendicular to it: $V$
        falls like $p\\cos\\theta/r^2$ and the $V=0$ line is the perpendicular bisector.</li>
        <li><b>Unequal pair (+2q, −q), separation $d=1.2$.</b> Exactly half of the flux leaving $+2q$ ends on $-q$, the rest escapes to
        infinity. Probe the axis to the right: there is a null point $\\mathbf E=0$ (where field lines split) at a distance
        $d/(\\sqrt2-1)\\approx2.9$ beyond the negative charge, i.e. at $x\\approx3.5$.</li>
        <li><b>Capacitor.</b> Between the rows the field lines are almost parallel and the equipotentials evenly spaced
        (uniform field); outside, the fields of the two rows nearly cancel. Watch the fringing field at the ends.</li>
        <li><b>Image charge.</b> In the grounded-plane preset every field line meets the conductor at right angles; the
        induced charge (coloured strip) peaks below the charge with $\\sigma=-q\\,d/2\\pi(x^2+d^2)^{3/2}$ and integrates to $-q$.</li>
        <li><b>Orbits.</b> Put a single negative charge, give the test charge $q_t>0$ a sideways launch speed $v_0\\approx\\sqrt{k|q q_t|/m r}$:
        it moves on a circle; other speeds give Kepler ellipses (or hyperbolae above the escape speed), while $\\mathcal E$ stays constant.</li>
      </ol>

      <h4>Limitations & further reading</h4>
      <p>Only the plane $z=0$ is shown, so the drawn line density is a 2D cut of a 3D flux tube: the number of lines leaving a charge is
      proportional to $q$, but the density in the plane is only qualitatively proportional to $|\\mathbf E|$. Charges are ideal points
      (the field is singular at them; the test-charge force is softened within $\\varepsilon$). Radiation and magnetic effects of the
      moving test charge are ignored. See D. J. Griffiths, <i>Introduction to Electrodynamics</i>, ch. 2–3; E. M. Purcell &amp; D. J. Morin,
      <i>Electricity and Magnetism</i>, ch. 1–3; J. D. Jackson, <i>Classical Electrodynamics</i>, ch. 1–2.</p>`,

    mount(api) {
      const P = api.params;
      const rng = new PM.RNG(20261001);
      const LUT = buildLUT();
      const GL = gaussLegendre(GL_N);

      // ---- state
      let charges = [], sel = 0, loaded = null;
      const sx = new Float64Array(MAXSRC), sy = new Float64Array(MAXSRC), sq = new Float64Array(MAXSRC);
      const sreal = new Uint8Array(MAXSRC);
      let ns = 0, ground = false;
      const S = new Float32Array(NX * NY);
      let gx0 = 0, gy0 = 0, hx = 1, hy = 1, smax = Math.asinh(VCLIP / V0);
      const seg = new Float32Array(SEG_CAP); let nseg = 0;
      const segZero = new Float32Array(SEG_CAP / 4); let nsegZero = 0;
      const lineBuf = new Float32Array(LINE_CAP), arcBuf = new Float32Array(LINE_CAP / 2), lineOff = new Int32Array(4000), lineLen = new Int32Array(4000); let nLines = 0;
      const arw = new Float32Array(ARW_CAP * 3); let nArw = 0;
      const AGX = 30, AGY = 19;
      const agx = new Float32Array(AGX * AGY), agy = new Float32Array(AGX * AGY), agu = new Float32Array(AGX * AGY), agv = new Float32Array(AGX * AGY), aga = new Float32Array(AGX * AGY);
      let fieldDirty = true, flowPhase = 0, layerDirty = true;
      const layer = document.createElement("canvas"), layerCtx = layer.getContext("2d");
      // heat-map raster
      const off = document.createElement("canvas"); off.width = NX; off.height = NY;
      const offCtx = off.getContext("2d"), img = offCtx.createImageData(NX, NY);
      // test charge
      const tc = new Float64Array(4), k1 = new Float64Array(4), k2 = new Float64Array(4), k3 = new Float64Array(4), k4 = new Float64Array(4), tmp = new Float64Array(4);
      let rebase = false;
      let tcActive = false, tcState = "", tcT = 0, tcE0 = 0, tcDrift = 0, simTime = 0;
      const trX = new Float32Array(TRAIL), trY = new Float32Array(TRAIL); let trN = 0, trHead = 0;
      let launch = { x: -2.6, y: 1.3 };
      let cursor = null, drag = -1;
      // profile plot buffers
      const NPF = 400; const pfx = new Float64Array(NPF), pfV = new Float64Array(NPF), pfE = new Float64Array(NPF);

      const M = api.metrics([
        { id: "E", label: "$|\\mathbf E|$ and direction at probe" },
        { id: "V", label: "Potential $V$ at probe" },
        { id: "Q", label: "Net charge $Q=\\sum q_i$" },
        { id: "p", label: "Dipole moment $|\\mathbf p|$" },
        { id: "gauss", label: "Gauss: $\\Phi/4\\pi k$ | $Q_{\\rm enc}$" },
        { id: "tc", label: "Test charge $|\\Delta\\mathcal E/\\mathcal E|$" },
      ]);
      const plots = api.plots([
        { id: "f", title: "Potential (colour), equipotentials and field lines — drag the charges", span: 2, aspect: 0.56, xlim: [-6, 6], ylim: [-3.4, 3.4], equal: true, xlabel: "x", ylabel: "y", grid: false, maxHeight: 640 },
        { id: "pf", title: "Profile along the dashed line through the probe: $V(x)$ and $E_x(x)$", span: 2, aspect: 0.2, minHeight: 170, maxHeight: 240, xlim: [-6, 6], ylim: [-3, 3], xlabel: "x", ylabel: "V, Eₓ" },
      ]);
      const pl = plots.f, cv = pl.canvas;
      const prevResize = pl.onResize;
      pl.onResize = () => { fieldDirty = true; layerDirty = true; if (prevResize) prevResize(); };
      cv.style.touchAction = "none";
      cv.style.cursor = "crosshair";

      // ------------------------------------------------------------ sources & field
      function buildSources() {
        ns = 0;
        for (const c of charges) { sx[ns] = c.x; sy[ns] = c.y; sq[ns] = c.q; sreal[ns] = 1; ns++; }
        if (ground) for (const c of charges) { sx[ns] = c.x; sy[ns] = 2 * YP - c.y; sq[ns] = -c.q; sreal[ns] = 0; ns++; }
        fieldDirty = true;
        rebase = true; // the potential changed: energy drift is measured from here on
      }
      let fEx = 0, fEy = 0, fV = 0;
      function fieldAt(x, y) {
        let ex = 0, ey = 0, v = 0;
        for (let k = 0; k < ns; k++) {
          const dx = x - sx[k], dy = y - sy[k], r2 = dx * dx + dy * dy + 1e-12, ir = 1 / Math.sqrt(r2), f = sq[k] * ir * ir * ir;
          ex += f * dx; ey += f * dy; v += sq[k] * ir;
        }
        fEx = ex; fEy = ey; fV = v;
      }

      function computeGrid() {
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        gx0 = x0; gy0 = y0; hx = (x1 - x0) / (NX - 1); hy = (y1 - y0) / (NY - 1);
        const d = img.data;
        for (let j = 0; j < NY; j++) {
          const y = gy0 + j * hy, row = (NY - 1 - j) * NX;
          for (let i = 0; i < NX; i++) {
            const x = gx0 + i * hx;
            let v = 0;
            if (!(ground && y < YP)) {
              for (let k = 0; k < ns; k++) { const dx = x - sx[k], dy = y - sy[k]; v += sq[k] / Math.sqrt(dx * dx + dy * dy + 1e-8); }
            }
            const vc = v > VCLIP ? VCLIP : v < -VCLIP ? -VCLIP : v;
            const s = Math.asinh(vc / V0);
            S[j * NX + i] = s;
            const li = Math.max(0, Math.min(511, Math.round(255.5 + (s / smax) * 255.5))) * 3, p = (row + i) * 4;
            d[p] = LUT[li]; d[p + 1] = LUT[li + 1]; d[p + 2] = LUT[li + 2]; d[p + 3] = 255;
          }
        }
        offCtx.putImageData(img, 0, 0);
      }

      const EX = new Float64Array(4), EY = new Float64Array(4), sig = new Float64Array(160);
      // marching squares on S for one level; zero-level segments go to a separate buffer (drawn dashed)
      function contourLevel(level, buf, cap, n) {
        for (let j = 0; j < NY - 1; j++) {
          const yb = gy0 + j * hy;
          for (let i = 0; i < NX - 1; i++) {
            const p = j * NX + i;
            const a = S[p] - level, b = S[p + 1] - level, c = S[p + NX + 1] - level, dd = S[p + NX] - level;
            const idx = (a > 0 ? 1 : 0) | (b > 0 ? 2 : 0) | (c > 0 ? 4 : 0) | (dd > 0 ? 8 : 0);
            if (idx === 0 || idx === 15) continue;
            if (n + 8 > cap) return n;
            const xl = gx0 + i * hx;
            // edge points: 0 bottom, 1 right, 2 top, 3 left
            const e0x = xl + (a / (a - b)) * hx, e0y = yb;
            const e1x = xl + hx, e1y = yb + (b / (b - c)) * hy;
            const e2x = xl + (dd / (dd - c)) * hx, e2y = yb + hy;
            const e3x = xl, e3y = yb + (a / (a - dd)) * hy;
            let s1 = -1, s2 = -1, s3 = -1, s4 = -1;
            switch (idx) {
              case 1: case 14: s1 = 3; s2 = 0; break;
              case 2: case 13: s1 = 0; s2 = 1; break;
              case 3: case 12: s1 = 3; s2 = 1; break;
              case 4: case 11: s1 = 1; s2 = 2; break;
              case 6: case 9: s1 = 0; s2 = 2; break;
              case 7: case 8: s1 = 3; s2 = 2; break;
              case 5: case 10: {
                const ctr = (a + b + c + dd) / 4 > 0;
                if ((idx === 5) === ctr) { s1 = 0; s2 = 1; s3 = 2; s4 = 3; } else { s1 = 3; s2 = 0; s3 = 1; s4 = 2; }
                break;
              }
            }
            EX[0] = e0x; EX[1] = e1x; EX[2] = e2x; EX[3] = e3x; EY[0] = e0y; EY[1] = e1y; EY[2] = e2y; EY[3] = e3y;
            const ex = EX, ey = EY;
            buf[n++] = ex[s1]; buf[n++] = ey[s1]; buf[n++] = ex[s2]; buf[n++] = ey[s2];
            if (s3 >= 0) { buf[n++] = ex[s3]; buf[n++] = ey[s3]; buf[n++] = ex[s4]; buf[n++] = ey[s4]; }
          }
        }
        return n;
      }
      function computeContours() {
        nseg = 0; nsegZero = 0;
        const dl = smax / (NLEV + 1);
        for (let k = -NLEV; k <= NLEV; k++) {
          if (k === 0) nsegZero = contourLevel(1e-9, segZero, segZero.length, 0);
          else nseg = contourLevel(k * dl, seg, SEG_CAP, nseg);
        }
      }

      // unit field direction at (x,y) times sgn; returns |E|
      let dX = 0, dY = 0;
      function dirAt(x, y, sgn) {
        fieldAt(x, y);
        const m = Math.hypot(fEx, fEy);
        if (m < 1e-12) { dX = 0; dY = 0; return 0; }
        dX = (sgn * fEx) / m; dY = (sgn * fEy) / m;
        return m;
      }
      // traces one field line into lineBuf at offset o; returns {n, term}
      let traceN = 0, traceTerm = "";
      function trace(x, y, sgn, startK, o, xmin, xmax, ymin, ymax) {
        let n = 0;
        lineBuf[o + n++] = x; lineBuf[o + n++] = y;
        traceTerm = "steps";
        for (let s = 0; s < MAXSTEP; s++) {
          if (o + n + 4 > LINE_CAP) { traceTerm = "cap"; break; }
          if (dirAt(x, y, sgn) === 0) { traceTerm = "null"; break; }
          const a1x = dX, a1y = dY;
          dirAt(x + 0.5 * DS * a1x, y + 0.5 * DS * a1y, sgn); const a2x = dX, a2y = dY;
          dirAt(x + 0.5 * DS * a2x, y + 0.5 * DS * a2y, sgn); const a3x = dX, a3y = dY;
          dirAt(x + DS * a3x, y + DS * a3y, sgn); const a4x = dX, a4y = dY;
          const nx = x + (DS / 6) * (a1x + 2 * a2x + 2 * a3x + a4x), ny = y + (DS / 6) * (a1y + 2 * a2y + 2 * a3y + a4y);
          if (ground && ny < YP) {
            const f = (y - YP) / (y - ny);
            lineBuf[o + n++] = x + f * (nx - x); lineBuf[o + n++] = YP; traceTerm = "plane"; break;
          }
          x = nx; y = ny;
          lineBuf[o + n++] = x; lineBuf[o + n++] = y;
          if (x < xmin || x > xmax || y < ymin || y > ymax) { traceTerm = "edge"; break; }
          let hit = -1;
          for (let k = 0; k < ns; k++) {
            if (k === startK && s < 10) continue;
            const ddx = x - sx[k], ddy = y - sy[k];
            if (ddx * ddx + ddy * ddy < R0 * R0 * 1.6) { hit = k; break; }
          }
          if (hit >= 0) { lineBuf[o + n++] = sx[hit]; lineBuf[o + n++] = sy[hit]; traceTerm = sq[hit] > 0 ? "pos" : "neg"; break; }
        }
        traceN = n;
      }
      function computeLines() {
        nLines = 0; nArw = 0;
        let o = 0;
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        const mx = 0.15 * (x1 - x0), my = 0.15 * (y1 - y0);
        const box = [x0 - mx, x1 + mx, y0 - my, y1 + my];
        for (const pass of [1, -1]) {
          for (let k = 0; k < ns; k++) {
            if (!sreal[k] || sq[k] * pass <= 0) continue;
            const nl = Math.max(1, Math.round(P.density * Math.abs(sq[k])));
            for (let j = 0; j < nl && nLines < lineOff.length; j++) {
              const th = (2 * Math.PI * j) / nl;
              trace(sx[k] + R0 * Math.cos(th), sy[k] + R0 * Math.sin(th), pass, k, o, box[0], box[1], box[2], box[3]);
              if (traceN < 4) continue;
              if (pass === -1) {
                if (traceTerm === "pos") continue; // already drawn from the positive charge
                // reverse in place so that every stored line runs along +E
                for (let a = 0, b = traceN - 2; a < b; a += 2, b -= 2) {
                  let t = lineBuf[o + a]; lineBuf[o + a] = lineBuf[o + b]; lineBuf[o + b] = t;
                  t = lineBuf[o + a + 1]; lineBuf[o + a + 1] = lineBuf[o + b + 1]; lineBuf[o + b + 1] = t;
                }
              }
              lineOff[nLines] = o; lineLen[nLines] = traceN / 2; nLines++;
              { const ao = o >> 1; arcBuf[ao] = 0; for (let q = 1; q < traceN / 2; q++) arcBuf[ao + q] = arcBuf[ao + q - 1] + Math.hypot(lineBuf[o + 2 * q] - lineBuf[o + 2 * q - 2], lineBuf[o + 2 * q + 1] - lineBuf[o + 2 * q - 1]); }
              // arrowheads every 1.6 units of arc length
              let acc = 0.8;
              for (let q = 2; q < traceN - 2 && nArw < ARW_CAP; q += 2) {
                const ddx = lineBuf[o + q] - lineBuf[o + q - 2], ddy = lineBuf[o + q + 1] - lineBuf[o + q - 1];
                acc -= Math.hypot(ddx, ddy);
                if (acc <= 0) { arw[3 * nArw] = lineBuf[o + q]; arw[3 * nArw + 1] = lineBuf[o + q + 1]; arw[3 * nArw + 2] = Math.atan2(ddy, ddx); nArw++; acc = 1.6; }
              }
              o += traceN;
            }
          }
        }
      }
      function computeArrowGrid() {
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        let lmax = -Infinity, lmin = Infinity;
        for (let j = 0; j < AGY; j++) for (let i = 0; i < AGX; i++) {
          const p = j * AGX + i, x = x0 + ((i + 0.5) / AGX) * (x1 - x0), y = y0 + ((j + 0.5) / AGY) * (y1 - y0);
          agx[p] = x; agy[p] = y;
          fieldAt(x, y);
          const m = Math.hypot(fEx, fEy);
          agu[p] = m > 0 ? fEx / m : 0; agv[p] = m > 0 ? fEy / m : 0;
          aga[p] = Math.log10(m + 1e-12);
          if (ground && y < YP) aga[p] = NaN;
          else { lmax = Math.max(lmax, aga[p]); lmin = Math.min(lmin, aga[p]); }
        }
        const hiL = Math.min(lmax, lmin + 3);
        for (let p = 0; p < AGX * AGY; p++) aga[p] = isFinite(aga[p]) ? PM.clamp((aga[p] - (hiL - 2.5)) / 2.5, 0.12, 1) : 0;
      }
      function gaussFlux() {
        if (!charges.length) return { phi: NaN, qenc: NaN };
        const c = charges[Math.min(sel, charges.length - 1)], R = P.gaussR;
        let phi = 0;
        for (let a = 0; a < GL_N; a++) {
          const u = GL.x[a], st = Math.sqrt(1 - u * u);
          for (let b = 0; b < PHI_N; b++) {
            const ph = (2 * Math.PI * (b + 0.5)) / PHI_N, nx = st * Math.cos(ph), ny = st * Math.sin(ph), nz = u;
            const X = c.x + R * nx, Y = c.y + R * ny, Z = R * nz;
            let en = 0;
            for (let k = 0; k < ns; k++) {
              const dx = X - sx[k], dy = Y - sy[k], r2 = dx * dx + dy * dy + Z * Z, f = sq[k] / (r2 * Math.sqrt(r2));
              en += f * (dx * nx + dy * ny + Z * nz);
            }
            phi += GL.w[a] * en;
          }
        }
        phi *= (2 * Math.PI / PHI_N) * R * R;
        let qenc = 0;
        for (let k = 0; k < ns; k++) if (Math.hypot(sx[k] - c.x, sy[k] - c.y) < R) qenc += sq[k];
        return { phi: phi / (4 * Math.PI), qenc };
      }
      let gaussCache = null;

      // ------------------------------------------------------------ test charge
      function deriv(s, out) {
        let ax = 0, ay = 0;
        for (let k = 0; k < ns; k++) {
          const dx = s[0] - sx[k], dy = s[1] - sy[k], r2 = dx * dx + dy * dy + EPS * EPS, f = sq[k] / (r2 * Math.sqrt(r2));
          ax += f * dx; ay += f * dy;
        }
        out[0] = s[2]; out[1] = s[3]; out[2] = P.qt * ax; out[3] = P.qt * ay;
      }
      function energy(s) {
        let u = 0;
        for (let k = 0; k < ns; k++) { const dx = s[0] - sx[k], dy = s[1] - sy[k]; u += sq[k] / Math.sqrt(dx * dx + dy * dy + EPS * EPS); }
        return 0.5 * (s[2] * s[2] + s[3] * s[3]) + P.qt * u;
      }
      function rk4(h) {
        deriv(tc, k1);
        for (let i = 0; i < 4; i++) tmp[i] = tc[i] + 0.5 * h * k1[i];
        deriv(tmp, k2);
        for (let i = 0; i < 4; i++) tmp[i] = tc[i] + 0.5 * h * k2[i];
        deriv(tmp, k3);
        for (let i = 0; i < 4; i++) tmp[i] = tc[i] + h * k3[i];
        deriv(tmp, k4);
        for (let i = 0; i < 4; i++) tc[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      }
      function pushTrail() {
        trX[trHead] = tc[0]; trY[trHead] = tc[1];
        trHead = (trHead + 1) % TRAIL; if (trN < TRAIL) trN++;
      }
      function releaseTest() {
        const a = (P.ang * Math.PI) / 180;
        tc[0] = launch.x; tc[1] = launch.y; tc[2] = P.v0 * Math.cos(a); tc[3] = P.v0 * Math.sin(a);
        tcActive = true; tcState = "moving"; tcT = 0; trN = 0; trHead = 0; tcDrift = 0;
        tcE0 = energy(tc); rebase = false;
        pushTrail();
      }
      function advanceTest(dt) {
        if (!tcActive || tcState !== "moving") return;
        if (rebase) { tcE0 = energy(tc); rebase = false; }
        let rem = dt, iter = 0, lastPush = 0;
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim, W = x1 - x0, H = y1 - y0;
        while (rem > 1e-12 && iter++ < 6000) {
          let rmin = Infinity, hitK = -1;
          for (let k = 0; k < ns; k++) { const r = Math.hypot(tc[0] - sx[k], tc[1] - sy[k]); if (r < rmin) { rmin = r; hitK = k; } }
          if (hitK >= 0 && rmin < 0.06 && sq[hitK] * P.qt < 0 && sreal[hitK]) { tcState = "captured"; break; }
          const v = Math.hypot(tc[2], tc[3]);
          const r = Math.max(rmin, EPS);
          let h = 0.04 * Math.min(r / (v + 0.2), Math.pow(r, 1.5));
          h = PM.clamp(h, 1e-5, 0.004);
          if (h > rem) h = rem;
          rk4(h); rem -= h; tcT += h;
          lastPush += h;
          if (lastPush > 0.01) { pushTrail(); lastPush = 0; }
          if (ground && tc[1] < YP) { tc[1] = YP; tcState = "absorbed by the conductor"; break; }
          if (tc[0] < x0 - 0.5 * W || tc[0] > x1 + 0.5 * W || tc[1] < y0 - 0.5 * H || tc[1] > y1 + 0.5 * H) { tcState = "left the region"; break; }
        }
        pushTrail();
        const E = energy(tc);
        tcDrift = Math.abs(E - tcE0) / Math.max(Math.abs(tcE0), 1e-3);
      }

      // ------------------------------------------------------------ interaction
      function toData(e) {
        const r = cv.getBoundingClientRect();
        const px = e.clientX - r.left, py = e.clientY - r.top;
        return { px, py, x: pl.invX(px), y: pl.invY(py) };
      }
      function pickCharge(px, py) {
        let best = -1, bd = 16;
        charges.forEach((c, k) => { const d = Math.hypot(pl.X(c.x) - px, pl.Y(c.y) - py); if (d < bd) { bd = d; best = k; } });
        return best;
      }
      function select(k) {
        sel = k;
        if (charges[k]) api.setControl("qsel", { value: charges[k].q });
        gaussCache = null;
      }
      function clampPos(c) {
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        c.x = PM.clamp(c.x, x0 + 0.1, x1 - 0.1);
        c.y = PM.clamp(c.y, (ground ? YP + 0.2 : y0 + 0.1), y1 - 0.1);
      }
      const onDown = (e) => {
        const d = toData(e);
        const k = pickCharge(d.px, d.py);
        layerDirty = true;
        if (k >= 0) { select(k); drag = k; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; }
        else if (!(ground && d.y < YP)) { launch = { x: d.x, y: d.y }; }
        cursor = d;
        api.invalidate();
      };
      const onMove = (e) => {
        const d = toData(e);
        cursor = d;
        if (drag >= 0 && charges[drag]) {
          charges[drag].x = d.x; charges[drag].y = d.y; clampPos(charges[drag]);
          buildSources(); gaussCache = null; layerDirty = true;
        } else cv.style.cursor = pickCharge(d.px, d.py) >= 0 ? "grab" : "crosshair";
        api.invalidate();
      };
      const onUp = () => { drag = -1; cv.style.cursor = "crosshair"; };
      const onLeave = () => { if (drag < 0) { cursor = null; api.invalidate(); } };
      cv.addEventListener("pointerdown", onDown);
      cv.addEventListener("pointermove", onMove);
      cv.addEventListener("pointerup", onUp);
      cv.addEventListener("pointercancel", onUp);
      cv.addEventListener("pointerleave", onLeave);

      function loadPreset() {
        ground = P.preset === "ground";
        charges = presetCharges(P.preset, rng);
        loaded = P.preset;
        launch = ground ? { x: -2.5, y: 1.2 } : P.preset === "cap" ? { x: -2.0, y: 0.6 } : P.preset === "single" ? { x: -2.5, y: 1.5 } : { x: -2.6, y: 1.3 };
        select(0);
        buildSources();
      }
      function addCharge(sign) {
        if (charges.length >= MAXREAL) { api.status("At most " + MAXREAL + " charges."); return; }
        const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
        let c = null;
        for (let t = 0; t < 200; t++) {
          const x = rng.uniform(x0 + 0.8, x1 - 0.8), y = rng.uniform(ground ? YP + 0.6 : y0 + 0.6, y1 - 0.6);
          if (!charges.some((o) => Math.hypot(o.x - x, o.y - y) < 0.9)) { c = { x, y, q: sign }; break; }
        }
        if (!c) c = { x: 0, y: 0.5, q: sign };
        charges.push(c);
        select(charges.length - 1);
        buildSources();
      }

      // ------------------------------------------------------------ drawing helpers
      function drawCharges() {
        pl.custom((ctx, p) => {
          // image charges (ghosts)
          if (ground) for (const c of charges) {
            const X = p.X(c.x), Y = p.Y(2 * YP - c.y), r = 7 + 3 * Math.sqrt(Math.abs(c.q));
            ctx.setLineDash([3, 3]); ctx.strokeStyle = c.q > 0 ? "rgba(120,170,255,0.7)" : "rgba(255,140,120,0.7)"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = "rgba(230,237,243,0.7)"; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(c.q > 0 ? "−" : "+", X, Y); ctx.fillText("image", X, Y + r + 9);
          }
          charges.forEach((c, k) => {
            const X = p.X(c.x), Y = p.Y(c.y), r = 7 + 3 * Math.sqrt(Math.abs(c.q));
            const col = c.q > 0 ? "#ff5a4a" : c.q < 0 ? "#4a8dff" : "#8b98a8";
            const g = ctx.createRadialGradient(X, Y, r * 0.6, X, Y, r * 2.4);
            g.addColorStop(0, c.q > 0 ? "rgba(255,90,74,0.55)" : c.q < 0 ? "rgba(74,141,255,0.55)" : "rgba(139,152,168,0.4)");
            g.addColorStop(1, "rgba(0,0,0,0)");
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X, Y, r * 2.4, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X, Y, r, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = "rgba(255,255,255,0.85)"; ctx.lineWidth = 1.2; ctx.stroke();
            ctx.fillStyle = "#fff"; ctx.font = "bold 13px system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
            ctx.fillText(c.q > 0 ? "+" : c.q < 0 ? "−" : "0", X, Y + 0.5);
            if (k === sel) {
              ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6; ctx.setLineDash([4, 3]);
              ctx.beginPath(); ctx.arc(X, Y, r + 5, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
            }
          });
        });
      }

      return {
        reset() {
          if (loaded !== P.preset || !charges.length) loadPreset();
          layerDirty = true;
          simTime = 0;
          releaseTest();
        },
        onParam(id, v) {
          layerDirty = true;
          if (id === "qsel" && charges[sel]) { charges[sel].q = v; buildSources(); gaussCache = null; }
          if (id === "density") fieldDirty = true;
          if (id === "qt") rebase = true;
          if (id === "gaussR") gaussCache = null;
          if (id === "showArrows") fieldDirty = true;
        },
        onAction(id) {
          layerDirty = true;
          if (id === "release") { releaseTest(); api.play(); }
          else if (id === "clear") { tcActive = false; trN = 0; }
          else if (id === "reload") { loadPreset(); releaseTest(); }
          else if (id === "addPos") addCharge(1);
          else if (id === "addNeg") addCharge(-1);
          else if (id === "remove") {
            if (charges.length > 1) { charges.splice(sel, 1); select(Math.min(sel, charges.length - 1)); buildSources(); }
            else api.status("At least one charge must remain.");
          }
        },
        step(dt) {
          simTime += dt;
          flowPhase += dt * 30;
          advanceTest(dt);
        },
        render() {
          if (fieldDirty) {
            computeGrid(); computeContours();
            computeLines(); computeArrowGrid();
            fieldDirty = false; gaussCache = null; layerDirty = true;
          }
          const [x0, x1] = pl.visibleXlim, [y0, y1] = pl.visibleYlim;
          if (layerDirty) {
          pl.clear();
          // potential map
          if (P.showHeat) pl.custom((ctx, p) => {
            ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
            ctx.drawImage(off, p.X(gx0 - hx / 2), p.Y(gy0 + (NY - 0.5) * hy), NX * hx * p.sx, NY * hy * p.sy);
          });
          // conductor + induced surface charge
          if (ground) {
            pl.rect(x0 - 1, y0 - 1, x1 + 1, YP, { color: "#262f3b", alpha: 1 });
            pl.custom((ctx, p) => {
              ctx.strokeStyle = "rgba(139,152,168,0.35)"; ctx.lineWidth = 1; ctx.beginPath();
              const Y0 = p.Y(YP), Y1 = p.Y(y0);
              for (let X = p.X(x0) - 300; X < p.X(x1); X += 12) { ctx.moveTo(X, Y1); ctx.lineTo(X + (Y1 - Y0), Y0); }
              ctx.stroke();
              // σ = ε0 E_y(x, YP+) = E_y/(4π)
              const n = 160;
              let smaxS = 1e-9;
              for (let i = 0; i < n; i++) { fieldAt(x0 + ((i + 0.5) / n) * (x1 - x0), YP + 1e-6); sig[i] = fEy / (4 * Math.PI); smaxS = Math.max(smaxS, Math.abs(sig[i])); }
              const w = (p.X(x1) - p.X(x0)) / n + 0.6;
              for (let i = 0; i < n; i++) {
                const t = Math.min(1, Math.sqrt(Math.abs(sig[i]) / smaxS));
                ctx.fillStyle = sig[i] < 0 ? `rgba(80,150,255,${t})` : `rgba(255,90,70,${t})`;
                ctx.fillRect(p.X(x0) + (i * (p.X(x1) - p.X(x0))) / n, Y0, w, 7);
              }
              ctx.strokeStyle = "#c9d1d9"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p.X(x0), Y0); ctx.lineTo(p.X(x1), Y0); ctx.stroke();
              ctx.fillStyle = "#c9d1d9"; ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "top";
              ctx.fillText("grounded conductor (V = 0) — strip: induced charge σ", p.X(x1) - 8, Y0 + 12);
            });
          }
          // equipotentials
          if (P.showEq) pl.custom((ctx, p) => {
            ctx.strokeStyle = "rgba(230,237,243,0.38)"; ctx.lineWidth = 0.9;
            ctx.beginPath();
            for (let s = 0; s < nseg; s += 4) { ctx.moveTo(p.X(seg[s]), p.Y(seg[s + 1])); ctx.lineTo(p.X(seg[s + 2]), p.Y(seg[s + 3])); }
            ctx.stroke();
            ctx.strokeStyle = "rgba(230,237,243,0.6)"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.1;
            ctx.beginPath();
            for (let s = 0; s < nsegZero; s += 4) { ctx.moveTo(p.X(segZero[s]), p.Y(segZero[s + 1])); ctx.lineTo(p.X(segZero[s + 2]), p.Y(segZero[s + 3])); }
            ctx.stroke(); ctx.setLineDash([]);
          });
          // arrow grid
          if (P.showArrows) pl.custom((ctx, p) => {
            ctx.lineWidth = 1.3;
            const L = 0.36 * Math.min((x1 - x0) / AGX, (y1 - y0) / AGY) * p.sx;
            for (let q = 0; q < AGX * AGY; q++) {
              if (aga[q] <= 0) continue;
              const X = p.X(agx[q]), Y = p.Y(agy[q]), u = agu[q], v = -agv[q];
              ctx.strokeStyle = ctx.fillStyle = `rgba(165,214,255,${aga[q]})`;
              ctx.beginPath(); ctx.moveTo(X - u * L, Y - v * L); ctx.lineTo(X + u * L, Y + v * L); ctx.stroke();
              const hx2 = X + u * L, hy2 = Y + v * L;
              ctx.beginPath(); ctx.moveTo(hx2, hy2); ctx.lineTo(hx2 - 5 * u + 3 * v, hy2 - 5 * v - 3 * u); ctx.lineTo(hx2 - 5 * u - 3 * v, hy2 - 5 * v + 3 * u); ctx.closePath(); ctx.fill();
            }
          });
          // field lines
          if (P.showLines) pl.custom((ctx, p) => {
            ctx.lineJoin = "round"; ctx.lineCap = "round";
            const path = () => {
              ctx.beginPath();
              for (let l = 0; l < nLines; l++) {
                const o = lineOff[l], n = lineLen[l];
                ctx.moveTo(p.X(lineBuf[o]), p.Y(lineBuf[o + 1]));
                for (let q = 1; q < n; q++) ctx.lineTo(p.X(lineBuf[o + 2 * q]), p.Y(lineBuf[o + 2 * q + 1]));
              }
            };
            path();
            ctx.strokeStyle = "rgba(255,244,214,0.55)"; ctx.lineWidth = 1.3; ctx.stroke();
            ctx.fillStyle = "rgba(255,244,214,0.9)";
            ctx.beginPath();
            for (let a = 0; a < nArw; a++) {
              const X = p.X(arw[3 * a]), Y = p.Y(arw[3 * a + 1]), th = -arw[3 * a + 2];
              ctx.moveTo(X + 6 * Math.cos(th), Y + 6 * Math.sin(th));
              ctx.lineTo(X + 6 * Math.cos(th + 2.5), Y + 6 * Math.sin(th + 2.5));
              ctx.lineTo(X + 6 * Math.cos(th - 2.5), Y + 6 * Math.sin(th - 2.5));
              ctx.closePath();
            }
            ctx.fill();
          });
          // Gauss sphere equator
          if (!gaussCache) gaussCache = gaussFlux();
          const cs = charges[Math.min(sel, charges.length - 1)];
          if (cs) {
            const ok = Math.abs(gaussCache.phi - gaussCache.qenc) < 1e-3 * Math.max(1, Math.abs(gaussCache.qenc));
            pl.circle(cs.x, cs.y, P.gaussR, { fill: false, stroke: ok ? "rgba(63,185,80,0.95)" : "rgba(245,158,11,0.95)", strokeWidth: 1.6 });
            pl.custom((ctx, p) => {
              ctx.strokeStyle = "rgba(0,0,0,0)";
              ctx.fillStyle = ok ? "#3fb950" : "#f59e0b"; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
              ctx.fillText("Gauss sphere", p.X(cs.x + P.gaussR * 0.72) + 3, p.Y(cs.y + P.gaussR * 0.72) - 2);
            });
          }
          // launch marker
          pl.custom((ctx, p) => {
            const X = p.X(launch.x), Y = p.Y(launch.y);
            ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(X - 7, Y); ctx.lineTo(X + 7, Y); ctx.moveTo(X, Y - 7); ctx.lineTo(X, Y + 7); ctx.stroke();
            if (P.v0 > 0) {
              const a = (P.ang * Math.PI) / 180, L = 14 + 10 * P.v0;
              ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + L * Math.cos(a), Y - L * Math.sin(a)); ctx.stroke(); ctx.setLineDash([]);
            }
          });
          // snapshot of the static layer
          layer.width = cv.width; layer.height = cv.height;
          layerCtx.drawImage(cv, 0, 0);
          layerDirty = false;
          } else {
            pl.clear();
            pl.custom((ctx) => { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.drawImage(layer, 0, 0); ctx.restore(); });
          }
          // flowing dots along the field lines (direction of E)
          if (P.showLines && P.flow) pl.custom((ctx, p) => {
            ctx.fillStyle = "rgba(255,252,240,0.95)";
            ctx.beginPath();
            const sp = 0.45, ph = (flowPhase * 0.02) % sp;
            for (let l = 0; l < nLines; l++) {
              const o = lineOff[l], n = lineLen[l], ao = o >> 1;
              let next = ph;
              for (let q = 1; q < n; q++) {
                const s1 = arcBuf[ao + q];
                while (next <= s1) {
                  const s0 = arcBuf[ao + q - 1], f = (next - s0) / (s1 - s0 || 1);
                  const x = lineBuf[o + 2 * q - 2] + f * (lineBuf[o + 2 * q] - lineBuf[o + 2 * q - 2]);
                  const y = lineBuf[o + 2 * q - 1] + f * (lineBuf[o + 2 * q + 1] - lineBuf[o + 2 * q - 1]);
                  const X = p.X(x), Y = p.Y(y);
                  ctx.moveTo(X + 1.9, Y); ctx.arc(X, Y, 1.9, 0, 2 * Math.PI);
                  next += sp;
                }
              }
            }
            ctx.fill();
          });
          drawCharges();
          pl.legend([
            { label: "V > 0", color: "#e0553e", type: "box" },
            { label: "V < 0", color: "#3f7fe0", type: "box" },
            { label: "field line (along E)", color: "#fff4d6" },
            { label: "equipotential", color: "rgba(230,237,243,0.6)" },
          ], "tr");
          // test charge trail and body
          if (tcActive && trN > 1) pl.custom((ctx, p) => {
            const col = P.qt >= 0 ? "255,170,60" : "110,200,255";
            const start = (trHead - trN + TRAIL) % TRAIL;
            const chunks = 8, per = Math.ceil(trN / chunks);
            for (let c = 0; c < chunks; c++) {
              const a = c * per, b = Math.min(trN - 1, (c + 1) * per);
              if (b <= a) continue;
              ctx.strokeStyle = `rgba(${col},${0.15 + (0.85 * (c + 1)) / chunks})`; ctx.lineWidth = 2.2;
              ctx.beginPath();
              for (let q = a; q <= b; q++) { const i = (start + q) % TRAIL, X = p.X(trX[i]), Y = p.Y(trY[i]); if (q === a) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
              ctx.stroke();
            }
            const X = p.X(tc[0]), Y = p.Y(tc[1]);
            ctx.fillStyle = `rgba(${col},0.25)`; ctx.beginPath(); ctx.arc(X, Y, 11, 0, 2 * Math.PI); ctx.fill();
            ctx.fillStyle = `rgb(${col})`; ctx.beginPath(); ctx.arc(X, Y, 5.5, 0, 2 * Math.PI); ctx.fill();
            ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.stroke();
          });

          // probe: cursor > test charge > launch point
          let probe;
          if (cursor && drag < 0) probe = { x: cursor.x, y: cursor.y };
          else if (tcActive) probe = { x: tc[0], y: tc[1] };
          else probe = { x: launch.x, y: launch.y };
          if (cursor && drag < 0) pl.custom((ctx, p) => {
            const X = p.X(probe.x), Y = p.Y(probe.y);
            ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.arc(X, Y, 6, 0, 2 * Math.PI); ctx.stroke();
          });
          pl.hline(probe.y, { color: "rgba(255,255,255,0.35)", dash: [6, 5], width: 1 });
          if (tcActive) pl.label([`test charge: ${tcState}`, `t = ${PM.fmt(tcT, 2)},  v = ${PM.fmt(Math.hypot(tc[2], tc[3]), 3)}`], "bl", { size: 11 });

          // metrics
          fieldAt(probe.x, probe.y);
          const inside = !(ground && probe.y < YP);
          const Em = inside ? Math.hypot(fEx, fEy) : 0, ang = (Math.atan2(fEy, fEx) * 180) / Math.PI;
          let onCharge = false;
          for (let k = 0; k < ns; k++) if (Math.hypot(probe.x - sx[k], probe.y - sy[k]) < R0) onCharge = true;
          if (onCharge) { M.set("E", "∞ (on a charge)"); M.set("V", "±∞ (on a charge)"); }
          else {
            M.set("E", inside ? `${PM.fmt(Em, 3)} ∠${ang.toFixed(0)}°` : "0 (inside conductor)");
            M.set("V", PM.fmt(inside ? fV : 0, 3));
          }
          let Q = 0, px = 0, py = 0;
          for (const c of charges) { Q += c.q; px += c.q * c.x; py += c.q * c.y; }
          M.set("Q", PM.fmt(Q, 2));
          M.set("p", PM.fmt(Math.hypot(px, py), 3) + (Math.abs(Q) > 1e-9 ? " (about origin)" : ""));
          M.set("gauss", `${PM.fmt(gaussCache.phi, 4)} | ${PM.fmt(gaussCache.qenc, 2)}`);
          M.set("tc", tcActive ? (tcDrift > 0 ? PM.fmt(tcDrift, 1) : "0") : "—");
          api.setTime(tcActive ? `t = ${PM.fmt(tcT, 2)}` : "");

          // profile plot
          const pf = plots.pf;
          let vmaxAbs = 0.2;
          for (let i = 0; i < NPF; i++) {
            const x = x0 + ((i + 0.5) / NPF) * (x1 - x0);
            pfx[i] = x;
            if (ground && probe.y < YP) { pfV[i] = 0; pfE[i] = 0; continue; }
            fieldAt(x, probe.y);
            pfV[i] = PM.clamp(fV, -VCLIP, VCLIP); pfE[i] = PM.clamp(fEx, -VCLIP, VCLIP);
          }
          for (let i = 0; i < NPF; i++) vmaxAbs = Math.max(vmaxAbs, Math.min(Math.abs(pfV[i]), 6), Math.min(Math.abs(pfE[i]), 6));
          pf.setLimits([x0, x1], [-vmaxAbs * 1.1, vmaxAbs * 1.1]);
          pf.clear();
          pf.hline(0, { color: "rgba(139,152,168,0.5)" });
          for (const c of charges) if (Math.abs(c.y - probe.y) < 0.15) pf.vline(c.x, { color: c.q > 0 ? "rgba(255,90,74,0.5)" : "rgba(74,141,255,0.5)", dash: [3, 3] });
          pf.line(pfx, pfV, { color: PlotColors.accent, width: 2 });
          pf.line(pfx, pfE, { color: PlotColors.accent3, width: 1.6 });
          pf.legend([{ label: "V(x)", color: PlotColors.accent }, { label: "Eₓ(x)", color: PlotColors.accent3 }], "tr");
          pf.label([`y = ${PM.fmt(probe.y, 2)}`], "tl", { size: 11 });
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
