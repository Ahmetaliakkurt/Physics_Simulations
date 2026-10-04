/* Charged particles in electric and magnetic fields — Lorentz force F = q(E + v×B) integrated with the Boris
   pusher: helices, E×B drift, velocity selector, magnetic bottle, grad-B drift and a mass spectrometer. */
(function () {
  "use strict";

  const TR = 1600;      // trail points per particle
  const NT = 720;       // time-series samples kept for the plots
  const NBIN = 140;     // mass-spectrometer detector bins
  const COLORS = ["#4fd1c5", "#f59e0b", "#f778ba", "#58a6ff", "#3fb950", "#d2a8ff"];
  const UMASS = [1, 2, 0.5, 3, 1.5, 4];           // mass factors, uniform field
  const XB_SPEC = [[1, 1], [-1, 1], [1, 2], [-1, 2], [1, 0.5], [-1, 0.5]];  // (charge sign, mass factor), E×B
  const SEL_V = [1, 0.75, 1.3, 0.9, 1.12, 0.55];  // speeds / (E/B), velocity selector
  const GB_SPEC = [[1, 1], [-1, 1], [1, 2], [-1, 2], [1, 0.5], [-1, 0.5]];
  const MS_MASS = [1, 1.15, 1.3, 1.5, 1.75, 2.1];  // mass spectrometer

  const SCEN = {
    uniform: "Uniform B: helix",
    exb: "Crossed E and B: E×B drift",
    selector: "Velocity selector (Wien filter)",
    mirror: "Magnetic mirror / bottle",
    gradb: "Grad-B drift",
    massspec: "Mass spectrometer",
  };

  App.register({
    id: "charged-particle-fields",
    category: "classical",
    group: "Electromagnetism",
    order: 22,
    title: "Charged Particles in Electric & Magnetic Fields",
    icon: "🌀",
    subtitle: "Several charged particles move under the Lorentz force in six classic field configurations — helices, drifts, a velocity filter, a magnetic bottle and a mass spectrometer — integrated live with the Boris algorithm.",
    notes: [{
      type: "info",
      html: "Drag the 3D view to rotate it (wheel = zoom). Each colour is a different particle species (different $q/m$, speed or pitch angle). " +
        "Blue arrows show $\\mathbf B$, yellow arrows $\\mathbf E$. Units are dimensionless: with $q=m=B=1$ the cyclotron frequency is $\\omega_c=1$.",
    }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "scen", type: "select", label: "Scenario", value: "uniform", rebuild: true,
        options: Object.keys(SCEN).map((k) => ({ value: k, label: SCEN[k] })) },
      { type: "section", label: "Particle" },
      { id: "qs", type: "select", label: "Charge sign", value: "1", options: [{ value: "1", label: "positive (+)" }, { value: "-1", label: "negative (−)" }] },
      { id: "q", type: "slider", label: "Charge magnitude $|q|$", min: 0.25, max: 3, step: 0.25, value: 1 },
      { id: "m", type: "slider", label: "Mass $m$", min: 0.25, max: 5, step: 0.25, value: 1 },
      { id: "v0", type: "slider", label: "Initial speed $v_0$", min: 0.2, max: 3, step: 0.1, value: 1,
        visibleIf: (p) => p.scen !== "selector" },
      { id: "pitch", type: "slider", label: "Pitch angle $\\alpha$ (between $\\mathbf v$ and $\\mathbf B$)", min: 5, max: 90, step: 1, value: 60, fmt: (v) => v + "°",
        visibleIf: (p) => p.scen === "uniform" || p.scen === "mirror" || p.scen === "gradb",
        help: "In the bottle, particle $k$ of $n$ starts with $\\alpha_k = \\alpha\\,k/n$." },
      { id: "np", type: "slider", label: "Number of particles / species", min: 1, max: 6, step: 1, value: 3 },
      { type: "section", label: "Fields" },
      { id: "B", type: "slider", label: "Magnetic field $|\\mathbf B|$ (at the centre)", min: 0.2, max: 3, step: 0.1, value: 1 },
      { id: "E", type: "slider", label: "Electric field $|\\mathbf E|$", min: 0, max: 2, step: 0.05, value: 0.4,
        visibleIf: (p) => p.scen === "exb" || p.scen === "selector" },
      { id: "Rm", type: "slider", label: "Mirror ratio $R_m=B_{\\max}/B_{\\min}$", min: 1.5, max: 10, step: 0.5, value: 4, visibleIf: (p) => p.scen === "mirror" },
      { id: "Lg", type: "slider", label: "Gradient length $L_\\nabla / r_L$", min: 3, max: 40, step: 1, value: 10, visibleIf: (p) => p.scen === "gradb",
        help: "$B_z = B_0(1 + x/L_\\nabla)$; the drift formula needs $r_L \\ll L_\\nabla$." },
      { type: "section", label: "Integrator" },
      { id: "wdt", type: "slider", label: "Time step $\\omega_c\\Delta t$", min: 0.02, max: 1, step: 0.02, value: 0.1 },
      { id: "rk4", type: "checkbox", label: "Also integrate particle 1 with RK4 (dashed, for comparison)", value: false,
        visibleIf: (p) => p.scen === "uniform" || p.scen === "mirror" || p.scen === "gradb" },
    ],
    theory: (p) => `
      <h4>The physical system</h4>
      <p>Non-relativistic point charges (charge $q$, mass $m$) moving in prescribed static fields $\\mathbf E(\\mathbf r)$, $\\mathbf B(\\mathbf r)$.
      The particles do not interact with each other and do not radiate; their own fields are neglected. Units are dimensionless:
      charge, mass and field strengths are measured in arbitrary reference units, time in units of $1/\\omega_{c0}$ where
      $\\omega_{c0}$ is the cyclotron frequency for $q=m=B=1$, and lengths in units of $v/\\omega_{c0}$. Current scenario:
      <b>${SCEN[p.scen] || ""}</b>.</p>
      <table>
        <tr><th>Scenario</th><th>Fields</th></tr>
        <tr><td>Uniform B</td><td>$\\mathbf B=B\\hat{\\mathbf z}$, $\\mathbf E=0$</td></tr>
        <tr><td>E×B drift</td><td>$\\mathbf B=B\\hat{\\mathbf z}$, $\\mathbf E=E\\hat{\\mathbf x}$</td></tr>
        <tr><td>Velocity selector</td><td>$\\mathbf B=B\\hat{\\mathbf z}$, $\\mathbf E=E\\hat{\\mathbf y}$ in a slab, beam along $+x$, exit slit</td></tr>
        <tr><td>Magnetic bottle</td><td>$B_z=B_0(1+z^2/L^2)$, $B_{x,y}=-B_0\\,(x,y)\\,z/L^2$ (so $\\nabla\\cdot\\mathbf B=0$)</td></tr>
        <tr><td>Grad-B drift</td><td>$\\mathbf B=B_0(1+x/L_\\nabla)\\hat{\\mathbf z}$</td></tr>
        <tr><td>Mass spectrometer</td><td>$\\mathbf B=B\\hat{\\mathbf z}$ for $y>0$, detector on $y=0$</td></tr>
      </table>

      <h4>Equations being solved</h4>
      <div class="callout">$$ m\\frac{d\\mathbf v}{dt}=q\\left(\\mathbf E+\\mathbf v\\times\\mathbf B\\right),\\qquad \\frac{d\\mathbf r}{dt}=\\mathbf v $$</div>
      <p>The magnetic force does no work ($\\mathbf v\\cdot(\\mathbf v\\times\\mathbf B)=0$), so in a pure magnetic field the kinetic energy is constant.
      In uniform $\\mathbf B$ the motion splits into free streaming along $\\mathbf B$ and a circle across it, with</p>
      $$ \\omega_c=\\frac{|q|B}{m},\\qquad r_L=\\frac{m v_\\perp}{|q|B},\\qquad \\text{pitch}=\\frac{2\\pi v_\\parallel}{\\omega_c}. $$
      <p>Adding a perpendicular $\\mathbf E$ produces the guiding-centre drift $\\mathbf v_E=\\mathbf E\\times\\mathbf B/B^2$, independent of $q$ and $m$.
      The guiding centre is $\\mathbf R=\\mathbf r+m\\,(\\mathbf v\\times\\mathbf B)/(qB^2)$; in uniform fields $\\dot{\\mathbf R}=\\mathbf v_E+v_\\parallel\\hat{\\mathbf b}$ exactly.
      A <b>velocity selector</b> passes undeflected only particles with $qE=qvB$, i.e. $v=E/B$. In a slowly varying field
      ($r_L\\ll$ scale length) the magnetic moment $\\mu=mv_\\perp^2/2B$ is an adiabatic invariant; with energy conservation this gives the
      mirror force $F_\\parallel=-\\mu\\,\\partial B/\\partial s$, reflection where $B=B_0/\\sin^2\\alpha_0$, and the <b>loss cone</b>
      $\\sin^2\\alpha_0\\lt 1/R_m$ of particles that escape. A field gradient gives the <b>grad-B drift</b>
      $\\mathbf v_{\\nabla B}=\\dfrac{mv_\\perp^2}{2qB}\\,\\dfrac{\\mathbf B\\times\\nabla B}{B^2}$, opposite for opposite charges. In the <b>mass spectrometer</b>
      ions of equal speed land at $x=2r_L=2mv/(qB)$, so position measures $m/q$.</p>

      <h4>How the simulation solves them</h4>
      <p>Each particle is advanced with the <b>Boris pusher</b>, the standard particle-in-cell integrator:</p>
      $$ \\mathbf v^-=\\mathbf v^n+\\tfrac{q\\Delta t}{2m}\\mathbf E,\\quad \\mathbf t=\\tfrac{q\\Delta t}{2m}\\mathbf B,\\quad \\mathbf s=\\tfrac{2\\mathbf t}{1+t^2},\\quad
      \\mathbf v'=\\mathbf v^-+\\mathbf v^-\\times\\mathbf t,\\quad \\mathbf v^+=\\mathbf v^-+\\mathbf v'\\times\\mathbf s,$$
      $$ \\mathbf v^{n+1}=\\mathbf v^++\\tfrac{q\\Delta t}{2m}\\mathbf E,\\qquad \\mathbf r^{n+1}=\\mathbf r^n+\\Delta t\\,\\mathbf v^{n+1}. $$
      <p><b>Why Boris:</b> the magnetic part $\\mathbf v^-\\to\\mathbf v^+$ is an exact rotation, so $|\\mathbf v|$ is conserved to round-off in a pure
      magnetic field for <i>any</i> time step — even in the non-uniform bottle field — and the scheme is time-reversible and volume-preserving,
      so errors do not accumulate secularly. Its only error is a phase error: the gyration angle per step is $2\\arctan(\\omega_c\\Delta t/2)$
      instead of $\\omega_c\\Delta t$ (and the discrete orbit radius is enlarged by $\\sqrt{1+(\\omega_c\\Delta t/2)^2}$). Classical RK4, by contrast, is more accurate per step but slowly loses energy (amplitude factor
      $\\approx1-(\\omega_c\\Delta t)^6/144$ per step) — tick the RK4 option and enlarge $\\omega_c\\Delta t$ to see it in the energy plot.</p>
      <ul>
        <li>Each particle has its own step $\\Delta t=(\\omega_c\\Delta t)/\\omega_{c,\\max}$ (using the largest field it can meet), and all particles are
        advanced to the common display time every frame; one reference gyration lasts about 1.6 s of real time at speed 1.</li>
        <li>Open directions are periodic for display (helix along $z$, drifts along $y$) — the particle is wrapped back and its trail broken.</li>
        <li><b>Metrics</b> (particle 1): $\\omega_c$ from the accumulated rotation angle of $\\mathbf v_\\perp-\\mathbf v_E$ divided by elapsed time;
        $r_L$ from half the extent of the orbit across $\\mathbf B$; the drift velocity from the displacement of the guiding centre $\\mathbf R$ divided by time;
        $\\mu$ from the local field. The monitored conserved quantity is the total energy $\\tfrac12mv^2+q\\phi$ ($\\phi=-\\mathbf E\\cdot\\mathbf r$).</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Uniform B:</b> the species with masses $1,2,0.5$ gyrate with $\\omega_c\\propto1/m$ and radii $\\propto m$; the measured $\\omega_c$ matches $|q|B/m$
        to $\\sim(\\omega_c\\Delta t)^2/12$. Set $\\alpha=90^\\circ$ for pure circles.</li>
        <li><b>Energy:</b> with RK4 on and $\\omega_c\\Delta t=0.8$ the RK4 particle spirals inwards and its kinetic energy decays, while Boris stays flat.</li>
        <li><b>E×B:</b> positive and negative, light and heavy particles all drift together at $E/B$ in the $-y$ direction — only the size and sense of the
        loops differ. With $v_0=0.2$ and $E/B=0.4$ the particles start almost at rest in the lab and trace near-cycloids; with large $v_0$ they make wide loops.</li>
        <li><b>Velocity selector:</b> only the species with $v=E/B$ goes straight through the slit; slower ones bend one way, faster ones the other.</li>
        <li><b>Bottle:</b> with $R_m=4$ the loss cone is $\\alpha_{lc}=\\arcsin(1/2)=30^\\circ$: particles with smaller pitch angle escape, larger ones bounce;
        $\\mu$ stays constant to a few per cent while $v_\\perp$ changes a lot.</li>
        <li><b>Mass spectrometer:</b> the peaks on the detector sit at $2mv/(qB)$; doubling $B$ halves all distances.</li>
      </ol>

      <h4>Limitations & further reading</h4>
      <p>Non-relativistic, test particles in fixed fields (no space charge, no radiation, no collisions); the bottle field is the paraxial
      (first-order) model. See D. J. Griffiths, <i>Introduction to Electrodynamics</i>, §5.1; F. F. Chen, <i>Introduction to Plasma Physics and
      Controlled Fusion</i>, ch. 2; J. D. Jackson, <i>Classical Electrodynamics</i>, ch. 12; C. K. Birdsall &amp; A. B. Langdon, <i>Plasma Physics via
      Computer Simulation</i> (Boris pusher); H. Qin et al., Phys. Plasmas 20, 084503 (2013), on why the Boris algorithm works so well.</p>`,

    mount(api) {
      const P = api.params;
      const S0 = P.scen;
      const rng = new PM.RNG(4242);
      let q0 = 1, ell = 1; // charge of the reference species and its Larmor radius at speed v0 (set in reset)
      let S = 1, Zm = 1, Lm = 1, Lg = 1, slab = 1, slitX = 1, slitW = 0.1, ESel = 0;
      let parts = [], T = 0, timeRate = 1;
      let ghost = null;
      const tsT = new Float64Array(NT); let tsN = 0, tsH = 0;
      const hist = new Float64Array(NBIN);
      const selPts = []; let launched = 0, passed = 0;
      const tmpX = new Float64Array(TR + 2), tmpY = new Float64Array(TR + 2);
      const Eo = new Float64Array(3), Bo = new Float64Array(3);

      // ------------------------------------------------------------ fields
      function field(x, y, z, E, B) {
        E[0] = E[1] = E[2] = 0; B[0] = B[1] = 0; B[2] = P.B;
        switch (S0) {
          case "exb": E[0] = P.E; break;
          case "selector": if (Math.abs(x) < slab) E[1] = ESel; else B[2] = 0; break;
          case "mirror": { const f = P.B / (Lm * Lm); B[2] = P.B + f * z * z; B[0] = -f * x * z; B[1] = -f * y * z; break; }
          case "gradb": B[2] = P.B * Math.max(0.05, 1 + x / Lg); break;
          case "massspec": if (y < 0) B[2] = 0; break;
        }
      }
      function setupGeometry() {
        q0 = parseFloat(P.qs) * P.q;
        ell = (P.m * (S0 === "selector" ? (P.E > 0 ? P.E : 0.05) / P.B : P.v0)) / (P.q * P.B);
        switch (S0) {
          case "uniform": break;
          case "exb": S = Math.max(3 * ell * 2, 4 * (P.m * P.E / (P.q * P.B * P.B)) * 2, 3 * ell); break;
          case "selector": ESel = P.E > 0 ? P.E : 0.05; S = 4 * (P.m * (ESel / P.B)) / (P.q * P.B); slab = 0.6 * S; slitX = 0.68 * S; slitW = 0.05 * S; break;
          case "mirror": Zm = 16 * ell; Lm = Zm / Math.sqrt(P.Rm - 1); S = Zm * 1.1; break;
          case "gradb": Lg = P.Lg * ell; S = 6 * ell; break;
          case "massspec": S = 2.3 * ell * MS_MASS[P.np - 1] + 0.5 * ell; break;
        }
        if (S0 === "uniform") {
          let mx = 0; for (let k = 0; k < P.np; k++) mx = Math.max(mx, UMASS[k]);
          S = Math.max(2.3 * ell * Math.sin((P.pitch * Math.PI) / 180) * mx, 2.5 * ell);
        }
        if (S0 === "exb") {
          let mx = 0; for (let k = 0; k < P.np; k++) mx = Math.max(mx, XB_SPEC[k][1]);
          const vE = P.E / P.B, rmax = (P.m * mx * (P.v0 + vE)) / (P.q * P.B);
          S = Math.max(2.4 * rmax, 3 * ell);
        }
      }
      function bmax(p) {
        if (S0 === "mirror") return P.B * P.Rm * 1.05;
        if (S0 === "gradb") return P.B * Math.max(1, 1 + (2 * S) / Lg);
        return P.B;
      }

      // ------------------------------------------------------------ particles
      function newTrail() { return { x: new Float32Array(TR), y: new Float32Array(TR), z: new Float32Array(TR), n: 0, h: 0 }; }
      function push(tr, x, y, z) { tr.x[tr.h] = x; tr.y[tr.h] = y; tr.z[tr.h] = z; tr.h = (tr.h + 1) % TR; if (tr.n < TR) tr.n++; }
      function makeParticle(k) {
        const a = (P.pitch * Math.PI) / 180;
        let q = q0, m = P.m, r = [0, 0, 0], v = [0, 0, 0], label = "";
        switch (S0) {
          case "uniform": m = P.m * UMASS[k]; v = [P.v0 * Math.sin(a), 0, P.v0 * Math.cos(a)]; r = [0, 0, -0.85 * S]; label = `m = ${PM.fmt(m, 2)}`; break;
          case "exb": q = q0 * XB_SPEC[k][0]; m = P.m * XB_SPEC[k][1]; v = [P.v0, 0, 0]; label = `q = ${PM.fmt(q, 2)}, m = ${PM.fmt(m, 2)}`; break;
          case "selector": { const vv = (ESel / P.B) * SEL_V[k]; v = [vv, 0, 0]; r = [-0.98 * S, 0, 0]; label = `v = ${PM.fmt(SEL_V[k], 2)} E/B`; break; }
          case "mirror": { const ak = (a * (k + 1)) / P.np; v = [0, P.v0 * Math.sin(ak), P.v0 * Math.cos(ak)]; r = [0, 0, 0]; label = `α₀ = ${((ak * 180) / Math.PI).toFixed(0)}°`; break; }
          case "gradb": q = q0 * GB_SPEC[k][0]; m = P.m * GB_SPEC[k][1]; v = [P.v0 * Math.sin(a), 0, P.v0 * Math.cos(a)]; r = [0, 0, -0.85 * S]; label = `q = ${PM.fmt(q, 2)}, m = ${PM.fmt(m, 2)}`; break;
          case "massspec": {
            m = P.m * MS_MASS[k];
            const sp = P.v0 * (1 + 0.004 * rng.gauss()), th = 0.02 * rng.gauss();
            v = [sp * Math.sin(th), sp * Math.cos(th), 0]; r = [0, 0, 0]; label = `m = ${PM.fmt(m, 2)}`; break;
          }
        }
        const p = { k, q, m, r: Float64Array.from(r), v: Float64Array.from(v), t: T, color: COLORS[k], label, alive: true, state: "", tr: newTrail(),
          ke: new Float64Array(NT), aux: new Float64Array(NT), r0: Float64Array.from(r) };
        p.dt = P.wdt * m / (Math.abs(q) * bmax(p));
        field(p.r[0], p.r[1], p.r[2], Eo, Bo);
        p.E0 = energy(p);
        p.mu0 = mu(p);
        p.gc0 = gc(p);
        push(p.tr, p.r[0], p.r[1], p.r[2]);
        // diagnostics
        p.ang = vperpAngle(p); p.angAcc = 0; p.tStart = T; p.xmin = p.r[0]; p.xmax = p.r[0]; p.ymin = p.r[1]; p.ymax = p.r[1];
        p.muMin = p.mu0; p.muMax = p.mu0;
        return p;
      }
      function energy(p) {
        const ke = 0.5 * p.m * (p.v[0] * p.v[0] + p.v[1] * p.v[1] + p.v[2] * p.v[2]);
        let phi = 0;
        if (S0 === "exb") phi = -P.E * p.r[0];
        return ke + p.q * phi;
      }
      function mu(p) {
        field(p.r[0], p.r[1], p.r[2], Eo, Bo);
        const b = Math.hypot(Bo[0], Bo[1], Bo[2]) || 1e-12;
        const vb = (p.v[0] * Bo[0] + p.v[1] * Bo[1] + p.v[2] * Bo[2]) / b;
        const v2 = p.v[0] * p.v[0] + p.v[1] * p.v[1] + p.v[2] * p.v[2];
        return (p.m * Math.max(v2 - vb * vb, 0)) / (2 * b);
      }
      function gc(p) {
        field(p.r[0], p.r[1], p.r[2], Eo, Bo);
        const b2 = Bo[0] * Bo[0] + Bo[1] * Bo[1] + Bo[2] * Bo[2] || 1e-12, f = p.m / (p.q * b2);
        const cx = p.v[1] * Bo[2] - p.v[2] * Bo[1], cy = p.v[2] * Bo[0] - p.v[0] * Bo[2], cz = p.v[0] * Bo[1] - p.v[1] * Bo[0];
        return [p.r[0] + f * cx, p.r[1] + f * cy, p.r[2] + f * cz];
      }
      function vperpAngle(p) {
        const vEx = S0 === "exb" ? 0 : 0, vEy = S0 === "exb" ? -P.E / P.B : 0;
        return Math.atan2(p.v[1] - vEy, p.v[0] - vEx);
      }
      function boris(p, h) {
        field(p.r[0], p.r[1], p.r[2], Eo, Bo);
        const c = (p.q * h) / (2 * p.m);
        const vmx = p.v[0] + c * Eo[0], vmy = p.v[1] + c * Eo[1], vmz = p.v[2] + c * Eo[2];
        const tx = c * Bo[0], ty = c * Bo[1], tz = c * Bo[2], t2 = tx * tx + ty * ty + tz * tz;
        const sx = (2 * tx) / (1 + t2), sy = (2 * ty) / (1 + t2), sz = (2 * tz) / (1 + t2);
        const px = vmx + (vmy * tz - vmz * ty), py = vmy + (vmz * tx - vmx * tz), pz = vmz + (vmx * ty - vmy * tx);
        const vpx = vmx + (py * sz - pz * sy), vpy = vmy + (pz * sx - px * sz), vpz = vmz + (px * sy - py * sx);
        p.v[0] = vpx + c * Eo[0]; p.v[1] = vpy + c * Eo[1]; p.v[2] = vpz + c * Eo[2];
        p.r[0] += h * p.v[0]; p.r[1] += h * p.v[1]; p.r[2] += h * p.v[2];
      }
      // RK4 ghost (6-vector)
      const gk = [new Float64Array(6), new Float64Array(6), new Float64Array(6), new Float64Array(6)], gt = new Float64Array(6);
      function gderiv(s, out, q, m) {
        field(s[0], s[1], s[2], Eo, Bo);
        const c = q / m;
        out[0] = s[3]; out[1] = s[4]; out[2] = s[5];
        out[3] = c * (Eo[0] + s[4] * Bo[2] - s[5] * Bo[1]);
        out[4] = c * (Eo[1] + s[5] * Bo[0] - s[3] * Bo[2]);
        out[5] = c * (Eo[2] + s[3] * Bo[1] - s[4] * Bo[0]);
      }
      function rk4Ghost(g, h) {
        const y = g.s;
        gderiv(y, gk[0], g.q, g.m);
        for (let i = 0; i < 6; i++) gt[i] = y[i] + 0.5 * h * gk[0][i];
        gderiv(gt, gk[1], g.q, g.m);
        for (let i = 0; i < 6; i++) gt[i] = y[i] + 0.5 * h * gk[1][i];
        gderiv(gt, gk[2], g.q, g.m);
        for (let i = 0; i < 6; i++) gt[i] = y[i] + h * gk[2][i];
        gderiv(gt, gk[3], g.q, g.m);
        for (let i = 0; i < 6; i++) y[i] += (h / 6) * (gk[0][i] + 2 * gk[1][i] + 2 * gk[2][i] + gk[3][i]);
      }

      function relaunch(k) {
        const p = makeParticle(k);
        parts[k] = p;
        if (S0 === "selector" || S0 === "massspec") launched++;
      }
      function wrap(p) {
        if (S0 === "uniform" || S0 === "gradb") {
          if (p.r[2] > S) { p.r[2] -= 2 * S; p.gc0[2] -= 2 * S; push(p.tr, NaN, NaN, NaN); }
          if (p.r[2] < -S) { p.r[2] += 2 * S; p.gc0[2] += 2 * S; push(p.tr, NaN, NaN, NaN); }
        }
        if (S0 === "exb" || S0 === "gradb") {
          if (p.r[1] > S) { p.r[1] -= 2 * S; p.wrapY = (p.wrapY || 0) + 2 * S; push(p.tr, NaN, NaN, NaN); }
          if (p.r[1] < -S) { p.r[1] += 2 * S; p.wrapY = (p.wrapY || 0) - 2 * S; push(p.tr, NaN, NaN, NaN); }
        }
      }
      function advance(p, Tend) {
        let guard = 0, lastPush = p.t;
        const minGap = (S * 0.004) / Math.max(1e-6, Math.hypot(p.v[0], p.v[1], p.v[2]));
        while (p.t + p.dt <= Tend && guard++ < 20000) {
          const h = p.dt; // fixed step; the remainder is carried over to the next frame
          if (!p.alive) { p.t = Tend; break; }
          const xOld = p.r[0], yOld = p.r[1];
          boris(p, h); p.t += h;
          // rotation-angle bookkeeping (frequency measurement)
          const an = vperpAngle(p); let d = an - p.ang; d = Math.atan2(Math.sin(d), Math.cos(d)); p.angAcc += d; p.ang = an;
          p.xmin = Math.min(p.xmin, p.r[0]); p.xmax = Math.max(p.xmax, p.r[0]);
          if (S0 === "mirror" || S0 === "gradb") { const u = mu(p); p.muMin = Math.min(p.muMin, u); p.muMax = Math.max(p.muMax, u); }
          // scenario boundaries
          if (S0 === "mirror" && Math.abs(p.r[2]) > Zm) { p.alive = false; p.state = "escaped"; }
          if (S0 === "selector") {
            if (xOld < slitX && p.r[0] >= slitX) {
              const ok = Math.abs(p.r[1]) < slitW;
              if (selPts.length < 600) selPts.push([SEL_V[p.k], p.r[1] / S, ok]);
              if (ok) passed++; else { p.alive = false; p.state = "blocked"; p.deadT = T; }
            }
            if (p.r[0] > S || Math.abs(p.r[1]) > S) { p.alive = false; p.deadT = T; }
          }
          if (S0 === "massspec" && yOld >= 0 && p.r[1] < 0 && p.t > p.tStart + 0.1) {
            const b = Math.floor(((p.r[0] + S) / (2 * S)) * NBIN);
            if (b >= 0 && b < NBIN) hist[b]++;
            p.alive = false; p.deadT = T;
          }
          wrap(p);
          if (p.t - lastPush >= minGap || !p.alive) { push(p.tr, p.r[0], p.r[1], p.r[2]); lastPush = p.t; }
        }
      }

      // ------------------------------------------------------------ layout
      const metricSets = {
        uniform: [["wc", "$\\omega_c$: theory $|q|B/m$ | measured"], ["rl", "$r_L$: theory | measured"], ["pitchL", "Pitch $2\\pi v_\\parallel/\\omega_c$"], ["dE", "Boris $|\\Delta K/K|$"], ["dEr", "RK4 $|\\Delta K/K|$"]],
        exb: [["vd", "Drift $v_y$: measured | $-E/B$"], ["wc", "$\\omega_c$: theory | measured"], ["rl", "$r_L$ (drift frame): theory | measured"], ["dE", "Total-energy drift $|\\Delta\\mathcal E/\\mathcal E|$"]],
        selector: [["vsel", "Selected speed $E/B$"], ["trans", "Through the slit / launched"], ["wc", "$\\omega_c$ (reference particle)"], ["rl", "$r_L$ at $v=E/B$"]],
        mirror: [["lc", "Loss cone $\\arcsin\\sqrt{1/R_m}$"], ["trap", "Trapped / total"], ["mu", "$\\mu$ of particle 1: min–max / mean"], ["dE", "Boris $|\\Delta K/K|$"], ["dEr", "RK4 $|\\Delta K/K|$"]],
        gradb: [["vd", "Drift $v_y$ (particle 1): measured | theory"], ["vd2", "Drift $v_y$ (particle 2): measured | theory"], ["mu", "$\\mu$ variation (particle 1)"], ["eps", "Adiabaticity $r_L/L_\\nabla$"], ["dE", "Boris $|\\Delta K/K|$"]],
        massspec: [["pred", "Predicted landing $x=2mv/qB$"], ["meas", "Measured peak positions"], ["hits", "Ions detected"], ["res", "Resolving power $m/\\Delta m$ (adjacent)"]],
      };
      const M = api.metrics(metricSets[S0].map(([id, label]) => ({ id, label })));
      const projInfo = {
        uniform: ["x", "y", "Projection on the plane ⟂ B (x–y)"], exb: ["x", "y", "x–y plane: trochoids drifting along E×B"],
        selector: ["x", "y", "Top view: beam, field slab (shaded) and exit slit"], mirror: ["z", "x", "Side view (z–x): bouncing between the mirrors"],
        gradb: ["x", "y", "x–y plane (B grows to the right)"], massspec: ["x", "y", "Top view: semicircles onto the detector (y = 0)"],
      }[S0];
      const auxTitle = {
        uniform: "Velocity components of particle 1", exb: "Guiding-centre displacement along y vs theory −(E/B)t",
        selector: "Deflection at the slit vs speed", mirror: "Magnetic moment μ(t)/μ(0) — adiabatic invariant",
        gradb: "Guiding-centre displacement along y vs grad-B theory", massspec: "Detector: counts vs landing position",
      }[S0];
      const plots = api.plots([
        { id: "v3", type: "3d", title: "3D trajectories (drag to rotate)", aspect: 0.85, extent: 1, yaw: 0.7, pitch: ["exb", "selector", "massspec"].indexOf(S0) >= 0 ? 0.95 : 0.35 },
        { id: "pr", title: projInfo[2], aspect: 0.85, xlim: [-1, 1], ylim: [-1, 1], equal: true, xlabel: projInfo[0], ylabel: projInfo[1] },
        { id: "en", title: "Kinetic energy K(t) = ½mv²", aspect: 0.55, xlim: [0, 1], ylim: [0, 1], xlabel: "t", ylabel: "K" },
        { id: "ax", title: auxTitle, aspect: 0.55, xlim: [0, 1], ylim: [0, 1], xlabel: S0 === "selector" ? "v / (E/B)" : S0 === "massspec" ? "x" : "t", ylabel: S0 === "selector" ? "y at slit / S" : S0 === "massspec" ? "counts" : S0 === "uniform" ? "v" : S0 === "mirror" ? "μ/μ₀" : "ΔY" },
      ]);

      // ------------------------------------------------------------ drawing helpers
      function drawTrail3(v, tr, color, dash) {
        const c = v.ctx, B = v._basis(), n = tr.n;
        if (n < 2) return;
        const start = (tr.h - n + TR) % TR, chunks = 6, per = Math.ceil(n / chunks);
        c.lineWidth = 1.8; c.lineJoin = "round"; c.setLineDash(dash ? [5, 4] : []);
        for (let ch = 0; ch < chunks; ch++) {
          const a = ch * per, b = Math.min(n - 1, (ch + 1) * per);
          if (b <= a) continue;
          c.strokeStyle = color; c.globalAlpha = 0.18 + (0.82 * (ch + 1)) / chunks;
          c.beginPath(); let pen = false;
          for (let q = a; q <= b; q++) {
            const i = (start + q) % TR;
            if (!isFinite(tr.x[i])) { pen = false; continue; }
            const pp = v.project(tr.x[i], tr.y[i], tr.z[i], B);
            if (pen) c.lineTo(pp[0], pp[1]); else { c.moveTo(pp[0], pp[1]); pen = true; }
          }
          c.stroke();
        }
        c.globalAlpha = 1; c.setLineDash([]);
      }
      function trailProj(tr, ia, ib) {
        const n = tr.n, start = (tr.h - n + TR) % TR, A = [tr.x, tr.y, tr.z];
        for (let q = 0; q < n; q++) { const i = (start + q) % TR; tmpX[q] = A[ia][i]; tmpY[q] = A[ib][i]; }
        return n;
      }
      function fieldArrows3(v) {
        const ext = S;
        const arrowsB = [], arrowsE = [];
        if (S0 === "mirror") {
          // field lines of the bottle: rho(z) = rho0 sqrt(B0/B(z))
          const c = v.ctx, B = v._basis();
          c.strokeStyle = "rgba(88,166,255,0.45)"; c.lineWidth = 1;
          for (const r0 of [0.35 * Zm / 4, 0.7 * Zm / 4]) for (let a = 0; a < 8; a++) {
            const ca = Math.cos((a * Math.PI) / 4), sa = Math.sin((a * Math.PI) / 4);
            c.beginPath();
            for (let i = 0; i <= 60; i++) {
              const z = -Zm + (2 * Zm * i) / 60, rho = r0 / Math.sqrt(1 + (z * z) / (Lm * Lm));
              const pp = v.project(rho * ca, rho * sa, z, B);
              if (i) c.lineTo(pp[0], pp[1]); else c.moveTo(pp[0], pp[1]);
            }
            c.stroke();
          }
          // coils
          for (const zc of [-Zm, Zm]) {
            const R = 0.45 * Zm, xs = [], ys = [], zs = [];
            for (let i = 0; i <= 48; i++) { const t = (2 * Math.PI * i) / 48; xs.push(R * Math.cos(t)); ys.push(R * Math.sin(t)); zs.push(zc); }
            v.line3(xs, ys, zs, { color: "#ffa657", width: 3, alpha: 0.85 });
          }
          v.arrow3(0, 0, -0.3 * Zm, 0, 0, 0.3 * Zm, { color: "#58a6ff", width: 2 });
          return;
        }
        const g = [-0.7, 0, 0.7];
        if (S0 === "massspec") {
          for (const x of g) for (const y of [0.15, 0.5, 0.85]) arrowsB.push([x * ext, y * ext, -0.3 * ext, 0.35 * ext]);
        } else {
          for (const x of g) for (const y of g) {
            let len = 0.35 * ext;
            if (S0 === "gradb") len *= Math.max(0.05, 1 + (x * ext) / Lg);
            if (S0 === "selector" && Math.abs(x * ext) > slab) continue;
            arrowsB.push([x * ext, y * ext, -ext, len]);
          }
        }
        for (const [x, y, z, L] of arrowsB) v.arrow3(x, y, z, x, y, z + L, { color: "rgba(88,166,255,0.8)", width: 1.6 });
        if (S0 === "exb") for (const y of g) for (const z of g) arrowsE.push([-0.95 * ext, y * ext, z * ext, 0, 0.3 * ext]);
        if (S0 === "selector" && P.E > 0) for (const x of [-0.4, 0.4]) for (const z of g) arrowsE.push([x * ext, -0.9 * ext, z * ext, 1, 0.3 * ext]);
        for (const [x, y, z, ax, L] of arrowsE) {
          if (ax === 0) v.arrow3(x, y, z, x + L, y, z, { color: "rgba(245,200,60,0.9)", width: 1.6 });
          else v.arrow3(x, y, z, x, y + L, z, { color: "rgba(245,200,60,0.9)", width: 1.6 });
        }
      }

      return {
        reset() {
          setupGeometry();
          T = 0; tsN = 0; tsH = 0; hist.fill(0); selPts.length = 0; launched = 0; passed = 0;
          parts = [];
          for (let k = 0; k < P.np; k++) parts.push(makeParticle(k));
          const p0 = parts[0];
          const refPeriod = (2 * Math.PI * p0.m) / (Math.abs(p0.q) * P.B);
          timeRate = refPeriod / 1.6;
          // in the selector and spectrometer the particles enter one after another
          if (S0 === "selector" || S0 === "massspec") {
            launched = 1;
            parts.forEach((p, k) => { p.alive = k === 0; p.deadT = -k * 0.25 * timeRate * 1.6; p.state = k === 0 ? "" : "waiting"; });
          }
          ghost = null;
          if (P.rk4 && (S0 === "uniform" || S0 === "mirror" || S0 === "gradb")) {
            ghost = { s: Float64Array.from([p0.r[0], p0.r[1], p0.r[2], p0.v[0], p0.v[1], p0.v[2]]), q: p0.q, m: p0.m, dt: p0.dt, t: 0, tr: newTrail(), ke: new Float64Array(NT), K0: 0.5 * p0.m * P.v0 * P.v0, alive: true };
          }
          const ext = S * 1.05;
          plots.v3.opts.extent = ext;
          const pr = plots.pr;
          if (S0 === "mirror") pr.setLimits([-1.05 * Zm, 1.05 * Zm], [-0.45 * Zm, 0.45 * Zm]);
          else if (S0 === "massspec") pr.setLimits(q0 > 0 ? [-0.3 * S, S] : [-S, 0.3 * S], [-0.2 * S, 1.0 * S]);
          else pr.setLimits([-S, S], [-S, S]);
        },
        step(dt) {
          const Tn = T + dt * timeRate;
          for (let k = 0; k < parts.length; k++) {
            const p = parts[k];
            if (!p.alive) {
              if ((S0 === "selector" || S0 === "massspec") && T - (p.deadT || 0) > 0.55 * timeRate * 1.6) { relaunch(k); parts[k].t = T; }
              continue;
            }
            advance(p, Tn);
          }
          if (ghost && ghost.alive) {
            let guard = 0;
            while (ghost.t + ghost.dt <= Tn && guard++ < 20000) {
              const h = ghost.dt;
              rk4Ghost(ghost, h); ghost.t += h;
              const s = ghost.s;
              if (S0 === "mirror" && Math.abs(s[2]) > Zm) { ghost.alive = false; break; }
              if (s[2] > S) { s[2] -= 2 * S; push(ghost.tr, NaN, NaN, NaN); }
              if (s[2] < -S) { s[2] += 2 * S; push(ghost.tr, NaN, NaN, NaN); }
              if (S0 === "gradb") { if (s[1] > S) { s[1] -= 2 * S; push(ghost.tr, NaN, NaN, NaN); } if (s[1] < -S) { s[1] += 2 * S; push(ghost.tr, NaN, NaN, NaN); } }
              push(ghost.tr, s[0], s[1], s[2]);
            }
          }
          T = Tn;
          // time series
          tsT[tsH] = T;
          for (const p of parts) {
            p.ke[tsH] = 0.5 * p.m * (p.v[0] * p.v[0] + p.v[1] * p.v[1] + p.v[2] * p.v[2]);
            if (S0 === "mirror") p.aux[tsH] = p.alive ? mu(p) / p.mu0 : NaN;
            else if (S0 === "exb" || S0 === "gradb") { const g = gc(p); p.aux[tsH] = g[1] + (p.wrapY || 0) - p.gc0[1]; }
            else if (S0 === "uniform") p.aux[tsH] = p.v[0];
          }
          if (S0 === "uniform") { const p = parts[0]; p.vy = p.vy || new Float64Array(NT); p.vz = p.vz || new Float64Array(NT); p.vy[tsH] = p.v[1]; p.vz[tsH] = p.v[2]; }
          if (ghost) { const s = ghost.s; ghost.ke[tsH] = 0.5 * ghost.m * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]); }
          tsH = (tsH + 1) % NT; if (tsN < NT) tsN++;
        },
        render() {
          // ---------------- 3D
          const v = plots.v3;
          v.clear(); v.box(S * 1.0); v.axes(S * 0.5);
          fieldArrows3(v);
          if (S0 === "selector") {
            const c = v.ctx, B = v._basis();
            const pl = [[slitX, slitW, S], [slitX, -S, -slitW]];
            c.fillStyle = "rgba(200,210,220,0.25)"; c.strokeStyle = "rgba(200,210,220,0.7)";
            for (const [x, y0, y1] of pl) {
              const pts = [[x, y0, -0.4 * S], [x, y1, -0.4 * S], [x, y1, 0.4 * S], [x, y0, 0.4 * S]].map((q) => v.project(q[0], q[1], q[2], B));
              c.beginPath(); pts.forEach((q, i) => (i ? c.lineTo(q[0], q[1]) : c.moveTo(q[0], q[1]))); c.closePath(); c.fill(); c.stroke();
            }
          }
          if (S0 === "massspec") {
            v.line3([0, S], [0, 0], [0, 0], { color: "#c9d1d9", width: 4, alpha: 0.6 });
          }
          if (ghost) drawTrail3(v, ghost.tr, "#e6edf3", true);
          for (const p of parts) drawTrail3(v, p.tr, p.color, false);
          for (const p of parts) if (p.alive || p.state === "escaped" || p.state === "blocked") v.point3(p.r[0], p.r[1], p.r[2], { color: p.alive ? p.color : "#8b98a8", size: 4.5 });
          v.text(`t = ${PM.fmt(T, 2)}`, 10, 18, { color: "#8b98a8" });

          // ---------------- projection
          const pr = plots.pr, [ia, ib] = S0 === "mirror" ? [2, 0] : [0, 1];
          pr.clear();
          if (S0 === "selector") {
            pr.rect(-slab, -S, slab, S, { color: "#f5c83c", alpha: 0.07 });
            pr.rect(slitX, slitW, slitX + 0.03 * S, S, { color: "#c9d1d9", alpha: 0.8 });
            pr.rect(slitX, -S, slitX + 0.03 * S, -slitW, { color: "#c9d1d9", alpha: 0.8 });
          }
          if (S0 === "gradb") {
            for (let i = 0; i < 24; i++) { const x0 = -S + (2 * S * i) / 24; pr.rect(x0, -S, x0 + (2 * S) / 24, S, { color: "#58a6ff", alpha: 0.03 + 0.12 * (i / 23) }); }
          }
          if (S0 === "mirror") {
            for (const r0 of [0.35 * Zm / 4, 0.7 * Zm / 4]) for (const sg of [1, -1])
              pr.fn((z) => (sg * r0) / Math.sqrt(1 + (z * z) / (Lm * Lm)), { color: "rgba(88,166,255,0.5)", width: 1 });
            pr.vline(-Zm, { color: "#ffa657", width: 3 }); pr.vline(Zm, { color: "#ffa657", width: 3 });
          }
          if (S0 === "massspec") {
            pr.hline(0, { color: "#c9d1d9", width: 3 });
            for (let k = 0; k < P.np; k++) { const x = (2 * P.m * MS_MASS[k] * P.v0) / (Math.abs(q0) * P.B) * Math.sign(q0); pr.segment(x, -0.08 * S, x, 0, { color: COLORS[k], width: 2 }); }
          }
          if (S0 === "uniform") for (const p of parts) {
            const r = (p.m * P.v0 * Math.sin((P.pitch * Math.PI) / 180)) / (Math.abs(p.q) * P.B);
            pr.circle(p.gc0[0], p.gc0[1], r, { fill: false, stroke: "rgba(230,237,243,0.35)", strokeWidth: 1 });
          }
          if (ghost) { const n = trailProj(ghost.tr, ia, ib); pr.line(tmpX.subarray(0, n), tmpY.subarray(0, n), { color: "#e6edf3", width: 1.2, dash: [5, 4] }); }
          for (const p of parts) {
            const n = trailProj(p.tr, ia, ib);
            pr.line(tmpX.subarray(0, n), tmpY.subarray(0, n), { color: p.color, width: 1.7 });
            if (p.alive) pr.circle(p.r[ia], p.r[ib], 4.5, { px: true, color: p.color, stroke: "#fff", strokeWidth: 1 });
          }
          pr.legend(parts.map((p) => ({ label: p.label + (p.state === "escaped" ? " — escaped" : ""), color: p.color })), "tr");

          // ---------------- time series
          const n = tsN, start = (tsH - n + NT) % NT;
          const tX = new Float64Array(n);
          for (let q = 0; q < n; q++) tX[q] = tsT[(start + q) % NT];
          const seq = (arr) => { const o = new Float64Array(n); for (let q = 0; q < n; q++) o[q] = arr[(start + q) % NT]; return o; };
          const t0 = n ? tX[0] : 0, t1 = Math.max(n ? tX[n - 1] : 1, t0 + 1e-6);
          const en = plots.en;
          let kmax = 1e-9, kmin = Infinity;
          const keS = parts.map((p) => seq(p.ke));
          for (const a of keS) for (let q = 0; q < n; q++) { kmax = Math.max(kmax, a[q]); kmin = Math.min(kmin, a[q]); }
          const gS = ghost ? seq(ghost.ke) : null;
          if (gS) for (let q = 0; q < n; q++) { kmax = Math.max(kmax, gS[q]); kmin = Math.min(kmin, gS[q]); }
          if (!isFinite(kmin)) kmin = 0;
          const pad = Math.max(0.08 * (kmax - kmin), 0.05 * kmax);
          en.setLimits([t0, t1], [Math.max(0, kmin - pad), kmax + pad]);
          en.clear();
          keS.forEach((a, k) => en.line(tX, a, { color: parts[k].color, width: 1.8 }));
          if (gS) { en.line(tX, gS, { color: "#e6edf3", width: 1.4, dash: [5, 4] }); en.legend([{ label: "Boris (particle 1)", color: parts[0].color }, { label: "RK4 (particle 1)", color: "#e6edf3", dash: [5, 4] }], "br"); }

          const ax = plots.ax;
          if (S0 === "massspec") {
            let hm = 1; for (let b = 0; b < NBIN; b++) hm = Math.max(hm, hist[b]);
            ax.setLimits(q0 > 0 ? [-0.3 * S, S] : [-S, 0.3 * S], [0, hm * 1.15]);
            ax.clear();
            const xs = new Float64Array(NBIN); for (let b = 0; b < NBIN; b++) xs[b] = -S + ((b + 0.5) * 2 * S) / NBIN;
            ax.bars(xs, hist, (2 * S) / NBIN, { color: PlotColors.accent, alpha: 0.8 });
            for (let k = 0; k < P.np; k++) ax.vline((2 * P.m * MS_MASS[k] * P.v0) / (Math.abs(q0) * P.B) * Math.sign(q0), { color: COLORS[k], dash: [4, 3] });
          } else if (S0 === "selector") {
            ax.setLimits([0.4, 1.4], [-1, 1]);
            ax.clear();
            ax.rect(0.4, -slitW / S, 1.4, slitW / S, { color: "#3fb950", alpha: 0.15 });
            ax.vline(1, { color: "rgba(230,237,243,0.5)", dash: [4, 3] });
            for (const [vv, y, ok] of selPts) ax.circle(vv, y, 4, { px: true, color: ok ? "#3fb950" : "#f85149", alpha: 0.8 });
            ax.legend([{ label: "passed the slit", color: "#3fb950", type: "dot" }, { label: "blocked", color: "#f85149", type: "dot" }], "tr");
          } else if (S0 === "uniform") {
            const p = parts[0];
            const a = seq(p.aux), b = p.vy ? seq(p.vy) : a, c = p.vz ? seq(p.vz) : a;
            const vm = Math.max(P.v0, 1e-3) * 1.15;
            ax.setLimits([t0, t1], [-vm, vm]);
            ax.clear();
            ax.line(tX, a, { color: "#f78166", width: 1.6 }); ax.line(tX, b, { color: "#7ee787", width: 1.6 }); ax.line(tX, c, { color: "#79c0ff", width: 2 });
            ax.legend([{ label: "vₓ", color: "#f78166" }, { label: "v_y", color: "#7ee787" }, { label: "v_z", color: "#79c0ff" }], "br");
          } else {
            const auxS = parts.map((p) => seq(p.aux));
            let lo = Infinity, hi = -Infinity;
            for (const a of auxS) for (let q = 0; q < n; q++) if (isFinite(a[q])) { lo = Math.min(lo, a[q]); hi = Math.max(hi, a[q]); }
            if (S0 === "mirror") { lo = Math.min(lo, 0.9); hi = Math.max(hi, 1.1); }
            const theo = [];
            if (S0 === "exb" || S0 === "gradb") {
              for (const p of parts) {
                const vpp = P.v0 * Math.sin((P.pitch * Math.PI) / 180), vd = S0 === "exb" ? -P.E / P.B : (p.m * vpp * vpp) / (2 * p.q * P.B * Lg);
                theo.push(vd);
                const yEnd = vd * (t1 - p.tStart);
                lo = Math.min(lo, 0, yEnd); hi = Math.max(hi, 0, yEnd);
              }
            }
            if (!isFinite(lo)) { lo = -1; hi = 1; }
            const pd = 0.08 * (hi - lo || 1);
            ax.setLimits([t0, t1], [lo - pd, hi + pd]);
            ax.clear();
            if (S0 === "mirror") ax.hline(1, { color: "rgba(230,237,243,0.5)", dash: [4, 3] });
            auxS.forEach((a, k) => ax.line(tX, a, { color: parts[k].color, width: 1.8 }));
            theo.forEach((vd, k) => ax.line([parts[k].tStart, t1], [0, vd * (t1 - parts[k].tStart)], { color: parts[k].color, width: 1.2, dash: [5, 4], alpha: 0.8 }));
            if (theo.length) ax.legend([{ label: "numerical (guiding centre)", color: "#c9d1d9" }, { label: "drift theory", color: "#c9d1d9", dash: [5, 4] }], "tr");
          }

          // ---------------- metrics
          const p = parts[0], el = Math.max(p.t - p.tStart, 1e-9);
          const wTh = (Math.abs(p.q) * P.B) / p.m, wMeas = Math.abs(p.angAcc) / el;
          const fmtPair = (a, b, d) => `${PM.fmt(a, d)} | ${PM.fmt(b, d)}`;
          const turned = Math.abs(p.angAcc) > 2 * Math.PI;
          if (S0 === "uniform" || S0 === "exb") {
            M.set("wc", turned ? fmtPair(wTh, wMeas, 4) : `${PM.fmt(wTh, 4)} | (wait one turn)`);
            const vperp = S0 === "exb" ? Math.hypot(P.v0, P.E / P.B) : P.v0 * Math.sin((P.pitch * Math.PI) / 180);
            const rTh = (p.m * vperp) / (Math.abs(p.q) * P.B);
            M.set("rl", turned ? fmtPair(rTh, (p.xmax - p.xmin) / 2, 3) : `${PM.fmt(rTh, 3)} | …`);
          }
          if (S0 === "uniform") M.set("pitchL", PM.fmt((2 * Math.PI * P.v0 * Math.cos((P.pitch * Math.PI) / 180)) / wTh, 3));
          if (S0 === "uniform" || S0 === "mirror" || S0 === "gradb") {
            const K0 = 0.5 * p.m * P.v0 * P.v0, K = 0.5 * p.m * (p.v[0] ** 2 + p.v[1] ** 2 + p.v[2] ** 2);
            M.set("dE", PM.fmt(Math.abs(K - K0) / K0, 1));
            if (ghost) { const s = ghost.s, Kg = 0.5 * ghost.m * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2); M.set("dEr", PM.fmt(Math.abs(Kg - ghost.K0) / ghost.K0, 1)); }
            else M.set("dEr", "— (tick the RK4 option)");
          }
          if (S0 === "exb") {
            const g = gc(p), vdm = (g[1] + (p.wrapY || 0) - p.gc0[1]) / el;
            M.set("vd", fmtPair(vdm, -P.E / P.B, 4));
            M.set("dE", PM.fmt(Math.abs(energy(p) - p.E0) / Math.max(Math.abs(p.E0), 1e-9), 1));
          }
          if (S0 === "selector") {
            M.set("vsel", PM.fmt(ESel / P.B, 3) + (P.E === 0 ? " (E = 0: using E = 0.05)" : ""));
            M.set("trans", `${passed} / ${launched}`);
            M.set("wc", PM.fmt(wTh, 3));
            M.set("rl", PM.fmt((P.m * ESel / P.B) / (P.q * P.B), 3));
          }
          if (S0 === "mirror") {
            const lc = (Math.asin(Math.sqrt(1 / P.Rm)) * 180) / Math.PI;
            M.set("lc", `${lc.toFixed(1)}°`);
            const trapped = parts.filter((q) => q.alive).length;
            M.set("trap", `${trapped} / ${parts.length}`);
            const mm = (p.muMin + p.muMax) / 2;
            M.set("mu", p.mu0 > 0 ? PM.fmt((p.muMax - p.muMin) / mm, 2) : "—");
          }
          if (S0 === "gradb") {
            for (let k = 0; k < Math.min(2, parts.length); k++) {
              const q = parts[k], g = gc(q), elk = Math.max(q.t - q.tStart, 1e-9);
              const vperp = P.v0 * Math.sin((P.pitch * Math.PI) / 180);
              const th = (q.m * vperp * vperp) / (2 * q.q * P.B * Lg);
              M.set(k === 0 ? "vd" : "vd2", fmtPair((g[1] + (q.wrapY || 0) - q.gc0[1]) / elk, th, 3));
            }
            if (parts.length < 2) M.set("vd2", "— (needs 2 particles)");
            const mm = (p.muMin + p.muMax) / 2;
            M.set("mu", p.mu0 > 0 ? PM.fmt((p.muMax - p.muMin) / mm, 2) : "—");
            M.set("eps", PM.fmt((p.m * P.v0 * Math.sin((P.pitch * Math.PI) / 180)) / (Math.abs(p.q) * P.B) / Lg, 3));
          }
          if (S0 === "massspec") {
            const pred = [], meas = [];
            const gapMin = P.np > 1 ? (2 * P.m * (MS_MASS[1] - MS_MASS[0]) * P.v0) / (Math.abs(q0) * P.B) : S;
            const win = Math.min(0.04 * S, 0.45 * gapMin);
            for (let k = 0; k < P.np; k++) {
              const x = (2 * P.m * MS_MASS[k] * P.v0) / (Math.abs(q0) * P.B) * Math.sign(q0);
              pred.push(PM.fmt(x, 2));
              let sw = 0, sx = 0;
              for (let b = 0; b < NBIN; b++) { const xc = -S + ((b + 0.5) * 2 * S) / NBIN; if (Math.abs(xc - x) < win) { sw += hist[b]; sx += hist[b] * xc; } }
              meas.push(sw > 0 ? PM.fmt(sx / sw, 2) : "…");
            }
            M.set("pred", pred.join(", "));
            M.set("meas", meas.join(", "));
            let tot = 0; for (let b = 0; b < NBIN; b++) tot += hist[b];
            M.set("hits", String(tot));
            M.set("res", P.np > 1 ? PM.fmt(MS_MASS[0] / (MS_MASS[1] - MS_MASS[0]), 2) + " needed; spread ≈ 0.4 % in v" : "—");
          }
          api.setTime(`t = ${PM.fmt(T, 2)}`);
        },
      };
    },
  });
})();
