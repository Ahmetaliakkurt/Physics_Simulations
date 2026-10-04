/* Planetary orbits and Kepler's laws — reduced two-body (central-force) problem in units AU, yr, GM = 4π² AU³/yr²,
 * integrated live with a time-symmetric adaptive leapfrog (kick–drift–kick velocity Verlet). Optional perturbations:
 * an extra 1/r³ force (exact apsidal precession) or a GR-like 1/r⁴ correction. */
(function () {
  "use strict";

  const GM = 4 * Math.PI * Math.PI, SQGM = 2 * Math.PI, R_STAR = 0.00465, ETA = 0.002, TWO_PI = 2 * Math.PI;
  const NSEC = 12; // equal-time sectors per reference period
  const PLANETS = [["Mercury", 0.387, 0.241], ["Venus", 0.723, 0.615], ["Earth", 1, 1], ["Mars", 1.524, 1.881],
    ["Jupiter", 5.203, 11.86], ["Saturn", 9.537, 29.46], ["Uranus", 19.19, 84.01], ["Neptune", 30.07, 164.8]];

  App.register({
    id: "kepler-orbits",
    category: "classical",
    group: "Mechanics",
    order: 12,
    title: "Planetary Orbits & Kepler's Laws",
    icon: "🪐",
    subtitle: "A planet launched from distance $r_0$ with any speed and direction moves on an ellipse, parabola or hyperbola around the Sun; " +
      "the page shows the orbit with equal-time sectors (2nd law), the effective potential with its turning points, and the period " +
      "against the semi-major axis (3rd law).",
    notes: [
      { type: "info", html: "Units: astronomical units (AU), years and solar masses, so $GM_\\odot=4\\pi^2\\ \\mathrm{AU^3/yr^2}$ and the Earth's orbit " +
        "($r_0=1$, $v_0=v_c$) has period exactly 1 yr. Use the buttons for the circular and escape speeds. Turn on a <b>perturbation</b> to make the " +
        "ellipse precess into a rosette — the shaded sectors stay equal, because the 2nd law only needs a central force." },
    ],
    animated: true,
    speed: { min: 0.1, max: 8, value: 1, step: 0.1 },
    controls: [
      { id: "r0", type: "slider", label: "Initial distance $r_0$", min: 0.3, max: 5, step: 0.01, value: 1, unit: "AU" },
      { id: "vr", type: "slider", label: "Initial speed $v_0/v_c$", min: 0.2, max: 1.8, step: 0.0001, value: 0.85, fmt: (v) => v.toFixed(4),
        help: "In units of the circular speed $v_c=\\sqrt{GM/r_0}$ (6.283 AU/yr = 29.8 km/s at 1 AU). Tangential launch: 1 → circle; " +
          "$\\sqrt2\\approx1.4142$ → parabola (escape speed $v_{\\rm esc}=\\sqrt{2GM/r_0}$); larger → hyperbola." },
      { id: "gamma", type: "slider", label: "Launch angle $\\gamma$ from the tangential direction", min: -80, max: 80, step: 1, value: 0, fmt: (v) => v.toFixed(0) + "°",
        help: "0° = perpendicular to the radius; positive = pointing outwards." },
      { id: "setCirc", type: "button", label: "Circular orbit (v₀ = v_c, γ = 0°)" },
      { id: "setEsc", type: "button", label: "Escape speed (v₀ = √2 v_c, parabola)" },
      { type: "section", label: "Perturbation" },
      { id: "pert", type: "select", label: "Extra force", value: "none", options: [
        { value: "none", label: "None (pure Kepler 1/r²)" },
        { value: "inv3", label: "Extra 1/r³ force: −GMλ/r³" },
        { value: "gr", label: "GR-like correction: −3GML²/(c²r⁴)" },
      ] },
      { id: "lam", type: "slider", label: "Strength $\\lambda$", min: -0.05, max: 0.15, step: 0.005, value: 0.03, unit: "AU",
        visibleIf: (p) => p.pert === "inv3", help: "λ > 0 attractive (prograde precession), λ < 0 repulsive (retrograde). If λ exceeds the semi-latus rectum p = L²/GM the planet spirals into the star." },
      { id: "cl", type: "slider", label: "Speed of light $c$ (artificially small)", min: 10, max: 200, step: 1, value: 40, unit: "AU/yr",
        visibleIf: (p) => p.pert === "gr", help: "The real value is 63 240 AU/yr, which gives Mercury's famous 43″ per century; a small c makes the effect visible." },
      { type: "section", label: "Display" },
      { id: "trail", type: "slider", label: "Trail length (orbital periods)", min: 0.25, max: 10, step: 0.25, value: 1.5, live: true },
      { id: "sectors", type: "checkbox", label: "Equal-time sectors (Kepler's 2nd law)", value: true, live: true },
      { id: "showVel", type: "checkbox", label: "Velocity vector", value: true, live: true },
      { id: "showConic", type: "checkbox", label: "Initial Kepler conic and turning-point circles", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>Two point masses — a star $M$ and a planet $m$ — interact by Newtonian gravity. Introducing the relative coordinate
      $\\mathbf r=\\mathbf r_m-\\mathbf r_M$ and the reduced mass $\\mu=mM/(m+M)$ turns the two-body problem into the motion of a single
      particle in the central potential $-G(M+m)\\mu/r$; for $m\\ll M$ this is simply the planet moving around a fixed star at the origin
      (the focus). Everything is per unit (reduced) mass. Units: lengths in astronomical units (AU), times in years, so that
      $GM=4\\pi^2\\ \\mathrm{AU^3/yr^2}$ for the Sun; speeds are in AU/yr (1 AU/yr = 4.74 km/s). The planet starts at
      $\\mathbf r_0=(r_0,0)$ with speed $v_0$ in units of the circular speed $v_c=\\sqrt{GM/r_0}$, at angle $\\gamma$ from the
      tangential direction. The star is a disc of radius $R_\\odot=0.00465$ AU; reaching it ends the run.</p>

      <h4>Equations being solved</h4>
      <p>Newton's equation for a central force, with an optional perturbation:</p>
      <div class="callout">$$\\ddot{\\mathbf r}=-\\frac{GM}{r^3}\\,\\mathbf r\\left[1+\\delta(r)\\right],\\qquad
        \\delta(r)=\\begin{cases}0 & \\text{pure Kepler}\\\\ \\lambda/r & \\text{extra } 1/r^3 \\text{ force}\\\\ 3L^2/(c^2r^2) & \\text{GR-like}\\end{cases}$$</div>
      <p>A central force exerts no torque, so the specific angular momentum $L=|\\mathbf r\\times\\dot{\\mathbf r}|=r^2\\dot\\varphi$ is
      conserved: the areal velocity is $dA/dt=L/2$, which <b>is</b> Kepler's second law. The energy
      $E=\\tfrac12\\dot r^2+\\tfrac12 r^2\\dot\\varphi^2+U(r)$ is conserved too; eliminating $\\dot\\varphi$ gives a 1-D problem in the
      <b>effective potential</b></p>
      $$E=\\tfrac12\\dot r^2+U_{\\rm eff}(r),\\qquad U_{\\rm eff}(r)=U(r)+\\frac{L^2}{2r^2},\\qquad
        U(r)=-\\frac{GM}{r}-\\frac{GM\\lambda}{2r^2}\\ \\ \\text{or}\\ \\ -\\frac{GM}{r}-\\frac{GML^2}{c^2r^3}.$$
      <p>The motion is confined to $U_{\\rm eff}(r)\\le E$, between the turning points $r_p$ (pericentre) and $r_a$ (apocentre). For pure
      Kepler the orbit is the conic $r(\\varphi)=p/[1+e\\cos(\\varphi-\\varpi)]$ with semi-latus rectum $p=L^2/GM$, eccentricity
      $e=\\sqrt{1+2EL^2/(GM)^2}$ and semi-major axis $a=-GM/2E$: an ellipse for $E&lt;0$ (Kepler's 1st law), a parabola for $E=0$
      ($v_0=v_{\\rm esc}=\\sqrt2\\,v_c$) and a hyperbola for $E&gt;0$. Integrating $dA/dt=L/2$ over one orbit with $A=\\pi ab$ gives the
      3rd law, $T^2=4\\pi^2a^3/GM$, i.e. $T[\\mathrm{yr}]=a[\\mathrm{AU}]^{3/2}$ in these units. With the $1/r^3$ term the orbit is still
      exactly solvable: $r=p/[1+e\\cos(k\\varphi)]$ with $k=\\sqrt{1-\\lambda/p}$, so the pericentre advances by
      $\\Delta\\varpi=2\\pi(1/k-1)$ per radial period. The GR-like term (the Schwarzschild correction written as a force) gives, to first
      order, $\\Delta\\varpi\\approx6\\pi GM/(c^2p)$.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Integrator.</b> Kick–drift–kick leapfrog (velocity Verlet):
        $\\mathbf v_{1/2}=\\mathbf v_n+\\tfrac h2\\mathbf a_n$, $\\mathbf r_{n+1}=\\mathbf r_n+h\\mathbf v_{1/2}$,
        $\\mathbf v_{n+1}=\\mathbf v_{1/2}+\\tfrac h2\\mathbf a_{n+1}$. It is symplectic and time-reversible, so energy errors oscillate
        instead of drifting, and for a central force it conserves $L$ <em>exactly</em> (the kicks are radial, the drift keeps
        $\\mathbf r\\times\\mathbf v$).</li>
        <li><b>Adaptive steps.</b> The step follows the local dynamical time, $h=\\eta\\,r^{3/2}/\\sqrt{GM}$ with $\\eta=0.002$
        (≈ 3000 steps per circular orbit, many more near a close pericentre). To keep the scheme time-symmetric, $h$ is taken as the
        average of the values at the start and at the (iteratively predicted) end of the step; this removes the secular energy drift
        that a one-sided adaptive step would cause. The playback is scaled so that one reference period takes about 5 s at ×1.</li>
        <li><b>Equal-time sectors.</b> The reference period ($T=a^{3/2}$ for ellipses, $r_0^{3/2}$ otherwise) is split into 12 equal
        intervals $\\Delta t$. The area swept is accumulated as the triangle $\\tfrac12(\\mathbf r_n\\times\\mathbf r_{n+1})$ of every drift;
        a sector boundary falling inside a step is interpolated along the straight drift. Because the scheme conserves $L$, each sector
        must equal $L\\Delta t/2$ to rounding error — the bar chart shows this.</li>
        <li><b>Metrics.</b> $E$ and $L$ from the current state; osculating $e=|\\mathbf v\\times\\mathbf L/GM-\\hat{\\mathbf r}|$ and
        $a=-GM/2E_{\\rm K}$ from the Kepler part; the measured period is the time between successive passages through the
        starting direction (one full $360°$ turn); the measured precession is the average advance of the pericentre angle, found by
        interpolating the sign change of $\\dot r=\\mathbf r\\cdot\\mathbf v/r$. It is compared with the prediction: $2\\pi(1/k-1)$ for the
        $1/r^3$ force and, for the GR-like term, the exact apsidal angle
        $\\Delta\\varphi=2\\int_{r_p}^{r_a}L\\,dr\\,/\\big(r^2\\sqrt{2(E-U_{\\rm eff})}\\big)-2\\pi$, evaluated by a 4000-point
        quadrature after the substitution $r=r_p+\\tfrac12(r_a-r_p)(1-\\cos u)$ that removes the endpoint singularities. The accuracy check is $|\\Delta E/E_0|$ (energy is not
        built into the scheme, so it honestly measures the integration error).</li>
        <li><b>Effective potential.</b> $U_{\\rm eff}(r)$ is drawn with the energy line; the turning points are found by stepping outwards and
        inwards from $r_0$ and bisecting $E-U_{\\rm eff}(r)=0$. The vertical bar at the current $r$ is the radial kinetic energy
        $\\tfrac12\\dot r^2$.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>The Earth.</b> Press <em>Circular orbit</em>: $e=0$, $a=1$ AU, measured period 1.000 yr, $E=-GM/2=-19.74$ AU²/yr²; the energy
        line touches the minimum of $U_{\\rm eff}$.</li>
        <li><b>Kepler's 2nd law.</b> Set $v_0/v_c=0.6$: the sectors near pericentre are short and wide, those near apocentre long and thin,
        yet the bars are equal; the speed at pericentre exceeds that at apocentre by the factor $r_a/r_p=(1+e)/(1-e)$.</li>
        <li><b>Kepler's 3rd law.</b> Change $r_0$ and $v_0$ and watch the red point move along the line $T=a^{3/2}$ through the planets;
        e.g. $r_0=5.2$, $v_0=v_c$ gives Jupiter's 11.9 yr.</li>
        <li><b>Escape.</b> Press <em>Escape speed</em>: $E=0$, $e=1$, a parabola; at $v_0/v_c=1.6$ the hyperbola has
        $e=v_0^2/v_c^2-1=1.56$ and $U_{\\rm eff}$ has a single turning point.</li>
        <li><b>Apsidal precession.</b> Choose the $1/r^3$ force with $\\lambda=0.03$ AU at $v_0/v_c=0.85$, $\\gamma=0$
        ($p=0.7225$ AU): predicted $\\Delta\\varpi=2\\pi(1/\\sqrt{1-0.0415}-1)=7.7°$ per orbit, matching the measured value. Increase the trail
        to 10 periods to see the rosette; a negative $\\lambda$ makes it precess backwards.</li>
        <li><b>Relativity in slow motion.</b> With the GR-like term and $c=40$ AU/yr the pericentre advances by about 50° per orbit, more
        than the first-order estimate $6\\pi GM/(c^2p)\\approx37°$ because the correction is not small; at $c=200$ AU/yr the two agree
        to a few per cent, and halving $c$ roughly quadruples the advance. Low $L$ (large $|\\gamma|$) lets the planet fall over the centrifugal barrier into the star — the
        relativistic capture that has no Newtonian analogue.</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>A point star with no tides, no other planets and no relativistic effects beyond the toy correction; the GR term reproduces the
      perihelion advance but not the full Schwarzschild dynamics. See Goldstein, Poole &amp; Safko, <em>Classical Mechanics</em>, ch. 3;
      J. R. Taylor, <em>Classical Mechanics</em>, ch. 8; Landau &amp; Lifshitz, <em>Mechanics</em>, §§14–15; and for the integrator,
      Hairer, Lubich &amp; Wanner, <em>Geometric Numerical Integration</em>.</p>`,

    mount(api) {
      const P = api.params;
      const plots = api.plots([
        { id: "orb", title: "Orbit — star at the focus (origin); shaded: equal-time sectors", aspect: 1, equal: true, xlabel: "x (AU)", ylabel: "y (AU)" },
        { id: "ueff", title: "Effective potential $U_{\\rm eff}(r)$ and energy $E$ (per unit mass, AU²/yr²)", aspect: 1, xlabel: "r (AU)", ylabel: "energy (AU²/yr²)" },
        { id: "area", title: "Kepler II — area swept in each of the last 12 equal time intervals", aspect: 0.62, xlabel: "sector (most recent on the right)", ylabel: "area (AU²)" },
        { id: "k3", title: "Kepler III — period vs semi-major axis, $T=a^{3/2}$", aspect: 0.62, xlog: true, ylog: true, xlim: [0.1, 100], ylim: [0.02, 1500],
          xlabel: "semi-major axis a (AU)", ylabel: "period T (yr)" },
      ]);
      const M = api.metrics([
        { id: "E", label: "Energy $E$ (AU²/yr²)" },
        { id: "L", label: "Angular momentum $L$ (AU²/yr)" },
        { id: "e", label: "Eccentricity $e$" },
        { id: "a", label: "Semi-major axis $a$ (AU)" },
        { id: "T", label: "Period: measured / $a^{3/2}$ (yr)" },
        { id: "dE", label: "Energy error $|\\Delta E/E_0|$" },
        { id: "pr", label: "Precession per orbit: measured / predicted" },
        { id: "v", label: "$v_0$ / $v_c$ / $v_{\\rm esc}$ (AU/yr)" },
      ]);

      // ---------------------------------------------------------------- state
      const CAP = 16384;
      const TT = new Float64Array(CAP), TX = new Float64Array(CAP), TY = new Float64Array(CAP);
      let tStart = 0, tCount = 0;
      const SBT = new Float64Array(NSEC + 2), SBX = new Float64Array(NSEC + 2), SBY = new Float64Array(NSEC + 2), SA = new Float64Array(NSEC + 2);
      let sbN = 0; // number of stored boundaries (≤ NSEC+1); areas SA[k] belong to sector [k, k+1]
      const PX = new Float64Array(CAP + 8), PY = new Float64Array(CAP + 8);
      const NU = 500, UR = new Float64Array(NU), UE = new Float64Array(NU), UU = new Float64Array(NU), UC = new Float64Array(NU);
      const CX = new Float64Array(721), CY = new Float64Array(721);
      const BX = new Float64Array(NSEC), BH = new Float64Array(NSEC);
      let x, y, vx, vy, ax, ay, t, L, L2, c2, lam, pertMode, E0, status, Tref, Tsec, tNextB, areaAcc, rate, Rview;
      let lastTrailT, lastTrailX, lastTrailY, thetaAcc, nTurns, lastTurnT, Tmeas, periPhi = [], rLo, rHi, kep, vScale;
      let AX = 0, AY = 0, predGR = NaN;

      function acc(px, py) {
        const r2 = px * px + py * py, r = Math.sqrt(r2);
        let f = -GM / (r2 * r);
        if (pertMode === 1) f *= 1 + lam / r;
        else if (pertMode === 2) f *= 1 + (3 * L2) / (c2 * r2);
        AX = f * px; AY = f * py;
      }
      function U(r) {
        let u = -GM / r;
        if (pertMode === 1) u -= (GM * lam) / (2 * r * r);
        else if (pertMode === 2) u -= (GM * L2) / (c2 * r * r * r);
        return u;
      }
      const Ueff = (r) => U(r) + L2 / (2 * r * r);
      const energy = () => 0.5 * (vx * vx + vy * vy) + U(Math.hypot(x, y));

      /** Osculating Kepler elements of a state (pure 1/r² part). */
      function elements(px, py, pvx, pvy) {
        const r = Math.hypot(px, py), v2 = pvx * pvx + pvy * pvy, rv = px * pvx + py * pvy;
        const eps = 0.5 * v2 - GM / r, Lz = px * pvy - py * pvx;
        const ex = ((v2 - GM / r) * px - rv * pvx) / GM, ey = ((v2 - GM / r) * py - rv * pvy) / GM;
        const e = Math.hypot(ex, ey), parab = Math.abs(eps) < 1e-9 * GM / r;
        return { eps: parab ? 0 : eps, L: Lz, e: parab ? 1 : e, w: Math.atan2(ey, ex), a: parab ? Infinity : -GM / (2 * eps), p: (Lz * Lz) / GM };
      }
      /** Turning point found by stepping from r0 inwards (dir = -1) or outwards (+1) and bisecting E − U_eff = 0. */
      function turning(r0, E, dir) {
        const f = (r) => E - Ueff(r);
        let a = r0;
        for (let k = 1; k <= 700; k++) {
          const b = r0 * Math.pow(1.02, dir * k);
          if (dir < 0 && b < R_STAR) return null;
          if (f(b) < 0) {
            let lo = a, hi = b;
            for (let it = 0; it < 80; it++) { const mid = 0.5 * (lo + hi); if (f(mid) >= 0) lo = mid; else hi = mid; }
            return 0.5 * (lo + hi);
          }
          a = b;
        }
        return null;
      }
      function pushTrail() {
        const i = (tStart + tCount) % CAP;
        TT[i] = t; TX[i] = x; TY[i] = y;
        if (tCount < CAP) tCount++; else tStart = (tStart + 1) % CAP;
        lastTrailT = t; lastTrailX = x; lastTrailY = y;
      }
      function pushBoundary(bt, bx, by, area) {
        if (sbN === NSEC + 1) { // drop the oldest
          for (let k = 0; k < NSEC; k++) { SBT[k] = SBT[k + 1]; SBX[k] = SBX[k + 1]; SBY[k] = SBY[k + 1]; SA[k] = SA[k + 1]; }
          sbN--;
        }
        if (sbN > 0) SA[sbN - 1] = area;
        SBT[sbN] = bt; SBX[sbN] = bx; SBY[sbN] = by; sbN++;
      }

      function stepOnce() {
        const r0 = Math.hypot(x, y), tau0 = (ETA * r0 * Math.sqrt(r0)) / SQGM;
        let h = tau0;
        for (let it = 0; it < 2; it++) { // time-symmetric step: average of the start and predicted end values
          const qx = x + h * (vx + 0.5 * h * ax), qy = y + h * (vy + 0.5 * h * ay), r1 = Math.hypot(qx, qy);
          h = 0.5 * (tau0 + (ETA * r1 * Math.sqrt(r1)) / SQGM);
        }
        const xo = x, yo = y, rdo = x * vx + y * vy, tOld = t;
        vx += 0.5 * h * ax; vy += 0.5 * h * ay;
        x += h * vx; y += h * vy;
        acc(x, y); ax = AX; ay = AY;
        vx += 0.5 * h * ax; vy += 0.5 * h * ay;
        t += h;
        // area swept during the (straight) drift
        const dA = 0.5 * (xo * y - x * yo);
        let fDone = 0;
        while (t >= tNextB) {
          const f = (tNextB - tOld) / h;
          areaAcc += (f - fDone) * dA; fDone = f;
          pushBoundary(tNextB, xo + f * (x - xo), yo + f * (y - yo), areaAcc);
          areaAcc = 0; tNextB += Tsec;
        }
        areaAcc += (1 - fDone) * dA;
        // polar angle accumulated → sidereal period
        const dth = Math.atan2(xo * y - yo * x, xo * x + yo * y);
        const thOld = thetaAcc; thetaAcc += dth;
        if (Math.floor(thetaAcc / TWO_PI) > Math.floor(thOld / TWO_PI) && dth > 0) {
          const target = Math.floor(thetaAcc / TWO_PI) * TWO_PI, tc = tOld + ((target - thOld) / dth) * h;
          Tmeas = tc - lastTurnT; lastTurnT = tc; nTurns++;
        }
        // pericentre passage (sign change of r·v) → apsidal angle
        const rd = x * vx + y * vy;
        if (kep.e > 1e-3 && rdo < 0 && rd >= 0) {
          const f = rdo / (rdo - rd);
          periPhi.push(Math.atan2(yo + f * (y - yo), xo + f * (x - xo)));
          if (periPhi.length > 12) periPhi.shift();
        }
        // trail sampling: every Tref/600 or every 1.5° of turning
        const cr = lastTrailX * y - lastTrailY * x, dt2 = lastTrailX * x + lastTrailY * y;
        if (t - lastTrailT >= Tref / 600 || Math.abs(Math.atan2(cr, dt2)) > 0.026) pushTrail();
        const r = Math.hypot(x, y);
        if (r < R_STAR) status = "crashed";
        else if (r > 60 * Rview && E0 >= 0) status = "escaped";
      }

      function reset() {
        pertMode = P.pert === "inv3" ? 1 : P.pert === "gr" ? 2 : 0;
        lam = P.lam; c2 = P.cl * P.cl;
        const r0 = P.r0, vc = Math.sqrt(GM / r0), vr = Math.abs(P.vr - Math.SQRT2) < 1e-4 ? Math.SQRT2 : P.vr;
        const v0 = vr * vc, g = (P.gamma * Math.PI) / 180;
        x = r0; y = 0; vx = v0 * Math.sin(g); vy = v0 * Math.cos(g);
        if (P.gamma === 0) vx = 0;
        L = x * vy - y * vx; L2 = L * L;
        acc(x, y); ax = AX; ay = AY;
        t = 0; E0 = energy(); status = "ok";
        kep = elements(x, y, vx, vy);
        if (pertMode === 0 && kep.eps === 0) E0 = 0;
        const Tc = Math.pow(r0, 1.5);
        const boundK = kep.eps < 0 && isFinite(kep.a);
        Tref = boundK && Math.pow(kep.a, 1.5) < 40 * Tc ? Math.pow(kep.a, 1.5) : Tc;
        Tsec = Tref / NSEC; tNextB = Tsec; areaAcc = 0; sbN = 0;
        tStart = 0; tCount = 0; thetaAcc = 0; nTurns = 0; lastTurnT = 0; Tmeas = NaN; periPhi = [];
        pushTrail(); pushBoundary(0, x, y, 0);
        rate = Tref / 5;

        // turning points of the radial motion
        rLo = turning(r0, E0, -1); rHi = E0 < 0 ? turning(r0, E0, +1) : null;
        predGR = pertMode === 2 ? apsidalQuadrature() : NaN;

        // ---- orbit view
        let xs0 = Math.min(0, x), xs1 = Math.max(0, x), ys0 = Math.min(0, y), ys1 = Math.max(0, y);
        const grow = (px, py) => { xs0 = Math.min(xs0, px); xs1 = Math.max(xs1, px); ys0 = Math.min(ys0, py); ys1 = Math.max(ys1, py); };
        const rCap = rHi && rHi < 40 * r0 ? rHi : Math.max(3 * (rLo || r0), 1.6 * r0);
        if (pertMode === 0) {
          for (let k = 0; k <= 720; k++) {
            const ph = (k / 720) * TWO_PI, den = 1 + kep.e * Math.cos(ph - kep.w);
            if (den <= 1e-9) continue;
            const r = kep.p / den;
            if (r <= rCap * 1.0001) grow(r * Math.cos(ph), r * Math.sin(ph));
          }
        } else { grow(-rCap, -rCap); grow(rCap, rCap); }
        const half = 0.5 * Math.max(xs1 - xs0, ys1 - ys0) * 1.28 + 1e-3, cx = 0.5 * (xs0 + xs1), cy = 0.5 * (ys0 + ys1);
        plots.orb.setLimits([cx - half, cx + half], [cy - half, cy + half]);
        Rview = 2 * half;
        vScale = (0.22 * 2 * half) / Math.max(rLo ? Math.abs(L) / rLo : v0, Math.sqrt(2 * Math.max(E0 - U(rCap), 0)), v0, 1e-9);

        // ---- effective potential curve
        const rx = rHi && rHi < 40 * r0 ? 1.3 * rHi : Math.max(2.5 * (rLo || r0), 1.5 * r0);
        let umin = Infinity;
        for (let i = 0; i < NU; i++) {
          const r = (rx * (i + 1)) / NU;
          UR[i] = r; UE[i] = Ueff(r); UU[i] = U(r); UC[i] = L2 / (2 * r * r);
          if (r > 0.15 * (rLo || r0) && UE[i] < umin) umin = UE[i];
        }
        const scale = Math.max(Math.abs(umin), Math.abs(E0), GM / rx);
        const ylo = Math.min(umin, E0) - 0.3 * scale, yhi = Math.max(E0, 0) + 0.95 * scale;
        plots.ueff.setLimits([0, rx], [ylo, yhi]);
        // clip the curves to a band around the view: huge pixel coordinates make canvas rasterisation very slow
        const yr = yhi - ylo, cl = (u) => Math.max(ylo - yr, Math.min(yhi + yr, u));
        for (let i = 0; i < NU; i++) { UE[i] = cl(UE[i]); UU[i] = cl(UU[i]); UC[i] = cl(UC[i]); }

        // ---- Kepler-III and sector plots
        for (let k = 0; k < NSEC; k++) BX[k] = k + 1;
        plots.area.setLimits([0.3, NSEC + 0.7], [0, 1.3 * 0.5 * Math.abs(L) * Tsec]);
        M.set("v", `${PM.fmt(v0, 3)} / ${PM.fmt(vc, 3)} / ${PM.fmt(Math.SQRT2 * vc, 3)}`);
      }

      function step(dt) {
        if (status !== "ok") return;
        const tTarget = t + dt * rate;
        let guard = 0;
        while (t < tTarget && status === "ok" && guard++ < 300000) stepOnce();
      }

      // ---------------------------------------------------------------- drawing
      function trailIndexFrom(tmin) { // first logical index with time ≥ tmin (binary search)
        let lo = 0, hi = tCount;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (TT[(tStart + mid) % CAP] < tmin) lo = mid + 1; else hi = mid; }
        return lo;
      }
      /** Apsidal advance per radial period (degrees): Δφ = 2∫ L dr / (r²√(2(E − U_eff))) − 2π, by quadrature. */
      function apsidalQuadrature() {
        if (!rLo || !rHi || (rHi - rLo) / rHi < 1e-6) return NaN;
        const n = 4000, half = 0.5 * (rHi - rLo);
        let s = 0;
        for (let i = 0; i < n; i++) {
          const u = (Math.PI * (i + 0.5)) / n, r = rLo + half * (1 - Math.cos(u));
          s += (Math.abs(L) / (r * r)) * (half * Math.sin(u)) / Math.sqrt(Math.max(2 * (E0 - Ueff(r)), 1e-300));
        }
        return (2 * s * (Math.PI / n) - TWO_PI) * (180 / Math.PI);
      }
      function predictedPrecession() { // degrees per orbit
        const p = L2 / GM;
        if (pertMode === 0) return 0;
        if (pertMode === 1) return lam < p ? 360 * (1 / Math.sqrt(1 - lam / p) - 1) : NaN;
        return predGR;
      }

      function render() {
        const p = plots.orb;
        p.clear();
        const [X0, X1] = p.visibleXlim, span = X1 - X0;
        p.hline(0, { color: PlotColors.muted, alpha: 0.25 }); p.vline(0, { color: PlotColors.muted, alpha: 0.25 });
        // reference conic and turning-point circles
        if (P.showConic) {
          let n = 0;
          for (let k = 0; k <= 720; k++) {
            const ph = (k / 720) * TWO_PI, den = 1 + kep.e * Math.cos(ph - kep.w);
            if (den <= 1e-9) { CX[n] = NaN; CY[n] = NaN; n++; continue; }
            const r = kep.p / den;
            if (r > 3 * span) { CX[n] = NaN; CY[n] = NaN; n++; continue; } // far branch of a parabola/hyperbola: off screen
            CX[n] = r * Math.cos(ph); CY[n] = r * Math.sin(ph); n++;
          }
          p.line(CX.subarray(0, n), CY.subarray(0, n), { color: "#ffffff", width: 1, dash: [4, 5], alpha: 0.35 });
          for (const rr of [rLo, rHi]) if (rr) p.circle(0, 0, rr, { fill: false, stroke: "rgba(139,152,168,0.45)", strokeWidth: 1 });
          if (kep.e > 1e-3 && kep.eps < 0 && pertMode === 0) { // empty focus
            const fx = -2 * kep.a * kep.e * Math.cos(kep.w), fy = -2 * kep.a * kep.e * Math.sin(kep.w);
            p.circle(fx, fy, 3.5, { px: true, fill: false, stroke: PlotColors.muted });
          }
        }
        // equal-time sectors
        if (P.sectors && sbN >= 1) {
          for (let k = 0; k < sbN; k++) {
            const tk = SBT[k], tk1 = k + 1 < sbN ? SBT[k + 1] : t;
            let n = 0;
            PX[n] = 0; PY[n] = 0; n++;
            PX[n] = SBX[k]; PY[n] = SBY[k]; n++;
            for (let i = trailIndexFrom(tk); i < tCount && n < CAP; i++) {
              const j = (tStart + i) % CAP;
              if (TT[j] >= tk1) break;
              PX[n] = TX[j]; PY[n] = TY[j]; n++;
            }
            if (k + 1 < sbN) { PX[n] = SBX[k + 1]; PY[n] = SBY[k + 1]; n++; } else { PX[n] = x; PY[n] = y; n++; }
            const col = k + 1 < sbN ? (k % 2 ? PlotColors.accent2 : PlotColors.accent) : PlotColors.accent3;
            p.poly(PX.subarray(0, n), PY.subarray(0, n), { color: col, alpha: k + 1 < sbN ? 0.22 : 0.3 });
          }
        }
        // trail, drawn in 8 chunks of increasing opacity
        const i0 = E0 >= 0 ? 0 : trailIndexFrom(t - P.trail * Tref), nT = tCount - i0; // unbound: keep the whole path
        if (nT > 1) {
          const chunks = 8;
          for (let c = 0; c < chunks; c++) {
            const a = i0 + Math.floor((c * nT) / chunks), b = Math.min(tCount, i0 + Math.floor(((c + 1) * nT) / chunks) + 1);
            let n = 0;
            for (let i = a; i < b; i++) { const j = (tStart + i) % CAP; PX[n] = TX[j]; PY[n] = TY[j]; n++; }
            if (c === chunks - 1) { PX[n] = x; PY[n] = y; n++; }
            p.line(PX.subarray(0, n), PY.subarray(0, n), { color: PlotColors.accent, width: 2, alpha: 0.15 + (0.85 * (c + 1)) / chunks });
          }
        }
        // star
        p.circle(0, 0, 13, { px: true, color: PlotColors.accent3, alpha: 0.18 });
        p.circle(0, 0, Math.max(R_STAR, 0), { color: "#ffd27a" });
        p.circle(0, 0, 5, { px: true, color: "#ffd27a" });
        if (P.showVel && status === "ok") p.arrow(x, y, x + vx * vScale, y + vy * vScale, { color: PlotColors.pink, width: 2, head: 8 });
        p.circle(x, y, 6, { px: true, color: status === "crashed" ? PlotColors.bad : "#79c0ff", stroke: "#ffffff", strokeWidth: 1.2 });
        const kind = E0 < -1e-12 ? (kep.e < 1e-4 && pertMode === 0 ? "circular orbit" : "bound orbit (E < 0)") : Math.abs(E0) <= 1e-12 ? "parabolic orbit (E = 0)" : "hyperbolic orbit (E > 0)";
        const st = status === "crashed" ? "collided with the star" : status === "escaped" ? "escaped to infinity" : kind;
        p.label([`t = ${PM.fmt(t, 3)} yr`, st], "tl", { size: 11.5 });
        const leg = [{ label: "trail", color: PlotColors.accent }];
        if (P.showVel) leg.push({ label: "velocity", color: PlotColors.pink });
        if (P.showConic) leg.push({ label: "initial Kepler conic", color: "#ffffff", dash: [4, 5] }, { label: "turning-point circles", color: PlotColors.muted });
        p.legend(leg, "tr");

        // effective potential
        const pu = plots.ueff;
        pu.clear();
        const r = Math.hypot(x, y);
        pu.hline(0, { color: PlotColors.muted, alpha: 0.4 });
        if (rLo || rHi) {
          const a = rLo || 0, b = rHi || pu.xlim[1];
          let n = 0;
          for (let i = 0; i < NU; i++) if (UR[i] >= a && UR[i] <= b) { PX[n] = UR[i]; PY[n] = Math.min(UE[i], E0); n++; }
          if (n > 1) pu.fill(PX.subarray(0, n), PY.subarray(0, n), E0, { color: PlotColors.accent, alpha: 0.15 });
        }
        pu.line(UR, UU, { color: PlotColors.blue, width: 1.2, dash: [5, 4], alpha: 0.8 });
        pu.line(UR, UC, { color: PlotColors.accent2, width: 1.2, dash: [2, 4], alpha: 0.8 });
        pu.line(UR, UE, { color: PlotColors.accent, width: 2.2 });
        pu.hline(E0, { color: PlotColors.accent3, width: 1.6 });
        for (const rr of [rLo, rHi]) if (rr) { pu.vline(rr, { color: PlotColors.muted, dash: [3, 4] }); pu.points([rr], [E0], { color: PlotColors.accent3, size: 4 }); }
        if (status === "ok") {
          const yr0 = pu.ylim[1] - pu.ylim[0], ue = Math.max(pu.ylim[0] - yr0, Math.min(pu.ylim[1] + yr0, Ueff(r)));
          pu.segment(r, ue, r, E0, { color: PlotColors.pink, width: 2.5 });
          pu.points([r], [ue], { color: "#79c0ff", size: 5, stroke: "#ffffff" });
        }
        pu.legend([{ label: "U_eff = U + L²/2r²", color: PlotColors.accent }, { label: "U(r)", color: PlotColors.blue, dash: [5, 4] },
          { label: "L²/2r² (centrifugal)", color: PlotColors.accent2, dash: [2, 4] }, { label: "energy E", color: PlotColors.accent3 },
          { label: "½ṙ² at current r", color: PlotColors.pink }], "tr");
        // turning-point labels at the foot of their vertical lines, on whichever side has room
        const yr = pu.ylim[1] - pu.ylim[0], xRight = pu.m.l + pu._v.pw;
        const tpLabel = (rr, txt, lift) => {
          const X = pu.X(rr), wpx = 6.4 * txt.length + 10, right = X + 6 + wpx < xRight;
          pu.text(rr, pu.ylim[0] + (0.05 + lift) * yr, txt, { dx: right ? 6 : -6, align: right ? "left" : "right", size: 11, color: PlotColors.accent3, bg: "#0f151c" });
        };
        if (rLo) tpLabel(rLo, `r_p = ${PM.fmt(rLo, 4)} AU`, rHi ? 0 : 0.08);
        else pu.label(["no inner turning point (plunge)"], "bl", { size: 11 });
        if (rHi) tpLabel(rHi, `r_a = ${PM.fmt(rHi, 4)} AU`, 0.08);
        else pu.label(["unbound: no outer turning point"], "br", { size: 11 });

        // sector areas
        const pa = plots.area, ref = 0.5 * Math.abs(L) * Tsec;
        pa.clear();
        const nS = Math.max(0, sbN - 1);
        let dev = 0;
        for (let k = 0; k < NSEC; k++) BH[k] = 0;
        for (let k = 0; k < nS; k++) { BH[NSEC - nS + k] = SA[k]; dev = Math.max(dev, Math.abs(SA[k] - ref) / ref); }
        const cols = []; for (let k = 0; k < NSEC; k++) cols.push(k % 2 ? PlotColors.accent2 : PlotColors.accent);
        pa.bars(BX, BH, 0.7, { colors: cols, alpha: 0.75 });
        pa.hline(ref, { color: PlotColors.accent3, dash: [6, 4], width: 1.5 });
        pa.label([`expected L·Δt/2 = ${PM.fmt(ref, 4)} AU²  (Δt = ${PM.fmt(Tsec, 4)} yr)`, nS ? `max relative deviation ${PM.fmt(dev, 2)}` : "waiting for the first sector…"], "tl", { size: 11 });

        // Kepler III
        const pk = plots.k3;
        pk.clear();
        pk.line([0.1, 100], [Math.pow(0.1, 1.5), 1000], { color: PlotColors.muted, width: 1.5, dash: [6, 4] });
        for (const [nm, a, T] of PLANETS) { pk.points([a], [T], { color: PlotColors.blue, size: 4 }); pk.text(a, T, nm, { dx: 7, dy: 8, size: 10.5, color: PlotColors.muted }); }
        const kNow = elements(x, y, vx, vy);
        if (kep.eps < 0 && isFinite(kep.a) && kep.a > 0.1 && kep.a < 100) {
          pk.circle(kep.a, Math.pow(kep.a, 1.5), 7, { px: true, fill: false, stroke: PlotColors.bad, strokeWidth: 1.6 });
          if (isFinite(Tmeas)) pk.points([kep.a], [Tmeas], { color: PlotColors.bad, size: 4.5 });
          pk.label(["this orbit: ○ T = a³ᐟ²,  ● measured"], "br", { size: 11, color: PlotColors.bad });
        } else pk.label(["this orbit is not a bound ellipse"], "br", { size: 11, color: PlotColors.muted });

        // metrics
        const E = energy();
        M.set("E", PM.fmt(E, 4));
        M.set("L", PM.fmt(L, 4));
        M.set("e", pertMode === 0 && E0 === 0 ? "1 (parabola)" : PM.fmt(kNow.e, 4));
        const parab = pertMode === 0 && E0 === 0; // exact escape-speed launch: show the ideal parabola values
        M.set("a", parab || kNow.eps === 0 ? "∞ (parabola)" : kNow.eps < 0 ? PM.fmt(kNow.a, 4) : PM.fmt(kNow.a, 4) + " (hyperbola)");
        M.set("T", (isFinite(Tmeas) ? PM.fmt(Tmeas, 4) : "—") + " / " + (!parab && kNow.eps < 0 ? PM.fmt(Math.pow(kNow.a, 1.5), 4) : "∞"));
        const norm = Math.abs(E0) > 0.01 * GM / P.r0 ? Math.abs(E0) : GM / P.r0;
        M.set("dE", PM.fmt(Math.abs(E - E0) / norm, 2) + (norm === Math.abs(E0) ? "" : " (of GM/r₀)"));
        let meas = NaN;
        if (periPhi.length >= 2) {
          let s = 0;
          for (let k = 1; k < periPhi.length; k++) { let d = periPhi[k] - periPhi[k - 1]; d -= TWO_PI * Math.round(d / TWO_PI); s += d; }
          meas = (s / (periPhi.length - 1)) * (180 / Math.PI);
        }
        const pred = predictedPrecession();
        M.set("pr", (isFinite(meas) ? PM.fmt(meas, 3) + "°" : "—") + " / " + (isFinite(pred) ? PM.fmt(pred, 3) + "°" : "plunge"));
        api.setTime(`t = ${PM.fmt(t, 3)} yr`);
      }

      return {
        reset, step, render,
        onAction(id) {
          if (id === "setCirc") { api.setControl("vr", { value: 1 }); api.setControl("gamma", { value: 0 }); }
          if (id === "setEsc") api.setControl("vr", { value: Math.SQRT2 });
          reset();
        },
      };
    },
  });
})();
