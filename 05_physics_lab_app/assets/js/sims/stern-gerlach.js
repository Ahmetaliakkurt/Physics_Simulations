/* Stern–Gerlach: live comparison of classical, semiclassical and quantum (two-component split-step) models. */
(function () {
  "use strict";

  const YB1 = 2, YB2 = 5, YS = 7, YMAX = 10, ZLIM = 20;     // field region, screen, plot limits
  const NZ = 1024, ZMAX = 25, DZ = (2 * ZMAX) / NZ, SIG = 0.6, DTQ = 0.01;
  const CAP = 40000, NB = 80, BW = (2 * ZLIM) / NB, RECENT = 500;
  const NYT = 200, NZT = 200;                                // quantum beam trace (y × z)

  const C_CL = PlotColors.accent3, C_SC = "#f5d742", C_UP = PlotColors.accent, C_DN = PlotColors.bad;

  App.register({
    id: "stern-gerlach",
    category: "quantum",
    order: 38,
    title: "Stern–Gerlach Experiment",
    icon: "🧲",
    subtitle: "Spin splitting in an inhomogeneous magnetic field: a classical, a semiclassical and a quantum (two-component Pauli) model run side by side, each building up its own screen pattern.",
    notes: [{
      type: "info",
      html: "The same experiment in three frameworks: <b>(1) classical</b> — the magnetic moment component $\\mu_z$ comes from a continuous " +
        "distribution of orientations and the screen shows one broad band; <b>(2) semiclassical</b> — $\\mu_z$ is restricted by hand to $\\pm1$, " +
        "“old quantum theory” style; <b>(3) quantum</b> — the ↑ and ↓ components of a spin-½ wave packet evolve under the Pauli equation and " +
        "separate by themselves inside the field. Quantum screen hits are drawn with the Born rule from $|\\psi_\\uparrow|^2+|\\psi_\\downarrow|^2$. " +
        "Only the third model produces the observed two-beam result from first principles.",
    }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "F0", type: "slider", label: "Gradient force $F_0=\\mu\\,\\partial B_z/\\partial z$", min: 1, max: 10, step: 0.5, value: 5 },
      { id: "vy", type: "slider", label: "Forward velocity $v_y$", min: 0.5, max: 5, step: 0.1, value: 2 },
      { id: "rate", type: "slider", label: "Beam flux (particles / time unit)", min: 50, max: 1500, step: 50, value: 400, live: true },
      { id: "spread", type: "checkbox", label: "Give classical particles the quantum packet's initial spread", value: false,
        help: "When on, the initial $z$ and $p_z$ of the classical/semiclassical particles are drawn from the Wigner distribution of the quantum packet ($\\Delta z=\\sigma/\\sqrt2$, $\\Delta p=\\hbar/\\sigma\\sqrt2$)." },
      { id: "seed", type: "number", label: "Random seed", min: 0, max: 9999, step: 1, value: 42 },
      { type: "section", label: "Display" },
      { id: "trace", type: "checkbox", label: "Show the quantum beam trace", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>Neutral particles with a magnetic moment $\\boldsymbol\\mu$ (silver atoms in the 1922 experiment, where $\\boldsymbol\\mu$ comes from
      one unpaired electron spin) fly along $y$ at constant speed $v_y$. Between $y=2$ and $y=5$ they pass a magnet whose field
      $B_z(z)=B_0+B'z$ grows linearly with height; outside it there is no force. The detection screen stands at $y_s=7$. Each particle starts at
      $z=0$ (optionally with the quantum packet's spread). Units: $\\hbar=m=\\mu=1$, and the strength of the inhomogeneity is expressed through the
      force $F_0=\\mu B'$ on a fully aligned moment. A steady stream of particles (rate set by "Beam flux") is emitted; the quantum panel
      evolves one wave packet at a time and repeats it.</p>

      <h4>Equations being solved</h4>
      <p>The energy of a moment in a field is $U=-\\boldsymbol\\mu\\cdot\\mathbf B$, so the force is</p>
      <div class="callout">$$ \\mathbf F=\\nabla(\\boldsymbol\\mu\\cdot\\mathbf B)\\;\\Rightarrow\\; F_z=\\mu_z\\,\\frac{\\partial B_z}{\\partial z}=F_0\\,\\frac{\\mu_z}{\\mu},\\qquad
        z_s=\\frac{F_z}{m}\\,\\tau\\left(\\frac{\\tau}{2}+\\frac{y_s-5}{v_y}\\right),\\quad \\tau=\\frac{3}{v_y}. $$</div>
      <p>$z_s$ is the screen deflection: uniform acceleration for the time $\\tau$ spent in the magnet, then straight-line drift to the screen.
      (The transverse components of $\\boldsymbol\\mu$ precess rapidly about the strong field $B_0$ — Larmor precession — so their force averages
      out and $\\mu_z$ stays constant.)</p>
      <ul>
        <li><b>Classical:</b> random orientations, $\\mu_z=\\mu\\cos\\vartheta$ with $\\cos\\vartheta$ uniform in $[-1,1]$ (isotropic) → a continuous band
        $|z|\\le z_s$.</li>
        <li><b>Semiclassical:</b> $\\mu_z=\\pm\\mu$ imposed by hand → two spots at $\\pm z_s$, but nothing explains why.</li>
        <li><b>Quantum:</b> the state is a two-component spinor $\\Psi=(\\psi_\\uparrow,\\psi_\\downarrow)^T$ obeying the Pauli equation
        $i\\hbar\\,\\partial_t\\Psi=\\big[p^2/2m-\\mu\\,\\boldsymbol\\sigma\\cdot\\mathbf B\\big]\\Psi$. Keeping $B_z$ only (the standard
        approximation justified by the precession argument above), the Hamiltonian is diagonal and each component feels its own linear potential:
        $$ i\\hbar\\frac{\\partial\\psi_{\\uparrow,\\downarrow}}{\\partial t}=\\left[-\\frac{\\hbar^2}{2m}\\frac{\\partial^2}{\\partial z^2}\\mp F_0\\,z\\,\\chi(t)\\right]\\psi_{\\uparrow,\\downarrow},
           \\qquad \\psi_{\\uparrow}(z,0)=\\psi_{\\downarrow}(z,0)=\\tfrac{1}{\\sqrt2}(\\pi\\sigma^2)^{-1/4}e^{-z^2/2\\sigma^2}, $$
        where $\\chi(t)=1$ while the packet ($y=v_yt$) is inside the magnet, $\\sigma=0.6$ and the constant $\\mp\\mu B_0$ is dropped (it only adds a
        phase). The spin starts along $+x$, an equal superposition of ↑ and ↓; the two eigenvalues $\\pm\\hbar/2$ of $S_z$ become two spatially
        separated packets, and the Born probabilities are $P_{\\uparrow,\\downarrow}=\\int|\\psi_{\\uparrow,\\downarrow}|^2dz=\\tfrac12$.</li>
      </ul>
      <p>For a linear potential Ehrenfest's theorem is exact: each component's centre follows the classical trajectory, so
      $\\langle z\\rangle_\\uparrow-\\langle z\\rangle_\\downarrow=2z_s$ at the screen. Likewise the Wigner function evolves exactly by the classical
      Liouville equation — what is quantum here is not the shape of the packets but the fact that $\\mu_z$ takes only two values.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Classical and semiclassical beams:</b> the trajectory is piecewise analytic (free flight, constant acceleration, free flight), so each
        particle's position is evaluated exactly from its emission time — no integration error. Emission is a continuous stream at the chosen
        rate; on reaching $y_s$ a particle is binned into the 80-bin screen histogram.</li>
        <li><b>Quantum packet:</b> split-step Fourier on $N_z=1024$ points in $-25\\le z\\le25$ ($\\Delta z\\approx0.049$, $|k|\\le64$) with fixed step
        $\\Delta t=0.01$ (up to 12 steps per frame):
        $$ \\psi_{\\uparrow,\\downarrow}\\leftarrow e^{\\pm iF_0z\\,f\\Delta t/2\\hbar}\\;\\mathcal F^{-1}\\!\\left[e^{-i\\hbar k^2\\Delta t/2m}\\,\\mathcal F\\!\\left[e^{\\pm iF_0z\\,f\\Delta t/2\\hbar}\\,\\psi_{\\uparrow,\\downarrow}\\right]\\right], $$
        where $f\\in[0,1]$ is the fraction of the step spent inside the magnet, so the field switches on and off at exactly the right time. For a linear
        potential this Strang splitting is very accurate (the commutator $[p^2,z]$ only produces a constant momentum kick). Absorbing layers at
        $|z|>20$ multiply $\\psi$ by $e^{-0.2(|z|-20)^2}$ each step, removing deflected waves that leave the plotted range.</li>
        <li><b>Detections (Born rule):</b> when the packet reaches $y_s$, $|\\psi_\\uparrow|^2$ and $|\\psi_\\downarrow|^2$ are stored as cumulative
        distributions. For every semiclassical screen hit one quantum hit is drawn: first the spin (↑ with probability $P_\\uparrow$), then $z$ by
        inverse-transform sampling; probability lost in the absorbers counts as "off-screen".</li>
        <li><b>Metrics:</b> classical std$(z)$ of the hits (expected $z_s/\\sqrt3$ for a uniform band without spread), the semiclassical beam separation
        $|\\bar z_+-\\bar z_-|$, and the quantum $\\langle z\\rangle_\\uparrow-\\langle z\\rangle_\\downarrow$ at the screen, both to be compared with the predicted
        $2z_s$. The quantum value is an accuracy check of the solver, since Ehrenfest makes it exact; it comes out a few per cent
        low only when the outer tails of the spread packets reach the absorbers at $|z|>20$ (with the defaults ≈ 25.7 instead of 26.25).</li>
      </ul>

      <h4>What to try</h4>
      <ul>
        <li><b>Defaults</b> ($F_0=5$, $v_y=2$): $\\tau=1.5$, $z_s=5\\cdot1.5\\,(0.75+1)\\approx13.1$, so the semiclassical separation reads 26.25 (the quantum one ≈ 25.7, see above) and the classical std ≈ $13.1/\\sqrt3\\approx7.6$.</li>
        <li><b>Weak gradient.</b> Lower $F_0$ to 1: $z_s\\approx2.6$, comparable to the spread the packet acquires during the flight
        ($\\sigma(t)$ grows from 0.6 to ≈ 4); the two quantum beams overlap and the split is barely resolved — as in real experiments with
        too weak a magnet.</li>
        <li><b>Slow atoms.</b> $v_y=1$ doubles $\\tau$ and the drift time: $z_s=5\\cdot3\\,(1.5+2)=52.5$ — beyond the screen, so all hits are counted
        "off-screen". Raise $v_y$ or lower $F_0$ to bring them back.</li>
        <li><b>Initial spread.</b> Tick "Give classical particles the quantum packet's initial spread": the semiclassical histogram becomes identical
        to the quantum one (exact Wigner–Liouville correspondence for linear forces), while the classical band only gets blurred edges.</li>
      </ul>

      <h4>Limitations &amp; further reading</h4>
      <p>One-dimensional transverse model with only the $B_z$ component (Maxwell's $\\nabla\\cdot\\mathbf B=0$ requires a transverse component too,
      whose effect averages out by precession); a sharp-edged field region; no orbital motion along $x$; spin-½ only. Reading: J. J. Sakurai,
      <i>Modern Quantum Mechanics</i>, §1.1; D. J. Griffiths &amp; D. F. Schroeter, <i>Introduction to Quantum Mechanics</i>, §4.4.2; B. Friedrich &amp;
      D. Herschbach, "Stern and Gerlach: how a bad cigar helped reorient atomic physics", <i>Physics Today</i> 56(12), 53 (2003).</p>`,

    mount(api) {
      const P = api.params;
      const plots = api.plots([
        { id: "c", title: "1. Classical (Newton–Maxwell): continuous μ<sub>z</sub> distribution", aspect: 0.34, minHeight: 220, xlim: [0, YMAX], ylim: [-ZLIM, ZLIM], ylabel: "deflection z" },
        { id: "ch", title: "Screen histogram", aspect: 1.0, minHeight: 220, xlim: [0, 1.15], ylim: [-ZLIM, ZLIM], margin: { l: 34, r: 10 } },
        { id: "s", title: "2. Semiclassical (old quantum theory): μ<sub>z</sub> = ±1 imposed by hand", aspect: 0.34, minHeight: 220, xlim: [0, YMAX], ylim: [-ZLIM, ZLIM], ylabel: "deflection z" },
        { id: "sh", title: "Screen histogram", aspect: 1.0, minHeight: 220, xlim: [0, 1.15], ylim: [-ZLIM, ZLIM], margin: { l: 34, r: 10 } },
        { id: "q", title: "3. Quantum (Pauli equation): the wave packet splits", aspect: 0.34, minHeight: 220, xlim: [0, YMAX], ylim: [-ZLIM, ZLIM], xlabel: "flight direction y", ylabel: "deflection z" },
        { id: "qh", title: "Screen histogram and |ψ|²", aspect: 1.0, minHeight: 220, xlim: [0, 1.15], ylim: [-ZLIM, ZLIM], xlabel: "relative intensity", margin: { l: 34, r: 10 } },
      ]);
      // 3:1 column layout (wide screens)
      const box = plots.c.container.parentElement;
      const wide = window.matchMedia("(min-width: 1100px)");
      const applyCols = () => { box.style.gridTemplateColumns = wide.matches ? "minmax(0,3fr) minmax(0,1fr)" : ""; };
      applyCols();
      if (wide.addEventListener) wide.addEventListener("change", applyCols);

      const Mt = api.metrics([
        { id: "t", label: "Time $t$" },
        { id: "cstd", label: "Classical: screen std$(z)$" },
        { id: "sdz", label: "Semiclassical: beam separation $|\\bar z_+-\\bar z_-|$" },
        { id: "qdz", label: "Quantum (at screen): $\\langle z\\rangle_\\uparrow-\\langle z\\rangle_\\downarrow$" },
        { id: "zth", label: "Predicted separation $2z_s$" },
      ]);

      // ---- particle pools (ring buffers)
      function pool() {
        return { te: new Float64Array(CAP), a: new Float64Array(CAP), z0: new Float64Array(CAP), v0: new Float64Array(CAP),
          head: 0, tail: 0, hist: new Float64Array(NB), rz: new Float64Array(RECENT), rn: 0,
          n: 0, out: 0, sz: 0, szz: 0, np: 0, sp: 0, nm: 0, sm: 0, xs: new Float64Array(CAP), ys: new Float64Array(CAP) };
      }
      const cl = pool(), sc = pool();
      // ---- quantum
      const uR = new Float64Array(NZ), uI = new Float64Array(NZ), dR = new Float64Array(NZ), dI = new Float64Array(NZ);
      const tR = new Float64Array(NZ), tI = new Float64Array(NZ), vhR = new Float64Array(NZ), vhI = new Float64Array(NZ);
      const absorb = new Float64Array(NZ), zg = new Float64Array(NZ);
      for (let i = 0; i < NZ; i++) zg[i] = -ZMAX + i * DZ;
      const kz = PM.fftk(NZ, DZ);
      const rhoU = new Float64Array(NZ), rhoD = new Float64Array(NZ);
      const scrU = new Float64Array(NZ), scrD = new Float64Array(NZ), cdfU = new Float64Array(NZ), cdfD = new Float64Array(NZ);
      const qHistU = new Float64Array(NB), qHistD = new Float64Array(NB), qRz = new Float64Array(RECENT), qRs = new Int8Array(RECENT);
      const trace = new Float32Array(NZT * NYT), traceIdx = new Int32Array(NZT);
      for (let r = 0; r < NZT; r++) traceIdx[r] = Math.round((-ZLIM + ((r + 0.5) * 2 * ZLIM) / NZT + ZMAX) / DZ);
      // drawing buffers
      const i0 = Math.floor((-ZLIM + ZMAX) / DZ), i1 = Math.ceil((ZLIM + ZMAX) / DZ), NW = i1 - i0 + 1;
      const polyX = new Float64Array(2 * NW), polyY = new Float64Array(2 * NW), lineX = new Float64Array(NW), lineY = new Float64Array(NW);
      for (let k = 0; k < NW; k++) lineY[k] = zg[i0 + k];
      const hbX = new Float64Array(NB), hbY = new Float64Array(NB);
      let rng, t, tq, qAcc = 0, qSep = NaN, qReach = 1, qOut = 0, emitCarry, pScale, haveScreen, pUp, qn, qRn, lastCol, zth;

      function zAt(tau, a, z0, v0) {
        const tin = YB1 / P.vy, tout = YB2 / P.vy;
        let z = z0 + v0 * tau;
        if (tau > tin) {
          if (tau <= tout) { const d = tau - tin; z += 0.5 * a * d * d; }
          else { const d = tout - tin; z += 0.5 * a * d * d + a * d * (tau - tout); }
        }
        return z;
      }
      function emit(pl, te, a) {
        const k = pl.head % CAP;
        pl.te[k] = te; pl.a[k] = a;
        if (P.spread) { pl.z0[k] = rng.gauss(0, SIG / Math.SQRT2); pl.v0[k] = rng.gauss(0, 1 / (SIG * Math.SQRT2)); }
        else { pl.z0[k] = 0; pl.v0[k] = 0; }
        pl.head++;
        if (pl.head - pl.tail > CAP) pl.tail = pl.head - CAP;
      }
      function binOf(z) { const b = Math.floor((z + ZLIM) / BW); return b >= 0 && b < NB ? b : -1; }
      /** Record particles that reached the screen; returns the number of new hits. */
      function collect(pl, split) {
        const tS = YS / P.vy;
        let n = 0;
        while (pl.tail < pl.head) {
          const k = pl.tail % CAP;
          if (t - pl.te[k] < tS) break;
          const z = zAt(tS, pl.a[k], pl.z0[k], pl.v0[k]);
          const b = binOf(z); if (b >= 0) pl.hist[b]++; else pl.out++;
          pl.rz[pl.rn % RECENT] = z; pl.rn++;
          pl.n++; pl.sz += z; pl.szz += z * z;
          if (split) { if (pl.a[k] > 0) { pl.np++; pl.sp += z; } else { pl.nm++; pl.sm += z; } }
          pl.tail++; n++;
        }
        return n;
      }
      function initQuantum() {
        const c0 = Math.pow(Math.PI * SIG * SIG, -0.25) / Math.SQRT2;
        for (let i = 0; i < NZ; i++) { const g = c0 * Math.exp(-(zg[i] * zg[i]) / (2 * SIG * SIG)); uR[i] = g; dR[i] = g; uI[i] = 0; dI[i] = 0; }
        tq = 0; lastCol = -1;
      }
      function setVhalf(f) {
        // ↑: V = −F0 z → e^{-iVΔt f/2} = e^{+iF0 z Δt f/2}; the ↓ component uses the complex conjugate
        for (let i = 0; i < NZ; i++) { const ph = 0.5 * P.F0 * zg[i] * DTQ * f; vhR[i] = Math.cos(ph); vhI[i] = Math.sin(ph); }
      }
      let vhF = -1;
      function kick(re, im, conj) {
        const s = conj ? -1 : 1;
        for (let i = 0; i < NZ; i++) {
          const a = vhR[i], b = s * vhI[i], r = re[i], m = im[i];
          re[i] = r * a - m * b; im[i] = r * b + m * a;
        }
      }
      function kinetic(re, im) {
        PM.fft(re, im, false);
        for (let i = 0; i < NZ; i++) { const a = tR[i], b = tI[i], r = re[i], m = im[i]; re[i] = r * a - m * b; im[i] = r * b + m * a; }
        PM.fft(re, im, true);
      }
      function qStep() {
        const tin = YB1 / P.vy, tout = YB2 / P.vy;
        const f = Math.max(0, Math.min(tq + DTQ, tout) - Math.max(tq, tin)) / DTQ;
        if (f > 0 && f !== vhF) { setVhalf(f); vhF = f; }
        if (f > 0) { kick(uR, uI, false); kick(dR, dI, true); }
        kinetic(uR, uI); kinetic(dR, dI);
        if (f > 0) { kick(uR, uI, false); kick(dR, dI, true); }
        for (let i = 0; i < NZ; i++) { const A = absorb[i]; uR[i] *= A; uI[i] *= A; dR[i] *= A; dI[i] *= A; }
        tq += DTQ;
        const y = P.vy * tq;
        // beam trace: write the total density into the y columns just passed
        const col = Math.min(NYT - 1, Math.floor((y / YMAX) * NYT));
        if (col > lastCol && y <= YMAX) {
          let mx = 0;
          for (let r = 0; r < NZT; r++) { const i = traceIdx[r], v = uR[i] * uR[i] + uI[i] * uI[i] + dR[i] * dR[i] + dI[i] * dI[i]; if (v > mx) mx = v; }
          for (let c = lastCol + 1; c <= col; c++) for (let r = 0; r < NZT; r++) {
            const i = traceIdx[r], v = uR[i] * uR[i] + uI[i] * uI[i] + dR[i] * dR[i] + dI[i] * dI[i];
            trace[r * NYT + c] = mx > 1e-4 ? v / mx : 0;
          }
          lastCol = col;
        }
        if (!haveScreen && y >= YS) captureScreen();
        if (y > YMAX + 0.3) initQuantum();
      }
      function captureScreen() {
        let su = 0, sd = 0;
        for (let i = 0; i < NZ; i++) {
          scrU[i] = uR[i] * uR[i] + uI[i] * uI[i]; scrD[i] = dR[i] * dR[i] + dI[i] * dI[i];
          su += scrU[i]; sd += scrD[i]; cdfU[i] = su; cdfD[i] = sd;
        }
        pUp = su / (su + sd || 1);
        qReach = (su + sd) * DZ;   // probability not lost in the absorbers (reaches the screen)
        let a = 0, b = 0; for (let i = 0; i < NZ; i++) { a += scrU[i] * zg[i]; b += scrD[i] * zg[i]; }
        qSep = (su > 1e-12 && sd > 1e-12) ? a / su - b / sd : NaN;
        haveScreen = true;
      }
      function sampleQ(n) {
        if (!haveScreen) return;
        for (let q = 0; q < n; q++) {
          if (rng.next() > qReach) { qn++; qOut++; continue; }
          const up = rng.next() < pUp, cdf = up ? cdfU : cdfD, tot = cdf[NZ - 1];
          const u = rng.next() * tot;
          let lo = 0, hi = NZ - 1;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
          const z = zg[lo] + (rng.next() - 0.5) * DZ, b = binOf(z);
          if (b >= 0) (up ? qHistU : qHistD)[b]++; else qOut++;
          qRz[qRn % RECENT] = z; qRs[qRn % RECENT] = up ? 1 : 0; qRn++; qn++;
        }
      }

      // ---- drawing helpers
      function drawMagnet(p) {
        p.rect(YB1, -ZLIM, YB2, ZLIM, { color: "#58a6ff", alpha: 0.06 });
        const xm = 0.5 * (YB1 + YB2);
        p.poly([YB1, YB2, YB2, xm + 0.25, xm - 0.25, YB1], [ZLIM, ZLIM, ZLIM - 2.2, ZLIM - 3.6, ZLIM - 3.6, ZLIM - 2.2], { color: PlotColors.bad, alpha: 0.45, stroke: "rgba(248,81,73,0.8)" });
        p.poly([YB1, YB2, YB2, YB1], [-ZLIM, -ZLIM, -ZLIM + 2.2, -ZLIM + 2.2], { color: PlotColors.blue, alpha: 0.45, stroke: "rgba(88,166,255,0.8)" });
        p.text(xm, ZLIM - 1.3, "N", { align: "center", size: 11, bold: true });
        p.text(xm, -ZLIM + 1.1, "S", { align: "center", size: 11, bold: true });
        p.vline(YS, { color: PlotColors.text, width: 2, dash: [6, 4], alpha: 0.8 });
      }
      function drawRecent(p, arr, n, color, signs) {
        const m = Math.min(n, RECENT);
        p.custom((c, pl) => {
          const X = pl.X(YS);
          if (!signs) { c.fillStyle = color; c.globalAlpha = 0.55; }
          for (let k = 0; k < m; k++) {
            const idx = (n - 1 - k) % RECENT;
            if (signs) { c.fillStyle = signs[idx] ? C_UP : C_DN; c.globalAlpha = 0.6; }
            c.fillRect(X - 4 + ((k * 7919) % 9), pl.Y(arr[idx]) - 1, 2, 2);
          }
          c.globalAlpha = 1;
        });
      }
      function drawParticles(p, pl, color) {
        const tS = YS / P.vy, stride = Math.max(1, Math.ceil((pl.head - pl.tail) / 6000));
        let n = 0;
        for (let q = pl.tail; q < pl.head; q += stride) {
          const k = q % CAP, tau = t - pl.te[k];
          if (tau < 0 || tau >= tS) continue;
          pl.xs[n] = P.vy * tau; pl.ys[n] = zAt(tau, pl.a[k], pl.z0[k], pl.v0[k]); n++;
        }
        p.points(pl.xs.subarray(0, n), pl.ys.subarray(0, n), { color, size: 1.1, alpha: 0.7, square: true });
      }
      function hbars(p, hist, norm, color, alpha) {
        if (norm <= 0) return;
        for (let b = 0; b < NB; b++) if (hist[b] > 0) {
          const z0 = -ZLIM + b * BW;
          p.rect(0, z0, Math.min(hist[b] / norm, 1.15), z0 + BW * 0.9, { color, alpha });
        }
      }
      function hmax(h) { let m = 0; for (let b = 0; b < NB; b++) if (h[b] > m) m = h[b]; return m; }

      return {
        reset() {
          rng = new PM.RNG((P.seed | 0) + 1);
          for (const pl of [cl, sc]) {
            pl.head = pl.tail = 0; pl.hist.fill(0); pl.rn = 0;
            pl.n = pl.out = pl.sz = pl.szz = pl.np = pl.sp = pl.nm = pl.sm = 0;
          }
          for (let i = 0; i < NZ; i++) {
            const ph = -0.5 * kz[i] * kz[i] * DTQ; tR[i] = Math.cos(ph); tI[i] = Math.sin(ph);
            const z = zg[i], sw = 5;
            absorb[i] = z > ZMAX - sw ? Math.exp(-0.2 * (z - (ZMAX - sw)) ** 2) : z < -ZMAX + sw ? Math.exp(-0.2 * (z + ZMAX - sw) ** 2) : 1;
          }
          vhF = -1;
          initQuantum();
          trace.fill(0); qHistU.fill(0); qHistD.fill(0); qRn = 0; qn = 0;
          haveScreen = false; pUp = 0.5; qSep = NaN; qReach = 1; qOut = 0;
          t = 0; emitCarry = 0; qAcc = 0;
          let mx = 0; for (let i = 0; i < NZ; i++) mx = Math.max(mx, 2 * uR[i] * uR[i]);
          pScale = 1.6 / mx;
          const tau = (YB2 - YB1) / P.vy;
          zth = P.F0 * tau * (tau / 2 + (YS - YB2) / P.vy);
        },
        step(dt) {
          // quantum packet (fixed inner steps)
          const tEnd = t + dt;
          qAcc += dt;
          let guard = 0;
          while (qAcc >= DTQ && guard < 12) { qStep(); qAcc -= DTQ; guard++; }
          if (guard >= 12) qAcc = 0;
          // classical beams: continuous emission
          emitCarry += P.rate * dt;
          const n = Math.floor(emitCarry); emitCarry -= n;
          for (let k = 0; k < n; k++) {
            const te = t + rng.next() * dt;
            emit(cl, te, P.F0 * rng.uniform(-1, 1));
            emit(sc, te, P.F0 * (rng.next() < 0.5 ? -1 : 1));
          }
          t = tEnd;
          collect(cl, false);
          const nh = collect(sc, true);
          sampleQ(nh);
        },
        render() {
          const tS = YS / P.vy;
          const pc = plots.c, ps = plots.s, pq = plots.q;
          for (const p of [pc, ps, pq]) { p.clear(); drawMagnet(p); }
          drawParticles(pc, cl, C_CL);
          drawRecent(pc, cl.rz, cl.rn, C_CL);
          drawParticles(ps, sc, C_SC);
          drawRecent(ps, sc.rz, sc.rn, C_SC);
          // quantum: beam trace + instantaneous packet
          if (P.trace && lastCol >= 0) pq.heatmap(trace, NYT, NZT, { x0: 0, x1: YMAX, y0: -ZLIM, y1: ZLIM, vmin: 0, vmax: 1.6, cmap: "teal", alpha: 0.75 });
          drawMagnet(pq);
          const yq = P.vy * tq;
          if (yq <= YMAX + 0.3) {
            for (const [re, im, rho, col] of [[uR, uI, rhoU, C_UP], [dR, dI, rhoD, C_DN]]) {
              for (let k = 0; k < NW; k++) {
                const i = i0 + k; rho[i] = re[i] * re[i] + im[i] * im[i];
                polyX[k] = yq - pScale * rho[i]; polyY[k] = zg[i];
                polyX[2 * NW - 1 - k] = yq; polyY[2 * NW - 1 - k] = zg[i];
              }
              pq.poly(polyX, polyY, { color: col, alpha: 0.55, stroke: col });
            }
          }
          drawRecent(pq, qRz, qRn, null, qRs);
          pq.legend([{ label: "spin ↑", color: C_UP, type: "box" }, { label: "spin ↓", color: C_DN, type: "box" }], "tl");
          const lab = `t = ${PM.fmt(t, 2)}`;
          pc.label(lab, "br"); ps.label(lab, "br");
          pq.label(`packet: y = ${PM.fmt(yq, 2)}`, "br");

          // histograms
          const hc = plots.ch, hs = plots.sh, hq = plots.qh;
          hc.clear(); hbars(hc, cl.hist, hmax(cl.hist), C_CL, 0.75);
          hs.clear(); hbars(hs, sc.hist, hmax(sc.hist), C_SC, 0.75);
          hq.clear();
          let qm = 0; for (let b = 0; b < NB; b++) qm = Math.max(qm, qHistU[b] + qHistD[b]);
          hbars(hq, qHistU, qm, C_UP, 0.55); hbars(hq, qHistD, qm, C_DN, 0.55);
          // instantaneous packet densities (as in the original), scaled to the maximum of the sum
          let rm = 0; for (let k = 0; k < NW; k++) rm = Math.max(rm, rhoU[i0 + k] + rhoD[i0 + k]);
          if (rm > 1e-4 && yq <= YMAX + 0.3) {
            for (const [rho, col, dash] of [[rhoU, C_UP, null], [rhoD, C_DN, null], [null, PlotColors.text, [5, 4]]]) {
              for (let k = 0; k < NW; k++) { const i = i0 + k; lineX[k] = (rho ? rho[i] : rhoU[i] + rhoD[i]) / rm; }
              hq.line(lineX, lineY, { color: col, width: dash ? 1.3 : 1.8, dash: dash || undefined, alpha: dash ? 0.7 : 1 });
            }
          }
          const cnt = (n, o) => (o > 0 ? [`${n} hits`, `${Math.round((100 * o) / n)} % off-screen`] : `${n} hits`);
          hc.label(cnt(cl.n, cl.out), "br", { size: 11 }); hs.label(cnt(sc.n, sc.out), "br", { size: 11 }); hq.label(cnt(qn, qOut), "br", { size: 11 });
          hq.legend([{ label: "|ψ↑|²", color: C_UP }, { label: "|ψ↓|²", color: C_DN }, { label: "total", color: PlotColors.text, dash: [5, 4] }], "tr");
          for (const h of [hc, hs, hq]) h.hline(0, { color: PlotColors.muted, dash: [2, 4], alpha: 0.5 });
          if (t < tS) for (const h of [hc, hs]) h.label("no hits yet", "tl", { size: 11 });

          // metrics
          const cstd = cl.n > 1 ? Math.sqrt(Math.max(cl.szz / cl.n - (cl.sz / cl.n) ** 2, 0)) : NaN;
          const sdz = sc.np > 0 && sc.nm > 0 ? Math.abs(sc.sp / sc.np - sc.sm / sc.nm) : NaN;
          Mt.set("t", PM.fmt(t, 2));
          Mt.set("cstd", PM.fmt(cstd, 2));
          Mt.set("sdz", PM.fmt(sdz, 2));
          Mt.set("qdz", haveScreen ? PM.fmt(qSep, 2) : "—");
          Mt.set("zth", PM.fmt(2 * zth, 2));
          api.setTime(`t = ${PM.fmt(t, 2)}`);
        },
        destroy() { if (wide.removeEventListener) wide.removeEventListener("change", applyCols); },
      };
    },
  });
})();
