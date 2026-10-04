/* Ideal gas: ensemble average vs time average.
 * One page, one shared set of gas parameters, and an "Averaging" selector (rebuild):
 *   ensemble — M independent, non-interacting 2D boxes advanced in parallel; the wall pressure is
 *              averaged over the boxes at each instant (Gibbs ensemble average).
 *   time     — a single box whose disks also collide elastically with each other (cell-list
 *              neighbour search); the wall pressure is averaged over time.
 * Both modes compare the measured pressure with P = N k_B T / A, the speed histogram with the 2D
 * Maxwell–Boltzmann distribution, and monitor the kinetic energy. Units: k_B = 1, m = 1. */
(function () {
  "use strict";

  const MASS = 1;
  const MAXPTS = 1500;  // max points kept in a time series (older points are thinned 2:1)
  const SB = 40;        // speed-histogram bins
  const HB = 40;        // per-box pressure histogram bins
  const SIM_RATE = 0.5; // simulated seconds per real second at speed ×1
  const ETA_MAX = 0.3;  // max disk area fraction in the collision mode

  /** Time series that thins itself 2:1 whenever it exceeds MAXPTS points. */
  class Series {
    constructor() { this.t = []; this.a = []; this.b = []; this.c = []; this.stride = 1; this.k = 0; }
    push(t, a, b, c) {
      if (this.k++ % this.stride !== 0) return;
      this.t.push(t); this.a.push(a); this.b.push(b); this.c.push(c);
      if (this.t.length > MAXPTS) {
        const f = (x) => x.filter((_, i) => i % 2 === 0);
        this.t = f(this.t); this.a = f(this.a); this.b = f(this.b); this.c = f(this.c); this.stride *= 2;
      }
    }
  }
  /** Henderson equation of state for hard disks: Z = PA/(Nk_BT) as a function of the area fraction η. */
  const hendersonZ = (eta) => (1 + (eta * eta) / 8) / ((1 - eta) * (1 - eta));
  /** Largest N (multiple of 10) with disk area fraction ≤ ETA_MAX. */
  const nMaxFor = (L, r) => PM.clamp(Math.floor((ETA_MAX * L * L) / (Math.PI * r * r) / 10) * 10, 30, 400);

  function theory(p) {
    const ens = p.avg !== "time";
    const L = +p.L, r = parseFloat(p.r), N = +p.N, T = +p.T;
    const eta = (N * Math.PI * r * r) / (L * L);
    const P0 = (N * T) / (L * L), fs = L / (L - 2 * r);
    return `
<div class="callout">${ens
      ? `<b>Ensemble average selected.</b> $M$ independent copies of the box are prepared with the same $(N, L, T)$ and evolved side by side; at every instant the pressure is averaged over the copies:
$$\\langle P\\rangle_{\\rm ens}(t)=\\frac1M\\sum_{k=1}^{M}P_k(t)\\quad\\longrightarrow\\quad \\frac{Nk_BT}{A}.$$
For the current settings $Nk_BT/A=${PM.fmt(P0, 2)}$ and the finite-size prediction is $${PM.fmt(fs, 4)}\\times$ this value.`
      : `<b>Time average selected.</b> One box is followed for a long time and its wall pressure is averaged over time:
$$\\bar P(\\tau)=\\frac1\\tau\\int_0^\\tau P(t)\\,dt\\quad\\longrightarrow\\quad \\frac{Nk_BT}{A}\\,Z(\\eta).$$
For the current settings $Nk_BT/A=${PM.fmt(P0, 2)}$, the disk area fraction is $\\eta=${PM.fmt(eta, 4)}$ and the expected ratio including finite size and hard-disk repulsion is $\\approx${PM.fmt(fs * hendersonZ(eta), 3)}$.`}</div>

<h4>The physical system</h4>
<p>A two-dimensional gas of $N$ identical disks of mass $m$ and radius $r$ in a square box of side $L$ (area $A=L^2$) with hard, smooth walls that reflect specularly.
Units: $k_B=1$ and $m=1$, so the temperature $T$ is an energy per particle, speeds are in m/s, lengths in m and time in s. Initial velocities are drawn either with uniformly
distributed magnitudes in $[1,v_{\\max}]$ and random directions (deliberately <i>not</i> Maxwellian) or with Gaussian components, and are then rescaled so that every box holds
exactly $E_k=Nk_BT$ — the equipartition value for two translational degrees of freedom. The two modes differ in what is averaged:</p>
<ul>
<li><b>Ensemble average:</b> $M$ independent boxes; the disks do not interact with each other (an ideal gas) and only bounce off the walls; $r$ only keeps the centres a distance $r$ from the walls.</li>
<li><b>Time average:</b> one box in which the disks also collide elastically with each other (a hard-disk gas), so energy is exchanged between particles.</li>
</ul>

<h4>Equations being solved</h4>
<p>Between collisions every particle moves freely, $\\dot{\\mathbf r}_i=\\mathbf v_i$. At a wall the normal component reverses, $v_\\perp\\to-v_\\perp$, and the wall receives the impulse $2m|v_\\perp|$.
In 2D the pressure is force per unit length of wall, so over a time $\\tau$</p>
$$P=\\frac{1}{4L\\,\\tau}\\sum_{\\text{wall hits}}2m|v_\\perp| .$$
<p>Kinetic theory: a particle bouncing between two walls a distance $\\ell=L-2r$ apart (its centre cannot get closer than $r$) hits one of them every $2\\ell/|v_x|$, so it exerts the mean force
$mv_x^2/\\ell$. Summing over particles and the four walls and using equipartition, $\\langle\\tfrac12mv_x^2\\rangle=\\langle\\tfrac12mv_y^2\\rangle=\\tfrac12k_BT$, gives</p>
$$P=\\frac{Nk_BT}{L\\,(L-2r)}\\;\\xrightarrow{\\;r\\ll L\\;}\\;\\frac{Nk_BT}{A},\\qquad E_k=\\sum_i\\tfrac12 m v_i^2=Nk_BT .$$
<p>The two averages are</p>
$$\\langle P\\rangle_{\\rm ens}(t)=\\frac1M\\sum_{k=1}^{M}P_k(t),\\qquad \\bar P(\\tau)=\\frac1\\tau\\int_0^\\tau P(t)\\,dt .$$
<p>The <b>ergodic hypothesis</b> states that for a system that explores its whole energy surface, $\\lim_{\\tau\\to\\infty}\\bar A(\\tau)=\\langle A\\rangle_{\\rm ens}$ for any observable $A$.
The statistical error shrinks like $1/\\sqrt M$ for the ensemble and like $\\sqrt{\\tau_c/\\tau}$ for the time average ($\\tau_c$: correlation time).</p>
<p>An elastic collision of two equal disks exchanges the velocity components along the unit vector $\\hat{\\mathbf n}$ joining their centres:
$\\mathbf v_i\\to\\mathbf v_i+[(\\mathbf v_j-\\mathbf v_i)\\cdot\\hat{\\mathbf n}]\\,\\hat{\\mathbf n}$, $\\mathbf v_j\\to\\mathbf v_j-[(\\mathbf v_j-\\mathbf v_i)\\cdot\\hat{\\mathbf n}]\\,\\hat{\\mathbf n}$, conserving momentum and kinetic energy.
Such collisions drive the speeds to the 2D Maxwell–Boltzmann distribution</p>
$$f(v)=\\frac{mv}{k_BT}\\,e^{-mv^2/2k_BT},\\qquad v_{\\rm mp}=\\sqrt{k_BT/m},\\quad \\langle v\\rangle=\\sqrt{\\pi k_BT/2m},\\quad v_{\\rm rms}=\\sqrt{2k_BT/m}.$$
<p>Hard disks are not an ideal gas: excluded area raises the pressure. With density $n=N/A$ and area fraction $\\eta=N\\pi r^2/A$,
the second virial coefficient gives $Z\\equiv PA/(Nk_BT)\\approx1+B_2n$ with $B_2=\\pi\\sigma^2/2=2\\pi r^2$ (i.e. $Z\\approx1+2\\eta$, $\\sigma=2r$);
the simulation compares with Henderson's equation of state $Z=(1+\\eta^2/8)/(1-\\eta)^2$.</p>

<h4>How the simulation solves them</h4>
<ul>
<li><b>Ensemble mode.</b> $M\\times N$ particles are stored in flat arrays. Each step of length $\\Delta t$ moves every particle exactly ($\\mathbf r\\to\\mathbf r+\\mathbf v\\Delta t$); a particle that crossed a wall is mirrored back and its impulse $2m|v_\\perp|$ is added to its own box.
The thin red curve is the instantaneous ensemble average $\\langle P\\rangle_{\\rm ens}=\\sum_k\\Delta p_k/(M\\cdot4L\\Delta t)$; the orange curve is its running time average (the metric);
the purple curve is box 1 alone, time-averaged. The histogram shows each box's pressure in the last step, or each box's own time average. Without collisions the speeds never change,
so the speed histogram (all boxes) is computed once.</li>
<li><b>Time mode.</b> The internal step is $h=\\min(\\Delta t,\\;0.1\\,r/v_{\\rm rms})$: a typical pair overlaps by only ~10% of $r$ before the contact is detected, and no pair can pass through another unless it approaches at more than 40 $v_{\\rm rms}$. After each move, overlapping pairs that are still approaching
are found with a cell list (cells of side $\\ge2r$, about one particle per cell, cost $O(N)$), receive the collision rule above and are pushed apart to contact along $\\hat{\\mathbf n}$.
Wall impulses are accumulated and $\\bar P(t)=\\sum\\Delta p/(4L\\,t)$ (orange); the red curve is a running average over 0.25 s. The speed histogram is an exponential moving average over ≈0.3 s.</li>
<li><b>Energy check.</b> Free flight, wall reflection and the collision rule all conserve $E_k$ exactly, so in both modes $E_k/(Nk_B)$ must stay equal to $T$; in the time mode the plot of $E_k/E_0$ shows
round-off-level drift. The optional rescaling $\\mathbf v\\to\\mathbf v\\sqrt{E_0/E_k}$ after every step (a crude thermostat) is therefore normally unnecessary. Total momentum is <i>not</i> conserved because the walls exert forces.</li>
<li><b>Measured / theory</b> is the measured average pressure divided by $Nk_BT/L^2$; the expected value is $L/(L-2r)$ for the ensemble and $\\approx L/(L-2r)\\cdot Z(\\eta)$ with collisions.
Simulated time runs at 0.5 s per real second at speed ×1.</li>
</ul>

<h4>What to try</h4>
<ol>
<li><b>1/√M.</b> In ensemble mode compare $M=100$ with $M=4000$: the scatter of the red curve shrinks by $\\sqrt{40}\\approx6.3$, while its mean stays at $L/(L-2r)\\cdot Nk_BT/L^2$.</li>
<li><b>Pressure does not need equilibrium.</b> With uniform initial speeds the ensemble-mode speed histogram stays flat-topped forever (no collisions, no thermalisation), yet the pressure agrees with $Nk_BT/A$, because it depends only on $\\langle v_\\perp^2\\rangle$.</li>
<li><b>Relaxation.</b> Switch to the time average with the same settings: within a few collision times (mean free path $\\ell\\approx1/(2\\sqrt2\\,n r)$, ≈3.5 m for $N=100$, $L=10$, $r=0.1$) the histogram relaxes to the Maxwell–Boltzmann curve.</li>
<li><b>Non-ideality.</b> In time mode set $r=0.2$ and the largest $N$: $\\eta\\approx0.29$ and the measured pressure is about twice $Nk_BT/A$, as Henderson's $Z\\approx2.0$ predicts. With $r=0.02$ the ratio returns to $\\approx1.00$ and the two averages agree — ergodicity at work.</li>
<li><b>Scaling.</b> Doubling $T$ doubles $P$; halving $L$ at fixed $N$ quadruples it.</li>
</ol>

<h4>Limitations &amp; further reading</h4>
<p>Classical, two-dimensional, equal masses. Collisions are time-stepped rather than event-driven, so contacts are resolved up to $O(v h)$ and very dense systems are approximate.
The non-interacting gas is not ergodic: in a square box each particle keeps its own $|v_x|$ and $|v_y|$ forever, which is why only collisions produce the Maxwell–Boltzmann distribution.
Further reading: F. Reif, <i>Fundamentals of Statistical and Thermal Physics</i>; D. V. Schroeder, <i>An Introduction to Thermal Physics</i>; M. Kardar, <i>Statistical Physics of Particles</i>;
M. P. Allen &amp; D. J. Tildesley, <i>Computer Simulation of Liquids</i>; D. Henderson, Mol. Phys. 30, 971 (1975).</p>`;
  }

  App.register({
    id: "ideal-gas",
    category: "statistical",
    order: 23,
    title: "Ideal Gas: Ensemble vs Time Average",
    icon: "🎈",
    subtitle: "Measure the pressure of a 2D gas either by averaging over many independent boxes at one instant or by following one colliding box over time, and compare both with P = Nk<sub>B</sub>T/A and the Maxwell–Boltzmann distribution.",
    notes: [{ type: "info", html: "Use the <b>Averaging</b> selector to choose what the average is taken over. The gas parameters are shared, so switching back and forth compares the two averages for the same $(N, L, T)$ — the ergodic hypothesis says they must agree." }],
    animated: true,
    speed: { min: 0.1, max: 5, value: 1, step: 0.1 },
    controls: [
      { id: "avg", type: "select", label: "Averaging", value: "ensemble", rebuild: true,
        options: [{ value: "ensemble", label: "Ensemble average (M boxes)" }, { value: "time", label: "Time average (one box)" }] },
      { id: "infoEns", type: "info", visibleIf: (p) => p.avg !== "time",
        html: "<b>Ensemble average:</b> many independent boxes, no particle–particle collisions; the pressure is averaged over all boxes at each instant." },
      { id: "infoTime", type: "info", visibleIf: (p) => p.avg === "time",
        html: "<b>Time average:</b> one box whose disks also collide elastically with each other; the pressure is averaged over time." },
      { type: "section", label: "Gas (shared by both modes)" },
      { id: "N", type: "slider", label: "Particles per box $N$", min: 10, max: 400, step: 10, value: 100,
        help: "With collisions the maximum is limited so that the disks cover at most 30% of the box." },
      { id: "L", type: "slider", label: "Box side $L$", min: 5, max: 20, step: 0.5, value: 10, unit: "m" },
      { id: "T", type: "slider", label: "Temperature $T$ (energy units, $k_B = 1$)", min: 50, max: 1000, step: 10, value: 300 },
      { id: "r", type: "select", label: "Disk radius $r$", value: "0.1",
        options: ["0.02", "0.05", "0.1", "0.15", "0.2"].map((v) => ({ value: v, label: v + " m" })),
        help: "Without collisions $r$ only keeps the centres $r$ away from the walls; with collisions the disks have diameter $2r$." },
      { id: "init", type: "select", label: "Initial speeds", value: "uniform",
        options: [{ value: "uniform", label: "Uniform speeds (not thermal)" }, { value: "mb", label: "Maxwell–Boltzmann" }],
        help: "Uniform: magnitudes uniform in $[1, v_{\\max}]$, random directions. Maxwell–Boltzmann: Gaussian velocity components. Either way the speeds are rescaled so that each box holds exactly $E_k = Nk_BT$." },
      { id: "vmax", type: "slider", label: "Speed range $[1, v_{\\max}]$ before rescaling", min: 1.5, max: 10, step: 0.5, value: 5,
        visibleIf: (p) => p.init === "uniform" },
      { id: "dt", type: "select", label: "Time step / pressure sampling $\\Delta t$", value: "0.005",
        options: ["0.001", "0.0025", "0.005", "0.01"].map((v) => ({ value: v, label: v + " s" })) },
      { id: "secEns", type: "section", label: "Ensemble average", visibleIf: (p) => p.avg !== "time" },
      { id: "M", type: "slider", label: "Number of boxes $M$", min: 100, max: 4000, step: 100, value: 1000, visibleIf: (p) => p.avg !== "time" },
      { id: "hist", type: "select", label: "Per-box pressure histogram", value: "inst", live: true, visibleIf: (p) => p.avg !== "time",
        options: [{ value: "inst", label: "Instantaneous (last step)" }, { value: "cum", label: "Each box's own time average" }] },
      { id: "secTime", type: "section", label: "Time average", visibleIf: (p) => p.avg === "time" },
      { id: "rescale", type: "checkbox", label: "Rescale $E_k$ to $E_0$ after every step (thermostat)", value: false, live: true, visibleIf: (p) => p.avg === "time",
        help: "Elastic collisions already conserve energy exactly, so this changes nothing but round-off." },
    ],
    theory,

    mount(api) {
      const P = api.params;
      const ENS = P.avg !== "time";
      const plots = api.plots([
        { id: "box", title: ENS ? "Four of the M boxes (colour: speed |v|)" : "The box (colour: speed |v|)", aspect: 0.9, maxHeight: 520, equal: true, axes: false, colorbar: true },
        { id: "pres", title: ENS ? "Pressure: ensemble average" : "Pressure: time average of one box", aspect: 0.9, maxHeight: 520, xlabel: "t (s)", ylabel: "P (force / length)" },
        { id: "spd", title: ENS ? "Speed distribution of all boxes vs 2D Maxwell–Boltzmann" : "Speed distribution (moving average) vs 2D Maxwell–Boltzmann", aspect: 0.62, xlabel: "speed |v| (m/s)", ylabel: "f(|v|)" },
        ENS
          ? { id: "aux", title: "Pressure of the individual boxes", aspect: 0.62, xlabel: "P", ylabel: "probability density" }
          : { id: "aux", title: "Energy and momentum (conservation check)", aspect: 0.62, xlabel: "t (s)", ylabel: "E_k / E₀" },
      ]);
      const M = api.metrics([
        { id: "t", label: "Time $t$" },
        { id: "pm", label: ENS ? "Measured $\\overline{\\langle P\\rangle}_{\\rm ens}$" : "Measured $\\bar P$ (time average)" },
        { id: "pt", label: "Ideal gas $Nk_BT/L^2$" },
        { id: "ratio", label: "Measured / theory" },
        { id: "exp", label: ENS ? "Expected $L/(L-2r)$" : "Expected $\\frac{L}{L-2r}Z(\\eta)$" },
        { id: "T", label: "Temperature $E_k/(Nk_B)$" },
        ENS ? { id: "x", label: "Particles simulated" } : { id: "x", label: "Collision rate" },
      ]);
      const cLUT = Array.from({ length: 64 }, (_, k) => colormap("plasma", 0.08 + 0.9 * (k / 63)));
      let L, rad, T, dt, N, t, steps, acc, Ptheo, Pfs, Pexp, vmp, spdX, spdMB, spdMax, ser, cols;

      function initVelocities(rng, vx, vy, i0, n) {
        let ke = 0;
        for (let i = i0; i < i0 + n; i++) {
          let a, b;
          if (P.init === "mb") { a = rng.gauss(0, Math.sqrt(T / MASS)); b = rng.gauss(0, Math.sqrt(T / MASS)); }
          else { const ang = rng.uniform(0, 2 * Math.PI), s = rng.uniform(1, P.vmax); a = s * Math.cos(ang); b = s * Math.sin(ang); }
          vx[i] = a; vy[i] = b; ke += 0.5 * MASS * (a * a + b * b);
        }
        return ke;
      }
      function speedColor(v) { return cLUT[Math.min(63, Math.floor((v / (3 * vmp)) * 63))]; }
      function commonReset() {
        L = P.L; rad = parseFloat(P.r); T = P.T; dt = parseFloat(P.dt);
        t = 0; steps = 0; acc = 0; ser = new Series();
        Ptheo = (N * T) / (L * L); Pfs = (N * T) / (L * (L - 2 * rad));
        vmp = Math.sqrt(T / MASS); spdMax = 4 * vmp;
        spdX = PM.linspace(0, spdMax, 300);
        spdMB = spdX.map((v) => ((MASS * v) / T) * Math.exp((-MASS * v * v) / (2 * T)));
      }
      /** Pressure axis: data expected within center·(1 ± 3.5 s); the lower part is left free for the legend. */
      function pressureLimits(center, s, x1) {
        plots.pres.setLimits([0, x1], [Math.min(center * (1 - 7.4 * s), Ptheo * (1 - 2 * s)), center * (1 + 3.5 * s)]);
      }
      function drawPressure(lines, legend, corner) {
        const pp = plots.pres;
        pp.clear();
        pp.hline(Pfs, { color: PlotColors.muted, dash: [2, 4], width: 1.2 });
        pp.hline(Ptheo, { color: PlotColors.blue, dash: [7, 4], width: 1.8 });
        if (!ENS) pp.hline(Pexp, { color: PlotColors.good, dash: [10, 4, 2, 4], width: 1.4 });
        if (ser.t.length > 1) for (const [ys, o] of lines) pp.line(ser.t, ys, o);
        pp.legend(legend, corner);
      }
      function drawSpeeds(centers, counts, width, label) {
        const ps = plots.spd;
        ps.clear();
        ps.bars(centers, counts, width, { color: PlotColors.accent2, alpha: 0.7, gap: 1 });
        ps.line(spdX, spdMB, { color: PlotColors.accent3, width: 2.2 });
        ps.vline(vmp, { color: PlotColors.muted, dash: [3, 4], width: 1 });
        ps.legend([{ label, color: PlotColors.accent2, type: "box" }, { label: "2D Maxwell–Boltzmann", color: PlotColors.accent3 }], "tr");
      }
      function setCommonMetrics(pm, Tnow) {
        M.set("t", PM.fmt(t, 2) + " s");
        M.set("pm", steps ? PM.fmt(pm, 2) : "—");
        M.set("pt", PM.fmt(Ptheo, 2));
        M.set("ratio", steps ? PM.fmt(pm / Ptheo, 3) : "—");
        M.set("exp", PM.fmt(Pexp / Ptheo, 3));
        M.set("T", PM.fmt(Tnow, 1));
        api.setTime(`t = ${PM.fmt(t, 2)} s`);
      }

      // ================================================================ ensemble mode
      if (ENS) {
        let nM, NT, X, Y, VX, VY, dpBox, cumBox, sumP, budget = 0, spdHist, boxCols, histC, histE, hlo, hhi, ymaxH = 1, keBox0;
        const bx = new Float64Array(400), by = new Float64Array(400);
        function setHistRange() {
          if (P.hist === "inst") {
            const hits = (2 * N * Math.sqrt((2 * T) / (Math.PI * MASS)) * dt) / (L - 2 * rad);
            hlo = 0; hhi = Pfs * (1 + 4.5 / Math.sqrt(Math.max(hits, 0.05)));
          } else { hlo = Pfs * 0.5; hhi = Pfs * 1.5; }
          histC.fill(0); histE = null; ymaxH = 1e-9;
          plots.aux.setLimits([hlo, hhi]);
        }
        function physStep() {
          const lo = rad, hi = L - rad;
          let tot = 0;
          for (let b = 0; b < nM; b++) {
            let dp = 0;
            const e = (b + 1) * N;
            for (let i = b * N; i < e; i++) {
              const vx = VX[i], vy = VY[i];
              let x = X[i] + vx * dt, y = Y[i] + vy * dt;
              if (x < lo) { x = Math.min(2 * lo - x, hi); VX[i] = -vx; dp += 2 * MASS * Math.abs(vx); }
              else if (x > hi) { x = Math.max(2 * hi - x, lo); VX[i] = -vx; dp += 2 * MASS * Math.abs(vx); }
              if (y < lo) { y = Math.min(2 * lo - y, hi); VY[i] = -vy; dp += 2 * MASS * Math.abs(vy); }
              else if (y > hi) { y = Math.max(2 * hi - y, lo); VY[i] = -vy; dp += 2 * MASS * Math.abs(vy); }
              X[i] = x; Y[i] = y;
            }
            dpBox[b] = dp; cumBox[b] += dp; tot += dp;
          }
          steps++; t = steps * dt;
          const pInst = tot / nM / (dt * 4 * L);
          sumP += pInst;
          ser.push(t, pInst, sumP / steps, cumBox[0] / (t * 4 * L));
        }
        return {
          reset() {
            nM = Math.round(P.M); N = Math.round(P.N); NT = nM * N;
            commonReset();
            Pexp = Pfs;
            X = new Float32Array(NT); Y = new Float32Array(NT); VX = new Float32Array(NT); VY = new Float32Array(NT);
            dpBox = new Float64Array(nM); cumBox = new Float64Array(nM);
            const rng = new PM.RNG(20240 + nM + N);
            for (let b = 0; b < nM; b++) {
              for (let i = b * N; i < (b + 1) * N; i++) { X[i] = rng.uniform(rad, L - rad); Y[i] = rng.uniform(rad, L - rad); }
              const f = Math.sqrt((N * T) / initVelocities(rng, VX, VY, b * N, N));
              for (let i = b * N; i < (b + 1) * N; i++) { VX[i] *= f; VY[i] *= f; }
            }
            // kinetic energy check (Float32 storage): temperature of box 1
            keBox0 = 0; for (let i = 0; i < N; i++) keBox0 += 0.5 * MASS * (VX[i] * VX[i] + VY[i] * VY[i]);
            sumP = 0;
            // expected relative fluctuation of ⟨P⟩_ens per step: ~1/sqrt(hits per step × M)
            const hits = (2 * N * Math.sqrt((2 * T) / (Math.PI * MASS)) * dt) / (L - 2 * rad);
            pressureLimits(Pfs, PM.clamp(1.3 / Math.sqrt(hits * nM), 0.008, 0.12), 2);
            budget = 0;
            // speed histogram: no collisions, so it never changes
            const ns = Math.min(NT, 300000), stride = Math.max(1, Math.floor(NT / ns)), sp = new Float64Array(ns);
            for (let k = 0; k < ns; k++) { const i = k * stride; sp[k] = Math.hypot(VX[i], VY[i]); }
            spdHist = PM.histogram(sp, SB, 0, spdMax, true);
            plots.spd.setLimits([0, spdMax], [0, Math.max(PM.max(spdHist.counts), PM.max(spdMB)) * 1.15]);
            boxCols = [];
            for (let i = 0; i < Math.min(4, nM) * N; i++) boxCols.push(speedColor(Math.hypot(VX[i], VY[i])));
            cols = new Array(N);
            const bw = 2 * L + L * 0.08;
            plots.box.setLimits([-0.02 * L, bw + 0.02 * L], [-0.02 * L, bw + 0.02 * L]);
            histC = new Float64Array(HB); histE = null;
            setHistRange();
          },
          onParam(id) { if (id === "hist" && histC) setHistRange(); },
          step(dts) {
            // at most ~5·10⁵ particle moves per frame: very large ensembles advance every few frames
            acc += dts * SIM_RATE;
            budget = Math.min(budget + 5e5, 5e5 + NT);
            let n = Math.floor(acc / dt);
            acc -= n * dt;
            while (n > 0 && budget >= NT) { physStep(); budget -= NT; n--; }
            if (n > 0) acc = 0; // drop what could not be done in this frame
            if (t > plots.pres.xlim[1]) plots.pres.setLimits([0, t * 1.6]);
          },
          render() {
            // four sample boxes
            const pb = plots.box;
            pb.clear();
            const nb = Math.min(4, nM), gap = L * 0.08;
            for (let b = 0; b < nb; b++) {
              const ox = (b % 2) * (L + gap), oy = (1 - Math.floor(b / 2)) * (L + gap);
              pb.rect(ox, oy, ox + L, oy + L, { color: "#0b1016", stroke: "#58a6ff", strokeWidth: 1.5 });
              for (let k = 0; k < N; k++) { bx[k] = ox + X[b * N + k]; by[k] = oy + Y[b * N + k]; cols[k] = boxCols[b * N + k]; }
              pb.points(bx.subarray(0, N), by.subarray(0, N), { colors: cols, size: Math.max(1.6, rad * pb.sx), alpha: 0.95 });
              const pbx = t > 0 ? cumBox[b] / (t * 4 * L) : 0;
              pb.text(ox + 0.03 * L, oy + L - 0.05 * L, `box ${b + 1}: P̄ = ${t > 0 ? PM.fmt(pbx, 1) : "—"}`, { size: 11, bg: "#0b1016", baseline: "top" });
            }
            pb.colorbar({ cmap: "plasma", vmin: 0, vmax: 3 * vmp, label: "|v| (m/s)" });

            drawPressure([
              [ser.a, { color: PlotColors.bad, width: 1, alpha: 0.45 }],
              [ser.c, { color: PlotColors.accent2, width: 1.4, alpha: 0.85 }],
              [ser.b, { color: PlotColors.accent3, width: 2.4 }],
            ], [
              { label: "⟨P⟩ens at each instant", color: PlotColors.bad },
              { label: "running time average of ⟨P⟩ens", color: PlotColors.accent3 },
              { label: "box 1 alone, time-averaged", color: PlotColors.accent2 },
              { label: `NkT/L² = ${PM.fmt(Ptheo, 1)}`, color: PlotColors.blue, dash: [7, 4] },
              { label: "finite size NkT/[L(L−2r)]", color: PlotColors.muted, dash: [2, 4] },
            ], "br");

            // per-box pressure histogram
            const ph = plots.aux;
            const vals = new Float64Array(nM);
            if (P.hist === "inst") for (let b = 0; b < nM; b++) vals[b] = dpBox[b] / (dt * 4 * L);
            else for (let b = 0; b < nM; b++) vals[b] = t > 0 ? cumBox[b] / (t * 4 * L) : 0;
            const hst = PM.histogram(vals, HB, hlo, hhi, true);
            if (steps > 0) {
              if (P.hist === "inst" && histE) for (let k = 0; k < HB; k++) histC[k] += (hst.counts[k] - histC[k]) * 0.15;
              else histC.set(hst.counts);
              histE = true;
            }
            ymaxH = Math.max(ymaxH * 0.995, PM.max(histC) * 1.15, 1e-9);
            ph.setLimits(null, [0, ymaxH]);
            ph.clear();
            ph.bars(hst.centers, histC, hst.width, { color: PlotColors.accent, alpha: 0.75, gap: 1 });
            let mean = 0; for (let b = 0; b < nM; b++) mean += vals[b]; mean /= nM;
            ph.vline(Ptheo, { color: PlotColors.blue, dash: [7, 4], width: 1.6 });
            if (steps > 0) ph.vline(mean, { color: PlotColors.bad, width: 1.6 });
            ph.legend([{ label: `mean over boxes ${steps ? PM.fmt(mean, 1) : "—"}`, color: PlotColors.bad }, { label: `NkT/L² = ${PM.fmt(Ptheo, 1)}`, color: PlotColors.blue, dash: [7, 4] }], "tr");
            ph.label(P.hist === "inst" ? `${nM} boxes · last step` : `${nM} boxes · averaged over ${PM.fmt(t, 2)} s`, "tl", { size: 11 });

            drawSpeeds(spdHist.centers, spdHist.counts, spdHist.width, "simulation (constant: no collisions)");

            setCommonMetrics(steps ? sumP / steps : NaN, keBox0 / N);
            M.set("x", NT.toLocaleString("en-US"));
          },
        };
      }

      // ================================================================ time-average mode
      let x, y, vx, vy, head, next, ncell, cs, accMom, ncol, colRate, E0, sub, h, pEma, spdC, spdCen, spdW, emin, emax, spBuf;
      function kinetic() { let k = 0; for (let i = 0; i < N; i++) k += vx[i] * vx[i] + vy[i] * vy[i]; return 0.5 * MASS * k; }
      function limitN() {
        const mx = nMaxFor(P.L, parseFloat(P.r));
        api.setControl("N", { max: mx, value: Math.min(P.N, mx) });
      }
      function zeroMomentum() {
        let mx = 0, my = 0;
        for (let i = 0; i < N; i++) { mx += vx[i]; my += vy[i]; }
        mx /= N; my /= N;
        for (let i = 0; i < N; i++) { vx[i] -= mx; vy[i] -= my; }
      }
      function collide() {
        head.fill(-1);
        for (let i = 0; i < N; i++) {
          const cx = Math.min(ncell - 1, Math.max(0, Math.floor(x[i] / cs))), cy = Math.min(ncell - 1, Math.max(0, Math.floor(y[i] / cs)));
          const c = cx * ncell + cy;
          next[i] = head[c]; head[c] = i;
        }
        const d2 = 4 * rad * rad;
        for (let cx = 0; cx < ncell; cx++) for (let cy = 0; cy < ncell; cy++) {
          for (let i = head[cx * ncell + cy]; i >= 0; i = next[i]) {
            for (let dx = -1; dx <= 1; dx++) {
              const nx = cx + dx; if (nx < 0 || nx >= ncell) continue;
              for (let dy = -1; dy <= 1; dy++) {
                const ny = cy + dy; if (ny < 0 || ny >= ncell) continue;
                for (let j = head[nx * ncell + ny]; j >= 0; j = next[j]) {
                  if (j <= i) continue;
                  const ex = x[j] - x[i], ey = y[j] - y[i], dd = ex * ex + ey * ey;
                  if (dd < d2 && dd > 1e-12) {
                    const dist = Math.sqrt(dd), ux = ex / dist, uy = ey / dist;
                    const dot = (vx[j] - vx[i]) * ux + (vy[j] - vy[i]) * uy;
                    if (dot < 0) { // approaching: exchange the normal velocity components
                      vx[i] += dot * ux; vy[i] += dot * uy; vx[j] -= dot * ux; vy[j] -= dot * uy;
                      const ov = 0.5 * (2 * rad - dist);
                      x[i] -= ov * ux; y[i] -= ov * uy; x[j] += ov * ux; y[j] += ov * uy;
                      ncol++;
                    }
                  }
                }
              }
            }
          }
        }
      }
      /** One sampling interval Δt = sub internal steps of length h. Returns the wall impulse. */
      function physStep() {
        const lo = rad, hi = L - rad;
        let dp = 0;
        for (let s = 0; s < sub; s++) {
          for (let i = 0; i < N; i++) {
            let xi = x[i] + vx[i] * h, yi = y[i] + vy[i] * h;
            if (xi < lo) { xi = Math.min(2 * lo - xi, hi); dp += 2 * MASS * Math.abs(vx[i]); vx[i] = -vx[i]; }
            else if (xi > hi) { xi = Math.max(2 * hi - xi, lo); dp += 2 * MASS * Math.abs(vx[i]); vx[i] = -vx[i]; }
            if (yi < lo) { yi = Math.min(2 * lo - yi, hi); dp += 2 * MASS * Math.abs(vy[i]); vy[i] = -vy[i]; }
            else if (yi > hi) { yi = Math.max(2 * hi - yi, lo); dp += 2 * MASS * Math.abs(vy[i]); vy[i] = -vy[i]; }
            x[i] = xi; y[i] = yi;
          }
          collide();
          // the push-apart may move a disk slightly past a wall: clamp (no impulse, velocity unchanged)
          for (let i = 0; i < N; i++) { if (x[i] < lo) x[i] = lo; else if (x[i] > hi) x[i] = hi; if (y[i] < lo) y[i] = lo; else if (y[i] > hi) y[i] = hi; }
          if (P.rescale) {
            const ke = kinetic();
            if (ke > 1e-10) { const f = Math.sqrt(E0 / ke); for (let i = 0; i < N; i++) { vx[i] *= f; vy[i] *= f; } }
          }
        }
        accMom += dp;
        steps++; t = steps * dt;
        const pInst = dp / (dt * 4 * L);
        const a = 1 - Math.exp(-dt / 0.25);
        pEma = steps === 1 ? pInst : pEma + (pInst - pEma) * a;
        ser.push(t, pEma, accMom / (t * 4 * L), kinetic() / E0);
      }
      function speedHist() {
        for (let i = 0; i < N; i++) spBuf[i] = Math.hypot(vx[i], vy[i]);
        return PM.histogram(spBuf, SB, 0, spdMax, true);
      }
      return {
        reset() {
          limitN();
          N = Math.round(P.N);
          commonReset();
          const eta = (N * Math.PI * rad * rad) / (L * L);
          Pexp = Pfs * hendersonZ(eta);
          x = new Float64Array(N); y = new Float64Array(N); vx = new Float64Array(N); vy = new Float64Array(N);
          spBuf = new Float64Array(N);
          const rng = new PM.RNG(7 + N);
          // place on a jittered grid (no initial overlaps)
          const ncol0 = Math.floor(Math.sqrt(N)) + 1, rows = Math.ceil(N / ncol0);
          const sp = (L - 2 * rad) / Math.max(ncol0, rows);
          let k = 0;
          for (let i = 0; i < rows && k < N; i++) for (let j = 0; j < ncol0 && k < N; j++, k++) {
            x[k] = PM.clamp(rad + sp * (j + 0.5) + rng.uniform(-sp * 0.1, sp * 0.1), rad, L - rad);
            y[k] = PM.clamp(rad + sp * (i + 0.5) + rng.uniform(-sp * 0.1, sp * 0.1), rad, L - rad);
          }
          initVelocities(rng, vx, vy, 0, N);
          zeroMomentum();
          E0 = N * T;
          const f = Math.sqrt(E0 / kinetic());
          for (let i = 0; i < N; i++) { vx[i] *= f; vy[i] *= f; }
          // internal step: typical overlap at detection ~0.1 r (keeps the collisional pressure accurate)
          const vrms = Math.sqrt((2 * T) / MASS);
          sub = Math.max(1, Math.ceil(dt / ((0.1 * rad) / vrms))); h = dt / sub;
          // cell list: side ≥ 2r, about one particle per cell
          cs = Math.max(2 * rad, L / Math.ceil(Math.sqrt(N)));
          ncell = Math.max(1, Math.floor(L / cs)); cs = L / ncell;
          head = new Int32Array(ncell * ncell); next = new Int32Array(N);
          accMom = 0; ncol = 0; colRate = 0; pEma = 0;
          spdW = spdMax / SB; spdCen = new Float64Array(SB);
          for (let q = 0; q < SB; q++) spdCen[q] = (q + 0.5) * spdW;
          spdC = new Float64Array(SB); spdC.set(speedHist().counts);
          plots.spd.setLimits([0, spdMax], [0, PM.max(spdMB) * 1.9]);
          const hitRate = (2 * N * Math.sqrt((2 * T) / (Math.PI * MASS))) / (L - 2 * rad);
          pressureLimits(Pexp, PM.clamp(1.3 / Math.sqrt(hitRate * 0.5), 0.02, 0.12), 2);
          plots.aux.setLimits([0, 2], [0.99, 1.01]);
          emin = emax = 1;
          plots.box.setLimits([0, L], [0, L]);
          cols = new Array(N);
        },
        onParam(id) { if (id === "L" || id === "r") limitN(); },
        step(dts) {
          acc += dts * SIM_RATE;
          const cap = Math.max(1, Math.floor(1.5e5 / (N * sub)));
          let n = Math.floor(acc / dt);
          if (n > cap) { n = cap; acc = 0; } else acc -= n * dt;
          const c0 = ncol;
          for (let k = 0; k < n; k++) physStep();
          if (n > 0) {
            colRate += ((ncol - c0) / (n * dt) - colRate) * Math.min(1, n * dt * 2);
            const hs = speedHist(), a = Math.min(1, n * dt * 3);
            for (let q = 0; q < SB; q++) spdC[q] += (hs.counts[q] - spdC[q]) * a;
          }
          if (steps > 40) { // widen the pressure axis if the measured value leaves it (dense hard-disk gas)
            const v = ser.b[ser.b.length - 1], yl = plots.pres.ylim;
            if (v > yl[1] * 0.95) plots.pres.setLimits(null, [yl[0], v * 1.25]);
            else if (v < yl[0] * 1.05) plots.pres.setLimits(null, [v * 0.8, yl[1]]);
          }
          if (t > plots.pres.xlim[1]) { plots.pres.setLimits([0, t * 1.6]); plots.aux.setLimits([0, t * 1.6]); }
        },
        render() {
          const pb = plots.box;
          pb.clear();
          pb.rect(0, 0, L, L, { color: "#0b1016", stroke: "#58a6ff", strokeWidth: 2 });
          for (let i = 0; i < N; i++) cols[i] = speedColor(Math.hypot(vx[i], vy[i]));
          pb.points(x, y, { colors: cols, size: Math.max(1.8, rad * pb.sx), alpha: 0.92 });
          pb.colorbar({ cmap: "plasma", vmin: 0, vmax: 3 * vmp, label: "|v| (m/s)" });
          pb.label([`N = ${N}`, `η = ${PM.fmt((N * Math.PI * rad * rad) / (L * L), 3)}`], "tl", { size: 11.5, color: PlotColors.good });

          drawPressure([
            [ser.a, { color: PlotColors.bad, width: 1, alpha: 0.5 }],
            [ser.b, { color: PlotColors.accent3, width: 2.4 }],
          ], [
            { label: "P, running 0.25 s average", color: PlotColors.bad },
            { label: "time average P̄(t)", color: PlotColors.accent3 },
            { label: `NkT/L² = ${PM.fmt(Ptheo, 1)}`, color: PlotColors.blue, dash: [7, 4] },
            { label: "finite size NkT/[L(L−2r)]", color: PlotColors.muted, dash: [2, 4] },
            { label: "hard disks: × Z(η)", color: PlotColors.good, dash: [10, 4, 2, 4] },
          ], "br");

          drawSpeeds(spdCen, spdC, spdW, "simulation (moving average)");

          const pe = plots.aux;
          if (ser.t.length > 1) {
            for (let k = Math.max(0, ser.c.length - 50); k < ser.c.length; k++) { emin = Math.min(emin, ser.c[k]); emax = Math.max(emax, ser.c[k]); }
            const half = Math.max(0.005, (emax - emin) * 0.8);
            pe.setLimits(null, [(emin + emax) / 2 - half, (emin + emax) / 2 + half]);
          }
          pe.clear();
          pe.hline(1, { color: PlotColors.accent3, dash: [6, 4], width: 1.2 });
          if (ser.t.length > 1) pe.line(ser.t, ser.c, { color: PlotColors.good, width: 2 });
          const ke = kinetic();
          let px = 0, py = 0; for (let i = 0; i < N; i++) { px += MASS * vx[i]; py += MASS * vy[i]; }
          pe.label([`E_k / E₀ − 1 = ${PM.fmt(ke / E0 - 1, 2)}`, `total momentum: pₓ = ${PM.fmt(px, 2)}, p_y = ${PM.fmt(py, 2)}`], "tl", { size: 11.5 });

          setCommonMetrics(steps ? accMom / (t * 4 * L) : NaN, ke / N);
          M.set("x", PM.fmt(colRate, 0) + " /s");
        },
      };
    },
  });
})();
