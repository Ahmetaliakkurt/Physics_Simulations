/* Central limit theorem — a live stream of walkers whose steps are right / left / stay with a U(0,1) step length.
 * Walkers start continuously in groups; the final positions of those that complete N steps fill the histogram. */
App.register({
  id: "central-limit-theorem",
  category: "statistical",
  order: 21,
  title: "Central Limit Theorem",
  icon: "🔔",
  subtitle: "Walkers take steps of random direction and random length; although a single step is flat, lopsided and has a spike at zero, the histogram of the sum of $N$ steps fills out a Gaussian, while measured skewness and kurtosis decay as $k^{-1/2}$ and $k^{-1}$.",
  notes: [{ type: "info", html: "At each step a particle moves right (probability $p$), left ($q$) or stays put ($n=1-p-q$); the step length is drawn afresh from a uniform distribution between 0 and 1 m. The distribution of a single step (lower left) is far from Gaussian, yet the sum of $N$ steps converges to a Gaussian, as the central limit theorem demands." }],
  animated: true,
  speed: { min: 0.1, max: 5, value: 1, step: 0.1 },
  controls: [
    { id: "N", type: "slider", label: "Number of steps $N$", min: 10, max: 300, step: 10, value: 100 },
    { id: "W", type: "slider", label: "Number of walks $M$", min: 1000, max: 100000, step: 1000, value: 20000,
      fmt: (v) => v.toLocaleString("en-US") },
    { id: "p", type: "slider", label: "Probability of a right step $p$", min: 0, max: 1, step: 0.05, value: 0.4 },
    { id: "q", type: "slider", label: "Probability of a left step $q$", min: 0, max: 1, step: 0.05, value: 0.4,
      help: "Automatically limited so that $p+q\\le 1$; the probability of staying put is $n=1-p-q$." },
    { id: "seed", type: "number", label: "Random seed", min: 0, max: 999999, step: 1, value: 7 },
    { type: "section", label: "Display" },
    { id: "S", type: "slider", label: "Trajectories drawn", min: 10, max: 300, step: 10, value: 80, live: true },
    { id: "band", type: "checkbox", label: "Show the theoretical $\\langle x\\rangle \\pm 2\\sigma$ envelope", value: true, live: true },
    { id: "finish", type: "button", label: "⏩ Finish all remaining walks now" },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A particle on a line starts at $x=0$ and takes $N$ independent steps. In each step it first chooses a direction
    $d\\in\\{+1,-1,0\\}$ with probabilities $p$, $q$ and $n=1-p-q$, and then a length $u$ drawn uniformly from $[0,1]$ m, so the
    displacement is $X=d\\,u$ (in metres). The final position is the sum $S_N=X_1+\\dots+X_N$. This is a caricature of many physical
    sums of random contributions: a Brownian particle receiving kicks of random size, the total energy of many weakly coupled
    subsystems, or the measurement error built from many small independent errors. Parameters: $N$ (steps), $M$ (number of
    simulated walks), $p$, $q$, and a random seed.</p>

    <h4>Equations being solved</h4>
    <p><b>The single-step distribution</b> is a mixture of two flat pieces and a delta function,</p>
    $$\\rho_1(x)=q\\,\\mathbf 1_{[-1,0)}(x)+p\\,\\mathbf 1_{(0,1]}(x)+n\\,\\delta(x),$$
    <p>which is shown in the lower-left plot: it is anything but Gaussian. Because $d$ and $u$ are independent,
    $\\langle X^m\\rangle=\\langle d^m\\rangle\\langle u^m\\rangle$ with $\\langle d^m\\rangle=p+(-1)^mq$ and
    $\\langle u^m\\rangle=\\int_0^1u^m\\,du=\\frac1{m+1}$:</p>
    $$\\langle X^m\\rangle=\\frac{p+(-1)^m q}{m+1}\\quad\\Rightarrow\\quad
      \\mu_1=\\langle X\\rangle=\\frac{p-q}{2},\\qquad \\sigma_1^2=\\langle X^2\\rangle-\\mu_1^2=\\frac{p+q}{3}-\\frac{(p-q)^2}{4}.$$
    <p><b>Central limit theorem.</b> For independent, identically distributed steps with finite variance, the cumulants of the sum
    are $N$ times the single-step cumulants; after standardising, all cumulants beyond the second vanish as $N\\to\\infty$:</p>
    <div class="callout">$$S_N=\\sum_{i=1}^{N}X_i\\ \\xrightarrow{\\;N\\to\\infty\\;}\\ \\mathcal N\\!\\big(N\\mu_1,\\ N\\sigma_1^2\\big),\\qquad
      \\rho_N(x)\\approx\\frac{1}{\\sqrt{2\\pi N\\sigma_1^2}}\\exp\\!\\Big[-\\frac{(x-N\\mu_1)^2}{2N\\sigma_1^2}\\Big].$$</div>
    <p><b>How fast?</b> The deviation from a Gaussian is measured by the skewness $\\gamma_1=\\kappa_3/\\kappa_2^{3/2}$ and the excess
    kurtosis $\\gamma_2=\\kappa_4/\\kappa_2^2$ (both zero for a Gaussian). With the single-step cumulants obtained from the raw moments
    $m_j=\\langle X^j\\rangle$,</p>
    $$\\kappa_3=m_3-3m_2\\mu_1+2\\mu_1^3,\\qquad \\kappa_4=m_4-4m_3\\mu_1-3m_2^2+12m_2\\mu_1^2-6\\mu_1^4,$$
    <p>and the additivity $\\kappa_j(S_k)=k\\,\\kappa_j$, the shape after $k$ steps is</p>
    $$\\gamma_1(k)=\\frac{\\kappa_3}{\\sqrt k\\,\\sigma_1^3},\\qquad \\gamma_2(k)=\\frac{\\kappa_4}{k\\,\\sigma_1^4}.$$
    <p>The skewness decays like $k^{-1/2}$ and the excess kurtosis like $k^{-1}$ — the leading terms of the Edgeworth expansion. The
    Berry–Esseen theorem bounds the distance to the Gaussian CDF by $C\\,\\langle|X-\\mu_1|^3\\rangle/(\\sigma_1^3\\sqrt N)$.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Monte Carlo:</b> each step uses two uniform random numbers from a seeded mulberry32 generator: one picks the direction
      (right if $r&lt;p$, left if $p\\le r&lt;p+q$, otherwise stay), the other is the length $u$. Positions are double precision.</li>
      <li><b>Walker stream:</b> as in the random-walk page, cohorts of $C=\\lceil M/(12\\cdot\\text{rate})\\rceil$ walkers start on every
      clock tick, with $\\text{rate}=\\mathrm{clamp}(N/1.6,30,320)$ ticks/s, and at most $N$ cohorts are in flight.</li>
      <li><b>Histogram:</b> 80 bins spanning $N\\mu_1\\pm4\\sqrt N\\sigma_1$; bar height = count$/(M_{\\text{done}}\\,\\Delta x)$, a
      probability density directly comparable with the Gaussian curve of the callout (exact mean and variance, no fit).</li>
      <li><b>Moments:</b> for every step index $k$ the running sums $\\sum x,\\sum x^2,\\sum x^3,\\sum x^4$ over all walkers that reached
      step $k$ give the central moments, e.g. $c_3=e_3-3me_2+2m^3$ and $c_4=e_4-4me_3+6m^2e_2-3m^4$ with $e_j=\\langle x^j\\rangle$, hence
      measured $\\gamma_1=c_3/c_2^{3/2}$ and $\\gamma_2=c_4/c_2^2-3$. These dots are compared with the theoretical curves
      $\\gamma_1/\\sqrt k$ and $\\gamma_2/k$ in the lower-right plot.</li>
      <li><b>Metrics:</b> for the final distribution, measured vs. theoretical mean $N\\mu_1$, standard deviation
      $\\sqrt N\\sigma_1$, skewness and excess kurtosis. With $p=q=0$ the distribution is a delta function at $0$ and is reported as
      degenerate.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Symmetric, no rests</b> ($p=q=0.5$): steps are uniform on $[-1,1]$, so $\\mu_1=0$, $\\sigma_1^2=1/3$, and for $N=100$,
      $\\sigma=\\sqrt{100/3}\\approx5.77$ m. The single step has $\\gamma_2=-1.2$ (flatter than Gaussian), so at $N=100$ the excess
      kurtosis is only $-0.012$.</li>
      <li><b>Only forward</b> ($p=1$, $q=0$): $S_N$ is a sum of $U(0,1)$ variables (Irwin–Hall distribution) with mean $N/2$ and
      variance $N/12$; the walkers move as a tight bundle, $\\sigma=2.89$ m for $N=100$.</li>
      <li><b>Rare, one-sided kicks</b> ($p=0.1$, $q=0$): the single-step skewness is $\\approx3.7$, so with $N=10$ the histogram is
      strongly skewed ($\\gamma_1\\approx1.2$); raise $N$ to 300 and it drops to $\\approx0.22$ — convergence is slow when the step
      distribution is lopsided.</li>
      <li><b>Mostly resting</b> ($p=q=0.05$): 90% of the steps are zero, yet the sum is again Gaussian with
      $\\sigma_1^2=0.1/3$; the spike at zero in the single-step plot does not survive summation.</li>
      <li>Compare the measured skewness dots with the $k^{-1/2}$ curve: the agreement improves with $M$ (statistical error of
      a sample skewness $\\approx\\sqrt{6/M}$).</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The CLT needs finite variance; for heavy-tailed steps (e.g. Lévy flights with $\\rho(x)\\sim|x|^{-1-\\alpha}$, $\\alpha&lt;2$) the
    sum converges to a non-Gaussian stable law, which this page cannot show since its step lengths are bounded. The histogram
    covers $\\pm4\\sigma$, so rare tail events outside it are not displayed. References: F. Reif, <i>Fundamentals of Statistical and
    Thermal Physics</i>, §1.10–1.11; W. Feller, <i>An Introduction to Probability Theory and Its Applications</i>, Vol. 2;
    N. G. van Kampen, <i>Stochastic Processes in Physics and Chemistry</i>, ch. 1; K. Huang, <i>Introduction to Statistical
    Physics</i>.</p>`,

  mount(api) {
    const P = api.params;
    const M = api.metrics([
      { id: "mean", label: "Mean (m): measured / theory" },
      { id: "std", label: "Std. dev. (m): measured / theory" },
      { id: "g1", label: "Skewness $\\gamma_1$: measured / theory" },
      { id: "g2", label: "Excess kurtosis $\\gamma_2$: measured / theory" },
    ]);
    const plots = api.plots([
      { id: "traj", title: "Individual trajectories (live)", aspect: 0.78, xlabel: "step number (time)", ylabel: "position (m)" },
      { id: "hist", title: "Final-position distribution vs. Gaussian theory", aspect: 0.78, xlabel: "probability density (1/m)", ylabel: "final position (m)" },
      { id: "step", title: "Distribution of a single step (not Gaussian!)", aspect: 0.55, xlim: [-1.3, 1.3], ylim: [0, 1.5], xlabel: "step Δx (m)", ylabel: "density" },
      { id: "shape", title: "Approach to the Gaussian: skewness and excess kurtosis", aspect: 0.55, xlabel: "step k", ylabel: "γ" },
    ]);
    const NB = 80;
    let N, W, p, q, C, K, rate, acc, launched, finished, nDone, done, finishedAll = false;
    let pos, age, size, cid, paths, counts, S1, S2, S3, S4, cnt, ylo, yhi, hlo, hw, mu1, var1, g1, g2, degenerate, rs;

    function rnd() {
      let t = (rs = (rs + 0x6d2b79f5) | 0);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    const r2 = (x) => Math.round(x * 100) / 100;
    function updateQLabel() {
      api.setControl("q", { label: `Probability of a left step $q$ (stay: $n$ = ${PM.fmt(Math.max(0, 1 - P.p - P.q), 2)})` });
    }

    function tick1() {
      const NP1 = N + 1, pq = p + q;
      for (let s = 0; s < N; s++) {
        const a = age[s];
        if (a < 0 || a >= N) continue;
        const off = s * C, n = size[s];
        let s1 = 0, s2 = 0, s3 = 0, s4 = 0;
        for (let i = off, e = off + n; i < e; i++) {
          const u = rnd(), L = rnd();
          let x = pos[i];
          if (u < p) x += L; else if (u < pq) x -= L;
          pos[i] = x;
          const x2 = x * x;
          s1 += x; s2 += x2; s3 += x2 * x; s4 += x2 * x2;
        }
        const na = a + 1;
        age[s] = na;
        S1[na] += s1; S2[na] += s2; S3[na] += s3; S4[na] += s4; cnt[na] += n;
        paths[s * NP1 + na] = pos[off];
        if (na === N) {
          for (let i = off, e = off + n; i < e; i++) {
            const b = Math.floor((pos[i] - hlo) / hw);
            if (b >= 0 && b < NB) counts[b]++;
          }
          nDone += n; finished++;
        }
      }
      if (launched < K) {
        const s = launched % N, off = s * C;
        const n = launched === K - 1 ? W - C * (K - 1) : C;
        for (let i = off; i < off + n; i++) pos[i] = 0;
        age[s] = 0; size[s] = n; cid[s] = launched; paths[s * NP1] = 0;
        cnt[0] += n;
        launched++;
      }
      if (finished >= K) done = true;
    }

    function moments(k) { // measured (mean, std, γ1, γ2) at step k
      const c = cnt[k];
      if (c < 2) return null;
      const m = S1[k] / c, e2 = S2[k] / c, e3 = S3[k] / c, e4 = S4[k] / c;
      const c2 = e2 - m * m, c3 = e3 - 3 * m * e2 + 2 * m * m * m, c4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4;
      if (!(c2 > 1e-14)) return { m, s: 0, g1: NaN, g2: NaN };
      return { m, s: Math.sqrt(c2), g1: c3 / c2 ** 1.5, g2: c4 / (c2 * c2) - 3 };
    }

    function hbars(pl, lo, w, vals, color, alpha) {
      pl.custom((c) => {
        c.fillStyle = color; c.globalAlpha = alpha;
        for (let j = 0; j < vals.length; j++) {
          if (!(vals[j] > 0)) continue;
          const y0 = pl.Y(lo + (j + 1) * w), y1 = pl.Y(lo + j * w), x0 = pl.X(0), x1 = pl.X(vals[j]);
          c.fillRect(x0, y0, x1 - x0, Math.max(y1 - y0 - 0.6, 0.6));
        }
      });
    }

    return {
      onParam(id, v) {
        if (id === "p" && P.q > 1 - v + 1e-9) api.setControl("q", { value: r2(1 - v) });
        if (id === "q" && v > 1 - P.p + 1e-9) api.setControl("q", { value: r2(1 - P.p) });
      },
      reset() {
        updateQLabel();
        if (finishedAll && !api.isPlaying) api.play();
        N = P.N; W = P.W; p = P.p; q = Math.min(P.q, 1 - p);
        rs = (P.seed | 0) ^ 0x85ebca6b;
        rate = PM.clamp(N / 1.6, 30, 320);
        C = Math.max(1, Math.ceil(W / (rate * 12)));
        K = Math.ceil(W / C);
        pos = new Float64Array(N * C);
        age = new Int32Array(N).fill(-1); size = new Int32Array(N); cid = new Int32Array(N);
        paths = new Float32Array(N * (N + 1));
        counts = new Float64Array(NB);
        S1 = new Float64Array(N + 1); S2 = new Float64Array(N + 1); S3 = new Float64Array(N + 1); S4 = new Float64Array(N + 1); cnt = new Float64Array(N + 1);
        acc = 0; launched = 0; finished = 0; nDone = 0; done = false; finishedAll = false;
        // moments and cumulants of a single step
        mu1 = 0.5 * (p - q);
        const m2 = (p + q) / 3, m3 = (p - q) / 4, m4 = (p + q) / 5;
        var1 = m2 - mu1 * mu1;
        degenerate = !(var1 > 1e-12);
        const k3 = m3 - 3 * m2 * mu1 + 2 * mu1 ** 3, k4 = m4 - 4 * m3 * mu1 - 3 * m2 * m2 + 12 * m2 * mu1 * mu1 - 6 * mu1 ** 4;
        g1 = degenerate ? 0 : k3 / var1 ** 1.5; g2 = degenerate ? 0 : k4 / (var1 * var1);
        const mu = N * mu1, sd = Math.sqrt(Math.max(N * var1, 0));
        if (degenerate) { hlo = -1; hw = 2 / NB; } else { hlo = mu - 4 * sd; hw = (8 * sd) / NB; }
        ylo = Math.min(-0.5, degenerate ? -1 : mu - 4.3 * sd); yhi = Math.max(0.5, degenerate ? 1 : mu + 4.3 * sd);
        plots.traj.setLimits([0, N], [ylo, yhi]);
        const peak = degenerate ? 1 : 1 / (sd * Math.sqrt(2 * Math.PI));
        plots.hist.setLimits([0, peak * 1.35], [ylo, yhi]);
        const gmax = Math.min(4, Math.max(0.3, g1, g2)), gmin = Math.max(-4, Math.min(0, g1, g2));
        plots.shape.setLimits([0, N], [Math.min(-0.3, gmin * 1.15), gmax * 1.15]);
        tick1();
      },
      step(dt) {
        if (done) { if (!finishedAll) { finishedAll = true; api.pause(); } return; }
        acc += dt * rate;
        let n = 0;
        while (acc >= 1 && n < 40 && !done) { tick1(); acc -= 1; n++; }
        if (n >= 40) acc = 0;
      },
      onAction(id) {
        if (id === "finish") { while (!done) tick1(); }
      },
      render() {
        const NP1 = N + 1, mu = N * mu1, sd = Math.sqrt(Math.max(N * var1, 0));
        // --- trajectories
        const pt = plots.traj;
        pt.clear();
        if (P.band && !degenerate) {
          const ks = PM.linspace(0, N, 101), up = new Float64Array(101), lo = new Float64Array(101), mm = new Float64Array(101);
          for (let i = 0; i < 101; i++) { const k = ks[i], s = 2 * Math.sqrt(k * var1); mm[i] = k * mu1; up[i] = mm[i] + s; lo[i] = mm[i] - s; }
          pt.fill(ks, up, lo, { color: PlotColors.accent3, alpha: 0.08 });
          pt.line(ks, up, { color: PlotColors.accent3, width: 1, dash: [4, 4], alpha: 0.7 });
          pt.line(ks, lo, { color: PlotColors.accent3, width: 1, dash: [4, 4], alpha: 0.7 });
          pt.line(ks, mm, { color: PlotColors.accent3, width: 1.4, alpha: 0.9 });
        }
        pt.hline(0, { color: PlotColors.text, dash: [6, 4], width: 1.2, alpha: 0.55 });
        let active = 0; for (let s = 0; s < N; s++) if (age[s] >= 0) active++;
        const every = Math.max(1, Math.ceil(active / P.S));
        pt.custom((c) => {
          c.lineWidth = 1.2; c.lineJoin = "round";
          for (let s = 0; s < N; s++) {
            const a = age[s];
            if (a < 1 || cid[s] % every !== 0) continue;
            c.strokeStyle = colormap("viridis", 0.25 + 0.75 * ((cid[s] * 0.137) % 1));
            c.globalAlpha = a === N ? 0.22 : 0.55;
            c.beginPath();
            const base = s * NP1;
            c.moveTo(pt.X(0), pt.Y(paths[base]));
            for (let k = 1; k <= a; k++) c.lineTo(pt.X(k), pt.Y(paths[base + k]));
            c.stroke();
            if (a < N) { c.globalAlpha = 0.95; c.fillStyle = c.strokeStyle; c.beginPath(); c.arc(pt.X(a), pt.Y(paths[base + a]), 2.2, 0, 7); c.fill(); }
          }
          c.globalAlpha = 1;
        });
        pt.label(`p = ${PM.fmt(p, 2)} · q = ${PM.fmt(q, 2)} · n = ${PM.fmt(Math.max(0, 1 - p - q), 2)}`, "tl");

        // --- final-position histogram (horizontal)
        const ph = plots.hist;
        ph.clear();
        if (degenerate) {
          if (nDone) ph.rect(0, -0.02, 1, 0.02, { color: PlotColors.accent2, alpha: 0.8 });
          ph.label(["p = q = 0: every particle stays put;", "the distribution is a δ function at x = 0."], "tl");
        } else {
          const dens = new Float64Array(NB);
          for (let j = 0; j < NB; j++) dens[j] = nDone ? counts[j] / (nDone * hw) : 0;
          hbars(ph, hlo, hw, dens, PlotColors.accent2, 0.75);
          const ys = PM.linspace(mu - 4 * sd, mu + 4 * sd, 400), fx = new Float64Array(400);
          for (let i = 0; i < 400; i++) fx[i] = Math.exp(-0.5 * ((ys[i] - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
          ph.line(fx, ys, { color: PlotColors.accent3, width: 2.2 });
          ph.hline(mu, { color: PlotColors.accent3, dash: [5, 4], width: 1, alpha: 0.6 });
        }
        if (!degenerate) ph.legend([{ label: `simulation (${nDone.toLocaleString("en-US")})`, color: PlotColors.accent2, type: "box" }, { label: "theory (Gaussian)", color: PlotColors.accent3 }], mu > (ylo + yhi) / 2 ? "br" : "tr");

        // --- single-step distribution
        const ps = plots.step, n0 = Math.max(0, 1 - p - q);
        ps.clear();
        if (q > 0) ps.rect(-1, 0, 0, q, { color: PlotColors.blue, alpha: 0.55, stroke: PlotColors.blue });
        if (p > 0) ps.rect(0, 0, 1, p, { color: PlotColors.accent, alpha: 0.55, stroke: PlotColors.accent });
        if (n0 > 1e-9) {
          ps.arrow(0, 0, 0, Math.min(n0, 1.05), { color: PlotColors.accent3, width: 2.4 });
          ps.text(0, Math.min(n0, 1.05), `n·δ(x), n = ${PM.fmt(n0, 2)}`, { dx: -8, dy: 2, align: "right", color: PlotColors.accent3, size: 12, bg: "#0f151c" });
        }
        ps.vline(mu1, { color: PlotColors.text, dash: [4, 4], width: 1 });
        ps.label([`μ₁ = ${PM.fmt(mu1, 3)} m`, `σ₁ = ${PM.fmt(Math.sqrt(Math.max(var1, 0)), 3)} m`], "tl");
        ps.legend([{ label: `left: q·U(−1,0)`, color: PlotColors.blue, type: "box" }, { label: `right: p·U(0,1)`, color: PlotColors.accent, type: "box" }, { label: "stay: n·δ(x)", color: PlotColors.accent3 }], "tr");

        // --- shape: skewness and excess kurtosis
        const pg = plots.shape;
        pg.clear();
        pg.hline(0, { color: PlotColors.muted, dash: [5, 4], width: 1 });
        if (!degenerate) {
          pg.fn((k) => (k >= 1 ? g1 / Math.sqrt(k) : NaN), { color: PlotColors.accent, width: 2, samples: 300 });
          pg.fn((k) => (k >= 1 ? g2 / k : NaN), { color: PlotColors.pink, width: 2, samples: 300 });
          const kk = [], a1 = [], a2 = [];
          const stride = Math.max(1, Math.ceil(N / 100));
          for (let k = 1; k <= N; k += stride) { const m = moments(k); if (m && isFinite(m.g1)) { kk.push(k); a1.push(m.g1); a2.push(m.g2); } }
          if (kk.length) {
            pg.points(kk, a1, { color: PlotColors.accent, size: 2.2, alpha: 0.85 });
            pg.points(kk, a2, { color: PlotColors.pink, size: 2.2, alpha: 0.85 });
          }
          pg.legend([{ label: "γ₁ measured / theory ∝ k^−1/2", color: PlotColors.accent }, { label: "γ₂ measured / theory ∝ 1/k", color: PlotColors.pink }], "tr");
        } else pg.label("All steps are zero: degenerate distribution", "tl");

        // --- metrics (final distribution)
        const mN = nDone > 1 ? moments(N) : null;
        M.set("mean", `${mN ? PM.fmt(mN.m, 3) : "—"} / ${PM.fmt(mu, 3)}`);
        M.set("std", `${mN ? PM.fmt(mN.s, 3) : "—"} / ${PM.fmt(sd, 3)}`);
        M.set("g1", `${mN ? PM.fmt(mN.g1, 3) : "—"} / ${degenerate ? "—" : PM.fmt(g1 / Math.sqrt(N), 3)}`);
        M.set("g2", `${mN ? PM.fmt(mN.g2, 3) : "—"} / ${degenerate ? "—" : PM.fmt(g2 / N, 3)}`);
        api.setTime(done ? `complete ✓ (${W.toLocaleString("en-US")} walks)` : `${nDone.toLocaleString("en-US")} / ${W.toLocaleString("en-US")} walks`);
      },
    };
  },
});
