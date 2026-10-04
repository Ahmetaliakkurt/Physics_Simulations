/* Binomial distribution and the 1D random walk — a live stream of walkers.
 * Walkers start continuously in small groups (cohorts); each takes N steps and, when it finishes,
 * its final position is added to the histogram. All positions are kept in typed arrays. */
App.register({
  id: "random-walk-binomial",
  category: "statistical",
  order: 20,
  title: "Random Walk and the Binomial Distribution",
  icon: "🎲",
  subtitle: "Thousands of ±1 random walkers set off in a steady stream; live trajectories on the left, and the histogram of their final positions converging to the exact binomial law on the right.",
  notes: [{ type: "info", html: "At every step a walker moves right (+1) with probability $p$ and left (−1) with probability $1-p$. Walkers start in groups; when a walker has completed its $N$ steps its final position is added to the histogram, which converges to the theoretical binomial distribution. The lower plots show the convergence rate and the $\\sqrt{k}$ diffusive spreading." }],
  animated: true,
  speed: { min: 0.1, max: 5, value: 1, step: 0.1 },
  controls: [
    { id: "N", type: "slider", label: "Number of steps $N$", min: 10, max: 500, step: 10, value: 100 },
    { id: "W", type: "slider", label: "Number of walks (trials) $M$", min: 1000, max: 100000, step: 1000, value: 20000,
      fmt: (v) => v.toLocaleString("en-US") },
    { id: "p", type: "slider", label: "Probability of a right step $p$", min: 0.05, max: 0.95, step: 0.01, value: 0.5 },
    { id: "seed", type: "number", label: "Random seed", min: 0, max: 999999, step: 1, value: 42 },
    { type: "section", label: "Display" },
    { id: "S", type: "slider", label: "Trajectories drawn", min: 5, max: 200, step: 5, value: 60, live: true },
    { id: "band", type: "checkbox", label: "Show the theoretical $\\langle x\\rangle \\pm 2\\sigma$ envelope", value: true, live: true },
    { id: "finish", type: "button", label: "⏩ Finish all remaining walks now" },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A particle lives on a one-dimensional lattice with unit spacing and starts at $x=0$. At each tick of a discrete clock it
    jumps one site to the right with probability $p$ or one site to the left with probability $1-p$, independently of all earlier
    jumps (a Markov process with no memory). After $N$ steps its position is $x_N$. This is the simplest model of diffusion — a
    molecule kicked by its neighbours, a polymer chain of $N$ freely jointed links projected on one axis, a spin-½ paramagnet
    ($k$ spins up out of $N$) or a gambler's fortune. Units: lattice spacing $=1$, time step $=1$; $p$ is dimensionless; $M$ walks
    are simulated with a reproducible random seed.</p>

    <h4>Equations being solved</h4>
    <p><b>Binomial law.</b> Let $k$ be the number of right steps. Any particular sequence with $k$ rights and $N-k$ lefts has
    probability $p^k(1-p)^{N-k}$, and there are $\\binom{N}{k}=\\frac{N!}{k!(N-k)!}$ such sequences (choose which $k$ of the $N$ steps
    go right). Hence</p>
    <div class="callout">$$P(k)=\\binom{N}{k}p^k(1-p)^{N-k},\\qquad x_N=k-(N-k)=2k-N,\\qquad k=0,1,\\dots,N .$$</div>
    <p><b>Moments.</b> Write $x_N=\\sum_{i=1}^N s_i$ with independent steps $s_i=\\pm1$. One step has
    $\\langle s\\rangle=p-(1-p)=2p-1$ and $\\langle s^2\\rangle=1$, so $\\operatorname{Var}s=1-(2p-1)^2=4p(1-p)$. Means and variances
    of independent variables add:</p>
    $$\\langle x_N\\rangle=N(2p-1),\\qquad \\sigma_N^2=4Np(1-p),\\qquad \\sigma(k)=\\sqrt{4kp(1-p)}\\propto\\sqrt{k}.$$
    <p>The $\\sqrt{k}$ growth is the hallmark of diffusion. In the continuum limit the probability density obeys the
    drift–diffusion equation $\\partial_t\\rho=-v\\,\\partial_x\\rho+D\\,\\partial_x^2\\rho$ with drift $v=2p-1$ and diffusion constant
    $D=2p(1-p)$ (so that $\\sigma^2=2Dt$), and for large $N$ the binomial approaches the Gaussian
    $\\mathcal N\\big(N(2p-1),\\,4Np(1-p)\\big)$ (de Moivre–Laplace theorem, a special case of the central limit theorem).</p>
    <p><b>Why the curve is $P(k)/2$.</b> Because $x_N=2k-N$, the final position always has the parity of $N$ and neighbouring
    possible values are 2 apart. The histogram therefore uses bins of width 2 centred on $x=2k-N$, and a bin's height is a
    probability <em>density</em>: (fraction of walks in the bin)/(bin width). Its expected value is $P(k)/2$, which is what is drawn.
    This also makes the histogram directly comparable with the Gaussian density, whose peak is $1/(\\sigma\\sqrt{2\\pi})$.</p>
    <p><b>Convergence.</b> With $M$ completed walks the empirical frequency $\\hat f_k$ of each bin is binomially distributed with
    standard deviation $\\sqrt{P_k(1-P_k)/M}$; for a Gaussian error $\\langle|\\hat f_k-P_k|\\rangle=\\sqrt{2P_k(1-P_k)/(\\pi M)}$. The total
    variation distance $\\mathrm{TV}=\\tfrac12\\sum_k|\\hat f_k-P_k|$ is therefore expected to fall as</p>
    $$\\langle\\mathrm{TV}\\rangle\\approx\\tfrac12\\sum_k\\sqrt{\\frac{2P_k(1-P_k)}{\\pi M}}\\ \\propto\\ M^{-1/2}.$$

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Monte Carlo:</b> every step of every walker is drawn explicitly with a seeded mulberry32 generator
      (uniform $u\\in[0,1)$; right if $u&lt;p$). Positions are stored as 16-bit integers.</li>
      <li><b>Walker stream:</b> the clock runs at $\\mathrm{clamp}(N/1.6,\\,30,\\,320)$ ticks per second (one walk lasts about 1.6 s
      at speed ×1). On each tick a new cohort of $C=\\lceil M/(12\\cdot\\text{rate})\\rceil$ walkers starts and every walker in flight
      takes one step; at most $N$ cohorts are in flight at once (ring buffer). When a cohort completes its $N$-th step, its final
      positions are binned as $j=(x_N+N)/2=k$.</li>
      <li><b>Trajectories:</b> the first walker of each cohort records its full path; only every $\\lceil\\text{active}/S\\rceil$-th
      path is drawn (finished paths fade). The orange band is $\\langle x\\rangle\\pm2\\sigma(k)$ from the formulas above.</li>
      <li><b>Histogram:</b> bars $=\\text{counts}_k/(2M_{\\text{done}})$ against the exact $P(k)/2$ (computed with log-Gamma functions,
      so it is accurate for $N=500$).</li>
      <li><b>Convergence plot:</b> TV is recomputed every time a cohort finishes and plotted against the number of completed walks on
      log–log axes, together with the expected $M^{-1/2}$ curve.</li>
      <li><b>Diffusion plot:</b> running sums $\\sum x$ and $\\sum x^2$ over all walkers that have reached step $k$ give the measured
      $\\sigma(k)=\\sqrt{\\langle x^2\\rangle-\\langle x\\rangle^2}$, compared with $\\sqrt{4kp(1-p)}$.</li>
      <li><b>Metrics:</b> completed walks; measured vs. theoretical mean and standard deviation of $x_N$; the current TV distance.
      "Finish all remaining walks now" runs the remaining ticks without drawing.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Default ($N=100$, $p=0.5$):</b> $\\langle x\\rangle=0$, $\\sigma=\\sqrt{100}=10$; about 95% of the paths stay inside the
      $\\pm2\\sigma$ envelope, i.e. $|x|\\le20$.</li>
      <li><b>Drift:</b> $p=0.7$ gives $\\langle x\\rangle=40$ and $\\sigma=\\sqrt{400\\cdot0.21}\\approx9.17$ — the cloud moves at speed
      $v=0.4$ per step while still spreading like $\\sqrt{k}$.</li>
      <li><b>Skewness:</b> $N=10$, $p=0.9$: the histogram is visibly lopsided. The skewness of the binomial,
      $(1-2p)/\\sqrt{Np(1-p)}\\approx-0.84$, only decays as $N^{-1/2}$; at $N=500$ the shape is almost Gaussian.</li>
      <li><b>Statistics of sampling:</b> watch the TV curve hug the dashed $M^{-1/2}$ line; going from $M=5000$ to $M=20000$ halves
      the deviation from the binomial.</li>
      <li><b>Parity:</b> $N$ is a multiple of 10 here, so every final position is even and the histogram bars sit 2 apart; after an odd
      number of steps every walker is on an odd site. Set $N=10$ to see the 11 discrete bars clearly.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The walk is Markovian (no memory), the walkers do not interact and there are no boundaries; it lives on a 1D lattice, whereas real diffusion has a
    continuous step distribution and a finite correlation time, and in 2D/3D the recurrence properties differ (Pólya's theorem).
    References: F. Reif, <i>Fundamentals of Statistical and Thermal Physics</i>, ch. 1; W. Feller, <i>An Introduction to Probability
    Theory and Its Applications</i>, Vol. 1; D. V. Schroeder, <i>An Introduction to Thermal Physics</i>, ch. 2;
    J. Rudnick &amp; G. Gaspari, <i>Elements of the Random Walk</i>.</p>`,

  mount(api) {
    const P = api.params;
    const M = api.metrics([
      { id: "done", label: "Completed walks" },
      { id: "mean", label: "Mean $\\langle x_N\\rangle$: measured / theory" },
      { id: "std", label: "Std. dev. $\\sigma_N$: measured / theory" },
      { id: "tv", label: "TV distance (histogram ↔ binomial)" },
    ]);
    const plots = api.plots([
      { id: "traj", title: "Individual trajectories (live)", aspect: 0.78, xlabel: "step k", ylabel: "position x" },
      { id: "hist", title: "Final-position histogram and binomial $P(k)/2$", aspect: 0.78, xlabel: "probability density", ylabel: "final position x" },
      { id: "conv", title: "Convergence: total variation distance", aspect: 0.55, xlog: true, ylog: true, xlabel: "completed walks M", ylabel: "TV" },
      { id: "diff", title: "Diffusion: spread of the position after k steps", aspect: 0.55, xlabel: "step k", ylabel: "σ(k)" },
    ]);

    let N, W, p, C, K, rate, acc, tick, launched, finished, nDone, done;
    let pos, age, size, cid, paths, counts, pmf, S1, S2, cnt, convM, convTV, ylo, yhi, hpeak, rs, finishedAll = false;

    // Fast seedable mulberry32 (state variable rs, inlined for speed)
    function rnd() {
      let t = (rs = (rs + 0x6d2b79f5) | 0);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }

    function tvDistance() {
      let tv = 0;
      if (nDone === 0) return NaN;
      for (let j = 0; j <= N; j++) tv += Math.abs(counts[j] / nDone - pmf[j]);
      return 0.5 * tv;
    }
    function expectedTV(m) {
      let s = 0;
      for (let j = 0; j <= N; j++) s += Math.sqrt((2 * pmf[j] * (1 - pmf[j])) / (Math.PI * m));
      return 0.5 * s;
    }

    function tick1() {
      const NP1 = N + 1;
      let newly = false;
      for (let s = 0; s < N; s++) {
        const a = age[s];
        if (a < 0 || a >= N) continue;
        const off = s * C, n = size[s];
        let s1 = 0, s2 = 0;
        for (let i = off, e = off + n; i < e; i++) {
          const x = (pos[i] += rnd() < p ? 1 : -1);
          s1 += x; s2 += x * x;
        }
        const na = a + 1;
        age[s] = na;
        S1[na] += s1; S2[na] += s2; cnt[na] += n;
        paths[s * NP1 + na] = pos[off];
        if (na === N) {
          for (let i = off, e = off + n; i < e; i++) counts[(pos[i] + N) >> 1]++;
          nDone += n; finished++; newly = true;
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
      tick++;
      if (newly) { convM.push(nDone); convTV.push(tvDistance()); }
      if (finished >= K) done = true;
    }

    function hbars(pl, centers, vals, h, color, alpha) {
      pl.custom((c) => {
        c.fillStyle = color; c.globalAlpha = alpha;
        for (let j = 0; j < centers.length; j++) {
          if (!(vals[j] > 0)) continue;
          const y0 = pl.Y(centers[j] + h / 2), y1 = pl.Y(centers[j] - h / 2), x0 = pl.X(0), x1 = pl.X(vals[j]);
          c.fillRect(x0, y0, x1 - x0, Math.max(y1 - y0 - 0.6, 0.6));
        }
      });
    }

    return {
      reset() {
        N = P.N; W = P.W; p = P.p;
        rs = (P.seed | 0) ^ 0x9e3779b9;
        rate = PM.clamp(N / 1.6, 30, 320);             // ticks per second (one walk ≈ 1.6 s at speed ×1)
        C = Math.max(1, Math.ceil(W / (rate * 12)));   // walkers starting on each tick
        K = Math.ceil(W / C);
        pos = new Int16Array(N * C);
        age = new Int32Array(N).fill(-1); size = new Int32Array(N); cid = new Int32Array(N);
        paths = new Float32Array(N * (N + 1));
        counts = new Float64Array(N + 1); pmf = new Float64Array(N + 1);
        for (let j = 0; j <= N; j++) pmf[j] = PM.binomPMF(N, j, p);
        S1 = new Float64Array(N + 1); S2 = new Float64Array(N + 1); cnt = new Float64Array(N + 1);
        convM = []; convTV = [];
        if (finishedAll && !api.isPlaying) api.play();
        acc = 0; tick = 0; launched = 0; finished = 0; nDone = 0; done = false; finishedAll = false;
        // fixed axis limits (μ ± 4.5σ of the final distribution plus the starting point)
        const mu = N * (2 * p - 1), sd = Math.sqrt(4 * N * p * (1 - p));
        ylo = Math.max(-N - 1, Math.min(-1, mu - 4.5 * sd - 2));
        yhi = Math.min(N + 1, Math.max(1, mu + 4.5 * sd + 2));
        hpeak = PM.max(pmf) / 2;
        plots.traj.setLimits([0, N], [ylo, yhi]);
        plots.hist.setLimits([0, hpeak * 1.3], [ylo, yhi]);
        plots.diff.setLimits([0, N], [0, Math.max(sd, 0.5) * 1.3]);
        const tvMax = expectedTV(Math.max(1, C)), tvMin = expectedTV(W);
        plots.conv.setLimits([Math.max(1, C * 0.8), W * 1.25], [Math.max(tvMin / 4, 1e-4), Math.min(2, tvMax * 3)]);
        tick1(); // the first cohort starts
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
        const NP1 = N + 1, mu = N * (2 * p - 1), sd = Math.sqrt(4 * N * p * (1 - p));
        // --- trajectories
        const pt = plots.traj;
        pt.clear();
        if (P.band) {
          const ks = PM.linspace(0, N, 101), up = new Float64Array(101), lo = new Float64Array(101), mm = new Float64Array(101);
          for (let i = 0; i < 101; i++) { const k = ks[i], s = 2 * Math.sqrt(k * p * (1 - p)); mm[i] = k * (2 * p - 1); up[i] = mm[i] + 2 * s; lo[i] = mm[i] - 2 * s; }
          pt.fill(ks, up, lo, { color: PlotColors.accent3, alpha: 0.08 });
          pt.line(ks, up, { color: PlotColors.accent3, width: 1, dash: [4, 4], alpha: 0.7 });
          pt.line(ks, lo, { color: PlotColors.accent3, width: 1, dash: [4, 4], alpha: 0.7 });
          pt.line(ks, mm, { color: PlotColors.accent3, width: 1.4, alpha: 0.9 });
        }
        pt.hline(0, { color: PlotColors.text, dash: [6, 4], width: 1, alpha: 0.5 });
        let active = 0; for (let s = 0; s < N; s++) if (age[s] >= 0) active++;
        const every = Math.max(1, Math.ceil(active / P.S));
        pt.custom((c) => {
          c.lineWidth = 1.2; c.lineJoin = "round";
          for (let s = 0; s < N; s++) {
            const a = age[s];
            if (a < 1 || cid[s] % every !== 0) continue;
            c.strokeStyle = colormap("turbo", 0.1 + 0.8 * ((cid[s] * 0.137) % 1));
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
        pt.label(`p = ${PM.fmt(p, 2)} · N = ${N}`, "tl");

        // --- histogram (horizontal, sharing the position axis with the trajectory plot)
        const ph = plots.hist;
        ph.clear();
        const xs = new Float64Array(NP1), dens = new Float64Array(NP1), th = new Float64Array(NP1);
        for (let j = 0; j <= N; j++) { xs[j] = 2 * j - N; dens[j] = nDone ? counts[j] / (nDone * 2) : 0; th[j] = pmf[j] / 2; }
        hbars(ph, xs, dens, 2, PlotColors.accent, 0.7);
        ph.line(th, xs, { color: PlotColors.accent3, width: 2.2 });
        ph.hline(mu, { color: PlotColors.accent3, dash: [5, 4], width: 1, alpha: 0.6 });
        ph.legend([{ label: `simulation (${nDone.toLocaleString("en-US")})`, color: PlotColors.accent, type: "box" }, { label: "binomial P(k)/2", color: PlotColors.accent3 }], mu > (ylo + yhi) / 2 ? "br" : "tr");

        // --- convergence
        const pc = plots.conv;
        pc.clear();
        const mref = [], tref = [];
        for (let i = 0; i <= 60; i++) { const m = Math.exp(Math.log(Math.max(1, C * 0.8)) + (i / 60) * (Math.log(W * 1.25) - Math.log(Math.max(1, C * 0.8)))); mref.push(m); tref.push(expectedTV(m)); }
        pc.line(mref, tref, { color: PlotColors.muted, dash: [6, 4], width: 1.4 });
        if (convM.length > 1) pc.line(convM, convTV, { color: PlotColors.accent, width: 1.8 });
        if (convM.length) pc.circle(convM[convM.length - 1], convTV[convTV.length - 1], 3.5, { px: true, color: PlotColors.accent });
        pc.legend([{ label: "measured TV", color: PlotColors.accent }, { label: "expected fluctuation ∝ M^−1/2", color: PlotColors.muted, dash: [6, 4] }], "tr");

        // --- diffusion σ(k)
        const pd = plots.diff;
        pd.clear();
        pd.fn((k) => Math.sqrt(4 * k * p * (1 - p)), { color: PlotColors.accent3, width: 2, samples: 200 });
        const kk = [], sk = [];
        for (let k = 1; k <= N; k++) {
          if (cnt[k] < 2) continue;
          const m1 = S1[k] / cnt[k], v = S2[k] / cnt[k] - m1 * m1;
          kk.push(k); sk.push(Math.sqrt(Math.max(v, 0)));
        }
        if (kk.length) pd.points(kk, sk, { color: PlotColors.accent, size: N > 200 ? 1.4 : 2.2, alpha: 0.9 });
        pd.legend([{ label: "measured σ(k)", color: PlotColors.accent, type: "dot" }, { label: "√(4kp(1−p))", color: PlotColors.accent3 }], "tl");

        // --- metrics
        let m1 = NaN, s1 = NaN;
        if (nDone > 0) {
          let a = 0, b = 0;
          for (let j = 0; j <= N; j++) { const x = 2 * j - N; a += counts[j] * x; b += counts[j] * x * x; }
          m1 = a / nDone; s1 = Math.sqrt(Math.max(b / nDone - m1 * m1, 0));
        }
        M.set("done", `${nDone.toLocaleString("en-US")} / ${W.toLocaleString("en-US")}`);
        M.set("mean", `${PM.fmt(m1, 2)} / ${PM.fmt(mu, 2)}`);
        M.set("std", `${PM.fmt(s1, 2)} / ${PM.fmt(sd, 2)}`);
        M.set("tv", convTV.length ? PM.fmt(convTV[convTV.length - 1], 4) : "—");
        api.setTime(done ? "complete ✓" : `${PM.fmt((100 * nDone) / W, 1)} %`);
      },
    };
  },
});
