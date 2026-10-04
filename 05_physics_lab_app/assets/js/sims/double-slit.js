/* Quantum double slit — 2D time-dependent Schrödinger equation, live split-step Fourier (256×256). */
(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Fast 2D FFT: radix-4 "row-vector" butterflies (the column transform is
  // vectorised along whole rows) + blocked transpose. About 4× faster than PM.fft2.
  // No normalisation (the 1/N² of the inverse transform is folded into the kinetic phase array).
  // ------------------------------------------------------------------
  function makeColFFT(P, W) {
    let bits = 0; while ((1 << bits) < P) bits++;
    if ((1 << bits) !== P) throw new Error("FFT size must be a power of two");
    const rev = new Uint32Array(P);
    for (let i = 0; i < P; i++) { let r = 0, x = i; for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; } rev[i] = r; }
    const twr = new Float64Array(2 * P), twi = new Float64Array(2 * P);
    for (let h = 1; h <= P; h <<= 1) for (let k = 0; k < h; k++) { twr[h + k] = Math.cos(Math.PI * k / h); twi[h + k] = -Math.sin(Math.PI * k / h); }
    return function (re, im, sgn) {
      for (let i = 0; i < P; i++) {
        const j = rev[i];
        if (j > i) {
          const a = i * W, b = j * W;
          for (let q = 0; q < W; q++) { let t = re[a + q]; re[a + q] = re[b + q]; re[b + q] = t; t = im[a + q]; im[a + q] = im[b + q]; im[b + q] = t; }
        }
      }
      let h = 1;
      if (bits & 1) {
        for (let st = 0; st < P; st += 2) {
          const a = st * W, b = a + W;
          for (let q = 0; q < W; q++) { const br = re[b + q], bi = im[b + q]; re[b + q] = re[a + q] - br; im[b + q] = im[a + q] - bi; re[a + q] += br; im[a + q] += bi; }
        }
        h = 2;
      }
      for (; h < P; h <<= 2) {
        const s = h << 2;
        for (let st = 0; st < P; st += s) for (let k = 0; k < h; k++) {
          const w1r = twr[h + k], w1i = sgn * twi[h + k];
          const w2r = twr[2 * h + k], w2i = sgn * twi[2 * h + k];
          const w3r = w2i * sgn, w3i = -w2r * sgn;
          const p0 = (st + k) * W, p1 = p0 + h * W, p2 = p1 + h * W, p3 = p2 + h * W;
          for (let q = 0; q < W; q++) {
            const a0r = re[p0 + q], a0i = im[p0 + q];
            let t = re[p1 + q], u = im[p1 + q];
            const a1r = t * w1r - u * w1i, a1i = t * w1i + u * w1r;
            const a2r = re[p2 + q], a2i = im[p2 + q];
            t = re[p3 + q]; u = im[p3 + q];
            const a3r = t * w1r - u * w1i, a3i = t * w1i + u * w1r;
            const y0r = a0r + a1r, y0i = a0i + a1i, y1r = a0r - a1r, y1i = a0i - a1i;
            const y2r = a2r + a3r, y2i = a2i + a3i, y3r = a2r - a3r, y3i = a2i - a3i;
            const v2r = y2r * w2r - y2i * w2i, v2i = y2r * w2i + y2i * w2r;
            const v3r = y3r * w3r - y3i * w3i, v3i = y3r * w3i + y3i * w3r;
            re[p0 + q] = y0r + v2r; im[p0 + q] = y0i + v2i;
            re[p2 + q] = y0r - v2r; im[p2 + q] = y0i - v2i;
            re[p1 + q] = y1r + v3r; im[p1 + q] = y1i + v3i;
            re[p3 + q] = y1r - v3r; im[p3 + q] = y1i - v3i;
          }
        }
      }
    };
  }
  function transposeSq(a, N) {
    const B = 16;
    for (let jb = 0; jb < N; jb += B) for (let ib = jb; ib < N; ib += B)
      for (let j = jb; j < jb + B; j++) for (let i = ib === jb ? j + 1 : ib; i < ib + B; i++) {
        const p = j * N + i, q = i * N + j, t = a[p]; a[p] = a[q]; a[q] = t;
      }
  }
  /** 2D FFT for a square N×N grid. The forward result is stored transposed, i.e. in [kx][ky] order. */
  function makeFFT2(N) {
    const col = makeColFFT(N, N);
    return {
      forward(re, im) { col(re, im, 1); transposeSq(re, N); transposeSq(im, N); col(re, im, 1); },
      inverse(re, im) { col(re, im, -1); transposeSq(re, N); transposeSq(im, N); col(re, im, -1); },
    };
  }

  const N = 256, L = 30, dx = L / N, X0 = -10, Y0 = 0, DT = 0.0125, V0 = 120, BAR = 0.3;
  const PLATE_W = 150, PLATE_H = 180, HBINS = 60, HIT_CAP = 60000;

  App.register({
    id: "double-slit",
    category: "quantum",
    order: 41,
    title: "Double-Slit Experiment",
    icon: "🎯",
    subtitle: "A 2D wave packet diffracts through two slits: the interference pattern emerges from the time-dependent Schrödinger equation and builds up on the screen one particle detection at a time.",
    notes: [{
      type: "info",
      html: "The interference pattern is not drawn by hand: a single wave packet is evolved live, every frame, towards a high wall with two slits " +
        "using the time-dependent Schrödinger equation (split-step Fourier, 256×256 grid). Every dot on the screen on the right is the detection of " +
        "one particle; detections are drawn at random from the probability flux (Born rule) and gradually rebuild the interference pattern of the " +
        "wave function — wave–particle duality.",
    }],
    animated: true,
    speed: { min: 0.2, max: 1.5, value: 1, step: 0.1 },
    controls: [
      { id: "sep", type: "slider", label: "Slit separation $d$", min: 1, max: 10, step: 0.5, value: 4 },
      { id: "width", type: "slider", label: "Slit width $w$", min: 0.5, max: 4, step: 0.1, value: 1.6,
        help: "Limited to $w\\le d-0.2$ so that the two slits never merge." },
      { id: "k0", type: "slider", label: "Initial wavenumber $k_0$", min: 2, max: 12, step: 0.5, value: 6 },
      { id: "sigma", type: "slider", label: "Packet width $\\sigma$", min: 0.5, max: 3, step: 0.1, value: 1.2 },
      { id: "xs", type: "slider", label: "Screen position $x_s$", min: 4, max: 12, step: 0.5, value: 9,
        help: "Distance $D$ of the detection screen from the slits." },
      { type: "section", label: "Detection" },
      { id: "nshot", type: "slider", label: "Particles per packet", min: 500, max: 20000, step: 500, value: 6000, live: true,
        help: "Number of particles one wave packet stands for; the fraction that reaches the screen appears as dots." },
      { id: "loop", type: "checkbox", label: "Send a new packet when the old one has gone (keep accumulating)", value: true, live: true },
      { type: "section", label: "Display" },
      { id: "scale", type: "select", label: "Colour scale", value: "sqrt", live: true,
        options: [{ value: "linear", label: "Linear |ψ|²" }, { value: "sqrt", label: "Square root (fringes stand out)" }, { value: "log", label: "Logarithmic" }] },
      { id: "showHist", type: "checkbox", label: "Show the hit histogram", value: true, live: true },
      { id: "showPred", type: "checkbox", label: "Mark predicted maxima (two point sources)", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A particle moves in a $30\\times30$ box (centred at the origin). A thin wall at $|x|\\lt 0.3$ blocks it everywhere except two slits of width $w$
      centred at $y=\\pm d/2$. The particle starts at $(x_0,y_0)=(-10,0)$ as a circular Gaussian packet moving towards the wall with mean wavenumber
      $k_0$ (de Broglie wavelength $\\lambda=2\\pi/k_0$); a detection screen is placed at $x=x_s=D$ behind the wall. Units: $\\hbar=m=1$, so the mean
      energy is $E=k_0^2/2$ and the group velocity $k_0$.</p>

      <h4>Equations being solved</h4>
      $$ i\\hbar\\frac{\\partial\\psi}{\\partial t}=\\left[-\\frac{\\hbar^2}{2m}\\nabla^2+V(x,y)\\right]\\psi,\\qquad
         \\psi(x,y,0)\\propto \\exp\\!\\left[-\\frac{(x-x_0)^2+(y-y_0)^2}{4\\sigma^2}\\right]e^{ik_0x}, $$
      <p>with $V=V_0=120$ inside the wall (outside the slits) and $V=0$ elsewhere. The wall is far above the packet energy ($E\\le72$), so tunnelling
      through it is negligible: for a barrier of thickness $a=0.6$, $T\\approx16\\frac{E}{V_0}\\big(1-\\frac{E}{V_0}\\big)e^{-2\\kappa a}$ with
      $\\kappa=\\sqrt{2(V_0-E)}$ gives $\\sim10^{-8}$ at $k_0=6$ and $\\sim3\\times10^{-5}$ at $k_0=12$. Behind the wall each slit acts as a source of
      outgoing waves (Huygens' principle). At a point $y$ on the screen the two path lengths differ by
      $\\Delta(y)=\\sqrt{D^2+(y+d/2)^2}-\\sqrt{D^2+(y-d/2)^2}$, and maxima appear where $\\Delta=n\\lambda$; for $D\\gg d,y$ this becomes</p>
      <div class="callout">$$ y_n\\approx n\\,\\frac{\\lambda D}{d},\\qquad \\Delta y=\\frac{\\lambda D}{d}=\\frac{2\\pi D}{k_0 d},\\qquad
        P(\\text{detect in }dy)\\propto j_x(D,y,t)\\,dy\\,dt,\\quad j_x=\\frac{\\hbar}{m}\\,\\mathrm{Im}\\big(\\psi^*\\partial_x\\psi\\big). $$</div>
      <p>The second relation is the Born rule applied to a screen: the probability that the particle crosses the screen line in $dy$ during $dt$ is the
      probability current through it. Each single detection is random; only the histogram of many detections reveals $|\\psi|^2$'s fringes.
      The slits' finite width adds a single-slit envelope $\\propto\\mathrm{sinc}^2(\\pi w\\,y/\\lambda D)$ that modulates the fringe heights.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Grid:</b> $256\\times256$ points, $\\Delta x=30/256\\approx0.117$ (about 4.5 points per wavelength at the largest $k_0=12$); the wall is 5 grid
        cells thick. Representable wavenumbers $|k|\\le\\pi/\\Delta x\\approx26.8$.</li>
        <li><b>Integrator:</b> Strang split-step Fourier,
        $$ \\psi(t+\\Delta t)= e^{-iV\\Delta t/2\\hbar}\\,\\mathcal{F}^{-1}\\!\\left[e^{-i\\hbar k^2\\Delta t/2m}\\,\\mathcal{F}\\!\\left[e^{-iV\\Delta t/2\\hbar}\\psi(t)\\right]\\right], $$
        with a hand-optimised radix-4 2D FFT; consecutive half-steps in $V$ are merged, and at most two steps run per frame.
        <b>Time step $\\Delta t=0.0125$:</b> the method is unconditionally stable, but its error per step grows like $\\Delta t^3$ times nested commutators
        of $T$ and $V$, which are huge for a 120-high wall only 5 cells thick. With a large step this error appears as spurious <i>leakage</i> of
        probability through the wall (not suppressed by $e^{-2\\kappa a}$). The chosen step keeps the wall phase $V_0\\Delta t=1.5$ rad per step and the
        packet moves at most ≈ 1.3 cells per step; the leakage is then far below the visible level.</li>
        <li><b>Boundaries:</b> a sponge in the outer 12 % of the box multiplies $\\psi$ by $e^{-3(\\Delta t/0.015)\\,s^2}$ per step, where $s\\in[0,1]$
        is the normalised depth (dashed square), so waves are absorbed instead of re-entering through the periodic FFT boundary.</li>
        <li><b>Detections:</b> after every step the current $j_x=\\mathrm{Im}(\\psi^*\\partial_x\\psi)$ (central differences) is evaluated along the screen
        column; its positive part sets both the number of hits in this step, $N_{\\rm shot}\\int j_x\\,dy\\,\\Delta t$ (fraction carried over), and — via
        inverse-transform sampling of its cumulative sum — where they land. The teal curve is $\\int|\\psi(D,y,t)|^2dt$ accumulated on the same column.</li>
        <li><b>Metrics:</b> "Through the slits" $=\\int_{x\\gt 0.3}|\\psi|^2$; "Probability in box" $=\\sum|\\psi|^2\\Delta x^2$ (the conserved norm, which
        decreases only by absorption in the sponge). "Measured fringe spacing" is the median distance between neighbouring maxima of the accumulated
        screen intensity above 20 % of its peak (positions refined by parabolic interpolation); it is compared with $\\lambda D/d$ and with the exact
        two-point-source spacing $y_1$ solving $\\Delta(y_1)=\\lambda$, whose maxima $y_n$ are marked by ticks on the screen plot.</li>
      </ul>

      <h4>What to try</h4>
      <ul>
        <li><b>Fringe spacing.</b> Defaults ($k_0=6$, $d=4$, $D=9$): $\\lambda D/d\\approx2.36$, while the exact two-point-source geometry gives
        $y_1\\approx2.50$ because $D/d$ is only 2.25. After a few packets the measured spacing settles near 2.4: real slits are not points
        (they are 1.6 wide) and the packet contains a spread of wavenumbers, both of which shift the observed maxima slightly.</li>
        <li><b>Scaling laws.</b> Double $k_0$ to 12: the spacing halves. Increase $d$ to 8: the fringes crowd together ($\\propto1/d$). Move the screen to
        $x_s=4$: they get closer ($\\propto D$).</li>
        <li><b>Born rule.</b> Watch the first few dozen dots: they look random. After a few thousand hits the histogram (orange) matches the teal
        $\\int|\\psi|^2dt$ curve.</li>
        <li><b>Single-slit envelope.</b> Widen the slits ($w=3$ with $d=4$): the outer fringes fade, because the single-slit diffraction envelope narrows as
        $\\lambda D/w$.</li>
        <li><b>Long wavelength.</b> $k_0=2$: $\\lambda\\approx3.1$ is comparable to $d$, only the central maximum and one or two side fringes fit on the screen,
        and more of the packet is reflected by the wall.</li>
      </ul>

      <h4>Limitations &amp; further reading</h4>
      <p>2D, spinless and non-interacting: one particle at a time, as in single-electron double-slit experiments. The screen is an idealised
      non-absorbing line (detections do not disturb the evolution), and the wavelength is only a few grid cells at the highest $k_0$, so the fine
      structure there is approximate. Reading: R. P. Feynman, <i>The Feynman Lectures on Physics</i> Vol. III, ch. 1; A. Tonomura et al., "Demonstration of
      single-electron buildup of an interference pattern", <i>Am. J. Phys.</i> 57, 117 (1989); E. Hecht, <i>Optics</i>, ch. 9–10.</p>`,

    mount(api) {
      const P = api.params, NN = N * N;
      const fft = makeFFT2(N);
      const re = new Float64Array(NN), im = new Float64Array(NN), dens = new Float64Array(NN);
      const V = new Float64Array(NN);
      const vhR = new Float64Array(NN), vhI = new Float64Array(NN);     // e^{-iVdt/2}
      const vaR = new Float64Array(NN), vaI = new Float64Array(NN);     // e^{-iVdt}·A
      const vhaR = new Float64Array(NN), vhaI = new Float64Array(NN);   // e^{-iVdt/2}·A
      const kR = new Float64Array(NN), kI = new Float64Array(NN);       // e^{-ik²dt/2}/N²
      const acc = new Float64Array(N), flux = new Float64Array(N), cdf = new Float64Array(N);
      const plate = new Float32Array(PLATE_W * PLATE_H), hist = new Float64Array(HBINS);
      const accX = new Float64Array(N), yGrid = new Float64Array(N);
      for (let j = 0; j < N; j++) yGrid[j] = -L / 2 + j * dx;
      const histX = new Float64Array(HBINS), histY = new Float64Array(HBINS);
      const rng = new PM.RNG(20240917);
      let t, tShot, simAcc, hitCarry, nHits, iS, shots, normNow, transNow;

      const M = api.metrics([
        { id: "t", label: "Time $t$" },
        { id: "trans", label: "Through the slits ($x>0.3$)" },
        { id: "norm", label: "Probability in box" },
        { id: "hits", label: "Detections / packets" },
        { id: "fr", label: "Small-angle spacing $\\lambda D/d$" },
        { id: "fr2", label: "Two-source spacing $y_1$" },
        { id: "frm", label: "Measured fringe spacing" },
      ]);
      const plots = api.plots([
        { id: "psi", title: "Probability density |ψ(x,y,t)|²", aspect: 1, xlim: [-L / 2, L / 2], ylim: [-L / 2, L / 2], equal: true, xlabel: "x", ylabel: "y", grid: false, maxHeight: 640 },
        { id: "scr", title: "Detection screen: single-particle hits and accumulated intensity", aspect: 1, xlim: [0, 1.12], ylim: [-L / 2, L / 2], xlabel: "relative intensity", ylabel: "y on the screen", grid: false, maxHeight: 640 },
      ]);

      function slitOpen(y) {
        const d = P.sep, w = Math.min(P.width, d - 0.2);
        return Math.abs(y - d / 2) < w / 2 || Math.abs(y + d / 2) < w / 2;
      }
      function buildOperators() {
        const kx = PM.fftk(N, dx);
        const inv = 1 / NN, edge = (L / 2) * (1 - 0.12), band = L / 2 - edge;
        for (let j = 0; j < N; j++) {
          const y = yGrid[j];
          for (let i = 0; i < N; i++) {
            const x = -L / 2 + i * dx, p = j * N + i;
            const v = Math.abs(x) < BAR && !slitOpen(y) ? V0 : 0;
            V[p] = v;
            const dxn = PM.clamp((Math.abs(x) - edge) / band, 0, 1), dyn = PM.clamp((Math.abs(y) - edge) / band, 0, 1);
            const dd = Math.max(dxn, dyn), A = Math.exp(-3 * (DT / 0.015) * dd * dd);
            vhR[p] = Math.cos(-v * DT / 2); vhI[p] = Math.sin(-v * DT / 2);
            vaR[p] = A * Math.cos(-v * DT); vaI[p] = A * Math.sin(-v * DT);
            vhaR[p] = A * vhR[p]; vhaI[p] = A * vhI[p];
            // the spectrum is in [kx][ky] order: p = i(kx)*N + j(ky) — k² is symmetric so this is fine
            const ph = -0.5 * (kx[i] * kx[i] + kx[j] * kx[j]) * DT;
            kR[p] = inv * Math.cos(ph); kI[p] = inv * Math.sin(ph);
          }
        }
      }
      function launch() {
        const s2 = 4 * P.sigma * P.sigma;
        let nrm = 0;
        for (let j = 0; j < N; j++) {
          const y = yGrid[j];
          for (let i = 0; i < N; i++) {
            const x = -L / 2 + i * dx, p = j * N + i;
            const g = Math.exp(-((x - X0) * (x - X0) + (y - Y0) * (y - Y0)) / s2);
            re[p] = g * Math.cos(P.k0 * x); im[p] = g * Math.sin(P.k0 * x);
            nrm += g * g;
          }
        }
        const f = 1 / Math.sqrt(nrm * dx * dx);
        for (let p = 0; p < NN; p++) { re[p] *= f; im[p] *= f; }
        tShot = 0; shots++;
      }
      function mul(aR, aI) {
        for (let p = 0; p < NN; p++) {
          const r = re[p], m = im[p];
          re[p] = r * aR[p] - m * aI[p]; im[p] = r * aI[p] + m * aR[p];
        }
      }
      function mulK() {
        for (let p = 0; p < NN; p++) {
          const r = re[p], m = im[p];
          re[p] = r * kR[p] - m * kI[p]; im[p] = r * kI[p] + m * kR[p];
        }
      }
      function detect() {
        // screen column (V = 0, no damping): accumulated density + probability current
        let tot = 0;
        for (let j = 0; j < N; j++) {
          const p = j * N + iS;
          const r = re[p], m = im[p];
          acc[j] += (r * r + m * m) * DT;
          const dr = (re[p + 1] - re[p - 1]) / (2 * dx), di = (im[p + 1] - im[p - 1]) / (2 * dx);
          const jx = r * di - m * dr; // Im(ψ* ∂xψ)
          const f = jx > 0 ? jx : 0;
          flux[j] = f; tot += f;
          cdf[j] = tot;
        }
        if (tot <= 0) return;
        hitCarry += P.nshot * tot * dx * DT;
        let n = Math.floor(hitCarry);
        if (rng.next() < hitCarry - n) n++;
        hitCarry -= n;
        if (hitCarry < 0) hitCarry = 0;
        for (let q = 0; q < n && nHits < HIT_CAP; q++) {
          const u = rng.next() * tot;
          let lo = 0, hi = N - 1;
          while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
          const y = yGrid[lo] + (rng.next() - 0.5) * dx;
          const r = Math.floor(((y + L / 2) / L) * PLATE_H), c = Math.floor(rng.next() * PLATE_W);
          if (r >= 0 && r < PLATE_H) plate[r * PLATE_W + c] += 1;
          const b = Math.floor(((y + L / 2) / L) * HBINS);
          if (b >= 0 && b < HBINS) hist[b] += 1;
          nHits++;
        }
      }
      function stepN(n) {
        mul(vhR, vhI);
        for (let s = 0; s < n; s++) {
          fft.forward(re, im);
          mulK();
          fft.inverse(re, im);
          if (s < n - 1) mul(vaR, vaI); else mul(vhaR, vhaI);
          t += DT; tShot += DT;
          detect();
        }
      }
      /** n-th maximum of two point sources at (0, ±d/2) seen on the line x = D: path difference Δ(y) = nλ (bisection). */
      function twoSourceY(n, lam, D) {
        if (n === 0) return 0;
        const d = P.sep, target = Math.abs(n) * lam;
        if (target >= d) return NaN;
        const diff = (y) => Math.hypot(D, y + d / 2) - Math.hypot(D, y - d / 2);
        let lo = 0, hi = 1e3;
        for (let it = 0; it < 60; it++) { const mid = 0.5 * (lo + hi); if (diff(mid) < target) lo = mid; else hi = mid; }
        return Math.sign(n) * 0.5 * (lo + hi);
      }
      /** Median spacing of neighbouring maxima of the accumulated screen intensity (above 20 % of the peak). */
      const peakY = new Float64Array(N), gaps = new Float64Array(N);
      function measuredSpacing() {
        let amax = 0; for (let j = 0; j < N; j++) if (acc[j] > amax) amax = acc[j];
        if (amax <= 0 || nHits < 100) return NaN;
        let np = 0;
        for (let j = 1; j < N - 1; j++) {
          const a = acc[j - 1], b = acc[j], c = acc[j + 1];
          if (b > a && b >= c && b > 0.2 * amax) {
            const den = a - 2 * b + c, off = den !== 0 ? 0.5 * (a - c) / den : 0; // parabolic refinement
            peakY[np++] = yGrid[j] + off * dx;
          }
        }
        if (np < 2) return NaN;
        for (let q = 0; q < np - 1; q++) gaps[q] = peakY[q + 1] - peakY[q];
        const g = Array.from(gaps.subarray(0, np - 1)).sort((u, v) => u - v);
        return g[(g.length - 1) >> 1];
      }
      function measure() {
        let mx = 0, tot = 0, tr = 0;
        const iB = Math.ceil((BAR + L / 2) / dx);
        for (let j = 0; j < N; j++) {
          const o = j * N;
          for (let i = 0; i < N; i++) {
            const p = o + i, d = re[p] * re[p] + im[p] * im[p];
            dens[p] = d; tot += d;
            if (i >= iB) tr += d;
            if (d > mx) mx = d;
          }
        }
        normNow = tot * dx * dx; transNow = tr * dx * dx;
        return mx;
      }

      return {
        reset() {
          if (P.width > P.sep - 0.2) api.setControl("width", { value: Math.max(0.5, Math.round((P.sep - 0.2) * 10) / 10) });
          api.setControl("width", { max: Math.max(0.6, Math.min(4, Math.round((P.sep - 0.2) * 10) / 10)) });
          buildOperators();
          t = 0; shots = 0; simAcc = 0; hitCarry = 0; nHits = 0;
          iS = Math.round((P.xs + L / 2) / dx);
          acc.fill(0); plate.fill(0); hist.fill(0);
          launch();
          measure();
        },
        step(dt) {
          simAcc += dt;
          let n = Math.floor(simAcc / DT);
          if (n > 2) { n = 2; simAcc = 0; } else simAcc -= n * DT;
          if (n > 0) stepN(n);
          if (P.loop && tShot > 2 && normNow < 0.004) launch();
        },
        render() {
          const vmax = measure();
          const pp = plots.psi;
          pp.clear();
          pp.heatmap(dens, N, N, { x0: -L / 2 - dx / 2, x1: L / 2 - dx / 2, y0: -L / 2 - dx / 2, y1: L / 2 - dx / 2, vmin: 0, vmax: Math.max(vmax, 1e-12), cmap: "inferno", scale: P.scale });
          // the wall (the parts outside the slits)
          const d = P.sep, w = Math.min(P.width, d - 0.2);
          const segs = [[-L / 2, -d / 2 - w / 2], [-d / 2 + w / 2, d / 2 - w / 2], [d / 2 + w / 2, L / 2]];
          for (const [a, b] of segs) if (b > a) pp.rect(-BAR, a, BAR, b, { color: "#d0d7de", alpha: 0.55, stroke: "rgba(255,255,255,0.8)" });
          const edge = (L / 2) * (1 - 0.12);
          pp.rect(-edge, -edge, edge, edge, { fill: false, stroke: "rgba(139,152,168,0.45)", dash: [4, 4] });
          pp.vline(-L / 2 + iS * dx, { color: PlotColors.accent, dash: [6, 4], width: 1.4 });
          pp.label([`t = ${PM.fmt(t, 2)}`], "tl");
          pp.label(["screen"], "tr", { color: PlotColors.accent });

          // ---- screen
          const ps = plots.scr;
          ps.clear();
          let pmax = 0; for (let q = 0; q < plate.length; q++) if (plate[q] > pmax) pmax = plate[q];
          if (nHits > 0) ps.heatmap(plate, PLATE_W, PLATE_H, { x0: 0, x1: 1.12, y0: -L / 2, y1: L / 2, vmin: 0, vmax: Math.max(1.1, Math.min(pmax, 3)), cmap: "ice", smooth: false });
          let amax = 0, asum = 0; for (let j = 0; j < N; j++) { amax = Math.max(amax, acc[j]); asum += acc[j]; }
          if (P.showHist && nHits >= 150 && amax > 0) {
            // scale the histogram so that its area equals the area under the accumulated-intensity curve
            const bw = L / HBINS, area = (asum * dx) / amax;
            let hs = 0; for (let b = 0; b < HBINS; b++) hs += hist[b];
            const f = area / (hs * bw);
            for (let b = 0; b < HBINS; b++) {
              const y0 = -L / 2 + b * bw;
              if (hist[b] > 0) ps.rect(0, y0, Math.min(hist[b] * f, 1.12), y0 + bw * 0.92, { color: PlotColors.accent3, alpha: 0.35 });
            }
          }
          if (amax > 0 && asum * dx > 2e-4) {
            for (let j = 0; j < N; j++) accX[j] = acc[j] / amax;
            ps.line(accX, yGrid, { color: PlotColors.accent, width: 2 });
          }
          const items = [{ label: "single hits", color: "#b4dcff", type: "dot" }, { label: "∫|ψ(xₛ,y,t)|² dt", color: PlotColors.accent }];
          if (P.showHist) items.push({ label: "hit histogram", color: PlotColors.accent3, type: "box" });
          ps.legend(items, "tr");
          ps.label([`${nHits} hits`], "bl");
          const lam = (2 * Math.PI) / P.k0, D = -L / 2 + iS * dx;
          const y1 = twoSourceY(1, lam, D);
          if (P.showPred && isFinite(y1)) {
            ps.custom((c, pl) => {
              c.strokeStyle = PlotColors.pink; c.lineWidth = 2; c.beginPath();
              for (let n = -40; n <= 40; n++) {
                const yn = twoSourceY(n, lam, D);
                if (!isFinite(yn) || Math.abs(yn) > L / 2) continue;
                const Y = pl.Y(yn), X = pl.X(1.12);
                c.moveTo(X - 12, Y); c.lineTo(X, Y);
              }
              c.stroke();
            });
          }

          M.set("t", PM.fmt(t, 2));
          M.set("trans", PM.fmt(transNow, 3));
          M.set("norm", PM.fmt(normNow, 3));
          M.set("hits", `${nHits} / ${shots}`);
          M.set("fr", PM.fmt((lam * D) / P.sep, 2));
          M.set("fr2", isFinite(y1) ? PM.fmt(y1, 2) : "—");
          const fm = measuredSpacing();
          M.set("frm", isFinite(fm) ? PM.fmt(fm, 2) : "— (accumulating)");
          api.setTime(`t = ${PM.fmt(t, 2)}`);
        },
      };
    },
  });
})();
