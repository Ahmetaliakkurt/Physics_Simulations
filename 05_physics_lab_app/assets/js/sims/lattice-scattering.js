/* 2D wave-packet scattering from periodic arrays of Gaussian bumps — live split-step Fourier on a 256×256 grid. */
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


  const N = 256, L = 60, dx = L / N, DT = 1 / 60, A_LAT = 4, R_BUMP = 0.8, R0 = 15, THICK = 6;
  const dk = (2 * Math.PI) / L, KMAX = (N / 2) * dk;
  const GAMMA = 0.7, GLUT = 2048;

  /** Scatterer positions (same ordering and spacing as the author's original code). */
  function latticeSites(kind) {
    const pts = [], a = A_LAT;
    if (kind === "sc" || kind === "bcc") {
      for (let i = 0; i < L / 2 - 1e-9; i += a) for (let j = -L / 2; j < L / 2 + a - 1e-9; j += a) pts.push([i, j]);
      if (kind === "bcc")
        for (let i = a / 2; i < L / 2 - 1e-9; i += a) for (let j = -L / 2 + a / 2; j < L / 2 + a - 1e-9; j += a) pts.push([i, j]);
    } else {
      const dxf = (a * Math.sqrt(3)) / 2;
      let row = 0;
      for (let i = 0; i < L / 2 - 1e-9; i += dxf) {
        const sh = row % 2 === 1 ? a / 2 : 0;
        for (let j = -L / 2 - a; j < L / 2 + a - 1e-9; j += a) pts.push([i, j + sh]);
        row++;
      }
    }
    return pts;
  }
  /** Reciprocal-lattice vectors G (inside |G| < gmax) of the infinite 2D lattice behind each arrangement. */
  function reciprocal(kind, gmax) {
    const a = A_LAT, out = [], g = (2 * Math.PI) / a;
    let b1, b2;
    if (kind === "fcc") { b1 = [-g / Math.sqrt(3), g]; b2 = [(2 * g) / Math.sqrt(3), 0]; } // triangular lattice, a1=(0,a), a2=(a√3/2,a/2)
    else { b1 = [g, 0]; b2 = [0, g]; }
    const H = Math.ceil((2 * gmax) / g) + 2;
    for (let h = -H; h <= H; h++) for (let k = -H; k <= H; k++) {
      if (h === 0 && k === 0) continue;
      if (kind === "bcc" && ((h + k) & 1)) continue; // centred cell: structure factor 1 + (−1)^{h+k} vanishes for odd h+k
      const gx = h * b1[0] + k * b2[0], gy = h * b1[1] + k * b2[1];
      if (gx * gx + gy * gy < gmax * gmax) out.push([gx, gy]);
    }
    return out;
  }

  App.register({
    id: "lattice-scattering",
    category: "quantum",
    order: 35,
    title: "Wave Packet Scattering from a Lattice",
    icon: "🧱",
    subtitle: "A 2D wave packet scatters from a periodic array of Gaussian bumps: interference in real space, discrete Laue/Bragg spots on the elastic ring in momentum space.",
    notes: [{
      type: "info",
      html: "The “centred” and “face-centred” options are <b>two-dimensional analogues</b> named after the 3D BCC and FCC Bravais lattices: " +
        "a square lattice with an extra site in every cell centre, and a triangular (close-packed) lattice. They do not model 3D crystal " +
        "symmetry or 3D Bragg conditions. The diffraction pattern in momentum space, however, is genuine physics: the wave scattered by a " +
        "periodic potential concentrates in discrete directions fixed by the reciprocal lattice (the 2D Laue condition).",
    }],
    animated: true,
    speed: { min: 0.25, max: 2, value: 1, step: 0.05 },
    controls: [
      { id: "lat", type: "select", label: "Arrangement", value: "sc",
        options: [{ value: "sc", label: "Square lattice" }, { value: "bcc", label: "Centred arrangement (BCC-like, 2D)" }, { value: "fcc", label: "Face-centred arrangement (FCC-like, 2D)" }] },
      { id: "theta", type: "slider", label: "Angle of incidence $\\theta$", min: -45, max: 45, step: 1, value: 0, unit: "°" },
      { id: "sig", type: "select", label: "Packet width $\\sigma$", value: "0.25",
        options: [{ value: "0.25", label: "σ = a/4 (broad in k)" }, { value: "0.5", label: "σ = a/2" }, { value: "1", label: "σ = a (sharp in k)" }] },
      { id: "k0", type: "slider", label: "Momentum magnitude $|k_0|$", min: 4, max: 12, step: 0.5, value: 10,
        help: "The largest wavenumber representable on the 256×256 grid is π/Δx ≈ 13.4." },
      { id: "V0", type: "slider", label: "Bump height $V_0$", min: 0, max: 150, step: 5, value: 80,
        help: "$V_0=0$: free packet (for comparison)." },
      { type: "section", label: "Display" },
      { id: "showSites", type: "checkbox", label: "Show scatterer positions", value: true, live: true },
      { id: "kscale", type: "select", label: "Momentum panel scale", value: "log", live: true,
        options: [{ value: "log", label: "Logarithmic (4 decades)" }, { value: "gamma", label: "|ψ̃|^0.7 (fixed scale)" }] },
      { id: "showRing", type: "checkbox", label: "Show the elastic ring $|k|=|k_0|$", value: true, live: true },
      { id: "showEwald", type: "checkbox", label: "Show Ewald construction $k_0+G$", value: true, live: true,
        help: "Reciprocal-lattice points shifted to the tip of $k_0$; the ones lying on the ring (within the packet's spread $1/2\\sigma$) satisfy the Laue condition and are circled." },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A spinless particle moves in the plane, in a square box of side $L=60$ (centred at the origin). The right half-plane $x\\ge0$
      is filled with identical repulsive Gaussian bumps — a toy "crystal" with lattice constant $a=4$ and bump radius $r_0=0.8$. The
      particle starts at distance $R_0=15$ from the origin as a Gaussian packet aimed at the origin with momentum
      $\\mathbf k_0=|k_0|(\\cos\\theta,\\sin\\theta)$. Units: $\\hbar=m=1$, so the mean kinetic energy is $E=|k_0|^2/2$ and $V_0$ is
      measured in the same units (e.g. $|k_0|=10$ gives $E=50$). Three arrangements of the bumps are offered:</p>
      <ul>
        <li><b>Square lattice</b> — sites at $(ma, na)$.</li>
        <li><b>Centred arrangement (BCC-like, 2D)</b> — the square lattice plus one extra site in every cell centre. In 2D this is
        itself a square lattice of spacing $a/\\sqrt2$ rotated by 45°.</li>
        <li><b>Face-centred arrangement (FCC-like, 2D)</b> — rows spaced $a\\sqrt3/2$ apart, every second row shifted by $a/2$: the
        triangular (close-packed) lattice, the 2D analogue of a close-packed FCC (111) plane.</li>
      </ul>

      <h4>Equations being solved</h4>
      <p>The two-dimensional time-dependent Schrödinger equation with the bump potential:</p>
      $$ i\\hbar\\frac{\\partial\\psi}{\\partial t}=\\left[-\\frac{\\hbar^2}{2m}\\nabla^2+V(\\mathbf r)\\right]\\psi,\\qquad
         V(\\mathbf r)=V_0\\sum_{\\mathbf R} \\exp\\!\\left(-\\frac{|\\mathbf r-\\mathbf R|^2}{r_0^2}\\right), $$
      <p>with initial state $\\psi(\\mathbf r,0)\\propto e^{-|\\mathbf r-\\mathbf r_0|^2/4\\sigma^2}\\,e^{i\\mathbf k_0\\cdot\\mathbf r}$,
      $\\mathbf r_0=-R_0(\\cos\\theta,\\sin\\theta)$. Its momentum distribution is a Gaussian of width $\\Delta k=1/2\\sigma$ around $\\mathbf k_0$.</p>
      <p><b>Why momentum space shows peaks.</b> In the Born approximation the scattered amplitude in direction $\\mathbf k'$ is
      proportional to the Fourier transform $\\tilde V(\\mathbf q)$ at the momentum transfer $\\mathbf q=\\mathbf k'-\\mathbf k_0$. For a
      periodic potential $\\tilde V(\\mathbf q)=f(\\mathbf q)\\,S(\\mathbf q)$: the form factor of one bump,
      $f(q)=\\pi r_0^2V_0\\,e^{-q^2r_0^2/4}$, times the structure factor $S(\\mathbf q)=\\sum_{\\mathbf R}e^{-i\\mathbf q\\cdot\\mathbf R}$, which is
      sharply peaked at reciprocal-lattice vectors $\\mathbf G$ (defined by $e^{i\\mathbf G\\cdot\\mathbf R}=1$ for every site). Energy
      conservation (elastic scattering) adds $|\\mathbf k'|=|\\mathbf k_0|$. Together they give the 2D Laue condition:</p>
      <div class="callout">$$ \\mathbf k'=\\mathbf k_0+\\mathbf G,\\qquad |\\mathbf k'|=|\\mathbf k_0|
        \\quad\\Longleftrightarrow\\quad 2\\,\\mathbf k_0\\cdot\\mathbf G+G^2=0 \\quad\\Longleftrightarrow\\quad 2d\\sin\\theta_B=n\\lambda $$</div>
      <p>The last form is Bragg's law for a family of lattice lines with spacing $d=2\\pi n/|\\mathbf G|$ and glancing angle $\\theta_B$.
      Geometrically (Ewald construction): draw the reciprocal lattice with its origin at the tip of $\\mathbf k_0$; every point that falls on
      the circle $|\\mathbf k|=|\\mathbf k_0|$ is an allowed diffracted beam. The reciprocal lattices used are
      $\\mathbf G=\\frac{2\\pi}{a}(h,k)$ for the square lattice, the same with $h+k$ even for the centred arrangement (the structure factor
      $1+(-1)^{h+k}$ of the two-site cell kills the odd ones), and the hexagonal lattice spanned by
      $\\mathbf b_1=\\frac{2\\pi}{a}(-1/\\sqrt3,\\,1)$, $\\mathbf b_2=\\frac{2\\pi}{a}(2/\\sqrt3,\\,0)$ for the triangular one, $|\\mathbf G_{min}|=4\\pi/(\\sqrt3a)$.
      Because the crystal only fills a half-plane, the momentum parallel to its surface is the strictly conserved one,
      $k'_y=k_{0y}+2\\pi n/a$ (a grating condition); the Bragg condition along $x$ is softened by the finite depth, and the packet's own
      spread $\\Delta k=1/2\\sigma$ blurs every spot. With $V_0$ comparable to $E$ the scattering is strong (multiple scattering, beyond Born),
      but the reciprocal-lattice selection rule still holds because it follows from periodicity alone (Bloch's theorem).</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Grid:</b> $256\\times256$ points, $\\Delta x=L/256\\approx0.234$; wavenumbers $k=2\\pi n/L$ with $|k|\\le\\pi/\\Delta x\\approx13.4$.</li>
        <li><b>Integrator:</b> second-order Strang split-step Fourier with $\\Delta t=1/60$:
        $$ \\psi(t+\\Delta t)= e^{-iV\\Delta t/2\\hbar}\\,\\mathcal F^{-1}\\!\\left[e^{-i\\hbar k^2\\Delta t/2m}\\,\\mathcal F\\!\\left[e^{-iV\\Delta t/2\\hbar}\\,\\psi(t)\\right]\\right]. $$
        The kinetic factor is exact in Fourier space; the local error per step is $\\mathcal O(\\Delta t^3)$ from the commutator $[T,V]$.
        Consecutive half-steps of $V$ are merged into one full step, and each FFT pair is a hand-optimised radix-4 2D transform. At most
        two steps are taken per animation frame, so one second of real time ≈ one time unit at speed ×1.</li>
        <li><b>Boundaries:</b> the FFT makes the box periodic, so a 6-unit sponge layer along all four edges multiplies $\\psi$ by
        $e^{-W\\Delta t}$ each step, $W=\\delta_x^2+\\delta_y^2$, where $\\delta$ is the penetration depth into the layer — equivalent to an
        imaginary potential $-iW$. It absorbs outgoing waves instead of letting them re-enter from the opposite side.</li>
        <li><b>Plots:</b> left, $|\\psi(x,y,t)|^2$ with fixed colour scale (40 % of the initial peak). Right, $|\\tilde\\psi(k_x,k_y,t)|^2$
        obtained by an extra 2D FFT (every second frame while playing), shown on a logarithmic scale spanning 4 decades or as
        $(|\\tilde\\psi|^2/\\max_0)^{0.7}$ on a fixed scale; the dashed ring is $|\\mathbf k|=|\\mathbf k_0|$ and the small dots are $\\mathbf k_0+\\mathbf G$.</li>
        <li><b>Metrics:</b> "Probability in box" $=\\sum|\\psi|^2\\Delta x^2$ is the monitored conserved quantity — exactly 1 for a unitary
        evolution, it decreases only when probability enters the absorbing layer. "In the lattice region" is the same sum over $x>0$.</li>
      </ul>

      <h4>What to try</h4>
      <ul>
        <li><b>Free packet first.</b> Set $V_0=0$: the momentum distribution stays a single blob at $\\mathbf k_0$ (no forces), while the
        real-space packet spreads. Then restore $V_0=80$ and watch new spots appear on the ring.</li>
        <li><b>Sharp beams need a wide packet.</b> Switch $\\sigma$ from $a/4$ to $a$: $\\Delta k$ drops from 0.5 to 0.125 and the
        smeared ring breaks up into discrete beams lying on horizontal lines $k'_y=k_{0y}+2\\pi n/a$ (spacing $\\approx1.57$ for $a=4$) — the
        grating condition imposed by the periodicity along the crystal surface.</li>
        <li><b>Ewald construction.</b> The dots $\\mathbf k_0+\\mathbf G$ form the reciprocal lattice hung on the tip of $\\mathbf k_0$. Every bright
        line passes through a row of dots, and the circled dots (on the ring within $\\Delta k$) are the beams that also satisfy the bulk
        Bragg condition along $x$. Change $|k_0|$ or $\\theta$ and watch which dots the ring passes through.</li>
        <li><b>Compare arrangements.</b> The centred arrangement removes every odd-$(h+k)$ reflection (half of the dots disappear and the rest
        form a lattice rotated by 45°); the triangular arrangement has a hexagonal reciprocal lattice with $|\\mathbf G_{min}|\\approx1.81$.</li>
        <li><b>Low energy.</b> At $|k_0|=4$ ($E=8\\ll V_0$) the bump tops are classically forbidden, but $V$ drops below $E$ already
        $r_0\\sqrt{\\ln(V_0/E)}\\approx1.2$ from each centre, leaving open channels between the bumps: roughly half of the probability still
        enters the lattice. The reflected wave shows the specular beam ($k'_x=-k_{0x}$, $k'_y=k_{0y}$) plus a few grating orders.</li>
      </ul>

      <h4>Limitations &amp; further reading</h4>
      <p>A 2D toy model: the "BCC-like" and "FCC-like" arrangements are planar analogues, not 3D crystals, and the crystal is finite
      (it fills only $0\\le x\\lt 30$ inside the box, about 8 lattice planes deep). The grid resolves $|k|\\le13.4$, so very fast packets alias. Reading: N. W. Ashcroft &amp;
      N. D. Mermin, <i>Solid State Physics</i>, ch. 5–6 (reciprocal lattice, Laue/Bragg, Ewald); C. Kittel, <i>Introduction to Solid State
      Physics</i>, ch. 2; J. J. Sakurai, <i>Modern Quantum Mechanics</i>, §6.2 (Born approximation).</p>`,

    mount(api) {
      const P = api.params, NN = N * N;
      const fft = makeFFT2(N);
      const re = new Float64Array(NN), im = new Float64Array(NN);
      const wr = new Float64Array(NN), wi = new Float64Array(NN);        // copy for the momentum view
      const dens = new Float64Array(NN), kdisp = new Float64Array(NN);
      const V = new Float64Array(NN);
      const vhR = new Float64Array(NN), vhI = new Float64Array(NN);
      const vaR = new Float64Array(NN), vaI = new Float64Array(NN);
      const vhaR = new Float64Array(NN), vhaI = new Float64Array(NN);
      const kR = new Float64Array(NN), kI = new Float64Array(NN);
      const kx = PM.fftk(N, dx);
      const glut = new Float64Array(GLUT + 1);
      for (let q = 0; q < GLUT; q++) glut[q] = Math.pow(q / GLUT, GAMMA);
      glut[GLUT] = 1;
      let t, simAcc, sites, gvecs, rVmax, kVmax, normNow, pRight, frameNo = 0;

      const M = api.metrics([
        { id: "t", label: "Time $t$" },
        { id: "E", label: "Energy $E=k_0^2/2$" },
        { id: "dk", label: "Packet spread $\\Delta k=1/2\\sigma$" },
        { id: "norm", label: "Probability in box" },
        { id: "pin", label: "In the lattice region ($x>0$)" },
      ]);
      const plots = api.plots([
        { id: "real", title: "Real space: |ψ(x,y,t)|²", aspect: 1, xlim: [-L / 2, L / 2], ylim: [-L / 2, L / 2], equal: true, xlabel: "x", ylabel: "y", grid: false, maxHeight: 620 },
        { id: "mom", title: "Momentum space: |ψ̃(kx,ky)|² (Laue/Bragg spots)", aspect: 1, xlim: [-KMAX, KMAX], ylim: [-KMAX, KMAX], equal: true, xlabel: "kx", ylabel: "ky", grid: false, maxHeight: 620 },
      ]);

      function buildOperators() {
        sites = latticeSites(P.lat);
        gvecs = reciprocal(P.lat, 2 * KMAX);
        V.fill(0);
        const V0 = P.V0, cut = 4 * R_BUMP, ir2 = 1 / (R_BUMP * R_BUMP);
        // add each bump only inside its own bounding box (fast)
        for (const [sx, sy] of sites) {
          const i0 = Math.max(0, Math.ceil((sx - cut + L / 2) / dx)), i1 = Math.min(N - 1, Math.floor((sx + cut + L / 2) / dx));
          const j0 = Math.max(0, Math.ceil((sy - cut + L / 2) / dx)), j1 = Math.min(N - 1, Math.floor((sy + cut + L / 2) / dx));
          for (let j = j0; j <= j1; j++) {
            const y = -L / 2 + j * dx;
            for (let i = i0; i <= i1; i++) {
              const x = -L / 2 + i * dx;
              V[j * N + i] += V0 * Math.exp(-((x - sx) * (x - sx) + (y - sy) * (y - sy)) * ir2);
            }
          }
        }
        const sc = DT; // sponge: amplitude factor e^{-W·Δt} per step, W = δx² + δy²
        for (let j = 0; j < N; j++) {
          const y = -L / 2 + j * dx;
          for (let i = 0; i < N; i++) {
            const x = -L / 2 + i * dx, p = j * N + i;
            let s = 0;
            const ex = Math.abs(x) - (L / 2 - THICK), ey = Math.abs(y) - (L / 2 - THICK);
            if (ex > 0) s += ex * ex;
            if (ey > 0) s += ey * ey;
            const A = Math.exp(-sc * s), v = V[p];
            vhR[p] = Math.cos(-v * DT / 2); vhI[p] = Math.sin(-v * DT / 2);
            vaR[p] = A * Math.cos(-v * DT); vaI[p] = A * Math.sin(-v * DT);
            vhaR[p] = A * vhR[p]; vhaI[p] = A * vhI[p];
            // spectrum is in [kx][ky] order after the forward FFT; k² is symmetric so the index order does not matter
            const ph = -0.5 * (kx[i] * kx[i] + kx[j] * kx[j]) * DT;
            kR[p] = Math.cos(ph) / NN; kI[p] = Math.sin(ph) / NN;
          }
        }
      }
      function launch() {
        const th = (P.theta * Math.PI) / 180, kxx = P.k0 * Math.cos(th), kyy = P.k0 * Math.sin(th);
        const x0 = -R0 * Math.cos(th), y0 = -R0 * Math.sin(th), sg = parseFloat(P.sig) * A_LAT, s4 = 4 * sg * sg;
        let nrm = 0;
        for (let j = 0; j < N; j++) {
          const y = -L / 2 + j * dx;
          for (let i = 0; i < N; i++) {
            const x = -L / 2 + i * dx, p = j * N + i;
            const g = Math.exp(-((x - x0) * (x - x0) + (y - y0) * (y - y0)) / s4), ph = kxx * x + kyy * y;
            re[p] = g * Math.cos(ph); im[p] = g * Math.sin(ph); nrm += g * g;
          }
        }
        const f = 1 / Math.sqrt(nrm * dx * dx);
        for (let p = 0; p < NN; p++) { re[p] *= f; im[p] *= f; }
      }
      function mul(aR, aI) {
        for (let p = 0; p < NN; p++) { const r = re[p], m = im[p]; re[p] = r * aR[p] - m * aI[p]; im[p] = r * aI[p] + m * aR[p]; }
      }
      function stepN(n) {
        mul(vhR, vhI);
        for (let s = 0; s < n; s++) {
          fft.forward(re, im);
          for (let p = 0; p < NN; p++) { const r = re[p], m = im[p]; re[p] = r * kR[p] - m * kI[p]; im[p] = r * kI[p] + m * kR[p]; }
          fft.inverse(re, im);
          if (s < n - 1) mul(vaR, vaI); else mul(vhaR, vhaI);
          t += DT;
        }
      }
      /** |ψ|² and the fftshifted |ψ̃(k)|²; returns their maxima. */
      function compute(withK) {
        let rmax = 0, tot = 0, right = 0;
        const iMid = N / 2;
        for (let j = 0; j < N; j++) {
          const o = j * N;
          for (let i = 0; i < N; i++) {
            const p = o + i, d = re[p] * re[p] + im[p] * im[p];
            dens[p] = d; tot += d; if (i >= iMid) right += d;
            if (d > rmax) rmax = d;
          }
        }
        normNow = tot * dx * dx; pRight = right * dx * dx;
        if (!withK) return { rmax, kmax: 0 };
        wr.set(re); wi.set(im);
        fft.forward(wr, wi);
        let kmax = 0;
        const h = N / 2;
        for (let a = 0; a < N; a++) {       // a: kx index
          const col = (a + h) % N, o = a * N;
          for (let b = 0; b < N; b++) {     // b: ky index
            const p = o + b, d = wr[p] * wr[p] + wi[p] * wi[p];
            kdisp[((b + h) % N) * N + col] = d;
            if (d > kmax) kmax = d;
          }
        }
        return { rmax, kmax };
      }
      function setMetrics() {
        M.set("t", PM.fmt(t, 2));
        M.set("E", PM.fmt((P.k0 * P.k0) / 2, 1));
        M.set("dk", PM.fmt(1 / (2 * parseFloat(P.sig) * A_LAT), 3));
        M.set("norm", PM.fmt(normNow, 4));
        M.set("pin", PM.fmt(pRight, 3));
        api.setTime(`t = ${PM.fmt(t, 2)}`);
      }
      function gammaMap(vmax) {
        const inv = GLUT / vmax;
        for (let p = 0; p < NN; p++) {
          const q = kdisp[p] * inv;
          kdisp[p] = q >= GLUT ? 1 : glut[q | 0];
        }
      }
      function drawEwald(pm) {
        const th = (P.theta * Math.PI) / 180, k0x = P.k0 * Math.cos(th), k0y = P.k0 * Math.sin(th);
        const tol = 1 / (2 * parseFloat(P.sig) * A_LAT);
        pm.custom((c, p) => {
          c.fillStyle = "rgba(165,214,255,0.55)";
          for (const [gx, gy] of gvecs) {
            const qx = k0x + gx, qy = k0y + gy;
            if (Math.abs(qx) > KMAX || Math.abs(qy) > KMAX) continue;
            c.fillRect(p.X(qx) - 1.25, p.Y(qy) - 1.25, 2.5, 2.5);
          }
          c.strokeStyle = "#a5d6ff"; c.lineWidth = 1.3;
          c.beginPath();
          for (const [gx, gy] of gvecs) {
            const qx = k0x + gx, qy = k0y + gy;
            if (Math.abs(qx) > KMAX || Math.abs(qy) > KMAX) continue;
            if (Math.abs(Math.hypot(qx, qy) - P.k0) > tol) continue;
            const X = p.X(qx), Y = p.Y(qy);
            c.moveTo(X + 6, Y); c.arc(X, Y, 6, 0, 2 * Math.PI);
          }
          c.stroke();
        });
      }

      return {
        reset() {
          buildOperators();
          launch();
          t = 0; simAcc = 0;
          const m = compute(true); frameNo = 0;
          rVmax = 0.4 * m.rmax;               // as in the original: 40 % of the initial maximum
          kVmax = m.kmax;                     // gamma scale relative to the initial momentum-space maximum
        },
        step(dt) {
          simAcc += dt;
          let n = Math.floor(simAcc / DT);
          if (n > 2) { n = 2; simAcc = 0; } else simAcc -= n * DT;
          if (n > 0) stepN(n);
        },
        render() {
          // the momentum panel (one extra 2D FFT) is refreshed only every second frame while playing
          const doK = !api.isPlaying || (frameNo++ & 1) === 0;
          const cm = compute(doK);
          const pr = plots.real;
          pr.clear();
          pr.heatmap(dens, N, N, { x0: -L / 2 - dx / 2, x1: L / 2 - dx / 2, y0: -L / 2 - dx / 2, y1: L / 2 - dx / 2, vmin: 0, vmax: rVmax, cmap: "turbo" });
          if (P.showSites && P.V0 > 0) {
            pr.custom((c, p) => {
              c.strokeStyle = "rgba(255,255,255,0.38)"; c.lineWidth = 1;
              c.beginPath();
              const rr = Math.max(R_BUMP * p.sx, 1.5);
              for (const [sx, sy] of sites) { const X = p.X(sx), Y = p.Y(sy); c.moveTo(X + rr, Y); c.arc(X, Y, rr, 0, 2 * Math.PI); }
              c.stroke();
            });
          }
          pr.vline(0, { color: "#a5d6ff", dash: [5, 4], alpha: 0.35 });
          const ed = L / 2 - THICK;
          pr.rect(-ed, -ed, ed, ed, { fill: false, stroke: "rgba(139,152,168,0.45)", dash: [4, 4] });
          pr.label(`t = ${PM.fmt(t, 2)}`, "tl");

          if (!doK) { setMetrics(); return; }
          const pm = plots.mom, kx0 = -KMAX - dk / 2, kx1 = KMAX - dk / 2;
          pm.clear();
          if (P.kscale === "log") {
            pm.heatmap(kdisp, N, N, { x0: kx0, x1: kx1, y0: kx0, y1: kx1, vmin: cm.kmax * 1e-4, vmax: cm.kmax, scale: "log", cmap: "inferno" });
          } else {
            gammaMap(kVmax);
            pm.heatmap(kdisp, N, N, { x0: kx0, x1: kx1, y0: kx0, y1: kx1, vmin: 0, vmax: 0.6, cmap: "inferno" });
          }
          pm.hline(0, { color: PlotColors.muted, dash: [2, 4], alpha: 0.5 });
          pm.vline(0, { color: PlotColors.muted, dash: [2, 4], alpha: 0.5 });
          if (P.showRing) {
            pm.custom((c, p) => {
              c.strokeStyle = PlotColors.bad; c.globalAlpha = 0.75; c.lineWidth = 1.5; c.setLineDash([6, 4]);
              c.beginPath(); c.arc(p.X(0), p.Y(0), P.k0 * p.sx, 0, 2 * Math.PI); c.stroke();
            });
          }
          if (P.showEwald && P.V0 > 0) drawEwald(pm);
          const th = (P.theta * Math.PI) / 180;
          pm.circle(P.k0 * Math.cos(th), P.k0 * Math.sin(th), 3.5, { px: true, fill: false, stroke: PlotColors.accent, strokeWidth: 1.5 });
          const items = [{ label: "incident k₀", color: PlotColors.accent, type: "dot" }];
          if (P.showRing) items.push({ label: "|k| = |k₀|", color: PlotColors.bad, dash: [6, 4] });
          if (P.showEwald && P.V0 > 0) items.push({ label: "k₀ + G", color: "#a5d6ff", type: "dot" });
          pm.legend(items, "tl");

          setMetrics();
        },
      };
    },
  });
})();
