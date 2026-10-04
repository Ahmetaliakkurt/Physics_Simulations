/* Hydrogen atom (Z=1) — probability cloud of ψ_nlm = R_nl(r)·Y_l^m(θ,φ) (importance sampling, live 3D rotation)
 * and a heat map of |ψ|² in a chosen plane. */
(function () {
  "use strict";

  const LETTERS = "spdfgh";
  /** Reset the context state before Plot.clear() (clears a clip stack left over from a resize). */
  function hardClear(p) {
    const c = p.ctx;
    if (typeof c.reset === "function") c.reset(); else p.canvas.width = p.canvas.width;
    p._clipped = false;
    p.clear();
  }


  // ------------------------------------------------------------ orbital helpers
  /** Name of a real orbital: pₓ, d(xy) … */
  function realLabel(l, m) {
    const T = {
      1: { "-1": "y", 0: "z", 1: "x" },
      2: { "-2": "xy", "-1": "yz", 0: "z²", 1: "xz", 2: "x²−y²" },
      3: { "-3": "y(3x²−y²)", "-2": "xyz", "-1": "yz²", 0: "z³", 1: "xz²", 2: "z(x²−y²)", 3: "x(x²−3y²)" },
    };
    if (l === 0) return "";
    if (T[l]) return T[l][m];
    return "m = " + (m > 0 ? "+" + m + ", cos" : m < 0 ? "−" + -m + ", sin" : "0");
  }
  function orbitalName(n, l, m, real) {
    const base = n + LETTERS[l];
    if (l === 0) return base;
    if (real) return base + " (" + realLabel(l, m) + ")";
    return base + " (m = " + (m > 0 ? "+" : "") + m + ")";
  }

  /** Angular part of the orbital: signed value (real) or amplitude + phase (complex). */
  function makeAngular(l, m, real) {
    const am = Math.abs(m), N = PM.sphHarmNorm(l, m);
    const cs = am % 2 ? -1 : 1; // (−1)^|m|
    // upper bound of |Y|²: max_c P² (× 2 for real orbitals with m≠0)
    let pmax = 0;
    for (let i = 0; i <= 2000; i++) { const p = PM.assocLegendre(l, am, -1 + i / 1000); if (p * p > pmax) pmax = p * p; }
    const ymax = N * N * pmax * (real && m !== 0 ? 2 : 1) * 1.01;
    return {
      ymax,
      /** returns the signed real-orbital value, or |Y| and its phase for complex orbitals: out=[value or |Y|, phase] */
      eval(c, phi, out) {
        const P = N * PM.assocLegendre(l, am, c);
        if (real) {
          let v = P;
          if (m > 0) v = Math.SQRT2 * cs * P * Math.cos(m * phi);
          else if (m < 0) v = Math.SQRT2 * cs * P * Math.sin(am * phi);
          out[0] = v; out[1] = v >= 0 ? 0 : Math.PI;
        } else {
          const amp = m < 0 ? cs * P : P; // Y_l^{-|m|} = (−1)^|m| conj(Y_l^{|m|})
          out[0] = Math.abs(amp);
          out[1] = m * phi + (amp < 0 ? Math.PI : 0);
        }
        return out;
      },
    };
  }

  /** Radial table: R(r), r²R², cumulative distribution. */
  function makeRadial(n, l, Z) {
    const NR = 3000, rmax = (4 * n * (n + l + 2)) / Z, h = rmax / (NR - 1);
    const r = new Float64Array(NR), R = new Float64Array(NR), p = new Float64Array(NR), cdf = new Float64Array(NR);
    for (let i = 0; i < NR; i++) { r[i] = i * h; R[i] = PM.radialH(n, l, r[i], Z); p[i] = r[i] * r[i] * R[i] * R[i]; }
    for (let i = 1; i < NR; i++) cdf[i] = cdf[i - 1] + 0.5 * h * (p[i] + p[i - 1]);
    const tot = cdf[NR - 1] || 1;
    for (let i = 0; i < NR; i++) cdf[i] /= tot;
    const quant = (q) => { let a = 0, b = NR - 1; while (b - a > 1) { const c = (a + b) >> 1; if (cdf[c] < q) a = c; else b = c; } return r[b]; };
    let imax = 0; for (let i = 1; i < NR; i++) if (p[i] > p[imax]) imax = i;
    const nodes = [];
    for (let i = 2; i < NR - 1; i++) if (R[i] * R[i - 1] < 0 && r[i] < quant(0.9999)) nodes.push(r[i - 1] - (R[i - 1] * h) / (R[i] - R[i - 1]));
    return {
      r, R, p, cdf, h, NR, rmp: r[imax], pmax: p[imax], quant, nodes,
      /** u ∈ [0,1) → r (inverse-transform sampling) */
      inv(u) {
        let a = 0, b = NR - 1;
        while (b - a > 1) { const c = (a + b) >> 1; if (cdf[c] <= u) a = c; else b = c; }
        const d = cdf[b] - cdf[a];
        return r[a] + (d > 0 ? ((u - cdf[a]) / d) * h : 0);
      },
      Rat(x) { const f = x / h, i = Math.min(Math.floor(f), NR - 2); const t = f - i; return R[i] * (1 - t) + R[i + 1] * t; },
    };
  }

  /** Sample a point cloud from the |ψ|² distribution. */
  function sampleCloud(rad, ang, count, seed, C) {
    const rng = new PM.RNG(seed), out = [0, 0];
    let k = 0, tries = 0;
    const maxTries = count * 400;
    while (k < count && tries < maxTries) {
      tries++;
      const c = 2 * rng.next() - 1, phi = 2 * Math.PI * rng.next();
      ang.eval(c, phi, out);
      if (rng.next() * ang.ymax >= out[0] * out[0]) continue;
      const r = rad.inv(rng.next()), s = Math.sqrt(Math.max(0, 1 - c * c));
      C.x[k] = r * s * Math.cos(phi); C.y[k] = r * s * Math.sin(phi); C.z[k] = r * c;
      const Rv = rad.Rat(r);
      C.rho[k] = Rv * Rv * out[0] * out[0];
      let ph = out[1] + (Rv < 0 ? Math.PI : 0);
      ph = ((ph % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
      C.phase[k] = ph;
      k++;
    }
    C.n = k;
    return C;
  }

  function parseRGB(s) { const m = s.match(/\d+/g); return [+m[0], +m[1], +m[2]]; }
  function hsl2rgb(h, s, l) {
    const a = s * Math.min(l, 1 - l);
    const f = (n) => { const k = (n + h / 30) % 12; return Math.round(255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)))); };
    return [f(0), f(8), f(4)];
  }

  const D = 4; // depth layers (far-to-near shading)
  const BG = [15, 21, 28];
  let TONE = null; // tone-mapping tables for additive blending (one per channel)
  function toneLUT() {
    if (TONE) return TONE;
    TONE = [0, 1, 2].map((ch) => {
      const L = new Uint8ClampedArray(4096);
      for (let i = 0; i < 4096; i++) L[i] = BG[ch] + (255 - BG[ch]) * (1 - Math.exp(-(i / 4) / 170));
      return L;
    });
    return TONE;
  }
  /**
   * Fast point cloud: points are "splatted" straight into a pixel buffer (instead of fillRect),
   * with additive (tone-mapped) or depth-sorted alpha blending.
   * pal: Float32Array(D*K*3) depth-shaded RGB.
   */
  function drawCloud(v, C, pal, K, size, alpha, glow, ext) {
    const W = v.canvas.width, H = v.canvas.height, dpr = v.dpr, n = C.n;
    if (!C.acc || C.accW !== W || C.accH !== H) {
      C.acc = new Float32Array(W * H * 3); C.img = v.ctx.createImageData(W, H); C.accW = W; C.accH = H;
      C.u32 = new Uint32Array(C.img.data.buffer);
    }
    const acc = C.acc, u32 = C.u32;
    const B = v._basis();
    const { cy, sy, cp, sp } = B, s = B.s * dpr, cx = B.cx * dpr, cz = B.cz * dpr;
    const cnt = C.cnt; cnt.fill(0);
    let bx0 = W, bx1 = 0, by0 = H, by1 = 0;
    for (let i = 0; i < n; i++) {
      const x = C.x[i], y = C.y[i], z = C.z[i];
      const xr = x * cy - y * sy, yr = x * sy + y * cy;
      const up = z * cp - yr * sp, depth = z * sp + yr * cp;
      const px = cx + xr * s, py = cz - up * s;
      C.px[i] = px; C.py[i] = py;
      if (px < bx0) bx0 = px; if (px > bx1) bx1 = px; if (py < by0) by0 = py; if (py > by1) by1 = py;
      let d = Math.floor(((depth / ext + 1) / 2) * D); d = d < 0 ? 0 : d >= D ? D - 1 : d;
      const key = d * K + C.ci[i];
      C.key[i] = key; cnt[key + 1]++;
    }
    const w = Math.max(1, Math.round(2 * size * dpr)), h0 = (w - 1) / 2;
    // only the bounding rectangle of the points is processed
    const X0 = Math.max(0, Math.floor(bx0 - w)), X1 = Math.min(W, Math.ceil(bx1 + w + 1));
    const Y0 = Math.max(0, Math.floor(by0 - w)), Y1 = Math.min(H, Math.ceil(by1 + w + 1));
    if (n === 0 || X1 <= X0 || Y1 <= Y0) return;
    const splat = (i, r, g, b, blend) => {
      const x0 = Math.round(C.px[i] - h0), y0 = Math.round(C.py[i] - h0);
      const xa = x0 < 0 ? 0 : x0, xb = x0 + w > W ? W : x0 + w, ya = y0 < 0 ? 0 : y0, yb = y0 + w > H ? H : y0 + w;
      for (let yy = ya; yy < yb; yy++) {
        const qe = (yy * W + xb) * 3;
        if (blend) for (let q = (yy * W + xa) * 3; q < qe; q += 3) { acc[q] += (r - acc[q]) * alpha; acc[q + 1] += (g - acc[q + 1]) * alpha; acc[q + 2] += (b - acc[q + 2]) * alpha; }
        else for (let q = (yy * W + xa) * 3; q < qe; q += 3) { acc[q] += r; acc[q + 1] += g; acc[q + 2] += b; }
      }
    };
    if (glow) {
      for (let y = Y0; y < Y1; y++) acc.fill(0, (y * W + X0) * 3, (y * W + X1) * 3);
      const a = alpha * 255 * 4 * 0.9;
      for (let i = 0; i < n; i++) { const k3 = C.key[i] * 3; splat(i, pal[k3] * a, pal[k3 + 1] * a, pal[k3 + 2] * a, false); }
      const T = toneLUT(), T0 = T[0], T1 = T[1], T2 = T[2];
      for (let y = Y0; y < Y1; y++) {
        for (let p = y * W + X0, pe = y * W + X1, q = p * 3; p < pe; p++, q += 3) {
          const r = acc[q], g = acc[q + 1], b = acc[q + 2];
          u32[p] = 0xff000000 | (T2[b < 4095 ? b | 0 : 4095] << 16) | (T1[g < 4095 ? g | 0 : 4095] << 8) | T0[r < 4095 ? r | 0 : 4095];
        }
      }
    } else {
      for (let y = Y0; y < Y1; y++) for (let q = (y * W + X0) * 3, qe = (y * W + X1) * 3; q < qe; q += 3) { acc[q] = BG[0]; acc[q + 1] = BG[1]; acc[q + 2] = BG[2]; }
      for (let q = 1; q < cnt.length; q++) cnt[q] += cnt[q - 1];
      const start = C.start; start.set(cnt);
      for (let i = 0; i < n; i++) C.order[start[C.key[i]]++] = i;
      for (let qq = 0; qq < n; qq++) { const i = C.order[qq], k3 = C.key[i] * 3; splat(i, pal[k3] * 255, pal[k3 + 1] * 255, pal[k3 + 2] * 255, true); }
      for (let y = Y0; y < Y1; y++) {
        for (let p = y * W + X0, pe = y * W + X1, q = p * 3; p < pe; p++, q += 3) u32[p] = 0xff000000 | ((acc[q + 2] | 0) << 16) | ((acc[q + 1] | 0) << 8) | (acc[q] | 0);
      }
    }
    v.ctx.putImageData(C.img, 0, 0, X0, Y0, X1 - X0, Y1 - Y0);
  }


  App.register({
    id: "hydrogen-orbitals",
    category: "quantum",
    order: 34,
    title: "Hydrogen Atom 3D Orbitals",
    icon: "🌐",
    subtitle: "The exact hydrogen eigenstates ψ_nlm: an electron cloud sampled from |ψ_nlm|² that you can rotate in 3D, a planar cut of the density and the radial distribution.",
    notes: [
      { type: "info", html: "Drag the cloud on the left to rotate it and use the mouse wheel to zoom. The heat map on the right is $|\\psi|^2$ in the chosen plane (default $y=0$, i.e. the $x$–$z$ plane); dashed circles mark the radial node spheres." },
    ],
    animated: true,
    speed: { min: 0, max: 3, value: 1, step: 0.1 },
    controls: [
      { type: "section", label: "Quantum numbers" },
      { id: "n", type: "slider", label: "Principal quantum number $n$", min: 1, max: 6, step: 1, value: 3 },
      { id: "l", type: "slider", label: "Angular momentum $l$ &nbsp;($0 \\le l \\lt  n$)", min: 0, max: 2, step: 1, value: 1, visibleIf: (p) => p.n > 1 },
      { id: "lInfo", type: "info", html: "Angular momentum: $l = 0$ — the only value allowed for $n=1$.", visibleIf: (p) => p.n <= 1 },
      { id: "m", type: "slider", label: "Magnetic quantum number $m$ &nbsp;($-l \\le m \\le l$)", min: -1, max: 1, step: 1, value: 0, visibleIf: (p) => p.n > 1 && p.l > 0 },
      { id: "mInfo", type: "info", html: "Magnetic quantum number: $m = 0$ — the only value allowed for $l=0$.", visibleIf: (p) => !(p.n > 1 && p.l > 0) },
      { id: "kind", type: "select", label: "Angular part", value: "complex",
        options: [{ value: "complex", label: "Complex Y_l^m (L_z eigenstate)" }, { value: "real", label: "Real orbital (pₓ, d(xy), …)" }] },
      { type: "section", label: "Point cloud" },
      { id: "count", type: "slider", label: "Number of sample points", min: 2000, max: 40000, step: 1000, value: 12000 },
      { id: "color", type: "select", label: "Colouring", value: "density", live: true,
        options: [{ value: "density", label: "Probability density |ψ|² (log)" }, { value: "phase", label: "Sign / phase of ψ" }] },
      { id: "size", type: "slider", label: "Point size", min: 0.5, max: 2.5, step: 0.1, value: 1.1, unit: "px", live: true },
      { id: "alpha", type: "slider", label: "Opacity", min: 0.05, max: 1, step: 0.01, value: 0.35, live: true },
      { id: "glow", type: "checkbox", label: "Additive (glowing) blending", value: true, live: true },
      { id: "rotate", type: "checkbox", label: "Auto-rotate", value: true, live: true },
      { type: "section", label: "Cross-section" },
      { id: "plane", type: "select", label: "Cut plane", value: "xz", live: true,
        options: [{ value: "xz", label: "x–z plane (y = 0)" }, { value: "xy", label: "x–y plane (z = 0)" }, { value: "yz", label: "y–z plane (x = 0)" }] },
      { id: "scale", type: "select", label: "Colour scale", value: "log", live: true,
        options: [{ value: "linear", label: "Linear" }, { value: "sqrt", label: "Square root" }, { value: "log", label: "Logarithmic (4 decades)" }] },
      { id: "nodes", type: "checkbox", label: "Show radial node spheres", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A single electron bound to a point nucleus of charge $+Ze$ ($Z=1$, hydrogen) by the Coulomb force; the nucleus is treated as infinitely heavy and
      spin and relativistic corrections are ignored. Lengths are measured in Bohr radii $a_0=4\\pi\\varepsilon_0\\hbar^2/me^2=0.529$ Å and energies in eV.
      The state is fixed by three quantum numbers: $n=1,2,\\dots$ (here up to 6), $l=0,\\dots,n-1$ and $m=-l,\\dots,l$. $\\theta$ is the polar angle measured from
      the $z$ axis, $\\phi$ the azimuth in the $xy$ plane.</p>

      <h4>Equations being solved</h4>
      <p>The time-independent Schrödinger equation $\\big[-\\tfrac{\\hbar^2}{2m}\\nabla^2-\\tfrac{Ze^2}{4\\pi\\varepsilon_0 r}\\big]\\psi=E\\psi$ separates in spherical
      coordinates. Writing $\\psi=R(r)Y(\\theta,\\phi)$, the angular part gives $\\hat L^2Y=\\hbar^2l(l+1)Y$, $\\hat L_zY=m\\hbar Y$ and the radial part the effective 1D problem
      $-\\tfrac{\\hbar^2}{2m}u''+\\big[-\\tfrac{Ze^2}{4\\pi\\varepsilon_0 r}+\\tfrac{\\hbar^2l(l+1)}{2mr^2}\\big]u=Eu$ for $u=rR$. Normalisable solutions exist only for</p>
      <div class="callout">
      $$\\psi_{nlm}(r,\\theta,\\phi)=R_{nl}(r)\\,Y_l^m(\\theta,\\phi),\\qquad E_n=-\\frac{13.6\\ \\text{eV}\\,Z^2}{n^2}$$
      </div>
      $$R_{nl}(r)=\\sqrt{\\left(\\frac{2Z}{na_0}\\right)^3\\frac{(n-l-1)!}{2n\\,(n+l)!}}\\;e^{-\\rho/2}\\rho^{\\,l}\\,L_{n-l-1}^{2l+1}(\\rho),\\qquad \\rho=\\frac{2Zr}{na_0}$$
      $$Y_l^m(\\theta,\\phi)=\\sqrt{\\frac{2l+1}{4\\pi}\\frac{(l-|m|)!}{(l+|m|)!}}\\;P_l^{|m|}(\\cos\\theta)\\,e^{im\\phi},\\qquad Y_l^{-|m|}=(-1)^{|m|}\\,\\overline{Y_l^{|m|}}$$
      <p>$L_k^{\\alpha}$ are generalised Laguerre polynomials and $P_l^{m}$ associated Legendre functions (Condon–Shortley phase included). The energy depends on
      $n$ only, so each level is $n^2$-fold degenerate (without spin). The complex $Y_l^m$ are eigenstates of $\\hat L_z$ and give densities symmetric about the
      $z$ axis; the <em>real</em> orbitals $\\sqrt2\\,(-1)^m\\,\\mathrm{Re}/\\mathrm{Im}\\,Y_l^{|m|}$ (the $p_x$, $d_{xy}$, … of chemistry) are equally valid energy eigenstates and show
      lobes. $R_{nl}$ has $n-l-1$ radial nodes, the angular part $l$ nodal planes or cones, and $\\langle r\\rangle=\\tfrac12[3n^2-l(l+1)]\\,a_0/Z$.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Radial table.</b> $R_{nl}$ is evaluated on 3000 points in $0\\le r\\le4n(n+l+2)\\,a_0$ with the Laguerre three-term recurrence. The radial probability
        $P(r)=r^2R_{nl}^2$ is integrated by the trapezoidal rule into a cumulative distribution $F(r)$ (normalised to 1).</li>
        <li><b>Sampling the cloud.</b> Points are drawn from $|\\psi|^2\\,dV=r^2R^2\\,dr\\;|Y|^2\\,d\\cos\\theta\\,d\\phi$: $r$ by inverse-transform sampling of $F$
        (binary search + linear interpolation), the angles by rejection sampling of $|Y|^2$ with $\\cos\\theta$ and $\\phi$ uniform — uniform $\\cos\\theta$ already includes the
        $\\sin\\theta$ of the volume element. A fixed seed makes the cloud reproducible. Up to 40 000 points.</li>
        <li><b>Rendering.</b> Each frame the points are rotated and projected orthographically and "splatted" straight into a pixel buffer, either additively with
        tone mapping (glow) or depth-sorted in four depth layers with alpha blending. Colour encodes $\\log_{10}|\\psi|^2$ (3 decades) or the sign/phase of $\\psi$
        ($\\pm$ for real orbitals, $\\arg\\psi=m\\phi$ as hue for complex ones).</li>
        <li><b>Cross-section.</b> $|\\psi|^2$ is computed exactly on a $181\\times181$ grid in the chosen plane, normalised to its maximum and shown on a linear, square-root
        or 4-decade logarithmic scale. If the plane lies on an angular nodal surface (e.g. $z=0$ for $2p_z$) this is detected and reported.</li>
        <li><b>Metrics.</b> $E_n=-13.6057/n^2$ eV; the most probable radius $r_{mp}$ is the maximum of $r^2R^2$ on the grid; $\\langle r\\rangle$ uses the analytic formula;
        radial nodes are sign changes of $R$ located by linear interpolation, which must equal $n-l-1$ — a built-in correctness check.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>1s</b> ($n=1$): a spherical cloud; $r_{mp}=1\\,a_0$ (the Bohr radius) while $\\langle r\\rangle=1.5\\,a_0$, because the distribution has a long tail.</li>
        <li><b>2s and 3s</b>: one radial node at $r=2a_0$ for 2s; two for 3s at $r=(9\\pm3\\sqrt3)/2\\,a_0\\approx1.90$ and $7.10\\,a_0$ (dashed circles and lines).</li>
        <li><b>2p, m = 0</b> in the $x$–$y$ cut plane: the plane is the nodal plane of $2p_z$ and the map reports $\\psi\\equiv0$. Switch to the $x$–$z$ plane to see the two lobes.</li>
        <li><b>Complex vs real.</b> $n=3$, $l=2$, $m=2$ complex gives a doughnut around the $z$ axis; colour by phase and the hue winds twice ($m=2$) around it. Switch to "real"
        and the same state becomes the four-lobed $d_{x^2-y^2}$.</li>
        <li><b>Circular orbits.</b> For $l=n-1$ there are no radial nodes and $r_{mp}=n^2a_0$ (the Bohr-model radius): 9 $a_0$ for 3d, 36 $a_0$ for 6h.</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>Non-relativistic, spinless, infinitely heavy nucleus (the reduced-mass correction is 0.05 %); the point cloud is a Monte-Carlo sample, so its graininess is
      statistical. See D. J. Griffiths &amp; D. F. Schroeter, <em>Introduction to Quantum Mechanics</em>, §4.2; B. H. Bransden &amp; C. J. Joachain,
      <em>Physics of Atoms and Molecules</em>, ch. 3; L. D. Landau &amp; E. M. Lifshitz, <em>Quantum Mechanics</em>, §36.</p>`,

    mount(api) {
      const P = api.params, Z = 1;
      const M = api.metrics([
        { id: "orb", label: "State" },
        { id: "E", label: "Energy $E_n$" },
        { id: "rmp", label: "Most probable radius $r_{mp}$" },
        { id: "rav", label: "Mean radius $\\langle r\\rangle$" },
        { id: "nodes", label: "Nodes (radial / angular)" },
      ]);
      const plots = api.plots([
        { id: "v3", type: "3d", title: "Electron cloud |ψ_nlm|² <span style='color:var(--muted);font-weight:400'>(drag to rotate)</span>", aspect: 0.9, maxHeight: 600, yaw: 0.8, pitch: 0.35 },
        { id: "cut", title: "Planar cross-section |ψ|²", aspect: 0.9, maxHeight: 600, equal: true, colorbar: true, xlabel: "x (a₀)", ylabel: "z (a₀)" },
        { id: "rad", title: "Radial probability distribution r²R²ₙₗ(r)", span: 2, aspect: 0.26, minHeight: 200, xlabel: "r (a₀)", ylabel: "r²R²" },
      ]);
      const view = plots.v3, pc = plots.cut;
      let dirty2d = true;
      plots.cut.onResize = () => { dirty2d = true; api.invalidate(); };
      plots.rad.onResize = () => { dirty2d = true; api.invalidate(); };
      let dragging = false;
      const onDown = () => { dragging = true; }, onUp = () => { dragging = false; };
      view.canvas.addEventListener("pointerdown", onDown);
      window.addEventListener("pointerup", onUp);

      const K = 24, NMAX = 40000, NG = 181;
      const C = {
        n: 0, x: new Float32Array(NMAX), y: new Float32Array(NMAX), z: new Float32Array(NMAX),
        rho: new Float32Array(NMAX), phase: new Float32Array(NMAX), ci: new Uint8Array(NMAX),
        px: new Float32Array(NMAX), py: new Float32Array(NMAX), key: new Uint16Array(NMAX),
        order: new Uint32Array(NMAX), cnt: new Uint32Array(D * K + 1), start: new Uint32Array(D * K + 1),
      };
      const cutRaw = new Float64Array(NG * NG), cutZ = new Float64Array(NG * NG);
      let rad, ang, ext = 5, cutExt = 5, pal = new Float32Array(D * K * 3), legend = [], name = "", real = false, qn = [1, 0, 0];
      let cutMin = 0, cutMax = 1, cutLabel = "", cutNodal = false;

      function clampQN() {
        const n = P.n;
        if (P.l > n - 1) api.setControl("l", { max: Math.max(1, n - 1), value: n - 1 });
        else api.setControl("l", { max: Math.max(1, n - 1) });
        P.l = Math.min(P.l, n - 1);
        const l = P.l, m = l === 0 ? 0 : PM.clamp(P.m, -l, l);
        if (l === 0) api.setControl("m", { min: -1, max: 1, value: 0 });
        else api.setControl("m", { min: -l, max: l, value: m });
        P.m = m;
      }

      function buildPalette() {
        legend = [];
        const base = [];
        if (P.color === "density") {
          for (let k = 0; k < K; k++) base.push(parseRGB(colormap("viridis", 0.12 + 0.88 * (k / (K - 1)))));
          let rmax = 0; for (let i = 0; i < C.n; i++) if (C.rho[i] > rmax) rmax = C.rho[i];
          for (let i = 0; i < C.n; i++) {
            const t = PM.clamp(1 + Math.log10(C.rho[i] / rmax + 1e-30) / 3, 0, 0.9999);
            C.ci[i] = (t * K) | 0;
          }
          legend = [{ label: "high |ψ|²", color: `rgb(${base[K - 1].join(",")})` }, { label: "low |ψ|²", color: `rgb(${base[4].join(",")})` }];
        } else if (real || qn[2] === 0) {
          const plus = [88, 166, 255], minus = [248, 110, 73];
          for (let k = 0; k < K; k++) base.push(k < K / 2 ? plus : minus);
          for (let i = 0; i < C.n; i++) C.ci[i] = Math.abs(C.phase[i] - Math.PI) < 1 ? K - 1 : 0;
          legend = [{ label: "ψ > 0", color: "rgb(88,166,255)" }, { label: "ψ < 0", color: "rgb(248,110,73)" }];
        } else {
          for (let k = 0; k < K; k++) base.push(hsl2rgb((360 * k) / K, 0.85, 0.6));
          for (let i = 0; i < C.n; i++) C.ci[i] = Math.min(K - 1, Math.floor((C.phase[i] / (2 * Math.PI)) * K));
          legend = [{ label: "hue = phase arg ψ = mφ", color: "rgb(240,120,120)" }];
        }
        for (let d = 0; d < D; d++) {
          const sh = (0.42 + 0.58 * ((d + 0.5) / D)) / 255;
          for (let k = 0; k < K; k++) { const b = base[k], o = (d * K + k) * 3; pal[o] = b[0] * sh; pal[o + 1] = b[1] * sh; pal[o + 2] = b[2] * sh; }
        }
      }

      /** |ψ|² grid in the selected plane. */
      function computeCut() {
        const E = cutExt, out = [0, 0];
        let mx = 0;
        for (let j = 0; j < NG; j++) {
          const w = -E + (2 * E * j) / (NG - 1);
          for (let i = 0; i < NG; i++) {
            const u = -E + (2 * E * i) / (NG - 1);
            let x, y, z;
            if (P.plane === "xz") { x = u; y = 0; z = w; } else if (P.plane === "xy") { x = u; y = w; z = 0; } else { x = 0; y = u; z = w; }
            const r = Math.sqrt(x * x + y * y + z * z);
            const c = r > 1e-12 ? z / r : 1, phi = Math.atan2(y, x);
            const Rv = r < rad.r[rad.NR - 1] ? rad.Rat(r) : 0;
            ang.eval(c, phi, out);
            const v = Rv * Rv * out[0] * out[0];
            cutRaw[j * NG + i] = v; if (v > mx) mx = v;
          }
        }
        // does the plane lie entirely on an angular nodal surface? (e.g. z=0 for 2p_z)
        let ref = 0;
        for (let i = 0; i < rad.NR; i += 4) { const v = rad.R[i] * rad.R[i]; if (v > ref) ref = v; }
        cutNodal = mx <= 1e-9 * ref * ang.ymax;
        const sc = P.scale;
        for (let k = 0; k < NG * NG; k++) {
          const t = mx > 0 ? cutRaw[k] / mx : 0;
          cutZ[k] = sc === "sqrt" ? Math.sqrt(t) : sc === "log" ? Math.max(-4, Math.log10(t + 1e-300)) : t;
        }
        cutMin = sc === "log" ? -4 : 0; cutMax = sc === "log" ? 0 : 1;
        cutLabel = sc === "sqrt" ? "√(|ψ|²/max)" : sc === "log" ? "log₁₀(|ψ|²/max)" : "|ψ|² / max";
        const lab = { xz: ["x (a₀)", "z (a₀)"], xy: ["x (a₀)", "y (a₀)"], yz: ["y (a₀)", "z (a₀)"] }[P.plane];
        pc.setLabels(lab[0], lab[1]);
        pc.setLimits([-E, E], [-E, E]);
        return mx;
      }

      function compute() {
        real = P.kind === "real";
        const n = P.n, l = Math.min(P.l, n - 1), m = PM.clamp(P.m, -l, l);
        qn = [n, l, m];
        rad = makeRadial(n, l, Z);
        ang = makeAngular(l, m, real);
        sampleCloud(rad, ang, Math.min(P.count, NMAX), 4321 + n * 97 + l * 13 + m, C);
        ext = rad.quant(0.97) * 1.05;
        cutExt = ext;
        view.opts.extent = ext;
        name = orbitalName(n, l, m, real);
        buildPalette();
        computeCut();
        const rp = rad.quant(0.9995) * 1.05;
        plots.rad.setLimits([0, rp], [0, rad.pmax * 1.18]);
        const rav = (3 * n * n - l * (l + 1)) / 2;
        M.set("orb", name);
        M.set("E", PM.fmt(-13.6057 / (n * n), 3) + " eV");
        M.set("rmp", PM.fmt(rad.rmp, 3) + " a₀");
        M.set("rav", PM.fmt(rav, 2) + " a₀");
        M.set("nodes", `${n - l - 1} / ${l}`);
      }

      function planeSquare() {
        const e = ext, pts = [[-e, -e], [e, -e], [e, e], [-e, e], [-e, -e]];
        const xs = [], ys = [], zs = [];
        for (const [u, w] of pts) {
          if (P.plane === "xz") { xs.push(u); ys.push(0); zs.push(w); }
          else if (P.plane === "xy") { xs.push(u); ys.push(w); zs.push(0); }
          else { xs.push(0); ys.push(u); zs.push(w); }
        }
        return [xs, ys, zs];
      }

      return {
        reset() { clampQN(); compute(); dirty2d = true; },
        onParam(id) {
          dirty2d = true;
          if (id === "n" || id === "l") clampQN();
          if (id === "color") buildPalette();
          if (id === "plane" || id === "scale") computeCut();
        },
        step(dt) { if (P.rotate && !dragging) view.yaw += 0.35 * dt; },
        render() {
          view.clear();
          drawCloud(view, C, pal, K, P.size, P.alpha, P.glow, ext);
          view.ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
          view.box(ext, { color: "rgba(139,152,168,0.16)" });
          const [sx, sy, sz] = planeSquare();
          view.line3(sx, sy, sz, { color: PlotColors.accent3, width: 1, alpha: 0.45, dash: [5, 4] });
          view.axes(ext * 0.9);
          view.text(name, 12, 22, { size: 15 });
          view.text(`${C.n.toLocaleString("en-US")} points · box ±${PM.fmt(ext, 1)} a₀`, 12, 40, { size: 11.5, color: PlotColors.muted });
          const c = view.ctx;
          legend.forEach((it, k) => {
            const y = view.H - 14 - (legend.length - 1 - k) * 17;
            c.fillStyle = it.color; c.beginPath(); c.arc(18, y - 4, 4.5, 0, 2 * Math.PI); c.fill();
            view.text(it.label, 30, y, { size: 11.5 });
          });
          view.text("cut plane", view.W - 12, view.H - 14, { size: 11, color: PlotColors.accent3, align: "right" });

          if (dirty2d) { // the 2D plots are redrawn only when something changed
            dirty2d = false;
          hardClear(pc);
          pc.heatmap(cutZ, NG, NG, { x0: -cutExt, x1: cutExt, y0: -cutExt, y1: cutExt, vmin: cutMin, vmax: cutMax, cmap: "inferno", colorbar: cutLabel });
          if (P.nodes && rad.nodes.length) {
            pc.custom((ctx, p) => {
              ctx.strokeStyle = "rgba(79,209,197,0.85)"; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
              for (const rn of rad.nodes) { ctx.beginPath(); ctx.arc(p.X(0), p.Y(0), rn * p.sx, 0, 2 * Math.PI); ctx.stroke(); }
              ctx.setLineDash([]);
            });
          }
          if (cutNodal) pc.text(0, 0, "This plane is an angular nodal plane: ψ ≡ 0", { align: "center", bg: "#0f151c", size: 12.5, color: PlotColors.accent3 });
          pc.label(name + "  ·  " + { xz: "y = 0", xy: "z = 0", yz: "x = 0" }[P.plane], "tl", { size: 11.5 });

          const pr = plots.rad;
          hardClear(pr);
          pr.fill(rad.r, rad.p, 0, { color: PlotColors.accent3, alpha: 0.22 });
          pr.line(rad.r, rad.p, { color: PlotColors.accent3, width: 2 });
          for (const rn of rad.nodes) pr.vline(rn, { color: PlotColors.accent, dash: [5, 4], width: 1.1 });
          pr.vline(rad.rmp, { color: PlotColors.text, dash: [6, 4], width: 1.2, alpha: 0.7 });
          const rav = (3 * qn[0] * qn[0] - qn[1] * (qn[1] + 1)) / 2;
          pr.vline(rav, { color: PlotColors.accent2, dash: [3, 3], width: 1.3 });
          pr.label([`r_mp = ${PM.fmt(rad.rmp, 2)} a₀`, `⟨r⟩ = ${PM.fmt(rav, 2)} a₀`], "tl", { size: 11.5 });
          pr.legend([{ label: "r²R²ₙₗ", color: PlotColors.accent3 }, { label: "most probable r", color: PlotColors.text, dash: [6, 4] }, { label: "⟨r⟩", color: PlotColors.accent2, dash: [3, 3] }, { label: "radial node", color: PlotColors.accent, dash: [5, 4] }]);
          }
        },
        destroy() { view.canvas.removeEventListener("pointerdown", onDown); window.removeEventListener("pointerup", onUp); },
      };
    },
  });
})();
