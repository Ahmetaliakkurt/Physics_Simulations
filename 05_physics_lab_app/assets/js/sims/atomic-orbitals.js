/* Atomic orbitals — hydrogen-like ψ_nlm = R_nl(r)·Y_l^m(θ,φ) with an effective nuclear charge Z_eff.
 * Exact sampling of |ψ|² (radial part: inverse CDF, angular part: acceptance–rejection) and a
 * live-rotating 3D point cloud. */
(function () {
  "use strict";

  // Effective nuclear charges (one value per element: approximately the valence-electron Z_eff)
  const ATOM_DATA = {
    H: 1.0, He: 1.69, Li: 1.28, Be: 1.91, B: 2.42, C: 3.14, N: 3.83, O: 4.45, F: 5.10, Ne: 5.85,
    Na: 2.51, Mg: 3.31, Al: 4.07, Si: 4.29, P: 4.89, S: 5.48, Cl: 6.12, Ar: 6.76,
    K: 3.50, Ca: 4.40, Sc: 4.63, Ti: 4.82, V: 5.12, Cr: 5.13, Mn: 5.43, Fe: 5.73,
    Co: 6.03, Ni: 6.33, Cu: 6.63, Zn: 6.93, Br: 9.03, Kr: 9.73, Ag: 10.5, Au: 12.0,
  };
  // Atomic numbers of the tabulated elements (for Slater's rules)
  const ZNUC = {
    H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10, Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
    K: 19, Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26, Co: 27, Ni: 28, Cu: 29, Zn: 30, Br: 35, Kr: 36, Ag: 47, Au: 79,
  };
  // Madelung (n + l, then n) filling order
  const MADELUNG = [[1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [4, 0], [3, 2], [4, 1], [5, 0], [4, 2], [5, 1], [6, 0], [4, 3], [5, 2], [6, 1], [7, 0], [5, 3], [6, 2], [7, 1]];
  // Ground-state exceptions among the tabulated elements: one s electron moves into the d shell
  const EXCEPT = { Cr: [4, 3], Cu: [4, 3], Ag: [5, 4], Au: [6, 5] }; // [n of the s shell, n of the d shell]
  /** Ground-state configuration as {"n,l": occupancy}. */
  function groundConfig(sym) {
    let left = ZNUC[sym] || 1;
    const cfg = {};
    for (const [n, l] of MADELUNG) {
      if (left <= 0) break;
      const c = Math.min(left, 2 * (2 * l + 1));
      cfg[n + "," + l] = c; left -= c;
    }
    const ex = EXCEPT[sym];
    if (ex) { cfg[ex[0] + ",0"] -= 1; cfg[ex[1] + ",2"] = (cfg[ex[1] + ",2"] || 0) + 1; }
    return cfg;
  }
  /** Slater grouping key: (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)… → [n, 0] for s/p, [n, l − 1] for d, f, g … */
  const sgroup = (n, l) => (l <= 1 ? n * 10 : n * 10 + l - 1);
  /**
   * Slater's-rule effective charge felt by one electron in the (n, l) sub-shell.
   * If (n, l) is empty in the ground state, the outermost electron is promoted into it (an excited configuration).
   */
  function slaterZeff(sym, n, l) {
    const Z = ZNUC[sym] || 1, cfg = groundConfig(sym), key = n + "," + l;
    if (cfg[key] > 0) cfg[key] -= 1;
    else {
      let best = null, bn = -1, bl = -1;
      for (const k in cfg) { if (!(cfg[k] > 0)) continue; const [kn, kl] = k.split(",").map(Number); if (kn > bn || (kn === bn && kl > bl)) { bn = kn; bl = kl; best = k; } }
      if (best) cfg[best] -= 1;
    }
    const g = sgroup(n, l);
    let S = 0;
    for (const k in cfg) {
      const c = cfg[k]; if (!(c > 0)) continue;
      const [kn, kl] = k.split(",").map(Number), gk = sgroup(kn, kl);
      if (gk > g) continue;                                   // outer groups do not screen
      if (gk === g) S += c * (n === 1 ? 0.30 : 0.35);         // same group
      else if (l <= 1) S += c * (kn === n - 1 ? 0.85 : 1.0);  // s, p: shell n−1 → 0.85, deeper → 1.00
      else S += c;                                            // d, f, …: everything inside → 1.00
    }
    return Z - S;
  }
  const LETTERS = "spdfgh";
  /** Reset the context state before Plot.clear() (clears any clip stack left over from a resize). */
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
    return base + " (m = " + (m > 0 ? "+" + m : m < 0 ? "−" + -m : "0") + ")";
  }

  /** Angular part of the orbital: signed value (real) or amplitude + phase (complex). */
  function makeAngular(l, m, real) {
    const am = Math.abs(m), N = PM.sphHarmNorm(l, m);
    const cs = am % 2 ? -1 : 1; // (−1)^|m|
    // upper bound of |Y|²: max_c P² (×2 for real orbitals with m ≠ 0)
    let pmax = 0;
    for (let i = 0; i <= 2000; i++) { const p = PM.assocLegendre(l, am, -1 + i / 1000); if (p * p > pmax) pmax = p * p; }
    const ymax = N * N * pmax * (real && m !== 0 ? 2 : 1) * 1.01;
    return {
      ymax,
      /** returns the signed real-orbital value, or |Y| and the phase angle for complex Y: out = [value or |Y|, phase] */
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
   * pal: Float32Array(D*K*3) of depth-shaded RGB.
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
    // only the rectangle covered by the points is processed
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
    id: "atomic-orbitals",
    category: "statistical",
    order: 24,
    title: "Atomic Orbitals",
    icon: "🧿",
    subtitle: "Hydrogen-like orbitals in a central Coulomb field: a 3D cloud of points drawn from $|\\psi_{nlm}|^2$ for chosen quantum numbers $n,l,m$ and element, plus the radial probability distribution and radial wave function.",
    notes: [
      { type: "info", html: "The electron is assumed to move in a hydrogen-like potential $-Z_{eff}/r$, where $Z_{eff}$ is either a single tabulated value per element (the charge felt by the valence electron) or Slater's-rule value for the chosen $n,l$. Because screening in a real many-electron atom depends on the orbital, this is an approximation for orbital sizes. Drag the cloud to rotate it, use the mouse wheel to zoom." },
    ],
    animated: true,
    speed: { min: 0, max: 3, value: 1, step: 0.1 },
    controls: [
      { id: "atom", type: "select", label: "Element", value: "H",
        options: Object.keys(ATOM_DATA).map((k) => ({ value: k, label: `${k}  (Z = ${ZNUC[k]})  ·  Z_eff ${ATOM_DATA[k].toFixed(2)}` })) },
      { id: "zmode", type: "select", label: "Effective nuclear charge $Z_{eff}$", value: "table",
        options: [{ value: "table", label: "Tabulated (valence electron)" }, { value: "slater", label: "Slater's rules for this n, l" }],
        help: "Slater's rules screen the chosen electron by all the others; if the chosen sub-shell is empty in the ground state, the outermost electron is promoted into it." },
      { type: "section", label: "Quantum numbers" },
      { id: "n", type: "slider", label: "Principal quantum number $n$", min: 1, max: 6, step: 1, value: 3 },
      { id: "l", type: "slider", label: "Orbital angular momentum $l$ &nbsp;($0 \\le l \\le n-1$)", min: 0, max: 2, step: 1, value: 2, visibleIf: (p) => p.n > 1 },
      { id: "lInfo", type: "info", html: "Orbital angular momentum: $l = 0$ — the only possible value for $n=1$.", visibleIf: (p) => p.n <= 1 },
      { id: "m", type: "slider", label: "Magnetic quantum number $m$ &nbsp;($-l \\le m \\le l$)", min: -2, max: 2, step: 1, value: 1, visibleIf: (p) => p.n > 1 && p.l > 0 },
      { id: "mInfo", type: "info", html: "Magnetic quantum number: $m = 0$ — the only possible value for $l=0$.", visibleIf: (p) => !(p.n > 1 && p.l > 0) },
      { id: "kind", type: "select", label: "Angular part", value: "real",
        options: [{ value: "real", label: "Real orbitals (lobes: pₓ, d(xy), …)" }, { value: "complex", label: "Complex Yₗᵐ (rings about z)" }],
        help: "Real orbitals are linear combinations of $Y_l^{m}$ and $Y_l^{-m}$; they give the lobe shapes familiar from chemistry." },
      { type: "section", label: "Display" },
      { id: "color", type: "select", label: "Colouring", value: "phase", live: true,
        options: [{ value: "phase", label: "Sign / phase of ψ" }, { value: "density", label: "Density |ψ|² (log scale)" }] },
      { id: "count", type: "slider", label: "Number of sample points", min: 2000, max: 40000, step: 1000, value: 16000 },
      { id: "size", type: "slider", label: "Point size", min: 0.5, max: 2.5, step: 0.1, value: 1.1, unit: "px", live: true },
      { id: "alpha", type: "slider", label: "Opacity", min: 0.05, max: 1, step: 0.01, value: 0.4, live: true },
      { id: "glow", type: "checkbox", label: "Additive (glowing) blending", value: true, live: true },
      { id: "rotate", type: "checkbox", label: "Auto-rotate", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>One electron of mass $m_e$ moves in the spherically symmetric Coulomb field of an effective nuclear charge $Z_{eff}e$:</p>
      $$V(r)=-\\frac{Z_{eff}\\,e^2}{4\\pi\\varepsilon_0\\,r}.$$
      <p>For hydrogen ($Z_{eff}=1$) and hydrogen-like ions this is exact (apart from fine structure). In a many-electron atom the
      other electrons partly <b>screen</b> the nucleus of charge $Ze$; replacing their effect by a reduced charge
      $Z_{eff}=Z-S$ (central-field / independent-particle approximation) turns the problem back into a hydrogen-like one.
      Lengths are in Bohr radii $a_0=4\\pi\\varepsilon_0\\hbar^2/(m_ee^2)=0.529$ Å, energies in Rydbergs (13.6 eV). The parameters are
      the element, the choice of $Z_{eff}$, the quantum numbers $n=1\\dots6$, $0\\le l\\le n-1$, $-l\\le m\\le l$, and whether the complex
      eigenfunctions or the real (chemists') orbitals are shown.</p>

      <h4>Equations being solved</h4>
      <p>The time-independent Schrödinger equation $\\big[-\\tfrac{\\hbar^2}{2m_e}\\nabla^2+V(r)\\big]\\psi=E\\psi$ separates in spherical
      coordinates ($\\theta$: polar angle from the $z$ axis, $\\phi$: azimuth in the $xy$ plane), and its bound states are</p>
      <div class="callout">$$\\psi_{nlm}(r,\\theta,\\phi)=R_{nl}(r)\\,Y_l^m(\\theta,\\phi),\\qquad E_n=-\\frac{Z_{eff}^2}{n^2}\\,13.6\\ \\text{eV}.$$</div>
      <p>The radial function involves the generalised Laguerre polynomials ($a_0=1$):</p>
      $$R_{nl}(r)=\\sqrt{\\Big(\\frac{2Z_{eff}}{n}\\Big)^3\\frac{(n-l-1)!}{2n\\,(n+l)!}}\\;e^{-\\rho/2}\\rho^{\\,l}L_{n-l-1}^{2l+1}(\\rho),\\qquad \\rho=\\frac{2Z_{eff}r}{n},$$
      <p>normalised so that $\\int_0^\\infty R_{nl}^2r^2dr=1$. It has $n-l-1$ radial nodes. The angular part is a spherical harmonic,</p>
      $$Y_l^m(\\theta,\\phi)=\\sqrt{\\frac{2l+1}{4\\pi}\\frac{(l-m)!}{(l+m)!}}\\;P_l^{m}(\\cos\\theta)\\,e^{im\\phi}\\ \\ (m\\ge0),\\qquad
        Y_l^{-m}=(-1)^m\\,\\overline{Y_l^{m}},$$
      <p>where $P_l^m(x)=(-1)^m(1-x^2)^{m/2}\\,d^mP_l/dx^m$ is the associated Legendre function including the Condon–Shortley phase. $|Y_l^m|^2$ does not depend on $\\phi$, so the complex
      orbitals are symmetric about the $z$ axis. The lobed <b>real orbitals</b> of chemistry are the combinations</p>
      $$Y_{lm}^{\\text{real}}=\\begin{cases}\\sqrt2\\,(-1)^m\\,\\mathrm{Re}\\,Y_l^{|m|}\\propto P_l^{|m|}(\\cos\\theta)\\cos(m\\phi), & m&gt;0\\\\ Y_l^0, & m=0\\\\ \\sqrt2\\,(-1)^m\\,\\mathrm{Im}\\,Y_l^{|m|}\\propto P_l^{|m|}(\\cos\\theta)\\sin(|m|\\phi), & m&lt;0\\end{cases}$$
      <p>which have $l$ angular nodal surfaces. Useful radial measures are the most probable radius $r_{mp}$ (maximum of
      $P(r)=r^2R_{nl}^2$) and the mean radius</p>
      $$\\langle r\\rangle=\\frac{3n^2-l(l+1)}{2Z_{eff}}\\,a_0,$$
      <p>so orbitals swell like $n^2$ and shrink like $1/Z_{eff}$: changing $Z_{eff}$ only rescales the picture, $r\\to r/Z_{eff}$.</p>
      <p><b>Effective charge.</b> Two options are offered. <i>Tabulated</i>: one value per element, approximately the
      Clementi–Raimondi effective charge felt by the outermost (valence) electron. <i>Slater's rules</i>: the electrons are grouped as
      (1s)(2s,2p)(3s,3p)(3d)(4s,4p)(4d)(4f)(5s,5p)…, the ground-state configuration is built by the Madelung rule (with the
      exceptions Cr, Cu, Ag, Au), and the screening of the chosen $n,l$ electron is $S=\\sum(\\text{coefficient})\\times(\\text{number of
      electrons})$:</p>
      <table>
        <tr><th>other electron in…</th><th>chosen electron is $ns$ or $np$</th><th>chosen electron is $nd$ or $nf$</th></tr>
        <tr><td>the same group</td><td>0.35 (0.30 within 1s)</td><td>0.35</td></tr>
        <tr><td>shell $n-1$</td><td>0.85</td><td>1.00</td></tr>
        <tr><td>shells $n-2$ and below</td><td>1.00</td><td>1.00</td></tr>
        <tr><td>groups to the right (outer)</td><td>0</td><td>0</td></tr>
      </table>
      <p>Examples: C 2p: $Z_{eff}=6-(3\\times0.35+2\\times0.85)=3.25$; Na 3s: $11-(8\\times0.85+2)=2.20$; Fe 3d: $26-(5\\times0.35+18)=6.25$.
      If the chosen sub-shell is empty in the ground state (e.g. H 3d), the outermost electron is promoted into it; an excited
      electron far outside the core sees $Z_{eff}\\approx1$ (Rydberg states).</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Radial table:</b> $R_{nl}$ is evaluated (Laguerre three-term recurrence) on 3000 points of $[0,\\,4n(n+l+2)/Z_{eff}]$, which
        contains essentially all of the probability. $P(r)=r^2R^2$ is integrated with the trapezoidal rule to a normalised cumulative
        distribution $F(r)$; $r_{mp}$ is the grid maximum of $P$; nodes are located by sign changes with linear interpolation.</li>
        <li><b>Sampling the cloud:</b> since $|\\psi|^2dV=\\big[R_{nl}^2r^2dr\\big]\\big[|Y|^2d\\Omega\\big]$ factorises, $r$ and the
        direction are drawn independently and <em>exactly</em>. The radius uses inverse-transform sampling, $r=F^{-1}(u)$ with
        $u\\sim U(0,1)$ (binary search + linear interpolation). The direction uses acceptance–rejection: $\\cos\\theta\\sim U(-1,1)$ and
        $\\phi\\sim U(0,2\\pi)$ are uniform on the sphere ($d\\Omega=d\\cos\\theta\\,d\\phi$) and a candidate is kept with probability
        $|Y(\\theta,\\phi)|^2/\\max|Y|^2$. The local density of points is therefore proportional to the probability of finding the
        electron there. A fixed seed per $(n,l,m)$ makes the cloud reproducible.</li>
        <li><b>Colour:</b> either the sign of the real wave function (blue $\\psi&gt;0$, red $\\psi&lt;0$, including the sign of $R_{nl}$, so
        radial nodes show as colour changes), the phase $\\arg\\psi=m\\phi$ on a colour wheel for complex orbitals, or
        $\\log_{10}|\\psi|^2$ over three decades.</li>
        <li><b>Rendering:</b> up to 40 000 points are projected orthographically and splatted directly into a pixel buffer with four
        depth-shading layers, using additive tone-mapped blending ("glow") or depth-sorted alpha blending. The view box half-width is
        the radius containing 97% of the probability.</li>
        <li><b>Metrics:</b> the orbital name, the $Z_{eff}$ used, $r_{mp}$, $\\langle r\\rangle$ from the formula above, and the number of
        radial / angular nodes $n-l-1$ / $l$.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>H 1s:</b> $r_{mp}=1\\,a_0$ (the Bohr radius) and $\\langle r\\rangle=1.5\\,a_0$ — the mean exceeds the peak because of the
        long exponential tail.</li>
        <li><b>H 3s, 3p, 3d:</b> 2, 1 and 0 radial nodes; the 3d radial distribution is a single peak at $r_{mp}=n^2=9\\,a_0$, as for every
        "circular" orbital $l=n-1$.</li>
        <li><b>Real vs. complex:</b> for $n=2$, $l=1$, $m=\\pm1$ the complex orbitals are identical rings around $z$ with opposite phase
        winding, while the real ones are the dumbbells $p_x$ and $p_y$.</li>
        <li><b>Screening:</b> choose Fe, $n=3$, $l=2$: the tabulated valence value (5.73) and Slater's 3d value (6.25) differ, and Slater's
        4s value is only 3.75 — inner and outer electrons see very different charges. Switch to Au 6s ($Z_{eff}=3.70$ by Slater).</li>
        <li><b>Rydberg orbital:</b> Na with Slater's rules, $n=3$, $l=2$: the promoted electron sees $Z_{eff}=1$ and the cloud is as large
        as hydrogen's 3d ($\\langle r\\rangle=10.5\\,a_0$).</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>A single $Z_{eff}$ cannot reproduce the true self-consistent (Hartree–Fock) radial functions: real outer orbitals have inner
      oscillations and tails governed by the ionisation energy, and spin–orbit coupling, relativistic effects (important for Au) and
      electron correlation are ignored. Slater's rules are empirical and rough to ~10%. References: D. J. Griffiths,
      <i>Introduction to Quantum Mechanics</i>, ch. 4; B. H. Bransden &amp; C. J. Joachain, <i>Physics of Atoms and Molecules</i>;
      J. C. Slater, Phys. Rev. <b>36</b>, 57 (1930); E. Clementi &amp; D. L. Raimondi, J. Chem. Phys. <b>38</b>, 2686 (1963);
      J. J. Sakurai, <i>Modern Quantum Mechanics</i>, ch. 3.</p>`,

    mount(api) {
      const P = api.params;
      const M = api.metrics([
        { id: "orb", label: "Orbital" },
        { id: "z", label: "Effective charge $Z_{eff}$" },
        { id: "rmp", label: "Most probable radius $r_{mp}$" },
        { id: "rav", label: "Mean radius $\\langle r\\rangle$" },
        { id: "nodes", label: "Nodes (radial / angular)" },
      ]);
      const plots = api.plots([
        { id: "v3", type: "3d", title: "3D probability cloud |ψ<sub>nlm</sub>|² &nbsp;<span style='color:var(--muted);font-weight:400'>(drag: rotate · wheel: zoom)</span>", span: 2, aspect: 0.58, maxHeight: 620, yaw: 0.8, pitch: 0.38 },
        { id: "rad", title: "Radial probability distribution r²R²ₙₗ(r)", aspect: 0.62, xlabel: "r (a₀)", ylabel: "r²R²" },
        { id: "rfun", title: "Radial wave function Rₙₗ(r)", aspect: 0.62, xlabel: "r (a₀)", ylabel: "Rₙₗ" },
      ]);
      const view = plots.v3;
      let dirty2d = true;
      plots.rad.onResize = () => { dirty2d = true; api.invalidate(); };
      plots.rfun.onResize = () => { dirty2d = true; api.invalidate(); };
      let dragging = false;
      const onDown = () => { dragging = true; }, onUp = () => { dragging = false; };
      view.canvas.addEventListener("pointerdown", onDown);
      window.addEventListener("pointerup", onUp);

      const K = 24, NMAX = 40000;
      const C = {
        n: 0, x: new Float32Array(NMAX), y: new Float32Array(NMAX), z: new Float32Array(NMAX),
        rho: new Float32Array(NMAX), phase: new Float32Array(NMAX), ci: new Uint8Array(NMAX),
        px: new Float32Array(NMAX), py: new Float32Array(NMAX), key: new Uint16Array(NMAX),
        order: new Uint32Array(NMAX), cnt: new Uint32Array(D * K + 1), start: new Uint32Array(D * K + 1),
      };
      let rad, ang, ext = 5, pal = new Float32Array(D * K * 3), legend = [], name = "", Z = 1, real = true;

      function clampQN() {
        const n = P.n;
        if (P.l > n - 1) api.setControl("l", { max: Math.max(1, n - 1), value: n - 1 });
        else api.setControl("l", { max: Math.max(1, n - 1) });
        P.l = Math.min(P.l, n - 1);
        const l = P.l;
        const m = l === 0 ? 0 : PM.clamp(P.m, -l, l);
        if (l === 0) api.setControl("m", { min: -1, max: 1, value: 0 });
        else api.setControl("m", { min: -l, max: l, value: m });
        P.m = m;
      }

      function buildPalette() {
        legend = [];
        let base = [];
        if (P.color === "density") {
          const cmaps = ["plasma", "viridis", "inferno", "turbo", "magma", "ice"];
          const cm = cmaps[P.l % cmaps.length];
          for (let k = 0; k < K; k++) base.push(parseRGB(colormap(cm, 0.18 + 0.82 * (k / (K - 1)))));
          let rmax = 0; for (let i = 0; i < C.n; i++) if (C.rho[i] > rmax) rmax = C.rho[i];
          for (let i = 0; i < C.n; i++) {
            const t = PM.clamp(1 + Math.log10(C.rho[i] / rmax + 1e-30) / 3, 0, 0.9999);
            C.ci[i] = (t * K) | 0;
          }
          legend = [{ label: "high |ψ|²", color: `rgb(${base[K - 1].join(",")})`, type: "dot" }, { label: "low |ψ|²", color: `rgb(${base[3].join(",")})`, type: "dot" }];
        } else if (real || P.m === 0) {
          // sign: + blue, − orange-red
          const plus = [88, 166, 255], minus = [248, 110, 73];
          for (let k = 0; k < K; k++) base.push(k < K / 2 ? plus : minus);
          for (let i = 0; i < C.n; i++) C.ci[i] = Math.abs(C.phase[i] - Math.PI) < 1 ? K - 1 : 0;
          legend = [{ label: "ψ > 0", color: "rgb(88,166,255)", type: "dot" }, { label: "ψ < 0", color: "rgb(248,110,73)", type: "dot" }];
        } else {
          for (let k = 0; k < K; k++) base.push(hsl2rgb((360 * k) / K, 0.85, 0.6));
          for (let i = 0; i < C.n; i++) C.ci[i] = Math.min(K - 1, Math.floor((C.phase[i] / (2 * Math.PI)) * K));
          legend = [{ label: "phase arg ψ = mφ (colour wheel)", color: "rgb(240,120,120)", type: "dot" }];
        }
        for (let d = 0; d < D; d++) {
          const sh = (0.42 + 0.58 * ((d + 0.5) / D)) / 255;
          for (let k = 0; k < K; k++) { const b = base[k], o = (d * K + k) * 3; pal[o] = b[0] * sh; pal[o + 1] = b[1] * sh; pal[o + 2] = b[2] * sh; }
        }
      }

      function compute() {
        real = P.kind === "real";
        const n = P.n, l = Math.min(P.l, n - 1), m = PM.clamp(P.m, -l, l);
        Z = P.zmode === "slater" ? slaterZeff(P.atom, n, l) : ATOM_DATA[P.atom] || 1;
        rad = makeRadial(n, l, Z);
        ang = makeAngular(l, m, real);
        sampleCloud(rad, ang, Math.min(P.count, NMAX), 1234 + n * 97 + l * 13 + m, C);
        ext = rad.quant(0.97) * 1.05;
        view.opts.extent = ext;
        name = orbitalName(n, l, m, real);
        buildPalette();
        // radial plot limits
        const rp = rad.quant(0.9995) * 1.05;
        plots.rad.setLimits([0, rp], [0, rad.pmax * 1.18]);
        let rmin = 0, rmx = 0;
        for (let i = 0; i < rad.NR; i++) if (rad.r[i] <= rp) { rmin = Math.min(rmin, rad.R[i]); rmx = Math.max(rmx, rad.R[i]); }
        const span = rmx - rmin;
        plots.rfun.setLimits([0, rp], [rmin < 0 ? rmin - 0.12 * span : -0.06 * span, rmx + 0.12 * span]);
        const rav = (3 * n * n - l * (l + 1)) / (2 * Z);
        M.set("orb", name);
        M.set("z", PM.fmt(Z, 2) + (P.zmode === "slater" ? " (Slater)" : " (table)"));
        M.set("rmp", PM.fmt(rad.rmp, 3) + " a₀");
        M.set("rav", PM.fmt(rav, 3) + " a₀");
        M.set("nodes", `${n - l - 1} / ${l}`);
      }

      return {
        reset() { clampQN(); compute(); dirty2d = true; },
        onParam(id) {
          dirty2d = true;
          if (id === "n" || id === "l") clampQN();
          if (id === "color") buildPalette();
        },
        step(dt) { if (P.rotate && !dragging) view.yaw += 0.35 * dt; },
        render() {
          view.clear();
          drawCloud(view, C, pal, K, P.size, P.alpha, P.glow, ext);
          view.ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
          view.box(ext, { color: "rgba(139,152,168,0.16)" });
          view.axes(ext * 0.9);
          const o = view.project(0, 0, 0);
          view.ctx.fillStyle = "#ffffff"; view.ctx.beginPath(); view.ctx.arc(o[0], o[1], 2.2, 0, 2 * Math.PI); view.ctx.fill();
          view.text(`${name}  ·  ${P.atom} (Z_eff = ${PM.fmt(Z, 2)})`, 12, 22, { size: 14 });
          view.text(`box half-width ${PM.fmt(ext, 1)} a₀  ·  ${C.n.toLocaleString("en-US")} points`, 12, 40, { size: 11.5, color: PlotColors.muted });
          const c = view.ctx;
          legend.forEach((it, k) => {
            const y = view.H - 14 - (legend.length - 1 - k) * 17;
            c.fillStyle = it.color; c.beginPath(); c.arc(18, y - 4, 4.5, 0, 2 * Math.PI); c.fill();
            view.text(it.label, 30, y, { size: 11.5 });
          });

          if (dirty2d) { // the 2D plots are redrawn only when they change
            dirty2d = false;
          // radial probability
          const pr = plots.rad;
          hardClear(pr);
          pr.fill(rad.r, rad.p, 0, { color: PlotColors.accent, alpha: 0.2 });
          pr.line(rad.r, rad.p, { color: PlotColors.accent, width: 2 });
          for (const rn of rad.nodes) pr.vline(rn, { color: PlotColors.muted, dash: [2, 4], width: 1 });
          pr.vline(rad.rmp, { color: PlotColors.text, dash: [6, 4], width: 1.2, alpha: 0.7 });
          const rav = (3 * P.n * P.n - P.l * (P.l + 1)) / (2 * Z);
          pr.vline(rav, { color: PlotColors.accent3, dash: [3, 3], width: 1.2, alpha: 0.8 });
          pr.label([`r_mp = ${PM.fmt(rad.rmp, 2)} a₀`, `⟨r⟩ = ${PM.fmt(rav, 2)} a₀`], "tl", { size: 11.5 });
          pr.legend([{ label: "r²R²ₙₗ", color: PlotColors.accent }, { label: "r_mp", color: PlotColors.text, dash: [6, 4] }, { label: "⟨r⟩", color: PlotColors.accent3, dash: [3, 3] }]);

          const pf = plots.rfun;
          hardClear(pf);
          pf.hline(0, { color: PlotColors.muted, width: 1 });
          pf.line(rad.r, rad.R, { color: PlotColors.accent2, width: 2 });
          for (const rn of rad.nodes) { pf.vline(rn, { color: PlotColors.muted, dash: [2, 4], width: 1 }); pf.circle(rn, 0, 3.5, { px: true, color: PlotColors.accent3 }); }
          pf.label(rad.nodes.length ? `${rad.nodes.length} radial node${rad.nodes.length > 1 ? "s" : ""}` : "no radial nodes", "tr");
          }
        },
        destroy() { view.canvas.removeEventListener("pointerdown", onDown); window.removeEventListener("pointerup", onUp); },
      };
    },
  });
})();
