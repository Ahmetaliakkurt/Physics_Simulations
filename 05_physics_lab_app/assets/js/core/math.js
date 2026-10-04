/* =====================================================================
 * PM — numerical core of the Physics Simulation Lab (runs in the browser)
 * All arrays are Float64Array (or Float32Array).
 * 2D arrays are row-major: index = j*nx + i  (i: x, j: y)
 * ===================================================================== */
(function () {
  "use strict";
  const PM = {};

  // ----------------------------------------------------------- basics
  PM.linspace = function (a, b, n) {
    const out = new Float64Array(n);
    if (n === 1) { out[0] = a; return out; }
    const h = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) out[i] = a + i * h;
    return out;
  };
  PM.zeros = (n) => new Float64Array(n);
  PM.clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  PM.lerp = (a, b, t) => a + (b - a) * t;
  PM.mod = (a, n) => ((a % n) + n) % n;
  PM.sum = function (a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; };
  PM.mean = (a) => PM.sum(a) / a.length;
  PM.std = function (a) {
    const m = PM.mean(a); let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / a.length);
  };
  PM.max = function (a) { let m = -Infinity; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; };
  PM.min = function (a) { let m = Infinity; for (let i = 0; i < a.length; i++) if (a[i] < m) m = a[i]; return m; };
  /** Trapezoidal-rule integral (uniform spacing dx). */
  PM.trapz = function (y, dx) {
    let s = 0; for (let i = 1; i < y.length; i++) s += 0.5 * (y[i] + y[i - 1]);
    return s * dx;
  };

  /** Histogram: {centers, counts, width}. density=true normalises the area to 1. */
  PM.histogram = function (data, bins, lo, hi, density) {
    const counts = new Float64Array(bins);
    const w = (hi - lo) / bins;
    let n = 0;
    for (let i = 0; i < data.length; i++) {
      const k = Math.floor((data[i] - lo) / w);
      if (k >= 0 && k < bins) { counts[k]++; n++; }
      else if (data[i] === hi) { counts[bins - 1]++; n++; }
    }
    if (density && n > 0) for (let k = 0; k < bins; k++) counts[k] /= n * w;
    const centers = new Float64Array(bins);
    for (let k = 0; k < bins; k++) centers[k] = lo + (k + 0.5) * w;
    return { centers, counts, width: w };
  };

  // ----------------------------------------------------------- RNG
  /** Fast seedable RNG (mulberry32) + Gaussian deviates (Box–Muller). */
  PM.RNG = class {
    constructor(seed) { this.s = (seed >>> 0) || 1; this._spare = null; }
    next() {
      let t = (this.s += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
    int(a, b) { return a + Math.floor(this.next() * (b - a + 1)); }
    gauss(mu = 0, sigma = 1) {
      if (this._spare !== null) { const v = this._spare; this._spare = null; return mu + sigma * v; }
      let u, v, s;
      do { u = 2 * this.next() - 1; v = 2 * this.next() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
      const f = Math.sqrt(-2 * Math.log(s) / s);
      this._spare = v * f;
      return mu + sigma * u * f;
    }
  };

  // ----------------------------------------------------------- FFT
  const fftCache = new Map();
  function fftPlan(n) {
    let p = fftCache.get(n);
    if (p) return p;
    if ((n & (n - 1)) !== 0) throw new Error("FFT size must be a power of two: " + n);
    const rev = new Uint32Array(n);
    let bits = 0; while ((1 << bits) < n) bits++;
    for (let i = 0; i < n; i++) {
      let r = 0, x = i;
      for (let b = 0; b < bits; b++) { r = (r << 1) | (x & 1); x >>= 1; }
      rev[i] = r;
    }
    const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    p = { rev, cos, sin };
    fftCache.set(n, p);
    return p;
  }
  /** In-place complex FFT. inverse=true gives the inverse transform normalised by 1/n. */
  PM.fft = function (re, im, inverse, n, offset, stride) {
    n = n || re.length; offset = offset || 0; stride = stride || 1;
    const { rev, cos, sin } = fftPlan(n);
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const a = offset + i * stride, b = offset + j * stride;
        let t = re[a]; re[a] = re[b]; re[b] = t;
        t = im[a]; im[a] = im[b]; im[b] = t;
      }
    }
    const sgn = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = cos[k * step], wi = sgn * sin[k * step];
          const a = offset + (start + k) * stride, b = offset + (start + k + half) * stride;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { const a = offset + i * stride; re[a] /= n; im[a] /= n; }
  };
  /** 2D FFT (row-major nx*ny array). */
  PM.fft2 = function (re, im, nx, ny, inverse) {
    for (let j = 0; j < ny; j++) PM.fft(re, im, inverse, nx, j * nx, 1);
    for (let i = 0; i < nx; i++) PM.fft(re, im, inverse, ny, i, nx);
  };
  /** Angular wavenumbers k = 2π·fftfreq(n, d). */
  PM.fftk = function (n, d) {
    const k = new Float64Array(n);
    for (let i = 0; i < n; i++) k[i] = (2 * Math.PI / (n * d)) * (i < n / 2 ? i : i - n);
    return k;
  };
  /** Shift zero frequency to the centre (for display). */
  PM.fftshift2 = function (a, nx, ny, out) {
    out = out || new a.constructor(a.length);
    const hx = nx >> 1, hy = ny >> 1;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++)
      out[((j + hy) % ny) * nx + ((i + hx) % nx)] = a[j * nx + i];
    return out;
  };

  // ----------------------------------------------------------- ODE
  /**
   * Classic RK4 step. f(t, y, dydt) fills dydt. y is updated in place.
   * Pass the returned workspace ws back in to avoid allocations.
   */
  PM.rk4 = function (f, t, y, h, ws) {
    const n = y.length;
    if (!ws || ws.n !== n) ws = { n, k1: new Float64Array(n), k2: new Float64Array(n), k3: new Float64Array(n), k4: new Float64Array(n), tmp: new Float64Array(n) };
    const { k1, k2, k3, k4, tmp } = ws;
    f(t, y, k1);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * h * k1[i];
    f(t + 0.5 * h, tmp, k2);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * h * k2[i];
    f(t + 0.5 * h, tmp, k3);
    for (let i = 0; i < n; i++) tmp[i] = y[i] + h * k3[i];
    f(t + h, tmp, k4);
    for (let i = 0; i < n; i++) y[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    return ws;
  };

  // ----------------------------------------------------------- linear algebra
  /** Thomas algorithm: a (sub-diagonal), b (diagonal), c (super-diagonal), d (right-hand side). */
  PM.solveTridiag = function (a, b, c, d) {
    const n = b.length, cp = new Float64Array(n), dp = new Float64Array(n), x = new Float64Array(n);
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0];
    for (let i = 1; i < n; i++) {
      const m = b[i] - a[i] * cp[i - 1];
      cp[i] = i < n - 1 ? c[i] / m : 0;
      dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    x[n - 1] = dp[n - 1];
    for (let i = n - 2; i >= 0; i--) x[i] = dp[i] - cp[i] * x[i + 1];
    return x;
  };
  /**
   * Symmetric tridiagonal matrix (diagonal d, off-diagonal e; e.length = n-1):
   * the LOWEST k eigenvalues (Sturm bisection) and normalised eigenvectors (inverse iteration).
   * O(k·n) for finite-difference Hamiltonians — fast even for large n.
   */
  PM.tridiagLowest = function (d, e, k) {
    const n = d.length;
    k = Math.min(k, n);
    const count = (x) => { // number of eigenvalues below x
      let c = 0, q = d[0] - x;
      if (q < 0) c++;
      for (let i = 1; i < n; i++) {
        const qq = q === 0 ? 1e-300 : q;
        q = d[i] - x - (e[i - 1] * e[i - 1]) / qq;
        if (q < 0) c++;
      }
      return c;
    };
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const r = (i > 0 ? Math.abs(e[i - 1]) : 0) + (i < n - 1 ? Math.abs(e[i]) : 0);
      lo = Math.min(lo, d[i] - r); hi = Math.max(hi, d[i] + r);
    }
    const values = new Float64Array(k), vectors = [];
    for (let m = 0; m < k; m++) {
      let a = lo, b = hi;
      for (let it = 0; it < 100; it++) {
        const mid = 0.5 * (a + b);
        if (count(mid) > m) b = mid; else a = mid;
        if (b - a < 1e-13 * Math.max(1, Math.abs(mid))) break;
      }
      values[m] = 0.5 * (a + b);
    }
    // inverse iteration
    const sub = new Float64Array(n), sup = new Float64Array(n);
    for (let i = 0; i < n - 1; i++) { sub[i + 1] = e[i]; sup[i] = e[i]; }
    for (let m = 0; m < k; m++) {
      const shift = values[m] + 1e-10 * Math.max(1, Math.abs(values[m]));
      const diag = new Float64Array(n);
      for (let i = 0; i < n; i++) diag[i] = d[i] - shift;
      let v = new Float64Array(n);
      const rng = new PM.RNG(12345 + m);
      for (let i = 0; i < n; i++) v[i] = rng.uniform(-1, 1);
      for (let it = 0; it < 4; it++) {
        v = PM.solveTridiag(sub, diag, sup, v);
        // orthogonalise against previous vectors (near-degenerate eigenvalues)
        for (let p = 0; p < vectors.length; p++) {
          if (Math.abs(values[p] - values[m]) < 1e-6 * Math.max(1, Math.abs(values[m]))) {
            let dot = 0; for (let i = 0; i < n; i++) dot += v[i] * vectors[p][i];
            for (let i = 0; i < n; i++) v[i] -= dot * vectors[p][i];
          }
        }
        let nrm = 0; for (let i = 0; i < n; i++) nrm += v[i] * v[i];
        nrm = Math.sqrt(nrm) || 1;
        for (let i = 0; i < n; i++) v[i] /= nrm;
      }
      // sign convention: first significant amplitude positive
      let s = 0; for (let i = 0; i < n; i++) { if (Math.abs(v[i]) > 1e-6) { s = Math.sign(v[i]); break; } }
      if (s < 0) for (let i = 0; i < n; i++) v[i] = -v[i];
      vectors.push(v);
    }
    return { values, vectors };
  };

  // ----------------------------------------------------------- special functions
  const factCache = [1];
  PM.factorial = function (n) {
    for (let i = factCache.length; i <= n; i++) factCache[i] = factCache[i - 1] * i;
    return factCache[n];
  };
  PM.logGamma = function (x) { // Lanczos
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
      -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - PM.logGamma(1 - x);
    x -= 1; let a = c[0]; const t = x + g + 0.5;
    for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
  };
  PM.binomPMF = function (n, k, p) {
    if (k < 0 || k > n) return 0;
    if (p <= 0) return k === 0 ? 1 : 0;
    if (p >= 1) return k === n ? 1 : 0;
    return Math.exp(PM.logGamma(n + 1) - PM.logGamma(k + 1) - PM.logGamma(n - k + 1) + k * Math.log(p) + (n - k) * Math.log(1 - p));
  };
  PM.erf = function (x) { // Abramowitz–Stegun 7.1.26
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  };
  /** Generalised Laguerre L_n^(α)(x) (three-term recurrence). */
  PM.genLaguerre = function (n, alpha, x) {
    if (n === 0) return 1;
    let L0 = 1, L1 = 1 + alpha - x;
    for (let k = 1; k < n; k++) {
      const L2 = ((2 * k + 1 + alpha - x) * L1 - (k + alpha) * L0) / (k + 1);
      L0 = L1; L1 = L2;
    }
    return L1;
  };
  /** Associated Legendre P_l^m(x), m ≥ 0, including the Condon–Shortley phase. */
  PM.assocLegendre = function (l, m, x) {
    let pmm = 1;
    if (m > 0) {
      const s = Math.sqrt(Math.max(0, (1 - x) * (1 + x)));
      let f = 1;
      for (let i = 1; i <= m; i++) { pmm *= -f * s; f += 2; }
    }
    if (l === m) return pmm;
    let pmmp1 = x * (2 * m + 1) * pmm;
    if (l === m + 1) return pmmp1;
    let pll = 0;
    for (let ll = m + 2; ll <= l; ll++) {
      pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
      pmm = pmmp1; pmmp1 = pll;
    }
    return pll;
  };
  /** Normalisation N_lm of the spherical harmonic: Y = N·P_l^|m|(cosθ)·e^{imφ} */
  PM.sphHarmNorm = function (l, m) {
    const am = Math.abs(m);
    return Math.sqrt(((2 * l + 1) / (4 * Math.PI)) * PM.factorial(l - am) / PM.factorial(l + am));
  };
  /** Hydrogen-like radial wavefunction R_nl(r) (units of a0 = 1). */
  PM.radialH = function (n, l, r, Z) {
    Z = Z || 1;
    const rho = 2 * Z * r / n;
    const norm = Math.sqrt(Math.pow(2 * Z / n, 3) * PM.factorial(n - l - 1) / (2 * n * PM.factorial(n + l)));
    return norm * Math.exp(-rho / 2) * Math.pow(rho, l) * PM.genLaguerre(n - l - 1, 2 * l + 1, rho);
  };

  // ----------------------------------------------------------- formatting
  PM.fmt = function (x, d) {
    if (!isFinite(x)) return "—";
    d = d === undefined ? 3 : d;
    const ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-3 || ax >= 1e5)) {
      const [m, e] = x.toExponential(Math.max(0, d - 1)).split("e");
      const sup = { "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
      return m + "×10" + e.split("").map((c) => sup[c]).join("");
    }
    return x.toFixed(d);
  };

  window.PM = PM;
})();
