/* Taylor-series approximation — coefficients from closed-form derivatives, animated term-by-term build-up. */
App.register({
  id: "taylor-series",
  category: "classical",
  group: "Mathematical Methods",
  order: 41,
  title: "Taylor Series Approximation",
  icon: "📈",
  subtitle: "A function (solid) and its Taylor polynomial $T_N$ (dashed) built from the derivatives at one point: as terms are added the region where the polynomial is accurate grows — for $\\ln x$ only up to the radius of convergence.",
  notes: [{ type: "info", html: "Press <b>Play</b> to add terms one by one (zero coefficients are skipped), or set $N$ by hand. The lower-left plot shows the error $|f-T_N|$ on a log scale, the lower-right plot the size of the coefficients $|c_n|$, whose decay rate determines the radius of convergence." }],
  animated: true,
  speed: { min: 0.2, max: 4, value: 1, step: 0.1 },
  controls: [
    { id: "fn", type: "select", label: "Function", value: "sin", options: [
      { value: "sin", label: "sin(x)   [a = π/2]" },
      { value: "cos", label: "cos(x)   [a = 0]" },
      { value: "exp", label: "exp(x)   [a = 0]" },
      { value: "ln", label: "ln(x)   [a = 1]" },
    ] },
    { id: "N", type: "slider", label: "Number of terms $N$", min: 1, max: 50, step: 1, value: 1, live: true,
      help: "$N$ terms: powers $n = 0,1,\\dots,N-1$ (terms with a zero coefficient included)." },
    { id: "xw", type: "slider", label: "Displayed range $\\pm$ (multiples of π)", min: 1, max: 10, step: 1, value: 4, live: true },
    { type: "section", label: "Animation" },
    { id: "Nmax", type: "slider", label: "Maximum number of terms in the animation", min: 2, max: 50, step: 1, value: 40, live: true,
      help: "Play adds the terms one at a time and starts over after reaching this limit. Changing $N$ by hand pauses the animation." },
    { id: "showTerm", type: "checkbox", label: "Also show the term being added", value: true, live: true },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>Almost every approximation in physics — small-angle pendulum, harmonic approximation of a potential minimum, linear response,
    perturbation theory, the low-velocity limit of relativistic energy — is a truncated Taylor series. This page shows four
    elementary functions and their Taylor polynomials about a fixed expansion point $a$:</p>
    <table>
      <tr><th>function</th><th>expansion point $a$</th><th>domain shown</th><th>radius of convergence $R$</th></tr>
      <tr><td>$\\sin x$</td><td>$\\pi/2$</td><td>all $x$</td><td>$\\infty$</td></tr>
      <tr><td>$\\cos x$</td><td>$0$</td><td>all $x$</td><td>$\\infty$</td></tr>
      <tr><td>$e^x$</td><td>$0$</td><td>all $x$</td><td>$\\infty$</td></tr>
      <tr><td>$\\ln x$</td><td>$1$</td><td>$x&gt;0$</td><td>$1$</td></tr>
    </table>
    <p>All quantities are dimensionless. The controls are the number of terms $N$ (powers $0,\\dots,N-1$) and the half-width of
    the displayed $x$ window in multiples of $\\pi$.</p>

    <h4>Equations being solved</h4>
    <p>Taylor's theorem with the Lagrange form of the remainder: if $f$ has $N$ continuous derivatives between $a$ and $x$,</p>
    <div class="callout">$$f(x)=\\underbrace{\\sum_{n=0}^{N-1}\\frac{f^{(n)}(a)}{n!}(x-a)^n}_{T_N(x)}\\;+\\;R_N(x),\\qquad
      R_N(x)=\\frac{f^{(N)}(\\xi)}{N!}(x-a)^N\\ \\text{ for some }\\xi\\text{ between }a\\text{ and }x .$$</div>
    <p>The coefficients $c_n=f^{(n)}(a)/n!$ follow from closed-form derivatives:</p>
    $$\\frac{d^n}{dx^n}\\sin x=\\sin\\!\\Big(x+\\frac{n\\pi}{2}\\Big)\\ \\Rightarrow\\ \\sin^{(n)}\\!\\Big(\\frac{\\pi}{2}\\Big)=\\cos\\frac{n\\pi}{2},\\qquad
      \\cos^{(n)}(0)=\\cos\\frac{n\\pi}{2},\\qquad (e^x)^{(n)}\\big|_0=1,$$
    $$\\frac{d^n}{dx^n}\\ln x=\\frac{(-1)^{n+1}(n-1)!}{x^n}\\ (n\\ge1)\\ \\Rightarrow\\ c_0=0,\\quad c_n=\\frac{(-1)^{n+1}}{n}.$$
    <p>Hence ($\\cos(n\\pi/2)$ is $1,0,-1,0,\\dots$, so only even powers survive for $\\sin$ about $\\pi/2$ and for $\\cos$ about $0$):</p>
    $$\\sin x=\\sum_{m=0}^{\\infty}\\frac{(-1)^m}{(2m)!}\\Big(x-\\frac{\\pi}{2}\\Big)^{2m},\\quad
      \\cos x=\\sum_{m=0}^{\\infty}\\frac{(-1)^m x^{2m}}{(2m)!},\\quad
      e^x=\\sum_{n=0}^{\\infty}\\frac{x^n}{n!},\\quad
      \\ln x=\\sum_{n=1}^{\\infty}\\frac{(-1)^{n+1}}{n}(x-1)^n .$$
    <p><b>Error bounds.</b> For $\\sin$ and $\\cos$ every derivative is bounded by 1, so $|R_N|\\le|x-a|^N/N!$; for the exponential
    $|R_N|\\le e^{\\max(0,x)}|x|^N/N!$. Since $N!\\approx(N/e)^N\\sqrt{2\\pi N}$, the error is small roughly where $|x-a|\\lesssim N/e$:
    the zone of validity grows <em>linearly</em> with $N$, and the series converges for every $x$ ($R=\\infty$, entire functions).</p>
    <p><b>Radius of convergence.</b> By the Cauchy–Hadamard formula $R=1/\\limsup_n|c_n|^{1/n}$. For $\\ln x$, $|c_n|=1/n$ gives
    $|c_n|^{1/n}\\to1$, i.e. $R=1$: the series converges for $0&lt;x\\le2$ (conditionally at $x=2$, where it becomes the alternating
    harmonic series $\\ln2=1-\\tfrac12+\\tfrac13-\\dots$) and diverges for $x&gt;2$ because the terms $|x-1|^n/n$ grow geometrically.
    $R$ equals the distance from $a=1$ to the nearest singularity of $\\ln z$ in the complex plane, the branch point $z=0$.
    For the factorial coefficients of the other three functions $|c_n|^{1/n}\\to0$, i.e. $R=\\infty$ — visible in the coefficient
    plot as bars that fall ever faster on the log scale, whereas for $\\ln x$ they barely decrease.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Coefficients:</b> the 50 coefficients $c_0,\\dots,c_{49}$ are computed once from the closed forms above
      ($1/n!$ via a cached factorial table) — no numerical differentiation is used.</li>
      <li><b>Evaluation:</b> $T_N$ is evaluated at 1000 equally spaced points of the window with Horner's scheme
      $T_N=c_0+u(c_1+u(c_2+\\dots))$, $u=x-a$, which needs $N$ multiplications per point and is numerically stable.</li>
      <li><b>Error plot:</b> $\\log_{10}|f(x)-T_N(x)|$, floored at $10^{-17}$; the dashed line marks $10^{-2}$. Double precision
      limits the attainable error to about $10^{-16}$ relative; far from $a$ with many terms, large alternating terms (up to
      $|x-a|^n/n!\\sim e^{|x-a|}$) cancel, so a cancellation floor $\\sim10^{-16}e^{|x-a|}$ appears.</li>
      <li><b>Animation:</b> a continuous term index $\\nu$ increases at 2.2 terms/s (times the speed factor). The next term
      $c_N(x-a)^N$ is blended in with a smooth-step weight $w=t^2(3-2t)$, $t=\\nu-N$, so the curve morphs smoothly; terms with
      zero coefficient are skipped because they do not change the curve. After reaching the maximum the picture holds for 1.6 s and
      restarts.</li>
      <li><b>Metrics:</b> $N$; the polynomial degree (highest power with $c_n\\neq0$); $a$; the maximum of $|f-T_N|$ over the window
      (ignoring $x\\le0$ for $\\ln$); and the interval between the leftmost and rightmost sample points where $|f-T_N|&lt;0.01$.
      The status line shows the first terms of $T_N$ in closed form.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>sin, $N=3$:</b> $T_3=1-\\tfrac12(x-\\pi/2)^2$ — the harmonic (parabolic) approximation of a maximum. Increase $N$: the
      accurate region grows by about one half-period every few terms; with $N=40$ the curve is indistinguishable from $\\sin x$
      over $\\pm4\\pi$.</li>
      <li><b>exp, range $\\pm\\pi$:</b> with $N=4$ (degree 3) the polynomial dives to $-\\infty$ on the left, with $N=5$ it turns up to
      $+\\infty$, while $e^x\\to0$. A small number $e^{-|x|}$ is the result of cancellation between large terms, so the left side
      needs many more terms than the right; watch the error plot become asymmetric.</li>
      <li><b>ln, increase $N$:</b> inside the green band $0&lt;x&lt;2$ the error falls like $|x-1|^N/N$; outside it the polynomial
      explodes no matter how many terms are used. Check the coefficient plot: $|c_n|=1/n$ decays only algebraically.</li>
      <li><b>ln near $x=2$:</b> convergence is painfully slow (error $\\approx 1/(2N)$ for the alternating harmonic series), a
      reminder that "convergent" and "useful" are different things.</li>
      <li>Compare the width of the $|f-T_N|&lt;0.01$ interval for cos at $N=11$ and $N=21$: it roughly doubles, as the estimate
      $|x|\\lesssim N/e$ predicts.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>Only real-valued functions with closed-form derivatives are offered; the series is truncated at 50 terms and evaluated in
    double precision, so beyond about $|x-a|\\approx 30$ round-off rather than truncation dominates. Analytic continuation and
    other approximants (Padé, Chebyshev) that converge beyond $R$ are not shown.
    References: M. L. Boas, <i>Mathematical Methods in the Physical Sciences</i> (ch. 1); Arfken, Weber &amp; Harris,
    <i>Mathematical Methods for Physicists</i>; T. M. Apostol, <i>Calculus</i>, Vol. 1; for numerical aspects, Press et al.,
    <i>Numerical Recipes</i>.</p>`,

  mount(api) {
    const P = api.params, NS = 1000, NC = 50;
    const FUN = {
      sin: { a: Math.PI / 2, aTex: "\\tfrac{\\pi}{2}", name: "sin x", ylim: [-3, 3], f: Math.sin, R: Infinity },
      cos: { a: 0, aTex: "0", name: "cos x", ylim: [-3, 3], f: Math.cos, R: Infinity },
      exp: { a: 0, aTex: "0", name: "eˣ", ylim: [-5, 20], f: Math.exp, R: Infinity },
      ln: { a: 1, aTex: "1", name: "ln x", ylim: [-5, 5], f: (x) => (x > 0 ? Math.log(x) : NaN), R: 1 },
    };
    // closed-form coefficients c_n = f⁽ⁿ⁾(a)/n!
    function coeffs(key) {
      const c = new Float64Array(NC);
      for (let n = 0; n < NC; n++) {
        if (key === "sin" || key === "cos") { const r = n % 4; c[n] = (r === 0 ? 1 : r === 2 ? -1 : 0) / PM.factorial(n); }
        else if (key === "exp") c[n] = 1 / PM.factorial(n);
        else c[n] = n === 0 ? 0 : (n % 2 === 1 ? 1 : -1) / n;
      }
      return c;
    }

    // log10 values are drawn on a linear axis; tick labels read 10ⁿ
    const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    function pow10(v) { const e = Math.round(v); if (Math.abs(v - e) > 1e-6) return ""; return e === 0 ? "1" : "10" + String(e).split("").map((ch) => SUP[ch] || ch).join(""); }
    const M = api.metrics([
      { id: "N", label: "Number of terms $N$" },
      { id: "deg", label: "Polynomial degree" },
      { id: "a", label: "Expansion point $a$" },
      { id: "err", label: "Max. error in the window" },
      { id: "good", label: "Region where $|f-T_N|<0.01$" },
    ]);
    const plots = api.plots([
      { id: "main", title: "Function and Taylor approximation", span: 2, aspect: 0.4, xlim: [-4 * Math.PI, 4 * Math.PI], ylim: [-3, 3], xlabel: "x", ylabel: "f(x)", minHeight: 260 },
      { id: "err", title: "Absolute error $|f(x)-T_N(x)|$ (log scale)", aspect: 0.62, xlim: [-4 * Math.PI, 4 * Math.PI], ylim: [-16, 6], ytickFormat: pow10, xlabel: "x", ylabel: "|f − T|" },
      { id: "coef", title: "Coefficients $|c_n|$ (log scale, terms in use highlighted)", aspect: 0.62, xlim: [-0.8, 49.8], ylim: [-66, 1], ytickFormat: pow10, xlabel: "n", ylabel: "|cₙ|" },
    ]);

    const xs = new Float64Array(NS), yt = new Float64Array(NS), yp = new Float64Array(NS), yterm = new Float64Array(NS);
    const ye = new Float64Array(NS), ypc = new Float64Array(NS), ytc = new Float64Array(NS), ytermc = new Float64Array(NS);
    let key = null, C = null, F = null;
    let nu = 1;          // continuous "term position" (animation)
    let hold = 0, lastStatus = "";
    const nIdx = new Float64Array(NC), nAbs = new Float64Array(NC);
    for (let n = 0; n < NC; n++) nIdx[n] = n;

    function ensureFun() {
      if (key === P.fn) return;
      key = P.fn; F = FUN[key]; C = coeffs(key);
      for (let n = 0; n < NC; n++) nAbs[n] = C[n] !== 0 ? Math.log10(Math.abs(C[n])) : NaN;
      plots.main.setLimits(null, F.ylim);
      plots.coef.setLimits(null, key === "ln" ? [-2.5, 0.5] : [-66, 1]);
    }
    const nextNonzero = (n) => { let m = n; while (m < NC && C[m] === 0) m++; return m; };

    // T_N(x) by Horner's scheme, N terms
    function taylor(x, N) {
      const u = x - F.a;
      let s = 0;
      for (let n = N - 1; n >= 0; n--) s = s * u + C[n];
      return s;
    }

    function texPoly(N) {
      const u = F.a === 0 ? "x" : `(x-${F.aTex})`, parts = [];
      let shown = 0, total = 0;
      for (let n = 0; n < N; n++) if (C[n] !== 0) total++;
      for (let n = 0; n < N && shown < 5; n++) {
        if (C[n] === 0) continue;
        const sgn = C[n] < 0 ? "-" : "+";
        let mag;
        if (key === "ln") mag = n === 1 ? u : `\\frac{${u}^{${n}}}{${n}}`;
        else mag = n === 0 ? "1" : n === 1 ? u : `\\frac{${u}^{${n}}}{${n}!}`;
        parts.push({ sgn, mag });
        shown++;
      }
      let s = parts.map((p, i) => (i === 0 ? (p.sgn === "-" ? "-" : "") : ` ${p.sgn} `) + p.mag).join("");
      if (!s) s = "0";
      if (total > shown) s += " + \\cdots";
      return `$$T_{${N}}(x) = ${s}$$`;
    }

    function clampY(src, dst, lo, hi) {
      for (let i = 0; i < src.length; i++) { const v = src[i]; dst[i] = isFinite(v) ? (v < lo ? lo : v > hi ? hi : v) : NaN; }
    }

    return {
      reset() { ensureFun(); nu = P.N; hold = 0; },
      onParam(id, v) {
        if (id === "fn") { ensureFun(); nu = 1; api.setControl("N", { value: 1 }); }
        if (id === "N") { nu = v; hold = 0; api.pause(); } // a manual choice pauses the animation
        if (id === "Nmax" && nu > v) { nu = 1; api.setControl("N", { value: 1 }); }
      },
      step(dt) {
        ensureFun();
        const Nmax = P.Nmax;
        if (hold > 0) { hold -= dt; if (hold <= 0) { nu = 1; api.setControl("N", { value: 1 }); } return; }
        let N = Math.floor(nu);
        // skip zero-coefficient terms (the curve would not change)
        const nz = nextNonzero(N);
        if (nz > N) { nu = Math.min(nz, Nmax) + (nu - N); N = Math.floor(nu); }
        nu += dt * 2.2;
        if (nu >= Nmax) { nu = Nmax; hold = 1.6; }
        const Nn = Math.floor(nu);
        if (Nn !== P.N) api.setControl("N", { value: Nn });
      },
      render() {
        ensureFun();
        const xw = P.xw * Math.PI;
        const pm = plots.main, pe = plots.err, pc = plots.coef;
        pm.setLimits([-xw, xw]); pe.setLimits([-xw, xw]);
        const N = Math.max(1, Math.min(NC, Math.floor(nu)));
        // transition: the next non-zero term is blended in smoothly
        const playing = api.isPlaying && hold <= 0 && N < P.Nmax;
        const fr = playing ? nu - N : 0;
        const al = fr * fr * (3 - 2 * fr);
        const nNext = N < NC ? N : -1; // power of the term being added
        const [y0, y1] = F.ylim, span = y1 - y0;
        let maxErr = 0, goodL = NaN, goodR = NaN;
        for (let i = 0; i < NS; i++) {
          const x = -xw + (2 * xw * i) / (NS - 1);
          xs[i] = x;
          yt[i] = F.f(x);
          let t = taylor(x, N);
          yterm[i] = nNext >= 0 ? C[nNext] * Math.pow(x - F.a, nNext) : 0;
          if (al > 0) t += al * yterm[i];
          yp[i] = t;
          const e = Math.abs(yt[i] - t);
          ye[i] = isFinite(e) ? Math.log10(Math.max(e, 1e-17)) : NaN;
          if (isFinite(e)) { maxErr = Math.max(maxErr, e); if (e < 0.01) { if (!(goodL <= x)) goodL = x; goodR = x; } }
        }
        clampY(yt, ytc, y0 - 3 * span, y1 + 3 * span);
        clampY(yp, ypc, y0 - 3 * span, y1 + 3 * span);
        clampY(yterm, ytermc, y0 - 3 * span, y1 + 3 * span);

        // --- main plot
        pm.clear();
        if (F.R < Infinity) {
          pm.rect(F.a - F.R, y0 - span, F.a + F.R, y1 + span, { color: PlotColors.good, alpha: 0.07 });
          pm.text(F.a, y0 + 0.06 * span, "convergence region |x − 1| < 1", { color: PlotColors.good, size: 10.5, align: "center" });
        }
        pm.hline(0, { color: PlotColors.muted, width: 1, alpha: 0.6 });
        pm.vline(F.a, { color: PlotColors.good, dash: [2, 4], width: 1.4 });
        if (P.showTerm && al > 0 && C[nNext] !== 0) pm.line(xs, ytermc, { color: PlotColors.accent3, width: 1.2, dash: [3, 3], alpha: 0.5 * al + 0.2 });
        pm.line(xs, ytc, { color: PlotColors.accent, width: 2.4 });
        pm.line(xs, ypc, { color: PlotColors.bad, width: 2, dash: [8, 5] });
        pm.circle(F.a, taylor(F.a, 1), 4.5, { px: true, color: PlotColors.good });
        const leg = [{ label: `exact function: ${F.name}`, color: PlotColors.accent }, { label: `Taylor polynomial (${N} term${N === 1 ? "" : "s"})`, color: PlotColors.bad, dash: [8, 5] }];
        if (P.showTerm && al > 0 && C[nNext] !== 0) leg.push({ label: `term being added (n = ${nNext})`, color: PlotColors.accent3, dash: [3, 3] });
        leg.push({ label: `expansion point a = ${PM.fmt(F.a, 2)}`, color: PlotColors.good, dash: [2, 4] });
        pm.legend(leg, "tr");
        pm.label(`N = ${N}${al > 0 ? " → " + (N + 1) : ""}`, "tl", { bold: true, size: 13 });

        // --- error
        pe.clear();
        pe.hline(-2, { color: PlotColors.muted, dash: [4, 4], width: 1 });
        pe.vline(F.a, { color: PlotColors.good, dash: [2, 4], width: 1.2 });
        if (F.R < Infinity) { pe.vline(F.a - F.R, { color: PlotColors.good, width: 1, alpha: 0.5 }); pe.vline(F.a + F.R, { color: PlotColors.good, width: 1, alpha: 0.5 }); }
        pe.line(xs, ye, { color: PlotColors.accent3, width: 1.5 });

        // --- coefficients
        pc.clear();
        const cols = [];
        for (let n = 0; n < NC; n++) cols.push(n < N ? PlotColors.accent2 : "rgba(139,152,168,0.35)");
        const base = pc.ylim[0];
        const hs = new Float64Array(NC);
        for (let n = 0; n < NC; n++) hs[n] = isFinite(nAbs[n]) ? nAbs[n] : base;
        pc.bars(nIdx, hs, 0.8, { colors: cols, base, alpha: 0.9 });
        if (al > 0 && nNext >= 0 && isFinite(nAbs[nNext])) pc.bars([nNext], [nAbs[nNext]], 0.8, { color: PlotColors.accent3, base, alpha: 0.4 + 0.6 * al });

        // --- metrics / formula
        let deg = -1; for (let n = 0; n < N; n++) if (C[n] !== 0) deg = n;
        M.set("N", String(N));
        M.set("deg", deg < 0 ? "—" : String(deg));
        M.set("a", PM.fmt(F.a, 4));
        M.set("err", PM.fmt(maxErr, 3));
        M.set("good", isFinite(goodL) ? `[${PM.fmt(goodL, 2)}, ${PM.fmt(goodR, 2)}]` : "—");
        const st = key + ":" + N;
        if (st !== lastStatus) { lastStatus = st; api.status(texPoly(N)); }
        api.setTime(`N = ${N}`);
      },
    };
  },
});
