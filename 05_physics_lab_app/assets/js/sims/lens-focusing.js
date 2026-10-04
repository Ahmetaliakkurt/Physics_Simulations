/* Focusing light and designing a symmetric biconvex lens — live calculation, ray diagram and dispersion. */
App.register({
  id: "lens-focusing",
  category: "classical",
  group: "Optics",
  order: 30,
  title: "Laser Focusing & Lens Design",
  icon: "🔍",
  subtitle: "Place a lens and a target: the thin-lens equation gives the required focal length, the thick-lens lensmaker's equation the surface radius of a symmetric biconvex lens — and dispersion shows where other colours would focus.",
  animated: true,
  speed: { min: 0.2, max: 3, value: 1, step: 0.1 },
  notes: [
    { type: "info", html: "The wavelength dependence of the refractive index is modelled as $n(\\lambda) = 1.50 + 50000/\\lambda^2$ ($\\lambda$ in nm). Across the visible range this gives $\\Delta n \\approx 0.16$, whereas real optical glasses have $\\Delta n \\approx 0.01$–$0.02$: the dispersion is exaggerated on purpose to make its effect visible." },
  ],
  controls: [
    { id: "wl", type: "slider", label: "Wavelength $\\lambda$", min: 400, max: 750, step: 0.1, value: 632.8, unit: "nm", live: true,
      help: "632.8 nm is the red helium–neon laser line." },
    { id: "d", type: "slider", label: "Lens centre thickness $d$", min: 1, max: 20, step: 0.5, value: 5, unit: "mm", live: true },
    { type: "section", label: "Positions (measured from the source)" },
    { id: "s1", type: "slider", label: "Lens position $s_1$", min: 1, max: 499, step: 1, value: 100, unit: "mm", live: true },
    { id: "tp", type: "slider", label: "Target position $s_1 + s_2'$", min: 2, max: 500, step: 1, value: 200, unit: "mm", live: true,
      help: "The target (focus) must lie beyond the lens: $s_2' > 0$." },
    { type: "section", label: "Display" },
    { id: "pulses", type: "checkbox", label: "Animate light pulses along the rays", value: true, live: true },
    { id: "disp", type: "checkbox", label: "Show blue (450 nm) and red (700 nm) rays", value: false, live: true,
      help: "The lens is designed for the selected $\\lambda$; other colours focus at different points (chromatic aberration)." },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>A point source of monochromatic light (e.g. a laser fibre tip) sits on the optical axis at $x=0$. A lens placed at distance $s_1$
    must form a sharp image of the source on a target at $x=s_1+s_2'$. The lens is a symmetric biconvex singlet of centre thickness $d$
    made of a glass with refractive index $n(\\lambda)$, in air ($n_{\\rm air}=1$). The task — the everyday problem of an optical designer —
    is: <em>which focal length is needed, and with what surface radius $R$ must the glass be ground?</em> All lengths are in millimetres,
    wavelengths in nanometres. Assumptions: paraxial (Gaussian) optics, i.e. rays make small angles with the axis; the ray diagram
    treats the lens as thin, while the radius calculation keeps the thickness exactly.</p>

    <h4>Equations being solved</h4>
    <p><b>1. Required focal length.</b> With object distance $s_1$ and image distance $s_2'$ (both positive for a real object and real image),
    the Gaussian thin-lens equation gives</p>
    $$\\frac1f=\\frac1{s_1}+\\frac1{s_2'}\\qquad\\Longrightarrow\\qquad f=\\frac{s_1s_2'}{s_1+s_2'} .$$
    <p><b>2. Lensmaker's equation for a thick lens</b> (surface radii $R_1$, $R_2$ with the usual sign convention, centre thickness $d$):</p>
    $$\\frac1f=(n-1)\\left[\\frac1{R_1}-\\frac1{R_2}+\\frac{(n-1)\\,d}{n\\,R_1R_2}\\right].$$
    <p>For a symmetric biconvex lens $R_1=R$, $R_2=-R$, which turns it into a quadratic equation for $R$:</p>
    $$\\frac1f=(n-1)\\left[\\frac2R-\\frac{(n-1)\\,d}{n\\,R^2}\\right]\\quad\\Longrightarrow\\quad \\frac1f\\,R^2-2(n-1)\\,R+\\frac{(n-1)^2d}{n}=0 .$$
    <div class="callout">$$R=f\\,(n-1)\\left[1+\\sqrt{1-\\frac{d}{nf}}\\right]\\ \\xrightarrow{\\ d\\to0\\ }\\ R_0=2(n-1)f .$$</div>
    <p>The larger root is the physical one (the smaller root describes an extremely curved, nearly spherical lens). If $d&gt;nf$ the
    discriminant is negative: no real radius can give the requested focal length with that much glass.</p>
    <p><b>3. Dispersion</b> (a Cauchy-type model): $\\;n(\\lambda)=1.50+\\dfrac{50000}{\\lambda^2}\\;$ ($\\lambda$ in nm). Once $R$ is fixed for the design
    wavelength, the focal length at any other wavelength follows from the thick-lens equation with $n(\\lambda)$, and the image position from
    the thin-lens equation, $x_{\\rm img}(\\lambda)=s_1+\\big(1/f(\\lambda)-1/s_1\\big)^{-1}$. Since $n$ is larger for short wavelengths, blue light focuses
    closer to the lens than red — longitudinal chromatic aberration.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li>Everything is closed-form and recomputed every frame from the slider values: $f$ from step 1, $n(\\lambda)$, then $R$ from the quadratic
      (both roots, the larger positive one is kept; a negative discriminant or $s_2'\\le0$ is reported as an error instead of a result).</li>
      <li>Ray diagram: five rays leave the source at heights $0,\\pm25,\\pm45$ mm at the lens plane and are refracted (thin-lens model) to the target.
      With the dispersion option, two rays at 450 nm and 700 nm are traced through the <em>same</em> lens to their own image points.
      The moving dots are light pulses travelling along the rays (purely visual).</li>
      <li>Lens cross-section, drawn to scale: two circular arcs of radius $R$ whose centres are $R-d/2$ from the lens centre; the aperture
      is limited to 92 % of the knife-edge height $\\sqrt{dR-d^2/4}$ where the two surfaces meet.</li>
      <li>Dispersion plot: the image position $x_{\\rm img}(\\lambda)$ for 141 wavelengths between 400 and 750 nm; the dashed line is the target,
      which the curve meets exactly at the design wavelength.</li>
      <li>Metrics: $f$, $n(\\lambda)$, the thick-lens radius $R$, the thin-lens value $R_0=2(n-1)f$ for comparison (the difference shows the effect of
      the thickness), and the chromatic focal shift $x_{\\rm img}(700)-x_{\\rm img}(450)$. Consistency check: inserting $R$ back into the thick-lens
      equation reproduces $f$ to machine precision, so the dispersion curve passes through the target at the design $\\lambda$.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li>Defaults ($\\lambda=632.8$ nm, $s_1=s_2'=100$ mm, $d=5$ mm): $f=50$ mm, $n=1.625$, thin-lens $R_0=62.5$ mm and thick-lens $R\\approx61.5$ mm —
      for the same focal length a thick lens needs a slightly <em>smaller</em> radius (stronger curvature), because the second surface
      sees a beam already converging inside the glass.</li>
      <li>Increase $d$ to 20 mm: $R$ drops further; move the lens and target close together (e.g. $s_1=20$, target at 40 mm, $f=10$ mm) until
      $d&gt;nf$ — no real radius exists.</li>
      <li>Symmetric imaging $s_1=s_2'$ gives $f=s_1/2$ (unit magnification); moving the target far away makes $f\\to s_1$ (collimation).</li>
      <li>Turn on the blue and red rays: with the defaults they focus about 30–40 mm apart — the exaggerated model makes chromatic aberration
      obvious. Change $\\lambda$ and see that the curve is steeper at short wavelengths ($dn/d\\lambda\\propto\\lambda^{-3}$).</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>Paraxial optics only: spherical aberration, coma and field curvature are ignored, as are diffraction (the focus of a real Gaussian
    laser beam has a waist $w_0\\approx\\lambda f/\\pi w$) and the principal-plane shift of a thick lens in the ray diagram. The Cauchy model
    with exaggerated dispersion is illustrative, not a real glass. Reading: E. Hecht, <i>Optics</i>, ch. 5–6; F. L. Pedrotti, L. M. Pedrotti &amp;
    L. S. Pedrotti, <i>Introduction to Optics</i>, ch. 2 and 18; Born &amp; Wolf, <i>Principles of Optics</i>, ch. 4.</p>`,

  mount(api) {
    const P = api.params;
    const OFFS = [-45, -25, 0, 25, 45]; // ray heights at the lens (mm)

    const M = api.metrics([
      { id: "f", label: "Focal length $f$" },
      { id: "n", label: "Refractive index $n(\\lambda)$" },
      { id: "R", label: "Surface radius $R$ (thick lens)" },
      { id: "R0", label: "Thin lens: $R_0 = 2(n-1)f$" },
      { id: "ca", label: "Chromatic shift (450 → 700 nm)" },
    ]);
    const plots = api.plots([
      { id: "bench", title: "Optical bench (ray diagram)", span: 2, aspect: 0.3, xlim: [-62, 530], ylim: [-78, 78], xlabel: "position along the optical axis (mm)", ylabel: "height (mm)", minHeight: 230 },
      { id: "lens", title: "Symmetric biconvex lens cross-section (to scale)", aspect: 0.72, xlim: [-10, 10], ylim: [-10, 10], equal: true, xlabel: "mm", ylabel: "mm" },
      { id: "disp", title: "Dispersion: image position vs wavelength", aspect: 0.72, xlim: [400, 750], ylim: [0, 520], xlabel: "λ (nm)", ylabel: "image position (mm)" },
    ]);

    const nOf = (wl) => 1.5 + 50000 / (wl * wl);
    function radius(f, n, d) {
      if (!(f > 0)) return { R: null, err: "A positive focal length is required." };
      const a = 1 / f, b = -2 * (n - 1), c = ((n - 1) ** 2 * d) / n;
      const disc = b * b - 4 * a * c;
      if (disc < 0) return { R: null, err: "No real surface radius (complex root): d > n·f." };
      const R = Math.max((-b + Math.sqrt(disc)) / (2 * a), (-b - Math.sqrt(disc)) / (2 * a));
      if (!(R > 0)) return { R: null, err: "No positive surface radius found." };
      return { R, err: null };
    }
    // thick-lens focal length (R1 = R, R2 = −R)
    const focalOf = (R, n, d) => 1 / ((n - 1) * (2 / R - ((n - 1) * d) / (n * R * R)));
    function imagePos(s1, f) { const inv = 1 / f - 1 / s1; return inv > 0 ? s1 + 1 / inv : Infinity; }

    // wavelength → RGB (approximate visible-spectrum mapping)
    function wlColor(wl, alpha) {
      let r = 0, g = 0, b = 0;
      if (wl < 440) { r = -(wl - 440) / 60; b = 1; }
      else if (wl < 490) { g = (wl - 440) / 50; b = 1; }
      else if (wl < 510) { g = 1; b = -(wl - 510) / 20; }
      else if (wl < 580) { r = (wl - 510) / 70; g = 1; }
      else if (wl < 645) { r = 1; g = -(wl - 645) / 65; }
      else { r = 1; }
      let fct = 1;
      if (wl < 420) fct = 0.45 + 0.55 * (wl - 380) / 40; else if (wl > 700) fct = 0.45 + 0.55 * (780 - wl) / 80;
      const k = (v) => Math.round(255 * Math.pow(Math.max(0, v * fct), 0.8) * 0.85 + 38);
      return `rgba(${k(r)},${k(g)},${k(b)},${alpha === undefined ? 1 : alpha})`;
    }

    let phase = 0;
    const WLS = PM.linspace(400, 750, 141), nCurve = new Float64Array(141), fCurve = new Float64Array(141);

    function state() {
      const s1 = P.s1, s2 = P.tp - P.s1, n = nOf(P.wl);
      const ok = s1 > 0 && s2 > 0;
      const f = ok ? 1 / (1 / s1 + 1 / s2) : NaN;
      const r = ok ? radius(f, n, P.d) : { R: null, err: "The target must be placed beyond the lens (s₂′ > 0)." };
      return { s1, s2, n, ok, f, R: r.R, err: r.err };
    }

    // point at path length s along one ray (source → lens → focus)
    function rayPoint(h, s1, xf, s) {
      const L1 = Math.hypot(s1, h), L2 = Math.hypot(xf - s1, h);
      if (s <= L1) { const u = s / L1; return [u * s1, u * h]; }
      const u = Math.min((s - L1) / L2, 1);
      return [s1 + u * (xf - s1), h * (1 - u)];
    }

    function drawLensGlyph(p, x0, halfH, bulge, color) {
      // schematic biconvex lens on the bench
      p.custom((c) => {
        const X = p.X(x0), Yt = p.Y(halfH), Yb = p.Y(-halfH), bw = bulge;
        c.beginPath();
        c.moveTo(X, Yt);
        c.quadraticCurveTo(X + 2 * bw, (Yt + Yb) / 2, X, Yb);
        c.quadraticCurveTo(X - 2 * bw, (Yt + Yb) / 2, X, Yt);
        c.closePath();
        c.fillStyle = "rgba(88,166,255,0.22)"; c.fill();
        c.strokeStyle = color; c.lineWidth = 2; c.stroke();
      });
    }

    function drawBench(S) {
      const p = plots.bench, col = wlColor(P.wl);
      p.clear();
      p.hline(0, { color: PlotColors.muted, dash: [8, 5], width: 1, alpha: 0.7 });
      if (P.tp < 430 && P.s1 < 430) p.text(525, 7, "optical axis", { color: PlotColors.muted, size: 10, align: "right" });
      for (let x = 0; x <= 500; x += 50) p.segment(x, -78, x, -72, { color: PlotColors.muted, width: 1, alpha: 0.5 });
      // source
      p.circle(0, 0, 8, { px: true, color: col });
      p.circle(0, 0, 14, { px: true, color: col, alpha: 0.25 });
      p.text(-14, 0, "Source", { color: col, size: 11, bold: true, align: "right" });

      if (!S.ok) {
        drawLensGlyph(p, S.s1, 52, 6, PlotColors.blue);
        p.label("⚠ The target must be placed beyond the lens!", "tl", { color: PlotColors.bad, bold: true, size: 13 });
        p.circle(P.tp, 0, 6, { px: true, color: PlotColors.bad });
        return;
      }
      const xf = P.tp;
      // chromatic rays
      if (P.disp && S.R) {
        for (const wl2 of [450, 700]) {
          const n2 = nOf(wl2), f2 = focalOf(S.R, n2, P.d), xi = imagePos(S.s1, f2);
          const c2 = wlColor(wl2, 0.75);
          for (const h of [-45, 45]) {
            const xe = isFinite(xi) ? xi : 560;
            const ye = isFinite(xi) ? 0 : h;
            const ext = isFinite(xi) ? Math.min(40, 530 - xi) : 0; // continue a little beyond the focus
            const sl = (ye - h) / (xe - S.s1);
            p.segment(S.s1, h, xe + ext, ye + sl * ext, { color: c2, width: 1.2 });
          }
          if (isFinite(xi) && xi < 540) {
            p.circle(xi, 0, 4, { px: true, color: c2 });
            p.text(xi, wl2 === 450 ? 10 : -10, `${wl2} nm`, { color: c2, size: 10, align: "center" });
          }
        }
      }
      // main rays
      for (const h of OFFS) {
        p.segment(0, 0, S.s1, h, { color: col, width: 1.6, alpha: 0.85 });
        p.segment(S.s1, h, xf, 0, { color: col, width: 1.6, alpha: 0.6 });
        const sl = (0 - h) / (xf - S.s1), ext = Math.min(35, 530 - xf);
        if (ext > 0) p.segment(xf, 0, xf + ext, sl * ext, { color: col, width: 1, alpha: 0.25 });
      }
      // moving light pulses
      if (P.pulses) {
        const SP = 22; // pulse spacing (mm)
        p.custom((c) => {
          c.fillStyle = col;
          for (const h of OFFS) {
            const Ltot = Math.hypot(S.s1, h) + Math.hypot(xf - S.s1, h);
            for (let s = (phase % SP); s < Ltot; s += SP) {
              const [x, y] = rayPoint(h, S.s1, xf, s);
              const X = p.X(x), Y = p.Y(y);
              c.globalAlpha = 0.22; c.beginPath(); c.arc(X, Y, 5.5, 0, 2 * Math.PI); c.fill();
              c.globalAlpha = 1; c.beginPath(); c.arc(X, Y, 2.2, 0, 2 * Math.PI); c.fill();
            }
          }
          c.globalAlpha = 1;
        });
      }
      // lens
      const bulge = S.R ? PM.clamp(260 / S.R, 3, 14) : 6;
      drawLensGlyph(p, S.s1, 52, bulge, PlotColors.blue);
      p.text(S.s1, 62, "Lens", { color: PlotColors.blue, size: 11, bold: true, align: "center" });
      // focus
      const pulse = 1 + 0.25 * Math.sin(phase * 0.15);
      p.circle(xf, 0, 12 * pulse, { px: true, fill: false, stroke: col, strokeWidth: 1.5 });
      p.circle(xf, 0, 7, { px: true, fill: false, stroke: col, strokeWidth: 1.5 });
      p.circle(xf, 0, 4, { px: true, color: col });
      p.text(xf, -18, "Focus", { color: col, size: 11, bold: true, align: "center" });
      // dimensions
      const yd = -58;
      p.arrow(S.s1 / 2, yd, 0, yd, { color: PlotColors.accent3, width: 1.4, head: 7 });
      p.arrow(S.s1 / 2, yd, S.s1, yd, { color: PlotColors.accent3, width: 1.4, head: 7 });
      p.text(S.s1 / 2, yd + 9, `s₁ = ${PM.fmt(S.s1, 1)} mm`, { color: PlotColors.accent3, size: 11, bold: true, align: "center", bg: "#0f151c" });
      const xm = (S.s1 + xf) / 2;
      p.arrow(xm, yd, S.s1, yd, { color: PlotColors.accent2, width: 1.4, head: 7 });
      p.arrow(xm, yd, xf, yd, { color: PlotColors.accent2, width: 1.4, head: 7 });
      p.text(xm, S.s1 < 80 || S.s2 < 80 ? yd - 11 : yd + 9, `s₂′ = ${PM.fmt(S.s2, 1)} mm`, { color: "#b69cff", size: 11, bold: true, align: "center", bg: "#0f151c" });
      if (S.err) p.label("⚠ " + S.err, "tl", { color: PlotColors.bad, bold: true, size: 12 });
    }

    function drawLens(S) {
      const p = plots.lens;
      if (!S.R) {
        p.setLimits([-10, 10], [-10, 10]);
        p.clear();
        p.label(["No lens can be designed:", S.err || ""], "tl", { color: PlotColors.bad, size: 12 });
        return;
      }
      const R = S.R, d = P.d, n = S.n;
      // half-aperture: 92 % of the knife-edge height (edge thickness ≥ 0)
      const Hknife = d / 2 >= R ? R : Math.sqrt(d * R - (d * d) / 4);
      const Hap = 0.92 * Math.min(Hknife, R);
      const sag = (h) => R - Math.sqrt(Math.max(R * R - h * h, 0));
      const ext = Math.max(Hap * 1.35, d * 1.2, 2);
      p.setLimits([-ext * 1.2, ext * 1.2], [-ext, ext * 1.12]);
      p.clear();
      p.hline(0, { color: PlotColors.muted, dash: [5, 4], width: 1, alpha: 0.6 });
      const N = 90, xs = [], ys = [];
      for (let i = 0; i <= N; i++) { const h = Hap - (2 * Hap * i) / N; xs.push(d / 2 - sag(h)); ys.push(h); }
      for (let i = 0; i <= N; i++) { const h = -Hap + (2 * Hap * i) / N; xs.push(-d / 2 + sag(h)); ys.push(h); }
      p.poly(xs, ys, { color: "#58a6ff", alpha: 0.22, stroke: PlotColors.blue, strokeWidth: 2.4 });
      // circles of curvature (visible part)
      p.custom((c, pl) => {
        c.strokeStyle = "rgba(182,156,255,0.35)"; c.setLineDash([4, 4]); c.lineWidth = 1;
        for (const cx of [-d / 2 + R, d / 2 - R]) {
          c.beginPath(); c.arc(pl.X(cx), pl.Y(0), R * pl.sx, 0, 2 * Math.PI); c.stroke();
        }
        c.setLineDash([]);
      });
      // thickness arrow
      const yT = Hap + ext * 0.12;
      p.arrow(0, yT, -d / 2, yT, { color: PlotColors.accent3, width: 1.4, head: 6 });
      p.arrow(0, yT, d / 2, yT, { color: PlotColors.accent3, width: 1.4, head: 6 });
      p.segment(-d / 2, 0, -d / 2, yT, { color: PlotColors.accent3, width: 0.8, dash: [2, 3], alpha: 0.6 });
      p.segment(d / 2, 0, d / 2, yT, { color: PlotColors.accent3, width: 0.8, dash: [2, 3], alpha: 0.6 });
      p.text(0, yT + ext * 0.09, `d = ${PM.fmt(d, 2)} mm`, { color: PlotColors.accent3, size: 11, bold: true, align: "center" });
      p.label([`R₁ = +${PM.fmt(R, 2)} mm`, `R₂ = −${PM.fmt(R, 2)} mm`, `aperture ≈ ${PM.fmt(2 * Hap, 1)} mm`, `n = ${PM.fmt(n, 4)}`], "bl", { size: 11, color: "#b69cff" });
    }

    function drawDisp(S) {
      const p = plots.disp;
      if (!S.R) { p.clear(); p.label("No design — dispersion cannot be computed", "tl", { color: PlotColors.bad }); return; }
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < WLS.length; i++) {
        nCurve[i] = nOf(WLS[i]);
        const xi = imagePos(S.s1, focalOf(S.R, nCurve[i], P.d));
        fCurve[i] = xi;
        if (isFinite(xi)) { lo = Math.min(lo, xi); hi = Math.max(hi, xi); }
      }
      const span = isFinite(hi - lo) ? Math.max(hi - lo, 4) : 50;
      const y0 = Math.max(0, (isFinite(lo) ? lo : P.tp) - 0.25 * span), y1 = Math.min(1500, (isFinite(hi) ? hi : P.tp) + 0.25 * span);
      p.setLimits(null, [y0, Math.max(y1, y0 + 5)]);
      p.clear();
      p.custom((c, pl) => {
        const yy = pl.m.t, h = pl._v.ph;
        for (let w = 400; w < 750; w += 5) { c.fillStyle = wlColor(w + 2.5, 0.08); c.fillRect(pl.X(w), yy, pl.X(w + 5) - pl.X(w) + 1, h); }
      });
      p.hline(P.tp, { color: PlotColors.muted, dash: [5, 4], width: 1 });
      p.line(WLS, fCurve, { color: PlotColors.accent, width: 2 });
      p.vline(P.wl, { color: wlColor(P.wl), width: 1.2, dash: [3, 3] });
      p.circle(P.wl, P.tp, 5, { px: true, color: wlColor(P.wl), stroke: PlotColors.text });
      p.label([`n(λ): ${PM.fmt(nOf(400), 3)} (400 nm) → ${PM.fmt(nOf(750), 3)} (750 nm)`, `target: ${PM.fmt(P.tp, 0)} mm (dashed)`], "tr", { size: 11 });
    }

    return {
      reset() { phase = 0; },
      step(dt) { phase += dt * 70; if (phase > 1e6) phase = 0; },
      render() {
        const S = state();
        drawBench(S);
        drawLens(S);
        drawDisp(S);
        M.set("f", S.ok ? PM.fmt(S.f, 2) + " mm" : "—");
        M.set("n", PM.fmt(S.n, 4));
        M.set("R", S.R ? PM.fmt(S.R, 2) + " mm" : "—");
        M.set("R0", S.ok ? PM.fmt(2 * (S.n - 1) * S.f, 2) + " mm" : "—");
        if (S.R) {
          const xb = imagePos(S.s1, focalOf(S.R, nOf(450), P.d)), xr = imagePos(S.s1, focalOf(S.R, nOf(700), P.d));
          M.set("ca", isFinite(xb) && isFinite(xr) ? PM.fmt(xr - xb, 2) + " mm" : "—");
        } else M.set("ca", "—");
        api.setTime(`λ = ${PM.fmt(P.wl, 1)} nm`);
      },
    };
  },
});
