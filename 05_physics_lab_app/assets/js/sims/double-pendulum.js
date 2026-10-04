/* Chaotic double pendulum — 1 to 120 independent double pendulums integrated live with RK4.
 * One shared physical model, two analysis views (selected with a rebuild select):
 *   "Trajectories & sensitivity": trails, configuration-space torus, phase-space separation on a
 *                                  log scale with a Lyapunov fit, relative energy error.
 *   "Ensemble statistics":         θ₂ kernel density estimate, normalised entropy, circular standard
 *                                  deviation, mean tip distance, nearest-neighbour Lyapunov estimate,
 *                                  ensemble energy drift. */
(function () {
  "use strict";

  const H = 1 / 1000;          // RK4 internal step (s)
  const REC_DT = 1 / 30;       // sampling interval of the time series (simulated s)
  const TR = 600;              // trail ring-buffer length (frames)
  const PH = 360;              // configuration-space history length (frames)
  const MAX_TRAILS = 20;       // trails / histories / all arms are drawn only up to this many pendulums
  const MAX_PAIRS = 8;         // separation curves drawn in the trajectory view
  const NG = 150;              // KDE grid points
  const HU = Math.log2(2 * Math.PI); // entropy of the uniform distribution on a circle (bits)
  const TWO_PI = 2 * Math.PI;
  // Defaults that each analysis view loads when it is selected (only if the user has not changed them).
  const DEF = {
    traj: { n: 6, pert: "spread", logd: -3 },
    stats: { n: 60, pert: "noise", logd: -4 },
  };
  let lastView = null;

  const wrap = (a) => a - TWO_PI * Math.round(a / TWO_PI);
  const SUP = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  /** Format 10^x degrees for the logarithmic perturbation slider (x in half-decade steps). */
  function fmtDelta(x) {
    if (x >= 0) return "1°";
    const e = Math.floor(x + 1e-9), m = Math.pow(10, x - e);
    const pow = "10" + String(e).split("").map((c) => SUP[c]).join("");
    return (Math.abs(m - 1) < 1e-6 ? pow : m.toFixed(1) + "×" + pow) + "°";
  }

  /** Secondary (right-hand) y axis: maps [lo,hi] onto the left axis range and draws its ticks. */
  function twin(pl, lo, hi, color, label) {
    const [a, b] = pl.ylim, map = (v) => a + ((v - lo) / (hi - lo)) * (b - a);
    pl._unclip((c) => {
      const x = pl.m.l + pl._v.pw;
      c.font = '11px "Segoe UI", system-ui, sans-serif'; c.fillStyle = color; c.textAlign = "left"; c.textBaseline = "middle";
      const raw = (hi - lo) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
      const st = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
      for (let v = Math.ceil(lo / st) * st; v <= hi + st * 1e-9; v += st) {
        const vv = Math.abs(v) < st * 1e-6 ? 0 : v;
        const txt = vv === 0 ? "0" : (st < 1e-3 || Math.abs(vv) >= 1e5) ? PM.fmt(vv, 1) : vv.toFixed(st >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(st) - 1e-9)));
        c.fillText(txt, x + 5, pl.Y(map(v)));
      }
      c.save(); c.translate(pl.W - 9, pl.m.t + pl._v.ph / 2); c.rotate(-Math.PI / 2);
      c.textAlign = "center"; c.font = '12px "Segoe UI", system-ui, sans-serif'; c.fillText(label, 0, 0); c.restore();
    });
    return map;
  }

  function theory(p) {
    const stats = p.view === "stats";
    return `
<p>${stats
      ? "You are in the <b>Ensemble statistics</b> view: the panels describe how a cloud of nearly identical pendulums spreads over the available states."
      : "You are in the <b>Trajectories &amp; sensitivity</b> view: the panels follow individual pendulums and measure how fast neighbouring trajectories separate."}
Both views integrate exactly the same equations; switch the <i>Analysis view</i> selector to see the other set of diagnostics.</p>

<h4>The physical system</h4>
<p>Each pendulum is the textbook planar double pendulum: a point mass $m_1$ on a massless rigid rod of length $L_1$ hinged at a fixed pivot,
and a point mass $m_2$ hanging from the first mass on a massless rod of length $L_2$. The motion is confined to a vertical plane, gravity $g$ is uniform,
and there is no friction or air drag. The configuration is fixed by the two angles $\\theta_1,\\theta_2$ measured from the downward vertical.
Units are SI: lengths in m, $g$ in m/s², masses in kg with $m_1 = 1\\,$kg fixed (only the ratio $m_2/m_1$ matters for the motion), time in s,
angles in rad (degrees on the sliders). All $N$ pendulums have the same parameters and are released from rest at $(\\theta_1(0),\\theta_2(0))$ plus a tiny
offset added to both angles: evenly spaced, $\\theta_{1,2}\\to\\theta_{1,2}+i\\,\\delta$ for pendulum $i=0,\\dots,N-1$, or random,
$\\theta_{1,2}\\to\\theta_{1,2}+\\xi_i$ with $\\xi_i\\sim\\mathcal N(0,\\delta^2)$. The pendulums do not interact: the ensemble is $N$ copies of one deterministic system.</p>

<h4>Equations being solved</h4>
<p>With the bob positions $x_1=L_1\\sin\\theta_1$, $y_1=-L_1\\cos\\theta_1$, $x_2=x_1+L_2\\sin\\theta_2$, $y_2=y_1-L_2\\cos\\theta_2$ and $\\Delta=\\theta_1-\\theta_2$, the Lagrangian $\\mathcal L=T-V$ is</p>
$$\\mathcal L=\\tfrac12(m_1+m_2)L_1^2\\dot\\theta_1^2+\\tfrac12 m_2L_2^2\\dot\\theta_2^2+m_2L_1L_2\\dot\\theta_1\\dot\\theta_2\\cos\\Delta+(m_1+m_2)gL_1\\cos\\theta_1+m_2gL_2\\cos\\theta_2 .$$
<p>The two Euler–Lagrange equations are linear in $\\ddot\\theta_1,\\ddot\\theta_2$; solving them gives the first-order system integrated here ($\\omega_k=\\dot\\theta_k$):</p>
<div class="callout">
$$\\dot\\omega_1=\\frac{-g(2m_1+m_2)\\sin\\theta_1-m_2g\\sin(\\theta_1-2\\theta_2)-2\\sin\\Delta\\,m_2\\left(\\omega_2^2L_2+\\omega_1^2L_1\\cos\\Delta\\right)}{L_1\\left(2m_1+m_2-m_2\\cos2\\Delta\\right)}$$
$$\\dot\\omega_2=\\frac{2\\sin\\Delta\\left(\\omega_1^2L_1(m_1+m_2)+g(m_1+m_2)\\cos\\theta_1+\\omega_2^2L_2m_2\\cos\\Delta\\right)}{L_2\\left(2m_1+m_2-m_2\\cos2\\Delta\\right)}$$
</div>
<p>The energy $E=\\tfrac12(m_1+m_2)L_1^2\\omega_1^2+\\tfrac12m_2L_2^2\\omega_2^2+m_2L_1L_2\\omega_1\\omega_2\\cos\\Delta-(m_1+m_2)gL_1\\cos\\theta_1-m_2gL_2\\cos\\theta_2$ is exactly conserved.
Chaos means that a small separation $\\delta Z$ between two phase-space points $Z=(\\theta_1,\\omega_1,\\theta_2,\\omega_2)$ grows on average as
$|\\delta Z(t)|\\approx|\\delta Z(0)|\\,e^{\\lambda t}$ with a positive (maximal) Lyapunov exponent $\\lambda$, until it saturates at the size of the accessible region.
Prediction is therefore lost after $t^*\\approx\\lambda^{-1}\\ln(\\Delta_{\\rm sat}/|\\delta Z(0)|)$: each factor 10 of extra initial precision only buys $\\ln 10/\\lambda$ seconds.</p>
<p>For the ensemble, the angle $\\theta_2$ is treated as a circular variable. With the mean resultant length $\\bar R=\\left|\\tfrac1N\\sum_i e^{i\\theta_{2,i}}\\right|$, the circular standard deviation
is $\\sigma_c=\\sqrt{-2\\ln\\bar R}$ ($\\bar R=1\\Rightarrow\\sigma_c=0$). The density $\\rho(\\theta_2)$ is estimated with a Gaussian kernel (Silverman bandwidth $h=(3N/4)^{-1/5}s$) and its normalised entropy is</p>
$$\\frac{H}{H_{\\max}}=\\frac{-\\int_{-\\pi}^{\\pi}\\rho\\log_2\\rho\\,d\\theta_2}{\\log_2(2\\pi)}\\in[0,1],$$
<p>equal to 1 only for the uniform distribution.</p>

<h4>How the simulation solves them</h4>
<ul>
<li><b>Integrator.</b> The $4N$ numbers $(\\theta_1,\\omega_1,\\theta_2,\\omega_2)_i$ are stored in one array and advanced together with the classical fourth-order Runge–Kutta method.
Each animation frame advances the time by (frame time × speed), split into $\\lceil\\Delta t/1\\,\\text{ms}\\rceil$ equal sub-steps, so $h\\le 1$ ms (local error $O(h^5)$, global $O(h^4)$).
The run stops at the chosen duration; all time series are sampled every $1/30$ s of simulated time.</li>
<li><b>Separation (trajectory view).</b> For pendulum $i$ relative to pendulum 0,
$d_i=\\big[\\mathrm{w}(\\Delta\\theta_1)^2+\\Delta\\omega_1^2+\\mathrm{w}(\\Delta\\theta_2)^2+\\Delta\\omega_2^2\\big]^{1/2}$ (angles wrapped to $(-\\pi,\\pi]$, $\\omega$ in rad/s).
Up to ${MAX_PAIRS} pairs are drawn, plus their geometric mean $\\exp\\langle\\ln d_i\\rangle$. The estimate $\\lambda$ is the least-squares slope of $\\langle\\ln d_i\\rangle$ versus $t$
from $t=0.3$ s until the geometric mean reaches $0.3$ (saturation); it is drawn as the dashed line.</li>
<li><b>Configuration space.</b> $(\\theta_1,\\theta_2)$ wrapped to $(-\\pi,\\pi]^2$ — a torus with opposite edges identified. Recent history is shown for up to ${MAX_TRAILS} pendulums.</li>
<li><b>Statistics view.</b> The KDE of wrapped $\\theta_2$ is evaluated on ${NG} grid points and $H/H_{\\max}$ by the trapezoidal rule; $\\langle d\\rangle$ is the mean distance between the
outer bobs over all $N(N-1)/2$ pairs (m). For $\\lambda(t)$ every pendulum is paired at $t=0$ with its nearest neighbour in the embedding
$(\\sin\\theta_1,\\cos\\theta_1,\\omega_1,\\sin\\theta_2,\\cos\\theta_2,\\omega_2)$ and $\\lambda(t)=\\tfrac1t\\langle\\ln[d(t)/d(0)]\\rangle$; after saturation this finite-time estimate decays like $1/t$.</li>
<li><b>Accuracy check.</b> RK4 does not conserve energy exactly, so the energy error is the monitor of numerical accuracy. It is normalised by the potential-energy scale
$E_{\\rm ref}=g[(m_1+m_2)L_1+m_2L_2]$ rather than by $E_0$, which can vanish (e.g. both arms horizontal). The trajectory view plots $\\max_i|E_i(t)-E_i(0)|/E_{\\rm ref}$;
the statistics view shows the ensemble mean ± standard deviation of $E_i(t)-E_i(0)$ (J) and the mean of $|\\Delta E_i|/E_{\\rm ref}$. Typical values stay below $10^{-7}$.</li>
</ul>

<h4>What to try</h4>
<ol>
<li><b>Normal modes.</b> With equal masses and lengths, set $\\theta_1=10^\\circ$, $\\theta_2=14.1^\\circ$ ($\\theta_2=\\sqrt2\\,\\theta_1$): only the slow mode $\\omega_-=\\sqrt{(2-\\sqrt2)g/L}\\approx2.40$ rad/s is excited and
the configuration point moves on a straight line. $\\theta_2=-\\sqrt2\\,\\theta_1$ gives the fast mode $\\omega_+=\\sqrt{(2+\\sqrt2)g/L}\\approx5.79$ rad/s. Separations grow only linearly: $\\lambda\\approx0$.</li>
<li><b>Exponential sensitivity.</b> At $\\theta_1=\\theta_2=150^\\circ$ reduce $\\delta$ from $10^{-1}$° to $10^{-8}$°: the saturation time grows only from about 2.6 s to 15 s — on average $\\approx1.8$ s per decade, i.e. $\\ln10/\\lambda$ with $\\lambda\\approx1.3\\ \\text{s}^{-1}$. A hundred-million-fold better initial precision buys just a few Lyapunov times. The steps are irregular because the stretching rate varies along the orbit.</li>
<li><b>Energy barriers.</b> The upper arm can only go over the top if $E>(m_1+m_2)gL_1-m_2gL_2$ (= 9.81 J for the defaults), the lower arm if $E>-(m_1+m_2)gL_1+m_2gL_2$ (= −9.81 J).
At $\\theta_1=\\theta_2=90^\\circ$ ($E=0$) only the outer arm flips; the motion is chaotic but the upper arm never loops.</li>
<li><b>Mixing.</b> In the statistics view with $N=60$ and noise $10^{-4}$°, the KDE is a spike ($H/H_{\\max}\\approx0$) for several seconds, then within a few Lyapunov times it spreads; $H/H_{\\max}$ approaches 0.9–1 and $\\sigma_c$ exceeds 100°.</li>
<li><b>Heavy or light outer bob.</b> Change $m_2/m_1$ to 0.2 or 3 and compare the estimated $\\lambda$ and the time needed for the ensemble to spread.</li>
</ol>

<h4>Limitations &amp; further reading</h4>
<p>Point masses on massless rods, no friction; a real (compound) double pendulum differs quantitatively. Finite-time, finite-$N$ estimates of $\\lambda$ and of the entropy are noisy and biased
(the KDE smooths the distribution), and after $t\\sim\\lambda^{-1}\\ln(10^{16})$ individual trajectories are no longer accurate even in double precision — only their statistics are.
Further reading: J. R. Taylor, <i>Classical Mechanics</i>, ch. 11–12; S. H. Strogatz, <i>Nonlinear Dynamics and Chaos</i>; T. Shinbrot et al., Am. J. Phys. 60, 491 (1992); E. Ott, <i>Chaos in Dynamical Systems</i>.</p>`;
  }

  App.register({
    id: "double-pendulum",
    category: "classical",
    group: "Mechanics",
    order: 11,
    title: "Chaotic Double Pendulum",
    icon: "🌀",
    subtitle: "Up to 120 double pendulums released from almost identical angles, integrated live: watch them move together, then separate exponentially — deterministic chaos, seen trajectory by trajectory or as ensemble statistics.",
    notes: [{ type: "info", html: "All pendulums obey the same deterministic equations; they differ only by a tiny initial offset $\\delta$. Use the <b>Analysis view</b> selector to switch between following individual trajectories (separation, Lyapunov exponent, energy error) and statistics of the whole ensemble (angle distribution, entropy, circular spread)." }],
    animated: true,
    speed: { min: 0.1, max: 3, value: 1, step: 0.1 },
    controls: [
      { id: "view", type: "select", label: "Analysis view", value: "traj", rebuild: true,
        options: [{ value: "traj", label: "Trajectories & sensitivity" }, { value: "stats", label: "Ensemble statistics" }],
        help: "Same pendulums, different diagnostics. Switching loads the view's default ensemble (6 evenly spaced / 60 randomly perturbed) unless you changed it." },
      { type: "section", label: "Ensemble" },
      { id: "n", type: "slider", label: "Number of pendulums $N$", min: 1, max: 120, step: 1, value: 6 },
      { id: "pert", type: "select", label: "Initial perturbation", value: "spread",
        options: [{ value: "spread", label: "Evenly spaced: θ + i·δ" }, { value: "noise", label: "Random: θ + ξ, ξ ~ N(0, δ²)" }],
        help: "The same offset is added to both angles of pendulum $i$." },
      { id: "logd", type: "slider", label: "Perturbation size $\\delta$", min: -8, max: 0, step: 0.5, value: -3,
        fmt: fmtDelta, help: "Logarithmic slider from 10⁻⁸° to 1°." },
      { type: "section", label: "Initial state (released from rest)" },
      { id: "th1", type: "slider", label: "Upper arm angle $\\theta_1(0)$", min: -180, max: 180, step: 0.5, value: 179, unit: "°" },
      { id: "th2", type: "slider", label: "Lower arm angle $\\theta_2(0)$", min: -180, max: 180, step: 0.5, value: 179, unit: "°" },
      { type: "section", label: "Physical parameters" },
      { id: "g", type: "slider", label: "Gravity $g$", min: 1, max: 20, step: 0.01, value: 9.81, unit: "m/s²" },
      { id: "mr", type: "slider", label: "Mass ratio $m_2/m_1$ ($m_1$ = 1 kg)", min: 0.2, max: 3, step: 0.05, value: 1 },
      { id: "L1", type: "slider", label: "Upper arm length $L_1$", min: 0.5, max: 2, step: 0.05, value: 1, unit: "m" },
      { id: "L2", type: "slider", label: "Lower arm length $L_2$", min: 0.5, max: 2, step: 0.05, value: 1, unit: "m" },
      { id: "Tmax", type: "slider", label: "Run duration", min: 10, max: 120, step: 5, value: 30, unit: "s" },
      { type: "section", label: "Display" },
      { id: "trail", type: "slider", label: "Trail length", min: 0, max: TR, step: 10, value: 200, unit: "frames", live: true,
        help: `Trails and configuration-space histories are drawn when $N \\le ${MAX_TRAILS}$; larger ensembles show every outer bob as a dot.` },
    ],
    theory,

    mount(api) {
      const P = api.params;
      // Load the selected view's defaults when the view was just switched (keep user-changed values).
      if (lastView && lastView !== P.view && DEF[lastView] && DEF[P.view]) {
        const from = DEF[lastView], to = DEF[P.view];
        for (const k in to) if (String(P[k]) === String(from[k])) api.setControl(k, { value: to[k] });
      }
      lastView = P.view;
      const STATS = P.view === "stats";

      const plots = STATS
        ? api.plots([
          { id: "sim", title: "Double-pendulum ensemble (live)", aspect: 0.95, xlim: [-2.2, 2.2], ylim: [-2.2, 2.2], equal: true, axes: false, maxHeight: 540 },
          { id: "kde", title: "Distribution of the lower-arm angle θ₂ (KDE)", aspect: 0.95, xlim: [-Math.PI, Math.PI], ylim: [0, 1], xlabel: "θ₂ (rad)", ylabel: "probability density", maxHeight: 540 },
          { id: "div", title: "Mean bob distance ⟨d⟩ and Lyapunov estimate λ(t)", aspect: 0.62, ylim: [0, 3], xlabel: "t (s)", ylabel: "⟨d⟩ (m)", margin: { r: 56 } },
          { id: "ent", title: "Normalised entropy and circular standard deviation", aspect: 0.62, ylim: [-0.05, 1.1], xlabel: "t (s)", ylabel: "H / H_max", margin: { r: 56 } },
          { id: "en", title: "Energy conservation across the ensemble", span: 2, aspect: 0.26, xlabel: "t (s)", ylabel: "⟨E − E₀⟩ (J)", margin: { l: 70, r: 80 }, minHeight: 200 },
        ])
        : api.plots([
          { id: "sim", title: "Pendulums with trails (live)", aspect: 0.95, xlim: [-2.2, 2.2], ylim: [-2.2, 2.2], equal: true, axes: false, maxHeight: 540 },
          { id: "cfg", title: "Configuration space (θ₁, θ₂) — a torus", aspect: 0.95, xlim: [-Math.PI, Math.PI], ylim: [-Math.PI, Math.PI], xlabel: "θ₁ (rad)", ylabel: "θ₂ (rad)", maxHeight: 540 },
          { id: "div", title: "Phase-space separation from pendulum 1 (log scale)", aspect: 0.62, ylim: [1e-6, 100], ylog: true, xlabel: "t (s)", ylabel: "separation d(t)" },
          { id: "en", title: "Relative energy error |ΔE| / E_ref (worst pendulum)", aspect: 0.62, ylim: [1e-16, 1e-6], ylog: true, xlabel: "t (s)", ylabel: "|ΔE| / E_ref" },
        ]);
      const M = STATS
        ? api.metrics([
          { id: "t", label: "Time" },
          { id: "ent", label: "Normalised entropy $H/H_{\\max}$" },
          { id: "sig", label: "Circular std. $\\sigma_c(\\theta_2)$" },
          { id: "lyap", label: "Lyapunov estimate $\\lambda(t)$" },
          { id: "drift", label: "Mean $|\\Delta E|/E_{\\rm ref}$" },
        ])
        : api.metrics([
          { id: "t", label: "Time" },
          { id: "lyap", label: "Fitted Lyapunov exponent $\\lambda$" },
          { id: "tl", label: "Lyapunov time $1/\\lambda$" },
          { id: "E", label: "Energy $E_0$ of pendulum 1" },
          { id: "err", label: "Max. $|\\Delta E|/E_{\\rm ref}$" },
        ]);
      const enTitle = STATS ? plots.en.container.querySelector(".plot-title") : null;

      // ------------------------------------------------------------ state
      let n, y, ws, t, finished, m1 = 1, m2, L1, L2, g, Eref, E0, colors;
      let cap, nrec, lastRec, T;
      // trajectory view
      let pairIdx, D, LnM, EE, maxErr, lnBuf, fit;
      let trX, trY, cfA, cfB, head, sx, sy;
      // statistics view
      let nn, d0, wrapped, tipX, tipY, Dv, Ly, En, Sg, Em, Es, Dr, cur, kdeTop, lamLo, lamHi, sigHi, eHalf, drHi;
      const grid = PM.linspace(-Math.PI, Math.PI, NG), dens = new Float64Array(NG);

      function deriv(_t, s, out) {
        const a = 2 * m1 + m2, mm = m1 + m2;
        for (let k = 0; k < s.length; k += 4) {
          const t1 = s[k], w1 = s[k + 1], t2 = s[k + 2], w2 = s[k + 3];
          const d = t1 - t2, sd = Math.sin(d), cd = Math.cos(d);
          const den = a - m2 * Math.cos(2 * d);
          out[k] = w1;
          out[k + 1] = (-g * a * Math.sin(t1) - m2 * g * Math.sin(t1 - 2 * t2) - 2 * sd * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cd)) / (L1 * den);
          out[k + 2] = w2;
          out[k + 3] = (2 * sd * (w1 * w1 * L1 * mm + g * mm * Math.cos(t1) + w2 * w2 * L2 * m2 * cd)) / (L2 * den);
        }
      }
      function energy(k) {
        const t1 = y[k], w1 = y[k + 1], t2 = y[k + 2], w2 = y[k + 3];
        return 0.5 * (m1 + m2) * L1 * L1 * w1 * w1 + 0.5 * m2 * L2 * L2 * w2 * w2 + m2 * L1 * L2 * w1 * w2 * Math.cos(t1 - t2)
          - (m1 + m2) * g * L1 * Math.cos(t1) - m2 * g * L2 * Math.cos(t2);
      }
      /** Phase-space distance between pendulums a and b (angles wrapped, ω in rad/s). */
      function dist(a, b) {
        const p = 4 * a, q = 4 * b;
        const e1 = wrap(y[p] - y[q]), e2 = y[p + 1] - y[q + 1], e3 = wrap(y[p + 2] - y[q + 2]), e4 = y[p + 3] - y[q + 3];
        return Math.sqrt(e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4);
      }
      /** Distance in the 6-D embedding (sinθ₁, cosθ₁, ω₁, sinθ₂, cosθ₂, ω₂). */
      function featDist(a, b) {
        const p = 4 * a, q = 4 * b;
        const d1 = Math.sin(y[p]) - Math.sin(y[q]), d2 = Math.cos(y[p]) - Math.cos(y[q]), d3 = y[p + 1] - y[q + 1];
        const d4 = Math.sin(y[p + 2]) - Math.sin(y[q + 2]), d5 = Math.cos(y[p + 2]) - Math.cos(y[q + 2]), d6 = y[p + 3] - y[q + 3];
        return Math.sqrt(d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6);
      }
      const tipXof = (i) => L1 * Math.sin(y[4 * i]) + L2 * Math.sin(y[4 * i + 2]);
      const tipYof = (i) => -L1 * Math.cos(y[4 * i]) - L2 * Math.cos(y[4 * i + 2]);

      // ------------------------------------------------------------ trajectory-view analysis
      let curLn = 0;
      const curD = new Float64Array(MAX_PAIRS);
      function analyseTraj() {
        if (n > 1) {
          let s = 0;
          for (let i = 1; i < n; i++) s += Math.log(Math.max(dist(0, i), 1e-300));
          curLn = s / (n - 1);
          for (let k = 0; k < pairIdx.length; k++) curD[k] = Math.max(dist(0, pairIdx[k]), 1e-300);
        }
        let e = 0;
        for (let i = 0; i < n; i++) e = Math.max(e, Math.abs(energy(4 * i) - E0[i]) / Eref);
        maxErr = Math.max(maxErr, e);
        return e;
      }
      /** Least-squares slope of ⟨ln d⟩ between t = 0.3 s and saturation (geometric mean 0.3). */
      function fitLyapunov() {
        fit = null;
        if (n < 2 || nrec < 20) return;
        const lnSat = Math.log(0.3);
        let sx0 = 0, sy0 = 0, sxx = 0, sxy = 0, c = 0, t0 = NaN, t1 = NaN;
        for (let k = 0; k < nrec; k++) {
          if (LnM[k] > lnSat) break;
          if (T[k] < 0.3) continue;
          const xx = T[k], yy = LnM[k];
          sx0 += xx; sy0 += yy; sxx += xx * xx; sxy += xx * yy; c++;
          if (!isFinite(t0)) t0 = xx;
          t1 = xx;
        }
        if (c < 15 || t1 - t0 < 0.5) return;
        const lam = (c * sxy - sx0 * sy0) / (c * sxx - sx0 * sx0), b = (sy0 - lam * sx0) / c;
        fit = { lam, b, t0, t1 };
      }

      // ------------------------------------------------------------ statistics-view analysis
      function analyseStats() {
        // circular standard deviation of θ₂
        let ms = 0, mc = 0;
        for (let i = 0; i < n; i++) { const th = y[4 * i + 2]; ms += Math.sin(th); mc += Math.cos(th); wrapped[i] = wrap(th); }
        ms /= n; mc /= n;
        const R = PM.clamp(Math.hypot(ms, mc), 1e-15, 1);
        const sig = (Math.sqrt(-2 * Math.log(R)) * 180) / Math.PI;
        // KDE (Silverman bandwidth) and normalised entropy
        let ent = 0;
        dens.fill(0);
        let mean = 0; for (let i = 0; i < n; i++) mean += wrapped[i]; mean /= n;
        let v = 0; for (let i = 0; i < n; i++) v += (wrapped[i] - mean) * (wrapped[i] - mean);
        if (n > 1 && Math.sqrt(v / n) >= 1e-10) {
          const h = Math.pow(0.75 * n, -0.2) * Math.sqrt(v / (n - 1)), cut = 8 * h;
          const norm = 1 / (n * h * Math.sqrt(2 * Math.PI)), inv = 1 / (2 * h * h);
          for (let i = 0; i < n; i++) {
            const xi = wrapped[i];
            for (let q = 0; q < NG; q++) { const dx = grid[q] - xi; if (dx > cut || dx < -cut) continue; dens[q] += Math.exp(-dx * dx * inv); }
          }
          for (let q = 0; q < NG; q++) dens[q] *= norm;
          let hr = 0, px = NaN, pf = 0;
          for (let q = 0; q < NG; q++) {
            if (!(dens[q] > 1e-15)) continue;
            const f = dens[q] * Math.log2(dens[q]);
            if (isFinite(px)) hr += 0.5 * (f + pf) * (grid[q] - px);
            px = grid[q]; pf = f;
          }
          ent = PM.clamp(-hr / HU, 0, 1);
        }
        // mean distance between outer bobs, all pairs
        for (let i = 0; i < n; i++) { tipX[i] = tipXof(i); tipY[i] = tipYof(i); }
        let ds = 0, np = 0;
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { ds += Math.hypot(tipX[i] - tipX[j], tipY[i] - tipY[j]); np++; }
        const div = np ? ds / np : 0;
        // nearest-neighbour Lyapunov estimate
        let lam = NaN;
        if (t > 0 && n > 1) {
          let s = 0, c = 0;
          for (let i = 0; i < n; i++) {
            const dd = featDist(i, nn[i]);
            if (dd > 1e-15 && d0[i] > 1e-15) { s += Math.log(dd / d0[i]); c++; }
          }
          if (c) lam = s / c / t;
        }
        // energy drift: mean ± std of ΔE_i and mean |ΔE_i|/E_ref
        let em = 0, e2 = 0, dr = 0;
        for (let i = 0; i < n; i++) { const de = energy(4 * i) - E0[i]; em += de; e2 += de * de; dr += Math.abs(de) / Eref; }
        em /= n;
        const es = Math.sqrt(Math.max(e2 / n - em * em, 0));
        return { sig, ent, div, lam, em, es, dr: dr / n };
      }

      function record(force) {
        const e = STATS ? 0 : analyseTraj();
        if (STATS) cur = analyseStats();
        if (!force && t - lastRec < REC_DT - 1e-9) return;
        if (nrec >= cap) return;
        lastRec = t;
        const k = nrec++;
        T[k] = t;
        if (STATS) {
          Dv[k] = cur.div; Ly[k] = cur.lam; En[k] = cur.ent; Sg[k] = cur.sig; Em[k] = cur.em; Es[k] = cur.es; Dr[k] = cur.dr;
          if (isFinite(cur.lam)) { lamHi = Math.max(lamHi, cur.lam * 1.15); lamLo = Math.min(lamLo, cur.lam * 1.15); }
          sigHi = Math.max(sigHi, cur.sig * 1.15);
          eHalf = Math.max(eHalf, (Math.abs(cur.em) + cur.es) * 1.3);
          drHi = Math.max(drHi, cur.dr * 1.3);
        } else {
          LnM[k] = curLn;
          for (let p = 0; p < pairIdx.length; p++) D[p][k] = curD[p];
          EE[k] = Math.max(e, 1e-17);
        }
      }
      function sampleFrame() {
        if (STATS || n > MAX_TRAILS) return;
        const hh = head % TR, hc = head % PH;
        for (let i = 0; i < n; i++) {
          trX[i][hh] = tipXof(i); trY[i][hh] = tipYof(i);
          cfA[i][hc] = wrap(y[4 * i]); cfB[i][hc] = wrap(y[4 * i + 2]);
        }
        head++;
      }

      const mapArr = (src, len, f, out) => { for (let i = 0; i < len; i++) out[i] = f(isFinite(src[i]) ? src[i] : 0); return out; };
      let scratchA, scratchB;

      // ------------------------------------------------------------ render helpers
      function drawPendulums() {
        const ps = plots.sim;
        ps.clear();
        if (n <= MAX_TRAILS) {
          const tl = Math.min(P.trail, head, TR);
          if (tl > 1) {
            for (let i = 0; i < n; i++) {
              for (let k = 0; k < tl; k++) { const idx = (head - tl + k) % TR; sx[k] = trX[i][idx]; sy[k] = trY[i][idx]; }
              ps.line(sx.subarray(0, tl), sy.subarray(0, tl), { color: colors[i], width: 1.2, alpha: 0.45 });
            }
          }
          for (let i = 0; i < n; i++) {
            const t1 = y[4 * i], t2 = y[4 * i + 2];
            const x1 = L1 * Math.sin(t1), y1 = -L1 * Math.cos(t1), x2 = x1 + L2 * Math.sin(t2), y2 = y1 - L2 * Math.cos(t2);
            ps.line([0, x1, x2], [0, y1, y2], { color: colors[i], width: 2.2, alpha: 0.9 });
            ps.circle(x1, y1, 5, { px: true, color: colors[i] });
            ps.circle(x2, y2, 6.5 * Math.cbrt(m2), { px: true, color: colors[i], stroke: "#0f151c" });
          }
        } else {
          // large ensemble: a subset of arms (faint) + every outer bob as a dot
          const ns = 16;
          for (let k = 0; k < ns; k++) {
            const i = Math.round((k * (n - 1)) / (ns - 1));
            const t1 = y[4 * i], t2 = y[4 * i + 2];
            const x1 = L1 * Math.sin(t1), y1 = -L1 * Math.cos(t1), x2 = x1 + L2 * Math.sin(t2), y2 = y1 - L2 * Math.cos(t2);
            ps.line([0, x1, x2], [0, y1, y2], { color: colors[i], width: 1.3, alpha: 0.4 });
          }
          for (let i = 0; i < n; i++) { tipX[i] = tipXof(i); tipY[i] = tipYof(i); }
          ps.points(tipX, tipY, { colors, size: 3.2, alpha: 0.9 });
        }
        ps.circle(0, 0, 4, { px: true, color: PlotColors.text });
        ps.label([`t = ${PM.fmt(t, 2)} s`, `N = ${n} pendulum${n === 1 ? "" : "s"}`], "tl");
      }

      function renderTraj() {
        // configuration space
        const pc = plots.cfg;
        pc.clear();
        pc.hline(0, { color: PlotColors.muted, alpha: 0.3, width: 1 });
        pc.vline(0, { color: PlotColors.muted, alpha: 0.3, width: 1 });
        if (n <= MAX_TRAILS) {
          const hl = Math.min(head, PH);
          pc.custom((c, p) => {
            for (let i = 0; i < n; i++) {
              c.fillStyle = colors[i]; c.globalAlpha = 0.55;
              for (let k = 0; k < hl; k++) { const idx = (head - hl + k) % PH; c.fillRect(p.X(cfA[i][idx]) - 1, p.Y(cfB[i][idx]) - 1, 2, 2); }
            }
            c.globalAlpha = 1;
          });
        }
        for (let i = 0; i < n; i++) { tipX[i] = wrap(y[4 * i]); tipY[i] = wrap(y[4 * i + 2]); }
        pc.points(tipX, tipY, { colors, size: n > MAX_TRAILS ? 3 : 4.5, stroke: n > MAX_TRAILS ? null : "#0f151c" });
        pc.label("opposite edges are identified", "br", { size: 11, color: PlotColors.muted });

        // separation
        const pd = plots.div;
        pd.clear();
        if (n < 2) pd.label("Separation needs at least 2 pendulums", "tl");
        else if (nrec > 1) {
          const Ts = T.subarray(0, nrec);
          for (let p = 0; p < pairIdx.length; p++) pd.line(Ts, D[p].subarray(0, nrec), { color: colors[pairIdx[p]], width: 1.2, alpha: 0.6 });
          for (let k = 0; k < nrec; k++) lnBuf[k] = Math.exp(LnM[k]);
          pd.line(Ts, lnBuf.subarray(0, nrec), { color: PlotColors.text, width: 2.2 });
          fitLyapunov();
          if (fit) pd.line([fit.t0, fit.t1], [Math.exp(fit.b + fit.lam * fit.t0), Math.exp(fit.b + fit.lam * fit.t1)], { color: PlotColors.accent3, width: 2, dash: [7, 4] });
          pd.legend([
            { label: "pairs (1, i)", color: PlotColors.muted },
            { label: "geometric mean", color: PlotColors.text },
            { label: fit ? `fit ∝ exp(λt), λ = ${PM.fmt(fit.lam, 2)} s⁻¹` : "fit ∝ exp(λt): not enough data yet", color: PlotColors.accent3, dash: [7, 4] },
          ], "br");
        }
        // energy error
        const pe = plots.en;
        const top = Math.pow(10, Math.max(-6, Math.ceil(Math.log10(Math.max(maxErr, 1e-300)) + 1)));
        if (top !== pe.ylim[1]) pe.setLimits(null, [1e-16, top]);
        pe.clear();
        if (nrec > 1) pe.line(T.subarray(0, nrec), EE.subarray(0, nrec), { color: PlotColors.good, width: 1.4 });

        M.set("t", `${PM.fmt(t, 2)} / ${PM.fmt(P.Tmax, 0)} s`);
        M.set("lyap", fit ? PM.fmt(fit.lam, 2) + " s⁻¹" : "—");
        M.set("tl", fit && fit.lam > 1e-3 ? PM.fmt(1 / fit.lam, 2) + " s" : "—");
        M.set("E", PM.fmt(Math.abs(E0[0]) < 1e-9 * Eref ? 0 : E0[0], 3) + " J");
        M.set("err", PM.fmt(maxErr, 2));
      }

      function renderStats() {
        // KDE
        const pk = plots.kde;
        let mx = 0; for (let q = 0; q < NG; q++) mx = Math.max(mx, dens[q]);
        const target = PM.clamp(mx * 1.2, 0.4, 8);
        kdeTop += (target - kdeTop) * 0.15;
        pk.setLimits(null, [0, kdeTop]);
        pk.clear();
        pk.hline(1 / (2 * Math.PI), { color: PlotColors.muted, dash: [5, 4], width: 1 });
        pk.text(Math.PI * 0.95, 1 / (2 * Math.PI), "uniform", { dy: -9, align: "right", color: PlotColors.muted, size: 11 });
        if (n > 1) {
          pk.fill(grid, dens, 0, { color: PlotColors.accent, alpha: 0.25 });
          pk.line(grid, dens, { color: PlotColors.accent, width: 2 });
        }
        for (let i = 0; i < n; i++) scratchB[i] = kdeTop * 0.015;
        pk.points(wrapped, scratchB.subarray(0, n), { colors, size: 2.4, alpha: 0.9 });
        pk.label([`H/H_max = ${PM.fmt(cur.ent, 3)}`, `σc = ${PM.fmt(cur.sig, 1)}°`], "tr");
        if (n < 2) pk.label("A distribution needs at least 2 pendulums", "tl", { color: PlotColors.muted });
        else if (mx > kdeTop * 1.02) pk.label(`peak: ${PM.fmt(mx, 1)}`, "tl", { color: PlotColors.muted });

        const Ts = T.subarray(0, nrec);
        // divergence + Lyapunov
        const pd = plots.div;
        const dTop = (L1 + L2) * 1.5;
        if (pd.ylim[1] !== dTop) pd.setLimits(null, [0, dTop]);
        pd.clear();
        const mapL = twin(pd, lamLo, lamHi, PlotColors.accent3, "λ (1/s)");
        if (lamLo < 0) pd.hline(mapL(0), { color: PlotColors.accent3, dash: [2, 5], width: 0.8, alpha: 0.6 });
        if (nrec > 1) {
          pd.line(Ts, Dv.subarray(0, nrec), { color: PlotColors.good, width: 2 });
          pd.line(Ts, mapArr(Ly, nrec, mapL, scratchA).subarray(0, nrec), { color: PlotColors.accent3, width: 2, dash: [7, 4] });
        }
        pd.vline(t, { color: PlotColors.text, dash: [2, 4], width: 0.8, alpha: 0.5 });
        pd.legend([{ label: "⟨d⟩ mean bob distance", color: PlotColors.good }, { label: "λ(t) nearest neighbours (right axis)", color: PlotColors.accent3, dash: [7, 4] }], "tl");

        // entropy + circular std
        const pe = plots.ent;
        pe.clear();
        const mapS = twin(pe, 0, sigHi, PlotColors.pink, "circular σc (°)");
        pe.hline(1, { color: PlotColors.muted, dash: [5, 4], width: 1 });
        if (nrec > 1) {
          pe.line(Ts, En.subarray(0, nrec), { color: PlotColors.accent, width: 2 });
          pe.line(Ts, mapArr(Sg, nrec, mapS, scratchA).subarray(0, nrec), { color: PlotColors.pink, width: 2, dash: [7, 4] });
        }
        pe.vline(t, { color: PlotColors.text, dash: [2, 4], width: 0.8, alpha: 0.5 });
        pe.legend([{ label: "H / H_max", color: PlotColors.accent }, { label: "σc (right axis)", color: PlotColors.pink, dash: [7, 4] }], "br");

        // energy
        const pn = plots.en;
        pn.setLimits(null, [-eHalf, eHalf]);
        pn.clear();
        const mapD = twin(pn, 0, drHi, PlotColors.accent2, "|ΔE| / E_ref");
        if (nrec > 1) {
          for (let i = 0; i < nrec; i++) { scratchA[i] = Em[i] + Es[i]; scratchB[i] = Em[i] - Es[i]; }
          pn.fill(Ts, scratchA.subarray(0, nrec), scratchB.subarray(0, nrec), { color: PlotColors.accent3, alpha: 0.15 });
          pn.line(Ts, Em.subarray(0, nrec), { color: PlotColors.accent3, width: 2 });
          pn.line(Ts, mapArr(Dr, nrec, mapD, scratchA).subarray(0, nrec), { color: PlotColors.accent2, width: 2, dash: [7, 4] });
        }
        pn.vline(t, { color: PlotColors.text, dash: [2, 4], width: 0.8, alpha: 0.5 });
        pn.legend([{ label: "⟨E − E₀⟩ ± std", color: PlotColors.accent3 }, { label: "mean |ΔE| / E_ref (right axis)", color: PlotColors.accent2, dash: [7, 4] }], "tl");

        M.set("t", `${PM.fmt(t, 2)} / ${PM.fmt(P.Tmax, 0)} s`);
        M.set("ent", n > 1 ? PM.fmt(cur.ent, 3) : "—");
        M.set("sig", PM.fmt(cur.sig, 1) + "°");
        M.set("lyap", isFinite(cur.lam) ? PM.fmt(cur.lam, 3) + " s⁻¹" : "—");
        M.set("drift", PM.fmt(cur.dr, 2));
      }

      return {
        reset() {
          n = Math.round(P.n); t = 0; finished = false;
          g = P.g; m2 = P.mr; L1 = P.L1; L2 = P.L2;
          Eref = g * ((m1 + m2) * L1 + m2 * L2);
          const th1 = (P.th1 * Math.PI) / 180, th2 = (P.th2 * Math.PI) / 180, del = (Math.pow(10, P.logd) * Math.PI) / 180;
          const rng = new PM.RNG(42);
          y = new Float64Array(4 * n); ws = null;
          for (let i = 0; i < n; i++) {
            const off = P.pert === "noise" ? rng.gauss(0, del) : i * del;
            y[4 * i] = th1 + off; y[4 * i + 2] = th2 + off;
          }
          E0 = new Float64Array(n); for (let i = 0; i < n; i++) E0[i] = energy(4 * i);
          colors = Array.from({ length: n }, (_, i) => colormap(STATS ? "plasma" : "turbo", n === 1 ? 0.2 : 0.08 + (0.84 * i) / (n - 1)));
          tipX = new Float64Array(n); tipY = new Float64Array(n); wrapped = new Float64Array(n);
          // time series storage
          cap = Math.ceil(P.Tmax / REC_DT) + 4; nrec = 0; lastRec = -1;
          T = new Float64Array(cap); scratchA = new Float64Array(Math.max(cap, n)); scratchB = new Float64Array(Math.max(cap, n));
          const R = (L1 + L2) * 1.08;
          plots.sim.setLimits([-R, R], [-R, R]);
          if (STATS) {
            Dv = new Float64Array(cap); Ly = new Float64Array(cap); En = new Float64Array(cap); Sg = new Float64Array(cap);
            Em = new Float64Array(cap); Es = new Float64Array(cap); Dr = new Float64Array(cap);
            // nearest neighbours in the 6-D embedding at t = 0
            nn = new Int32Array(n); d0 = new Float64Array(n);
            for (let i = 0; i < n; i++) {
              let best = Infinity, bj = i;
              for (let j = 0; j < n; j++) { if (j === i) continue; const dd = featDist(i, j); if (dd < best) { best = dd; bj = j; } }
              nn[i] = bj; d0[i] = isFinite(best) ? best : 0;
            }
            kdeTop = 1; lamLo = -0.5; lamHi = 2; sigHi = 30; drHi = 1e-9;
            eHalf = 1e-9 * Eref;
            enTitle.textContent = `Energy conservation across the ensemble (⟨E₀⟩ = ${PM.fmt(PM.mean(E0), 3)} J, E_ref = ${PM.fmt(Eref, 3)} J)`;
            for (const id of ["div", "ent", "en"]) plots[id].setLimits([0, P.Tmax]);
          } else {
            const np = Math.min(MAX_PAIRS, n - 1);
            pairIdx = [];
            for (let k = 0; k < np; k++) pairIdx.push(np === 1 ? n - 1 : 1 + Math.round((k * (n - 2)) / (np - 1)));
            pairIdx = [...new Set(pairIdx)];
            D = pairIdx.map(() => new Float64Array(cap));
            LnM = new Float64Array(cap); EE = new Float64Array(cap); lnBuf = new Float64Array(cap);
            maxErr = 0; fit = null;
            // y-range of the separation plot: from a decade below the smallest initial distance
            let dmin = Infinity;
            for (let i = 1; i < n; i++) dmin = Math.min(dmin, dist(0, i));
            const lo = isFinite(dmin) && dmin > 0 ? Math.pow(10, Math.floor(Math.log10(dmin)) - 1) : 1e-6;
            plots.div.setLimits([0, P.Tmax], [Math.max(lo, 1e-14), 100]);
            plots.en.setLimits([0, P.Tmax], [1e-16, 1e-6]);
            // trails and configuration-space histories
            head = 0;
            if (n <= MAX_TRAILS) {
              trX = []; trY = []; cfA = []; cfB = [];
              for (let i = 0; i < n; i++) { trX.push(new Float64Array(TR)); trY.push(new Float64Array(TR)); cfA.push(new Float64Array(PH)); cfB.push(new Float64Array(PH)); }
              sx = new Float64Array(TR); sy = new Float64Array(TR);
            }
          }
          sampleFrame();
          record(true);
          if (!api.isPlaying) api.play();
        },
        step(dt) {
          if (finished) return;
          dt = Math.min(dt, P.Tmax - t);
          if (dt > 0) {
            const sub = Math.max(1, Math.ceil(dt / H)), h = dt / sub;
            for (let s = 0; s < sub; s++) { ws = PM.rk4(deriv, t, y, h, ws); t += h; }
          }
          const end = P.Tmax - t < 1e-9;
          if (end) t = P.Tmax;
          sampleFrame();
          record(end);
          if (end) { finished = true; api.pause(); }
        },
        render() {
          drawPendulums();
          if (STATS) renderStats(); else renderTraj();
          api.setTime(finished ? `t = ${PM.fmt(t, 2)} s · end of run (Restart to run again)` : `t = ${PM.fmt(t, 2)} s`);
        },
      };
    },
  });
})();
