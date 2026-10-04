/* Series RLC / RC / RL circuits — RK4 (sub-stepped) integration of Kirchhoff's voltage law with the exact analytic
 * solution overlaid; DC step, capacitor discharge and AC sine drive with resonance curve, phase and phasor diagram. */
(function () {
  "use strict";
  const IND = "#ffd166";
  const fmtNum = (x) => (x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toPrecision(2));
  const fmtR = (r) => (r >= 1000 ? fmtNum(r / 1000) + " kΩ" : fmtNum(r) + " Ω");
  const fmtL = (mH) => (mH >= 1000 ? fmtNum(mH / 1000) + " H" : fmtNum(mH) + " mH");
  const fmtC = (uF) => (uF >= 1000 ? fmtNum(uF / 1000) + " mF" : fmtNum(uF) + " μF");
  const fmtW = (w) => (w >= 1e4 ? PM.fmt(w, 3) : w.toFixed(w >= 100 ? 0 : w >= 10 ? 1 : 2));

  function txt(c, s, x, y, o) {
    o = o || {};
    c.font = (o.bold ? "600 " : "") + (o.size || 12) + 'px "Segoe UI", system-ui, sans-serif';
    c.textAlign = o.align || "left"; c.textBaseline = o.base || "middle";
    if (o.bg) { const w = c.measureText(s).width, h = (o.size || 12) + 6, bx = x - (c.textAlign === "center" ? w / 2 : c.textAlign === "right" ? w : 0) - 4;
      c.globalAlpha = 0.85; c.fillStyle = o.bg; c.fillRect(bx, y - h / 2, w + 8, h); c.globalAlpha = 1; }
    c.fillStyle = o.color || "#e6edf3"; c.fillText(s, x, y);
  }
  function arrowPx(c, x0, y0, x1, y1, col, w, head) {
    const a = Math.atan2(y1 - y0, x1 - x0), h = head || 8;
    c.strokeStyle = c.fillStyle = col; c.lineWidth = w || 2;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    c.beginPath(); c.moveTo(x1, y1);
    c.lineTo(x1 - h * Math.cos(a - 0.42), y1 - h * Math.sin(a - 0.42));
    c.lineTo(x1 - h * Math.cos(a + 0.42), y1 - h * Math.sin(a + 0.42)); c.closePath(); c.fill();
  }
  /** Range with hysteresis (grows at once, shrinks when the data use < 45 % of it). */
  function RangeScaler() {
    let lo = 0, hi = 0;
    return {
      fit(mn, mx) {
        mn = Math.min(mn, 0); mx = Math.max(mx, 0);
        const span = Math.max(mx - mn, 1e-15), pad = 0.1 * span, tl = mn - (mn < 0 ? pad : 0), th = mx + (mx > 0 ? pad : 0);
        if (tl < lo || th > hi || th - tl < 0.45 * (hi - lo)) { lo = tl; hi = th; }
        if (hi - lo < 1e-15) hi = lo + 1e-15;
        return [lo, hi];
      },
      reset() { lo = hi = 0; },
    };
  }
  function minmax(arrs, n, i0) {
    let mn = Infinity, mx = -Infinity;
    for (const a of arrs) for (let i = i0 || 0; i < n; i++) { const v = a[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    return isFinite(mn) ? [mn, mx] : [0, 1];
  }
  function firstIdx(T, n, t0) { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (T[m] < t0) lo = m + 1; else hi = m; } return lo; }

  App.register({
    id: "rlc-circuit",
    category: "classical",
    group: "Electromagnetism",
    order: 24,
    title: "RLC Circuits: Transients & Resonance",
    icon: "🔌",
    subtitle: "A series RLC (or RC / RL) circuit integrated live with RK4 and checked against the exact solution: charging and discharging transients (under-, critically and over-damped), and the driven AC response with resonance curve, phase and rotating phasors.",
    notes: [{ type: "info", html: "Moving dots show the current (speed ∝ $I$, they reverse when $I$ changes sign); the capacitor shows its charge, the inductor its magnetic field, the resistor glows with the dissipated power. Solid curves: RK4, dashed: analytic solution. Component sliders are logarithmic." }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "src", type: "select", label: "Source", value: "dc", rebuild: true, options: [
        { value: "dc", label: "DC step: switch on V₀ at t = 0 (charging)" },
        { value: "dis", label: "Discharge (source removed at t = 0)" },
        { value: "ac", label: "AC sine drive V₀ sin ωt" },
      ] },
      { id: "cfg", type: "select", label: "Circuit", value: "rlc", rebuild: true, options: [
        { value: "rlc", label: "Series RLC" },
        { value: "rc", label: "RC only (no inductor, L → 0)" },
        { value: "rl", label: "RL only (no capacitor)" },
      ] },
      { type: "section", label: "Components" },
      { id: "lR", type: "slider", label: "Resistance $R$", min: -1, max: 3, step: 0.001, value: 1, fmt: (v) => fmtR(Math.pow(10, v)) },
      { id: "lL", type: "slider", label: "Inductance $L$", min: -1, max: 3, step: 0.001, value: 2, fmt: (v) => fmtL(Math.pow(10, v)), visibleIf: (p) => p.cfg !== "rc" },
      { id: "lC", type: "slider", label: "Capacitance $C$", min: -1, max: 3, step: 0.001, value: 2, fmt: (v) => fmtC(Math.pow(10, v)), visibleIf: (p) => p.cfg !== "rl" },
      { id: "crit", type: "button", label: "Set critical damping  R = 2√(L/C)", visibleIf: (p) => p.cfg === "rlc" },
      { type: "section", label: "Source" },
      { id: "V0", type: "slider", label: "Voltage $V_0$", min: 1, max: 20, step: 0.5, value: 10, unit: "V",
        help: "DC: battery voltage. Discharge: initial capacitor voltage (RL: initial current $V_0/R$). AC: amplitude." },
      { id: "lw", type: "slider", label: "Drive frequency $\\omega/\\omega_0$", min: -1, max: 1, step: 0.002, value: -0.1, live: true,
        fmt: (v) => Math.pow(10, v).toFixed(3), visibleIf: (p) => p.src === "ac" },
      { id: "sweep", type: "checkbox", label: "Sweep the frequency slowly (0.1 → 10 and back)", value: false, live: true, visibleIf: (p) => p.src === "ac",
        help: "Measured amplitudes (dots) are added to the resonance curve. A fast sweep through a sharp resonance lags behind the steady-state curve." },
      { type: "section", label: "Display" },
      { id: "an", type: "checkbox", label: "Overlay the analytic solution (dashed)", value: true, live: true },
    ],
    theory: `
    <h4>The physical system</h4>
    <p>A resistor $R$ (Ω), an inductor $L$ (H) and a capacitor $C$ (F) are connected in series with a voltage source $V(t)$.
    The components are ideal and lumped (no wire resistance, no stray capacitance, no radiation), so the same current $I$ flows
    through all of them and the state of the circuit is fully described by the capacitor charge $q$ and the current $I=dq/dt$.
    Three sources are available: a DC step ($V=V_0$ switched on at $t=0$ with $q=I=0$), a discharge ($V=0$, the capacitor starts
    charged to $q_0=CV_0$; for the RL circuit the inductor starts with current $V_0/R$), and an AC drive $V=V_0\\sin\\omega t$.
    The RC and RL circuits are the limits $L\\to0$ and "capacitor replaced by a wire" ($1/C\\to0$); they are coded as separate
    first-order equations, so no division by zero ever occurs. Charges are plotted in mC, times in ms, energies in mJ.</p>

    <h4>Equations being solved</h4>
    <p>Kirchhoff's voltage law — the source voltage equals the sum of the drops $V_R=RI$, $V_L=L\\,dI/dt$, $V_C=q/C$ — gives a
    driven, damped harmonic oscillator:</p>
    <div class="callout">$$L\\ddot q+R\\dot q+\\frac{q}{C}=V(t)\\quad\\Longleftrightarrow\\quad
      \\ddot q+2\\zeta\\omega_0\\dot q+\\omega_0^2q=\\frac{V(t)}{L},\\qquad
      \\omega_0=\\frac{1}{\\sqrt{LC}},\\quad \\zeta=\\frac{R}{2}\\sqrt{\\frac{C}{L}} .$$</div>
    <p><b>Transients.</b> With $\\alpha=R/2L=\\zeta\\omega_0$ the homogeneous solutions $e^{st}$ have $s=-\\alpha\\pm\\sqrt{\\alpha^2-\\omega_0^2}$:</p>
    <ul>
      <li>underdamped ($\\zeta&lt;1$): $u(t)=e^{-\\alpha t}\\big[u_0\\cos\\omega_dt+\\frac{\\dot u_0+\\alpha u_0}{\\omega_d}\\sin\\omega_dt\\big]$, $\\omega_d=\\omega_0\\sqrt{1-\\zeta^2}$;</li>
      <li>critically damped ($\\zeta=1$): $u(t)=e^{-\\alpha t}\\big[u_0+(\\dot u_0+\\alpha u_0)t\\big]$ — the fastest return without overshoot;</li>
      <li>overdamped ($\\zeta&gt;1$): $u=Ae^{s_1t}+Be^{s_2t}$ with $A+B=u_0$, $s_1A+s_2B=\\dot u_0$; the slow root $s_1=-\\omega_0^2/(\\alpha+\\beta)$, $\\beta=\\sqrt{\\alpha^2-\\omega_0^2}$, sets the decay.</li>
    </ul>
    <p>Here $u=q-q_p$ is the deviation from the particular solution ($q_p=CV_0$ for the DC step, $0$ for the discharge). First-order limits:
    RC: $R\\dot q+q/C=V$, $q(t)=CV+(q_0-CV)e^{-t/RC}$; RL: $L\\dot I+RI=V$, $I(t)=V/R+(I_0-V/R)e^{-Rt/L}$.</p>
    <p><b>AC steady state.</b> With complex impedance $Z=R+i\\big(\\omega L-\\frac{1}{\\omega C}\\big)$ the current is
    $I(t)=\\frac{V_0}{|Z|}\\sin(\\omega t-\\varphi)$ with</p>
    $$|Z|=\\sqrt{R^2+\\Big(\\omega L-\\frac{1}{\\omega C}\\Big)^2},\\qquad \\tan\\varphi=\\frac{\\omega L-1/\\omega C}{R},\\qquad
      \\langle P\\rangle=\\tfrac12V_0I_0\\cos\\varphi .$$
    <p>The amplitude peaks at $\\omega=\\omega_0$ where $|Z|=R$ and $\\varphi=0$; the voltages across $L$ and $C$ are then $Q$ times the source
    amplitude and cancel each other. The quality factor and full width at half power are</p>
    $$Q=\\frac{\\omega_0}{\\Delta\\omega}=\\frac{1}{R}\\sqrt{\\frac{L}{C}}=\\frac{1}{2\\zeta},\\qquad \\Delta\\omega=\\frac{R}{L},\\qquad
      \\omega_\\pm=\\sqrt{\\omega_0^2+\\alpha^2}\\pm\\alpha .$$
    <p>For RC (high-pass for $V_R$, low-pass for $V_C$) and RL circuits the reference frequency is the corner frequency
    $\\omega_c=1/RC$ or $R/L$, where $|Z|=\\sqrt2R$ and $|\\varphi|=45^\\circ$. The full AC solution is the steady state plus a homogeneous
    transient fixed by the initial conditions; the dashed curve shows exactly that.</p>
    <p><b>Energy.</b> Multiplying the circuit equation by $I$: $\\;VI=\\frac{d}{dt}\\Big(\\frac{q^2}{2C}+\\frac12LI^2\\Big)+RI^2$ — the source's power goes
    into the electric energy of the capacitor, the magnetic energy of the inductor and Joule heat.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li>State vector $(q,\\,I,\\,Q_J,\\,W_s)$ with $\\dot Q_J=RI^2$ and $\\dot W_s=VI$, integrated with the classical 4th-order Runge–Kutta
      method. For RLC: $\\dot q=I$, $\\dot I=(V-RI-q/C)/L$; RC: $\\dot q=(V-q/C)/R$ with $I=\\dot q$; RL: $\\dot I=(V-RI)/L$, $\\dot q=I$ (charge passed).</li>
      <li>Step size: $h=\\min\\big(0.1/\\omega_0,\\ 0.5/|s_{\\rm fast}|,\\ T_{\\rm win}/3000\\big)$ (RC/RL: $0.3\\tau$), and at most 1/150 of the drive period.
      Each animation frame is split into as many RK4 sub-steps as needed (up to 20 000; extremely stiff over-damped
      circuits then simply play more slowly).</li>
      <li>Time is rescaled for display: a transient window of $T_{\\rm win}$ (≈ 6 decay times or 10 oscillation periods) plays in about 8 s;
      in AC mode one drive period takes 1.6 s. The toolbar shows the slow-motion factor.</li>
      <li>The analytic solution above is evaluated at the same times and drawn dashed; the metric "max |RK4 − exact|" shows the largest
      deviation of the current, relative to its peak. When the drive frequency is changed the analytic solution is re-started from
      the current state.</li>
      <li>Accuracy check: energy conservation $E_0+W_s=U_C+U_L+Q_J$ is monitored as a relative residual (RK4 error ∝ $h^4$).</li>
      <li>AC extras: the resonance curve $|I|(\\omega)=V_0/|Z|$ and phase $\\varphi(\\omega)$ (log frequency axis) with a marker at the current $\\omega$;
      dots are measured amplitudes — the peak $|I|$ in each drive period. The phasor diagram rotates at $\\omega$; the vertical
      projection of each arrow is the instantaneous steady-state value, and $\\vec V_R+\\vec V_L+\\vec V_C=\\vec V_s$ head-to-tail.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li>DC step with the defaults ($R=10\\ \\Omega$, $L=100$ mH, $C=100\\ \\mu$F): $\\omega_0=316$ rad/s ($f_0=50.3$ Hz), $\\zeta=0.158$ — the charge
      overshoots $CV_0$ by $e^{-\\pi\\zeta/\\sqrt{1-\\zeta^2}}\\approx60\\%$ and rings at $\\omega_d=312$ rad/s.</li>
      <li>Press "Set critical damping": $R=2\\sqrt{L/C}=63.2\\ \\Omega$, no overshoot. Increase $R$ further (overdamped): the approach becomes
      <em>slower</em> again, governed by $\\tau\\approx RC$.</li>
      <li>DC charging of the RC circuit: exactly half of the energy delivered by the battery, $CV_0^2$, is dissipated in $R$ — whatever $R$ is.</li>
      <li>AC at $\\omega=\\omega_0$: $\\varphi=0$, $|I|=V_0/R=1$ A; the phasors $V_L$ and $V_C$ are equal and opposite and $Q=3.16$ times larger than $V_0$.
      Turn on the sweep and compare the measured dots with the curve; reduce $R$ to 1 Ω ($Q=31.6$) and watch them lag behind the sharp peak.</li>
      <li>AC on RC: below $\\omega_c$ the current leads the voltage by almost $90^\\circ$ (capacitive), above it is nearly in phase.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>Lumped ideal components, valid while the circuit is much smaller than the wavelength $2\\pi c/\\omega$; real inductors have
    series resistance and capacitors leakage, both of which only add to $R$ here. Reading: Purcell &amp; Morin, <i>Electricity and
    Magnetism</i>, ch. 8 (AC circuits); Griffiths, <i>Introduction to Electrodynamics</i>, §7.2; Feynman Lectures vol. I, ch. 23–25
    (resonance) and vol. II, ch. 22; Horowitz &amp; Hill, <i>The Art of Electronics</i>, ch. 1.</p>`,

    mount(api) {
      const P = api.params, src = P.src || "dc", cfg = P.cfg || "rlc", AC = src === "ac";
      const hasL = cfg !== "rc", hasC = cfg !== "rl";
      let Rexact = null;
      api.setControl("lw", { label: cfg === "rlc" ? "Drive frequency $\\omega/\\omega_0$" : "Drive frequency $\\omega/\\omega_c$ (corner)" });

      // ------------------------------------------------------------ layout
      const qTitle = hasC ? "Capacitor charge $q(t)$" : "Charge passed $q(t)=\\int I\\,dt$";
      const layout = [{ id: "circ", title: "Circuit (live)", aspect: 0.66, axes: false, minHeight: 260 }];
      if (AC) layout.push({ id: "phas", title: "Phasor diagram (steady state, rotating at ω)", aspect: 0.66, equal: true, xlim: [-1, 1], ylim: [-1, 1], xlabel: "Re", ylabel: "Im → instantaneous value (V)", minHeight: 260 });
      else layout.push({ id: "en", title: "Energy: $U_C=q^2/2C$, $U_L=LI^2/2$, Joule heat", aspect: 0.66, xlabel: "t (ms)", ylabel: "energy (mJ)", minHeight: 260 });
      layout.push({ id: "q", title: qTitle, aspect: 0.5, xlabel: "t (ms)", ylabel: "q (mC)" });
      layout.push({ id: "I", title: "Current $I(t)$", aspect: 0.5, xlabel: "t (ms)", ylabel: "I (A)" });
      layout.push({ id: "V", title: "Voltages $V_s$, $V_R=RI$, $V_L=L\\dot I$, $V_C=q/C$", aspect: AC ? 0.5 : 0.3, span: AC ? 1 : 2, xlabel: "t (ms)", ylabel: "voltage (V)" });
      if (AC) {
        layout.push({ id: "en", title: "Stored energy $U_C$, $U_L$", aspect: 0.5, xlabel: "t (ms)", ylabel: "energy (mJ)" });
        layout.push({ id: "res", title: "Resonance curve $|I|(\\omega)=V_0/|Z|$", aspect: 0.55, xlog: true, xlim: [0.1, 10], ylim: [0, 1], xlabel: cfg === "rlc" ? "ω / ω₀" : "ω / ω_c", ylabel: "|I| (A)" });
        layout.push({ id: "ph", title: "Phase $\\varphi(\\omega)$ of the voltage relative to the current", aspect: 0.55, xlog: true, xlim: [0.1, 10], ylim: [-95, 95], xlabel: cfg === "rlc" ? "ω / ω₀" : "ω / ω_c", ylabel: "φ (degrees)" });
      }
      const metricDefs = AC ? [
        { id: "w", label: "Drive $\\omega$ / $f$" },
        { id: "Z", label: "Impedance $|Z|$" },
        { id: "phi", label: "Phase $\\varphi$ (V ahead of I)" },
        { id: "I0", label: "Amplitude $|I|$: measured / steady" },
        { id: "P", label: "Mean power $\\frac12V_0I_0\\cos\\varphi$" },
        { id: "Q", label: cfg === "rlc" ? "$Q$ factor / bandwidth $\\Delta\\omega$" : "Corner $\\omega_c$" },
        { id: "en", label: "Energy residual" },
      ] : [
        { id: "cls", label: "Damping" },
        { id: "w0", label: cfg === "rlc" ? "$\\omega_0=1/\\sqrt{LC}$ / $f_0$" : "Time constant $\\tau$" },
        { id: "tc", label: cfg === "rlc" ? "Decay time $1/\\alpha$ / $\\omega_d$" : "Final state" },
        { id: "now", label: "$q$ / $I$ now" },
        { id: "err", label: "max |RK4 − exact| / peak" },
        { id: "en", label: "Energy residual" },
      ];
      const Mt = api.metrics(metricDefs);
      const plots = api.plots(layout);

      // ------------------------------------------------------------ state
      let R, L, C, V0, w0, wref, zeta, alpha, Twin, h, hbase, slow, dtRec, E0;
      let w = 1, th0 = 0, tf0 = 0; // drive: phase θ(t) = th0 + w (t − tf0)
      const y = new Float64Array(4);
      let ws = null, t = 0, done = false, nextRec = 0, errMax = 0, Ipeak = 1e-30, limited = false;
      let dotOff = 0, sweepDir = 1, frame = 0, perPeak = 0, lastPer = 0, measAmp = NaN;
      const S = { n: 0, nmax: 4000, t: new Float64Array(4000) };
      const KEYS = ["q", "I", "qa", "Ia", "Vs", "VR", "VL", "VC", "UC", "UL", "QJ", "Etot"];
      for (const k of KEYS) S[k] = new Float64Array(4000);
      const NA = 600, an = { ta: 0, qa: 0, Ia: 0, tha: 0, w: 1, Z: 1, ph: 0, u0: 0, v0: 0, qp0: 0 };
      const anT = new Float64Array(NA), anQ = new Float64Array(NA), anI = new Float64Array(NA);
      const NR = 320, rW = new Float64Array(NR), rI = new Float64Array(NR), rP = new Float64Array(NR);
      const measW = [], measA = [];
      const sc = { q: RangeScaler(), I: RangeScaler(), V: RangeScaler(), e: RangeScaler() };
      const tmp = [0, 0];

      const Xof = (ww) => (hasL ? ww * L : 0) - (hasC ? 1 / (ww * C) : 0);
      const Zof = (ww) => Math.hypot(R, Xof(ww));
      const Vsrc = (tt) => (AC ? V0 * Math.sin(th0 + w * (tt - tf0)) : src === "dc" ? V0 : 0);

      function params() {
        R = Rexact !== null ? Rexact : Math.pow(10, P.lR);
        L = hasL ? Math.pow(10, P.lL) * 1e-3 : 0;
        C = hasC ? Math.pow(10, P.lC) * 1e-6 : Infinity;
        V0 = P.V0;
        if (cfg === "rlc") {
          w0 = 1 / Math.sqrt(L * C); zeta = (R / 2) * Math.sqrt(C / L); alpha = R / (2 * L); wref = w0;
          if (zeta < 1) { const wd = w0 * Math.sqrt(1 - zeta * zeta); Twin = Math.min(6 / alpha, (10 * 2 * Math.PI) / wd); hbase = 0.1 / w0; }
          else { const beta = Math.sqrt(Math.max(alpha * alpha - w0 * w0, 0)), sSlow = (w0 * w0) / (alpha + beta); Twin = 6 / sSlow; hbase = Math.min(0.5 / (alpha + beta), 0.1 / w0); }
        } else if (cfg === "rc") { const tau = R * C; wref = 1 / tau; Twin = 6 * tau; hbase = 0.3 * tau; }
        else { const tau = L / R; wref = 1 / tau; Twin = 6 * tau; hbase = 0.3 * tau; }
        h = Math.min(hbase, Twin / 3000);
        slow = Twin / 8; dtRec = Twin / 1500;
        E0 = src === "dis" ? (hasC ? 0.5 * C * V0 * V0 : 0.5 * L * (V0 / R) * (V0 / R)) : 0;
      }
      function setDrive(wNew) {
        // keep the drive phase continuous when ω changes
        th0 = th0 + w * (t - tf0); tf0 = t; w = wNew;
        if (AC) { const T = (2 * Math.PI) / w; h = Math.min(hbase, T / 150); slow = T / 1.6; dtRec = T / 120; Twin = 5 * T; }
      }

      // ------------------------------------------------------------ dynamics
      function deriv(tt, s, o) {
        const V = Vsrc(tt);
        if (cfg === "rc") { const I = (V - s[0] / C) / R; o[0] = I; o[1] = 0; o[2] = R * I * I; o[3] = V * I; return; }
        const I = s[1];
        o[0] = I;
        o[1] = (V - R * I - (hasC ? s[0] / C : 0)) / L;
        o[2] = R * I * I; o[3] = V * I;
      }
      function currentI() { return cfg === "rc" ? (Vsrc(t) - y[0] / C) / R : y[1]; }

      // ------------------------------------------------------------ analytic solution
      function partic(tt) {
        if (!AC) {
          const V = src === "dc" ? V0 : 0;
          if (cfg === "rl") { tmp[0] = (V / R) * tt; tmp[1] = V / R; } else { tmp[0] = C * V; tmp[1] = 0; }
          return tmp;
        }
        const psi = an.tha + an.w * (tt - an.ta);
        tmp[0] = -(V0 / (an.w * an.Z)) * Math.cos(psi - an.ph); tmp[1] = (V0 / an.Z) * Math.sin(psi - an.ph);
        return tmp;
      }
      function anchor() {
        an.ta = t; an.qa = y[0]; an.Ia = currentI(); an.tha = th0 + w * (t - tf0); an.w = w;
        if (AC) { an.Z = Zof(w); an.ph = Math.atan2(Xof(w), R); }
        const p = partic(t); an.qp0 = p[0]; an.u0 = an.qa - p[0]; an.v0 = an.Ia - p[1];
      }
      /** Exact solution at time tt → out[0] = q, out[1] = I. */
      function analytic(tt, out) {
        const s = tt - an.ta, p = partic(tt), qp = p[0], Ip = p[1];
        if (cfg === "rlc") {
          const u0 = an.u0, v0 = an.v0, a = alpha, wd2 = w0 * w0 - a * a;
          let u, up;
          if (Math.abs(wd2) < 1e-10 * w0 * w0) { // critical
            const b = v0 + a * u0, e = Math.exp(-a * s); u = e * (u0 + b * s); up = e * (b - a * (u0 + b * s));
          } else if (wd2 > 0) {
            const wd = Math.sqrt(wd2), b = (v0 + a * u0) / wd, e = Math.exp(-a * s), cs = Math.cos(wd * s), sn = Math.sin(wd * s);
            u = e * (u0 * cs + b * sn); up = e * ((-a * u0 + b * wd) * cs + (-a * b - u0 * wd) * sn);
          } else {
            const beta = Math.sqrt(-wd2), s1 = -(w0 * w0) / (a + beta), s2 = -a - beta, A = (v0 - s2 * u0) / (s1 - s2), B = u0 - A;
            const e1 = Math.exp(s1 * s), e2 = Math.exp(s2 * s); u = A * e1 + B * e2; up = A * s1 * e1 + B * s2 * e2;
          }
          out[0] = qp + u; out[1] = Ip + up;
        } else if (cfg === "rc") {
          const tc = R * C, u = an.u0 * Math.exp(-s / tc); out[0] = qp + u; out[1] = Ip - u / tc;
        } else {
          const tc = L / R, e = Math.exp(-s / tc); out[1] = Ip + an.v0 * e; out[0] = an.qa + (qp - an.qp0) + tc * an.v0 * (1 - e);
        }
        return out;
      }

      // ------------------------------------------------------------ recording
      const ao = [0, 0];
      function record() {
        if (S.n >= S.nmax) { // roll: drop the oldest half
          const hlf = S.nmax >> 1; S.t.copyWithin(0, hlf); for (const k of KEYS) S[k].copyWithin(0, hlf); S.n = S.nmax - hlf;
        }
        const i = S.n++, q = y[0], I = currentI(), V = Vsrc(t);
        const VC = hasC ? q / C : 0, VR = R * I, VL = hasL ? V - VR - VC : 0;
        S.t[i] = t * 1000; S.q[i] = q * 1000; S.I[i] = I; S.Vs[i] = V; S.VR[i] = VR; S.VL[i] = VL; S.VC[i] = VC;
        S.UC[i] = hasC ? (q * q) / (2 * C) * 1000 : 0; S.UL[i] = hasL ? 0.5 * L * I * I * 1000 : 0; S.QJ[i] = y[2] * 1000; S.Etot[i] = (E0 + y[3]) * 1000;
        if (!AC || !P.sweep) { analytic(t, ao); S.qa[i] = ao[0] * 1000; S.Ia[i] = ao[1]; } else { S.qa[i] = NaN; S.Ia[i] = NaN; }
        if (P.an && isFinite(S.Ia[i])) errMax = Math.max(errMax, Math.abs(I - S.Ia[i]));
      }

      // ------------------------------------------------------------ drawing: circuit
      function drawCircuit() {
        const p = plots.circ; p.clear();
        const I = currentI(), q = y[0], V = Vsrc(t);
        const Iscale = Math.max(Ipeak, 1e-30), rel = I / Iscale;
        p.custom((c, pl) => {
          const W = pl.W, H = pl.H, x0 = W * 0.16, x1 = W * 0.84, yT = H * 0.18, yB = H * 0.8, xm = (x0 + x1) / 2, ym = (yT + yB) / 2;
          // wires
          c.strokeStyle = "#8b98a8"; c.lineWidth = 2;
          c.beginPath(); c.moveTo(x0, ym - 22); c.lineTo(x0, yT); c.lineTo(xm - 48, yT); c.moveTo(xm + 48, yT); c.lineTo(x1, yT); c.lineTo(x1, ym - 44);
          c.moveTo(x1, ym + 44); c.lineTo(x1, yB); c.lineTo(xm + 6, yB); c.moveTo(xm - 6, yB); c.lineTo(x0, yB); c.lineTo(x0, ym + 22); c.stroke();
          if (!hasL) { c.beginPath(); c.moveTo(x1, ym - 44); c.lineTo(x1, ym + 44); c.stroke(); }
          if (!hasC) { c.beginPath(); c.moveTo(xm - 6, yB); c.lineTo(xm + 6, yB); c.stroke(); }
          // resistor (top) with glow ∝ power
          const Pn = Math.min(1, (R * I * I) / (R * Iscale * Iscale));
          if (Pn > 0.01) { c.save(); c.shadowColor = "rgba(255,120,60,0.9)"; c.shadowBlur = 20 * Pn; c.strokeStyle = `rgba(255,140,80,${0.3 + 0.6 * Pn})`; c.lineWidth = 7; c.beginPath(); c.moveTo(xm - 40, yT); c.lineTo(xm + 40, yT); c.stroke(); c.restore(); }
          c.strokeStyle = "#e6edf3"; c.lineWidth = 2; c.beginPath(); c.moveTo(xm - 48, yT);
          for (let k = 0; k <= 12; k++) c.lineTo(xm - 40 + (80 * k) / 12, yT + (k === 0 || k === 12 ? 0 : k % 2 ? -8 : 8));
          c.lineTo(xm + 48, yT); c.stroke();
          txt(c, `R = ${fmtR(R)}`, xm, yT - 20, { size: 12, align: "center", bold: true });
          // inductor (right) with field indicator
          if (hasL) {
            const UL = 0.5 * L * I * I, ULs = 0.5 * L * Iscale * Iscale, g = Math.min(1, UL / (ULs || 1));
            if (g > 0.01) { c.fillStyle = `rgba(88,166,255,${0.08 + 0.25 * g})`; c.fillRect(x1 - 16, ym - 44, 32, 88); }
            c.strokeStyle = "#e6edf3"; c.lineWidth = 2; c.beginPath();
            for (let k = 0; k < 5; k++) { const yc = ym - 44 + 8.8 + 17.6 * k; c.moveTo(x1, yc - 8.8); c.arc(x1, yc, 8.8, -Math.PI / 2, Math.PI / 2, false); }
            c.stroke();
            if (Math.abs(rel) > 0.03) {
              const len = 70 * Math.min(1, Math.abs(rel)), dir = I > 0 ? 1 : -1;
              arrowPx(c, x1 + 28, ym - (dir * len) / 2, x1 + 28, ym + (dir * len) / 2, "#58a6ff", 2.5, 8);
              txt(c, "B", x1 + 38, ym, { size: 12, bold: true, color: "#58a6ff" });
            }
            txt(c, `L = ${fmtL(L * 1000)}`, x1, ym - 58, { size: 12, align: "center", bold: true, bg: "#0f151c" });
          } else txt(c, "(no inductor)", x1 - 10, ym, { size: 11, align: "right", color: "#8b98a8" });
          // capacitor (bottom): right plate +q
          if (hasC) {
            const qs = Math.max(maxQ, 1e-30), fr = Math.min(1, Math.abs(q) / qs);
            c.strokeStyle = "#e6edf3"; c.lineWidth = 3;
            c.beginPath(); c.moveTo(xm - 6, yB - 22); c.lineTo(xm - 6, yB + 22); c.moveTo(xm + 6, yB - 22); c.lineTo(xm + 6, yB + 22); c.stroke();
            const nS = Math.round(5 * fr);
            for (let k = 0; k < nS; k++) {
              const yy = yB - 18 + (36 * (k + 0.5)) / Math.max(nS, 1);
              txt(c, q > 0 ? "+" : "−", xm + 16, yy, { size: 12, bold: true, align: "center", color: q > 0 ? "#f85149" : "#58a6ff" });
              txt(c, q > 0 ? "−" : "+", xm - 16, yy, { size: 12, bold: true, align: "center", color: q > 0 ? "#58a6ff" : "#f85149" });
            }
            if (fr > 0.03) { c.fillStyle = `rgba(245,158,11,${0.15 + 0.5 * fr})`; c.fillRect(xm - 5, yB - 20, 10, 40); }
            txt(c, `C = ${fmtC(C * 1e6)}`, xm, yB + 36, { size: 12, align: "center", bold: true });
            txt(c, `V_C = ${PM.fmt(q / C, 2)} V`, xm, yB - 34, { size: 11, align: "center", color: "#f59e0b" });
          } else txt(c, "(no capacitor)", xm, yB + 18, { size: 11, align: "center", color: "#8b98a8" });
          // source (left)
          c.strokeStyle = "#e6edf3"; c.lineWidth = 2;
          if (AC) {
            c.beginPath(); c.arc(x0, ym, 22, 0, 2 * Math.PI); c.stroke();
            c.beginPath(); for (let k = 0; k <= 30; k++) { const xx = x0 - 13 + (26 * k) / 30, yy = ym - 8 * Math.sin((2 * Math.PI * k) / 30); k ? c.lineTo(xx, yy) : c.moveTo(xx, yy); } c.stroke();
            txt(c, `${PM.fmt(V, 2)} V`, x0, ym + 36, { size: 12, align: "center", color: "#e6edf3", bg: "#0f151c" });
          } else if (src === "dc") {
            c.lineWidth = 3; c.beginPath(); c.moveTo(x0 - 16, ym - 22); c.lineTo(x0 + 16, ym - 22); c.stroke();
            c.lineWidth = 5; c.beginPath(); c.moveTo(x0 - 8, ym - 10); c.lineTo(x0 + 8, ym - 10); c.stroke();
            c.lineWidth = 3; c.beginPath(); c.moveTo(x0 - 16, ym + 2); c.lineTo(x0 + 16, ym + 2); c.stroke();
            c.lineWidth = 5; c.beginPath(); c.moveTo(x0 - 8, ym + 14); c.lineTo(x0 + 8, ym + 14); c.stroke();
            c.lineWidth = 2; c.beginPath(); c.moveTo(x0, ym + 14); c.lineTo(x0, ym + 22); c.stroke();
            txt(c, "+", x0 + 22, ym - 24, { size: 13, bold: true, color: "#f85149" });
            txt(c, `V₀ = ${PM.fmt(V0, 1)} V`, x0, ym + 38, { size: 12, align: "center", bg: "#0f151c" });
          } else {
            c.beginPath(); c.moveTo(x0, ym - 22); c.lineTo(x0, ym + 22); c.stroke();
            c.fillStyle = "#e6edf3"; c.beginPath(); c.arc(x0, ym - 22, 3, 0, 7); c.arc(x0, ym + 22, 3, 0, 7); c.fill();
            txt(c, "switch closed", x0, ym + 38, { size: 11, align: "center", color: "#8b98a8", bg: "#0f151c" });
          }
          // moving charges (clockwise for I > 0)
          const segs = [[x0, yB, x0, yT], [x0, yT, x1, yT], [x1, yT, x1, yB], [x1, yB, x0, yB]];
          const lens = segs.map((s) => Math.hypot(s[2] - s[0], s[3] - s[1])), per = lens.reduce((a, b) => a + b, 0);
          const nd = 26;
          c.fillStyle = IND; c.globalAlpha = 0.35 + 0.65 * Math.min(1, Math.abs(rel) * 3);
          for (let k = 0; k < nd; k++) {
            let s = (((dotOff + (k * per) / nd) % per) + per) % per, j = 0;
            while (s > lens[j] && j < 3) { s -= lens[j]; j++; }
            const g = segs[j], f = s / lens[j], X = g[0] + (g[2] - g[0]) * f, Y = g[1] + (g[3] - g[1]) * f;
            if (hasC && Math.abs(X - xm) < 8 && Math.abs(Y - yB) < 4) continue;
            c.beginPath(); c.arc(X, Y, 3.2, 0, 2 * Math.PI); c.fill();
          }
          c.globalAlpha = 1;
          txt(c, `I = ${PM.fmt(I, 3)} A`, xm, ym - 6, { size: 14, bold: true, align: "center", color: IND });
          txt(c, I >= 0 ? "clockwise" : "counter-clockwise", xm, ym + 14, { size: 11, align: "center", color: "#8b98a8" });
        });
      }

      // ------------------------------------------------------------ drawing: time plots
      function xWindow() {
        if (!AC) return [0, Twin * 1000];
        const tm = t * 1000, wd = Twin * 1000; return [Math.max(0, tm - wd), Math.max(tm, wd)];
      }
      function tplot(p, keys, scaler, fixed) {
        const [xa, xb] = xWindow(), i0 = firstIdx(S.t, S.n, xa);
        let lim;
        if (fixed) lim = fixed; else { const mm = minmax(keys.map((k) => S[k]), S.n, i0); lim = scaler.fit(mm[0], mm[1]); }
        p.setLimits([xa, xb], lim); p.clear(); p.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
        return S.t.subarray(0, S.n);
      }
      const sub = (k) => S[k].subarray(0, S.n);
      let limQ = null, limI = null, limE = null, maxQ = 1e-30;

      function drawPhasor() {
        const p = plots.phas, Z = Zof(w), ph = Math.atan2(Xof(w), R), I0 = V0 / Z;
        const VR = I0 * R, VL = hasL ? I0 * w * L : 0, VC = hasC ? I0 / (w * C) : 0;
        const lim = 1.12 * Math.max(V0, VR + Math.max(VL, VC), Math.hypot(VR, VL), Math.hypot(VR, VC));
        p.setLimits([-lim, lim], [-lim, lim]); p.clear();
        p.hline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 }); p.vline(0, { color: PlotColors.muted, alpha: 0.4, width: 1 });
        const psi = th0 + w * (t - tf0), aI = psi - ph;
        p.custom((c, pl) => { c.strokeStyle = "rgba(139,152,168,0.25)"; c.setLineDash([3, 4]); c.beginPath(); c.arc(pl.X(0), pl.Y(0), V0 * pl.sx, 0, 2 * Math.PI); c.stroke(); c.setLineDash([]); });
        const vs = [V0 * Math.cos(psi), V0 * Math.sin(psi)];
        let px = 0, py = 0;
        const chain = [["V_R", VR, aI, PlotColors.accent3], ["V_L", VL, aI + Math.PI / 2, PlotColors.blue], ["V_C", VC, aI - Math.PI / 2, PlotColors.accent2]];
        for (const [lab, mag, ang, col] of chain) {
          if (mag <= 0) continue;
          const nx = px + mag * Math.cos(ang), ny = py + mag * Math.sin(ang);
          if (mag > lim * 0.02) p.arrow(px, py, nx, ny, { color: col, width: 2.6, head: 10 });
          p.text(nx, ny, lab, { color: col, size: 12, bold: true, dx: 6, dy: -8 });
          px = nx; py = ny;
        }
        p.arrow(0, 0, vs[0], vs[1], { color: PlotColors.text, width: 2.2, head: 10 });
        p.text(vs[0], vs[1], "V_s", { color: PlotColors.text, size: 12, bold: true, dx: 6, dy: 10 });
        const iL = 0.55 * lim; // current phasor (own scale)
        p.arrow(0, 0, iL * Math.cos(aI), iL * Math.sin(aI), { color: IND, width: 1.6, head: 8, dash: [5, 3] });
        p.text(iL * Math.cos(aI), iL * Math.sin(aI), "I", { color: IND, size: 12, bold: true, dx: 6, dy: 8 });
        p.segment(vs[0], vs[1], lim, vs[1], { color: PlotColors.text, width: 1, dash: [2, 4], alpha: 0.6 });
        p.circle(lim * 0.97, vs[1], 4, { px: true, color: PlotColors.text });
        p.label(`φ = ${PM.fmt((ph * 180) / Math.PI, 1)}°`, "tl", { size: 12 });
      }
      function drawResonance() {
        const pr = plots.res, pp = plots.ph, Z = Zof(w), ph = (Math.atan2(Xof(w), R) * 180) / Math.PI, wr = w / wref;
        let mx = 0; for (let i = 0; i < NR; i++) mx = Math.max(mx, rI[i]);
        for (let i = 0; i < measA.length; i++) mx = Math.max(mx, measA[i]);
        pr.setLimits(null, [0, mx * 1.12]); pr.clear();
        if (cfg === "rlc") {
          const wp = (Math.sqrt(w0 * w0 + alpha * alpha) + alpha) / wref, wm = (Math.sqrt(w0 * w0 + alpha * alpha) - alpha) / wref;
          pr.rect(Math.max(wm, 0.1), 0, Math.min(wp, 10), mx * 1.12, { color: PlotColors.accent, alpha: 0.08 });
          pr.hline(V0 / R / Math.SQRT2, { color: PlotColors.muted, dash: [3, 4], width: 1 });
        } else pr.vline(1, { color: PlotColors.muted, dash: [3, 4], width: 1 });
        pr.line(rW, rI, { color: PlotColors.accent, width: 2 });
        if (measW.length) pr.points(measW, measA, { color: IND, size: 2.6, alpha: 0.85 });
        pr.vline(wr, { color: PlotColors.text, dash: [4, 4], width: 1 });
        pr.circle(wr, V0 / Z, 5, { px: true, color: PlotColors.bad, stroke: PlotColors.text });
        pr.legend([{ label: "V₀/|Z| (steady state)", color: PlotColors.accent }, { label: "measured peak |I|", color: IND, type: "dot" }], "tr");
        pp.clear();
        pp.hline(0, { color: PlotColors.muted, alpha: 0.5, width: 1 }); pp.hline(45, { color: PlotColors.muted, dash: [2, 4], width: 1, alpha: 0.5 }); pp.hline(-45, { color: PlotColors.muted, dash: [2, 4], width: 1, alpha: 0.5 });
        pp.line(rW, rP, { color: PlotColors.accent2, width: 2 });
        pp.vline(wr, { color: PlotColors.text, dash: [4, 4], width: 1 });
        pp.circle(wr, ph, 5, { px: true, color: PlotColors.bad, stroke: PlotColors.text });
        pp.label(["φ > 0: inductive (I lags V)", "φ < 0: capacitive (I leads V)"], "bl", { size: 11 });
      }

      // ------------------------------------------------------------ lifecycle
      function precompute() {
        errMax = 0;
        if (!AC) {
          let qmx = -Infinity, qmn = Infinity, imx = -Infinity, imn = Infinity, emx = E0;
          for (let i = 0; i < NA; i++) {
            const tt = (Twin * i) / (NA - 1); analytic(tt, ao); anT[i] = tt * 1000; anQ[i] = ao[0] * 1000; anI[i] = ao[1];
            qmx = Math.max(qmx, anQ[i]); qmn = Math.min(qmn, anQ[i]); imx = Math.max(imx, ao[1]); imn = Math.min(imn, ao[1]);
            if (src === "dc") emx = Math.max(emx, V0 * ao[0]);
            emx = Math.max(emx, hasC ? (ao[0] * ao[0]) / (2 * C) : 0, hasL ? 0.5 * L * ao[1] * ao[1] : 0);
          }
          const pad = (a, b) => { a = Math.min(a, 0); b = Math.max(b, 0); const s = (b - a) || 1e-12; return [a - (a < 0 ? 0.08 * s : 0), b + 0.08 * s]; };
          limQ = pad(qmn, qmx); limI = pad(imn, imx); limE = [0, Math.max(emx * 1000 * 1.08, 1e-12)];
          Ipeak = Math.max(Math.abs(imn), Math.abs(imx), 1e-30);
          maxQ = Math.max(Math.abs(qmn), Math.abs(qmx)) / 1000;
        } else {
          for (let i = 0; i < NR; i++) { const r = Math.pow(10, -1 + (2 * i) / (NR - 1)), ww = r * wref; rW[i] = r; rI[i] = V0 / Zof(ww); rP[i] = (Math.atan2(Xof(ww), R) * 180) / Math.PI; }
        }
      }
      function updateACScales() {
        const Z = Zof(w); Ipeak = Math.max(V0 / Z, 1e-30); maxQ = V0 / (w * Z);
      }

      const inst = {
        reset() {
          params();
          t = 0; done = false; ws = null; frame = 0; limited = false; perPeak = 0; lastPer = 0; measAmp = NaN;
          th0 = 0; tf0 = 0; w = AC ? Math.pow(10, P.lw) * wref : 1;
          if (AC) setDrive(w);
          y[2] = 0; y[3] = 0;
          if (src === "dis") { if (hasC) { y[0] = C * V0; y[1] = 0; } else { y[0] = 0; y[1] = V0 / R; } }
          else { y[0] = 0; y[1] = 0; }
          if (cfg === "rc") y[1] = currentI();
          measW.length = 0; measA.length = 0;
          S.n = 0; for (const k in sc) sc[k].reset();
          anchor(); precompute(); if (AC) updateACScales();
          record(); nextRec = dtRec; dotOff = 0;
        },
        onParam(id, v) {
          if (id === "lR" || id === "lL" || id === "lC") Rexact = null;
          if (id === "lw" && AC) { setDrive(Math.pow(10, v) * wref); anchor(); updateACScales(); }
          if (id === "sweep" && AC && !v) anchor();
        },
        onAction(id) {
          if (id === "crit" && cfg === "rlc") {
            L = Math.pow(10, P.lL) * 1e-3; C = Math.pow(10, P.lC) * 1e-6;
            Rexact = 2 * Math.sqrt(L / C);
            api.setControl("lR", { value: Math.log10(Rexact) });
            api.time = 0; inst.reset();
          }
        },
        step(dt) {
          if (done) { inst.reset(); return; }
          frame++;
          if (AC && P.sweep) {
            let x = Math.log10(w / wref) + sweepDir * dt * 0.1;
            if (x > 1) { x = 1; sweepDir = -1; } if (x < -1) { x = -1; sweepDir = 1; }
            setDrive(Math.pow(10, x) * wref); updateACScales();
            if (frame % 6 === 0) api.setControl("lw", { value: x });
          }
          const simDt = dt * slow;
          let n = Math.ceil(simDt / h); limited = n > 20000; if (n > 20000) n = 20000;
          const hh = Math.min(h, simDt / Math.max(n, 1));
          for (let i = 0; i < n && !done; i++) {
            ws = PM.rk4(deriv, t, y, hh, ws); t += hh;
            if (cfg === "rc") y[1] = currentI();
            if (AC) {
              const I = Math.abs(currentI()); if (I > perPeak) perPeak = I;
              const per = Math.floor((th0 + w * (t - tf0)) / (2 * Math.PI));
              if (per > lastPer) {
                lastPer = per; measAmp = perPeak;
                measW.push(w / wref); measA.push(perPeak); if (measW.length > 900) { measW.shift(); measA.shift(); }
                perPeak = 0;
              }
            }
            if (t >= nextRec) { record(); nextRec += dtRec; }
            if (!AC && t >= Twin) { done = true; record(); api.pause(); }
          }
          dotOff += dt * 110 * (currentI() / Math.max(Ipeak, 1e-30));
        },
        render() {
          drawCircuit();
          const T = S.t.subarray(0, S.n), showAn = P.an && !(AC && P.sweep);
          // charge
          let p = plots.q; tplot(p, ["q", "qa"], sc.q, AC ? null : limQ);
          if (!AC && hasC) p.hline(src === "dc" ? C * V0 * 1000 : 0, { color: PlotColors.muted, dash: [3, 4], width: 1 });
          if (showAn) { if (AC) p.line(T, sub("qa"), { color: PlotColors.text, width: 1.2, dash: [6, 4], alpha: 0.8 }); else p.line(anT, anQ, { color: PlotColors.text, width: 1.2, dash: [6, 4], alpha: 0.8 }); }
          p.line(T, sub("q"), { color: PlotColors.accent3, width: 2 });
          // current
          p = plots.I; tplot(p, ["I", "Ia"], sc.I, AC ? null : limI);
          if (showAn) { if (AC) p.line(T, sub("Ia"), { color: PlotColors.text, width: 1.2, dash: [6, 4], alpha: 0.8 }); else p.line(anT, anI, { color: PlotColors.text, width: 1.2, dash: [6, 4], alpha: 0.8 }); }
          p.line(T, sub("I"), { color: IND, width: 2 });
          p.legend(showAn ? [{ label: "RK4", color: IND }, { label: "exact", color: PlotColors.text, dash: [6, 4] }] : [{ label: "RK4", color: IND }], "tr");
          // voltages
          p = plots.V; tplot(p, ["Vs", "VR", "VL", "VC"], sc.V);
          const leg = [{ label: "V_s", color: PlotColors.text }];
          p.line(T, sub("Vs"), { color: PlotColors.text, width: 1.4, alpha: 0.8 });
          p.line(T, sub("VR"), { color: PlotColors.accent3, width: 1.7 }); leg.push({ label: "V_R", color: PlotColors.accent3 });
          if (hasL) { p.line(T, sub("VL"), { color: PlotColors.blue, width: 1.7 }); leg.push({ label: "V_L", color: PlotColors.blue }); }
          if (hasC) { p.line(T, sub("VC"), { color: PlotColors.accent2, width: 1.7 }); leg.push({ label: "V_C", color: PlotColors.accent2 }); }
          p.legend(leg, "tr");
          // energy
          p = plots.en;
          if (AC) {
            tplot(p, ["UC", "UL"], sc.e);
          } else tplot(p, [], null, limE);
          const lE = [];
          if (hasC) { p.line(T, sub("UC"), { color: PlotColors.accent2, width: 1.8 }); lE.push({ label: "U_C", color: PlotColors.accent2 }); }
          if (hasL) { p.line(T, sub("UL"), { color: PlotColors.blue, width: 1.8 }); lE.push({ label: "U_L", color: PlotColors.blue }); }
          if (!AC) {
            p.line(T, sub("QJ"), { color: PlotColors.bad, width: 1.8 }); lE.push({ label: "Joule heat Q_J", color: PlotColors.bad });
            p.line(T, sub("Etot"), { color: PlotColors.text, width: 1.3, dash: [5, 4] }); lE.push({ label: "E₀ + source work", color: PlotColors.text, dash: [5, 4] });
          }
          p.legend(lE, "tr");
          if (AC) { drawPhasor(); drawResonance(); }

          // metrics
          const I = currentI(), Etot = E0 + y[3], Est = (hasC ? (y[0] * y[0]) / (2 * C) : 0) + (hasL ? 0.5 * L * I * I : 0) + y[2];
          const enRes = Math.abs(Etot - Est) / Math.max(Math.abs(Etot), Est, 1e-30);
          Mt.set("en", t > 0 ? PM.fmt(enRes, 2) : "—");
          if (AC) {
            const Z = Zof(w), ph = Math.atan2(Xof(w), R), I0 = V0 / Z;
            Mt.set("w", `${fmtW(w)} rad/s / ${fmtW(w / (2 * Math.PI))} Hz`);
            Mt.set("Z", fmtR(Z));
            Mt.set("phi", PM.fmt((ph * 180) / Math.PI, 1) + "°");
            Mt.set("I0", `${isFinite(measAmp) ? PM.fmt(measAmp, 3) : "…"} / ${PM.fmt(I0, 3)} A`);
            Mt.set("P", PM.fmt(0.5 * V0 * I0 * Math.cos(ph), 3) + " W");
            Mt.set("Q", cfg === "rlc" ? `${PM.fmt((1 / R) * Math.sqrt(L / C), 3)} / ${fmtW(R / L)} rad/s` : `${fmtW(wref)} rad/s`);
          } else {
            if (cfg === "rlc") {
              const cls = Math.abs(zeta - 1) < 2e-3 ? "critically damped" : zeta < 1 ? "underdamped" : "overdamped";
              Mt.set("cls", `ζ = ${PM.fmt(zeta, 3)}, ${cls}`);
              Mt.set("w0", `${fmtW(w0)} rad/s / ${fmtW(w0 / (2 * Math.PI))} Hz`);
              if (zeta < 1) Mt.set("tc", `${PM.fmt(1000 / alpha, 3)} ms · ${fmtW(w0 * Math.sqrt(1 - zeta * zeta))} rad/s`);
              else { const beta = Math.sqrt(Math.max(alpha * alpha - w0 * w0, 0)); Mt.set("tc", `slowest 1/|s| = ${PM.fmt(1000 * (alpha + beta) / (w0 * w0), 3)} ms`); }
            } else {
              Mt.set("cls", cfg === "rc" ? "first order (RC)" : "first order (RL)");
              Mt.set("w0", `τ = ${cfg === "rc" ? "RC" : "L/R"} = ${PM.fmt(1000 / wref, 3)} ms`);
              Mt.set("tc", cfg === "rc" ? `q → ${PM.fmt((src === "dc" ? C * V0 : 0) * 1000, 3)} mC` : `I → ${PM.fmt(src === "dc" ? V0 / R : 0, 3)} A`);
            }
            Mt.set("now", `${PM.fmt(y[0] * 1000, 3)} mC / ${PM.fmt(I, 3)} A`);
            Mt.set("err", P.an ? PM.fmt(errMax / Math.max(Ipeak, 1e-30), 2) : "—");
          }
          const factor = slow > 0 ? 1 / slow : 1;
          api.setTime(`t = ${PM.fmt(t * 1000, 2)} ms · ${slow < 1 ? `slow motion ×1/${PM.fmt(factor, 0)}` : `time-lapse ×${PM.fmt(slow, 1)}`}${limited ? " (stiff: slowed)" : ""}${done ? " · finished" : ""}`);
        },
      };
      return inst;
    },
  });
})();
