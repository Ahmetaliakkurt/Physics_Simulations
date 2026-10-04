/* Hydrogen atom, n = 2 shell: Bohr → fine structure → Lamb shift → Zeeman splitting.
 * The levels rearrange with smooth transitions as stages are switched and the field strength changes. */
(function () {
  "use strict";

  // the 8 states of the n = 2 shell: term, m_j, Landé g_J
  const STATES = [
    { term: "2s₁/₂", key: "2s", mj: 0.5, g: 2.0, dash: [] },
    { term: "2s₁/₂", key: "2s", mj: -0.5, g: 2.0, dash: [7, 4] },
    { term: "2p₁/₂", key: "2p1", mj: 0.5, g: 2 / 3, dash: [] },
    { term: "2p₁/₂", key: "2p1", mj: -0.5, g: 2 / 3, dash: [7, 4] },
    { term: "2p₃/₂", key: "2p3", mj: 1.5, g: 4 / 3, dash: [] },
    { term: "2p₃/₂", key: "2p3", mj: 0.5, g: 4 / 3, dash: [7, 4] },
    { term: "2p₃/₂", key: "2p3", mj: -0.5, g: 4 / 3, dash: [10, 3, 2, 3] },
    { term: "2p₃/₂", key: "2p3", mj: -1.5, g: 4 / 3, dash: [2, 3] },
  ];
  const TERM_COLOR = { "2s": "#58a6ff", "2p1": "#3fb950", "2p3": "#f85149" };
  const STAGES = [
    { h: "1 · Ĥ₀", d: "Bohr / Schrödinger" },
    { h: "2 · + fine structure", d: "spin–orbit + relativistic" },
    { h: "3 · + Lamb", d: "QED / vacuum" },
    { h: "4 · + Zeeman", d: "external magnetic field" },
  ];
  const FLAGS = [null, "fs", "lamb", "zeeman"];
  const MU_B = 57.883818; // µeV/T
  const GHZ_PER_UEV = 0.2417990; // E/h

  const fracMJ = (mj) => (mj > 0 ? "+" : "−") + (Math.abs(mj) === 0.5 ? "1/2" : "3/2");

  App.register({
    id: "hydrogen-levels",
    category: "quantum",
    order: 33,
    title: "Hydrogen Atom Energy Levels",
    icon: "🔬",
    subtitle: "The eightfold degeneracy of hydrogen's n = 2 shell broken step by step as Hamiltonian corrections are added: fine structure, Lamb shift and Zeeman splitting.",
    notes: [
      { type: "info", html: "This diagram is not a numerical eigenvalue solution: the shifts of the 8 states of the $n=2$ shell are computed from the standard perturbative formulas, with default magnitudes equal to the measured values (fine structure ≈ 45 µeV, Lamb shift ≈ 4.4 µeV), to show how the degeneracy is lifted <b>step by step</b>: $\\hat H_0 \\to +\\hat H_{fs} \\to +\\hat H_{Lamb} \\to +\\hat H_{Zeeman}$. Switch terms on and off and the levels glide to their new positions." },
    ],
    animated: true,
    speed: { min: 0.2, max: 3, value: 1, step: 0.1 },
    controls: [
      { type: "section", label: "Hamiltonian terms" },
      { id: "fs", type: "checkbox", label: "Fine structure $\\hat H_{fs}$ (spin–orbit + relativistic)", value: true, live: true },
      { id: "lamb", type: "checkbox", label: "Lamb shift $\\hat H_{Lamb}$ (QED vacuum fluctuations)", value: true, live: true },
      { id: "zeeman", type: "checkbox", label: "Zeeman effect $\\hat H_{Zeeman}$ (external magnetic field)", value: true, live: true },
      { type: "section", label: "Magnitudes" },
      { id: "fsShift", type: "slider", label: "Fine-structure splitting ($2p_{3/2}$)", min: 0, max: 100, step: 1, value: 45, unit: "µeV", live: true },
      { id: "lambShift", type: "slider", label: "Lamb shift ($2s_{1/2}$)", min: 0, max: 15, step: 0.1, value: 4.3, unit: "µeV", live: true },
      { id: "B", type: "slider", label: "Magnetic field as $\\mu_B B$", min: 0, max: 20, step: 0.5, value: 8, unit: "µeV", live: true,
        help: "$\\mu_B B = 1$ µeV corresponds to $B \\approx 17.3$ mT." },
      { type: "section", label: "Animation" },
      { id: "sweep", type: "checkbox", label: "Sweep the B field (0 → 20 → 0)", value: false, live: true },
      { id: "seq", type: "button", label: "▶ Animate the stages", primary: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A hydrogen atom (one electron bound to a proton, nucleus treated as infinitely heavy) in its first excited shell, $n=2$, optionally placed in a
      uniform static magnetic field $\\mathbf B=B\\hat z$. Counting spin, the shell has 8 states, labelled here in the coupled basis $|n\\,l\\,j\\,m_j\\rangle$:
      $2s_{1/2}$ ($m_j=\\pm\\tfrac12$), $2p_{1/2}$ ($m_j=\\pm\\tfrac12$) and $2p_{3/2}$ ($m_j=\\pm\\tfrac12,\\pm\\tfrac32$). Energies are given in µeV
      ($1\\ \\mu\\text{eV}\\leftrightarrow 241.8$ MHz via $E=h\\nu$) and the field as the Zeeman energy scale $\\mu_BB$ with
      $\\mu_B=57.88\\ \\mu\\text{eV/T}$. Hyperfine structure (nuclear spin) is neglected.</p>

      <h4>Equations being solved</h4>
      <p>The Hamiltonian is built up in four stages, $\\hat H=\\hat H_0+\\hat H_{fs}+\\hat H_{Lamb}+\\hat H_Z$, each treated as a perturbation of the previous one.</p>
      <p><b>1. Bohr / Schrödinger.</b> $\\hat H_0=\\hat p^2/2m-e^2/4\\pi\\varepsilon_0r$ gives $E_n=-13.6\\ \\text{eV}/n^2$; all 8 states of $n=2$ are degenerate.</p>
      <p><b>2. Fine structure.</b> The relativistic kinetic correction, spin–orbit coupling and the Darwin term,</p>
      $$\\hat H_{fs}=-\\frac{\\hat p^4}{8m^3c^2}+\\frac{1}{2m^2c^2}\\frac{1}{r}\\frac{dV}{dr}\\,\\hat{\\mathbf L}\\cdot\\hat{\\mathbf S}+\\frac{\\pi\\hbar^2}{2m^2c^2}\\frac{e^2}{4\\pi\\varepsilon_0}\\,\\delta^3(\\mathbf r),$$
      <p>combine to a shift that depends only on $n$ and $j$:</p>
      $$E_{nj}=-\\frac{13.6\\ \\text{eV}}{n^2}\\left[1+\\frac{\\alpha^2}{n^2}\\left(\\frac{n}{j+\\tfrac12}-\\frac34\\right)\\right].$$
      <p>For $n=2$ this lowers the $j=\\tfrac12$ levels by 56.6 µeV and $j=\\tfrac32$ by 11.3 µeV, so $2p_{3/2}$ lies 45.3 µeV (10.97 GHz) above the still
      degenerate pair $2s_{1/2}$, $2p_{1/2}$.</p>
      <p><b>3. Lamb shift.</b> Coupling to the quantised electromagnetic vacuum (QED) smears the electron's position and raises mainly the $l=0$ states,
      whose wave function is non-zero at the nucleus. Bethe's estimate is</p>
      $$\\Delta E_{Lamb}(ns)\\approx\\frac{4\\alpha^5mc^2}{3\\pi n^3}\\,\\ln\\frac{mc^2}{\\langle\\Delta E\\rangle}\\approx4.3\\ \\mu\\text{eV}\\ (n=2),$$
      <p>with $\\langle\\Delta E\\rangle\\approx17.8$ Ry; the measured $2s_{1/2}$–$2p_{1/2}$ splitting is 1057.8 MHz = 4.37 µeV. The Dirac equation alone predicts zero.</p>
      <p><b>4. Zeeman effect (weak field).</b> $\\hat H_Z=\\frac{\\mu_B B}{\\hbar}(\\hat L_z+g_s\\hat S_z)$ with $g_s\\approx2$. When $\\mu_BB$ is small compared with the fine-structure
      splitting, first-order perturbation theory in the $|j\\,m_j\\rangle$ basis gives</p>
      <div class="callout">
      $$\\Delta E_Z=g_J\\,m_j\\,\\mu_B B,\\qquad g_J=1+\\frac{j(j+1)+s(s+1)-l(l+1)}{2j(j+1)}$$
      </div>
      <p>so $g_J=2$ for $2s_{1/2}$, $2/3$ for $2p_{1/2}$ and $4/3$ for $2p_{3/2}$, and all 8 states separate.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Energies.</b> Every state's energy at stage $k$ is the sum of the analytic shifts above: $+\\Delta_{fs}$ for $2p_{3/2}$ (stage 2),
        $+\\Delta_{Lamb}$ for $2s_{1/2}$ (stage 3) and $g_Jm_j\\mu_BB$ (stage 4). The zero of energy is the Dirac $j=\\tfrac12$ level, i.e. the common
        fine-structure shift of the whole shell is subtracted so that the diagram shows the splittings.</li>
        <li><b>Smooth transitions.</b> Each term carries a weight $W_k\\in[0,1]$ that relaxes towards 1 (on) or 0 (off) as $\\dot W=4.5\\,(W_{target}-W)$; the
        magnitudes and the field relax the same way, so changes glide instead of jumping. While paused, changes are applied instantly.</li>
        <li><b>Diagram.</b> Each stage column draws the levels; levels closer than 0.4 % of the vertical range are merged into one bar with a multiplicity label (×2, ×4…).
        Thin lines connect the same state between stages. Arrows give the fine-structure and Lamb splittings in µeV and in frequency $\\nu=\\Delta E/h$.</li>
        <li><b>Zeeman fan.</b> The lower plot shows the straight lines $E_i(B)=E_i^{(3)}+g_Jm_j\\mu_BB$ for $0\\le\\mu_BB\\le20$ µeV, with the current field marked.
        The real field is $B=(\\mu_BB)/\\mu_B$.</li>
        <li><b>Check.</b> The metric "distinct levels" counts the separate energies at the last stage: 1 → 2 → 3 → 8 as the terms are added
        (fewer at exact level crossings).</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li>Press <b>Animate the stages</b>: the 8-fold level splits into 4 + 4 (fine structure), then 2 + 2 + 4 (Lamb), then 8 separate levels (Zeeman).</li>
        <li><b>Slopes in the fan.</b> The slope of each line is $g_Jm_j$: ±2 for $2p_{3/2}$, $m_j=\\pm\\tfrac32$; ±1 for $2s_{1/2}$; ±2/3 for $2p_{3/2}$, $m_j=\\pm\\tfrac12$; only ±1/3 for $2p_{1/2}$.</li>
        <li><b>A level crossing.</b> $2s_{1/2}(m_j=-\\tfrac12)$ falls with slope −1 from 4.3 µeV while $2p_{1/2}(m_j=+\\tfrac12)$ rises with slope +1/3: they cross at
        $\\mu_BB=4.3\\cdot\\tfrac34\\approx3.2$ µeV, i.e. $B\\approx56$ mT. Lamb and Retherford used exactly such Zeeman tuning in 1947. Step $\\mu_BB$ from 3 to 3.5 and the two levels swap order.</li>
        <li>Switch the <b>Lamb shift off</b>: $2s_{1/2}$ and $2p_{1/2}$ become degenerate again — the Dirac prediction.</li>
        <li>Tick <b>Sweep the B field</b> and follow the moving marker along the fan; at $\\mu_BB\\approx13.6$ µeV $2p_{3/2}(m_j=-\\tfrac32)$ meets $2s_{1/2}(m_j=+\\tfrac12)$.</li>
      </ol>

      <h4>Limitations &amp; further reading</h4>
      <p>All shifts are first-order and linear. The weak-field Zeeman formula needs $\\mu_BB\\ll45$ µeV; at the top of the slider range the field mixes the
      $2p_{1/2}$ and $2p_{3/2}$ states with equal $m_j$ (onset of the Paschen–Back regime), so the true levels bend away from straight lines. Hyperfine structure
      (≈ 0.7 µeV for $2s$) is ignored. See D. J. Griffiths &amp; D. F. Schroeter, <em>Introduction to Quantum Mechanics</em>, ch. 7; H. A. Bethe &amp; E. E. Salpeter,
      <em>Quantum Mechanics of One- and Two-Electron Atoms</em>; W. E. Lamb &amp; R. C. Retherford, <em>Phys. Rev.</em> 72, 241 (1947).</p>`,

    mount(api) {
      const P = api.params;
      const M = api.metrics([
        { id: "fs", label: "Fine structure $\\Delta E(2p_{3/2}-2p_{1/2})$" },
        { id: "lamb", label: "Lamb shift $\\Delta E(2s_{1/2}-2p_{1/2})$" },
        { id: "B", label: "Actual field strength $B$" },
        { id: "lev", label: "Distinct energy levels" },
      ]);
      const plots = api.plots([
        { id: "diag", title: "n = 2 shell: degeneracy lifted by successive perturbations", span: 2, aspect: 0.46, maxHeight: 560,
          xlim: [-0.55, 4.95], ylim: [-20, 70], grid: false, ylabel: "Energy shift ΔE (µeV)", xtickFormat: () => "" },
        { id: "fan", title: "Zeeman fan: ΔE(B)", span: 2, aspect: 0.3, minHeight: 220, xlim: [0, 20], ylim: [-30, 80],
          xlabel: "μ_B B (µeV)", ylabel: "ΔE (µeV)" },
      ]);
      const W = [1, 0, 0, 0], WT = [1, 1, 1, 1];
      let fsCur = 0, lambCur = 0, Bcur = 0, seqT = -1, sweepPh = 0;
      let ylo = -20, yhi = 70, flo = -30, fhi = 80;

      function hardClear(p) {
        const c = p.ctx;
        if (typeof c.reset === "function") c.reset(); else p.canvas.width = p.canvas.width;
        p._clipped = false;
        p.clear();
      }
      const delta = (i, j, fs, lamb, B) => {
        const s = STATES[i];
        if (j === 1) return s.key === "2p3" ? fs : 0;
        if (j === 2) return s.key === "2s" ? lamb : 0;
        if (j === 3) return s.g * s.mj * B;
        return 0;
      };
      /** Energy at stage k (weighted sum of the shifts). */
      function energy(i, k, B) {
        let e = 0;
        for (let j = 1; j <= k; j++) e += W[j] * delta(i, j, fsCur, lambCur, B === undefined ? Bcur : B);
        return e;
      }
      function targets() {
        for (let j = 1; j <= 3; j++) {
          const on = !!P[FLAGS[j]];
          WT[j] = seqT >= 0 ? (on && seqT > 0.7 + (j - 1) * 1.4 ? 1 : 0) : on ? 1 : 0;
        }
      }
      function limitsFor(Bmax) {
        let lo = 0, hi = 0;
        for (let i = 0; i < 8; i++) for (let k = 0; k < 4; k++) for (const B of [Bcur, Bmax]) {
          const e = energy(i, k, B); lo = Math.min(lo, e); hi = Math.max(hi, e);
        }
        const span = Math.max(hi - lo, 12);
        return [lo - 0.12 * span, hi + 0.2 * span];
      }
      function groups(k) {
        const es = STATES.map((_, i) => energy(i, k));
        const idx = es.map((e, i) => i).sort((a, b) => es[a] - es[b]);
        const tol = Math.max(1e-6, (yhi - ylo) * 0.004);
        const out = [];
        for (const i of idx) {
          const g = out[out.length - 1];
          if (g && Math.abs(es[i] - g.e) < tol) g.m.push(i); else out.push({ e: es[i], m: [i] });
        }
        return out;
      }

      return {
        reset() {
          W[1] = W[2] = W[3] = 0; fsCur = P.fsShift; lambCur = P.lambShift; Bcur = P.B; seqT = 0.001; sweepPh = 0;
          [ylo, yhi] = limitsFor(P.B);
        },
        onAction(id) { if (id === "seq") { seqT = 0.001; W[1] = W[2] = W[3] = 0; api.play(); } },
        onParam(id, v) {
          if (id === "sweep" && v) sweepPh = Math.acos(PM.clamp(1 - Bcur / 10, -1, 1)) / (2 * Math.PI);
          if (!api.isPlaying) { // while paused, apply changes instantly
            seqT = -1; targets();
            for (let j = 1; j <= 3; j++) W[j] = WT[j];
            fsCur = P.fsShift; lambCur = P.lambShift; Bcur = P.B;
            [ylo, yhi] = limitsFor(Bcur); [flo, fhi] = [ylo, yhi];
          }
        },
        step(dt) {
          if (seqT >= 0) { seqT += dt; if (seqT > 0.7 + 3 * 1.4 + 0.8) seqT = -1; }
          targets();
          const a = 1 - Math.exp(-4.5 * dt);
          for (let j = 1; j <= 3; j++) W[j] += (WT[j] - W[j]) * a;
          fsCur += (P.fsShift - fsCur) * a;
          lambCur += (P.lambShift - lambCur) * a;
          if (P.sweep) {
            sweepPh += dt * 0.12;
            Bcur = 10 * (1 - Math.cos(2 * Math.PI * sweepPh));
            api.setControl("B", { value: Math.round(Bcur * 2) / 2 });
          } else Bcur += (P.B - Bcur) * a;
          const [tl, th] = limitsFor(P.sweep ? 20 : Bcur);
          const b = 1 - Math.exp(-3 * dt);
          ylo += (tl - ylo) * b; yhi += (th - yhi) * b;
          let lo = 0, hi = 0;
          for (let i = 0; i < 8; i++) for (const B of [0, 20]) { const e = energy(i, 2) + W[3] * STATES[i].g * STATES[i].mj * B; lo = Math.min(lo, e); hi = Math.max(hi, e); }
          const sp = Math.max(hi - lo, 12);
          flo += (lo - 0.1 * sp - flo) * b; fhi += (hi + 0.1 * sp - fhi) * b;
        },
        render() {
          const pd = plots.diag;
          pd.setLimits(null, [ylo, yhi]);
          hardClear(pd);
          pd.hline(0, { color: PlotColors.muted, dash: [2, 5], width: 1, alpha: 0.5 });
          const HW = 0.3;
          // connecting lines between stages
          for (let k = 0; k < 3; k++) {
            for (let i = 0; i < 8; i++) {
              const s = STATES[i];
              pd.segment(k + HW, energy(i, k), k + 1 - HW, energy(i, k + 1), { color: TERM_COLOR[s.key], dash: s.dash.length ? s.dash : [], width: 1.3, alpha: 0.6 });
            }
          }
          // level bars (degenerate groups merged)
          for (let k = 0; k < 4; k++) {
            const act = k === 0 || W[k] > 0.02;
            for (const g of groups(k)) {
              const keys = new Set(g.m.map((i) => STATES[i].key));
              const col = keys.size === 1 ? TERM_COLOR[STATES[g.m[0]].key] : PlotColors.text;
              pd.segment(k - HW, g.e, k + HW, g.e, { color: col, width: 3.2, alpha: act ? 1 : 0.5 });
              if (g.m.length > 1) pd.text(k, g.e, "×" + g.m.length, { size: 10.5, color: PlotColors.muted, align: "center", baseline: "bottom", dy: -4 });
            }
            // stage heading
            const on = k === 0 || WT[k] > 0.5;
            const col = on ? PlotColors.text : PlotColors.muted;
            pd.text(k, yhi, STAGES[k].h, { align: "center", baseline: "top", dy: 8, size: 13, bold: true, color: col });
            pd.text(k, yhi, STAGES[k].d + (on ? "" : " (off)"), { align: "center", baseline: "top", dy: 26, size: 11.5, color: PlotColors.muted });
          }
          // splitting arrows (with frequency equivalent)
          const gap = (k, iHi, iLo, txt) => {
            const e1 = energy(iHi, k), e0 = energy(iLo, k);
            if (Math.abs(e1 - e0) < (yhi - ylo) * 0.03) return;
            pd.arrow(k + 0.12, e0, k + 0.12, e1, { color: PlotColors.accent3, width: 1.2, head: 6 });
            pd.arrow(k + 0.12, e1, k + 0.12, e0, { color: PlotColors.accent3, width: 1.2, head: 6 });
            pd.text(k + 0.17, (e0 + e1) / 2, txt, { size: 11, color: PlotColors.accent3, bg: "#0f151c" });
          };
          if (W[1] > 0.05) gap(1, 4, 2, `${PM.fmt(fsCur * W[1], 1)} µeV ≈ ${PM.fmt(fsCur * W[1] * GHZ_PER_UEV, 2)} GHz`);
          if (W[2] > 0.05) gap(2, 0, 2, `${PM.fmt(lambCur * W[2], 2)} µeV ≈ ${PM.fmt(lambCur * W[2] * GHZ_PER_UEV * 1000, 0)} MHz`);
          // legend identifying the states
          pd.legend(STATES.map((s) => ({ label: `${s.term}  m_j = ${fracMJ(s.mj)}  (g = ${s.g === 2 ? "2" : s.g < 1 ? "2/3" : "4/3"})`, color: TERM_COLOR[s.key], dash: s.dash })), "br");

          // Zeeman fan
          const pf = plots.fan;
          pf.setLimits(null, [flo, fhi]);
          hardClear(pf);
          const Bs = [0, 20];
          for (let i = 0; i < 8; i++) {
            const s = STATES[i], e0 = energy(i, 2);
            const ys = Bs.map((B) => e0 + W[3] * s.g * s.mj * B);
            pf.line(Bs, ys, { color: TERM_COLOR[s.key], dash: s.dash, width: 1.8, alpha: 0.9 });
          }
          pf.vline(Bcur, { color: PlotColors.accent3, width: 1.4, dash: [5, 4] });
          for (let i = 0; i < 8; i++) pf.circle(Bcur, energy(i, 3), 4, { px: true, color: TERM_COLOR[STATES[i].key], stroke: "#0f151c" });
          pf.label(W[3] < 0.05 && !P.zeeman ? "Zeeman term off: levels independent of B" : `μ_B B = ${PM.fmt(Bcur, 1)} µeV  ≈  ${PM.fmt((Bcur / MU_B) * 1000, 0)} mT`, "tl");

          // metrics
          const fsE = fsCur * W[1], lE = lambCur * W[2];
          M.set("fs", `${PM.fmt(fsE, 1)} µeV`);
          M.set("lamb", `${PM.fmt(lE, 2)} µeV`);
          M.set("B", `≈ ${PM.fmt((Bcur / MU_B) * 1000, 0)} mT`);
          M.set("lev", String(groups(3).length) + " / 8");
          api.setTime(`μ_B B = ${PM.fmt(Bcur, 1)} µeV`);
        },
      };
    },
  });
})();
