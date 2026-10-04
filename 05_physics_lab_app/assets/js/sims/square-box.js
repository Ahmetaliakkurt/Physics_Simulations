/* 2D wave packet in an infinite square well — EXACT time evolution in the eigenstate basis (DST-I), re-evaluated live every frame. */
(function () {
  "use strict";

  const M = 255, MM = M * M;                       // interior grid points / sine modes per axis
  const QREV = 65536;                              // autocorrelation samples over one revival period
  const ENV = 900;                                 // envelope columns for the autocorrelation plot

  App.register({
    id: "square-box",
    category: "quantum",
    order: 37,
    title: "Wave Packet in a Square Box",
    icon: "🟦",
    subtitle: "Exact time evolution of a Gaussian packet in a 2D infinite square well, from its eigenstate expansion: wall reflections, interference, and quantum revivals at T_rev = 4mL²/πħ.",
    notes: [{
      type: "info",
      html: "The square box is <b>separable</b>: its eigenstates are products $\\sin(n\\pi x/L)\\sin(m\\pi y/L)$ and the motion splits into " +
        "two independent 1D problems. The system is therefore integrable — its classical counterpart is not chaotic, and the “scars” typical " +
        "of chaotic quantum billiards do not appear here. Quantum chaos shows up in irregular shapes whose classical dynamics is chaotic " +
        "(the stadium or the Sinai billiard). In exchange, the integer energy spectrum of the square box makes the packet return " +
        "<b>exactly</b> to its initial state at $t=T_{rev}$ — use the jump buttons to see it.",
    }],
    animated: true,
    speed: { min: 0.1, max: 20, value: 1, step: 0.1 },
    controls: [
      { id: "L", type: "slider", label: "Box size $L$", min: 10, max: 30, step: 1, value: 20 },
      { id: "mass", type: "slider", label: "Mass $m$", min: 0.5, max: 6, step: 0.5, value: 3 },
      { id: "sigma", type: "slider", label: "Packet width $\\sigma$", min: 0.5, max: 3, step: 0.1, value: 1 },
      { id: "kx", type: "slider", label: "Initial momentum $k_x$", min: -25, max: 25, step: 1, value: 18,
        help: "255×255 sine modes; the largest representable wavenumber is $255\\pi/L$ (≈ 26.7 for L = 30)." },
      { id: "ky", type: "slider", label: "Initial momentum $k_y$", min: -25, max: 25, step: 1, value: 14 },
      { type: "section", label: "Jump in time (exact evolution)" },
      { id: "jq", type: "button", label: "t → T_rev / 4" },
      { id: "jh", type: "button", label: "t → T_rev / 2  (mirror image)" },
      { id: "jf", type: "button", label: "t → T_rev  (full revival)" },
      { type: "section", label: "Display" },
      { id: "classic", type: "checkbox", label: "Show the classical particle and the ⟨r⟩ trail", value: true, live: true },
    ],
    theory: `
      <h4>The physical system</h4>
      <p>A particle of mass $m$ is confined to a square of side $L$ ($0\\lt x,y\\lt L$) by infinitely high walls — a quantum billiard. At $t=0$
      it is a Gaussian packet at the centre of the box with mean wave vector $(k_x,k_y)$,
      $\\psi(x,y,0)\\propto e^{-[(x-L/2)^2+(y-L/2)^2]/2\\sigma^2}\\,e^{i(k_xx+k_yy)}$ (so $|\\psi|^2$ has standard deviation $\\sigma/\\sqrt2$ per axis).
      Units: $\\hbar=1$; $L$ and $\\sigma$ are lengths, $k$ inverse lengths, energies are $\\hbar^2k^2/2m$ and the classical velocity is
      $\\mathbf v=\\hbar\\mathbf k/m$.</p>

      <h4>Equations being solved</h4>
      <p>The free Schrödinger equation inside the box with Dirichlet conditions on the walls,</p>
      $$ i\\hbar\\frac{\\partial\\psi}{\\partial t}=-\\frac{\\hbar^2}{2m}\\nabla^2\\psi,\\qquad \\psi=0\\ \\text{on the walls}. $$
      <p>Separation of variables gives the normalised eigenstates and energies</p>
      $$ \\phi_{nm}(x,y)=\\frac{2}{L}\\sin\\frac{n\\pi x}{L}\\,\\sin\\frac{m\\pi y}{L},\\qquad
         E_{nm}=E_1\\,(n^2+m^2),\\quad E_1=\\frac{\\pi^2\\hbar^2}{2mL^2},\\qquad n,m=1,2,\\dots $$
      <p>and any state evolves as $\\psi(t)=\\sum_{n,m}c_{nm}\\,e^{-iE_{nm}t/\\hbar}\\,\\phi_{nm}$ with $c_{nm}=\\langle\\phi_{nm}|\\psi(0)\\rangle$.
      Because all $E_{nm}$ are integer multiples of $E_1$, every phase returns to a multiple of $2\\pi$ at the same moment:</p>
      <div class="callout">$$ T_{rev}=\\frac{2\\pi\\hbar}{E_1}=\\frac{4mL^2}{\\pi\\hbar},\\qquad \\psi(x,y,T_{rev})=\\psi(x,y,0)\\ \\text{exactly.} $$</div>
      <p>At $T_{rev}/2$ the phases are $e^{-i\\pi(n^2+m^2)}=(-1)^{n+m}$, and since $\\phi_{nm}(L-x,L-y)=(-1)^{n+m}\\phi_{nm}(x,y)$
      the wave function becomes exactly the initial packet point-reflected through the box centre, $\\psi(x,y,T_{rev}/2)=\\psi(L-x,L-y,0)$, i.e. moving with momentum $-\\mathbf k$. At rational fractions such as
      $T_{rev}/4$ the wave function is a superposition of a few displaced copies of the packet (fractional revivals). The classical
      transit time $2mL/\\hbar k$ is much shorter, so in between the packet spreads over the box and looks like a random interference pattern.
      The return is monitored by the autocorrelation
      $|A(t)|^2=|\\langle\\psi(0)|\\psi(t)\\rangle|^2=\\big|\\sum_{n,m}|c_{nm}|^2e^{-iE_{nm}t/\\hbar}\\big|^2$.</p>

      <h4>How the simulation solves them</h4>
      <ul>
        <li><b>Basis and grid.</b> The box is sampled at $255\\times255$ interior points $x_i=i\\,L/256$; on this grid the 255 discrete sine vectors
        $S_{ni}=\\sqrt{2/256}\\,\\sin(\\pi ni/256)$ are exactly orthonormal (a type-I discrete sine transform, DST-I), and they are the samples of
        the box eigenfunctions. The initial packet is projected onto them, giving $255\\times255$ coefficients.</li>
        <li><b>Separability.</b> The initial packet is a product $f(x)g(y)$ and $E_{nm}=E_n+E_m$, so $c_{nm}=a_nb_m$ and
        $\\psi(x,y,t)=u(x,t)\\,v(y,t)$ with
        $$ u(x_i,t)=\\sum_{n=1}^{255}a_n\\,e^{-iE_1n^2t/\\hbar}\\,S_{ni}\\quad(\\text{same for }v). $$
        Each frame therefore costs two 1D sine sums of $255^2$ operations plus the outer product for $|\\psi|^2$ — no time stepping at all.
        This is an <b>exact</b> evolution of the grid representation: there is no integration error to accumulate, and the jump buttons
        evaluate $T_{rev}/4$, $T_{rev}/2$ or $T_{rev}$ directly. Phases are reduced modulo $2\\pi$ before the cosine is taken to keep full
        precision at large $t$.</li>
        <li><b>Plots.</b> Left: $|\\psi|^2$. Right: phase portrait — hue $=\\arg\\psi$, brightness $=|\\psi|$. Bottom: $|A(t)|^2$ over one
        full revival period. It is the product $A_x A_y$ of two 1D sums $A_x(t)=\\sum_n|a_n|^2e^{-iE_1n^2t/\\hbar}$; at the 65 536 sample times
        $t_q=qT_{rev}/65536$ these become $\\sum_n|a_n|^2e^{-2\\pi i\\,n^2q/65536}$, evaluated exactly by folding $n^2$ modulo 65 536 and one FFT.
        The plot shows the upper envelope of those samples.</li>
        <li><b>Classical comparison.</b> The dashed white circle is a classical point particle with the same start and $\\mathbf v=\\hbar\\mathbf k/m$,
        reflecting elastically off the walls; the teal dot and trail are the quantum mean position $\\langle\\mathbf r\\rangle$, which follows it
        while the packet is compact (Ehrenfest's theorem).</li>
        <li><b>Checks.</b> The norm $\\sum|\\psi|^2\\Delta x^2$ of the evaluated grid wave function stays 1 to rounding error, and
        $\\langle E\\rangle=\\sum E_{nm}|c_{nm}|^2$ is constant by construction; $|A|^2$ returns to 1.0000 at $T_{rev}$.</li>
      </ul>

      <h4>What to try</h4>
      <ul>
        <li><b>Full revival.</b> With the defaults ($L=20$, $m=3$) $T_{rev}=4\\cdot3\\cdot400/\\pi\\approx1528$. Press "t → T_rev": the scrambled
        pattern collapses back into the original packet and $|A|^2=1.0000$.</li>
        <li><b>Mirror revival.</b> Press "t → T_rev/2": the packet reappears at the point reflected through the box centre (the centre itself here,
        so the density looks like the initial one), but with momentum $-\\mathbf k$ — the phase stripes in the phase portrait run the other way.
        A packet with reversed momentum is practically orthogonal to the original, so $|A|^2\\approx0$ even though $|\\psi|^2$ is identical.</li>
        <li><b>Fractional revival.</b> At "t → T_rev/4" each 1D factor is an equal-weight superposition of the packet and its mirror image,
        $u=\\tfrac{1}{\\sqrt2}\\big(e^{-i\\pi/4}f(x)-e^{i\\pi/4}f(L-x)\\big)$. Here both copies sit at the centre and interfere
        into a checkerboard, and $|A|^2=(1/2)^2=0.2500$ exactly.</li>
        <li><b>Ehrenfest and spreading.</b> Play at speed ×1: for the first few wall bounces ⟨r⟩ (teal) tracks the classical particle (white);
        once the packet has spread over the box (velocity spread $\\Delta v=\\hbar/\\sqrt2m\\sigma\\approx0.24$, so after $t\\sim L/\\Delta v\\approx80$) ⟨r⟩ settles towards the centre while the classical particle
        keeps bouncing. Increase $m$ or $\\sigma$ to delay this.</li>
        <li><b>Revival scaling.</b> $T_{rev}\\propto mL^2$: halve $L$ to 10 and the revival time drops four-fold to ≈ 382.</li>
      </ul>

      <h4>Limitations &amp; further reading</h4>
      <p>The walls are perfectly hard, the box is ideal and the representation is limited to 255 modes per axis ($|k|\\lesssim255\\pi/L$).
      Being separable, the square is the textbook integrable billiard; quantum signatures of chaos (level repulsion, scarred eigenstates)
      require non-separable shapes such as the Bunimovich stadium or the Sinai billiard. Reading: R. W. Robinett, "Quantum wave packet revivals",
      <i>Physics Reports</i> 392, 1 (2004); Griffiths &amp; Schroeter, <i>Introduction to Quantum Mechanics</i>, §2.2 (infinite square well; its end-of-chapter problems include the revival time);
      H.-J. Stöckmann, <i>Quantum Chaos: An Introduction</i>; E. J. Heller, <i>The Semiclassical Way to Dynamics and Spectroscopy</i>.</p>`,

    mount(api) {
      const PR = api.params;
      // orthonormal sine table S[n][i] = √(2/(M+1))·sin(π n i/(M+1)), n,i = 1..M (row n-1)
      const S = new Float64Array(MM), sq = Math.sqrt(2 / (M + 1));
      for (let n = 1; n <= M; n++) for (let i = 1; i <= M; i++) S[(n - 1) * M + i - 1] = sq * Math.sin((Math.PI * n * i) / (M + 1));
      const aR = new Float64Array(M), aI = new Float64Array(M), bR = new Float64Array(M), bI = new Float64Array(M);
      const a2 = new Float64Array(M), b2 = new Float64Array(M);
      const uR = new Float64Array(M), uI = new Float64Array(M), vR = new Float64Array(M), vI = new Float64Array(M);
      const phR = new Float64Array(M + 1), phI = new Float64Array(M + 1);
      const dens = new Float64Array(MM);
      const envY = new Float64Array(ENV), envX = new Float64Array(ENV);
      const TRL = 400, trX = new Float64Array(TRL), trY = new Float64Array(TRL), tdX = new Float64Array(TRL), tdY = new Float64Array(TRL);
      const qR = new Float64Array(QREV), qI = new Float64Array(QREV), q2R = new Float64Array(QREV), q2I = new Float64Array(QREV);
      const off = document.createElement("canvas"); off.width = M; off.height = M;
      const offCtx = off.getContext("2d"), img = offCtx.createImageData(M, M);
      let L, dx, Es, Trev, t, initMax, Emean, trN, trHead, ex, ey, A2, nrm;

      const Mt = api.metrics([
        { id: "t", label: "Time $t$" },
        { id: "trev", label: "Revival time $T_{rev}=4mL^2/\\pi\\hbar$" },
        { id: "E", label: "Mean energy $\\langle E\\rangle$" },
        { id: "ac", label: "Autocorrelation $|\\langle\\psi_0|\\psi_t\\rangle|^2$" },
        { id: "nrm", label: "Norm $\\sum|\\psi|^2\\Delta x^2$" },
      ]);
      const plots = api.plots([
        { id: "dens", title: "Probability density |ψ(x,y,t)|²", aspect: 1, xlim: [0, 20], ylim: [0, 20], equal: true, xlabel: "x", ylabel: "y", grid: false, colorbar: true, margin: { r: 86 }, maxHeight: 620 },
        { id: "phase", title: "Phase portrait: hue = arg ψ, brightness = |ψ|", aspect: 1, xlim: [0, 20], ylim: [0, 20], equal: true, xlabel: "x", ylabel: "y", grid: false, margin: { r: 86 }, maxHeight: 620 },
        { id: "ac", title: "Autocorrelation |⟨ψ(0)|ψ(t)⟩|² over one revival period", span: 2, aspect: 0.22, minHeight: 170, xlim: [0, 1], ylim: [0, 1.05], xlabel: "t", ylabel: "|A(t)|²" },
      ]);

      /** 1D profile f(ξ) = e^{-(ξ-ξ0)²/2σ²} e^{ikξ} → orthonormal sine coefficients (Σ|c|² = 1). */
      function project(k, cR, cI, c2) {
        const x0 = L / 2, s2 = 2 * PR.sigma * PR.sigma, fR = uR, fI = uI;
        let nrm = 0;
        for (let i = 0; i < M; i++) {
          const x = (i + 1) * dx, g = Math.exp(-((x - x0) * (x - x0)) / s2);
          fR[i] = g * Math.cos(k * x); fI[i] = g * Math.sin(k * x); nrm += g * g;
        }
        const f = 1 / Math.sqrt(nrm);
        for (let n = 0; n < M; n++) {
          let sr = 0, si = 0;
          const o = n * M;
          for (let i = 0; i < M; i++) { sr += S[o + i] * fR[i]; si += S[o + i] * fI[i]; }
          cR[n] = sr * f; cI[n] = si * f; c2[n] = cR[n] * cR[n] + cI[n] * cI[n];
        }
      }
      /** u(ξ,t) = Σ_n c_n e^{-iE1 n² t} S[n][ξ]; returns the 1D overlap ⟨φ0|φt⟩. */
      function evolve1D(cR, cI, c2, oR, oI) {
        oR.fill(0); oI.fill(0);
        let ar = 0, ai = 0;
        for (let n = 1; n <= M; n++) {
          const zr = phR[n], zi = phI[n], k = n - 1;
          const wr = cR[k] * zr - cI[k] * zi, wi = cR[k] * zi + cI[k] * zr;
          ar += c2[k] * zr; ai += c2[k] * zi;
          if (wr * wr + wi * wi < 1e-30) continue;
          const o = k * M;
          for (let i = 0; i < M; i++) { const sv = S[o + i]; oR[i] += wr * sv; oI[i] += wi * sv; }
        }
        return [ar, ai];
      }
      function evalAt(time) {
        const a = PM.mod(Es * time, 2 * Math.PI);
        for (let n = 1; n <= M; n++) { const ang = -PM.mod(a * n * n, 2 * Math.PI); phR[n] = Math.cos(ang); phI[n] = Math.sin(ang); }
        const [xr, xi] = evolve1D(aR, aI, a2, uR, uI);
        const [yr, yi] = evolve1D(bR, bI, b2, vR, vI);
        const Ar = xr * yr - xi * yi, Ai = xr * yi + xi * yr;
        A2 = Ar * Ar + Ai * Ai;
        let mx = 0, sx = 0, sy = 0, nu = 0, nv = 0;
        const inv = 1 / (dx * dx);
        for (let i = 0; i < M; i++) { const q = uR[i] * uR[i] + uI[i] * uI[i]; sx += q * (i + 1) * dx; nu += q; }
        for (let j = 0; j < M; j++) nv += vR[j] * vR[j] + vI[j] * vI[j];
        nrm = nu * nv; // grid values are u_i·v_j/Δx, so Σ|ψ|²Δx² = (Σ|u|²)(Σ|v|²)
        for (let j = 0; j < M; j++) {
          const pv = (vR[j] * vR[j] + vI[j] * vI[j]) * inv, o = j * M;
          sy += pv * dx * dx * (j + 1) * dx;
          for (let i = 0; i < M; i++) { const d = pv * (uR[i] * uR[i] + uI[i] * uI[i]); dens[o + i] = d; if (d > mx) mx = d; }
        }
        ex = sx; ey = sy;
        return mx;
      }
      function pushTrail() { trX[trHead % TRL] = ex; trY[trHead % TRL] = ey; trHead++; trN = Math.min(trN + 1, TRL); }
      function foldFFT(c2, oR, oI) {
        oR.fill(0); oI.fill(0);
        for (let n = 1; n <= M; n++) oR[(n * n) % QREV] += c2[n - 1];
        PM.fft(oR, oI, false);
      }
      function buildAutocorr() {
        // A(t_q) = A_x·A_y,  A_x(t_q) = Σ_n |a_n|² e^{-2πi n² q/Q},  t_q = q·T_rev/Q  (folded FFT, exact at the sample times)
        foldFFT(a2, qR, qI); foldFFT(b2, q2R, q2I);
        envY.fill(0);
        for (let q = 0; q < QREV; q++) {
          const r = qR[q] * q2R[q] - qI[q] * q2I[q], i = qR[q] * q2I[q] + qI[q] * q2R[q], v = r * r + i * i;
          const b = Math.min(ENV - 1, Math.floor((q / QREV) * ENV));
          if (v > envY[b]) envY[b] = v;
        }
        envY[ENV - 1] = Math.max(envY[ENV - 1], envY[0]); // t = T_rev ≡ t = 0
        for (let b = 0; b < ENV; b++) envX[b] = ((b + 0.5) / ENV) * Trev;
        envX[0] = 0; envX[ENV - 1] = Trev;
      }
      function drawPhase(p, vmax) {
        const d = img.data, s = 1 / Math.sqrt(vmax), r3 = Math.sqrt(3) / 2;
        for (let j = 0; j < M; j++) {
          const row = (M - 1 - j) * M, vr = vR[j], vi = vI[j];
          for (let i = 0; i < M; i++) {
            const re = vr * uR[i] - vi * uI[i], im = vr * uI[i] + vi * uR[i];
            const a = Math.sqrt(re * re + im * im);
            let br = (a / dx) * s; if (br > 1) br = 1;
            const u = a > 0 ? re / a : 0, v = a > 0 ? im / a : 0, k = (row + i) * 4, g = br * 255;
            d[k] = g * (0.5 + 0.5 * u);
            d[k + 1] = g * (0.5 + 0.5 * (-0.5 * u + r3 * v));
            d[k + 2] = g * (0.5 + 0.5 * (-0.5 * u - r3 * v));
            d[k + 3] = 255;
          }
        }
        offCtx.putImageData(img, 0, 0);
        p.custom((c, pl) => {
          c.imageSmoothingEnabled = true;
          const x0 = pl.X(0.5 * dx), x1 = pl.X(L - 0.5 * dx), y0 = pl.Y(L - 0.5 * dx), y1 = pl.Y(0.5 * dx);
          c.drawImage(off, x0, y0, x1 - x0, y1 - y0);
        });
      }
      function reflect(u) { const w = PM.mod(u, 2 * L); return w > L ? 2 * L - w : w; }
      function jump(to) { t = to; trN = 0; trHead = 0; api.invalidate(); }

      return {
        reset() {
          L = PR.L; dx = L / (M + 1);
          Es = (Math.PI * Math.PI) / (2 * PR.mass * L * L);
          Trev = (2 * Math.PI) / Es;
          project(PR.kx, aR, aI, a2);
          project(PR.ky, bR, bI, b2);
          Emean = 0;
          for (let n = 1; n <= M; n++) Emean += Es * n * n * (a2[n - 1] + b2[n - 1]);
          buildAutocorr();
          for (const k of ["dens", "phase"]) plots[k].setLimits([0, L], [0, L]);
          plots.ac.setLimits([0, Trev]);
          t = 0; trN = 0; trHead = 0;
          initMax = evalAt(0);
        },
        step(dt) { t += dt; },
        onAction(id) {
          if (id === "jq") jump(Trev / 4);
          else if (id === "jh") jump(Trev / 2);
          else if (id === "jf") jump(Trev);
        },
        render() {
          const mx = evalAt(t);
          if (api.isPlaying || trN === 0) pushTrail();
          const vmax = Math.max(Math.min(mx * 0.85, initMax * 1.5), 1e-12);
          const pd = plots.dens, pp = plots.phase;
          pd.clear();
          pd.heatmap(dens, M, M, { x0: 0.5 * dx, x1: L - 0.5 * dx, y0: 0.5 * dx, y1: L - 0.5 * dx, vmin: 0, vmax, cmap: "inferno", colorbar: "|ψ|²" });
          pp.clear();
          drawPhase(pp, vmax);
          for (const p of [pd, pp]) p.rect(0, 0, L, L, { fill: false, stroke: PlotColors.accent, strokeWidth: 2.5 });
          if (PR.classic) {
            const vx = PR.kx / PR.mass, vy = PR.ky / PR.mass;
            const cx = reflect(L / 2 + vx * t), cy = reflect(L / 2 + vy * t);
            if (trN > 1) {
              const n = trN;
              for (let k = 0; k < n; k++) { const idx = (trHead - n + k) % TRL; tdX[k] = trX[idx]; tdY[k] = trY[idx]; }
              pd.line(tdX.subarray(0, n), tdY.subarray(0, n), { color: PlotColors.accent, width: 1.3, alpha: 0.7 });
            }
            pd.circle(ex, ey, 4.5, { px: true, color: PlotColors.accent, stroke: "#0f151c" });
            pd.circle(cx, cy, 5, { px: true, fill: false, stroke: PlotColors.white, strokeWidth: 1.8 });
            pd.legend([{ label: "⟨r⟩ (quantum)", color: PlotColors.accent, type: "dot" }, { label: "classical particle", color: PlotColors.white, type: "dot" }], "br");
          }
          pd.label(`t = ${PM.fmt(t, 2)}`, "tl");
          pp.label(`t / T_rev = ${PM.fmt(t / Trev, 4)}`, "tl");

          const pa = plots.ac;
          pa.clear();
          pa.fill(envX, envY, 0, { color: PlotColors.accent2, alpha: 0.35 });
          pa.line(envX, envY, { color: PlotColors.accent2, width: 1.2 });
          for (const [f, s] of [[0.25, "T/4"], [0.5, "T/2"], [0.75, "3T/4"]]) {
            pa.vline(f * Trev, { color: PlotColors.muted, dash: [3, 4], alpha: 0.6 });
            pa.text(f * Trev, 0.97, s, { dx: 4, align: "left", size: 11, color: PlotColors.muted });
          }
          const tm = PM.mod(t, Trev);
          pa.vline(tm, { color: PlotColors.accent3, width: 2 });
          pa.circle(tm, A2, 4, { px: true, color: PlotColors.accent3 });

          Mt.set("t", PM.fmt(t, 2));
          Mt.set("trev", PM.fmt(Trev, 1));
          Mt.set("E", PM.fmt(Emean, 2));
          Mt.set("ac", PM.fmt(A2, 4));
          Mt.set("nrm", PM.fmt(nrm, 6));
          api.setTime(`t = ${PM.fmt(t, 2)}`);
        },
      };
    },
  });
})();
