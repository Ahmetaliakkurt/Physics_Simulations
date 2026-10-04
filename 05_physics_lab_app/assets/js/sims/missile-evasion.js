/* Missile evasion — an aircraft escaping a turn-rate-limited pure-pursuit missile, real-time kinematic model. */
App.register({
  id: "missile-evasion",
  category: "special",
  order: 51,
  title: "Missile Evasion (AI Pilot)",
  icon: "✈️",
  subtitle: "An aircraft tries to escape a turn-rate-limited pure-pursuit missile: a live kinematic pursuit–evasion game with a look-ahead AI pilot and running episode statistics.",
  animated: true,
  speed: { min: 0.25, max: 4, value: 1, step: 0.05 },
  notes: [
    { type: "info", html: "The model is purely kinematic: each step the heading changes by at most the turn rate and the position is updated as <b>Δr = v·(cos ψ, sin ψ)·Δt</b>. Forces, mass, drag and thrust are not modelled — this is not a flight-dynamics simulator but an abstract pursuit–evasion game." },
  ],
  controls: [
    { id: "pilot", type: "select", label: "Aircraft pilot", value: "plan", options: [
      { value: "plan", label: "Look-ahead planner (AI)" },
      { value: "simple", label: "Simple heuristic: turn away from the missile" },
    ], help: "Every step the planner simulates a few manoeuvres ahead and picks the one that keeps the aircraft farthest from the missile." },
    { type: "section", label: "Aircraft" },
    { id: "vp", type: "slider", label: "Aircraft speed $v_a$", min: 0.5, max: 3, step: 0.1, value: 1.6, unit: "units/step" },
    { id: "wp", type: "slider", label: "Aircraft turn rate $\\omega_a$", min: 3, max: 15, step: 0.5, value: 9, unit: "°/step" },
    { type: "section", label: "Missile" },
    { id: "vm", type: "slider", label: "Missile speed $v_m$", min: 0.5, max: 4, step: 0.1, value: 2.2, unit: "units/step" },
    { id: "wm", type: "slider", label: "Missile turn rate $\\omega_m$", min: 1, max: 15, step: 0.5, value: 7, unit: "°/step",
      help: "The closer the missile's turn rate gets to the aircraft's, the harder the escape." },
    { id: "fuel", type: "slider", label: "Missile burn time", min: 100, max: 600, step: 10, value: 350, unit: "steps",
      help: "If the missile has not hit after this many steps the aircraft has escaped." },
    { type: "section", label: "Display" },
    { id: "trail", type: "slider", label: "Trail length", min: 0, max: 300, step: 10, value: 90, unit: "steps", live: true },
    { id: "showPlan", type: "checkbox", label: "Show the planned manoeuvre", value: true, live: true, visibleIf: (p) => p.pilot === "plan" },
    { id: "showLock", type: "checkbox", label: "Show the lock-on range", value: true, live: true },
    { id: "clr", type: "button", label: "Reset statistics" },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>Two point vehicles move in a rectangular arena of $140\\times90$ length units: an aircraft (the evader) and a missile (the pursuer). Both fly at
    constant speed — $v_a$ and $v_m$ in units per step — and can only change heading, by at most $\\omega_a$ and $\\omega_m$ degrees per step. Time is discrete:
    one step is one decision/update, and the animation plays 30 steps per second (× the speed control). Each episode the aircraft starts at $(42,45)$ with a random
    heading and the missile launches from a random point on the boundary, pointing at the aircraft. The episode ends with</p>
    <ul>
      <li><b>hit</b> — the distance falls below the kill radius $d_{hit}=3$;</li>
      <li><b>crash</b> — the aircraft leaves the arena (it may not fly out of the box);</li>
      <li><b>escape</b> — the missile runs out of fuel after the chosen burn time.</li>
    </ul>
    <p>The "lock-on" circle of radius 40 is only a visual cue.</p>

    <h4>Equations being solved</h4>
    <p><b>Aircraft</b> (unicycle model with bounded turn rate), with pilot command $u_k$:</p>
    $$\\psi_{k+1}=\\psi_k+\\mathrm{clip}(u_k,-\\omega_a,\\omega_a),\\qquad \\mathbf r_{k+1}=\\mathbf r_k+v_a(\\cos\\psi_{k+1},\\sin\\psi_{k+1}).$$
    <p><b>Missile</b> with <em>pure-pursuit</em> guidance: it tries to point its nose at the aircraft's current position, $\\lambda_k=\\mathrm{atan2}(y_a-y_m,\\,x_a-x_m)$
    being the line-of-sight angle, but its turn is limited:</p>
    <div class="callout">
    $$\\phi_{k+1}=\\phi_k+\\mathrm{clip}\\big(\\mathrm{wrap}(\\lambda_k-\\phi_k),\\,-\\omega_m,\\,\\omega_m\\big),\\qquad \\mathbf r^{m}_{k+1}=\\mathbf r^{m}_k+v_m(\\cos\\phi_{k+1},\\sin\\phi_{k+1})$$
    </div>
    <p>Here $\\mathrm{wrap}$ maps an angle to $(-\\pi,\\pi]$. Flying at constant speed with the maximum turn rate traces a circle of radius $R=v/\\omega$ ($\\omega$ in rad/step).
    With the defaults $R_a=1.6/0.157\\approx10.2$ and $R_m=2.2/0.122\\approx18.0$: the missile is 37 % faster but turns in a much wider circle.</p>
    <p><b>Why evasion is possible.</b> For unlimited turn rate, a faster pure-pursuit missile always catches a target. In a bounded arena, however, the aircraft cannot simply run away, and pure pursuit demands a turn rate equal to the
    line-of-sight rate $\\dot\\lambda=v_a\\sin\\theta_a/d$ (target velocity component across the line of sight over distance), which diverges as $d\\to0$. Once
    $\\dot\\lambda>\\omega_m$ the missile can no longer follow: a hard turn by the aircraft at close range makes the missile overshoot and fly a wide arc before it can come back.
    The game is decided by the speed ratio $v_m/v_a$ and the turn-radius ratio $R_m/R_a$.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Update.</b> The two difference equations above are applied exactly, one step at a time (no integrator is needed: the model is defined in discrete time).
      The number of steps per frame is $30\\,\\Delta t\\times$speed, capped at 8.</li>
      <li><b>Hit test without tunnelling.</b> Over one step the relative position is assumed to change linearly, $\\mathbf d(s)=\\mathbf d_0+s(\\mathbf d_1-\\mathbf d_0)$, $s\\in[0,1]$;
      the closest approach $\\min_s|\\mathbf d(s)|$ is compared with $d_{hit}$, so a fast missile cannot jump "through" the aircraft between two steps.</li>
      <li><b>Simple pilot.</b> If the missile is closer than 35 units the aircraft steers to the heading pointing directly away from it; within 20 units of a wall it adds a
      repulsion vector pointing back into the arena; otherwise it wanders gently ($\\pm2°$/step sinusoid). The command is clipped to $\\pm\\omega_a$.</li>
      <li><b>Look-ahead planner</b> (a small model-predictive controller). Every step it rolls out five constant-turn candidates — $u\\in\\{-1,-\\tfrac12,0,\\tfrac12,1\\}\\,\\omega_a$ —
      for $K=40$ steps, simulating the missile with its exact guidance law. Each rollout is scored by the worst moment along the horizon,
      $S=\\min_k\\min\\big(d_k,\\,1.5\\,w_k+d_{hit}\\big)$, where $w_k$ is the distance to the nearest wall; a predicted crash scores $-1000+k$ and a predicted hit $-500+k$
      (later failure is better). The best candidate is applied for one step, then everything is re-planned; the previous command is kept unless another is strictly better,
      which prevents chattering. The dotted lines show the chosen rollout.</li>
      <li><b>Statistics.</b> The distance plot shows $\\log_{10}d$ versus step with the hit and lock radii; the history bars give the survival time of the last 40 episodes coloured by outcome;
      the success rate is escapes / all episodes. The minimum distance metric uses the swept closest approach.</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li><b>Pilot comparison.</b> With the default settings the planner escapes in essentially every episode, while the simple "turn away" pilot is caught almost
      every time (about 1 escape in 40 in our tests): running straight away from a faster missile only delays the hit, and the walls force a turn anyway.</li>
      <li><b>A sharp threshold.</b> Keep the speeds and raise the missile turn rate: up to $\\omega_m=8°$/step ($R_m\\approx15.8$) the planner always escapes, at 8.5° roughly
      two episodes in three, and from 9°/step ($R_m\\approx14$) the missile wins almost every time. The overshoot trick only works while the missile's turn circle is wide enough.</li>
      <li><b>Agility alone is not enough.</b> With $\\omega_m=9°$ even $\\omega_a=15°$/step does not save the aircraft; conversely, with the default missile an aircraft limited to
      $\\omega_a=4°$/step is caught every time.</li>
      <li><b>Slow missile.</b> Set $v_m=1.4\\lt v_a$: the missile can never close on a fleeing target; the planner escapes every time and even the simple pilot survives most episodes
      (its failures are mostly wall crashes).</li>
      <li><b>Fast missile.</b> At $v_m=3$ the default $\\omega_m=7°$ always hits, but at $\\omega_m=5°$ ($R_m\\approx34$) the planner escapes again — speed and turn radius trade off.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>The model is 2D and kinematic: no aerodynamics, no energy loss in turns, no sensor noise, perfect knowledge of the opponent, and the missile uses plain pure pursuit
    instead of proportional navigation ($\\dot\\phi=N\\dot\\lambda$, $N\\approx3$–5), which real missiles use precisely because it avoids the high terminal turn rates
    exploited here. See R. Isaacs, <em>Differential Games</em> (1965) — the "homicidal chauffeur" problem; P. Zarchan, <em>Tactical and Strategic Missile Guidance</em>;
    N. A. Shneydor, <em>Missile Guidance and Pursuit</em>.</p>`,

  mount(api) {
    const P = api.params;
    const W = 140, H = 90, HIT = 3, LOCK = 40, SPS = 30; // steps per second (× speed)
    const K = 40, CANDS = [-1, -0.5, 0, 0.5, 1];
    const D2R = Math.PI / 180;
    const TRMAX = 620, HISTN = 40;
    const wrap = (a) => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };
    const clip = (v, a) => (v > a ? a : v < -a ? -a : v);

    const M = api.metrics([
      { id: "ep", label: "Episode" },
      { id: "surv", label: "Survival time (this episode)" },
      { id: "dmin", label: "Closest approach (this episode)" },
      { id: "score", label: "Escapes / hits / crashes" },
      { id: "rate", label: "Success rate" },
    ]);
    const plots = api.plots([
      { id: "arena", title: "Airspace", span: 2, aspect: 0.6, xlim: [-3, W + 3], ylim: [-3, H + 11], equal: true, axes: false, maxHeight: 620 },
      { id: "dist", title: "Aircraft–missile distance (this episode, log scale)", aspect: 0.58, xlim: [0, 350], ylim: [0, 2.3], xlabel: "step", ylabel: "distance",
        ytickFormat: (v) => { const e = Math.round(v); return Math.abs(v - e) < 1e-6 ? String(Math.pow(10, e)) : ""; } },
      { id: "hist", title: "Recent episodes: survival time", aspect: 0.58, xlim: [0.5, HISTN + 0.5], ylim: [0, 370], xlabel: "episode (newest on the right)", ylabel: "steps" },
    ]);

    // state
    let rng, ep = 0, stepN = 0, acc = 0, phase = "run", endT = 0, outcome = null, hitPt = null;
    let px, py, pa, mx, my, ma, dmin, lastCmd = 0, wander0 = 0;
    const trP = new Float64Array(TRMAX * 2), trM = new Float64Array(TRMAX * 2);
    let head = 0, count = 0;
    let distH = new Float64Array(0);
    const planP = new Float64Array(K * 2), planM = new Float64Array(K * 2);
    let planN = 0;
    const hist = []; // {steps, res}
    let nEsc = 0, nHit = 0, nWall = 0;

    // Missile guidance (one step). The state array s = [mx, my, ma] is updated in place.
    function missileStep(s, tx, ty, vm, wm) {
      const des = Math.atan2(ty - s[1], tx - s[0]);
      s[2] = wrap(s[2] + clip(wrap(des - s[2]), wm));
      s[0] += vm * Math.cos(s[2]); s[1] += vm * Math.sin(s[2]);
    }
    const ms = new Float64Array(3);

    // Forward simulation of a candidate manoeuvre → safety score (larger = better)
    function rollout(cmd, record) {
      const vp = P.vp, wp = P.wp * D2R, vm = P.vm, wm = P.wm * D2R;
      let x = px, y = py, a = pa;
      ms[0] = mx; ms[1] = my; ms[2] = ma;
      let score = Infinity;
      for (let k = 0; k < K; k++) {
        a += cmd * wp;
        x += vp * Math.cos(a); y += vp * Math.sin(a);
        const wall = Math.min(x, W - x, y, H - y);
        if (record) { planP[2 * k] = x; planP[2 * k + 1] = y; }
        if (wall < 1) { if (record) planN = k + 1; return -1000 + k; }
        missileStep(ms, x, y, vm, wm);
        if (record) { planM[2 * k] = ms[0]; planM[2 * k + 1] = ms[1]; }
        const d = Math.hypot(x - ms[0], y - ms[1]);
        if (d < HIT) { if (record) planN = k + 1; return -500 + k; }
        const s = Math.min(d, 1.5 * wall + HIT);
        if (s < score) score = s;
      }
      if (record) planN = K;
      return score;
    }
    function pilotPlan() {
      let best = lastCmd, bestS = rollout(lastCmd, false);
      for (const c of CANDS) { if (c === lastCmd) continue; const s = rollout(c, false); if (s > bestS + 1e-9) { bestS = s; best = c; } }
      lastCmd = best;
      if (P.showPlan) rollout(best, true); else planN = 0;
      return best * P.wp * D2R;
    }
    function pilotSimple() {
      planN = 0;
      const dx = mx - px, dy = my - py, d = Math.hypot(dx, dy);
      const wp = P.wp * D2R;
      // near a wall, steer back into the arena (simple repulsion vector)
      const m = 20, wx = (Math.max(0, m - px) - Math.max(0, px - (W - m))) / m, wy = (Math.max(0, m - py) - Math.max(0, py - (H - m))) / m;
      let des;
      if (d < 35) des = Math.atan2(dy, dx) + Math.PI;
      else if (wx || wy) des = pa;
      else return clip(2 * D2R * Math.sin((stepN + wander0) * 0.07), wp);
      const vx = Math.cos(des) + 3 * wx, vy = Math.sin(des) + 3 * wy;
      return clip(wrap(Math.atan2(vy, vx) - pa), wp);
    }

    function newEpisode() {
      ep++;
      rng = new PM.RNG(7919 * ep + 17);
      px = W * 0.3; py = H * 0.5; pa = rng.uniform(-0.6, 0.6);
      const side = rng.int(0, 3);
      if (side === 0) { mx = 0; my = rng.uniform(0, H); }
      else if (side === 1) { mx = W; my = rng.uniform(0, H); }
      else if (side === 2) { mx = rng.uniform(0, W); my = 0; }
      else { mx = rng.uniform(0, W); my = H; }
      ma = Math.atan2(py - my, px - mx);
      wander0 = rng.uniform(0, 90);
      stepN = 0; acc = 0; phase = "run"; outcome = null; hitPt = null; lastCmd = 0; planN = 0;
      head = 0; count = 0;
      distH = new Float64Array(P.fuel + 2);
      dmin = Math.hypot(px - mx, py - my);
      distH[0] = dmin;
      pushTrail();
      plots.dist.setLimits([0, P.fuel]);
      plots.hist.setLimits(null, [0, P.fuel * 1.08]);
    }
    function pushTrail() {
      const i = head % TRMAX;
      trP[2 * i] = px; trP[2 * i + 1] = py; trM[2 * i] = mx; trM[2 * i + 1] = my;
      head++; count = Math.min(count + 1, TRMAX);
    }
    function finish(res) {
      phase = "end"; outcome = res; endT = 0;
      if (res === "esc") nEsc++; else if (res === "hit") nHit++; else nWall++;
      hist.push({ steps: stepN, res });
      if (hist.length > HISTN) hist.shift();
    }

    function simStep() {
      const turn = P.pilot === "plan" ? pilotPlan() : pilotSimple();
      const ox = px, oy = py, omx = mx, omy = my;
      pa = wrap(pa + turn);
      px += P.vp * Math.cos(pa); py += P.vp * Math.sin(pa);
      ms[0] = mx; ms[1] = my; ms[2] = ma;
      missileStep(ms, px, py, P.vm, P.wm * D2R);
      mx = ms[0]; my = ms[1]; ma = ms[2];
      stepN++;
      // closest approach within the step (relative motion assumed linear)
      const r0x = omx - ox, r0y = omy - oy, r1x = mx - px, r1y = my - py;
      const ux = r1x - r0x, uy = r1y - r0y, uu = ux * ux + uy * uy;
      const s = uu > 0 ? PM.clamp(-(r0x * ux + r0y * uy) / uu, 0, 1) : 1;
      const dClose = Math.hypot(r0x + s * ux, r0y + s * uy);
      const d = Math.hypot(r1x, r1y);
      dmin = Math.min(dmin, dClose);
      distH[Math.min(stepN, distH.length - 1)] = d;
      pushTrail();
      if (dClose < HIT) { hitPt = [ox + s * (px - ox), oy + s * (py - oy)]; finish("hit"); return; }
      if (px < 1 || px > W - 1 || py < 1 || py > H - 1) { px = PM.clamp(px, 0, W); py = PM.clamp(py, 0, H); hitPt = [px, py]; finish("wall"); return; }
      if (stepN >= P.fuel) finish("esc");
    }

    // ------------------------------------------------------------ drawing helpers
    function drawJet(c, X, Y, ang, s, col) {
      c.save(); c.translate(X, Y); c.rotate(-ang); // screen y points down → flip the angle
      c.beginPath();
      c.moveTo(1.6 * s, 0);
      c.lineTo(0.2 * s, 0.22 * s); c.lineTo(-0.2 * s, 0.95 * s); c.lineTo(-0.55 * s, 0.95 * s); c.lineTo(-0.45 * s, 0.25 * s);
      c.lineTo(-0.9 * s, 0.2 * s); c.lineTo(-1.1 * s, 0.55 * s); c.lineTo(-1.25 * s, 0.55 * s); c.lineTo(-1.15 * s, 0);
      c.lineTo(-1.25 * s, -0.55 * s); c.lineTo(-1.1 * s, -0.55 * s); c.lineTo(-0.9 * s, -0.2 * s); c.lineTo(-0.45 * s, -0.25 * s);
      c.lineTo(-0.55 * s, -0.95 * s); c.lineTo(-0.2 * s, -0.95 * s); c.lineTo(0.2 * s, -0.22 * s);
      c.closePath();
      c.fillStyle = col; c.fill();
      c.strokeStyle = "#0f151c"; c.lineWidth = 1; c.stroke();
      c.restore();
    }
    function drawMissile(c, X, Y, ang, s, t) {
      c.save(); c.translate(X, Y); c.rotate(-ang);
      // exhaust flame
      const fl = 0.8 + 0.35 * Math.sin(t * 40) + 0.15 * Math.sin(t * 97);
      c.beginPath(); c.moveTo(-0.9 * s, 0.25 * s); c.lineTo(-(0.9 + 1.3 * fl) * s, 0); c.lineTo(-0.9 * s, -0.25 * s); c.closePath();
      c.fillStyle = "rgba(245,158,11,0.9)"; c.fill();
      c.beginPath(); c.moveTo(1.1 * s, 0); c.lineTo(0.6 * s, 0.22 * s); c.lineTo(-0.9 * s, 0.22 * s); c.lineTo(-0.9 * s, -0.22 * s); c.lineTo(0.6 * s, -0.22 * s); c.closePath();
      c.fillStyle = "#f85149"; c.fill();
      c.fillStyle = "#e6edf3"; c.fillRect(-0.9 * s, -0.5 * s, 0.35 * s, 1.0 * s);
      c.restore();
    }
    function trailXY(buf, n, out) {
      const xs = new Float64Array(n), ys = new Float64Array(n);
      for (let k = 0; k < n; k++) { const i = (head - n + k) % TRMAX; xs[k] = buf[2 * i]; ys[k] = buf[2 * i + 1]; }
      return [xs, ys];
    }

    let animT = 0;
    return {
      reset() {
        ep = 0; hist.length = 0; nEsc = nHit = nWall = 0;
        newEpisode();
      },
      onAction(id) { if (id === "clr") { hist.length = 0; nEsc = nHit = nWall = 0; } },
      step(dt) {
        animT += dt;
        if (phase === "end") {
          endT += dt;
          if (endT > 1.6) newEpisode();
          return;
        }
        acc += dt * SPS;
        let n = Math.floor(acc);
        acc -= n;
        n = Math.min(n, 8);
        for (let i = 0; i < n && phase === "run"; i++) simStep();
      },
      render() {
        const pa_ = plots.arena;
        pa_.clear();
        const d = Math.hypot(mx - px, my - py);
        const locked = d < LOCK && phase === "run";
        pa_.custom((c, pl) => {
          // arena and grid
          c.fillStyle = "#0c1219"; c.fillRect(pl.X(0), pl.Y(H), pl.X(W) - pl.X(0), pl.Y(0) - pl.Y(H));
          c.strokeStyle = "rgba(139,152,168,0.08)"; c.lineWidth = 1; c.beginPath();
          for (let x = 10; x < W; x += 10) { c.moveTo(pl.X(x), pl.Y(0)); c.lineTo(pl.X(x), pl.Y(H)); }
          for (let y = 10; y < H; y += 10) { c.moveTo(pl.X(0), pl.Y(y)); c.lineTo(pl.X(W), pl.Y(y)); }
          c.stroke();
          c.strokeStyle = "rgba(139,152,168,0.55)"; c.setLineDash([6, 5]); c.strokeRect(pl.X(0), pl.Y(H), pl.X(W) - pl.X(0), pl.Y(0) - pl.Y(H)); c.setLineDash([]);
        });
        if (P.showLock && locked) {
          pa_.circle(px, py, LOCK, { fill: false, stroke: "rgba(248,81,73,0.45)", strokeWidth: 1.2 });
          pa_.circle(px, py, LOCK, { color: PlotColors.bad, alpha: 0.04 });
        }
        // trails
        const tl = Math.min(P.trail + 1, count);
        if (tl > 1) {
          const [a1, b1] = trailXY(trP, tl), [a2, b2] = trailXY(trM, tl);
          pa_.line(a1, b1, { color: PlotColors.accent, width: 2, alpha: 0.75 });
          pa_.line(a2, b2, { color: PlotColors.bad, width: 1.8, alpha: 0.7, dash: [5, 4] });
        }
        // plan
        if (phase === "run" && P.pilot === "plan" && P.showPlan && planN > 1) {
          pa_.line(planP.filter((_, i) => i % 2 === 0 && i < 2 * planN), planP.filter((_, i) => i % 2 === 1 && i < 2 * planN), { color: PlotColors.accent, width: 1, alpha: 0.35, dash: [2, 4] });
          pa_.line(planM.filter((_, i) => i % 2 === 0 && i < 2 * planN), planM.filter((_, i) => i % 2 === 1 && i < 2 * planN), { color: PlotColors.bad, width: 1, alpha: 0.3, dash: [2, 4] });
        }
        // vehicles
        const S = pa_.sx * 2.6;
        pa_.custom((c, pl) => {
          const showPlane = !(phase === "end" && outcome !== "esc");
          if (showPlane) drawJet(c, pl.X(px), pl.Y(py), pa, S, outcome === "esc" ? PlotColors.good : PlotColors.accent);
          drawMissile(c, pl.X(mx), pl.Y(my), ma, S * 0.75, animT);
          if (phase === "end" && hitPt) {
            const u = Math.min(endT / 0.9, 1), X = pl.X(hitPt[0]), Y = pl.Y(hitPt[1]);
            for (let k = 0; k < 3; k++) {
              const r = (1.5 + 8 * u) * (1 - 0.25 * k) * pl.sx;
              c.globalAlpha = (1 - u) * (0.9 - 0.2 * k);
              c.fillStyle = ["#f85149", "#f59e0b", "#fcfdbf"][k];
              c.beginPath(); c.arc(X, Y, r, 0, 2 * Math.PI); c.fill();
            }
            c.globalAlpha = 1;
          }
        });
        const lines = [`Episode ${ep}  ·  step ${stepN}/${P.fuel}`, `distance ${PM.fmt(d, 1)}`];
        pa_.label(lines, "tl", { size: 12 });
        if (phase === "end") {
          const txt = outcome === "esc" ? "✔ ESCAPED — the missile ran out of fuel" : outcome === "hit" ? "💥 HIT" : "✖ CRASH — flew out of the arena";
          const col = outcome === "esc" ? PlotColors.good : outcome === "hit" ? PlotColors.accent3 : PlotColors.bad;
          pa_.textPx(pa_.m.l + pa_._v.pw / 2, pa_.m.t + 30, txt, { size: 17, bold: true, color: col, align: "center", bg: "#0f151c" });
        } else if (locked) {
          pa_.label("🔒 MISSILE LOCKED ON", "tr", { color: PlotColors.bad, bold: true, size: 12 });
        } else pa_.label("missile searching", "tr", { color: PlotColors.muted, size: 12 });
        pa_.legend([{ label: "Aircraft", color: PlotColors.accent }, { label: "Missile", color: PlotColors.bad, dash: [5, 4] }], "br");

        // distance plot
        const pd = plots.dist;
        pd.clear();
        pd.hline(Math.log10(HIT), { color: PlotColors.accent3, dash: [4, 4], width: 1 });
        pd.hline(Math.log10(LOCK), { color: PlotColors.bad, dash: [4, 4], width: 1, alpha: 0.7 });
        pd.text(P.fuel * 0.98, Math.log10(HIT) + 0.08, "hit radius", { color: PlotColors.accent3, size: 10, align: "right" });
        pd.text(P.fuel * 0.98, Math.log10(LOCK) + 0.08, "lock-on range", { color: PlotColors.bad, size: 10, align: "right" });
        const n = Math.min(stepN + 1, distH.length);
        if (n > 1) {
          const xs = new Float64Array(n), ys = new Float64Array(n);
          for (let i = 0; i < n; i++) { xs[i] = i; ys[i] = Math.log10(Math.max(distH[i], 0.5)); }
          pd.line(xs, ys, { color: PlotColors.accent, width: 1.6 });
        }

        // episode history
        const ph = plots.hist;
        ph.clear();
        if (hist.length) {
          const off = HISTN - hist.length;
          const xs = hist.map((_, i) => off + i + 1), hs = hist.map((h) => h.steps);
          const cols = hist.map((h) => (h.res === "esc" ? PlotColors.good : h.res === "hit" ? PlotColors.accent3 : PlotColors.bad));
          ph.bars(xs, hs, 0.8, { colors: cols, alpha: 0.85 });
        } else ph.label("No completed episodes yet", "tl", { color: PlotColors.muted });
        ph.legend([{ label: "escape", color: PlotColors.good, type: "box" }, { label: "hit", color: PlotColors.accent3, type: "box" }, { label: "crash", color: PlotColors.bad, type: "box" }], "tl");

        const tot = nEsc + nHit + nWall;
        M.set("ep", String(ep));
        M.set("surv", `${PM.fmt(stepN / SPS, 1)} s`);
        M.set("dmin", PM.fmt(dmin, 2));
        M.set("score", `${nEsc} / ${nHit} / ${nWall}`);
        M.set("rate", tot ? `${Math.round((100 * nEsc) / tot)} %` : "—");
        api.setTime(`t = ${PM.fmt(stepN / SPS, 1)} s`);
      },
    };
  },
});
