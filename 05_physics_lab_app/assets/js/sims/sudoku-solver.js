/* Sudoku solver — graph colouring by backtracking, with a live step-by-step visualisation.
 * The solver is a resumable state machine: each frame advances it by K steps according to the speed setting.
 * One "step" = one placement (assigning a colour/digit to a cell) or one backtrack (clearing a cell). */
(function () {
  const PRESETS = {
    default: "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    easy: "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    medium: "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    hard: "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    extreme: "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
  };
  const toText = (s) => Array.from({ length: 9 }, (_, r) => s.slice(9 * r, 9 * r + 9).split("").join(" ")).join("\n");
  const SAFETY_CAP = 2e9; // very high safety limit, only reached by pathological inputs

  /** Backtracking solver: first empty cell in row-major order, colours tried 1..9 in order. */
  class Solver {
    constructor(cells) {
      this.b = Int8Array.from(cells);
      this.row = new Uint16Array(9); this.col = new Uint16Array(9); this.box = new Uint16Array(9);
      const E = [];
      for (let i = 0; i < 81; i++) {
        const v = this.b[i], r = (i / 9) | 0, c = i % 9, x = ((r / 3) | 0) * 3 + ((c / 3) | 0);
        if (v) { this.row[r] |= 1 << v; this.col[c] |= 1 << v; this.box[x] |= 1 << v; } else E.push(i);
      }
      this.E = Int32Array.from(E);
      this.tried = new Uint8Array(E.length + 1);
      this.k = 0; this.steps = 0; this.backs = 0;
      this.status = E.length ? 0 : 1; // 0 running, 1 solved, 2 no solution, 3 safety limit
      this.last = -1; this.lastBack = false;
      this.track = null; // {visits, flash, depS, depK, every}
    }
    /** Advance by at most n steps. */
    run(n) {
      const b = this.b, row = this.row, col = this.col, box = this.box, E = this.E, tried = this.tried, NE = E.length, tr = this.track;
      let k = this.k, steps = this.steps, backs = this.backs, last = this.last, lastBack = this.lastBack;
      const stop = steps + n;
      while (this.status === 0 && steps < stop) {
        if (k === NE) { this.status = 1; break; }
        const i = E[k], r = (i / 9) | 0, c = i - 9 * r, x = ((r / 3) | 0) * 3 + ((c / 3) | 0);
        const used = row[r] | col[c] | box[x];
        let v = tried[k] + 1;
        while (v <= 9 && (used >> v) & 1) v++;
        if (v <= 9) { // place
          tried[k] = v; b[i] = v; row[r] |= 1 << v; col[c] |= 1 << v; box[x] |= 1 << v;
          k++; tried[k] = 0; steps++; last = i; lastBack = false;
          if (tr) tr.visits[i]++;
        } else { // backtrack: clear the previous cell
          tried[k] = 0; k--;
          if (k < 0) { this.status = 2; k = 0; break; }
          const j = E[k], rr = (j / 9) | 0, cc = j - 9 * rr, xx = ((rr / 3) | 0) * 3 + ((cc / 3) | 0), w = b[j];
          b[j] = 0; row[rr] &= ~(1 << w); col[cc] &= ~(1 << w); box[xx] &= ~(1 << w);
          steps++; backs++; last = j; lastBack = true;
          if (tr) { tr.visits[j]++; tr.flash[j] = 1; }
        }
        if (tr && steps % tr.every === 0) { tr.depS.push(steps); tr.depK.push(k); }
        if (steps >= SAFETY_CAP) { this.status = 3; break; }
      }
      if (k === NE && this.status === 0) this.status = 1;
      this.k = k; this.steps = steps; this.backs = backs; this.last = last; this.lastBack = lastBack;
    }
  }

  App.register({
    id: "sudoku-solver",
    category: "special",
    order: 50,
    title: "Sudoku Solver (Graph Colouring)",
    icon: "🧩",
    subtitle: "A live, step-by-step view of a backtracking search that solves 9×9 Sudoku as a graph-colouring problem: the board, the search load per cell and the search depth.",
    notes: [{ type: "info", html: "Sudoku is a <b>graph-colouring problem</b>: every cell is a <b>vertex</b>, and cells in the same row, column or 3×3 box are joined by an <b>edge</b>. The task is to assign the colours (digits) 1–9 so that no two neighbours share a colour. The algorithm takes the first empty cell, tries the colours in order and backtracks when it reaches a dead end." }],
    animated: true,
    speed: { min: 0.1, max: 4, value: 1, step: 0.1 },
    controls: [
      { id: "preset", type: "select", label: "Preset puzzle", value: "default", options: [
        { value: "default", label: "Default (classic example)" },
        { value: "easy", label: "Easy" },
        { value: "medium", label: "Medium" },
        { value: "hard", label: "Hard (Inkala, 2012)" },
        { value: "extreme", label: "Very hard for naive backtracking (≈5×10⁷ steps)" },
      ] },
      { id: "puzzle", type: "textarea", label: "Sudoku board (9 rows × 9 digits, 0 = empty)", rows: 10, value: toText(PRESETS.default) },
      { id: "load", type: "button", label: "✔ Load and validate puzzle" },
      { id: "default", type: "button", label: "↺ Default puzzle" },
      { type: "section", label: "Solver" },
      { id: "rate", type: "slider", label: "Solving speed", min: 0.7, max: 4.7, step: 0.01, value: 2.6, live: true,
        fmt: (v) => Math.round(Math.pow(10, v)).toLocaleString("en-US") + " steps/s" },
      { id: "instant", type: "button", label: "⚡ Solve instantly", primary: true },
      { type: "section", label: "Display" },
      { id: "colors", type: "checkbox", label: "Show digits as colours (graph colouring)", value: false, live: true },
    ],
    theory: `
      <h4>The problem</h4>
      <p>A Sudoku board is a 9×9 grid divided into nine 3×3 boxes. Some cells (the <em>givens</em> or clues) are pre-filled with digits 1–9; the
      goal is to fill the rest so that every row, every column and every box contains each digit exactly once. A well-posed puzzle has exactly one
      solution. You can type any board into the text box (0, “.” or “_” for empty cells); it is validated before solving.</p>

      <h4>Formulation</h4>
      <p>Build the <b>Sudoku graph</b> $G=(V,E)$: one vertex per cell ($|V|=81$) and an edge between any two cells that share a row, column or box.
      Each cell has $8$ row-mates, $8$ column-mates and $4$ further box-mates, so every vertex has degree 20 and</p>
      $$|E|=\\frac{81\\cdot20}{2}=810 .$$
      <p>A solution is a <em>proper colouring</em> $c:V\\to\\{1,\\dots,9\\}$ with</p>
      <div class="callout">
      $$c(u)\\neq c(v)\\quad\\text{for every edge }(u,v)\\in E,\\qquad c(v)=g_v\\ \\text{for every given cell } v .$$
      </div>
      <p>Each row is a 9-clique, so the chromatic number is $\\chi(G)=9$: nine colours are necessary and sufficient. Solving Sudoku is therefore a
      <em>precolouring-extension</em> problem — extend a partial 9-colouring to the whole graph. For general $n^2\\times n^2$ boards this problem is NP-complete
      (Yato &amp; Seta, 2003), so no known algorithm is polynomial in the worst case; for 9×9 boards, however, constraint propagation makes the search tiny.</p>

      <h4>How the solver works</h4>
      <ul>
        <li><b>Variable order.</b> The empty cells are listed once in row-major order $e_1,e_2,\\dots,e_n$; the search always works on the first unfilled one.</li>
        <li><b>Value order.</b> For cell $e_k$ the colours are tried in increasing order, starting just above the last colour tried there.</li>
        <li><b>Constraint check in $O(1)$.</b> Three arrays of 9-bit masks record which digits are used in each row, column and box; a colour is legal when its bit is
        clear in $\\text{row}[r]\\,|\\,\\text{col}[c]\\,|\\,\\text{box}[b]$.</li>
        <li><b>Place / backtrack.</b> If a legal colour exists it is placed and the search moves to $e_{k+1}$ (<em>placement</em>, teal flash). If none exists the
        previous cell $e_{k-1}$ is cleared and its next colour is tried (<em>backtrack</em>, red flash). Each placement or backtrack counts as one <b>step</b>.
        Reaching $k=n$ means solved; backtracking past $e_1$ proves there is no solution.</li>
        <li><b>Complexity.</b> This is a depth-first search of a tree with at most $9^n$ leaves ($n$ = number of empty cells, 49–64 here), but the constraints
        prune it drastically: the presets need from 351 steps (easy) to $5.3\\times10^7$ steps (the last preset, built to defeat this fixed ordering).</li>
        <li><b>Validation and look-ahead.</b> The givens are first checked for duplicates in any row, column or box. A second, invisible copy of the solver then runs
        ahead (up to $5\\times10^6$ steps at once, then ≈ 4 ms per frame) to learn the total step count and whether a solution exists, which sets the axis of the depth plot.</li>
        <li><b>Plots.</b> The board shows the current assignment; the heat map counts how many times each cell was changed (the search load); the depth plot shows the
        number of filled empty cells $k$ versus step — the search diving and retreating in the tree. "Solve instantly" runs the solver in 11 ms slices per frame until done.</li>
      </ul>

      <h4>What to try</h4>
      <ol>
        <li><b>Easy</b> preset: 49 empty cells, solved in 351 steps with 151 backtracks — almost pure forward filling, the depth curve is a near-straight ramp.</li>
        <li><b>Default</b> preset: 8 365 steps. Watch the heat map: cells late in row-major order are changed far more often than early ones.</li>
        <li><b>Hard (Inkala)</b>: 60 empty cells but only ≈ 99 000 steps — "hard" for humans is not the same as hard for this algorithm.</li>
        <li><b>Very hard for naive backtracking</b>: ≈ $5.3\\times10^7$ steps, because the first rows hold only one or two clues each, so a wrong early choice is discovered only very deep in the tree. Use "Solve instantly" (about a second).</li>
        <li><b>Break it</b>: put two 5s in the same row and press "Load and validate" — the duplicate cells are highlighted. Or create a board that is
        consistent but unsolvable; the look-ahead search proves that no solution exists.</li>
      </ol>

      <h4>Limitations</h4>
      <p>The fixed row-major cell order and ascending digit order are deliberately naive; heuristics such as "most constrained cell first" (minimum remaining values),
      constraint propagation (naked/hidden singles) or Knuth's Dancing Links reduce every preset to well under a thousand steps. The solver finds the first solution and does not
      check uniqueness. Further reading: D. E. Knuth, “Dancing Links”, <em>Millennial Perspectives in Computer Science</em> (2000); T. Yato &amp; T. Seta,
      <em>IEICE Trans. Fundamentals</em> E86-A, 1052 (2003); S. Russell &amp; P. Norvig, <em>Artificial Intelligence: A Modern Approach</em>, ch. 6 (constraint satisfaction).</p>`,

    mount(api) {
      const P = api.params;
      const M = api.metrics([
        { id: "steps", label: "Steps" },
        { id: "backs", label: "Backtracks" },
        { id: "filled", label: "Filled cells" },
        { id: "state", label: "Status" },
      ]);
      const plots = api.plots([
        { id: "board", title: "Board — <span style='color:#4fd1c5'>■</span> placement · <span style='color:#f85149'>■</span> backtrack", aspect: 1, xlim: [0, 9], ylim: [0, 9], equal: true, axes: false, maxHeight: 620, margin: { l: 6, r: 6, t: 6, b: 6 } },
        { id: "visits", title: "Changes per cell (search load)", aspect: 1, xlim: [0, 9], ylim: [0, 9], equal: true, axes: false, maxHeight: 620, colorbar: true, margin: { l: 6, t: 6, b: 6, r: 74 } },
        { id: "depth", title: "Search depth (number of empty cells filled)", span: 2, aspect: 0.24, minHeight: 190, xlabel: "step", ylabel: "depth" },
      ]);
      const note = document.createElement("div");
      note.className = "note warn"; note.style.display = "none";
      api.stage.insertBefore(note, api.stage.querySelector(".plots"));

      let givens, conflicts, vis, pre, err, acc, turbo, E0, totalKnown, autoPaused = false;
      const visits = new Float64Array(81), flash = new Float32Array(81);

      function showErr(msg) { err = msg; note.textContent = msg || ""; note.style.display = msg ? "" : "none"; }

      function parse(text) {
        const lines = String(text || "").split(/\r?\n/).map((l) => l.replace(/[\s,|;]/g, "").replace(/[.*_-]/g, "0")).filter((l) => l.length);
        if (lines.length !== 9) return { error: `The board must have 9 rows; it currently has ${lines.length} non-empty row(s).` };
        const cells = new Int8Array(81);
        for (let r = 0; r < 9; r++) {
          const bad = lines[r].match(/[^0-9]/);
          if (bad) return { error: `Row ${r + 1} contains an invalid character: “${bad[0]}”. Use only the digits 0–9 (0 = empty cell).` };
          if (lines[r].length !== 9) return { error: `Row ${r + 1} must contain 9 digits; found ${lines[r].length}.` };
          for (let c = 0; c < 9; c++) cells[9 * r + c] = lines[r].charCodeAt(c) - 48;
        }
        return { cells };
      }
      /** Check the givens for duplicates: {msg, bad:Set} */
      function validate(cells) {
        const bad = new Set(); let msg = null;
        const groups = [];
        for (let i = 0; i < 9; i++) {
          groups.push({ name: `row ${i + 1}`, idx: Array.from({ length: 9 }, (_, c) => 9 * i + c) });
          groups.push({ name: `column ${i + 1}`, idx: Array.from({ length: 9 }, (_, r) => 9 * r + i) });
          const br = ((i / 3) | 0) * 3, bc = (i % 3) * 3;
          groups.push({ name: `3×3 box ${i + 1}`, idx: Array.from({ length: 9 }, (_, k) => 9 * (br + ((k / 3) | 0)) + bc + (k % 3)) });
        }
        for (const g of groups) {
          const seen = {};
          for (const i of g.idx) {
            const v = cells[i];
            if (!v) continue;
            if (seen[v] !== undefined) { bad.add(i); bad.add(seen[v]); if (!msg) msg = `Invalid puzzle: the digit ${v} is given more than once in ${g.name}. The clues violate the Sudoku rules.`; }
            else seen[v] = i;
          }
        }
        return { msg, bad };
      }

      function load() {
        showErr(null);
        visits.fill(0); flash.fill(0);
        acc = 0; turbo = false; vis = null; pre = null; conflicts = new Set();
        const res = parse(P.puzzle);
        if (res.error) { givens = new Int8Array(81); showErr(res.error); setupDepth(); return; }
        givens = res.cells;
        const v = validate(givens);
        if (v.msg) { conflicts = v.bad; showErr(v.msg); setupDepth(); return; }
        vis = new Solver(givens);
        E0 = vis.E.length;
        // background look-ahead solve: determines the total step count and solvability
        pre = new Solver(givens);
        pre.run(5e6);
        totalKnown = pre.status !== 0;
        vis.track = { visits, flash, depS: [0], depK: [0], every: totalKnown ? Math.max(1, Math.ceil(pre.steps / 4000)) : 1000 };
        checkPre();
        setupDepth();
      }
      function checkPre() {
        if (!pre || pre.status === 0) return;
        totalKnown = true;
        if (pre.status === 2) showErr("This puzzle has no solution: every possibility was tried and none satisfies the Sudoku rules. Please check the input.");
        if (pre.status === 3) showErr("The search hit its (very high) safety limit; the puzzle is most likely unsolvable or extremely under-constrained.");
      }
      function setupDepth() {
        const tot = pre && totalKnown ? Math.max(pre.steps, 10) : 1000;
        plots.depth.setLimits([0, tot], [0, Math.max(E0 || 1, 1) * 1.08]);
      }
      const solvable = () => vis && pre && !(totalKnown && pre.status !== 1);

      function runFor(ms) { // fast solving within a time budget
        const t0 = performance.now();
        while (vis.status === 0 && performance.now() - t0 < ms) vis.run(50000);
      }

      return {
        reset() { load(); if (autoPaused && !api.isPlaying) api.play(); autoPaused = false; },
        onParam(id, v) {
          if (id === "preset" && PRESETS[v]) api.setControl("puzzle", { value: toText(PRESETS[v]) });
        },
        onAction(id) {
          if (id === "load" || id === "default") {
            if (id === "default") { api.setControl("preset", { value: "default" }); api.setControl("puzzle", { value: toText(PRESETS.default) }); }
            load(); api.play(); autoPaused = false;
          } else if (id === "instant" && solvable() && vis.status === 0) {
            turbo = true;
            runFor(60);
            if (vis.status === 0) api.play(); else flash.fill(0);
          }
        },
        step(dt) {
          // continue the background look-ahead (only for searches longer than 5·10⁶ steps)
          if (vis && pre && pre.status === 0 && vis.status !== 0) { // the visible solver has already finished the search
            pre.status = vis.status; pre.steps = vis.steps; checkPre(); setupDepth();
          }
          if (pre && pre.status === 0 && !turbo) {
            const t0 = performance.now();
            while (pre.status === 0 && performance.now() - t0 < 4) pre.run(100000);
            if (pre.status !== 0) { checkPre(); setupDepth(); }
          }
          for (let i = 0; i < 81; i++) if (flash[i] > 0) flash[i] = Math.max(0, flash[i] - dt / 0.45);
          if (!solvable() || vis.status !== 0) {
            if (vis && vis.status !== 0 && !(pre && pre.status === 0)) { flash.fill(0); autoPaused = true; api.pause(); }
            return;
          }
          if (turbo) { runFor(11); return; }
          acc += Math.pow(10, P.rate) * dt;
          const n = Math.floor(acc);
          if (n > 0) { acc -= n; vis.run(Math.min(n, 400000)); }
          if (!totalKnown && vis.steps > plots.depth.xlim[1]) plots.depth.setLimits([0, vis.steps * 1.5]);
        },
        render() {
          const pb = plots.board, b = vis ? vis.b : givens;
          pb.clear();
          const solved = vis && vis.status === 1;
          pb.custom((c) => {
            const cs = pb.sx; // cell size (pixels)
            for (let i = 0; i < 81; i++) {
              const r = (i / 9) | 0, col = i % 9, X = pb.X(col), Y = pb.Y(9 - r), v = b[i];
              let bg = givens[i] ? "#1f2630" : "#10151c";
              if (P.colors && v) bg = PlotCycle[v - 1];
              c.globalAlpha = P.colors && v ? (givens[i] ? 0.55 : 0.32) : 1;
              c.fillStyle = bg; c.fillRect(X, Y, cs, cs);
              c.globalAlpha = 1;
              if (conflicts.has(i)) { c.fillStyle = PlotColors.bad; c.globalAlpha = 0.6; c.fillRect(X, Y, cs, cs); c.globalAlpha = 1; }
              if (flash[i] > 0) { c.fillStyle = PlotColors.bad; c.globalAlpha = 0.65 * flash[i]; c.fillRect(X, Y, cs, cs); c.globalAlpha = 1; }
            }
            if (vis && !solved && vis.last >= 0 && vis.status === 0) {
              const i = vis.last, r = (i / 9) | 0, col = i % 9;
              c.fillStyle = vis.lastBack ? PlotColors.bad : PlotColors.accent;
              c.globalAlpha = 0.9; c.fillRect(pb.X(col), pb.Y(9 - r), cs, cs); c.globalAlpha = 1;
            }
            c.textAlign = "center"; c.textBaseline = "middle";
            for (let i = 0; i < 81; i++) {
              const v = b[i];
              if (!v) continue;
              const r = (i / 9) | 0, col = i % 9, hl = vis && !solved && i === vis.last && vis.status === 0;
              c.font = `${givens[i] ? 700 : 600} ${Math.round(cs * 0.56)}px "Segoe UI", system-ui, sans-serif`;
              c.fillStyle = hl ? "#06231f" : givens[i] ? PlotColors.text : P.colors ? "#ffffff" : PlotColors.accent;
              c.fillText(String(v), pb.X(col + 0.5), pb.Y(9 - r - 0.5) + 1);
            }
            // grid lines
            c.strokeStyle = "rgba(139,152,168,0.35)"; c.lineWidth = 1;
            c.beginPath();
            for (let k = 1; k < 9; k++) { if (k % 3 === 0) continue; c.moveTo(pb.X(k), pb.Y(0)); c.lineTo(pb.X(k), pb.Y(9)); c.moveTo(pb.X(0), pb.Y(k)); c.lineTo(pb.X(9), pb.Y(k)); }
            c.stroke();
            c.strokeStyle = solved ? PlotColors.good : "#c9d1d9"; c.lineWidth = 3;
            c.beginPath();
            for (let k = 0; k <= 9; k += 3) { c.moveTo(pb.X(k), pb.Y(0)); c.lineTo(pb.X(k), pb.Y(9)); c.moveTo(pb.X(0), pb.Y(k)); c.lineTo(pb.X(9), pb.Y(k)); }
            c.stroke();
          });

          // --- visit heat map
          const pv = plots.visits;
          pv.clear();
          const z = new Float64Array(81);
          let vmax = 1;
          for (let i = 0; i < 81; i++) { const r = (i / 9) | 0, col = i % 9; z[(8 - r) * 9 + col] = visits[i]; vmax = Math.max(vmax, visits[i]); }
          pv.heatmap(z, 9, 9, { x0: 0, x1: 9, y0: 0, y1: 9, vmin: 0, vmax, cmap: "magma", smooth: false, scale: vmax > 50 ? "sqrt" : "linear", colorbar: "number of changes" });
          pv.custom((c) => {
            c.textAlign = "center"; c.textBaseline = "middle";
            c.font = `600 ${Math.round(pv.sx * 0.34)}px "Segoe UI", system-ui, sans-serif`;
            for (let i = 0; i < 81; i++) if (givens[i]) { const r = (i / 9) | 0, col = i % 9; c.fillStyle = "rgba(230,237,243,0.55)"; c.fillText(String(givens[i]), pv.X(col + 0.5), pv.Y(9 - r - 0.5)); }
            c.strokeStyle = "#c9d1d9"; c.lineWidth = 2; c.globalAlpha = 0.8;
            c.beginPath();
            for (let k = 0; k <= 9; k += 3) { c.moveTo(pv.X(k), pv.Y(0)); c.lineTo(pv.X(k), pv.Y(9)); c.moveTo(pv.X(0), pv.Y(k)); c.lineTo(pv.X(9), pv.Y(k)); }
            c.stroke(); c.globalAlpha = 1;
          });

          // --- search depth
          const pd = plots.depth;
          pd.clear();
          if (vis) {
            pd.hline(E0, { color: PlotColors.good, dash: [6, 4], width: 1.2 });
            const tr = vis.track;
            let xs = tr.depS, ys = tr.depK;
            if (xs.length > 5000) { // thin out for drawing
              const st = Math.ceil(xs.length / 5000), nx = [], ny = [];
              for (let i = 0; i < xs.length; i += st) { nx.push(xs[i]); ny.push(ys[i]); }
              nx.push(xs[xs.length - 1]); ny.push(ys[ys.length - 1]); xs = nx; ys = ny;
            }
            if (xs.length > 1) {
              pd.fill(xs, ys, 0, { color: PlotColors.accent2, alpha: 0.18 });
              pd.line(xs, ys, { color: PlotColors.accent2, width: 1.3 });
            }
            pd.circle(vis.steps, vis.k, 4, { px: true, color: vis.status === 1 ? PlotColors.good : PlotColors.accent });
            pd.label(`solution depth = ${E0} empty cells`, "tr", { color: PlotColors.good });
          } else pd.label("Load a valid puzzle.", "tl");

          // --- metrics
          const fmtN = (x) => Math.round(x).toLocaleString("en-US");
          if (vis) {
            M.set("steps", totalKnown && pre.status === 1 ? `${fmtN(vis.steps)} / ${fmtN(pre.steps)}` : fmtN(vis.steps));
            M.set("backs", fmtN(vis.backs));
            M.set("filled", `${81 - E0 + vis.k} / 81`);
            const st = !solvable() ? "No solution ❌" : vis.status === 1 ? "Solved ✅" : turbo ? "Fast solving…" : pre.status === 0 ? "Searching…" : "Solving…";
            M.set("state", st);
            api.setTime(vis.status === 1 ? "complete ✓" : turbo ? "⚡ instant" : `${fmtN(Math.pow(10, P.rate))} steps/s`);
          } else {
            M.set("steps", "—"); M.set("backs", "—"); M.set("filled", "—"); M.set("state", "Invalid input");
            api.setTime("");
          }
        },
      };
    },
  });
})();
