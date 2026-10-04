/* =====================================================================
 * Plot / View3D — fast Canvas 2D rendering engine
 *
 *  const p = new Plot(konteyner, { aspect: 0.6, xlim:[0,1], ylim:[-1,1],
 *                                  xlabel:"x", ylabel:"y", equal:false });
 *  // her karede:
 *  p.clear();                 // background + grid + axes, then clips to the data area
 *  p.line(xs, ys, {color:"#4fd1c5", width:2});
 *  p.heatmap(z, nx, ny, {x0,x1,y0,y1, vmin, vmax, cmap:"inferno"});
 *
 * All drawing functions take DATA coordinates.
 * ===================================================================== */
(function () {
  "use strict";

  const THEME = {
    bg: "#131a22", grid: "rgba(139,152,168,0.13)", axis: "#2b3848", tick: "#8b98a8",
    text: "#e6edf3", font: '11px "Segoe UI", system-ui, sans-serif', labelFont: '12px "Segoe UI", system-ui, sans-serif',
  };
  const COLORS = {
    accent: "#4fd1c5", accent2: "#8b5cf6", accent3: "#f59e0b", good: "#3fb950", bad: "#f85149",
    blue: "#58a6ff", pink: "#f778ba", muted: "#8b98a8", text: "#e6edf3", white: "#ffffff",
  };
  const CYCLE = ["#4fd1c5", "#8b5cf6", "#f59e0b", "#58a6ff", "#f85149", "#3fb950", "#f778ba", "#a5d6ff", "#ffa657", "#d2a8ff"];

  // ------------------------------------------------------------ colormaps
  const CMAP_STOPS = {
    inferno: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
    magma: [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]],
    viridis: [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]],
    plasma: [[13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137], [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33]],
    turbo: [[48, 18, 59], [70, 107, 227], [41, 187, 236], [49, 242, 153], [163, 253, 61], [237, 208, 58], [251, 128, 34], [210, 49, 5], [122, 4, 3]],
    coolwarm: [[59, 76, 192], [98, 130, 234], [141, 176, 254], [184, 208, 249], [221, 221, 221], [245, 196, 173], [244, 154, 123], [222, 96, 77], [180, 4, 38]],
    teal: [[19, 26, 34], [20, 60, 70], [26, 110, 110], [60, 170, 160], [79, 209, 197], [190, 245, 240]],
    ice: [[19, 26, 34], [30, 50, 100], [50, 100, 190], [88, 166, 255], [180, 220, 255], [255, 255, 255]],
  };
  const LUTS = {};
  function lut(name) {
    if (LUTS[name]) return LUTS[name];
    const stops = CMAP_STOPS[name] || CMAP_STOPS.inferno;
    const L = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1);
      const k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
      for (let c = 0; c < 3; c++) L[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * f;
    }
    LUTS[name] = L;
    return L;
  }

  function niceStep(range, target) {
    const raw = range / Math.max(target, 1);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }
  function tickLabel(v, step) {
    if (Math.abs(v) < step * 1e-6) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) {
      const [m, e] = v.toExponential(1).split("e");
      return m.replace(/\.0$/, "") + "e" + e.replace("+", "");
    }
    const dec = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    return v.toFixed(Math.min(dec, 6));
  }

  // ================================================================== Plot
  class Plot {
    constructor(container, opts = {}) {
      this.opts = Object.assign({
        aspect: 0.62, xlim: [0, 1], ylim: [0, 1], xlabel: "", ylabel: "", equal: false,
        grid: true, axes: true, margin: null, colorbar: false, xlog: false, ylog: false,
        xtickFormat: null, ytickFormat: null, minHeight: 180, maxHeight: 900,
      }, opts);
      this.container = container;
      this.wrap = document.createElement("div");
      this.wrap.className = "plot-canvas-wrap";
      this.canvas = document.createElement("canvas");
      this.wrap.appendChild(this.canvas);
      container.appendChild(this.wrap);
      this.ctx = this.canvas.getContext("2d");
      this.xlim = this.opts.xlim.slice();
      this.ylim = this.opts.ylim.slice();
      this._clipped = false;
      this._off = document.createElement("canvas");
      this._offCtx = this._off.getContext("2d");
      this.onResize = null;
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(this.wrap);
      this._resize();
    }
    destroy() { this._ro.disconnect(); }

    _resize() {
      // Pop the previous frame's clip from the stack (otherwise axis labels freeze).
      if (this._clipped) { this.ctx.restore(); this._clipped = false; }
      const w = Math.max(this.wrap.clientWidth, 50);
      const h = Math.round(PM.clamp(w * this.opts.aspect, this.opts.minHeight, this.opts.maxHeight));
      this.wrap.style.height = h + "px";
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      this.W = w; this.H = h; this.dpr = dpr;
      const m = this.opts.margin || {};
      const axes = this.opts.axes;
      this.m = {
        l: m.l !== undefined ? m.l : axes ? (this.opts.ylabel ? 56 : 44) : 8,
        r: m.r !== undefined ? m.r : this.opts.colorbar ? 74 : 12,
        t: m.t !== undefined ? m.t : 10,
        b: m.b !== undefined ? m.b : axes ? (this.opts.xlabel ? 42 : 26) : 8,
      };
      this._computeTransform();
      this._clipped = false;
      if (this.onResize) this.onResize();
    }

    setLimits(xlim, ylim) {
      if (xlim) this.xlim = [xlim[0], xlim[1]];
      if (ylim) this.ylim = [ylim[0], ylim[1]];
      this._computeTransform();
    }
    setLabels(xlabel, ylabel) { if (xlabel !== undefined) this.opts.xlabel = xlabel; if (ylabel !== undefined) this.opts.ylabel = ylabel; }

    _computeTransform() {
      const pw = this.W - this.m.l - this.m.r, ph = this.H - this.m.t - this.m.b;
      let [x0, x1] = this.xlim, [y0, y1] = this.ylim;
      if (this.opts.xlog) { x0 = Math.log10(x0); x1 = Math.log10(x1); }
      if (this.opts.ylog) { y0 = Math.log10(y0); y1 = Math.log10(y1); }
      if (this.opts.equal) {
        const sx = pw / (x1 - x0), sy = ph / (y1 - y0), s = Math.min(sx, sy);
        const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
        x0 = cx - pw / s / 2; x1 = cx + pw / s / 2; y0 = cy - ph / s / 2; y1 = cy + ph / s / 2;
      }
      this._v = { x0, x1, y0, y1, pw, ph };
      this.sx = pw / (x1 - x0); this.sy = ph / (y1 - y0);
    }
    /** Data → pixel. */
    X(x) { if (this.opts.xlog) x = Math.log10(x); return this.m.l + (x - this._v.x0) * this.sx; }
    Y(y) { if (this.opts.ylog) y = Math.log10(Math.max(y, 1e-300)); return this.m.t + this._v.ph - (y - this._v.y0) * this.sy; }
    /** Pixel → data. */
    invX(px) { const v = this._v.x0 + (px - this.m.l) / this.sx; return this.opts.xlog ? Math.pow(10, v) : v; }
    invY(py) { const v = this._v.y0 + (this.m.t + this._v.ph - py) / this.sy; return this.opts.ylog ? Math.pow(10, v) : v; }
    get visibleXlim() { return [this._v.x0, this._v.x1]; }
    get visibleYlim() { return [this._v.y0, this._v.y1]; }

    _unclip(fn) {
      const c = this.ctx;
      if (this._clipped) c.restore();
      c.save();
      fn(c);
      c.restore();
      if (this._clipped) this._applyClip();
    }
    _applyClip() {
      const c = this.ctx;
      c.save();
      c.beginPath();
      c.rect(this.m.l, this.m.t, this._v.pw, this._v.ph);
      c.clip();
      this._clipped = true;
    }

    /** Background, grid, axes and labels; then clip to the data area. */
    clear() {
      const c = this.ctx, o = this.opts;
      if (this._clipped) { c.restore(); this._clipped = false; }
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      c.fillStyle = THEME.bg;
      c.fillRect(0, 0, this.W, this.H);
      const { l, t } = this.m, { pw, ph, x0, x1, y0, y1 } = this._v;
      c.fillStyle = "#0f151c";
      c.fillRect(l, t, pw, ph);
      if (o.axes) {
        c.font = THEME.font;
        c.lineWidth = 1;
        // x tikleri
        const xs = [];
        if (o.xlog) { for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) xs.push([e, "10" + supNum(e)]); }
        else { const st = niceStep(x1 - x0, Math.max(3, pw / 90)); for (let v = Math.ceil(x0 / st) * st; v <= x1 + st * 1e-9; v += st) xs.push([v, o.xtickFormat ? o.xtickFormat(v) : tickLabel(v, st)]); }
        const ys = [];
        if (o.ylog) { for (let e = Math.ceil(y0); e <= Math.floor(y1); e++) ys.push([e, "10" + supNum(e)]); }
        else { const st = niceStep(y1 - y0, Math.max(3, ph / 55)); for (let v = Math.ceil(y0 / st) * st; v <= y1 + st * 1e-9; v += st) ys.push([v, o.ytickFormat ? o.ytickFormat(v) : tickLabel(v, st)]); }
        const px = (v) => l + (v - x0) * this.sx, py = (v) => t + ph - (v - y0) * this.sy;
        if (o.grid) {
          c.strokeStyle = THEME.grid;
          c.beginPath();
          for (const [v] of xs) { const p = Math.round(px(v)) + 0.5; c.moveTo(p, t); c.lineTo(p, t + ph); }
          for (const [v] of ys) { const p = Math.round(py(v)) + 0.5; c.moveTo(l, p); c.lineTo(l + pw, p); }
          c.stroke();
        }
        c.strokeStyle = THEME.axis;
        c.strokeRect(l + 0.5, t + 0.5, pw - 1, ph - 1);
        c.fillStyle = THEME.tick;
        c.textAlign = "center"; c.textBaseline = "top";
        for (const [v, s] of xs) c.fillText(s, px(v), t + ph + 5);
        c.textAlign = "right"; c.textBaseline = "middle";
        for (const [v, s] of ys) c.fillText(s, l - 6, py(v));
        c.font = THEME.labelFont;
        c.fillStyle = THEME.text;
        if (o.xlabel) { c.textAlign = "center"; c.textBaseline = "bottom"; c.fillText(o.xlabel, l + pw / 2, this.H - 4); }
        if (o.ylabel) {
          c.save(); c.translate(13, t + ph / 2); c.rotate(-Math.PI / 2);
          c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(o.ylabel, 0, 0); c.restore();
        }
      }
      this._applyClip();
    }

    // -------------------------------------------------------------- primitives
    _style(o, def) {
      const c = this.ctx;
      c.strokeStyle = o.color || def || COLORS.accent;
      c.lineWidth = o.width || 1.6;
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 1;
      c.setLineDash(o.dash || []);
      c.lineJoin = "round"; c.lineCap = "round";
    }
    _reset() { const c = this.ctx; c.globalAlpha = 1; c.setLineDash([]); }

    /** Polyline from xs/ys arrays; NaN values break the line. */
    line(xs, ys, o = {}) {
      const c = this.ctx, n = Math.min(xs.length, ys.length);
      this._style(o);
      c.beginPath();
      let pen = false;
      for (let i = 0; i < n; i++) {
        const y = ys[i], x = xs[i];
        if (!isFinite(y) || !isFinite(x)) { pen = false; continue; }
        const X = this.X(x), Y = this.Y(y);
        if (pen) c.lineTo(X, Y); else { c.moveTo(X, Y); pen = true; }
      }
      c.stroke();
      this._reset();
    }
    /** Plot y = f(x) over the visible x range. */
    fn(f, o = {}) {
      const n = o.samples || 400, [a, b] = this.xlim;
      const xs = PM.linspace(a, b, n), ys = new Float64Array(n);
      for (let i = 0; i < n; i++) ys[i] = f(xs[i]);
      this.line(xs, ys, o);
    }
    /** Fill between ys1 and ys0 (array or constant baseline). */
    fill(xs, ys1, ys0, o = {}) {
      const c = this.ctx, n = xs.length;
      c.fillStyle = o.color || COLORS.accent;
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 0.35;
      c.beginPath();
      c.moveTo(this.X(xs[0]), this.Y(ys1[0]));
      for (let i = 1; i < n; i++) c.lineTo(this.X(xs[i]), this.Y(ys1[i]));
      for (let i = n - 1; i >= 0; i--) {
        const y0 = typeof ys0 === "number" || ys0 == null ? (ys0 || 0) : ys0[i];
        c.lineTo(this.X(xs[i]), this.Y(y0));
      }
      c.closePath();
      c.fill();
      this._reset();
    }
    /** Scatter points. o.colors: optional per-point colour array. */
    points(xs, ys, o = {}) {
      const c = this.ctx, n = Math.min(xs.length, ys.length), r = o.size || 2.5;
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 1;
      if (o.square) {
        c.fillStyle = o.color || COLORS.accent;
        for (let i = 0; i < n; i++) { if (o.colors) c.fillStyle = o.colors[i]; c.fillRect(this.X(xs[i]) - r, this.Y(ys[i]) - r, 2 * r, 2 * r); }
      } else if (o.colors) {
        for (let i = 0; i < n; i++) { c.fillStyle = o.colors[i]; c.beginPath(); c.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 2 * Math.PI); c.fill(); }
      } else {
        c.fillStyle = o.color || COLORS.accent;
        c.beginPath();
        for (let i = 0; i < n; i++) { const X = this.X(xs[i]), Y = this.Y(ys[i]); c.moveTo(X + r, Y); c.arc(X, Y, r, 0, 2 * Math.PI); }
        c.fill();
      }
      if (o.stroke) {
        c.strokeStyle = o.stroke; c.lineWidth = o.strokeWidth || 1;
        c.beginPath();
        for (let i = 0; i < n; i++) { const X = this.X(xs[i]), Y = this.Y(ys[i]); c.moveTo(X + r, Y); c.arc(X, Y, r, 0, 2 * Math.PI); }
        c.stroke();
      }
      this._reset();
    }
    /** Circle of radius r in data units (o.px=true: r in pixels). */
    circle(x, y, r, o = {}) {
      const c = this.ctx;
      const R = o.px ? r : r * this.sx;
      c.beginPath(); c.arc(this.X(x), this.Y(y), Math.max(R, 0.5), 0, 2 * Math.PI);
      if (o.fill !== false) { c.fillStyle = o.color || COLORS.accent; c.globalAlpha = o.alpha !== undefined ? o.alpha : 1; c.fill(); }
      if (o.stroke) { c.strokeStyle = o.stroke; c.lineWidth = o.strokeWidth || 1.5; c.globalAlpha = 1; c.stroke(); }
      this._reset();
    }
    segment(x0, y0, x1, y1, o = {}) {
      const c = this.ctx; this._style(o);
      c.beginPath(); c.moveTo(this.X(x0), this.Y(y0)); c.lineTo(this.X(x1), this.Y(y1)); c.stroke();
      this._reset();
    }
    /** Arrow (vector). */
    arrow(x0, y0, x1, y1, o = {}) {
      const c = this.ctx; this._style(o);
      const X0 = this.X(x0), Y0 = this.Y(y0), X1 = this.X(x1), Y1 = this.Y(y1);
      const a = Math.atan2(Y1 - Y0, X1 - X0), h = o.head || 9;
      c.beginPath(); c.moveTo(X0, Y0); c.lineTo(X1, Y1); c.stroke();
      c.fillStyle = o.color || COLORS.accent;
      c.beginPath(); c.moveTo(X1, Y1);
      c.lineTo(X1 - h * Math.cos(a - 0.4), Y1 - h * Math.sin(a - 0.4));
      c.lineTo(X1 - h * Math.cos(a + 0.4), Y1 - h * Math.sin(a + 0.4));
      c.closePath(); c.fill();
      this._reset();
    }
    vline(x, o = {}) {
      const c = this.ctx; this._style(Object.assign({ width: 1.2, color: COLORS.muted }, o));
      const X = this.X(x); c.beginPath(); c.moveTo(X, this.m.t); c.lineTo(X, this.m.t + this._v.ph); c.stroke(); this._reset();
    }
    hline(y, o = {}) {
      const c = this.ctx; this._style(Object.assign({ width: 1.2, color: COLORS.muted }, o));
      const Y = this.Y(y); c.beginPath(); c.moveTo(this.m.l, Y); c.lineTo(this.m.l + this._v.pw, Y); c.stroke(); this._reset();
    }
    /** Rectangle (data coordinates). */
    rect(x0, y0, x1, y1, o = {}) {
      const c = this.ctx;
      const X = this.X(Math.min(x0, x1)), Y = this.Y(Math.max(y0, y1));
      const w = Math.abs(this.X(x1) - this.X(x0)), h = Math.abs(this.Y(y1) - this.Y(y0));
      if (o.fill !== false) { c.fillStyle = o.color || COLORS.accent; c.globalAlpha = o.alpha !== undefined ? o.alpha : 1; c.fillRect(X, Y, w, h); }
      if (o.stroke) { c.globalAlpha = 1; c.strokeStyle = o.stroke; c.lineWidth = o.strokeWidth || 1; c.setLineDash(o.dash || []); c.strokeRect(X, Y, w, h); }
      this._reset();
    }
    /** Bar chart: centres xs, heights hs, width w (data units). */
    bars(xs, hs, w, o = {}) {
      const c = this.ctx;
      c.fillStyle = o.color || COLORS.accent;
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 0.75;
      const base = o.base || 0;
      for (let i = 0; i < xs.length; i++) {
        const X0 = this.X(xs[i] - w / 2), X1 = this.X(xs[i] + w / 2), Y0 = this.Y(base), Y1 = this.Y(hs[i]);
        if (o.colors) c.fillStyle = o.colors[i];
        c.fillRect(X0, Math.min(Y0, Y1), Math.max(X1 - X0 - (o.gap || 0), 0.5), Math.abs(Y1 - Y0));
      }
      this._reset();
    }
    /** Closed polygon. */
    poly(xs, ys, o = {}) {
      const c = this.ctx;
      c.beginPath();
      for (let i = 0; i < xs.length; i++) { const X = this.X(xs[i]), Y = this.Y(ys[i]); if (i) c.lineTo(X, Y); else c.moveTo(X, Y); }
      c.closePath();
      if (o.fill !== false) { c.fillStyle = o.color || COLORS.accent; c.globalAlpha = o.alpha !== undefined ? o.alpha : 0.5; c.fill(); }
      if (o.stroke) { c.globalAlpha = 1; c.strokeStyle = o.stroke; c.lineWidth = o.strokeWidth || 1; c.stroke(); }
      this._reset();
    }
    /** Text at a data coordinate. */
    text(x, y, s, o = {}) { this.textPx(this.X(x) + (o.dx || 0), this.Y(y) + (o.dy || 0), s, o); }
    /** Text at a pixel coordinate (relative to the canvas top-left). */
    textPx(px, py, s, o = {}) {
      const c = this.ctx;
      c.font = (o.bold ? "600 " : "") + (o.size || 12) + 'px "Segoe UI", system-ui, sans-serif';
      c.textAlign = o.align || "left"; c.textBaseline = o.baseline || "middle";
      if (o.bg) {
        const w = c.measureText(s).width, h = (o.size || 12) + 6;
        let bx = px - (c.textAlign === "center" ? w / 2 : c.textAlign === "right" ? w : 0) - 4;
        c.fillStyle = o.bg; c.globalAlpha = 0.85;
        c.fillRect(bx, py - h / 2, w + 8, h); c.globalAlpha = 1;
      }
      c.fillStyle = o.color || THEME.text;
      c.fillText(s, px, py);
    }
    /** Info label in a corner of the data area (e.g. "t = 1.25"). corner: tl|tr|bl|br */
    label(s, corner = "tl", o = {}) {
      const pad = 8, l = this.m.l, t = this.m.t, r = l + this._v.pw, b = t + this._v.ph;
      const lines = Array.isArray(s) ? s : [s];
      const lh = (o.size || 12) + 5;
      lines.forEach((line, k) => {
        const px = corner[1] === "l" ? l + pad : r - pad;
        const py = corner[0] === "t" ? t + pad + 8 + k * lh : b - pad - 8 - (lines.length - 1 - k) * lh;
        this.textPx(px, py, line, Object.assign({ align: corner[1] === "l" ? "left" : "right", bg: "#0f151c" }, o));
      });
    }
    /** Legend box. items: [{label, color, dash?, type:'line'|'box'|'dot'}] */
    legend(items, corner = "tr") {
      const c = this.ctx;
      c.font = '11.5px "Segoe UI", system-ui, sans-serif';
      const w = Math.max(...items.map((it) => c.measureText(it.label).width)) + 38, h = items.length * 17 + 8;
      const x = corner[1] === "r" ? this.m.l + this._v.pw - w - 8 : this.m.l + 8;
      const y = corner[0] === "t" ? this.m.t + 8 : this.m.t + this._v.ph - h - 8;
      c.fillStyle = "#0f151c"; c.globalAlpha = 0.88; c.fillRect(x, y, w, h); c.globalAlpha = 1;
      c.strokeStyle = THEME.axis; c.lineWidth = 1; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      items.forEach((it, k) => {
        const yy = y + 12 + k * 17;
        c.strokeStyle = it.color; c.fillStyle = it.color; c.lineWidth = 2.2; c.setLineDash(it.dash || []);
        if (it.type === "box") { c.globalAlpha = 0.7; c.fillRect(x + 8, yy - 5, 18, 10); c.globalAlpha = 1; }
        else if (it.type === "dot") { c.beginPath(); c.arc(x + 17, yy, 4, 0, 2 * Math.PI); c.fill(); }
        else { c.beginPath(); c.moveTo(x + 8, yy); c.lineTo(x + 26, yy); c.stroke(); }
        c.setLineDash([]);
        c.fillStyle = THEME.text; c.textAlign = "left"; c.textBaseline = "middle";
        c.fillText(it.label, x + 32, yy);
      });
    }

    /**
     * Heatmap. z: nx*ny (row-major, j=0 → bottom edge y0).
     * o: {x0,x1,y0,y1, vmin, vmax, cmap, smooth(true), scale:'linear'|'sqrt'|'log', colorbar(label)}
     */
    heatmap(z, nx, ny, o = {}) {
      const vmin = o.vmin !== undefined ? o.vmin : PM.min(z), vmax = o.vmax !== undefined ? o.vmax : PM.max(z);
      const L = lut(o.cmap || "inferno"), scale = o.scale || "linear";
      if (this._off.width !== nx || this._off.height !== ny) { this._off.width = nx; this._off.height = ny; this._img = null; }
      if (!this._img) this._img = this._offCtx.createImageData(nx, ny);
      const d = this._img.data, span = vmax - vmin || 1;
      const lmin = scale === "log" ? Math.log10(Math.max(vmin, vmax * 1e-6)) : 0, lmax = scale === "log" ? Math.log10(vmax) : 1;
      for (let j = 0; j < ny; j++) {
        const row = (ny - 1 - j) * nx;
        for (let i = 0; i < nx; i++) {
          let t = (z[j * nx + i] - vmin) / span;
          if (scale === "sqrt") t = Math.sqrt(Math.max(t, 0));
          else if (scale === "log") t = (Math.log10(Math.max(z[j * nx + i], 1e-300)) - lmin) / (lmax - lmin || 1);
          const k = (t <= 0 ? 0 : t >= 1 ? 255 : (t * 255) | 0) * 3, p = (row + i) * 4;
          d[p] = L[k]; d[p + 1] = L[k + 1]; d[p + 2] = L[k + 2]; d[p + 3] = 255;
        }
      }
      this._offCtx.putImageData(this._img, 0, 0);
      const x0 = o.x0 !== undefined ? o.x0 : this.xlim[0], x1 = o.x1 !== undefined ? o.x1 : this.xlim[1];
      const y0 = o.y0 !== undefined ? o.y0 : this.ylim[0], y1 = o.y1 !== undefined ? o.y1 : this.ylim[1];
      const c = this.ctx;
      c.imageSmoothingEnabled = o.smooth !== false;
      c.imageSmoothingQuality = "high";
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 1;
      c.drawImage(this._off, this.X(x0), this.Y(y1), this.X(x1) - this.X(x0), this.Y(y0) - this.Y(y1));
      c.globalAlpha = 1;
      if (o.colorbar !== undefined && o.colorbar !== false) this.colorbar({ cmap: o.cmap, vmin, vmax, label: typeof o.colorbar === "string" ? o.colorbar : "" });
    }
    /** Colour bar on the right (reserve the margin with opts.colorbar:true). */
    colorbar({ cmap, vmin, vmax, label }) {
      this._unclip((c) => {
        const L = lut(cmap || "inferno");
        const x = this.m.l + this._v.pw + 12, y = this.m.t, h = this._v.ph, w = 12;
        for (let i = 0; i < h; i++) {
          const k = Math.floor((1 - i / h) * 255) * 3;
          c.fillStyle = `rgb(${L[k]},${L[k + 1]},${L[k + 2]})`;
          c.fillRect(x, y + i, w, 1.5);
        }
        c.strokeStyle = THEME.axis; c.strokeRect(x + 0.5, y + 0.5, w, h - 1);
        c.fillStyle = THEME.tick; c.font = THEME.font; c.textAlign = "left"; c.textBaseline = "middle";
        const st = niceStep(vmax - vmin || 1, 5);
        for (let v = Math.ceil(vmin / st) * st; v <= vmax + st * 1e-9; v += st) {
          const yy = y + h - ((v - vmin) / (vmax - vmin || 1)) * h;
          c.fillText(tickLabel(v, st), x + w + 4, yy);
        }
        if (label) {
          c.save(); c.translate(this.W - 8, y + h / 2); c.rotate(-Math.PI / 2);
          c.textAlign = "center"; c.fillStyle = THEME.text; c.font = THEME.labelFont; c.fillText(label, 0, 0); c.restore();
        }
      });
    }
    /** Free drawing: fn(ctx, plot) — context clipped to the data area. */
    custom(fn) { fn(this.ctx, this); this._reset(); }
  }
  function supNum(e) {
    const m = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    return String(e).split("").map((ch) => m[ch] || ch).join("");
  }

  // ================================================================== View3D
  /**
   * Orthographic 3D view (drag to rotate, wheel to zoom).
   *  const v = new View3D(konteyner, {aspect:0.8, extent: 10, yaw:0.8, pitch:0.4, autoRotate:0});
   *  v.clear(); v.axes(); v.points(xs,ys,zs,{values, cmap, size, alpha});  v.surface(...)
   * Coordinates: z points up.
   */
  class View3D {
    constructor(container, opts = {}) {
      this.opts = Object.assign({ aspect: 0.8, extent: 1, yaw: 0.8, pitch: 0.45, zoom: 1, autoRotate: 0, minHeight: 240, maxHeight: 820 }, opts);
      this.yaw = this.opts.yaw; this.pitch = this.opts.pitch; this.zoom = this.opts.zoom;
      this.wrap = document.createElement("div");
      this.wrap.className = "plot-canvas-wrap";
      this.canvas = document.createElement("canvas");
      this.wrap.appendChild(this.canvas);
      container.appendChild(this.wrap);
      container.classList.add("draggable");
      this.ctx = this.canvas.getContext("2d");
      this.onChange = null; // redraw request after rotation
      this.onResize = null;
      let drag = null;
      this._down = (e) => { drag = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch }; this.canvas.setPointerCapture(e.pointerId); };
      this._move = (e) => {
        if (!drag) return;
        this.yaw = drag.yaw + (e.clientX - drag.x) * 0.01;
        this.pitch = PM.clamp(drag.pitch + (e.clientY - drag.y) * 0.01, -1.55, 1.55);
        this.userMoved = true;
        if (this.onChange) this.onChange();
      };
      this._up = () => { drag = null; };
      this._wheel = (e) => { e.preventDefault(); this.zoom = PM.clamp(this.zoom * Math.exp(-e.deltaY * 0.001), 0.3, 5); if (this.onChange) this.onChange(); };
      this.canvas.addEventListener("pointerdown", this._down);
      this.canvas.addEventListener("pointermove", this._move);
      this.canvas.addEventListener("pointerup", this._up);
      this.canvas.addEventListener("wheel", this._wheel, { passive: false });
      this._ro = new ResizeObserver(() => this._resize());
      this._ro.observe(this.wrap);
      this._resize();
    }
    destroy() { this._ro.disconnect(); }
    _resize() {
      const w = Math.max(this.wrap.clientWidth, 50);
      const h = Math.round(PM.clamp(w * this.opts.aspect, this.opts.minHeight, this.opts.maxHeight));
      this.wrap.style.height = h + "px";
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
      this.W = w; this.H = h; this.dpr = dpr;
      if (this.onResize) this.onResize();
    }
    _basis() {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const s = (Math.min(this.W, this.H) * 0.42 * this.zoom) / this.opts.extent;
      return { cy, sy, cp, sp, s, cx: this.W / 2, cz: this.H / 2 };
    }
    /** 3D → [px, py, depth] (larger depth = closer to the camera). */
    project(x, y, z, B) {
      B = B || this._basis();
      const xr = x * B.cy - y * B.sy, yr = x * B.sy + y * B.cy;
      const up = z * B.cp - yr * B.sp, depth = z * B.sp + yr * B.cp;
      return [B.cx + xr * B.s, B.cz - up * B.s, depth];
    }
    clear() {
      const c = this.ctx;
      c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      c.fillStyle = "#0f151c";
      c.fillRect(0, 0, this.W, this.H);
    }
    /** Axes (x reddish, y greenish, z blue) + labels. */
    axes(len, labels = ["x", "y", "z"]) {
      len = len || this.opts.extent;
      const c = this.ctx, B = this._basis(), o = this.project(0, 0, 0, B);
      const ax = [[len, 0, 0, "#f78166"], [0, len, 0, "#7ee787"], [0, 0, len, "#79c0ff"]];
      c.lineWidth = 1.2; c.font = '12px "Segoe UI", system-ui, sans-serif';
      ax.forEach(([x, y, z, col], k) => {
        const p = this.project(x, y, z, B);
        c.strokeStyle = col; c.globalAlpha = 0.8;
        c.beginPath(); c.moveTo(o[0], o[1]); c.lineTo(p[0], p[1]); c.stroke();
        c.globalAlpha = 1; c.fillStyle = col; c.fillText(labels[k], p[0] + 4, p[1] - 4);
      });
    }
    /** Wire-frame box [-e,e]^3. */
    box(e, o = {}) {
      e = e || this.opts.extent;
      const c = this.ctx, B = this._basis();
      const V = [[-e, -e, -e], [e, -e, -e], [e, e, -e], [-e, e, -e], [-e, -e, e], [e, -e, e], [e, e, e], [-e, e, e]].map((v) => this.project(v[0], v[1], v[2], B));
      const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
      c.strokeStyle = o.color || "rgba(139,152,168,0.25)"; c.lineWidth = 1;
      c.beginPath(); E.forEach(([a, b]) => { c.moveTo(V[a][0], V[a][1]); c.lineTo(V[b][0], V[b][1]); }); c.stroke();
    }
    /**
     * Point cloud. Colour by o.values + o.cmap (vmin/vmax), or a single o.color.
     * For speed, colours are binned into 48 buckets.
     */
    points(xs, ys, zs, o = {}) {
      const c = this.ctx, B = this._basis(), n = xs.length, r = o.size || 1.6;
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 0.6;
      if (!o.values) {
        c.fillStyle = o.color || COLORS.accent;
        for (let i = 0; i < n; i++) { const p = this.project(xs[i], ys[i], zs[i], B); c.fillRect(p[0] - r, p[1] - r, 2 * r, 2 * r); }
      } else {
        const L = lut(o.cmap || "turbo"), K = 48;
        const vmin = o.vmin !== undefined ? o.vmin : PM.min(o.values), vmax = o.vmax !== undefined ? o.vmax : PM.max(o.values);
        const buckets = Array.from({ length: K }, () => []);
        for (let i = 0; i < n; i++) {
          const t = PM.clamp((o.values[i] - vmin) / (vmax - vmin || 1), 0, 0.9999);
          buckets[(t * K) | 0].push(i);
        }
        for (let k = 0; k < K; k++) {
          const li = Math.floor(((k + 0.5) / K) * 255) * 3;
          c.fillStyle = `rgb(${L[li]},${L[li + 1]},${L[li + 2]})`;
          const idx = buckets[k];
          for (let q = 0; q < idx.length; q++) { const i = idx[q]; const p = this.project(xs[i], ys[i], zs[i], B); c.fillRect(p[0] - r, p[1] - r, 2 * r, 2 * r); }
        }
      }
      c.globalAlpha = 1;
    }
    /** 3D polyline. */
    line3(xs, ys, zs, o = {}) {
      const c = this.ctx, B = this._basis();
      c.strokeStyle = o.color || COLORS.accent; c.lineWidth = o.width || 1.8; c.globalAlpha = o.alpha !== undefined ? o.alpha : 1;
      c.setLineDash(o.dash || []);
      c.beginPath();
      for (let i = 0; i < xs.length; i++) { const p = this.project(xs[i], ys[i], zs[i], B); if (i) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]); }
      if (o.close) c.closePath();
      c.stroke(); c.globalAlpha = 1; c.setLineDash([]);
    }
    point3(x, y, z, o = {}) {
      const c = this.ctx, p = this.project(x, y, z);
      c.fillStyle = o.color || COLORS.accent3; c.beginPath(); c.arc(p[0], p[1], o.size || 5, 0, 2 * Math.PI); c.fill();
      if (o.label) { c.font = '12px "Segoe UI", system-ui, sans-serif'; c.fillStyle = COLORS.text; c.fillText(o.label, p[0] + 8, p[1] - 6); }
    }
    arrow3(x0, y0, z0, x1, y1, z1, o = {}) {
      const c = this.ctx, B = this._basis(), a = this.project(x0, y0, z0, B), b = this.project(x1, y1, z1, B);
      c.strokeStyle = c.fillStyle = o.color || COLORS.accent; c.lineWidth = o.width || 2.2;
      c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
      const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), h = 9;
      c.beginPath(); c.moveTo(b[0], b[1]);
      c.lineTo(b[0] - h * Math.cos(ang - 0.4), b[1] - h * Math.sin(ang - 0.4));
      c.lineTo(b[0] - h * Math.cos(ang + 0.4), b[1] - h * Math.sin(ang + 0.4)); c.closePath(); c.fill();
    }
    /**
     * Surface: X,Y,Z (nu*nv, row-major, index j*nu+i). Painter's algorithm + simple shading.
     * o: {cmap, values (for colour; defaults to Z), vmin, vmax, alpha, wire(color)}
     */
    surface(X, Y, Z, nu, nv, o = {}) {
      const c = this.ctx, B = this._basis(), L = lut(o.cmap || "viridis");
      const P = new Float64Array(nu * nv * 3);
      for (let k = 0; k < nu * nv; k++) { const p = this.project(X[k], Y[k], Z[k], B); P[3 * k] = p[0]; P[3 * k + 1] = p[1]; P[3 * k + 2] = p[2]; }
      const vals = o.values || Z;
      const vmin = o.vmin !== undefined ? o.vmin : PM.min(vals), vmax = o.vmax !== undefined ? o.vmax : PM.max(vals);
      const quads = [];
      for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nu - 1; i++) {
        const a = j * nu + i, b = a + 1, cc = a + nu + 1, d = a + nu;
        if (!isFinite(Z[a] + Z[b] + Z[cc] + Z[d])) continue;
        quads.push([a, b, cc, d, (P[3 * a + 2] + P[3 * b + 2] + P[3 * cc + 2] + P[3 * d + 2]) / 4]);
      }
      quads.sort((q1, q2) => q1[4] - q2[4]);
      c.globalAlpha = o.alpha !== undefined ? o.alpha : 0.95;
      c.lineWidth = 0.6;
      for (const [a, b, cc, d] of quads) {
        // simple lighting from the face normal
        const ux = X[b] - X[a], uy = Y[b] - Y[a], uz = Z[b] - Z[a], vx = X[d] - X[a], vy = Y[d] - Y[a], vz = Z[d] - Z[a];
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const nn = Math.hypot(nx, ny, nz) || 1; nx /= nn; ny /= nn; nz /= nn;
        const shade = 0.55 + 0.45 * Math.abs(nx * 0.3 + ny * 0.4 + nz * 0.86);
        const t = PM.clamp(((vals[a] + vals[b] + vals[cc] + vals[d]) / 4 - vmin) / (vmax - vmin || 1), 0, 1);
        const li = ((t * 255) | 0) * 3;
        c.fillStyle = `rgb(${(L[li] * shade) | 0},${(L[li + 1] * shade) | 0},${(L[li + 2] * shade) | 0})`;
        c.beginPath();
        c.moveTo(P[3 * a], P[3 * a + 1]); c.lineTo(P[3 * b], P[3 * b + 1]); c.lineTo(P[3 * cc], P[3 * cc + 1]); c.lineTo(P[3 * d], P[3 * d + 1]);
        c.closePath(); c.fill();
        if (o.wire) { c.strokeStyle = o.wire; c.stroke(); }
      }
      c.globalAlpha = 1;
    }
    text(s, px, py, o = {}) {
      const c = this.ctx; c.font = (o.size || 12) + 'px "Segoe UI", system-ui, sans-serif';
      c.fillStyle = o.color || COLORS.text; c.textAlign = o.align || "left"; c.fillText(s, px, py);
    }
  }

  window.Plot = Plot;
  window.View3D = View3D;
  window.PlotColors = COLORS;
  window.PlotCycle = CYCLE;
  window.colormap = function (name, t) { const L = lut(name), k = Math.floor(PM.clamp(t, 0, 1) * 255) * 3; return `rgb(${L[k]},${L[k + 1]},${L[k + 2]})`; };
})();
