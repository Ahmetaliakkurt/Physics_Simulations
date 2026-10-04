/* =====================================================================
 * App — registry, hash routing (#/sim-id), control panel and a single
 * requestAnimationFrame loop (60 fps).
 *
 * Simulation spec:
 *   App.register({
 *     id, category:'classical'|'statistical'|'quantum'|'special', group?, order, title, icon, subtitle,
 *     notes:[{type:'info'|'warn', html}], theory:"HTML + $LaTeX$",
 *     animated:true, speed:{min,max,value,step},
 *     controls:[ {id,type:'slider',label,min,max,step,value,live?,rebuild?,fmt?,help?,visibleIf?}, ... ],
 *     mount(api) { return { reset(), step(dt), render(), onParam(id,v), onAction(id), destroy() } }
 *   })
 *
 * On control change:  live:true  -> onParam(id,v) + redraw
 *                     (default)  -> onParam(id,v) then reset()
 *                     rebuild:true -> the whole page is rebuilt (mount() runs again) keeping all
 *                                     current parameter values — use it for "mode" selectors that
 *                                     change which plots/metrics exist.
 * ===================================================================== */
(function () {
  "use strict";

  const CATEGORIES = {
    classical: {
      label: "Classical Physics", icon: "🪐", color: "#4fd1c5",
      desc: "Mechanics, chaos, projectile motion with drag, electric and magnetic fields, circuits, induction, optics and the mathematical tools behind them.",
      groups: ["Mechanics", "Electromagnetism", "Optics", "Mathematical Methods"],
    },
    statistical: {
      label: "Statistical Physics", icon: "📊", color: "#8b5cf6",
      desc: "Random walks, the central limit theorem, quantum statistics, the ideal gas and hydrogen-like orbitals as probability clouds.",
    },
    quantum: {
      label: "Quantum Mechanics", icon: "⚛️", color: "#f59e0b",
      desc: "The Schrödinger equation solved live: wells, tunnelling, wave packets, the hydrogen atom, Stern–Gerlach and the double slit.",
    },
    special: {
      label: "Special Projects", icon: "🚀", color: "#3fb950",
      desc: "A graph-colouring Sudoku solver and a pursuit–evasion simulation.",
    },
  };

  const registry = [];
  let current = null; // {spec, inst, api, playing, speed, dirty}
  const $ = (sel, el = document) => el.querySelector(sel);
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "html") e.innerHTML = v;
      else if (k === "style") e.style.cssText = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) e.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    return e;
  };

  function renderMath(el) {
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(el, {
          delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }],
          throwOnError: false,
        });
      } catch (e) { console.warn(e); }
    }
  }

  /** Sims of a category, sorted by group (category's group order) then by `order`. */
  function simsOf(key) {
    const groups = CATEGORIES[key].groups || [];
    const gi = (s) => (s.group && groups.indexOf(s.group) >= 0 ? groups.indexOf(s.group) : groups.length);
    return registry.filter((s) => s.category === key).sort((a, b) => gi(a) - gi(b) || a.order - b.order);
  }
  /** Interleave group headings (as {heading}) into a sorted sim list. */
  function withHeadings(sims) {
    const out = []; let last = null;
    for (const s of sims) {
      if (s.group && s.group !== last) { out.push({ heading: s.group }); last = s.group; }
      out.push(s);
    }
    return out;
  }

  // ------------------------------------------------------------------ sidebar
  function buildSidebar() {
    const sb = $("#sidebar");
    sb.innerHTML = "";
    sb.appendChild(h("div", { class: "brand", onclick: () => (location.hash = "#/") },
      h("div", { class: "brand-logo" }, "⚛️"),
      h("div", {}, h("div", { class: "brand-title" }, "Physics Simulation Lab"), h("div", { class: "brand-sub" }, "Classical · Statistical · Quantum"))));
    sb.appendChild(h("a", { class: "nav-home", href: "#/", "data-id": "" }, "🏠  Home"));
    for (const [key, cat] of Object.entries(CATEGORIES)) {
      const sims = simsOf(key);
      if (!sims.length) continue;
      const box = h("div", { class: "nav-cat", style: `--cat-color:${cat.color}` });
      const head = h("button", { class: "nav-cat-head", onclick: () => box.classList.toggle("collapsed") },
        h("span", {}, `${cat.icon}  ${cat.label}`), h("span", { class: "chev" }, "▼"));
      const items = h("div", { class: "nav-items" },
        withHeadings(sims).map((s) => s.heading
          ? h("div", { class: "nav-group" }, s.heading)
          : h("a", { class: "nav-item", href: "#/" + s.id, "data-id": s.id }, h("span", { class: "ico" }, s.icon), s.title)));
      box.append(head, items);
      sb.appendChild(box);
    }
  }
  function markActive(id) {
    document.querySelectorAll("#sidebar [data-id]").forEach((a) => a.classList.toggle("active", a.dataset.id === id));
    const act = document.querySelector(`#sidebar [data-id="${id}"]`);
    if (act && act.closest(".nav-cat")) act.closest(".nav-cat").classList.remove("collapsed");
  }

  // ------------------------------------------------------------------ home page
  function showHome() {
    unmount();
    const main = $("#main");
    main.innerHTML = "";
    main.appendChild(h("div", { class: "hero" },
      h("div", { class: "kicker", style: "--cat-color:#4fd1c5" }, "Physics Simulation Lab"),
      h("h1", {}, "Classical, Statistical and Quantum Physics"),
      h("p", {}, `${registry.length} interactive simulations. Every one is computed live in your browser at 60 frames per second — ` +
        "no server, no installation. Pick a simulation from the menu or the list below; changes to the parameters take effect instantly, " +
        "and each page explains exactly which equations are being solved and how.")));
    const grid = h("div", { class: "cat-grid" });
    for (const [key, cat] of Object.entries(CATEGORIES)) {
      const sims = simsOf(key);
      if (!sims.length) continue;
      grid.appendChild(h("div", { class: "cat-card", style: `--cat-color:${cat.color}` },
        h("div", { style: "font-size:26px" }, cat.icon), h("h2", {}, cat.label),
        h("p", { style: "color:var(--muted);font-size:13px;margin:0 0 10px;line-height:1.5" }, cat.desc),
        h("ul", {}, withHeadings(sims).map((s) => s.heading
          ? h("li", { class: "cat-group" }, s.heading)
          : h("li", {}, h("a", { href: "#/" + s.id }, h("span", {}, s.icon), s.title))))));
    }
    main.appendChild(grid);
    markActive("");
    document.title = "Physics Simulation Lab";
  }

  // ------------------------------------------------------------------ controls
  function fmtVal(c, v) {
    if (c.fmt) return c.fmt(v);
    if (typeof v !== "number") return String(v);
    const step = c.step || 1;
    const dec = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
    return v.toFixed(dec) + (c.unit ? " " + c.unit : "");
  }
  function buildControls(spec, api, panel, preset) {
    const els = {};
    panel.appendChild(h("h3", {}, "Parameters"));
    for (const c0 of spec.controls || []) {
      const c = Object.assign({}, c0); // per-mount copy: setControl() never mutates the spec
      if (preset && c.id && preset[c.id] !== undefined) c.value = preset[c.id];
      if (c.type === "section") { const s = h("div", { class: "ctl-section" }, c.label); panel.appendChild(s); if (c.id) els[c.id] = { row: s, c }; continue; }
      if (c.type === "info") { const s = h("div", { class: "ctl-help", html: c.html }); panel.appendChild(s); if (c.id) els[c.id] = { row: s, c }; continue; }
      const row = h("div", { class: "ctl" });
      let input, valEl = null;
      const fire = (v) => {
        api.params[c.id] = v;
        if (valEl) valEl.textContent = fmtVal(c, v);
        onControlChange(c, v);
      };
      if (c.type === "slider") {
        valEl = h("span", { class: "val" }, fmtVal(c, c.value));
        input = h("input", { type: "range", min: c.min, max: c.max, step: c.step || 1, value: c.value });
        input.addEventListener("input", () => fire(parseFloat(input.value)));
        row.append(h("div", { class: "ctl-label" }, h("span", { html: c.label }), valEl), input);
      } else if (c.type === "select") {
        input = h("select", {}, (c.options || []).map((o) => {
          const val = typeof o === "object" ? o.value : o, lab = typeof o === "object" ? o.label : o;
          return h("option", { value: val, selected: String(val) === String(c.value) ? "selected" : null }, lab);
        }));
        input.addEventListener("change", () => fire(input.value));
        row.append(h("div", { class: "ctl-label" }, h("span", { html: c.label })), input);
      } else if (c.type === "checkbox") {
        input = h("input", { type: "checkbox" });
        input.checked = !!c.value;
        input.addEventListener("change", () => fire(input.checked));
        row.append(h("label", { class: "ctl-check" }, input, h("span", { html: c.label })));
      } else if (c.type === "number") {
        input = h("input", { type: "number", min: c.min, max: c.max, step: c.step || 1, value: c.value });
        input.addEventListener("change", () => { let v = parseFloat(input.value); if (!isFinite(v)) v = c.value; fire(v); });
        row.append(h("div", { class: "ctl-label" }, h("span", { html: c.label })), input);
      } else if (c.type === "textarea") {
        input = h("textarea", { rows: c.rows || 6, spellcheck: "false" });
        input.value = c.value || "";
        input.addEventListener("input", () => { api.params[c.id] = input.value; });
        row.append(h("div", { class: "ctl-label" }, h("span", { html: c.label })), input);
      } else if (c.type === "button") {
        input = h("button", { class: "btn full" + (c.primary ? " primary" : "") }, c.label);
        input.addEventListener("click", () => { if (current && current.inst.onAction) current.inst.onAction(c.id); current && (current.dirty = true); });
        row.append(input);
      }
      if (c.help) row.appendChild(h("div", { class: "ctl-help", html: c.help }));
      if (c.type !== "button") api.params[c.id] = c.value;
      panel.appendChild(row);
      els[c.id] = { row, input, valEl, c };
    }
    return els;
  }
  function onControlChange(c, v) {
    if (!current) return;
    if (c.rebuild) {
      const keep = Object.assign({}, current.api.params);
      const spec = current.spec, scroll = $("#main").scrollTop;
      showSim(spec, keep);
      $("#main").scrollTop = scroll;
      return;
    }
    applyVisibility();
    try {
      if (c.live && current.inst.onParam) current.inst.onParam(c.id, v);
      else if (!c.live) { if (current.inst.onParam) current.inst.onParam(c.id, v); resetSim(); }
    } catch (e) { showError(e); }
    current.dirty = true;
  }
  function applyVisibility() {
    if (!current) return;
    for (const { row, c } of Object.values(current.controlEls)) {
      if (c.visibleIf) row.style.display = c.visibleIf(current.api.params) ? "" : "none";
    }
  }
  function resetSim() {
    if (!current) return;
    current.api.time = 0;
    try { if (current.inst.reset) current.inst.reset(); } catch (e) { showError(e); }
    current.dirty = true;
  }
  function showError(e) {
    console.error(e);
    const box = $("#sim-error");
    if (box) { box.style.display = ""; box.textContent = "Error: " + (e && e.message ? e.message : e); }
  }

  // ------------------------------------------------------------------ simulation page
  function showSim(spec, preset) {
    unmount();
    const cat = CATEGORIES[spec.category];
    const main = $("#main");
    main.innerHTML = "";
    main.style.setProperty("--cat-color", cat.color);
    main.appendChild(h("div", { class: "page-header" },
      h("div", { class: "kicker" }, `${cat.icon} ${cat.label}${spec.group ? " · " + spec.group : ""}`),
      h("div", { class: "page-title" }, `${spec.icon} ${spec.title}`),
      h("p", { class: "page-sub", html: spec.subtitle || "" })));
    for (const n of spec.notes || []) main.appendChild(h("div", { class: "note" + (n.type === "warn" ? " warn" : ""), html: n.html }));
    const panel = h("aside", { class: "controls" });
    const stage = h("section", { class: "stage" });
    main.appendChild(h("div", { class: "sim-layout" }, panel, stage));
    const errBox = h("div", { id: "sim-error", class: "note warn", style: "display:none" });
    stage.appendChild(errBox);

    const api = {
      params: {}, time: 0, stage,
      invalidate() { if (current) current.dirty = true; },
      play() { if (current) { current.playing = true; syncToolbar(); } },
      pause() { if (current) { current.playing = false; syncToolbar(); } },
      get isPlaying() { return !!(current && current.playing); },
      setTime(text) { const t = $("#tb-time"); if (t) t.textContent = text; },
      status(html) { let s = $("#sim-status"); if (!s) { s = h("div", { id: "sim-status", class: "status" }); stage.appendChild(s); } s.innerHTML = html; renderMath(s); },
      metrics(defs) {
        const box = h("div", { class: "metrics" });
        const vals = {};
        defs.forEach((d) => { vals[d.id] = h("div", { class: "m-value" }, d.value || "—"); box.appendChild(h("div", { class: "metric" }, h("div", { class: "m-label", html: d.label }), vals[d.id])); });
        stage.insertBefore(box, plotsBox);
        renderMath(box);
        return { set(id, text) { if (vals[id] && vals[id].textContent !== text) vals[id].textContent = text; } };
      },
      /** layout: [{id, title, span:1|2, type:'2d'|'3d', ...Plot/View3D options}] */
      plots(layout) {
        const out = {};
        for (const L of layout) {
          const card = h("div", { class: "plot-card" + (L.span === 2 ? " span2" : "") });
          if (L.title) card.appendChild(h("div", { class: "plot-title", html: L.title }));
          plotsBox.appendChild(card);
          const p = L.type === "3d" ? new View3D(card, L) : new Plot(card, L);
          p.onResize = () => api.invalidate();
          if (L.type === "3d") p.onChange = () => api.invalidate();
          current.objects.push(p);
          out[L.id] = p;
        }
        renderMath(plotsBox);
        return out;
      },
      setControl(id, patch) {
        const e = current && current.controlEls[id];
        if (!e) return;
        Object.assign(e.c, patch);
        if (e.input && e.c.type === "slider") {
          if (patch.min !== undefined) e.input.min = patch.min;
          if (patch.max !== undefined) e.input.max = patch.max;
          if (patch.step !== undefined) e.input.step = patch.step;
          if (patch.value !== undefined) e.input.value = patch.value;
          api.params[id] = parseFloat(e.input.value);
          if (e.valEl) e.valEl.textContent = fmtVal(e.c, api.params[id]);
        } else if (e.input && patch.value !== undefined) {
          if (e.c.type === "checkbox") e.input.checked = !!patch.value; else e.input.value = patch.value;
          api.params[id] = patch.value;
        }
        if (patch.disabled !== undefined && e.input) e.input.disabled = patch.disabled;
        if (patch.label !== undefined) { const lab = e.row.querySelector(".ctl-label span, .ctl-check span"); if (lab) { lab.innerHTML = patch.label; renderMath(lab); } }
      },
    };

    // toolbar
    if (spec.animated) {
      const sp = Object.assign({ min: 0.1, max: 4, value: 1, step: 0.1 }, spec.speed || {});
      const playBtn = h("button", { class: "btn primary", id: "tb-play", onclick: () => { current.playing = !current.playing; syncToolbar(); } }, "⏸ Pause");
      const resetBtn = h("button", { class: "btn", onclick: () => resetSim() }, "⏮ Restart");
      const stepBtn = h("button", { class: "btn", title: "Single step", onclick: () => { if (current && current.inst.step) { current.playing = false; syncToolbar(); current.inst.step(1 / 60 * current.speed); current.dirty = true; } } }, "⏭ Step");
      const spVal = h("span", {}, "×" + sp.value.toFixed(1));
      const spIn = h("input", { type: "range", min: sp.min, max: sp.max, step: sp.step, value: sp.value });
      spIn.addEventListener("input", () => { current.speed = parseFloat(spIn.value); spVal.textContent = "×" + current.speed.toFixed(1); });
      stage.appendChild(h("div", { class: "toolbar" }, playBtn, resetBtn, stepBtn, h("div", { class: "spacer" }),
        h("div", { class: "speed" }, "Speed", spIn, spVal), h("div", { class: "time", id: "tb-time" }, "")));
    }
    const plotsBox = h("div", { class: "plots" });
    stage.appendChild(plotsBox);

    current = { spec, api, inst: {}, playing: !!spec.animated && spec.autoplay !== false, speed: (spec.speed && spec.speed.value) || 1, dirty: true, objects: [], controlEls: {} };
    current.controlEls = buildControls(spec, api, panel, preset);
    applyVisibility();

    if (spec.theory) {
      const theory = typeof spec.theory === "function" ? spec.theory(api.params) : spec.theory;
      const th = h("details", { class: "theory", open: "open" },
        h("summary", {}, "📚 The Physics — what this simulation solves"), h("div", { class: "body", html: theory }));
      main.appendChild(th);
    }
    renderMath(main);

    try {
      current.inst = spec.mount(api) || {};
      if (current.inst.reset) current.inst.reset();
    } catch (e) { showError(e); }
    syncToolbar();
    markActive(spec.id);
    document.title = spec.title + " · Physics Simulation Lab";
    if (!preset) main.scrollTop = 0;
  }
  function syncToolbar() {
    const b = $("#tb-play");
    if (b && current) { b.textContent = current.playing ? "⏸ Pause" : "▶ Play"; b.classList.toggle("primary", !current.playing); }
  }
  function unmount() {
    if (!current) return;
    try { if (current.inst.destroy) current.inst.destroy(); } catch (e) { console.error(e); }
    current.objects.forEach((o) => o.destroy && o.destroy());
    current = null;
  }

  // ------------------------------------------------------------------ main loop
  let last = performance.now(), fpsCount = 0, fpsT = last;
  function loop(ts) {
    const dt = Math.min((ts - last) / 1000, 0.05);
    last = ts;
    if (current) {
      try {
        if (current.playing && current.inst.step) { current.inst.step(dt * current.speed); current.dirty = true; }
        if (current.dirty && current.inst.render) { current.dirty = false; current.inst.render(); }
      } catch (e) { current.playing = false; syncToolbar(); showError(e); }
    }
    fpsCount++;
    if (ts - fpsT >= 1000) { App.fps = (fpsCount * 1000) / (ts - fpsT); fpsCount = 0; fpsT = ts; }
    requestAnimationFrame(loop);
  }

  function route() {
    const id = decodeURIComponent(location.hash.replace(/^#\/?/, ""));
    const spec = registry.find((s) => s.id === id);
    if (spec) showSim(spec); else showHome();
  }

  const App = {
    fps: 0,
    CATEGORIES,
    register(spec) { registry.push(spec); },
    get registry() { return registry; },
    get current() { return current; },
    start() {
      buildSidebar();
      window.addEventListener("hashchange", route);
      route();
      requestAnimationFrame(loop);
    },
  };
  window.App = App;
})();
