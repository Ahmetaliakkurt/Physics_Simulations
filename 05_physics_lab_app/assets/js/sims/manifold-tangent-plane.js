/* Tangent plane and tangent vectors of the manifold z = −x² − y² — drag-to-rotate 3D view.
 * The surface is the graph of f over a star-shaped domain with a wavy boundary R(θ) = R0 + A·sin(kθ). */
App.register({
  id: "manifold-tangent-plane",
  category: "classical",
  group: "Mathematical Methods",
  order: 40,
  title: "Manifold and Tangent Plane",
  icon: "📐",
  subtitle: "The downward paraboloid $z=-x^2-y^2$ drawn as a 2D manifold in 3D space, with the tangent plane $T_PM$ at a movable point $P$ and the two tangent vectors (the columns of the Jacobian) that span it.",
  notes: [{ type: "info", html: "Move $P$ with the $x_0$, $y_0$ sliders and watch the tangent plane tilt: its slope is the gradient $\\nabla f=(-2x_0,-2y_0)$. Drag the 3D view to rotate it, use the mouse wheel to zoom." }],
  animated: false,
  controls: [
    { type: "section", label: "Point P" },
    { id: "x0", type: "slider", label: "$x_0$", min: -1.5, max: 1.5, step: 0.05, value: 0.5, live: true },
    { id: "y0", type: "slider", label: "$y_0$", min: -1.5, max: 1.5, step: 0.05, value: 0.3, live: true },
    { type: "section", label: "Surface boundary" },
    { id: "R0", type: "slider", label: "Mean boundary radius $R_0$", min: 1.5, max: 4, step: 0.1, value: 3, live: true },
    { id: "A", type: "slider", label: "Boundary ripple amplitude $A$", min: 0, max: 0.8, step: 0.05, value: 0.3, live: true },
    { id: "k", type: "slider", label: "Number of boundary lobes $k$", min: 1, max: 8, step: 1, value: 4, live: true },
    { type: "section", label: "Display" },
    { id: "plane", type: "checkbox", label: "Show tangent plane", value: true, live: true },
    { id: "vecs", type: "checkbox", label: "Show tangent vectors $S_1, S_2$", value: true, live: true },
    { id: "grid", type: "checkbox", label: "Show surface mesh", value: false, live: true },
    { id: "rot", type: "checkbox", label: "Slow auto-rotation", value: true, live: true },
    { type: "info", html: "Drag the view with the mouse to rotate it, use the wheel to zoom." },
  ],
  theory: `
    <h4>The physical system</h4>
    <p>The object studied is a smooth two-dimensional surface embedded in ordinary Euclidean space $\\mathbb R^3$: the graph of the
    concave quadratic function</p>
    $$M=\\{(x,y,z)\\in\\mathbb R^3 : z=f(x,y)=-x^2-y^2\\},$$
    <p>a paraboloid opening downwards with its apex at the origin. Such a graph is the simplest example of a <b>manifold</b>: near every
    point it looks like a piece of the plane $\\mathbb R^2$, and the coordinates $(x,y)$ themselves serve as one global chart.
    Only a bounded patch is drawn, over a star-shaped domain whose boundary in polar coordinates is</p>
    $$x=\\rho\\,R(\\theta)\\cos\\theta,\\quad y=\\rho\\,R(\\theta)\\sin\\theta,\\qquad R(\\theta)=R_0+A\\sin(k\\theta),\\qquad \\rho\\in[0,1],\\ \\theta\\in[0,2\\pi].$$
    <p>The parameters are dimensionless lengths: $R_0$ is the mean boundary radius, $A$ the ripple amplitude and $k$ the number of
    lobes of the boundary curve (which is the bright closed curve on the rim of the surface). The point $P=(x_0,y_0,f(x_0,y_0))$ is
    chosen with the sliders. Nothing is time-dependent; this is pure geometry, but it is exactly the geometry behind configuration
    spaces in Lagrangian mechanics, constraint surfaces, and the linearisation of any nonlinear map.</p>

    <h4>Equations being solved</h4>
    <p>Write the surface as the image of the parametrisation (embedding) $\\Phi:\\mathbb R^2\\to\\mathbb R^3$,
    $\\Phi(x,y)=(x,\\,y,\\,f(x,y))$. Its <b>differential</b> at $P$ is the linear map given by the $3\\times2$ Jacobian matrix</p>
    $$D\\Phi\\big|_P=\\begin{pmatrix}1&0\\\\0&1\\\\ f_x & f_y\\end{pmatrix}_P,\\qquad f_x=\\frac{\\partial f}{\\partial x}=-2x_0,\\quad f_y=\\frac{\\partial f}{\\partial y}=-2y_0 .$$
    <p>Its two columns are the velocity vectors of the coordinate curves $x\\mapsto\\Phi(x,y_0)$ and $y\\mapsto\\Phi(x_0,y)$ through $P$:</p>
    $$S_1=\\frac{\\partial\\Phi}{\\partial x}=(1,\\,0,\\,f_x),\\qquad S_2=\\frac{\\partial\\Phi}{\\partial y}=(0,\\,1,\\,f_y).$$
    <p>The <b>tangent space</b> is the image of the differential, $T_PM=D\\Phi|_P(\\mathbb R^2)=\\operatorname{span}\\{S_1,S_2\\}$: every
    curve on $M$ through $P$ has a velocity that is a linear combination $aS_1+bS_2$. Attached at $P$, it becomes the tangent plane,
    which is nothing but the first-order Taylor polynomial of $f$:</p>
    <div class="callout">$$z_{T}(x,y)=f(x_0,y_0)+f_x\\,(x-x_0)+f_y\\,(y-y_0),\\qquad \\mathbf n=S_1\\times S_2=(-f_x,\\,-f_y,\\,1).$$</div>
    <p>The normal $\\mathbf n$ is perpendicular to both tangent vectors, and the tilt of the plane from the horizontal is
    $\\alpha=\\arccos(\\hat{\\mathbf n}\\cdot\\hat{\\mathbf z})=\\arctan|\\nabla f|$. The inner products of the tangent vectors define the
    induced metric (first fundamental form) $g_{ij}=S_i\\cdot S_j=\\delta_{ij}+f_if_j$, with $\\det g=1+|\\nabla f|^2$; its square root
    $|\\mathbf n|=\\sqrt{1+|\\nabla f|^2}$ is the factor by which a small area $dx\\,dy$ of the chart is stretched on the surface.
    Because $f$ is quadratic, the Taylor remainder is exact:</p>
    $$f(x,y)-z_T(x,y)=-\\big[(x-x_0)^2+(y-y_0)^2\\big]\\le 0,$$
    <p>so the whole surface lies below every one of its tangent planes (concavity) and touches it only at $P$.</p>

    <h4>How the simulation solves them</h4>
    <ul>
      <li><b>Surface mesh:</b> a polar grid of $72$ angles $\\times$ $24$ radial levels $\\rho_j=j/23$ is mapped through the boundary
      formula above, and $z=f(x,y)$ is evaluated exactly at each vertex. The quads are drawn with the painter's algorithm
      (depth-sorted) and simple Lambert-type shading; colour encodes the height $z$ (magma colour map).</li>
      <li><b>Tangent plane:</b> a $10\\times10$ patch covering $[x_0-1.5,x_0+1.5]\\times[y_0-1.5,y_0+1.5]$ on which $z_T$ is evaluated.
      Since the surface is entirely below the plane, the draw order is chosen from the side the camera is on: the sign of
      $\\mathbf n\\cdot\\hat{\\mathbf c}$ ($\\hat{\\mathbf c}$ = unit vector towards the camera) decides whether the plane or the surface
      is painted last.</li>
      <li><b>Vertical scale:</b> $z$ is multiplied by $s=1.35/(R_0+A)$ and shifted so that the patch fits on screen. A map
      $(x,y,z)\\mapsto(x,y,sz+c)$ is affine, so it sends tangent planes to tangent planes; the tangent vectors are drawn as
      $(1.2,0,1.2\\,s f_x)$ and $(0,1.2,1.2\\,s f_y)$. The scale factor is printed in the bottom-right corner.</li>
      <li><b>Metrics:</b> $z_P=-x_0^2-y_0^2$; $\\nabla f=(f_x,f_y)=(-2x_0,-2y_0)$; $|\\nabla f|=2\\sqrt{x_0^2+y_0^2}$; tilt
      $\\alpha=\\arctan|\\nabla f|$ (shown in degrees, true unscaled geometry); and the area factor $\\sqrt{\\det g}=\\sqrt{1+|\\nabla f|^2}$.</li>
      <li>The page is static except for the optional auto-rotation, which only changes the camera yaw (0.25 rad/s).</li>
    </ul>

    <h4>What to try</h4>
    <ol>
      <li>Set $x_0=y_0=0$: $P$ is the apex, $\\nabla f=0$, the plane is horizontal, $\\alpha=0^\\circ$ and the area factor is exactly 1.</li>
      <li>Set $y_0=0$ and vary $x_0$: $S_2=(0,1,0)$ stays horizontal while $S_1$ tilts. At $x_0=0.5$ one has $|\\nabla f|=1$ and
      $\\alpha=45^\\circ$; at $x_0=1.5$, $\\alpha=\\arctan 3\\approx71.6^\\circ$.</li>
      <li>Move $P$ around a circle $x_0^2+y_0^2=r^2$: $z_P$, $|\\nabla f|=2r$ and $\\alpha$ stay constant (rotational symmetry), only the
      direction of the gradient turns.</li>
      <li>Rotate the view until you look along the plane edge-on: the paraboloid touches the plane at $P$ only and curves away
      quadratically on every side, illustrating the exact remainder $-|\\Delta\\mathbf r|^2$.</li>
      <li>Change $R_0$, $A$ and $k$ (e.g. $A=0.8$, $k=6$): the boundary changes but the tangent plane at $P$ does not — the tangent
      space is a <em>local</em> object determined by derivatives at $P$ alone.</li>
    </ol>

    <h4>Limitations &amp; further reading</h4>
    <p>Only one surface, described by a single global chart, is shown; general manifolds need several overlapping charts, and the
    tangent space can also be defined intrinsically (via derivations or equivalence classes of curves) without any embedding.
    Curvature (second fundamental form, Gaussian curvature $K=4/(1+4r^2)^2$ for this surface) is not visualised.
    References: M. P. do Carmo, <i>Differential Geometry of Curves and Surfaces</i>; M. Spivak, <i>Calculus on Manifolds</i>;
    J. M. Lee, <i>Introduction to Smooth Manifolds</i>; M. L. Boas, <i>Mathematical Methods in the Physical Sciences</i>.</p>`,

  mount(api) {
    const P = api.params;
    const NT = 72, NR = 24;   // angular × radial resolution
    const NP = 10;            // tangent-plane mesh
    const f = (x, y) => -x * x - y * y;

    const M = api.metrics([
      { id: "z", label: "$z_P = f(x_0,y_0)$" },
      { id: "grad", label: "$\\nabla f=(f_x,\\,f_y)$" },
      { id: "g", label: "$|\\nabla f|$" },
      { id: "ang", label: "Plane tilt $\\alpha=\\arctan|\\nabla f|$" },
      { id: "area", label: "Area factor $\\sqrt{1+|\\nabla f|^2}$" },
    ]);
    const plots = api.plots([
      { id: "v", type: "3d", title: "Manifold $M: z=-x^2-y^2$ and tangent plane $T_PM$", span: 2, aspect: 0.6, extent: 3.4, yaw: 0.96, pitch: 0.42, minHeight: 360, maxHeight: 640 },
    ]);
    const V = plots.v;

    const X = new Float64Array(NT * NR), Y = new Float64Array(NT * NR), Z = new Float64Array(NT * NR), C = new Float64Array(NT * NR);
    const TX = new Float64Array(NP * NP), TY = new Float64Array(NP * NP), TZ = new Float64Array(NP * NP), TC = new Float64Array(NP * NP);
    const bx = new Float64Array(NT), by = new Float64Array(NT), bz = new Float64Array(NT), bf = new Float64Array(NT);
    let zs = 1, zoff = 0, dirty = true;

    function build() {
      const rmax = P.R0 + P.A;
      zs = 1.35 / rmax;              // visual z scale
      zoff = 0.5 * rmax * rmax * zs; // centre vertically
      V.opts.extent = 1.12 * rmax;   // fit the view to the boundary
      for (let j = 0; j < NR; j++) {
        const rn = j / (NR - 1);
        for (let i = 0; i < NT; i++) {
          const th = (2 * Math.PI * i) / (NT - 1);
          const R = rn * (P.R0 + P.A * Math.sin(P.k * th));
          const x = R * Math.cos(th), y = R * Math.sin(th), z = f(x, y);
          const q = j * NT + i;
          X[q] = x; Y[q] = y; Z[q] = z * zs + zoff; C[q] = z;
        }
      }
      for (let i = 0; i < NT; i++) {
        const q = (NR - 1) * NT + i;
        bx[i] = X[q]; by[i] = Y[q]; bz[i] = Z[q];
        bf[i] = -rmax * rmax * zs + zoff - 0.05; // projection onto the floor
      }
      // tangent plane
      const x0 = P.x0, y0 = P.y0, z0 = f(x0, y0), fx = -2 * x0, fy = -2 * y0;
      for (let j = 0; j < NP; j++) for (let i = 0; i < NP; i++) {
        const x = x0 - 1.5 + (3 * i) / (NP - 1), y = y0 - 1.5 + (3 * j) / (NP - 1);
        const q = j * NP + i;
        TX[q] = x; TY[q] = y; TZ[q] = (z0 + fx * (x - x0) + fy * (y - y0)) * zs + zoff; TC[q] = 0.72;
      }
      dirty = false;
    }

    // local rAF loop for the auto-rotation (the page itself is not animated)
    let raf = 0, lastTs = 0;
    function tick(ts) {
      raf = requestAnimationFrame(tick);
      const dt = lastTs ? Math.min((ts - lastTs) / 1000, 0.05) : 0;
      lastTs = ts;
      if (P.rot) { V.yaw += 0.25 * dt; api.invalidate(); }
    }
    raf = requestAnimationFrame(tick);

    function drawPlane() {
      V.surface(TX, TY, TZ, NP, NP, { values: TC, vmin: 0, vmax: 1, cmap: "viridis", alpha: 0.5, wire: "rgba(126,231,135,0.35)" });
      // plane outline
      const e = [[0, 0], [NP - 1, 0], [NP - 1, NP - 1], [0, NP - 1]].map(([i, j]) => j * NP + i);
      V.line3(e.map((q) => TX[q]), e.map((q) => TY[q]), e.map((q) => TZ[q]), { color: "#7ee787", width: 1.3, alpha: 0.8, close: true });
    }

    return {
      reset() { dirty = true; },
      onParam() { dirty = true; },
      destroy() { cancelAnimationFrame(raf); },
      render() {
        if (dirty) build();
        const x0 = P.x0, y0 = P.y0, z0 = f(x0, y0), fx = -2 * x0, fy = -2 * y0;
        const pz = z0 * zs + zoff;
        V.clear();
        // floor: boundary projection and axes
        V.line3(bx, by, bf, { color: "rgba(139,152,168,0.35)", width: 1, dash: [4, 4] });
        const L = P.R0 + P.A + 0.4, zb = bf[0];
        V.line3([-L, L], [0, 0], [zb, zb], { color: "rgba(247,129,102,0.45)", width: 1 });
        V.line3([0, 0], [-L, L], [zb, zb], { color: "rgba(126,231,135,0.45)", width: 1 });
        const ex = V.project(L, 0, zb), ey = V.project(0, L, zb);
        V.text("x", ex[0] + 4, ex[1], { color: "#f78166" });
        V.text("y", ey[0] + 4, ey[1], { color: "#7ee787" });

        // Unit vector towards the camera; plane normal n = (−fx, −fy, 1) (in scaled space (−fx·zs, −fy·zs, 1)).
        const cy = Math.cos(V.yaw), sy = Math.sin(V.yaw), cp = Math.cos(V.pitch), sp = Math.sin(V.pitch);
        const cam = [sy * cp, cy * cp, sp];
        const above = -fx * zs * cam[0] - fy * zs * cam[1] + cam[2] > 0;
        // The concave surface lies entirely below the plane: if the camera is above the plane, draw the surface first.
        if (P.plane && !above) drawPlane();
        V.surface(X, Y, Z, NT, NR, { values: C, cmap: "magma", alpha: 0.93, wire: P.grid ? "rgba(15,21,28,0.45)" : null });
        V.line3(bx, by, bz, { color: "#fcfdbf", width: 1.4, alpha: 0.7 });
        if (P.plane && above) drawPlane();

        if (P.vecs) {
          const s = 1.2;
          V.arrow3(x0, y0, pz, x0 + s, y0, pz + s * fx * zs, { color: PlotColors.blue, width: 3 });
          V.arrow3(x0, y0, pz, x0, y0 + s, pz + s * fy * zs, { color: PlotColors.accent3, width: 3 });
          const a = V.project(x0 + s, y0, pz + s * fx * zs), b = V.project(x0, y0 + s, pz + s * fy * zs);
          V.text("S₁", a[0] + 6, a[1] - 4, { color: PlotColors.blue, size: 13 });
          V.text("S₂", b[0] + 6, b[1] - 4, { color: PlotColors.accent3, size: 13 });
        }
        V.point3(x0, y0, pz, { color: PlotColors.white, size: 5.5, label: "P" });

        // legend
        const c = V.ctx;
        const items = [["Manifold M", "#e5507a"], ["Tangent plane TₚM", "#35b779"], ["S₁ = (1, 0, ∂f/∂x)", PlotColors.blue], ["S₂ = (0, 1, ∂f/∂y)", PlotColors.accent3]];
        c.font = '12px "Segoe UI", system-ui, sans-serif';
        c.fillStyle = "rgba(15,21,28,0.85)"; c.fillRect(10, 10, 176, items.length * 18 + 10);
        items.forEach(([t, col], i) => {
          c.fillStyle = col; c.fillRect(20, 20 + i * 18, 12, 8);
          c.fillStyle = PlotColors.text; c.textAlign = "left"; c.textBaseline = "middle"; c.fillText(t, 40, 24 + i * 18);
        });
        c.fillStyle = PlotColors.muted; c.textAlign = "right"; c.textBaseline = "alphabetic";
        c.fillText(`z axis scaled ×${PM.fmt(zs, 2)}`, V.W - 10, V.H - 10);
        c.textAlign = "left";

        const g = Math.hypot(fx, fy);
        M.set("z", PM.fmt(z0, 3));
        M.set("grad", `(${PM.fmt(fx, 2)}, ${PM.fmt(fy, 2)})`);
        M.set("g", PM.fmt(g, 3));
        M.set("ang", PM.fmt((Math.atan(g) * 180) / Math.PI, 1) + "°");
        M.set("area", PM.fmt(Math.sqrt(1 + g * g), 3));
      },
    };
  },
});
