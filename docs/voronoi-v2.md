# Voronoi v2 — the irregular-cell architecture

Replaces the fixed Goldberg hex mesh entirely. Direction set by review: the
planet is tessellated by true spherical Voronoi polygons — soap bubbles in a
dish — and the cell becomes a richer primitive than a sample point.

## 1. The tessellation

- **Generators**: points on the sphere drawn from a **density field** (below),
  with spacing proportional to local target cell size.
- **Cell size is dictated by the terrain, full stop.** The size distribution is
  deliberately wild — mountain-belt cells and abyssal-plain cells can differ by
  two orders of magnitude in area. This must never be homogenized: vanilla
  Lloyd relaxation equalizes cell sizes and is therefore FORBIDDEN; if any
  relaxation is used it must be density-weighted (each cell relaxes toward the
  centroid of the density mass it covers, preserving the contrast), or skipped
  entirely.
- **Tessellation**: spherical Delaunay via 3D convex hull of the generator
  points; the Voronoi diagram is its dual. Cells have any number of vertices
  (typically 4–9) and any size. No two alike, no lattice signature at any zoom.
- **Contract**: the existing `Mesh` API survives — cell centers, CSR neighbors,
  CSR corner lists (CCW), chunks, `cell_at` — it was built degree-agnostic, so
  the process crates and renderer keep consuming it. What changes is who makes
  it and what it carries.

## 2. Adaptive density = fidelity allocation

The **total cell budget** is the user's fidelity knob (config; replaces the
subdivision level). A **density field** decides where those cells go:

- high density: steep relief (|elevation gradient|), active plate boundaries,
  coastlines and shelves;
- low density: open plains, abyssal interior.

Mountain ranges are therefore *made of* many small clustered cells — the
dramatic ridge-and-valley geometry falls out of the tessellation — while a
plain of equal area might be a handful of big polygons.

Because mountains and coastlines EMERGE during the run, density must follow
them: the planet **re-tessellates periodically** (phase boundaries + every
~25 Myr of drift): compute the new density field from the current state, draw
new generators (deterministic from seed + epoch), tessellate, and resample
every field from the old mesh (per-cell: area-weighted from overlapped old
cells; columns: from the dominant contributor — mass-ledgered). Re-tessellation
is a pure function of (state, seed, epoch): checkpoints and determinism
survive. Cadence: at every phase boundary, plus fixed ~25 Myr epochs during
Drift/Refinement — terrain moves slowly enough between epochs that density
stays honest, and the amortized cost is a handful of re-tessellations per run.

## 3. Per-vertex elevation

Cells carry elevation **at each vertex** as well as the cell mean. Simulation
fields (crust, climate, strata) stay per-cell; vertex elevation is derived
(area-weighted blend of adjacent cells' isostatic elevation + coherent detail
noise) and is what the renderer displaces and the coastline threads through —
shorelines and ridgelines follow the irregular vertex geometry, not cell
membership steps.

## 4. River cells

Hydrology becomes an **edge-routed graph** on the tessellation instead of a
per-cell scalar:

- A cell's river state: `inflow edges`, `outflow edge` (or `spring` /
  `mouth` / `none`), and discharge, which grows downstream as rainfall
  accumulates.
- Routing: steepest descent on vertex/edge elevations; depression filling as
  today. A river is then a polyline threaded edge-to-edge through cells —
  renderable as a real winding channel, and the stream-power erosion carves
  the cells it crosses (which raises their density priority at the next
  re-tessellation: incised valleys refine themselves).
- Mouths hand deltas to the coast cell's shared edge; the existing facies
  logic keeps working per-cell.

## 5. What this buys

- Coastlines: irregular polygon outlines are the canvas the wiggly coasts sit
  on — no hex staircase at any resolution.
- Continental shelves / fragments: a landmass is a set of cells with organic
  outline; splitting Pangaea along a rift = cutting the cell graph along
  edges, which look like torn paper, not plotted curves.
- Fidelity where it matters: the same 500k-cell budget renders Himalayan
  drama AND planet-scale oceans, because the cells go where the relief is.

## 6. Cost & sequencing (estimate)

1. `iw-mesh` v2: density-driven generators, convex-hull Delaunay, Voronoi
   dual, chunking, `cell_at` — the hard core (hull at millions of points).
2. `iw-core`: vertex-elevation storage, river edge state, budget config.
3. Resampling machinery + re-tessellation driver in `iw-sim`/tectonics.
4. Hydrology rework to edge routing; renderer vertex-elevation path.
5. Pangaea rework (in flight, currently parked) rebased onto v2 mesh.
6. Tests migrated off hexagon assumptions; visual acceptance loop.

Phases land in that order, each keeping the workspace green.
