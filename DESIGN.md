# Infinite Worlds — High-Level Design

A native Rust application that generates Earth-like planets by simulating their geological
history from crustal formation to the present day, and renders the result in a Vulkan window
as an interactive 3D globe or a Mercator projection.

Status: **draft for review**.

---

## 1. Goals

- Model the planet as a true sphere tiled with polygons of roughly equal area — no quad
  spheres, no cube maps, no lat/long grids with polar singularities.
- Simulate the processes that shaped Earth's surface, in causal order: craton formation,
  supercontinent assembly, rifting, continental drift, subduction, orogeny, volcanism,
  ocean formation, then (in the geologically recent past) glaciation, rivers, wind, and
  rain sculpting the final terrain.
- Mountains and trenches are **emergent**, never painted: Himalaya-scale ranges arise from
  continent–continent collision and crustal thickening; Mariana-scale trenches arise from
  subduction. Target relief spread comparable to Earth (~20 km between highest peak and
  deepest trench).
- Full stratigraphic record: every cell carries a column of rock layers (igneous,
  sedimentary, metamorphic; granite, basalt, shale, limestone, marble, ...) built up and
  destroyed by the simulation itself.
- Classify the finished surface into the 14 WWF terrestrial biomes.
- Pure Rust. Vulkan rendering. Wayland-native on Linux when a Wayland session is
  present, automatic X11 fallback otherwise — no env-var workarounds either way. Ports
  cleanly to macOS and Windows.

## 2. Fidelity philosophy (read this first)

"Full realistic simulation" needs one honest caveat: nobody can numerically solve mantle
convection and 500 Myr of erosion at planet scale on a desktop. What we *can* do — and what
this design commits to — is **phenomenological models with correct physics at their core**:

- Plate motion driven by the real force balance (slab pull dominant, ridge push secondary),
  not random walks.
- Elevation from **isostasy** (crust floats on mantle; thick/light crust rides high,
  thin/dense crust rides low). This single mechanism is why continents exist, why
  collision makes Himalayas, and why old cold oceanic plates make trenches. It is the
  backbone of the whole simulation.
- Erosion from the **stream power law** and hillslope diffusion — the same equations used
  in academic landscape-evolution models (e.g. Fastscape).

Every process is a module with real units (meters, years, Pascals, kg/m³) and parameters
taken from published Earth values, tunable in config. The result is "geologically
plausible and internally consistent," which is the strongest claim any planet generator
can honestly make. Where we cheat, the doc says so explicitly.

The acceptance bar (agreed): **realistic-looking output**, not scientific accuracy.
Where a cheaper approximation produces equally convincing terrain, the cheaper
approximation wins.

## 3. Architecture

Hexagonal (ports & adapters), enforced by the crate graph. The domain — everything that
knows what a planet *is* and how it evolves — is pure Rust with zero dependencies on
Vulkan, winit, the filesystem, or any OS API. All I/O lives in adapter crates that plug
into ports (traits) defined at the domain boundary. The compiler enforces the direction:
adapters depend on the domain; the domain depends on nothing but math.

```
                 driving side                        driven side
┌──────────────┐        ┌────────────────────────┐        ┌───────────────────┐
│ iw-app       │        │      DOMAIN CORE       │        │ iw-render-vulkan  │
│ (winit loop, │─ port ─▶  iw-mesh  iw-tectonics │─ port ─▶ (ash, egui)       │
│  egui input) │        │  iw-geology iw-climate │        ├───────────────────┤
├──────────────┤        │  iw-surface iw-biomes  │─ port ─▶ iw-store-postcard │
│ iw-headless  │─ port ─▶  iw-sim (orchestrator) │        ├───────────────────┤
│ (CLI, CI)    │        │                        │─ port ─▶ iw-export-png     │
└──────────────┘        └────────────────────────┘        └───────────────────┘
```

**Ports** (traits, defined in the domain / application layer):

- `SimulationControl` — driving port: start, pause, single-step, run-to-phase-end,
  re-run phase with new params. Exercised by the GUI app and equally by the headless CLI.
- `PlanetView` — driven port: read-only, versioned snapshots of planet state handed to
  observers at safe points (double-buffered so the renderer never sees a half-stepped
  planet, and the sim never blocks on the renderer).
- `CheckpointStore` — driven port: save/load/list checkpoints. Production adapter is
  postcard+zstd on disk (§12); tests use an in-memory adapter.
- `MapExporter` — driven port: rasterized map output (PNG heightmaps, biome maps).
- `ProgressSink` — driven port: phase/step progress events for UI progress bars and CI
  logs.

**Adapters**: `iw-render-vulkan` (ash + egui, §9), `iw-app` (winit/Wayland shell, §11),
`iw-headless` (batch generation + golden tests in CI, no window), `iw-store-postcard`,
`iw-export-png`. Swapping any of these — e.g. a future wgpu renderer, or glTF export —
touches no domain code.

Domain-internal boundaries follow the same discipline: `iw-tectonics`, `iw-geology`,
`iw-climate`, `iw-surface`, and `iw-biomes` each expose a `Process` interface to `iw-sim`
(read planet state + timestep → field updates) and do not call each other directly;
cross-process coupling (e.g. erosion needs precipitation) flows through the shared,
explicitly-typed planet state, so each process is testable in isolation against a
synthetic planet.

Data flow is a pipeline over that shared `Planet` state:

```
seed ─▶ mesh ─▶ [Phase 1: crust] ─▶ [Phase 2: drift] ─▶ [Phase 3: refinement]
                     ─▶ [Phase 4: recent past] ─▶ [climate + biomes] ─▶ observers
        each phase boundary = checkpoint via CheckpointStore
```

Checkpoints matter: re-tuning sea level or re-running glaciation must not require
re-simulating 400 Myr of tectonics. Any phase can be re-run from the previous checkpoint
with new parameters.

Workspace layout:

```
infinite-worlds/
├── crates/
│   ├── iw-mesh            # domain: Goldberg sphere, adjacency, spherical geometry
│   ├── iw-sim             # domain: phase orchestration, ports, deterministic RNG
│   ├── iw-tectonics       # domain process: plates, subduction, orogeny, volcanism
│   ├── iw-geology         # domain process: strata, rock types, metamorphism, isostasy
│   ├── iw-climate         # domain process: temperature, winds, precipitation, sea level
│   ├── iw-surface         # domain process: rivers, glaciers, erosion, sediment
│   ├── iw-biomes          # domain process: biome classification
│   ├── iw-render-vulkan   # adapter: ash renderer, globe/Mercator views, egui UI
│   ├── iw-store-postcard  # adapter: checkpoint persistence
│   ├── iw-export-png      # adapter: map export
│   ├── iw-headless        # adapter/binary: batch generation, CI, golden tests
│   └── iw-app             # adapter/binary: winit event loop, wiring
└── DESIGN.md
```

## 4. Planet mesh

**Goldberg polyhedron** — the dual of a subdivided icosahedron. Exactly 12 pentagonal
cells, the rest hexagons, all within a few percent of equal area, no singularities.

- Subdivision level is a user-facing tunable (§9.1), range 6–10. **Default: level 8,
  ~655k cells, ~26 km cell pitch** at Earth radius while prototyping — expected to move
  to 9 once the pipeline is profiled. Levels 9 (~2.6M cells, ~13 km) and 10 (~10.5M
  cells, ~6.5 km) supported for maximum fidelity on machines with the RAM for it (§10);
  level 6 (~41k cells) for fast iteration during development.
- `iw-mesh` provides: cell centers (unit vectors), neighbor lists (CSR adjacency), cell
  areas, edge lengths, great-circle math, and a lat/long → cell lookup (cube-projection
  spatial index).
- All per-cell simulation state lives in flat `Vec`s indexed by cell id (struct-of-arrays)
  — cache-friendly and trivially parallelizable with rayon.
- One mesh serves both simulation and rendering. If relief still looks too coarse up
  close, a later **amplification pass** can add sub-cell detail with noise conditioned on
  the local geology (mountain roughness ≠ plains roughness). Explicitly deferred; not
  needed for v1.

## 5. Simulation timeline

Simulated time is divided into four phases with different timesteps. Durations are config
values; defaults below match your brief.

| Phase | Sim duration | Timestep | What runs |
|---|---|---|---|
| 1. Crustal formation | ~200 Myr | 1 Myr | craton seeding, accretion, supercontinent assembly |
| 2. Continental drift | ~200 Myr | 0.5 Myr | full tectonics, ocean fill, coarse erosion |
| 3. Refinement (to −2 Ma) | ~50–100 Myr | 0.25 Myr | tectonics continues, finer erosion, climate spins up |
| 4. Recent past (−2 Ma → 0) | 2 Myr | 1–10 kyr | glacial cycles, rivers, deltas, wind/rain, fjords; plates still creep |

Everything is driven by a seeded deterministic RNG (`rand_pcg`): same seed + same config →
identical planet, on every platform (this constrains us to deterministic math — no
platform-dependent floating point fast-math in sim crates).

### Phase 1 — Crustal formation (cratons → Rodinia analog)

- Seed 10–20 **craton nuclei** at random-but-spaced locations: small caps of thick
  (~40 km), old, buoyant felsic crust over a global basaltic proto-crust.
- Cratons drift under a simple early-mantle convection proxy (low-order spherical-harmonic
  flow field evolving slowly over the phase) and **accrete**: collisions weld cratons
  together with greenstone-belt-style seams, building continental shields.
- Bias forces late in the phase to sweep continents together into a supercontinent — our
  Rodinia analog — so Phase 2 starts from the historically interesting configuration:
  one landmass, one superocean.

### Phase 2 — Continental drift (the main tectonic engine)

Plates are **rigid spherical caps rotating about Euler poles** (exactly how real plate
motion is described). Each cell belongs to one plate; each plate has an angular velocity
updated each step from a force balance:

- **Slab pull**: plates with subducting edges get pulled toward their trenches — the
  dominant term, as on Earth.
- **Ridge push**: plates slide away from spreading ridges.
- **Basal drag**: resistance term proportional to plate speed.
- Resulting speeds calibrated to Earth's 2–10 cm/yr.

Boundary interactions, resolved per boundary segment each step:

- **Divergent (rift → ridge)**: supercontinent stress triggers rifting along weak sutures
  (the Phase-1 seams — so breakup echoes assembly, as on Earth). Rifts widen into ocean
  basins; new basaltic oceanic crust forms at the ridge, stamped with its formation age.
  Ocean floor **subsides with age** (thermal subsidence, √age law) — this alone produces
  realistic ocean bathymetry: shallow ridges, deep abyssal plains.
- **Convergent, ocean–continent**: dense oceanic plate subducts. Produces: a **trench**
  (flexural bending, deepest where the slab is oldest/densest — Mariana-class depths),
  a **volcanic arc** ~150–300 km behind the trench (Andes-style), andesitic volcanism,
  and accretion of scraped-off oceanic sediment onto the margin.
- **Convergent, ocean–ocean**: older, denser side subducts → island arcs and back-arc
  basins.
- **Convergent, continent–continent**: neither subducts. Crust **thickens** (up to
  ~70 km, Tibet-like); isostasy converts thickening into uplift → Himalaya-scale ranges
  with deep crustal roots. Suture zones close; plates weld.
- **Transform**: lateral slip, fault scars, no creation/destruction.
- **Hotspots**: a handful of fixed mantle plumes; plates drifting over them get volcanic
  chains with age progression (Hawaii-style), and occasional flood-basalt provinces.
- Plates **split** when rifts propagate through them and **merge** on continental
  collision, so plate count and geometry evolve freely.

**Isostasy** runs every step: elevation is computed from crustal thickness and density
(Airy model, with flexural smoothing so loads bend the lithosphere regionally instead of
spiking single cells). Elevation is therefore never edited directly — processes change
thickness, density, and load; elevation follows.

**Ocean fill**: the planet has a total water budget (tunable, in units of Earth oceans).
Each step, sea level is solved from the hypsometric curve so that exactly that volume
fills the lowest terrain. Turning the knob up drowns continents; down exposes shelves.

### Phase 3 — Refinement

Same engine, smaller timestep, plus: coarse fluvial/hillslope erosion coupled in (mountains
shed sediment into foreland basins and passive margins — building the sedimentary record),
and the climate model starts producing per-cell temperature/precipitation used by erosion.

### Phase 4 — Recent past (−2 Ma → now)

Tectonic velocities freeze to their final values (creep only); surface processes take over
at kyr resolution:

- **Glacial cycles**: ~10 cycles driven by a Milankovitch-like temperature oscillation.
  Ice sheets grow where accumulation exceeds melt; ice flows downhill, carving U-valleys,
  cirques, and — where ice streams reach the coast — **fjords**. Retreat leaves moraines,
  outwash plains, and overdeepened basins that fill as **great lakes**.
- **Rivers**: flow routing over the sphere (priority-flood depression filling → flow
  accumulation → discharge). Stream-power incision carves valleys and gorges; sediment
  transported downstream deposits as floodplains and **deltas** at river mouths.
- **Hillslope & wind**: diffusion smooths slopes; aeolian transport moves sand in arid
  cells (dune fields, loess deposition downwind).
- **Coasts**: wave erosion cuts cliffs on exposed coasts, builds barrier deposits on
  sheltered ones.

## 6. Geology: the stratigraphic column

The signature data structure. Every cell owns a **stack of strata**, bottom to top:

```rust
struct Stratum {
    rock: RockType,      // enum, see below
    thickness_m: f32,
    deposited_myr: f32,  // sim-time stamp
    grade: MetamorphicGrade, // none / low / medium / high
}
```

- **Igneous**: basalt & gabbro (oceanic crust, flood basalts), granite & diorite
  (plutons intruded under arcs and collision zones), andesite & rhyolite (volcanic
  surface flows), tuff (explosive eruptions).
- **Sedimentary**, created by the sim's own erosion/deposition: sandstone (coastal,
  fluvial), shale (deep/quiet water mud), limestone (warm shallow seas, biogenic),
  conglomerate (mountain-front fans), evaporites (restricted basins).
- **Metamorphic**, by transformation in place when burial depth / nearby magmatism push a
  stratum past pressure–temperature thresholds: shale → slate → schist → gneiss;
  limestone → marble; sandstone → quartzite; basalt → amphibolite. Grade recorded, so
  a collision belt ends up with the right zonation: high-grade core, low-grade flanks.

Processes read and write the column: deposition pushes strata, erosion pops them (exposing
older rock — canyon walls and shield interiors show their history), intrusions insert
sills, subduction destroys whole columns, collision folds and thickens them. Columns are
capped (~64 strata) with automatic merging of thin like layers to bound memory
(~655k cells × 64 strata ≈ workable; exact budget in §10).

The payoff: click any cell in the viewer and see its full geologic history — e.g. marble
under a mountain range tells you "this was a warm shallow sea, then got buried in a
continental collision."

## 7. Climate

Deliberately lightweight — just enough physics to drive erosion and biomes correctly:

- **Temperature**: latitudinal insolation curve + lapse rate (−6.5 °C/km) + ocean thermal
  moderation for maritime cells + a global offset for glacial cycles.
- **Winds**: prescribed belts (trades, westerlies, polar easterlies) with Coriolis
  deflection — not a GCM.
- **Precipitation**: moisture picked up over ocean, advected downwind, dropped by
  orographic lift (windward wet, leeward **rain shadow**) and convergence zones (ITCZ).
  This produces the patterns that matter: monsoon-ish wet east coasts, dry continental
  interiors, coastal deserts behind mountain walls.
- Seasonality approximated by computing summer/winter endpoints (axial tilt is a config
  knob) rather than a full annual cycle.

## 8. Biomes

The 14 WWF terrestrial biomes, classified per land cell from annual mean temperature,
precipitation, seasonality, and local state (elevation, flooding, coastal adjacency):

tropical & subtropical moist broadleaf forest · tropical & subtropical dry broadleaf
forest · tropical & subtropical coniferous forest · temperate broadleaf & mixed forest ·
temperate coniferous forest · boreal forest/taiga · tropical & subtropical grasslands,
savannas & shrublands · temperate grasslands, savannas & shrublands · flooded grasslands &
savannas · montane grasslands & shrublands · tundra · mediterranean forests, woodlands &
scrub · deserts & xeric shrublands · mangroves.

Classification is a Whittaker-style temperature×precipitation lookup with overrides for
the special cases (flooded: river-adjacent lowland with high water table; montane: high
elevation in low latitude; mangrove: tropical muddy coast; mediterranean: winter-wet /
summer-dry west coasts at ~30–45°).

## 9. Rendering & UI

- **Renderer**: the `iw-render-vulkan` adapter — `ash` (raw Vulkan bindings) + `vk-mem`
  for allocation, Vulkan 1.2 baseline. It consumes `PlanetView` snapshots; it never
  touches simulation internals.
- **Globe view**: the Goldberg mesh rendered as triangles (each cell fanned from its
  center), vertex elevation displaced along the normal with adjustable vertical
  exaggeration (real relief is invisible at true scale: 20 km on 6371 km radius).
  Arcball rotation + **deep zoom**: continuous from full-globe down to near-surface
  altitude (~50 km), close enough to inspect individual features — a fjord system, a
  volcanic arc, a delta.
- **Camera controls** — one camera, behavior blends with altitude:
  - **WASD** always means north / west / south / east. Zoomed out, it spins the globe
    under the camera with a little **momentum** (velocity damps after key release —
    flick and glide). Zoomed in, the same keys become surface travel: the camera tracks
    along great circles over the terrain at its current altitude, speed scaled to
    altitude so travel feels constant.
  - **Free look**: hold **right mouse button** and move the mouse to adjust the camera
    direction/angle — pitch and yaw from the current vantage point without moving it —
    stand off a
    mountain range at 50 km and look along the horizon, down into a trench, back over
    your shoulder. The look direction persists until a recenter key eases the view back
    to straight-down framing; WASD travel continues to work while free-looking.
  - Scroll zooms along the view ray; the orbital ↔ near-surface transition is
    continuous, no mode switch to manage.
  Engineering consequences, planned from the start:
  - Mesh built as **chunked patches** (grouped by icosahedron face region) for frustum
    and horizon culling — when zoomed in, only the visible sliver of a 2.6M-cell planet
    is drawn.
  - **Reverse-Z depth buffer**, so precision holds across the full-globe ↔ near-surface
    zoom range.
  - Zoom-proportional arcball sensitivity (rotation slows as you approach) and vertical
    exaggeration that eases toward 1× when close, so near-surface terrain doesn't look
    cartoonish.
  - Close zoom is the feature most likely to promote the geology-conditioned
    amplification pass (§4) from deferred to needed — at level 8 a cell is ~26 km, which
    reads fine at globe distance and visibly faceted at 50 km altitude. Decision point
    after M3, when there's real terrain to look at.
- **Mercator view**: same data, vertices reprojected in the vertex shader; toggle is
  instant, no separate asset. (Poles clamped at ±85° as usual for Mercator.)
- **Beauty view** (the default layer): satellite-realistic, NASA Blue Marble as the
  reference target (`modis_wonderglobe.jpg`) — deep saturated ocean blue with turquoise
  shallow-water tint on continental shelves, land colored from biome/land-cover
  (vegetation greens, desert tans, bare-rock browns), white polar ice and mountain
  glaciers, subtle specular sun glint on water, and a thin blue **atmospheric rim**
  (limb scattering) against a realistic starfield background. **Clouds**: implemented as
  a toggleable procedural layer, **off by default**. No moon for now.
- **Data layers**, switchable: relief/terrain color, biomes, plates (+ velocity arrows),
  crust age, crust type & thickness, surface rock type, temperature, precipitation, ice
  cover, drainage/rivers.
- **Cell inspector**: click a cell → stratigraphic column rendered as a stacked bar with
  rock names, ages, grades.
- **Time scrubbing**: checkpoints + periodic lightweight snapshots let you replay the
  planet's history with a time slider — watching the supercontinent break up is half the
  fun and an invaluable debugging tool.
- **UI**: `egui` (ash backend) for parameter panels, phase progress, layer selection.
  Simulation runs on worker threads; the UI thread never blocks. **Generation is a live
  spectacle**: the globe renders and updates continuously while phases run — you watch
  cratons collide, oceans open, and ice sheets advance in real time. No mid-run
  interactivity required; the sim streams `PlanetView` snapshots and the camera stays
  free.
- **Progress narration**: the `ProgressSink` events carry Maxis-grade snarky status
  lines, drawn from a per-phase pool and displayed in the progress bar — "Reticulating
  splines", "Subducting the evidence", "Aged basalt: subsiding as designed",
  "Convincing continents to commit", "Percolating plutons", "Applying orogeny liberally",
  "Un-flooding Kansas", "Negotiating with the mantle". The pool lives in a plain text
  asset so adding jokes never touches code.

### 9.1 Tunables

Presented in a "New Planet" panel; sensible defaults in parentheses. All are plumbed as
one serde `PlanetConfig` struct — saved into every checkpoint so a planet remembers how
it was made.

- **Seed** (random) — text field plus a 🎲 dice button: randomizes the seed and
  immediately regenerates with all current settings.
- **Mesh subdivision level** (8 during prototyping, later 9; range 6–10) — with a live
  readout of cell count and estimated RAM.
- **Phase durations** (200 / 200 / 75 / 2 Myr; each bounded to sane ranges) — total sim
  length is their sum; longer drift = more supercontinent cycles.
- **Water budget** (1.0 Earth oceans; 0.1–3.0) — drives sea level via hypsometric fill.
- **Global temperature offset** (0 °C; ±20 °C) — shifts the whole climate; drives
  ice extent, desert extent, biome belts.
- **Axial tilt** (23.4°; 0–45°) — seasonality strength, position of climate belts.
- **Precipitation multiplier** (1.0×; 0.25–4×) — global moisture scaling; arid world ↔
  jungle world.
- **Tectonic vigor** (1.0×; 0.25–2×) — scales plate speeds and volcanism; sleepy world ↔
  violent world.
- **Hotspot count** (8; 0–30).
- **Craton count** (14; 4–30) — few = huge shields; many = fragmented microcontinents.
- **Glacial cycle intensity** (1.0×; 0–2×) — depth of Phase-4 ice ages.
- Render-side (not part of the planet): vertical exaggeration, layer selection.

## 10. Performance

- **CPU-first**: all simulation in Rust on CPU via `rayon`. Tectonics is branchy and
  irregular (bad GPU fit); erosion's flow routing is a global sequential dependency
  (needs care even on CPU). At the 655k-cell prototyping default, a full ~500 Myr
  history is estimated at **minutes**; at level 9 (2.6M) roughly 4× that, tens of
  minutes — the number to defend in profiling before making 9 the default.
- Struct-of-arrays layout throughout; `f32` for fields, `f64` only where accumulation
  error matters (time, mass budgets).
- GPU compute (Vulkan compute shaders, reusing the render context) is the designated
  escape hatch — most likely needed for Phase-4 erosion at kyr steps once runs move to
  2.6M+ cells, the most parallel-friendly stage. Deferred until profiling says so; the
  `Process` interface (§3) is designed so a module can move to GPU without the
  orchestrator noticing.
- Memory: strata stored in a pooled arena with per-cell spans (most cells carry few
  layers; the 64-stratum cap bounds the worst case). Level 9 ≈ 2–4 GB total; level 10
  ≈ 8–16 GB — the New Planet panel shows the estimate next to the subdivision slider so
  nobody discovers this via the OOM killer.
- Mass conservation is asserted in debug builds (rock eroded = rock in transit +
  rock deposited + rock subducted); the classic failure mode of erosion sims is silently
  creating or destroying crust.

## 11. Platform & windowing

- **`winit`** for windowing, compiled on Linux with both `wayland` and `x11` features.
  winit's default backend selection already does the right thing: **Wayland when a
  Wayland session is present, X11 otherwise** — no environment variables, no user
  intervention. Same winit code paths drive macOS (AppKit) and Windows (Win32).
- Surface creation via `ash-window` + `raw-window-handle`: `VK_KHR_wayland_surface` /
  `VK_KHR_xcb_surface` on Linux (matching whichever backend winit picked),
  `VK_KHR_win32_surface` on Windows, and on macOS `VK_EXT_metal_surface` through
  **MoltenVK** (Vulkan-on-Metal; the standard, Khronos-supported route — LunarG SDK ships
  it). We stick to Vulkan core + portability-subset-safe features so MoltenVK is a
  packaging concern, not a code fork.
- Linux is the primary dev platform. No remote CI for now (local-only repo): the
  cross-platform contract is held by keeping platform code confined to the winit/ash
  adapter layer and building the other targets manually at milestones; hosted CI can be
  added when the repo gets a remote.

## 12. Persistence

- Checkpoint format: `postcard` (compact, serde-based) + zstd, versioned header.
  Full sim state — resumable and re-runnable per phase.
- Lightweight history snapshots (elevation, plates, sea level only) for the time
  scrubber, under a **tunable disk cap, 2 GB default**: the snapshot interval
  auto-adjusts to planet size and sim length so history always fits the cap; bigger
  planets scrub more coarsely rather than eating the disk.
- Export: PNG heightmap/biome/rock maps in equirectangular and Mercator; glTF mesh export
  is a possible later addition.

## 13. Testing

- Unit: spherical geometry, isostasy solver (analytic cases), metamorphic thresholds,
  flow routing on synthetic terrains.
- Property (proptest): mass conservation across random step sequences; no NaNs; sea-level
  solver converges for any hypsometry.
- Golden planets: fixed seeds run through `iw-headless` (no window, in-memory store
  adapter) with committed summary statistics (hypsometric curve, % land, plate count,
  biome area fractions) — catches drift from refactors. Ranges chosen loosely enough to
  allow intentional model changes. The hexagonal split is what makes this cheap: the
  entire domain runs headless with no GPU and no display server, so the full suite is
  just `cargo test` locally (and drops straight into hosted CI whenever the repo gets a
  remote).
- Determinism test: same seed twice → bit-identical output; when other platforms are
  built at milestones, Linux vs Windows vs macOS must match too.

## 14. Milestones

Each milestone is demoable; rendering arrives early because seeing the planet is how we
debug everything after it.

1. **M1 — Spinning rock**: workspace, Goldberg mesh, ash renderer on Wayland, arcball
   globe with deep zoom (chunked/culled mesh, reverse-Z) + Mercator toggle, egui shell.
   (The whole platform layer de-risked first.)
2. **M2 — Cratons**: Phase 1 running live in the viewer; plate/crust data layers.
3. **M3 — Drift**: full tectonic engine — rifting, subduction, collision, isostasy,
   ocean fill. The planet gets Himalayas and trenches here. Checkpoints + time scrubber.
4. **M4 — Stone**: stratigraphic columns, rock types, metamorphism, cell inspector.
5. **M5 — Water & ice**: climate model, Phase 4 (rivers, glaciers, fjords, deltas, wind).
6. **M6 — Life zones**: biome classification + biome layer; beauty-view polish (Blue
   Marble shading, atmosphere rim, starfield, optional clouds); export; macOS/Windows
   builds validated.

## 15. Decision log

Resolved in review, 2026-08-13:

1. **Cell count**: level 8 (~655k cells) default while prototyping, moving to level 9
   once profiled; user-tunable 6–10, level 10 supported.
2. **Renderer**: `ash` confirmed (MoltenVK on macOS).
3. **Interactivity**: watch-live generation with continuously updating globe; no mid-run
   parameter nudging. Snarky Maxis-style progress narration is a requirement (§9).
4. **Tunables**: full panel per §9.1 — seed + dice-reroll, subdivision level, phase
   durations, water budget, temperature offset, tilt, precipitation, tectonic vigor,
   hotspots, cratons, glacial intensity.
5. **Water budget**: Earth-like default, exposed knob.
6. **Fidelity**: approximation is acceptable; the bar is realistic-*looking* results,
   not scientific accuracy (§2).
7. **Platform**: Wayland-first with automatic X11 fallback on non-Wayland sessions;
   no env vars either way (§11).
8. **Beauty view**: satellite-realistic, NASA Blue Marble reference; clouds toggleable
   but off by default; realistic starfield; no moon (§9).
9. **History recording**: tunable disk cap, 2 GB default, auto-adjusting snapshot
   interval (§12).
10. **CI**: local-only for now; hosted CI deferred until the repo has a remote (§11,
    §13).
11. **Deep zoom**: camera zooms from full-globe to ~50 km altitude for feature
    inspection; chunked/culled mesh and reverse-Z planned from M1; amplification-pass
    decision revisited after M3 (§9).
12. **Camera controls**: WASD = N/W/S/E at all altitudes — globe rotation with momentum
    when zoomed out, altitude-scaled surface travel when zoomed in; right-mouse-hold
    free-look from the current vantage with recenter key; continuous orbital ↔
    near-surface blend, no explicit mode switch (§9).
