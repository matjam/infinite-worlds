# Infinite Worlds — Implementation Plan

Companion to `DESIGN.md` (the authoritative design). This document tells implementing
agents exactly what to build, where, and how completion is judged. Read DESIGN.md first,
then your work package below.

## 0. Ground rules for all agents

- **Ownership**: each work package (WP) owns specific paths. Do not edit files outside
  your ownership list. Shared contracts live in `crates/iw-core` and `crates/iw-mesh`
  (owned by the coordinator; if a contract blocks you, report back — do not change it
  unilaterally unless your WP explicitly grants it).
- **Environment**: Linux/Arch, Wayland session, rustc 1.97 stable (via mise), RTX 4090,
  24 cores, 30 GB RAM. `glslc` and `glslangValidator` are on PATH. Network access to
  crates.io works.
- **Definition of done** for every WP: `cargo build -p <crate>` clean,
  `cargo test -p <crate>` green, `cargo clippy -p <crate> -- -D warnings` clean,
  no `todo!()`/`unimplemented!()` left in your paths, public items have doc comments.
- **Dependencies**: use the versions pinned in `[workspace.dependencies]` in the root
  `Cargo.toml`. If a pinned version fails to resolve or its API differs from what you
  expected, find the correct current version/API (`cargo add`, docs.rs) and update the
  root pin — that is the one root-file edit you are allowed, and note it in your report.
- **Determinism**: no `HashMap` iteration order dependence in sim results (use sorted or
  `Vec` iteration; `FxHashMap` allowed for lookups whose order never touches results),
  no time/thread-count dependence, all randomness from the RNG streams handed to you via
  `StepCtx`. Floating point: plain `f32`/`f64` ops only, no fast-math flags.
- **Units**: SI-ish and explicit in names: `_m`, `_km`, `_myr`, `_c` (Celsius),
  `_mm_yr`, `_kg_m3`, `_rad`. Simulation time in Myr as `f64`.
- **Style**: rustfmt defaults. Comments state constraints and units, not narration.
- **Performance**: write the straightforward `rayon` par_iter version first; only
  optimize if your acceptance test exceeds the budget stated in your WP.

## 1. Workspace

```
crates/
  iw-mesh            (bottom; no iw deps)
  iw-core            (depends: iw-mesh)
  iw-tectonics       (depends: core, mesh)
  iw-geology         (depends: core, mesh)
  iw-climate         (depends: core, mesh)
  iw-surface         (depends: core, mesh)
  iw-biomes          (depends: core, mesh)
  iw-sim             (depends: core, mesh + all five process crates)
  iw-store-postcard  (depends: core, mesh)
  iw-export-png      (depends: core, mesh)
  iw-headless        (bin; depends: sim + adapters)
  iw-render-vulkan   (depends: core, mesh)
  iw-app             (bin; depends: sim, render, adapters)
assets/
  snark.txt          (progress narration lines, per-phase sections)
shaders/             (GLSL, compiled to SPIR-V by iw-render-vulkan build.rs via glslc)
```

Deviation from DESIGN.md §3 (approved): a bottom crate `iw-core` holds shared types and
ports so process crates don't depend on `iw-sim`; `iw-sim` is the orchestrator only.
GUI: egui (primary; imgui-rs is an approved fallback if egui↔ash integration fails).
Allocator: `gpu-allocator` (pure Rust) instead of vk-mem.

## 2. Core contracts (already implemented — read, don't guess)

`iw-core` is fully implemented by the coordinator. Key items (see source for detail):

- `PlanetConfig` — all §9.1 tunables + seed + subdivision level. serde.
- `Planet` — SoA state: `plate_id: Vec<u16>`, `crust_type: Vec<CrustType>`,
  `crust_thickness_m`, `crust_density_kg_m3`, `crust_age_myr`, `elevation_m`,
  `sediment_m`, `temperature_c`, `precip_mm_yr`, `ice_thickness_m`,
  `water_flux_m3_yr`, `lake_depth_m`, `biome: Vec<Biome>`, `columns: StrataColumns`,
  `plates: Vec<Plate>`, `hotspots: Vec<Hotspot>`, `sea_level_m: f32`,
  `time_myr: f64`, `phase: Phase`.
- `StrataColumns` — per-cell stack of `Stratum { rock: RockType, thickness_m: f32,
  deposited_myr: f32, grade: MetamorphicGrade }`, cap 64 with thin-layer merging.
  Ops: `deposit`, `erode`, `intrude`, `top_rock`, `total_thickness_m`.
- `RockType` (18 variants), `MetamorphicGrade`, `CrustType`, `Biome` (14 WWF + Ocean,
  Lake, IceSheet), `Phase` (CrustalFormation, Drift, Refinement, RecentPast).
- `Process` trait: `fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64,
  ctx: &mut StepCtx)`. `StepCtx` carries an `rng` (Pcg64Mcg, per-process stream),
  `progress: &dyn ProgressSink`, and a scratch-buffer pool.
- Ports: `CheckpointStore`, `ProgressSink` (+ `ProgressEvent`), `MapExporter`.
- `PlanetView` — cheap-to-clone snapshot (Arcs) of render-relevant fields + `version`,
  `time_myr`, `sea_level_m`, `phase`, per-cell top-rock byte, plus plate velocity
  vectors for the plate layer.

`iw-mesh` public API is fixed (skeleton committed; WP1 fills implementations):

- `Mesh::build(level: u8) -> Mesh` — Goldberg polyhedron, `10*4^level + 2` cells.
- Fields (all public, flat): `centers: Vec<Vec3>` (unit), `areas_km2`, CSR
  `neighbor_offsets`/`neighbors`, corner vertex pool `vertices: Vec<Vec3>` + CSR
  `corner_offsets`/`corners` (CCW seen from outside), `chunks: Vec<Chunk>` (~20–80
  patches grouped by parent icosahedron face, for frustum culling), `cell_chunk`.
- `cell_at(dir: Vec3) -> u32` lookup; `latlon`, local `east`/`north` bases,
  great-circle helpers. Radius constant `EARTH_RADIUS_KM = 6371.0`.

## 3. Work packages

### WP1 — iw-mesh implementation (model: Opus)

Owns: `crates/iw-mesh/**`.
Build subdivided icosahedron (frequency `2^level`), take the dual: cell per original
vertex, corners = centroids of incident triangles (normalized), sorted CCW. Exactly 12
pentagons at icosahedron vertices. Chunks: group cells by nearest icosahedron face
center; store chunk bounding cone (`center`, `cos_radius`). `cell_at`: coarse chunk cone
test then local walk via neighbors (greedy descent on angular distance; guaranteed by
convexity). Precompute per-cell lat/lon.
Budget: `Mesh::build(8)` < 10 s, < 1.5 GB peak; level 6 < 0.5 s (used in most tests).
Tests (required): cell count formula levels 3–6; exactly 12 pentagons; every hex has 6
neighbors, adjacency symmetric; area max/min ratio < 2.0 and total = sphere area ±0.5%;
`cell_at(centers[i]) == i` sampled; corners CCW and within cell circumradius; chunk
cones cover all their cells.

### WP2 — renderer bootstrap: window, Vulkan, globe, camera (model: Opus)

Owns: `crates/iw-render-vulkan/**`, `crates/iw-app/**`, `shaders/**`.
- winit 0.30 `ApplicationHandler`, both `wayland`+`x11` features; ash 0.38 +
  ash-window + gpu-allocator; FIFO present, 2 frames in flight; resize handling.
- Reverse-Z (D32, `GREATER` compare, cleared to 0.0) from the start.
- Globe pipeline: one vertex buffer per chunk (cell corner fans, per-vertex: position
  (unit sphere), cell id u32). Storage buffer with per-cell data (elevation, color).
  Vertex shader displaces along normal by elevation × exaggeration. Frustum + horizon
  cull chunks on CPU. Push constants: view-proj, camera pos, mode flags.
- Mercator mode: same buffers, vertex shader branch projects to plane
  (x = lon, y = ln(tan(π/4 + lat/2)) clamped ±85°), orthographic camera, pan/zoom.
- Camera per DESIGN §9: orbital arcball ↔ near-surface blend by altitude, WASD =
  N/W/S/E (momentum with exponential damping when zoomed out; great-circle travel,
  altitude-scaled speed when zoomed in), right-mouse-hold free-look (pitch/yaw, persists
  until recenter key `R`), scroll zoom along view ray, min altitude ~50 km. Vertical
  exaggeration eases toward 1× below ~500 km altitude.
- egui overlay (egui + egui-winit + egui-ash-renderer or hand-rolled pass; imgui-rs
  fallback approved): FPS, camera altitude, exaggeration slider, mode toggle. Placeholder
  side panel for later tunables.
- Shaders: GLSL 450, compiled by `build.rs` with `glslc` to SPIR-V, `include_bytes!`.
- `iw-app`: for now renders a `Mesh::build(6)` planet with procedural test elevation
  (a few spherical-harmonic bumps) and flat per-cell color by elevation. CLI flags:
  `--level N`, `--exit-after-secs S` (smoke), `--mercator`.
- Smoke test = acceptance: `cargo run -p iw-app -- --exit-after-secs 3` runs on the
  live Wayland session without validation errors (enable `VK_LAYER_KHRONOS_validation`
  in debug; treat validation messages as errors in the smoke run). Unit-test the camera
  math and Mercator projection (pure functions) — no GPU needed in `cargo test`.

### WP3 — iw-sim orchestrator (model: Opus)

Owns: `crates/iw-sim/**`, `assets/snark.txt`.
- `Simulation::new(config, mesh, processes)` builds `Planet`; per-process named RNG
  streams derived from seed (`seed ⊕ hash(process_name)` — stable across runs and
  process reordering).
- Phase schedule from config durations + timesteps (DESIGN §5 table). Each step: run
  processes in fixed order (tectonics, geology, climate, surface, biomes), advance
  `time_myr`, publish `PlanetView` (arc-swap) at a max rate (every N steps such that
  publishing ≲ 10/s of wall time), emit `ProgressEvent`s.
- Worker thread + `crossbeam_channel` commands: `Start`, `Pause`, `StepOnce`,
  `Regenerate(PlanetConfig)`, `RerunFromPhase(Phase, PlanetConfig)`, `Shutdown`.
  Checkpoint via `CheckpointStore` port at each phase boundary; `RerunFromPhase` loads
  the boundary checkpoint.
- `assets/snark.txt`: sections `[crust] [drift] [refine] [recent] [generic]`, ≥ 12
  lines each, Maxis-grade ("Reticulating splines" must appear). Loader picks
  deterministically-pseudo-randomly per event; parser tested.
- Processes injected as `Vec<Box<dyn Process>>` — use trivial no-op processes in tests;
  don't depend on the real process crates (iw-sim depends on them only in `iw-app`
  wiring later — keep iw-sim's Cargo deps to core+mesh so WPs stay parallel).
  [Correction: workspace Cargo.toml already lists iw-sim deps as core+mesh only.]
- Tests: phase schedule math, determinism (two runs, no-op processes + RNG-consuming
  dummy process → identical state hash), command handling (pause/step), snapshot
  version monotonicity, snark parser.

### WP4 — iw-tectonics (model: Opus)

Owns: `crates/iw-tectonics/**`. Implements `Process` (`TectonicsProcess`).
Phase behavior keyed off `planet.phase`:
- **CrustalFormation**: seed `config.craton_count` nuclei (Poisson-disc via RNG +
  min-distance rejection on cells): continental crust 40 km / 2700 kg/m³ / age 0;
  everything else oceanic 7 km / 3000 kg/m³. Cratons drift in a low-order flow field
  (sum of 3–5 rotating spherical harmonics, slowly evolving); accretion welds contacting
  cratons (record suture cells). Late-phase attractor sweeps continents toward one
  supercontinent. Track per-cell craton/plate membership.
- **Drift/Refinement/RecentPast**: full engine.
  - Plates: `Plate { euler_pole: DVec3, omega_rad_myr: f64, ... }`. Cell velocity =
    ω × r, scaled by `config.tectonic_vigor`. Target speeds 2–10 cm/yr.
  - Forces each step: slab pull ∝ subducting boundary length × slab age; ridge push ∝
    ridge length; basal drag ∝ −v. Update ω (bounded).
  - Advection: plates are rigid — cell fields don't advect; instead boundaries migrate:
    reassign boundary cells between plates when relative motion accumulates a cell
    width. Divergent: new oceanic crust (Basalt column, age 0) at the gap. Convergent:
    consume the subducting side's cells (columns destroyed → volatile flux), build
    trench via thinning/flexure on the subducting side and arc volcanism (Andesite
    deposit + thickening) 2–4 cells behind the overriding edge. Continent–continent:
    thicken both sides toward ≤ 70 km, fold (metamorphic grade bump via geology
    hooks — set `pending_metamorphism` flags), weld plates when convergence stalls.
  - Rifting: on supercontinent (large plate, high fraction continental), nucleate rift
    along old sutures with RNG; split plate (graph flood-fill either side of rift line).
  - Oceanic aging: `crust_age_myr += dt` on oceanic cells; thermal subsidence handled in
    geology via density increase with age (set density here: ρ(age)).
  - Hotspots: fixed points; overlying cell gets basalt deposit + thickening pulse.
- Write to: `plate_id`, `crust_*`, `plates`, columns (via `StrataColumns` ops), flags.
  Do NOT touch `elevation_m` (geology/isostasy owns it).
- Tests (level 5–6 mesh, short runs): craton count & spacing; plate partition = all
  cells, contiguous; speeds within 0–15 cm/yr; divergent boundary creates age-0 oceanic
  cells; convergent consumes area at expected rate; collision raises crust thickness;
  determinism (two identical runs → identical plate_id/crust fields).
Budget: level 6, Phase 1+2 (400 steps) < 60 s.

### WP5 — iw-geology (model: Opus)

Owns: `crates/iw-geology/**`. Implements `GeologyProcess`.
- **Isostasy** (each step, the elevation authority): Airy — root depth from column
  (crust thickness × density vs mantle 3300 kg/m³), elevation = buoyancy surplus;
  include sediment and ice loads (ice 917). Then flexural smoothing: 2–4 Jacobi
  passes of neighbor averaging weighted by rigidity. Oceanic thermal subsidence via
  ρ(age) set by tectonics falls out naturally. Calibrate constants so: 35 km/2700
  continental crust ≈ +0.8 km; 7 km/3000+ old oceanic ≈ −4..−6 km; 70 km thickened ≈
  +5..+8 km. A unit test locks these anchors (±30%).
- **Sea level**: hypsometric solve for `sea_level_m` from `config.water_budget` (1.0 =
  Earth's 1.335e9 km³ scaled to mesh area): sort/partition elevations, binary search the
  level where integrated basin volume = budget (account cell areas). Converge < 1 m.
- **Metamorphism**: for buried strata compute P (overburden) & T (geotherm 25 °C/km +
  arc/collision bonus flags from tectonics); apply transition table (shale→slate→schist
  →gneiss; limestone→marble; sandstone→quartzite; basalt→amphibolite) updating grade.
- **Igneous emplacement**: consume tectonics flags: arc → Diorite/Granite intrusion at
  depth + Andesite/Tuff at surface; hotspot → Basalt; collision melt → Granite plutons.
- **Mass ledger**: `debug_assert` accounting — Δ(total column mass) matches deposits −
  erosion − subduction losses fed through a `MassLedger` on StepCtx scratch.
- Tests: isostasy anchors; sea-level solver on synthetic hypsometries (analytic cases,
  budget 0 → level at global min); metamorphic table thresholds; determinism.

### WP6 — iw-climate (model: Opus)

Owns: `crates/iw-climate/**`. Implements `ClimateProcess` per DESIGN §7.
- Annual-mean T: `T = T_eq − ΔT·sin²(lat)` (T_eq ≈ 28 °C, pole ≈ −25 °C baseline),
  + `config.temperature_offset_c`, − 6.5 °C/km × max(elev − sea, 0), + maritime damping
  (distance-to-ocean BFS, capped), + glacial forcing input (set by sim in RecentPast:
  sinusoidal −6..0 °C). Summer/winter endpoints: ±seasonal amplitude ∝ tilt × sin(lat),
  continentality amplifies.
- Winds: 6 belts by latitude (trades/westerlies/polar easterlies) with Coriolis-tilted
  directions; per-cell unit wind vector on the tangent plane.
- Precipitation: iterative moisture advection — ocean cells evaporate (T-dependent),
  moisture advects downwind cell-to-cell (project wind onto neighbor directions),
  rains out ∝ moisture × (base + orographic lift (uphill Δelev) + ITCZ/convergence
  factor), × `config.precip_multiplier`. ~10 relaxation sweeps; deterministic order.
  Must produce: rain shadows behind ranges, wet equator, dry ~30° belts, dry deep
  interiors.
- Runs every step but cheap (< 15% of step time); cache belt geometry per mesh.
- Tests: lapse rate & latitude gradient; rain shadow on synthetic ridge planet (windward
  ≥ 2× leeward); precip belts on aquaplanet (equator max, ~30° min); determinism.

### WP7 — iw-surface (model: Opus)

Owns: `crates/iw-surface/**`. Implements `SurfaceProcess` per DESIGN §5 Phase 3/4.
- Active from Refinement (coarse) and RecentPast (full, kyr steps).
- Priority-flood (binary-heap) on the sphere: fill depressions → `lake_depth_m`
  (above sea level only), single-direction flow to steepest lower neighbor,
  accumulate discharge from `precip` × area (+ melt).
- Fluvial: stream power `E = K·A^0.5·S` (K scaled so major rivers incise ~1 km/Myr in
  uplifted terrain); erode via `columns.erode` (rock-hardness factor per RockType:
  granite/quartzite hard, shale/evaporite soft), transported sediment settles where
  capacity (∝ Q·S) drops: floodplains, lake floors, and **deltas** at river mouths
  (deposit Sandstone/Shale mix; conglomerate at mountain fronts, limestone in warm
  shallow seas away from clastic input, evaporites in closed dry basins).
- Hillslope: linear diffusion of `sediment_m` + weathering (bedrock → sediment, rate
  T/precip dependent).
- Glaciers (RecentPast): ice accumulates where snowfall > melt; SIA-lite flow (ice flux
  ∝ thickness² × slope downhill); glacial erosion ∝ sliding flux carving into columns —
  overdeepening allowed (below sea level → fjords when coastal, lakes inland);
  moraine/outwash deposition at termini.
- Aeolian: arid cells (precip < 250 mm) lose fine sediment downwind, deposit as dunes/
  loess. Coastal: wave erosion on exposed shoreline cells, deposition in bays.
- All erosion/deposition through `StrataColumns` + `sediment_m`; report to MassLedger.
- Tests: synthetic cone → radial drainage, mass conserved ±0.1%; depression filling
  leaves no undrained land pits; glacier on synthetic ridge carves below original
  valley floor; delta forms at synthetic river mouth; determinism.
Budget: level 6 RecentPast (2 Myr @ 5 kyr = 400 steps) < 90 s.

### WP8 — iw-biomes (model: Sonnet)

Owns: `crates/iw-biomes/**`. Implements `BiomeProcess` (runs last, cheap).
Whittaker table + overrides per DESIGN §8 using T (annual + seasonality), precip,
elevation, `lake_depth`, ice, coastal adjacency. Also export `biome_color(Biome) ->
[u8;3]` (satellite-look palette: vegetation greens, desert tans, tundra grey-browns) —
WP11 consumes it. Tests: aquaplanet → all ocean; hot+wet → tropical moist forest;
cold high lat → tundra/taiga; montane override on high tropical cells; all 14
reachable on a synthetic T/P sweep.

### WP9 — adapters (model: Sonnet)

Owns: `crates/iw-store-postcard/**`, `crates/iw-export-png/**`, `crates/iw-headless/**`.
- Store: postcard+zstd `Planet` checkpoints, 16-byte magic+version header, `save/load/
  list` in a planet directory; history ring of light snapshots respecting
  `config.history_cap_bytes` (thin snapshots at adaptive interval).
- Export: equirectangular (2:1) and Mercator PNGs of elevation (hypsometric tint),
  biome (WP8 palette), top rock type; supersample 2× via `cell_at`.
- Headless: clap CLI `iw-headless gen --seed N --level L --out dir [--phases ...]`
  wiring sim + real processes + store + export; prints golden summary stats (land %,
  hypsometric quartiles, plate count, biome fractions) as JSON. This binary IS the
  integration test used by WP12.

### WP10 — integration & UI (model: Opus)

Owns: `crates/iw-app/**` (takes over from WP2), plus read-only use of everything.
Wire real pipeline: New Planet panel (all §9.1 tunables, seed dice 🎲 regen), live
generation view (arc-swap snapshots → per-cell color/elevation SSBO updates ≤ 10 Hz),
progress bar with snark lines, data layers (beauty placeholder, elevation, biomes,
plates+velocity arrows, crust age/type/thickness, top rock, temperature, precip, ice,
discharge), cell picking (ray → `cell_at`) with inspector window showing the strata
column (colored stack, rock names, ages, grades), time scrubber over history snapshots,
pause/resume/step controls, RerunFromPhase buttons at phase boundaries.

### WP11 — beauty view (model: Opus)

Owns: shaders + the beauty-layer code paths in `iw-render-vulkan`/`iw-app`.
Blue Marble target per DESIGN §9: biome-driven land albedo with slope/relief shading
(analytic normals from elevation gradient), ocean depth gradient (deep #0b2a5e-ish →
shelf turquoise), specular sun glint, ice sheets/glaciers white with slight blue,
atmospheric limb scattering (screen-space rim on the sphere silhouette + subtle height
fog), starfield background (procedural, seeded, temperature-tinted stars, no external
assets), optional procedural cloud layer (2 octaves flow noise advected by wind field,
off by default). Sun direction slowly orbits or fixed at pleasing angle (toggle).

### WP12 — tests, golden planets, docs (model: Sonnet)

Owns: `tests/` (workspace-level), `README.md`, doc passes in any crate (doc comments
only — no behavior changes).
End-to-end determinism (headless seed 42 twice → identical stats JSON), golden stats
committed for seeds {42, 1337} level 6 with loose ranges (land 15–55%, plate count
6–30, ≥ 6 biomes present, max elev 2–12 km, min elev −12..−3 km), property tests, README
(build/run instructions, screenshots section stub, tunables table).

## 4. Execution order

```
Wave 1 (parallel): WP1 mesh · WP2 renderer · WP3 sim
Wave 2 (parallel): WP4 tectonics · WP5 geology · WP6 climate   (after Wave 1 core green)
Wave 3 (parallel): WP7 surface · WP8 biomes · WP9 adapters
Wave 4: WP10 integration  →  WP11 beauty  ·  WP12 tests/docs (parallel with WP11)
Final: coordinator verification, smoke run, commits at each wave boundary.
```

Waves 2/3 crates compile against core+mesh only, so they could start earlier; kept in
waves so integration problems surface with fewer moving parts.

## 5. Verification commands (coordinator, each wave)

```
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p iw-headless -- gen --seed 42 --level 6 --out /tmp/planet42   (Wave 3+)
cargo run -p iw-app -- --exit-after-secs 3                                 (Wave 1+)
```
