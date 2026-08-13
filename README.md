# Infinite Worlds

Infinite Worlds is a native Rust planet generator. Give it a seed and it simulates
475+ million years of a planet's geological history — crustal formation and
supercontinent assembly, continental drift with real plate tectonics (slab pull, ridge
push, subduction, orogeny, volcanism), a full stratigraphic rock record, then glaciation,
rivers, wind and rain sculpting the last two million years — and renders the result as an
interactive 3D globe (or Mercator projection) on a true equal-area Goldberg-polyhedron
sphere, no quad-sphere or lat/long grid artifacts. Mountains and trenches are never
painted: they emerge from isostasy (crust floats on the mantle) and the same force
balance that shapes real plate boundaries. The renderer is `ash`-based Vulkan,
Wayland-first on Linux with automatic X11 fallback, no environment-variable workarounds.
See `DESIGN.md` for the full design and `IMPLEMENTATION_PLAN.md` for how it was built.

## Screenshots

Placeholder — drop PNGs into `docs/screenshots/` (not yet created) and link them here
once the beauty-view renderer has something worth showing. In the meantime,
`iw-headless gen --maps` writes equirectangular PNGs (elevation, biome, top-rock-type)
straight from a headless run — see below.

## Build & run

Requires a recent stable Rust toolchain (developed against rustc 1.97; install via
[`mise`](https://mise.jdx.dev/) or [`rustup`](https://rustup.rs/)) and, for the graphical
app, a working Vulkan driver plus `glslc`/`glslangValidator` on `PATH` (used by
`iw-render-vulkan`'s build script to compile the GLSL shaders).

```sh
# Interactive viewer — generates a planet and renders it live.
cargo run --release -p iw-app

# Useful flags:
cargo run --release -p iw-app -- --level 7 --seed 12345 --mercator
cargo run --release -p iw-app -- --exit-after-secs 3   # smoke test, no window interaction

# Headless generation — no window, writes checkpoints + summary.json.
cargo run --release -p iw-headless -- gen --seed 42 --level 6 --out /tmp/planet42

# ...with equirectangular elevation/biome/top-rock PNGs:
cargo run --release -p iw-headless -- gen --seed 42 --level 6 --out /tmp/planet42 --maps

# ...with a custom phase schedule (crust,drift,refine,recent, in Myr):
cargo run --release -p iw-headless -- gen --seed 42 --level 4 --out /tmp/quick --durations 20,20,8,0.2

# Resume a previous run from the checkpoint at a phase boundary:
cargo run --release -p iw-headless -- resume --from /tmp/planet42 --phase refine
```

`iw-headless gen` and `resume` both print (and write to `<out>/summary.json`) a compact
golden-stats fingerprint: land fraction, sea level, elevation quartiles, plate count, and
per-biome / per-rock-type area fractions.

## Architecture

Hexagonal (ports & adapters). The domain — everything that knows what a planet *is* and
how it evolves — has zero dependency on Vulkan, winit, the filesystem or any OS API;
adapters depend on the domain, never the reverse.

```
   driving side                     DOMAIN CORE                    driven side
 iw-app (winit/egui)  ─▶  iw-mesh · iw-core · iw-sim  ─▶  iw-render-vulkan (ash, egui)
 iw-headless (CLI/CI) ─▶  iw-tectonics · iw-geology    ─▶  iw-store-postcard (checkpoints)
                          iw-climate · iw-surface       ─▶  iw-export-png (map PNGs)
                          iw-biomes
```

| Crate | What it is |
|---|---|
| `iw-mesh` | Goldberg-polyhedron planet mesh (dual of a subdivided icosahedron): cell centers, CSR adjacency, areas, corners, spherical/lat-lon helpers. No `iw-*` dependencies. |
| `iw-core` | Shared domain contract: `Planet` (struct-of-arrays state), `PlanetConfig`, `StrataColumns`/`RockType`/`Biome`, the `Process` trait, and the driven ports (`CheckpointStore`, `MapExporter`, `ProgressSink`). |
| `iw-tectonics` | `Process`: craton seeding & accretion, rigid rotating plates, slab pull / ridge push / basal drag, subduction/collision/rifting, hotspots. Owns crust fields and plate membership, never elevation. |
| `iw-geology` | `Process`: isostasy (the elevation authority), hypsometric sea-level solve, metamorphism, igneous emplacement, the mass-conservation ledger. |
| `iw-climate` | `Process`: latitudinal temperature + lapse rate, prescribed wind belts, iterative moisture advection and orographic precipitation. |
| `iw-surface` | `Process`: priority-flood drainage, stream-power fluvial erosion/deposition, hillslope diffusion, glaciers, aeolian and coastal processes. |
| `iw-biomes` | `Process`: Whittaker temperature×precipitation classification into the 14 WWF terrestrial biomes, plus ocean/lake/ice-sheet and the render palette. |
| `iw-sim` | Orchestrator: phase schedule, deterministic per-process RNG streams, checkpointing, snapshot publishing, progress narration. No dependency beyond `iw-core`/`iw-mesh`. |
| `iw-store-postcard` | Adapter: postcard+zstd `CheckpointStore` (full-fidelity, resumable) and a disk-capped history snapshot ring for the time scrubber. |
| `iw-export-png` | Adapter: equirectangular/Mercator PNG export of elevation, biome and top-rock maps. |
| `iw-headless` | Adapter/binary: CLI batch generation, resume, map export and golden-stats JSON — the integration surface `iw-golden`'s tests build on. |
| `iw-render-vulkan` | Adapter: `ash` Vulkan renderer — globe/Mercator pipelines, camera, egui overlay. |
| `iw-app` | Adapter/binary: `winit`/Wayland shell wiring the live simulation to the renderer. |
| `iw-golden` | Test-only crate: end-to-end determinism, golden-planet stat ranges, mass-ledger, config-sanitize and checkpoint-resume tests, run against the real pipeline. |

## Keyboard & mouse controls (`iw-app`)

One camera; behavior blends continuously with altitude (orbital far out, near-surface
travel when zoomed in). Full flags: `cargo run -p iw-app -- --help`.

| Input | Action |
|---|---|
| `W` `A` `S` `D` | Move north / west / south / east (always screen-independent) |
| Left drag | Spin the globe (pan, in Mercator mode) |
| Left click | Inspect the cell under the cursor |
| `I` | Inspect the cell at the centre of the view |
| Right drag (hold) | Free look — pitch/yaw without moving the camera |
| Scroll | Zoom along the view ray |
| `R` | Recenter the view |
| `G` / `M` | Toggle globe ↔ Mercator projection |
| `Space` | Pause / resume the simulation |
| `Shift`+`S` | Single-step the simulation (plain `S` moves south) |
| `N` | Toggle the New Planet panel |
| `F5` | Roll a new seed and regenerate |
| `1`–`9`, `0` | Select data layer by index |
| `-` | Water-flux data layer |
| `Tab` | Next data layer |
| `,` / `.` | Scrub the history one snapshot back / forward |
| `L` | Return to the live simulation |
| `F1` | Toggle this help |
| `Esc` | Quit |

## Testing

```sh
cargo test --workspace              # everything except slow/#[ignore]d tests
cargo test --workspace -- --ignored # the slow ones too (includes iw-golden's ~12s golden planets)
cargo test -p iw-golden -- --ignored --nocapture  # just the two golden-planet runs
```

`iw-golden` (`crates/iw-golden/tests/golden.rs`) runs the real five-process pipeline
in-process: end-to-end determinism, loose golden-planet stat ranges for seeds 42 and
1337 at full fidelity, mass-ledger balance (via the sim's own `debug_assert`, live in
dev-profile builds), `PlanetConfig::sanitize` property tests, and checkpoint-resume
equivalence.

## Tunables (`PlanetConfig`, DESIGN.md §9.1)

Every field is plumbed through one serde-able `PlanetConfig`, saved into every
checkpoint so a planet remembers how it was made. Ranges below are the hard clamps
applied by `PlanetConfig::sanitize` (`crates/iw-core/src/config.rs`); the New Planet
panel exposes the same ranges as slider bounds.

| Tunable | Default | Range | Effect |
|---|---|---|---|
| Seed | 42 | any `u64` | Fully determines the planet given the rest of the config; the 🎲 button randomizes it. |
| Mesh subdivision level | 8 (~655k cells) | 4–10 | Goldberg-sphere resolution; cells = `10·4^level + 2`. Level 6 (~41k cells) is used for fast iteration/tests, 9–10 for maximum fidelity. |
| Phase durations (crust / drift / refine / recent) | 200 / 200 / 75 / 2 Myr | 0–2000 Myr each | Length of each simulation era; total run length is their sum. More drift time = more supercontinent cycles. |
| Phase timesteps (crust / drift / refine / recent) | 1 / 0.5 / 0.25 / 0.005 Myr | 0.05–5 Myr (recent-past: 0.001–5 Myr) | Simulation step size per phase; smaller = finer temporal resolution, more steps. |
| Water budget | 1.0 Earth oceans | 0.0–3.0 | Total surface water volume; sea level is solved from the hypsometric curve to hold exactly this volume. Higher drowns continents, lower exposes shelves. |
| Global temperature offset | 0 °C | −20–20 °C | Uniform shift to the whole climate field; drives ice extent, desert extent, biome belts. |
| Axial tilt | 23.4° | 0–45° | Seasonality strength and the latitude of climate belts. |
| Precipitation multiplier | 1.0× | 0.25–4× | Global moisture scaling; arid world ↔ jungle world. |
| Tectonic vigor | 1.0× | 0.25–2× | Scales plate speeds and volcanism rate; sleepy world ↔ violent world. |
| Hotspot count | 8 | 0–30 | Number of fixed mantle plumes (Hawaii-style volcanic chains, flood basalts). |
| Craton count | 14 | 4–30 | Number of Phase-1 continental nuclei. Few = huge shields; many = fragmented microcontinents. |
| Glacial intensity | 1.0× | 0.0–2.0× | Amplitude of the recent-past glacial temperature cycles. |
| History cap | 2 GiB | uncapped (any `u64`) | Disk budget for the time-scrubber's lightweight snapshot ring; snapshot interval auto-adjusts to fit. |

## License

MIT (`license = "MIT"` in the workspace `Cargo.toml`).
