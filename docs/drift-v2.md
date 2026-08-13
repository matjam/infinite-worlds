# Drift v2 — rigid-cap crustal advection

Replaces the v1 static-crust drift engine. Root problem being fixed: crustal fields
never move, so continents can't translate, rifts scar instead of opening, hotspots
build one immortal cone instead of a chain, and boundaries stay ruler-straight.
All four user-visible defects trace to this.

## Model

Each `Plate` carries an accumulated rotation `accum: DQuat` (identity at birth,
serialized — checkpoint format bump v1→v2). Every drift step composes
`accum ← axisangle(euler_pole, omega·dt) · accum`. When any plate's accumulated
displacement `angle(accum) · R` reaches one cell pitch, a **remap step** runs for
all plates with pending rotation (in practice: every step at level ≥ 8, every few
steps at level ≤ 6).

### Pull-based remap

Double-buffer the advected fields (`plate_id`, `crust_type/thickness/density/age`,
`SUTURE` flag bit, `sediment_m`, `columns`). For every cell `d` (rayon):

- Candidate plates = owners of `d` and of its ring-k neighborhood, where
  `k = ceil(max_displacement / pitch) + 1` (small: 2–5 plates near boundaries,
  1 in interiors).
- For each candidate `p`: `src = mesh.cell_at(accum_p⁻¹ · center[d])` (seeded walk
  from `d`, converges in ~k hops). `p` claims `d` iff `owner_old(src) == p`.
- **1 claimant** → `d` pulls all advected fields from `src` (column cloned).
- **0 claimants** → divergent gap = seafloor spreading: fresh age-0 oceanic crust
  (Basalt/Gabbro column, ledger `created`), owner = claimant plate of the
  lowest-id claimed ring-1 neighbor, `RIFT` flag. This replaces v1's in-place
  ridge renewal — ridges now migrate and leave an age gradient.
- **≥2 claimants** → convergent overlap: polarity rules (continental over oceanic,
  else older/denser oceanic subducts). Winner pulls its fields; each oceanic
  loser's column is destroyed (ledger `subducted`, `SUBDUCTING` flag, trench
  flexure applied by the existing per-edge pass). Continent–continent: winner
  keeps its column, loser's crust thickness folds in via the existing shortening
  constant, `COLLISION`/`SUTURE` flags, weld impulse. This replaces the v1
  probabilistic `Transfer` block entirely.

After the remap, `accum ← identity` for remapped plates. Sub-cell discretization
error is accepted (zero-mean; the discrete grid is the truth).

Interior fast path: if `d` and its ring-1 all share one owner, only that plate is
tested (one `cell_at`). Estimated remap cost ~30–50 ms at level 8.

### What stays

Boundary edge classification by relative velocity (drives torque, arc volcanism,
trench flexure), force balance (slab pull / ridge push / drag / jitter), welding,
contiguity enforcement, hotspot deposition (chains now emerge naturally as plates
carry cones off the plume), thickness relaxation, ocean aging (now with a real
age gradient away from ridges).

### What goes

- The probabilistic per-edge `Transfer` reassignment (subsumed by remap).
- In-place ridge renewal.
- `MIN_SPEED_CAP_M_YR` floor: speeds run honest 2–10 cm/yr at every subdivision;
  remap ring-k absorbs multi-cell per-step displacement, so no saturation.

### Rifting v2 (organic)

Nucleate at max weakness = w1·SUTURE + w2·ridged-noise (iw_core::noise, planet
seed) + w3·thinness. Grow the rift path cell-by-cell from the nucleus in both
directions: next cell = neighbor maximizing weakness + direction-persistence −
revisit penalty, small rng tiebreak; stop at plate edge/ocean or length cap.
Flood-fill split on the path's two sides; diverging Euler poles; path gets RIFT
flags + crust thinning. No great circles. With advection, the two halves then
genuinely separate and the gap fills with ridge crust — scars heal.

### Boundary character

Initial boundaries come noise-warped (phase-1 handoff); rifts meander; subduction
overlap resolution follows the actual curved geometry of plate edges, and ridge
gaps trace the trailing edge, so transform-fault-like offsets appear where the
boundary direction flips relative to motion. Explicit ridge segmentation is
deferred until visual review says otherwise.

## Order of work

1. `Plate.accum` field (iw-core) + store version bump (iw-store-postcard v2) +
   fix all Plate construction sites.
2. Remap pass (new `advect.rs`), delete Transfer block, wire into `drift::step`.
3. Rift path growth (rewrite `split_plate`), remove speed-cap floor.
4. Retest: existing suite + new tests (continent translation over N steps, ridge
   age gradient monotone away from ridge, rift opens to >2-cell ocean within
   50 Myr, hotspot chain length, remap mass ledger balance, determinism/resume).
5. Recalibrate if stats drifted; then level-9 visual acceptance loop.
