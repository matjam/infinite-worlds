//! Test doubles: synthetic meshes, an in-memory checkpoint store, trivial
//! processes and a recording progress sink.
//!
//! Public (behind `#[doc(hidden)]`) so integration tests and other crates'
//! tests can use them without depending on real process crates or on
//! `Mesh::build`. Not part of the supported API.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use glam::Vec3;
use iw_core::{CheckpointStore, Planet, Process, ProgressEvent, ProgressSink, StepCtx};
use iw_mesh::{latlon_of, Chunk, Mesh, EARTH_RADIUS_KM};
use rand::RngCore;

/// Level reported by the synthetic meshes. `PlanetConfig::sanitize` clamps
/// `subdivision_level` to at least 4, and the simulation refuses to re-run a
/// phase when the config level and the mesh level disagree, so the fake meshes
/// claim the lowest level a sanitized config can hold. Cell counts do not match
/// `Mesh::expected_cells(4)`; nothing in `iw-sim` requires that they do.
pub const MIN_LEVEL: u8 = 4;

/// Assemble a mesh from cell centers and a neighbor list, filling the
/// remaining fields with consistent (if not geometrically exact) values.
/// Corners are fabricated: one shared vertex pool, three per cell.
fn synthetic_mesh(level: u8, centers: Vec<Vec3>, neighbors_of: Vec<Vec<u32>>) -> Mesh {
    let n = centers.len();
    assert_eq!(n, neighbors_of.len());
    let mut neighbor_offsets = Vec::with_capacity(n + 1);
    let mut neighbors = Vec::new();
    neighbor_offsets.push(0);
    for nb in &neighbors_of {
        neighbors.extend_from_slice(nb);
        neighbor_offsets.push(neighbors.len() as u32);
    }
    // Three fake corners per cell, taken from the centers pool so every index
    // is in range and the CSR arrays are well formed.
    let mut corner_offsets = Vec::with_capacity(n + 1);
    let mut corners = Vec::new();
    corner_offsets.push(0);
    for i in 0..n {
        for k in 0..3 {
            corners.push(((i + k) % n) as u32);
        }
        corner_offsets.push(corners.len() as u32);
    }
    let area = 4.0 * std::f32::consts::PI * EARTH_RADIUS_KM * EARTH_RADIUS_KM / n as f32;
    let latlon = centers.iter().map(|c| latlon_of(*c)).collect();
    let chunk = Chunk {
        cells: (0..n as u32).collect(),
        center: Vec3::Z,
        cos_radius: -1.0,
    };
    Mesh {
        level,
        vertices: centers.clone(),
        centers,
        areas_km2: vec![area; n],
        latlon,
        neighbor_offsets,
        neighbors,
        corner_offsets,
        corners,
        chunks: vec![chunk],
        cell_chunk: vec![0; n],
    }
}

/// Four cells at the vertices of a tetrahedron, each adjacent to the other
/// three. Enough for any code that only walks cells and neighbors.
pub fn tiny_mesh() -> Arc<Mesh> {
    let centers: Vec<Vec3> = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ]
    .iter()
    .map(|v| v.normalize())
    .collect();
    let neighbors_of = (0..4u32)
        .map(|i| (0..4u32).filter(|j| *j != i).collect())
        .collect();
    Arc::new(synthetic_mesh(MIN_LEVEL, centers, neighbors_of))
}

/// `n` cells evenly spaced around the equator, each adjacent to its two
/// neighbors. Useful when a test wants more than four cells.
pub fn ring_mesh(n: usize) -> Arc<Mesh> {
    assert!(n >= 3, "a ring needs at least 3 cells");
    let centers = (0..n)
        .map(|i| {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos(), a.sin(), 0.0)
        })
        .collect();
    let neighbors_of = (0..n)
        .map(|i| vec![((i + n - 1) % n) as u32, ((i + 1) % n) as u32])
        .collect();
    Arc::new(synthetic_mesh(MIN_LEVEL, centers, neighbors_of))
}

/// In-memory [`CheckpointStore`]. WP9 ships the postcard+zstd one.
#[derive(Default)]
pub struct MemoryStore {
    inner: Mutex<MemoryStoreInner>,
}

#[derive(Default)]
struct MemoryStoreInner {
    order: Vec<String>,
    planets: HashMap<String, Planet>,
}

impl MemoryStore {
    /// An empty store.
    pub fn new() -> MemoryStore {
        MemoryStore::default()
    }

    /// Tags written so far, oldest first (duplicates collapsed).
    pub fn tags(&self) -> Vec<String> {
        self.inner.lock().unwrap().order.clone()
    }

    /// Number of distinct tags stored.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().planets.len()
    }

    /// True when nothing has been saved.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl CheckpointStore for MemoryStore {
    fn save(&self, tag: &str, planet: &Planet) -> anyhow::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        if inner
            .planets
            .insert(tag.to_string(), planet.clone())
            .is_none()
        {
            inner.order.push(tag.to_string());
        }
        Ok(())
    }

    fn load(&self, tag: &str) -> anyhow::Result<Planet> {
        let inner = self.inner.lock().unwrap();
        inner
            .planets
            .get(tag)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no checkpoint {tag}"))
    }

    fn list(&self) -> anyhow::Result<Vec<String>> {
        Ok(self.tags())
    }
}

/// [`ProgressSink`] that keeps every event it is handed.
#[derive(Default)]
pub struct RecordingSink {
    events: Mutex<Vec<ProgressEvent>>,
}

impl RecordingSink {
    /// An empty recorder.
    pub fn new() -> RecordingSink {
        RecordingSink::default()
    }

    /// Copy of everything recorded so far.
    pub fn events(&self) -> Vec<ProgressEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl ProgressSink for RecordingSink {
    fn event(&self, ev: ProgressEvent) {
        self.events.lock().unwrap().push(ev);
    }
}

/// A process that does nothing at all.
pub struct NoopProcess;

impl Process for NoopProcess {
    fn name(&self) -> &'static str {
        "noop"
    }
    fn step(&mut self, _planet: &mut Planet, _mesh: &Mesh, _dt_myr: f64, _ctx: &mut StepCtx) {}
}

/// A process that perturbs `elevation_m` from its RNG stream, so two runs can
/// only agree if the whole RNG plumbing is deterministic.
pub struct RngNoiseProcess {
    name: &'static str,
}

impl Default for RngNoiseProcess {
    fn default() -> Self {
        RngNoiseProcess { name: "rng-noise" }
    }
}

impl RngNoiseProcess {
    /// Named instance (the name selects the RNG stream).
    pub fn new(name: &'static str) -> RngNoiseProcess {
        RngNoiseProcess { name }
    }
}

impl Process for RngNoiseProcess {
    fn name(&self) -> &'static str {
        self.name
    }

    fn step(&mut self, planet: &mut Planet, mesh: &Mesh, dt_myr: f64, ctx: &mut StepCtx) {
        for cell in 0..mesh.n_cells() {
            let r = (ctx.rng.next_u32() as f64 / u32::MAX as f64) as f32 - 0.5;
            // Mix in mesh geometry and dt so a wrong mesh or schedule shows up.
            // The offset keeps the factor non-zero for equatorial cells.
            planet.elevation_m[cell] += r * 100.0 * dt_myr as f32 * (mesh.centers[cell].z + 1.5);
        }
    }
}

/// A process that burns wall time, for exercising the worker thread's
/// responsiveness to commands mid-phase.
pub struct SlowProcess {
    /// Sleep taken on every step.
    pub per_step: Duration,
}

impl SlowProcess {
    /// Sleep `millis` per step.
    pub fn new(millis: u64) -> SlowProcess {
        SlowProcess {
            per_step: Duration::from_millis(millis),
        }
    }
}

impl Process for SlowProcess {
    fn name(&self) -> &'static str {
        "slow"
    }

    fn step(&mut self, _planet: &mut Planet, _mesh: &Mesh, _dt_myr: f64, _ctx: &mut StepCtx) {
        std::thread::sleep(self.per_step);
    }
}

/// A process that reports deposition it never balances with erosion — it
/// exists to prove the mass-ledger `debug_assert` in the step loop bites.
pub struct UnbalancedProcess;

impl Process for UnbalancedProcess {
    fn name(&self) -> &'static str {
        "unbalanced"
    }

    fn step(&mut self, _planet: &mut Planet, _mesh: &Mesh, _dt_myr: f64, ctx: &mut StepCtx) {
        ctx.ledger.deposited_m3 += 1.0e9;
    }
}
