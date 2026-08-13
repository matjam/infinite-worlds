use std::path::Path;

use iw_mesh::Mesh;

use crate::{Phase, Planet};

/// Progress events emitted by the simulation for UIs and logs.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    PhaseStarted {
        phase: Phase,
        time_myr: f64,
    },
    PhaseCompleted {
        phase: Phase,
        time_myr: f64,
    },
    /// Periodic step progress within a phase.
    Step {
        phase: Phase,
        step: u64,
        of: u64,
        time_myr: f64,
    },
    /// Human-facing narration ("Reticulating splines").
    Narration(String),
    /// Something notable happened (rift opened, collision started, ...).
    Milestone(String),
}

pub trait ProgressSink: Send + Sync {
    fn event(&self, ev: ProgressEvent);
}

/// Sink that drops everything (tests, benchmarks).
pub struct NullProgress;

impl ProgressSink for NullProgress {
    fn event(&self, _ev: ProgressEvent) {}
}

/// Persistence port for full simulation checkpoints (DESIGN.md §12).
pub trait CheckpointStore: Send + Sync {
    fn save(&self, tag: &str, planet: &Planet) -> anyhow::Result<()>;
    fn load(&self, tag: &str) -> anyhow::Result<Planet>;
    /// Available checkpoint tags, oldest first.
    fn list(&self) -> anyhow::Result<Vec<String>>;
}

/// Rasterized map export port (PNG heightmaps, biome maps, ...).
pub trait MapExporter: Send + Sync {
    fn export(&self, planet: &Planet, mesh: &Mesh, out_dir: &Path) -> anyhow::Result<()>;
}
