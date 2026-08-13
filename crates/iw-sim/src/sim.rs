//! The orchestrator: phase schedule, process invocation, checkpoints,
//! snapshot publishing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwapOption;
use iw_core::{
    rng_for, CheckpointStore, MassLedger, Phase, Planet, PlanetConfig, PlanetView, Process,
    ProgressEvent, ProgressSink, StepCtx,
};
use iw_mesh::Mesh;

use crate::narrator::Narrator;

/// Stable, filesystem-safe name of a phase, used in checkpoint tags.
pub fn phase_slug(phase: Phase) -> &'static str {
    match phase {
        Phase::CrustalFormation => "crustal_formation",
        Phase::Drift => "drift",
        Phase::Refinement => "refinement",
        Phase::RecentPast => "recent_past",
    }
}

/// Checkpoint tag written when `phase` completes: `phase-<slug>`.
pub fn phase_tag(phase: Phase) -> String {
    format!("phase-{}", phase_slug(phase))
}

/// Publish at most this often in wall time (both conditions must hold).
const DEFAULT_PUBLISH_INTERVAL: Duration = Duration::from_millis(250);
/// ...and never more often than every N steps.
const DEFAULT_PUBLISH_STEPS: u64 = 25;
/// A narration line roughly every this many steps, plus one per phase start.
const NARRATE_EVERY_STEPS: u64 = 20;
/// Target number of `ProgressEvent::Step` events per phase.
const STEP_EVENTS_PER_PHASE: u64 = 200;
/// Mass-ledger tolerance: residual must stay under this fraction of eroded
/// volume (plus a floor for the all-zero case).
const LEDGER_TOLERANCE: f64 = 1.0e-3;

/// Where the schedule currently stands: which phase, and how many of its steps
/// are already done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cursor {
    /// Index into `Phase::ALL`; `>= 4` means the run is finished.
    phase_idx: usize,
    step_in_phase: u64,
}

impl Cursor {
    const DONE: Cursor = Cursor {
        phase_idx: Phase::ALL.len(),
        step_in_phase: 0,
    };

    const START: Cursor = Cursor {
        phase_idx: 0,
        step_in_phase: 0,
    };

    /// Reconstruct the cursor from a monotonic step counter, as stored in a
    /// checkpoint. Phases with zero steps are skipped.
    fn from_step_index(config: &PlanetConfig, step_index: u64) -> Cursor {
        let mut rem = step_index;
        for (phase_idx, phase) in Phase::ALL.iter().enumerate() {
            let steps = config.steps_in(*phase);
            if rem < steps {
                return Cursor {
                    phase_idx,
                    step_in_phase: rem,
                };
            }
            rem -= steps;
        }
        Cursor::DONE
    }

    fn phase(&self) -> Option<Phase> {
        Phase::ALL.get(self.phase_idx).copied()
    }

    /// Skip forward over phases with no steps, so `phase()` always names a
    /// phase that will actually run.
    fn normalized(mut self, config: &PlanetConfig) -> Cursor {
        while let Some(phase) = self.phase() {
            if config.steps_in(phase) > 0 {
                break;
            }
            self.phase_idx += 1;
            self.step_in_phase = 0;
        }
        self
    }
}

/// Owns the planet and drives it through the phase schedule.
///
/// `Simulation` is single-threaded and synchronous; [`crate::SimHandle`] wraps
/// it in a worker thread for the GUI. Everything it talks to the outside world
/// through is a port: [`CheckpointStore`], [`ProgressSink`], and the published
/// [`PlanetView`] snapshot cell.
pub struct Simulation {
    planet: Planet,
    mesh: Arc<Mesh>,
    processes: Vec<Box<dyn Process>>,
    store: Arc<dyn CheckpointStore>,
    progress: Arc<dyn ProgressSink>,
    narrator: Narrator,
    ledger: MassLedger,

    cursor: Cursor,
    view: Arc<ArcSwapOption<PlanetView>>,
    version: u64,
    publish_steps: u64,
    publish_interval: Duration,
    last_publish_step: u64,
    last_publish_at: Instant,
}

impl Simulation {
    /// Build a fresh planet from `config` and prepare the schedule.
    ///
    /// `processes` run in the given order every step; an empty vec is valid and
    /// yields a planet that only advances time. `mesh` must already match
    /// `config.subdivision_level` — the simulation never builds a mesh itself
    /// (see [`crate::SimHandle::spawn`] for the rebuild-on-regenerate hook).
    pub fn new(
        config: PlanetConfig,
        mesh: Arc<Mesh>,
        processes: Vec<Box<dyn Process>>,
        store: Arc<dyn CheckpointStore>,
        progress: Arc<dyn ProgressSink>,
    ) -> Simulation {
        let mut config = config;
        config.sanitize();
        let cursor = Cursor::START.normalized(&config);
        let mut planet = Planet::new(config, mesh.n_cells());
        // The tessellation is simulation state (docs/voronoi-v2.md): keep the
        // generators with the planet so checkpoints can rebuild the mesh.
        planet.mesh_generators = mesh.generators.clone();
        Simulation {
            planet,
            mesh,
            processes,
            store,
            progress,
            narrator: Narrator::default(),
            ledger: MassLedger::default(),
            cursor,
            view: Arc::new(ArcSwapOption::empty()),
            version: 0,
            publish_steps: DEFAULT_PUBLISH_STEPS,
            publish_interval: DEFAULT_PUBLISH_INTERVAL,
            last_publish_step: 0,
            last_publish_at: Instant::now(),
        }
    }

    /// Read-only access to the live planet. Only safe to call between steps,
    /// which is all the worker thread ever does.
    pub fn planet(&self) -> &Planet {
        &self.planet
    }

    /// The mesh the planet is defined on.
    pub fn mesh(&self) -> &Arc<Mesh> {
        &self.mesh
    }

    /// The narration pool in use.
    pub fn narrator(&self) -> &Narrator {
        &self.narrator
    }

    /// Replace the narration pool (e.g. parsed from a file on disk that
    /// overrides the embedded asset).
    pub fn set_narrator(&mut self, narrator: Narrator) {
        self.narrator = narrator;
    }

    /// The shared snapshot cell, for handing to observers (renderer, UI).
    pub fn view_cell(&self) -> Arc<ArcSwapOption<PlanetView>> {
        Arc::clone(&self.view)
    }

    /// Most recently published snapshot, if any. Cheap: clones a few `Arc`s.
    pub fn latest_view(&self) -> Option<PlanetView> {
        self.view.load_full().map(|v| (*v).clone())
    }

    /// Phase currently being stepped, or `None` once the run is finished.
    pub fn current_phase(&self) -> Option<Phase> {
        self.cursor.phase()
    }

    /// True once every phase has run to completion.
    pub fn is_done(&self) -> bool {
        self.cursor.phase().is_none()
    }

    /// Steps completed so far (across all phases).
    pub fn step_index(&self) -> u64 {
        self.planet.step_index
    }

    /// Steps in the whole run under the current config.
    pub fn total_steps(&self) -> u64 {
        Phase::ALL
            .iter()
            .map(|p| self.planet.config.steps_in(*p))
            .sum()
    }

    /// Override the snapshot publish throttle. A snapshot is published when
    /// **both** `steps` steps and `interval` wall time have elapsed since the
    /// last one; `(1, Duration::ZERO)` publishes every step (tests).
    pub fn set_publish_throttle(&mut self, steps: u64, interval: Duration) {
        self.publish_steps = steps.max(1);
        self.publish_interval = interval;
    }

    /// Throw the planet away and start over from `config` on `mesh`.
    /// Snapshot versions keep increasing across a reset so observers can rely
    /// on monotonicity for the lifetime of the process.
    pub fn reset(&mut self, config: PlanetConfig, mesh: Arc<Mesh>) {
        let mut config = config;
        config.sanitize();
        self.mesh = mesh;
        self.planet = Planet::new(config, self.mesh.n_cells());
        self.cursor = Cursor::START.normalized(&self.planet.config);
        self.ledger.reset();
        self.publish();
    }

    /// Continue from a planet loaded elsewhere (checkpoint resume). The
    /// schedule position is derived from `planet.step_index` under the
    /// planet's own config. A planet carrying Voronoi generators brings its
    /// own tessellation: the mesh is rebuilt from them (deterministically).
    pub fn resume_from(&mut self, planet: Planet) -> anyhow::Result<()> {
        if !planet.mesh_generators.is_empty() && planet.mesh_generators.len() != self.mesh.n_cells()
        {
            self.mesh = Arc::new(Mesh::build_from_generators(&planet.mesh_generators));
        }
        if planet.n_cells() != self.mesh.n_cells() {
            anyhow::bail!(
                "checkpoint has {} cells, mesh has {}",
                planet.n_cells(),
                self.mesh.n_cells()
            );
        }
        self.cursor = Cursor::from_step_index(&planet.config, planet.step_index);
        self.planet = planet;
        self.cursor = self.cursor.normalized(&self.planet.config);
        self.ledger.reset();
        self.publish();
        Ok(())
    }

    /// Re-run `phase` (and everything after it) with `config`, starting from
    /// the checkpoint written when the previous phase completed.
    ///
    /// Fails if the checkpoint is missing or if `config` changes the mesh
    /// subdivision level, which would invalidate every per-cell array.
    /// Re-running the first phase is equivalent to [`Simulation::reset`].
    pub fn rerun_from_phase(&mut self, phase: Phase, config: PlanetConfig) -> anyhow::Result<()> {
        let mut config = config;
        config.sanitize();
        let Some(prev) = Phase::ALL.get(phase.index().wrapping_sub(1)).copied() else {
            // No earlier phase to resume from: a full regenerate.
            let mesh = Arc::clone(&self.mesh);
            self.reset(config, mesh);
            return Ok(());
        };
        let tag = phase_tag(prev);
        let mut planet = self
            .store
            .load(&tag)
            .map_err(|e| anyhow::anyhow!("loading checkpoint {tag}: {e}"))?;
        // The checkpoint carries its own tessellation (post-retess for that
        // era boundary); rebuild the mesh from it when it differs.
        if !planet.mesh_generators.is_empty() && planet.mesh_generators.len() != self.mesh.n_cells()
        {
            self.mesh = Arc::new(Mesh::build_from_generators(&planet.mesh_generators));
        }
        if planet.n_cells() != self.mesh.n_cells() {
            anyhow::bail!(
                "checkpoint {tag} has {} cells, mesh has {}",
                planet.n_cells(),
                self.mesh.n_cells()
            );
        }
        if planet.config.cell_budget != config.cell_budget {
            anyhow::bail!(
                "rerun_from_phase({phase:?}): cell budget changed ({} -> {}); \
                 regenerate instead",
                planet.config.cell_budget,
                config.cell_budget
            );
        }
        planet.config = config;
        planet.phase = phase;
        self.planet = planet;
        self.cursor = Cursor {
            phase_idx: phase.index(),
            step_in_phase: 0,
        }
        .normalized(&self.planet.config);
        self.ledger.reset();
        self.publish();
        Ok(())
    }

    /// Run every remaining step to completion, synchronously.
    pub fn run_headless(&mut self) {
        while self.step_once() {}
    }

    /// Advance one step. Returns `false` if the run was already finished.
    ///
    /// Emits phase events, narration and progress; publishes a snapshot when
    /// the throttle allows; checkpoints at every phase completion.
    pub fn step_once(&mut self) -> bool {
        let Some(phase) = self.cursor.phase() else {
            return false;
        };
        // The cursor invariant: `phase()` never names a zero-step phase.
        let steps = self.planet.config.steps_in(phase);
        if self.cursor.step_in_phase == 0 {
            self.on_phase_started(phase);
        }

        let dt_myr = self.planet.config.dt_myr(phase);
        self.run_processes(dt_myr);

        self.planet.step_index += 1;
        self.planet.time_myr += dt_myr;
        self.cursor.step_in_phase += 1;
        let step_in_phase = self.cursor.step_in_phase;

        let every = (steps / STEP_EVENTS_PER_PHASE).max(1);
        if step_in_phase.is_multiple_of(every) || step_in_phase == steps {
            self.progress.event(ProgressEvent::Step {
                phase,
                step: step_in_phase,
                of: steps,
                time_myr: self.planet.time_myr,
            });
        }
        if self.planet.step_index.is_multiple_of(NARRATE_EVERY_STEPS) {
            self.narrate(phase);
        }
        self.maybe_publish();

        if step_in_phase >= steps {
            self.on_phase_completed(phase);
            self.cursor = Cursor {
                phase_idx: self.cursor.phase_idx + 1,
                step_in_phase: 0,
            }
            .normalized(&self.planet.config);
        }
        true
    }

    /// Run every process once, in order, with its own RNG stream, and check
    /// the mass ledger balances.
    fn run_processes(&mut self, dt_myr: f64) {
        let Simulation {
            planet,
            mesh,
            processes,
            progress,
            ledger,
            ..
        } = self;
        ledger.reset();
        let seed = planet.config.seed;
        let step_index = planet.step_index;
        for process in processes.iter_mut() {
            let mut ctx = StepCtx {
                rng: rng_for(seed, process.name(), step_index),
                progress: progress.as_ref(),
                ledger,
            };
            process.step(planet, mesh, dt_myr, &mut ctx);
        }
        debug_assert!(
            ledger.residual_m3().abs() <= LEDGER_TOLERANCE * ledger.eroded_m3 + 1.0,
            "mass ledger residual {} m^3 exceeds {:.3}% of eroded {} m^3 at step {}",
            ledger.residual_m3(),
            LEDGER_TOLERANCE * 100.0,
            ledger.eroded_m3,
            step_index
        );
    }

    fn on_phase_started(&mut self, phase: Phase) {
        self.planet.phase = phase;
        self.progress.event(ProgressEvent::PhaseStarted {
            phase,
            time_myr: self.planet.time_myr,
        });
        self.narrate(phase);
        self.publish();
    }

    fn on_phase_completed(&mut self, phase: Phase) {
        self.planet.phase = phase;
        // Adaptive re-tessellation between eras (docs/voronoi-v2.md §2): the
        // terrain this era built decides where the next era's cells crowd.
        // Runs BEFORE the checkpoint so the checkpoint carries the new
        // generators and a rerun-from-here resumes on the new mesh.
        if phase.next().is_some() && !self.mesh.generators.is_empty() {
            let epoch = phase.index() as u64 + 1;
            let before = self.mesh.n_cells();
            crate::retess::retessellate(&mut self.planet, &mut self.mesh, epoch);
            self.progress.event(ProgressEvent::Milestone(format!(
                "re-tessellated for the next era: {} -> {} cells",
                before,
                self.mesh.n_cells()
            )));
        }
        let tag = phase_tag(phase);
        if let Err(e) = self.store.save(&tag, &self.planet) {
            log::error!("checkpoint {tag} failed: {e:#}");
        }
        self.progress.event(ProgressEvent::PhaseCompleted {
            phase,
            time_myr: self.planet.time_myr,
        });
        self.narrate(phase);
        self.publish();
    }

    fn narrate(&self, phase: Phase) {
        let line = self
            .narrator
            .line(phase, self.planet.config.seed, self.planet.step_index);
        self.progress
            .event(ProgressEvent::Narration(line.to_string()));
    }

    fn maybe_publish(&mut self) {
        let steps_since = self
            .planet
            .step_index
            .saturating_sub(self.last_publish_step);
        if steps_since < self.publish_steps {
            return;
        }
        if self.last_publish_at.elapsed() < self.publish_interval {
            return;
        }
        self.publish();
    }

    /// Capture and publish a snapshot unconditionally.
    fn publish(&mut self) {
        self.version += 1;
        let view = PlanetView::capture(self.version, &self.planet, &self.mesh);
        self.view.store(Some(Arc::new(view)));
        self.last_publish_step = self.planet.step_index;
        self.last_publish_at = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_maps_step_index_to_phase() {
        let c = PlanetConfig::default();
        // 200 / 400 / 300 / 400 steps.
        assert_eq!(Cursor::from_step_index(&c, 0), Cursor::START);
        assert_eq!(
            Cursor::from_step_index(&c, 199),
            Cursor {
                phase_idx: 0,
                step_in_phase: 199
            }
        );
        assert_eq!(
            Cursor::from_step_index(&c, 200),
            Cursor {
                phase_idx: 1,
                step_in_phase: 0
            }
        );
        assert_eq!(
            Cursor::from_step_index(&c, 900),
            Cursor {
                phase_idx: 3,
                step_in_phase: 0
            }
        );
        assert_eq!(Cursor::from_step_index(&c, 1300), Cursor::DONE);
        assert_eq!(Cursor::from_step_index(&c, 99_999), Cursor::DONE);
    }

    #[test]
    fn cursor_skips_zero_length_phases() {
        let c = PlanetConfig {
            phase_durations_myr: [0.0, 10.0, 0.0, 0.0],
            phase_dt_myr: [1.0, 1.0, 1.0, 1.0],
            ..PlanetConfig::default()
        };
        assert_eq!(
            Cursor::from_step_index(&c, 0),
            Cursor {
                phase_idx: 1,
                step_in_phase: 0
            }
        );
        assert_eq!(Cursor::from_step_index(&c, 10), Cursor::DONE);
    }

    #[test]
    fn tags_are_stable() {
        assert_eq!(
            phase_tag(Phase::CrustalFormation),
            "phase-crustal_formation"
        );
        assert_eq!(phase_tag(Phase::RecentPast), "phase-recent_past");
    }
}
