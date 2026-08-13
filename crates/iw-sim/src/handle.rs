//! Threaded driver: a [`Simulation`] on a worker thread, commanded over a
//! channel, observed through snapshots, status and a drained event ring.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use iw_core::{Phase, PlanetConfig, PlanetView, Process, ProgressEvent, ProgressSink};
use iw_mesh::Mesh;

use crate::sim::Simulation;

/// Commands accepted by the worker thread.
#[derive(Debug, Clone)]
pub enum SimCommand {
    /// Run steps continuously until paused or finished.
    Start,
    /// Stop stepping; the worker then blocks until the next command.
    Pause,
    /// Advance exactly one step and stay paused.
    StepOnce,
    /// Throw the planet away and rebuild it from this config. The mesh is
    /// rebuilt only if `subdivision_level` changed. Leaves the worker paused.
    Regenerate(PlanetConfig),
    /// Reload the checkpoint written before `phase` and re-run from there with
    /// this config. Leaves the worker paused. Rejected (logged) if the config
    /// changes the subdivision level.
    RerunFromPhase(Phase, PlanetConfig),
    /// Finish the current step, then exit the thread.
    Shutdown,
}

/// What the worker is doing right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimState {
    /// Constructed or regenerated, never started.
    Idle,
    /// Stepping through this phase.
    Running(Phase),
    /// Stopped part-way through this phase.
    Paused(Phase),
    /// Every phase has completed.
    Done,
}

/// Snapshot of worker progress, cheap to poll from the UI thread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimStatus {
    /// Current run state.
    pub state: SimState,
    /// Elapsed simulated time, Myr.
    pub time_myr: f64,
    /// Steps completed across all phases.
    pub step_index: u64,
    /// Steps in the whole run under the current config.
    pub total_steps: u64,
}

/// Builds (or rebuilds) a mesh for a subdivision level. Injected so `iw-sim`
/// never calls `Mesh::build` itself and stays cheap to test.
pub type MeshBuilder = Arc<dyn Fn(u8) -> Arc<Mesh> + Send + Sync>;

/// Default number of progress events buffered for the UI thread. When the UI
/// falls behind, the oldest events are dropped.
pub const DEFAULT_EVENT_CAPACITY: usize = 512;

/// Handle to a simulation running on its own thread.
///
/// Dropping the handle shuts the worker down and joins it.
pub struct SimHandle {
    tx: Sender<SimCommand>,
    status: Arc<ArcSwap<SimStatus>>,
    view: Arc<ArcSwapOption<PlanetView>>,
    events: Arc<EventRing>,
    join: Option<JoinHandle<()>>,
}

impl SimHandle {
    /// Start a worker thread owning a fresh [`Simulation`].
    ///
    /// The worker starts paused (`SimState::Idle`); send [`SimCommand::Start`]
    /// to begin. `progress` still receives every event; the handle additionally
    /// queues events for [`SimHandle::drain_events`].
    ///
    /// `processes` are reused across `Regenerate`, so a process holding cached
    /// state must reset it when it sees `planet.step_index == 0`.
    pub fn spawn(
        config: PlanetConfig,
        mesh: Arc<Mesh>,
        processes: Vec<Box<dyn Process>>,
        store: Arc<dyn iw_core::CheckpointStore>,
        progress: Arc<dyn ProgressSink>,
        mesh_builder: MeshBuilder,
    ) -> SimHandle {
        let events = Arc::new(EventRing::new(DEFAULT_EVENT_CAPACITY));
        let sink: Arc<dyn ProgressSink> = Arc::new(TeeSink {
            inner: progress,
            ring: Arc::clone(&events),
        });
        let sim = Simulation::new(config, mesh, processes, store, sink);
        let view = sim.view_cell();
        let status = Arc::new(ArcSwap::from_pointee(SimStatus {
            state: SimState::Idle,
            time_myr: sim.planet().time_myr,
            step_index: sim.step_index(),
            total_steps: sim.total_steps(),
        }));
        let (tx, rx) = crossbeam_channel::unbounded();
        let worker = Worker {
            sim,
            rx,
            status: Arc::clone(&status),
            mesh_builder,
            mode: Mode::Idle,
            pending_steps: 0,
        };
        let join = std::thread::Builder::new()
            .name("iw-sim".to_string())
            .spawn(move || worker.run())
            .expect("spawning the simulation thread");
        SimHandle {
            tx,
            status,
            view,
            events,
            join: Some(join),
        }
    }

    /// Send a command. Silently ignored once the worker has exited.
    pub fn send(&self, cmd: SimCommand) {
        if let Err(e) = self.tx.send(cmd) {
            log::warn!("sim worker is gone, dropping command: {e}");
        }
    }

    /// Begin (or resume) continuous stepping.
    pub fn start(&self) {
        self.send(SimCommand::Start);
    }

    /// Stop stepping after the current step.
    pub fn pause(&self) {
        self.send(SimCommand::Pause);
    }

    /// Advance exactly one step and remain paused.
    pub fn step_once(&self) {
        self.send(SimCommand::StepOnce);
    }

    /// Rebuild the planet from `config` (see [`SimCommand::Regenerate`]).
    pub fn regenerate(&self, config: PlanetConfig) {
        self.send(SimCommand::Regenerate(config));
    }

    /// Re-run from a phase boundary (see [`SimCommand::RerunFromPhase`]).
    pub fn rerun_from_phase(&self, phase: Phase, config: PlanetConfig) {
        self.send(SimCommand::RerunFromPhase(phase, config));
    }

    /// Current worker status.
    pub fn status(&self) -> SimStatus {
        **self.status.load()
    }

    /// Current run state (phase-carrying).
    pub fn state(&self) -> SimState {
        self.status().state
    }

    /// Elapsed simulated time, Myr.
    pub fn time_myr(&self) -> f64 {
        self.status().time_myr
    }

    /// Most recently published snapshot, if any.
    pub fn latest_view(&self) -> Option<PlanetView> {
        self.view.load_full().map(|v| (*v).clone())
    }

    /// The shared snapshot cell, for observers that want to poll it directly.
    pub fn view_cell(&self) -> Arc<ArcSwapOption<PlanetView>> {
        Arc::clone(&self.view)
    }

    /// Take every buffered progress event, oldest first.
    pub fn drain_events(&self) -> Vec<ProgressEvent> {
        self.events.drain()
    }

    /// Number of events dropped because the UI did not drain fast enough.
    pub fn dropped_events(&self) -> u64 {
        self.events.dropped()
    }

    /// Shut the worker down and wait for it. Equivalent to dropping the handle,
    /// but explicit (and it propagates a worker panic).
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = self.tx.send(SimCommand::Shutdown);
            if let Err(e) = join.join() {
                log::error!("sim worker panicked: {e:?}");
            }
        }
    }
}

impl Drop for SimHandle {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

/// Worker-internal run mode. Distinct from [`SimState`] because the phase is
/// read from the simulation when status is published.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Idle,
    Running,
    Paused,
    Done,
}

struct Worker {
    sim: Simulation,
    rx: Receiver<SimCommand>,
    status: Arc<ArcSwap<SimStatus>>,
    mesh_builder: MeshBuilder,
    mode: Mode,
    /// Steps queued by `StepOnce` while not running.
    pending_steps: u64,
}

impl Worker {
    fn run(mut self) {
        if self.sim.is_done() {
            self.mode = Mode::Done;
        }
        self.publish_status();
        loop {
            let stepping = self.mode == Mode::Running || self.pending_steps > 0;
            if stepping {
                // Stay responsive mid-phase: poll, never block.
                loop {
                    match self.rx.try_recv() {
                        Ok(cmd) => {
                            if !self.handle(cmd) {
                                return;
                            }
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => return,
                    }
                }
            } else {
                // Nothing to do: block until commanded.
                match self.rx.recv() {
                    Ok(cmd) => {
                        if !self.handle(cmd) {
                            return;
                        }
                        self.publish_status();
                        continue;
                    }
                    Err(_) => return,
                }
            }

            if self.mode != Mode::Running && self.pending_steps == 0 {
                self.publish_status();
                continue;
            }
            if self.sim.step_once() {
                self.pending_steps = self.pending_steps.saturating_sub(1);
            } else {
                self.mode = Mode::Done;
                self.pending_steps = 0;
            }
            self.publish_status();
        }
    }

    /// Returns false when the worker should exit.
    fn handle(&mut self, cmd: SimCommand) -> bool {
        match cmd {
            SimCommand::Start => {
                if self.mode != Mode::Done {
                    self.mode = Mode::Running;
                }
            }
            SimCommand::Pause => {
                self.pending_steps = 0;
                if self.mode == Mode::Running {
                    self.mode = Mode::Paused;
                }
            }
            SimCommand::StepOnce => {
                if self.mode != Mode::Done {
                    if self.mode == Mode::Running {
                        self.mode = Mode::Paused;
                    }
                    self.pending_steps += 1;
                }
            }
            SimCommand::Regenerate(config) => {
                let mesh = if config.subdivision_level != self.sim.mesh().level {
                    (self.mesh_builder)(config.subdivision_level)
                } else {
                    Arc::clone(self.sim.mesh())
                };
                self.sim.reset(config, mesh);
                self.mode = Mode::Idle;
                self.pending_steps = 0;
            }
            SimCommand::RerunFromPhase(phase, config) => {
                if config.subdivision_level != self.sim.mesh().level {
                    log::error!(
                        "rerun from {phase:?} rejected: config level {} != mesh level {}; \
                         use Regenerate to change the mesh",
                        config.subdivision_level,
                        self.sim.mesh().level
                    );
                } else if let Err(e) = self.sim.rerun_from_phase(phase, config) {
                    log::error!("rerun from {phase:?} failed: {e:#}");
                } else {
                    self.mode = Mode::Idle;
                    self.pending_steps = 0;
                }
            }
            SimCommand::Shutdown => return false,
        }
        true
    }

    fn publish_status(&self) {
        let phase = self.sim.current_phase();
        let state = match self.mode {
            Mode::Idle => phase.map_or(SimState::Done, |_| SimState::Idle),
            Mode::Running => phase.map_or(SimState::Done, SimState::Running),
            Mode::Paused => phase.map_or(SimState::Done, SimState::Paused),
            Mode::Done => SimState::Done,
        };
        self.status.store(Arc::new(SimStatus {
            state,
            time_myr: self.sim.planet().time_myr,
            step_index: self.sim.step_index(),
            total_steps: self.sim.total_steps(),
        }));
    }
}

/// Bounded, drop-oldest event buffer shared with the UI thread.
struct EventRing {
    cap: usize,
    queue: Mutex<VecDeque<ProgressEvent>>,
    dropped: Mutex<u64>,
}

impl EventRing {
    fn new(cap: usize) -> EventRing {
        EventRing {
            cap: cap.max(1),
            queue: Mutex::new(VecDeque::with_capacity(cap.max(1))),
            dropped: Mutex::new(0),
        }
    }

    fn push(&self, ev: ProgressEvent) {
        let mut q = self.queue.lock().unwrap();
        if q.len() == self.cap {
            q.pop_front();
            *self.dropped.lock().unwrap() += 1;
        }
        q.push_back(ev);
    }

    fn drain(&self) -> Vec<ProgressEvent> {
        let mut q = self.queue.lock().unwrap();
        q.drain(..).collect()
    }

    fn dropped(&self) -> u64 {
        *self.dropped.lock().unwrap()
    }
}

/// Forwards every event to the caller's sink and to the UI ring.
struct TeeSink {
    inner: Arc<dyn ProgressSink>,
    ring: Arc<EventRing>,
}

impl ProgressSink for TeeSink {
    fn event(&self, ev: ProgressEvent) {
        self.ring.push(ev.clone());
        self.inner.event(ev);
    }
}
