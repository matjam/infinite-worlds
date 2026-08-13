//! Worker-thread command handling: start, pause, single-step, regenerate,
//! rerun, shutdown — and the UI-facing event ring.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use iw_core::{NullProgress, Phase, PlanetConfig, Process, ProgressEvent};
use iw_mesh::Mesh;
use iw_sim::test_util::{
    ring_mesh, tiny_mesh, MemoryStore, RngNoiseProcess, SlowProcess, MIN_LEVEL,
};
use iw_sim::{MeshBuilder, SimHandle, SimState};

/// 10 ms per step, 100 steps per phase: long enough that commands land
/// mid-phase, short enough for a test suite.
const STEP_MS: u64 = 10;

fn config(seed: u64) -> PlanetConfig {
    PlanetConfig {
        seed,
        subdivision_level: MIN_LEVEL,
        phase_durations_myr: [100.0, 100.0, 100.0, 100.0],
        phase_dt_myr: [1.0, 1.0, 1.0, 1.0],
        ..PlanetConfig::default()
    }
}

fn processes() -> Vec<Box<dyn Process>> {
    vec![
        Box::new(SlowProcess::new(STEP_MS)),
        Box::new(RngNoiseProcess::default()),
    ]
}

fn counting_mesh_builder(calls: Arc<AtomicUsize>) -> MeshBuilder {
    Arc::new(move |cfg: &iw_core::PlanetConfig| {
        calls.fetch_add(1, Ordering::SeqCst);
        // Stand-in for a real tessellation; cell count irrelevant here.
        let level = cfg.subdivision_level;
        let mesh = ring_mesh(6 + level as usize);
        Arc::new(Mesh {
            level,
            ..Arc::try_unwrap(mesh).unwrap_or_else(|_| unreachable!())
        })
    })
}

fn spawn(handle_config: PlanetConfig, calls: Arc<AtomicUsize>) -> SimHandle {
    SimHandle::spawn(
        handle_config,
        tiny_mesh(),
        processes(),
        Arc::new(MemoryStore::new()),
        Arc::new(NullProgress),
        counting_mesh_builder(calls),
    )
}

/// Poll until `pred` holds or the deadline passes; returns whether it held.
fn wait_for(timeout_ms: u64, mut pred: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        if pred() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    pred()
}

#[test]
fn starts_idle_and_does_not_step_until_told() {
    let sim = spawn(config(1), Arc::new(AtomicUsize::new(0)));
    assert_eq!(sim.state(), SimState::Idle);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(sim.status().step_index, 0);
    assert_eq!(sim.status().total_steps, 400);
    sim.shutdown();
}

#[test]
fn pause_stops_the_run_mid_phase() {
    let sim = spawn(config(2), Arc::new(AtomicUsize::new(0)));
    sim.start();
    assert!(
        wait_for(2000, || sim.status().step_index >= 3),
        "worker never started stepping"
    );
    assert!(matches!(
        sim.state(),
        SimState::Running(Phase::CrustalFormation)
    ));

    sim.pause();
    assert!(
        wait_for(2000, || matches!(sim.state(), SimState::Paused(_))),
        "worker ignored Pause: {:?}",
        sim.state()
    );
    let stopped_at = sim.status().step_index;
    std::thread::sleep(Duration::from_millis(10 * STEP_MS));
    assert_eq!(
        sim.status().step_index,
        stopped_at,
        "steps kept coming after Pause"
    );

    // ...and Start picks it up again from where it stopped.
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index > stopped_at));
    sim.shutdown();
}

#[test]
fn step_once_advances_exactly_one_step() {
    let sim = spawn(config(3), Arc::new(AtomicUsize::new(0)));
    sim.step_once();
    assert!(
        wait_for(2000, || sim.status().step_index == 1),
        "StepOnce did not step: {:?}",
        sim.status()
    );
    std::thread::sleep(Duration::from_millis(5 * STEP_MS));
    assert_eq!(sim.status().step_index, 1, "StepOnce ran away");
    assert!(sim.status().time_myr > 0.0);

    sim.step_once();
    sim.step_once();
    assert!(wait_for(2000, || sim.status().step_index == 3));
    std::thread::sleep(Duration::from_millis(5 * STEP_MS));
    assert_eq!(sim.status().step_index, 3);
    sim.shutdown();
}

#[test]
fn regenerate_resets_time_and_rebuilds_the_tessellation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let sim = spawn(config(4), calls.clone());
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index >= 3));

    // Every regenerate is a fresh tessellation (a new seed or budget means
    // new generators), so the injected builder is called each time.
    sim.regenerate(config(5));
    assert!(
        wait_for(2000, || sim.status().step_index == 0),
        "Regenerate did not reset: {:?}",
        sim.status()
    );
    assert_eq!(sim.status().time_myr, 0.0);
    assert_eq!(
        sim.state(),
        SimState::Idle,
        "Regenerate must leave it paused"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "builder not called");

    let mut bigger = config(6);
    bigger.subdivision_level = MIN_LEVEL + 1;
    sim.regenerate(bigger);
    assert!(wait_for(2000, || calls.load(Ordering::SeqCst) == 2));
    assert!(wait_for(2000, || sim
        .latest_view()
        .map(|v| v.cells.elevation_m.len())
        == Some(6 + MIN_LEVEL as usize + 1)));

    // ...and the fresh planet still runs.
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index >= 2));
    sim.shutdown();
}

#[test]
fn rerun_from_phase_reloads_the_previous_checkpoint() {
    let store = Arc::new(MemoryStore::new());
    let mut cfg = config(7);
    // Four steps per phase so a full run finishes quickly.
    cfg.phase_durations_myr = [4.0, 4.0, 4.0, 4.0];
    let sim = SimHandle::spawn(
        cfg.clone(),
        tiny_mesh(),
        vec![Box::new(RngNoiseProcess::default())],
        store.clone(),
        Arc::new(NullProgress),
        counting_mesh_builder(Arc::new(AtomicUsize::new(0))),
    );
    sim.start();
    assert!(
        wait_for(4000, || sim.state() == SimState::Done),
        "run did not finish: {:?}",
        sim.status()
    );
    assert_eq!(sim.status().step_index, 16);

    sim.rerun_from_phase(Phase::RecentPast, cfg.clone());
    assert!(
        wait_for(2000, || sim.status().step_index == 12),
        "rerun did not rewind: {:?}",
        sim.status()
    );
    assert_eq!(sim.state(), SimState::Idle);
    sim.start();
    assert!(wait_for(4000, || sim.state() == SimState::Done));
    assert_eq!(sim.status().step_index, 16);

    // A cell-budget change is refused (it would invalidate every per-cell
    // array) and leaves the sim alone.
    let mut wrong = cfg;
    wrong.cell_budget += 12_345;
    sim.rerun_from_phase(Phase::Drift, wrong);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(sim.status().step_index, 16);
    assert_eq!(sim.state(), SimState::Done);
    sim.shutdown();
}

#[test]
fn shutdown_joins_even_while_running() {
    let sim = spawn(config(8), Arc::new(AtomicUsize::new(0)));
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index >= 2));
    let t = Instant::now();
    sim.shutdown();
    assert!(
        t.elapsed() < Duration::from_secs(2),
        "shutdown blocked for {:?}",
        t.elapsed()
    );
}

#[test]
fn events_are_queued_for_the_ui_and_drained_once() {
    let sim = spawn(config(9), Arc::new(AtomicUsize::new(0)));
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index >= 5));
    sim.pause();
    let first = sim.drain_events();
    assert!(
        first
            .iter()
            .any(|e| matches!(e, ProgressEvent::PhaseStarted { .. })),
        "no PhaseStarted in {first:?}"
    );
    assert!(first
        .iter()
        .any(|e| matches!(e, ProgressEvent::Narration(_))));
    assert!(first
        .iter()
        .any(|e| matches!(e, ProgressEvent::Step { .. })));
    // Draining is destructive.
    assert!(sim
        .drain_events()
        .iter()
        .all(|e| !matches!(e, ProgressEvent::PhaseStarted { .. })));
    assert_eq!(sim.dropped_events(), 0);
    sim.shutdown();
}

#[test]
fn views_are_published_from_the_worker() {
    let sim = spawn(config(10), Arc::new(AtomicUsize::new(0)));
    sim.start();
    assert!(wait_for(2000, || sim.latest_view().is_some()));
    let v = sim.latest_view().unwrap();
    assert_eq!(v.phase, Phase::CrustalFormation);
    assert_eq!(v.cells.elevation_m.len(), 4);
    assert_eq!(v.cells.top_rock.len(), 4);
    let cell = sim.view_cell();
    assert!(cell.load_full().is_some());
    sim.shutdown();
}

/// Deposits one distinct stratum per step so a queried column has structure.
struct DepositProcess;

impl Process for DepositProcess {
    fn name(&self) -> &'static str {
        "deposit"
    }

    fn step(
        &mut self,
        planet: &mut iw_core::Planet,
        _mesh: &Mesh,
        _dt_myr: f64,
        _ctx: &mut iw_core::StepCtx,
    ) {
        let time = planet.time_myr;
        // Alternating rock types so `deposit` cannot merge them into one.
        let rock = if planet.step_index.is_multiple_of(2) {
            iw_core::RockType::Basalt
        } else {
            iw_core::RockType::Shale
        };
        planet.columns.deposit(0, rock, 100.0, time);
    }
}

#[test]
fn query_column_answers_while_the_worker_runs() {
    let sim = SimHandle::spawn(
        config(11),
        tiny_mesh(),
        vec![
            Box::new(SlowProcess::new(STEP_MS)),
            Box::new(DepositProcess),
        ],
        Arc::new(MemoryStore::new()),
        Arc::new(NullProgress),
        counting_mesh_builder(Arc::new(AtomicUsize::new(0))),
    );
    sim.start();
    assert!(wait_for(2000, || sim.status().step_index >= 3));

    let column = sim
        .query_column(0, Duration::from_secs(2))
        .expect("worker answered the column query");
    assert!(
        column.len() >= 2,
        "expected a stacked column, got {column:?}"
    );
    assert!(column.iter().all(|s| s.thickness_m > 0.0));
    // Bottom-to-top: deposition times must not decrease.
    assert!(column
        .windows(2)
        .all(|w| w[0].deposited_myr <= w[1].deposited_myr));

    // A cell with nothing deposited, and an out-of-range cell, both answer.
    assert_eq!(
        sim.query_column(1, Duration::from_secs(2)),
        Some(Vec::new())
    );
    assert_eq!(
        sim.query_column(9999, Duration::from_secs(2)),
        Some(Vec::new())
    );
    sim.shutdown();
}

#[test]
fn query_column_after_shutdown_does_not_block() {
    let sim = spawn(config(12), Arc::new(AtomicUsize::new(0)));
    let rx = sim.request_column(0);
    sim.shutdown();
    // Either the worker answered before exiting, or the channel closed; both
    // resolve promptly rather than hanging the UI thread.
    let t = Instant::now();
    let _ = rx.recv_timeout(Duration::from_millis(500));
    assert!(t.elapsed() < Duration::from_millis(600));
}
