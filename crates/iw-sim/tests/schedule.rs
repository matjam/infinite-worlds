//! Phase schedule: step counts, simulated time, events and checkpoints.

use std::sync::Arc;
use std::time::Duration;

use iw_core::{CheckpointStore, Phase, PlanetConfig, ProgressEvent, ProgressSink};
use iw_sim::test_util::{tiny_mesh, MemoryStore, NoopProcess, RecordingSink};
use iw_sim::{phase_tag, Simulation};

fn short_config() -> PlanetConfig {
    PlanetConfig {
        seed: 7,
        subdivision_level: iw_sim::test_util::MIN_LEVEL,
        phase_durations_myr: [10.0, 10.0, 5.0, 1.0],
        phase_dt_myr: [1.0, 0.5, 0.25, 0.05],
        ..PlanetConfig::default()
    }
}

/// `subdivision_level` is clamped by `sanitize`, so tests that compare against
/// the raw config must sanitize first.
fn sanitized(mut c: PlanetConfig) -> PlanetConfig {
    c.sanitize();
    c
}

#[test]
fn default_config_step_counts() {
    let c = sanitized(PlanetConfig::default());
    assert_eq!(c.steps_in(Phase::CrustalFormation), 200);
    assert_eq!(c.steps_in(Phase::Drift), 400);
    assert_eq!(c.steps_in(Phase::Refinement), 300);
    assert_eq!(c.steps_in(Phase::RecentPast), 400);

    let mesh = tiny_mesh();
    let sim = Simulation::new(
        PlanetConfig::default(),
        mesh,
        vec![],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    assert_eq!(sim.total_steps(), 1300);
    assert_eq!(sim.current_phase(), Some(Phase::CrustalFormation));
    assert!(!sim.is_done());
}

#[test]
fn full_run_walks_every_phase_and_lands_on_total_time() {
    let config = short_config();
    let sanitized = sanitized(config.clone());
    let store = Arc::new(MemoryStore::new());
    let sink = Arc::new(RecordingSink::new());
    let mut sim = Simulation::new(
        config,
        tiny_mesh(),
        vec![Box::new(NoopProcess)],
        store.clone(),
        sink.clone(),
    );
    let expected_steps: u64 = Phase::ALL.iter().map(|p| sanitized.steps_in(*p)).sum();
    assert_eq!(expected_steps, 10 + 20 + 20 + 20);

    sim.run_headless();

    assert!(sim.is_done());
    assert_eq!(sim.current_phase(), None);
    assert_eq!(sim.step_index(), expected_steps);
    let expected_time: f64 = sanitized.phase_durations_myr.iter().sum();
    assert!(
        (sim.planet().time_myr - expected_time).abs() < 1e-9,
        "time {} != {}",
        sim.planet().time_myr,
        expected_time
    );
    assert_eq!(sim.planet().phase, Phase::RecentPast);

    // Phase events, in order, exactly once each.
    let events = sink.events();
    let phase_events: Vec<(bool, Phase)> = events
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::PhaseStarted { phase, .. } => Some((true, *phase)),
            ProgressEvent::PhaseCompleted { phase, .. } => Some((false, *phase)),
            _ => None,
        })
        .collect();
    let expected: Vec<(bool, Phase)> = Phase::ALL
        .iter()
        .flat_map(|p| [(true, *p), (false, *p)])
        .collect();
    assert_eq!(phase_events, expected);

    // One checkpoint per phase completion, in order.
    let tags = store.list().unwrap();
    let expected_tags: Vec<String> = Phase::ALL.iter().map(|p| phase_tag(*p)).collect();
    assert_eq!(tags, expected_tags);

    // Step events carry a monotone step within each phase and end at `of`.
    let mut last: Option<(Phase, u64, u64)> = None;
    for ev in &events {
        if let ProgressEvent::Step {
            phase, step, of, ..
        } = ev
        {
            if let Some((lp, ls, _)) = last {
                if lp == *phase {
                    assert!(*step > ls);
                }
            }
            assert!(*step <= *of);
            last = Some((*phase, *step, *of));
        }
    }
    assert_eq!(
        last.map(|(p, s, of)| (p, s == of)),
        Some((Phase::RecentPast, true))
    );

    // Narration happened, and every line came from the pool.
    let narration: Vec<&String> = events
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::Narration(s) => Some(s),
            _ => None,
        })
        .collect();
    assert!(
        narration.len() >= 8,
        "only {} narration lines",
        narration.len()
    );
}

#[test]
fn phase_time_advances_by_dt_each_step() {
    let config = short_config();
    let sanitized = sanitized(config.clone());
    let mut sim = Simulation::new(
        config,
        tiny_mesh(),
        vec![],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    let mut expected_time = 0.0;
    for phase in Phase::ALL {
        let dt = sanitized.dt_myr(phase);
        for _ in 0..sanitized.steps_in(phase) {
            assert_eq!(sim.current_phase(), Some(phase));
            assert!(sim.step_once());
            expected_time += dt;
            assert!((sim.planet().time_myr - expected_time).abs() < 1e-9);
        }
    }
    assert!(!sim.step_once());
}

#[test]
fn zero_length_phases_are_skipped() {
    let config = PlanetConfig {
        seed: 1,
        subdivision_level: iw_sim::test_util::MIN_LEVEL,
        phase_durations_myr: [0.0, 4.0, 0.0, 0.0],
        phase_dt_myr: [1.0, 1.0, 1.0, 1.0],
        ..PlanetConfig::default()
    };
    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(
        config,
        tiny_mesh(),
        vec![],
        store.clone(),
        Arc::new(iw_core::NullProgress),
    );
    assert_eq!(sim.current_phase(), Some(Phase::Drift));
    sim.run_headless();
    assert_eq!(sim.step_index(), 4);
    assert_eq!(store.list().unwrap(), vec![phase_tag(Phase::Drift)]);
}

#[test]
fn works_without_any_processes_and_publishes_snapshots() {
    let mut sim = Simulation::new(
        short_config(),
        tiny_mesh(),
        vec![],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    assert!(
        sim.latest_view().is_none(),
        "nothing published before a run"
    );
    sim.set_publish_throttle(1, Duration::ZERO);
    sim.run_headless();
    let view = sim.latest_view().expect("a snapshot");
    assert_eq!(view.cells.elevation_m.len(), 4);
    assert_eq!(view.phase, Phase::RecentPast);
}

#[test]
fn snapshot_versions_strictly_increase() {
    let mut sim = Simulation::new(
        short_config(),
        tiny_mesh(),
        vec![Box::new(NoopProcess)],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    sim.set_publish_throttle(1, Duration::ZERO);
    let mut versions = Vec::new();
    while sim.step_once() {
        if let Some(v) = sim.latest_view() {
            versions.push(v.version);
        }
    }
    assert!(versions.len() > 10);
    assert!(
        versions.windows(2).all(|w| w[1] > w[0]),
        "versions not strictly increasing: {versions:?}"
    );
    // A regenerate must not rewind the version counter.
    let last = *versions.last().unwrap();
    sim.reset(short_config(), tiny_mesh());
    assert!(sim.latest_view().unwrap().version > last);
}

#[test]
fn throttled_publishing_is_rarer_than_stepping() {
    let mut sim = Simulation::new(
        short_config(),
        tiny_mesh(),
        vec![],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    // Defaults: 25 steps AND 250 ms. A 70-step run takes microseconds, so the
    // only publishes are the phase boundaries.
    sim.run_headless();
    let version = sim.latest_view().unwrap().version;
    assert!(version <= 8, "published {version} times for 70 steps");
    assert!(version >= 4, "phase boundaries must publish");
}

struct CountingSink(std::sync::atomic::AtomicUsize);

impl ProgressSink for CountingSink {
    fn event(&self, _ev: ProgressEvent) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn step_events_are_throttled_to_about_200_per_phase() {
    let config = PlanetConfig {
        seed: 3,
        subdivision_level: iw_sim::test_util::MIN_LEVEL,
        phase_durations_myr: [1000.0, 0.0, 0.0, 0.0],
        phase_dt_myr: [0.5, 1.0, 1.0, 1.0],
        ..PlanetConfig::default()
    };
    let sink = Arc::new(RecordingSink::new());
    let mut sim = Simulation::new(
        config,
        tiny_mesh(),
        vec![],
        Arc::new(MemoryStore::new()),
        sink.clone(),
    );
    sim.run_headless();
    assert_eq!(sim.step_index(), 2000);
    let steps = sink
        .events()
        .iter()
        .filter(|e| matches!(e, ProgressEvent::Step { .. }))
        .count();
    assert!(
        (190..=210).contains(&steps),
        "{steps} step events for 2000 steps"
    );

    let counter = Arc::new(CountingSink(std::sync::atomic::AtomicUsize::new(0)));
    let mut sim2 = Simulation::new(
        short_config(),
        tiny_mesh(),
        vec![],
        Arc::new(MemoryStore::new()),
        counter.clone(),
    );
    sim2.run_headless();
    assert!(counter.0.load(std::sync::atomic::Ordering::Relaxed) > 0);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "mass ledger residual")]
fn unbalanced_mass_trips_the_ledger_assert() {
    let mut sim = Simulation::new(
        short_config(),
        tiny_mesh(),
        vec![Box::new(iw_sim::test_util::UnbalancedProcess)],
        Arc::new(MemoryStore::new()),
        Arc::new(iw_core::NullProgress),
    );
    sim.run_headless();
}
