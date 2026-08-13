//! Same seed, same config, same planet — including across a checkpoint resume.

use std::sync::Arc;

use iw_core::{CheckpointStore, NullProgress, Phase, Planet, PlanetConfig, Process};
use iw_sim::test_util::{
    ring_mesh, tiny_mesh, MemoryStore, NoopProcess, RngNoiseProcess, MIN_LEVEL,
};
use iw_sim::{phase_tag, Simulation};

fn config(seed: u64) -> PlanetConfig {
    PlanetConfig {
        seed,
        subdivision_level: MIN_LEVEL,
        phase_durations_myr: [10.0, 10.0, 5.0, 0.5],
        phase_dt_myr: [1.0, 0.5, 0.25, 0.05],
        ..PlanetConfig::default()
    }
}

fn processes() -> Vec<Box<dyn Process>> {
    vec![
        Box::new(RngNoiseProcess::new("tectonics-ish")),
        Box::new(NoopProcess),
        Box::new(RngNoiseProcess::new("surface-ish")),
    ]
}

/// FNV-1a over the raw bits of the elevation field.
fn elevation_hash(planet: &Planet) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for e in &planet.elevation_m {
        for b in e.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn run_full(seed: u64) -> (Simulation, u64) {
    let mut sim = Simulation::new(
        config(seed),
        tiny_mesh(),
        processes(),
        Arc::new(MemoryStore::new()),
        Arc::new(NullProgress),
    );
    sim.run_headless();
    let hash = elevation_hash(sim.planet());
    (sim, hash)
}

#[test]
fn two_runs_agree_and_actually_moved_the_planet() {
    let (a, ha) = run_full(42);
    let (_b, hb) = run_full(42);
    assert_eq!(ha, hb, "same seed must give the same elevation field");
    let (_c, hc) = run_full(43);
    assert_ne!(ha, hc, "a different seed must give a different planet");
    assert!(
        a.planet().elevation_m.iter().any(|e| *e != 0.0),
        "the dummy process never moved anything, so the test proves nothing"
    );
}

#[test]
fn process_order_does_not_change_a_process_stream() {
    // Streams are derived from the process name, not its position.
    let mesh = tiny_mesh();
    let mut forward = Simulation::new(
        config(9),
        Arc::clone(&mesh),
        vec![
            Box::new(RngNoiseProcess::new("a")),
            Box::new(RngNoiseProcess::new("b")),
        ],
        Arc::new(MemoryStore::new()),
        Arc::new(NullProgress),
    );
    let mut reversed = Simulation::new(
        config(9),
        mesh,
        vec![
            Box::new(RngNoiseProcess::new("b")),
            Box::new(RngNoiseProcess::new("a")),
        ],
        Arc::new(MemoryStore::new()),
        Arc::new(NullProgress),
    );
    forward.run_headless();
    reversed.run_headless();
    // Both processes only add to elevation, so reordering may only shuffle
    // float rounding, never draw different numbers.
    let worst = forward
        .planet()
        .elevation_m
        .iter()
        .zip(&reversed.planet().elevation_m)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1.0e-2,
        "streams moved with process order: {worst} m"
    );
    assert!(forward.planet().elevation_m.iter().any(|e| e.abs() > 1.0));
}

#[test]
fn resume_mid_phase_matches_an_uninterrupted_run() {
    let (_, expected) = run_full(42);

    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(
        config(42),
        tiny_mesh(),
        processes(),
        store.clone(),
        Arc::new(NullProgress),
    );
    // Stop 27 steps in: mid-Drift, nowhere near a phase boundary.
    for _ in 0..27 {
        assert!(sim.step_once());
    }
    assert_eq!(sim.current_phase(), Some(Phase::Drift));
    store.save("mid-run", sim.planet()).unwrap();
    drop(sim);

    let mut resumed = Simulation::new(
        config(42),
        tiny_mesh(),
        processes(),
        store.clone(),
        Arc::new(NullProgress),
    );
    resumed.resume_from(store.load("mid-run").unwrap()).unwrap();
    assert_eq!(resumed.step_index(), 27);
    assert_eq!(resumed.current_phase(), Some(Phase::Drift));
    resumed.run_headless();

    assert_eq!(elevation_hash(resumed.planet()), expected);
}

#[test]
fn rerun_from_phase_boundary_matches_an_uninterrupted_run() {
    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(
        config(42),
        tiny_mesh(),
        processes(),
        store.clone(),
        Arc::new(NullProgress),
    );
    sim.run_headless();
    let expected = elevation_hash(sim.planet());
    assert!(store
        .list()
        .unwrap()
        .contains(&phase_tag(Phase::Refinement)));

    // Re-run the last phase from the Refinement checkpoint with the same config.
    sim.rerun_from_phase(Phase::RecentPast, config(42)).unwrap();
    assert_eq!(sim.current_phase(), Some(Phase::RecentPast));
    sim.run_headless();
    assert_eq!(elevation_hash(sim.planet()), expected);
}

#[test]
fn rerun_of_the_first_phase_is_a_regenerate() {
    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(
        config(42),
        tiny_mesh(),
        processes(),
        store.clone(),
        Arc::new(NullProgress),
    );
    sim.run_headless();
    let expected = elevation_hash(sim.planet());
    sim.rerun_from_phase(Phase::CrustalFormation, config(42))
        .unwrap();
    assert_eq!(sim.step_index(), 0);
    assert_eq!(sim.planet().time_myr, 0.0);
    sim.run_headless();
    assert_eq!(elevation_hash(sim.planet()), expected);
}

#[test]
fn rerun_rejects_a_budget_change_and_a_missing_checkpoint() {
    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(
        config(42),
        tiny_mesh(),
        processes(),
        store.clone(),
        Arc::new(NullProgress),
    );
    // No checkpoints written yet.
    let err = sim.rerun_from_phase(Phase::Drift, config(42)).unwrap_err();
    assert!(
        format!("{err:#}").contains("phase-crustal_formation"),
        "{err:#}"
    );

    // With a checkpoint present, a cell-budget change is refused: it would
    // invalidate every per-cell array.
    store.save("phase-crustal_formation", sim.planet()).unwrap();
    let mut wrong = config(42);
    wrong.cell_budget += 54_321;
    let err = sim.rerun_from_phase(Phase::Drift, wrong).unwrap_err();
    assert!(format!("{err:#}").contains("cell budget"), "{err:#}");
}

#[test]
fn resume_rejects_a_planet_from_another_mesh() {
    let store = Arc::new(MemoryStore::new());
    let big = Simulation::new(
        config(1),
        ring_mesh(9),
        vec![],
        store.clone(),
        Arc::new(NullProgress),
    );
    store.save("wide", big.planet()).unwrap();
    let mut small = Simulation::new(
        config(1),
        tiny_mesh(),
        vec![],
        store.clone(),
        Arc::new(NullProgress),
    );
    let err = small.resume_from(store.load("wide").unwrap()).unwrap_err();
    assert!(format!("{err:#}").contains("cells"), "{err:#}");
}
