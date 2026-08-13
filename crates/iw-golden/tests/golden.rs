//! WP12 golden/integration tests: run the real simulation pipeline
//! in-process — the same five processes `iw-headless` and `iw-app` wire up,
//! in the same order — and check determinism, coarse physical plausibility
//! for two fixed seeds, mass conservation, config sanitization, and
//! checkpoint-resume equivalence.
//!
//! Fast tests (level 4, ~130 steps) run on every `cargo test`. The two
//! full-fidelity golden-planet tests (level 6, default phase durations,
//! ~12s each measured in release; see each test's doc comment) are
//! `#[ignore]`d — run them explicitly:
//!
//! ```text
//! cargo test -p iw-golden -- --ignored
//! ```

use std::sync::Arc;

use iw_core::{CheckpointStore, NullProgress, Phase, Planet, PlanetConfig, Process};
use iw_mesh::Mesh;
use iw_sim::{test_util::MemoryStore, Simulation};
use iw_store_postcard::FileStore;
use proptest::prelude::*;

/// The five real processes, in the fixed order the sim requires — mirrors
/// `iw-headless`'s `build_processes` (`crates/iw-headless/src/processes.rs`)
/// and `iw-app`'s, minus the `--skip` escape hatch neither test needs.
fn processes() -> Vec<Box<dyn Process>> {
    vec![
        Box::new(iw_tectonics::TectonicsProcess::default()),
        Box::new(iw_geology::GeologyProcess::default()),
        Box::new(iw_climate::ClimateProcess::default()),
        Box::new(iw_surface::SurfaceProcess::default()),
        Box::new(iw_biomes::BiomeProcess),
    ]
}

/// A small, fast config shared by the non-ignored tests: level 4 (2,562
/// cells) and short phase durations (20/20/8/0.2 Myr => 132 steps total
/// under the default per-phase timesteps) — enough to exercise every
/// process without a multi-second runtime.
fn short_config(seed: u64) -> PlanetConfig {
    let mut config = PlanetConfig {
        seed,
        subdivision_level: 4,
        phase_durations_myr: [20.0, 20.0, 8.0, 0.2],
        ..PlanetConfig::default()
    };
    config.sanitize();
    config
}

/// Full default schedule (200/200/75/2 Myr, 1300 steps) at the golden-planet
/// subdivision level.
fn golden_config(seed: u64) -> PlanetConfig {
    let mut config = PlanetConfig {
        seed,
        subdivision_level: 6,
        ..PlanetConfig::default()
    };
    config.sanitize();
    config
}

fn run_to_completion(config: PlanetConfig, mesh: Arc<Mesh>) -> Planet {
    let store = Arc::new(MemoryStore::new());
    let mut sim = Simulation::new(config, mesh, processes(), store, Arc::new(NullProgress));
    sim.run_headless();
    assert!(sim.is_done(), "simulation did not reach completion");
    sim.planet().clone()
}

/// Bit-pattern view of `elevation_m`, for equality checks that mean exactly
/// "byte-identical" (plain `==` would also call `-0.0` and `0.0` equal).
fn elevation_bits(planet: &Planet) -> Vec<u32> {
    planet.elevation_m.iter().map(|f| f.to_bits()).collect()
}

#[test]
fn end_to_end_determinism() {
    let mesh = Arc::new(Mesh::build(4));
    let config = short_config(42);

    let a = run_to_completion(config.clone(), Arc::clone(&mesh));
    let b = run_to_completion(config, mesh);

    assert_eq!(
        elevation_bits(&a),
        elevation_bits(&b),
        "elevation_m diverged between identical runs"
    );
    assert_eq!(a.biome, b.biome, "biome diverged between identical runs");
    assert_eq!(
        a.plate_id, b.plate_id,
        "plate_id diverged between identical runs"
    );
    assert_eq!(
        a.sea_level_m.to_bits(),
        b.sea_level_m.to_bits(),
        "sea_level_m diverged between identical runs"
    );
}

/// The mass ledger is checked by a `debug_assert!` inside
/// `Simulation::run_processes` (`crates/iw-sim/src/sim.rs`) every step; this
/// test's only job is to run the full five-process pipeline in the dev
/// profile, where `debug_assert!` is live (the workspace root `Cargo.toml`
/// sets `[profile.dev.package."*"] opt-level = 3` so dev builds stay fast
/// without disabling debug assertions). A silent mass leak in any process
/// would panic mid-run; reaching completion is the pass condition.
#[test]
fn mass_ledger_balanced() {
    let mesh = Arc::new(Mesh::build(4));
    let planet = run_to_completion(short_config(7), mesh);
    assert!(planet.step_index > 0);
}

proptest! {
    /// Arbitrary (including out-of-range and negative) `PlanetConfig`
    /// fields, sanitized, must land inside the documented ranges
    /// (`PlanetConfig::sanitize`, DESIGN.md §9.1) and sanitizing twice must
    /// equal sanitizing once.
    #[test]
    fn config_sanitize_roundtrip(
        seed in any::<u64>(),
        subdivision_level in any::<u8>(),
        d0 in -1.0e4f64..1.0e4,
        d1 in -1.0e4f64..1.0e4,
        d2 in -1.0e4f64..1.0e4,
        d3 in -1.0e4f64..1.0e4,
        dt0 in -10.0f64..10.0,
        dt1 in -10.0f64..10.0,
        dt2 in -10.0f64..10.0,
        dt3 in -10.0f64..10.0,
        water_budget in -10.0f64..10.0,
        temperature_offset_c in -100.0f32..100.0,
        axial_tilt_deg in -100.0f32..100.0,
        precip_multiplier in -10.0f32..10.0,
        tectonic_vigor in -10.0f32..10.0,
        hotspot_count in any::<u32>(),
        craton_count in any::<u32>(),
        glacial_intensity in -10.0f32..10.0,
        history_cap_bytes in any::<u64>(),
    ) {
        let mut config = PlanetConfig {
            seed,
            subdivision_level,
            phase_durations_myr: [d0, d1, d2, d3],
            phase_dt_myr: [dt0, dt1, dt2, dt3],
            water_budget,
            temperature_offset_c,
            axial_tilt_deg,
            precip_multiplier,
            tectonic_vigor,
            hotspot_count,
            craton_count,
            glacial_intensity,
            history_cap_bytes,
        };
        config.sanitize();

        prop_assert!((4..=10).contains(&config.subdivision_level));
        for d in config.phase_durations_myr {
            prop_assert!((0.0..=2000.0).contains(&d));
        }
        for (i, dt) in config.phase_dt_myr.iter().enumerate() {
            let lo = if i == 3 { 0.001 } else { 0.05 };
            prop_assert!((lo..=5.0).contains(dt));
        }
        prop_assert!((0.0..=3.0).contains(&config.water_budget));
        prop_assert!((-20.0..=20.0).contains(&config.temperature_offset_c));
        prop_assert!((0.0..=45.0).contains(&config.axial_tilt_deg));
        prop_assert!((0.25..=4.0).contains(&config.precip_multiplier));
        prop_assert!((0.25..=2.0).contains(&config.tectonic_vigor));
        prop_assert!(config.hotspot_count <= 30);
        prop_assert!((4..=30).contains(&config.craton_count));
        prop_assert!((0.0..=2.0).contains(&config.glacial_intensity));

        let mut twice = config.clone();
        twice.sanitize();
        prop_assert_eq!(config, twice, "sanitize is not idempotent");
    }
}

/// Run to completion while checkpointing (`FileStore` in a tempdir, exactly
/// like `iw-headless gen`), then start a *second* `Simulation` and resume it
/// from the checkpoint written when Drift completed
/// (`rerun_from_phase(Refinement, ..)` loads the previous phase's tag — see
/// `Simulation::rerun_from_phase`), run that to completion too, and check
/// the final elevation matches the original run bit-for-bit. Exercises both
/// `FileStore` round-tripping and the "resume reproduces a straight-through
/// run" determinism guarantee documented in `iw-tectonics`'s crate docs.
#[test]
fn checkpoint_resume_matches() {
    let mesh = Arc::new(Mesh::build(4));
    let config = short_config(99);

    let dir = tempfile::tempdir().expect("tempdir");
    let store: Arc<dyn CheckpointStore> =
        Arc::new(FileStore::new(dir.path().to_path_buf()).expect("open FileStore"));

    let mut original = Simulation::new(
        config.clone(),
        Arc::clone(&mesh),
        processes(),
        Arc::clone(&store),
        Arc::new(NullProgress),
    );
    original.run_headless();
    assert!(original.is_done());
    let original_elevation = elevation_bits(original.planet());

    let mut resumed = Simulation::new(
        config.clone(),
        Arc::clone(&mesh),
        processes(),
        Arc::clone(&store),
        Arc::new(NullProgress),
    );
    resumed
        .rerun_from_phase(Phase::Refinement, config)
        .expect("resume from the drift-phase checkpoint");
    resumed.run_headless();
    assert!(resumed.is_done());

    assert_eq!(
        original_elevation,
        elevation_bits(resumed.planet()),
        "elevation_m after checkpoint resume diverged from the original run"
    );
}

/// Area-weighted land fraction (`elevation_m >= sea_level_m`), matching
/// `iw-headless`'s `compute_summary`.
fn land_fraction(planet: &Planet, mesh: &Mesh) -> f64 {
    let mut land = 0.0f64;
    let mut total = 0.0f64;
    for (&e, &area) in planet.elevation_m.iter().zip(mesh.areas_km2.iter()) {
        total += area as f64;
        if e >= planet.sea_level_m {
            land += area as f64;
        }
    }
    land / total
}

/// Area fraction of cells classified as `biome`.
fn biome_fraction(planet: &Planet, mesh: &Mesh, biome: iw_core::Biome) -> f64 {
    let mut area = 0.0f64;
    let mut total = 0.0f64;
    for (&b, &a) in planet.biome.iter().zip(mesh.areas_km2.iter()) {
        total += a as f64;
        if b == biome {
            area += a as f64;
        }
    }
    area / total
}

/// Loose stat-range checks shared by both golden-planet tests. Bands are
/// deliberately wide (DESIGN.md §13: "ranges chosen loosely enough to allow
/// intentional model changes") — see each assertion's comment for the
/// actual value measured on 2026-08-13 via:
/// `cargo run --release -p iw-headless -- gen --seed <42|1337> --level 6`.
fn assert_golden_stats(planet: &Planet, mesh: &Mesh) {
    let land = land_fraction(planet, mesh);
    assert!(
        (0.15..0.45).contains(&land),
        "land_fraction {land} outside 0.15..0.45"
    ); // measured: 0.2695 (seed 42), 0.2775 (seed 1337)

    let plate_count = planet
        .plate_id
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert!(
        (3..30).contains(&plate_count),
        "plate_count {plate_count} outside 3..30"
    ); // measured: 6 (both seeds)

    assert!(
        (-500.0..500.0).contains(&planet.sea_level_m),
        "sea_level_m {} outside -500..500",
        planet.sea_level_m
    ); // measured: 15.5 m (seed 42), -28.0 m (seed 1337)

    let elevation_max = planet
        .elevation_m
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (1500.0..9000.0).contains(&elevation_max),
        "elevation max {elevation_max} outside 1500..9000"
    ); // measured: 4742 m (seed 42), 4368 m (seed 1337)

    let elevation_min = planet
        .elevation_m
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    assert!(
        (-11000.0..-6000.0).contains(&elevation_min),
        "elevation min {elevation_min} outside -11000..-6000"
    ); // measured: -8733 m (seed 42), -8571 m (seed 1337)

    let ice = biome_fraction(planet, mesh, iw_core::Biome::IceSheet);
    assert!(
        (0.01..0.12).contains(&ice),
        "IceSheet fraction {ice} outside 0.01..0.12"
    ); // measured: 0.0561 (seed 42), 0.0554 (seed 1337)

    let terrestrial_present = iw_core::Biome::TERRESTRIAL
        .iter()
        .filter(|&&b| biome_fraction(planet, mesh, b) > 0.001)
        .count();
    assert!(
        terrestrial_present >= 8,
        "only {terrestrial_present} terrestrial biomes above 0.1% of surface"
    ); // measured: 12 (seed 42), 13 (seed 1337)

    let mut has_sedimentary = false;
    let mut has_igneous = false;
    let mut has_metamorphic = false;
    for cell in 0..planet.n_cells() as u32 {
        if let Some(rock) = planet.columns.top_rock(cell) {
            has_sedimentary |= rock.is_sedimentary();
            has_igneous |= rock.is_igneous();
            has_metamorphic |= rock.is_metamorphic();
        }
    }
    assert!(has_sedimentary, "no sedimentary rock on any column top");
    assert!(has_igneous, "no igneous rock on any column top");
    assert!(has_metamorphic, "no metamorphic rock on any column top");
    // measured (seed 42): sedimentary e.g. Shale 17.0%, igneous e.g. Basalt
    // 29.1%, metamorphic e.g. Slate 0.08% + Schist 0.03% + Quartzite 0.002%.
}

/// Full default phase schedule (200/200/75/2 Myr) at subdivision level 6
/// (40,962 cells), seed 42. Measured runtime: ~12.4s
/// (`cargo run --release -p iw-headless -- gen --seed 42 --level 6`); the
/// dev-profile `cargo test --ignored` run is comparable because the
/// workspace overrides `opt-level = 3` for all packages in the dev profile.
#[test]
#[ignore = "slow (~12s): full default phase schedule at level 6"]
fn golden_planet_seed_42() {
    let mesh = Arc::new(Mesh::build(6));
    let planet = run_to_completion(golden_config(42), Arc::clone(&mesh));
    assert_golden_stats(&planet, &mesh);
}

/// Same as `golden_planet_seed_42` with seed 1337. Measured runtime: ~12.2s.
#[test]
#[ignore = "slow (~12s): full default phase schedule at level 6"]
fn golden_planet_seed_1337() {
    let mesh = Arc::new(Mesh::build(6));
    let planet = run_to_completion(golden_config(1337), Arc::clone(&mesh));
    assert_golden_stats(&planet, &mesh);
}
