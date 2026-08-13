//! Acceptance tests for biome classification (IMPLEMENTATION_PLAN.md §3 WP8).

use std::collections::HashMap;

use glam::Vec3;
use iw_biomes::*;
use iw_core::{rng_for, Biome, MassLedger, NullProgress, Planet, PlanetConfig, Process, StepCtx};
use iw_mesh::Mesh;

fn config(level: u8) -> PlanetConfig {
    PlanetConfig {
        subdivision_level: level,
        ..PlanetConfig::default()
    }
}

fn dir_at(lat_deg: f32, lon_deg: f32) -> Vec3 {
    let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
    Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin())
}

/// All-ocean planet: uniform abyssal floor, sea level 0.
fn aquaplanet(level: u8) -> (Planet, Mesh) {
    let mesh = Mesh::build(level);
    let mut planet = Planet::new(config(level), mesh.n_cells());
    planet.elevation_m.fill(-4000.0);
    planet.sea_level_m = 0.0;
    (planet, mesh)
}

/// All-ocean planet with a single elevated cell at `(lat_deg, lon_deg)`, so
/// that cell's neighbors are all ocean (distance-to-ocean == 1).
fn island_planet(level: u8, lat_deg: f32, lon_deg: f32, elev_m: f32) -> (Planet, Mesh, u32) {
    let (mut planet, mesh) = aquaplanet(level);
    let cell = mesh.cell_at(dir_at(lat_deg, lon_deg));
    planet.elevation_m[cell as usize] = elev_m;
    (planet, mesh, cell)
}

/// Planet with no ocean anywhere: every cell above sea level. Distance to
/// ocean therefore saturates at its maximum everywhere.
fn all_land_planet(level: u8, elev_m: f32) -> (Planet, Mesh) {
    let mesh = Mesh::build(level);
    let mut planet = Planet::new(config(level), mesh.n_cells());
    planet.elevation_m.fill(elev_m);
    planet.sea_level_m = 0.0;
    (planet, mesh)
}

fn run(planet: &mut Planet, mesh: &Mesh, process: &mut BiomeProcess, steps: usize) {
    let progress = NullProgress;
    for _ in 0..steps {
        let mut ledger = MassLedger::default();
        let mut ctx = StepCtx {
            rng: rng_for(planet.config.seed, "biomes", planet.step_index),
            progress: &progress,
            ledger: &mut ledger,
        };
        process.step(planet, mesh, 1.0, &mut ctx);
        planet.step_index += 1;
    }
}

// --- water / ice -----------------------------------------------------------

#[test]
fn aquaplanet_is_all_ocean() {
    let (mut planet, mesh) = aquaplanet(4);
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert!(planet.biome.iter().all(|b| *b == Biome::Ocean));
}

#[test]
fn frozen_cell_is_ice_sheet() {
    let (mut planet, mesh) = aquaplanet(5);
    let cell = mesh.cell_at(dir_at(80.0, 0.0)) as usize;
    // Ice over open ocean must still classify as ice sheet, ahead of Ocean.
    planet.ice_thickness_m[cell] = 50.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[cell], Biome::IceSheet);
}

#[test]
fn lake_cell_is_lake() {
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    let cell = mesh.cell_at(dir_at(0.0, 0.0)) as usize;
    planet.lake_depth_m[cell] = 2.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[cell], Biome::Lake);
}

// --- Whittaker core --------------------------------------------------------

#[test]
fn hot_wet_land_is_tropical_moist_broadleaf() {
    let (mut planet, mesh, cell) = island_planet(5, 5.0, 0.0, 300.0);
    let c = cell as usize;
    planet.temperature_c[c] = 27.0;
    planet.precip_mm_yr[c] = 2000.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[c], Biome::TropicalMoistBroadleaf);
}

#[test]
fn hot_dry_land_is_desert() {
    let (mut planet, mesh, cell) = island_planet(5, 5.0, 0.0, 300.0);
    let c = cell as usize;
    planet.temperature_c[c] = 30.0;
    planet.precip_mm_yr[c] = 100.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[c], Biome::Desert);
}

#[test]
fn cold_wet_land_is_boreal_taiga() {
    // lat 50 with the default axial tilt gives enough seasonal amplitude to
    // keep the summer extreme above the tundra line (>= 10 C) even though
    // the annual mean sits in the boreal band, isolating the Whittaker
    // boreal branch from the standalone tundra override.
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    let cell = mesh.cell_at(dir_at(50.0, 0.0)) as usize;
    planet.temperature_c[cell] = 0.0;
    planet.precip_mm_yr[cell] = 500.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[cell], Biome::BorealTaiga);
}

#[test]
fn cold_summer_is_tundra() {
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    // Low latitude keeps seasonal amplitude small, so a very cold annual
    // mean still isn't enough to raise the summer extreme to 10 C.
    let cell = mesh.cell_at(dir_at(20.0, 0.0)) as usize;
    planet.temperature_c[cell] = -8.0;
    planet.precip_mm_yr[cell] = 400.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[cell], Biome::Tundra);
}

// --- overrides ---------------------------------------------------------

#[test]
fn montane_override_fires_at_3000m_tropical() {
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    let cell = mesh.cell_at(dir_at(10.0, 0.0)) as usize;
    planet.elevation_m[cell] = 3000.0; // 2500 m above the 500 m plain, well
                                       // above the 500 m sea-level baseline.
    planet.temperature_c[cell] = 25.0;
    planet.precip_mm_yr[cell] = 1000.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[cell], Biome::MontaneGrassland);
}

#[test]
fn mangrove_on_tropical_coast() {
    let (mut planet, mesh, cell) = island_planet(5, 2.0, 0.0, 20.0);
    let c = cell as usize;
    planet.temperature_c[c] = 28.0;
    planet.precip_mm_yr[c] = 1200.0;
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[c], Biome::Mangrove);
}

#[test]
fn flooded_grassland_near_lake() {
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    let lake = mesh.cell_at(dir_at(0.0, 0.0));
    let flooded = mesh.neighbors_of(lake)[0] as usize;
    planet.lake_depth_m[lake as usize] = 2.0;
    planet.elevation_m[flooded] = 100.0; // < 200 m lowland
    planet.temperature_c[flooded] = 15.0; // warm
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(planet.biome[flooded], Biome::FloodedGrassland);
}

// --- process contract ----------------------------------------------------

#[test]
fn process_contract() {
    let process = BiomeProcess;
    assert_eq!(process.name(), "biomes");
    let boxed: Box<dyn Process> = Box::new(BiomeProcess::new());
    assert_eq!(boxed.name(), "biomes");
}

#[test]
fn writes_only_biome_field() {
    let (mut planet, mesh) = all_land_planet(5, 500.0);
    let before = planet.clone();
    run(&mut planet, &mesh, &mut BiomeProcess::new(), 3);

    assert_eq!(planet.elevation_m, before.elevation_m);
    assert_eq!(planet.sea_level_m, before.sea_level_m);
    assert_eq!(planet.temperature_c, before.temperature_c);
    assert_eq!(planet.precip_mm_yr, before.precip_mm_yr);
    assert_eq!(planet.ice_thickness_m, before.ice_thickness_m);
    assert_eq!(planet.lake_depth_m, before.lake_depth_m);
    assert_eq!(planet.water_flux_m3_yr, before.water_flux_m3_yr);
    assert_eq!(planet.crust_thickness_m, before.crust_thickness_m);
    assert_eq!(planet.plate_id, before.plate_id);
}

// --- reachability sweep ----------------------------------------------------

/// Sweep a synthetic T x P x (elevation, latitude, coastal distance, lake
/// adjacency, high-flux, seasonal amplitude) grid through the pure
/// classifier and assert every one of the 14 terrestrial biomes appears at
/// least once. Prints per-biome counts (run with `--nocapture` to see them).
#[test]
fn all_fourteen_terrestrial_biomes_reachable() {
    let temps = [
        -15.0, -8.0, -3.0, 0.0, 3.0, 6.0, 10.0, 15.0, 18.0, 22.0, 26.0, 30.0,
    ];
    let precips = [
        50.0, 150.0, 300.0, 450.0, 700.0, 950.0, 1300.0, 1700.0, 2000.0,
    ];
    let elevs = [10.0f32, 100.0, 300.0, 900.0, 1500.0, 3000.0];
    let lats_deg = [2.0f32, 15.0, 25.0, 32.0, 40.0, 55.0, 70.0, 85.0];
    let dists: [u8; 4] = [0, 1, 3, 6];
    let amps = [2.0f32, 5.0, 9.0, 14.0];

    let mut counts: HashMap<Biome, u64> = HashMap::new();
    let mut total = 0u64;

    for &t in &temps {
        for &p in &precips {
            for &elev in &elevs {
                for &lat in &lats_deg {
                    for &dist in &dists {
                        for &amp in &amps {
                            for &lake_adjacent in &[false, true] {
                                for &water_flux_high in &[false, true] {
                                    let inputs = CellInputs {
                                        temperature_c: t,
                                        summer_c: t + amp,
                                        winter_c: t - amp,
                                        precip_mm_yr: p,
                                        elevation_above_sea_m: elev,
                                        lat_rad: lat.to_radians(),
                                        dist_to_ocean: dist,
                                        lake_adjacent,
                                        water_flux_high,
                                        ice_thickness_m: 0.0,
                                        is_ocean: false,
                                        lake_depth_m: 0.0,
                                    };
                                    let b = classify_cell(&inputs);
                                    *counts.entry(b).or_insert(0) += 1;
                                    total += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("biome sweep: {total} combinations");
    for b in Biome::TERRESTRIAL {
        println!("  {:32} {}", b.name(), counts.get(&b).copied().unwrap_or(0));
    }
    for b in Biome::TERRESTRIAL {
        assert!(
            counts.get(&b).copied().unwrap_or(0) > 0,
            "biome {:?} unreachable in sweep",
            b
        );
    }
}

// --- determinism and statelessness ------------------------------------

fn fingerprint(planet: &Planet) -> Vec<u8> {
    planet.biome.iter().map(|b| *b as u8).collect()
}

/// Land band spanning a range of latitudes/longitudes with varied elevation,
/// temperature and precipitation, so the determinism/statelessness/timing
/// tests exercise a realistic mix of every override and Whittaker branch.
fn varied_planet(level: u8) -> (Planet, Mesh) {
    let mesh = Mesh::build(level);
    let mut planet = Planet::new(config(level), mesh.n_cells());
    for i in 0..mesh.n_cells() {
        let (lat, lon) = (
            mesh.latlon[i][0].to_degrees(),
            mesh.latlon[i][1].to_degrees(),
        );
        let land = (lon.abs() < 120.0) && (lat.abs() < 75.0);
        if land {
            let lat_frac = (lat.abs() / 75.0).clamp(0.0, 1.0);
            planet.elevation_m[i] = 50.0 + 3500.0 * ((i as f32 * 0.618).fract());
            planet.temperature_c[i] = 28.0 - 50.0 * lat_frac;
            planet.precip_mm_yr[i] = 200.0 + 1800.0 * ((i as f32 * 0.381).fract());
            if i % 97 == 0 {
                planet.lake_depth_m[i] = 1.5;
            }
            if i % 53 == 0 {
                planet.water_flux_m3_yr[i] = 5.0e6;
            }
            if lat.abs() > 65.0 && i % 5 == 0 {
                planet.ice_thickness_m[i] = 20.0;
            }
        } else {
            planet.elevation_m[i] = -3000.0;
        }
    }
    planet.sea_level_m = 0.0;
    (planet, mesh)
}

#[test]
fn deterministic_across_runs() {
    let (mut a, mesh) = varied_planet(5);
    let mut b = a.clone();
    run(&mut a, &mesh, &mut BiomeProcess::new(), 3);
    run(&mut b, &mesh, &mut BiomeProcess::new(), 3);
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn stateless_across_process_instances() {
    let (mut planet, mesh) = varied_planet(5);
    let mut warm = BiomeProcess::new();
    run(&mut planet, &mesh, &mut warm, 2);

    // Mutate the water state mid-run so a stale cache would show up.
    for i in 0..mesh.n_cells() {
        if i % 31 == 0 {
            planet.lake_depth_m[i] = 3.0;
        }
    }
    let mut cold_start = planet.clone();
    run(&mut planet, &mesh, &mut warm, 1);
    run(&mut cold_start, &mesh, &mut BiomeProcess::new(), 1);
    assert_eq!(fingerprint(&planet), fingerprint(&cold_start));
}

// --- biome_color -----------------------------------------------------------

#[test]
fn biome_colors_are_all_distinct() {
    let all = [
        Biome::Unclassified,
        Biome::TropicalMoistBroadleaf,
        Biome::TropicalDryBroadleaf,
        Biome::TropicalConifer,
        Biome::TemperateBroadleaf,
        Biome::TemperateConifer,
        Biome::BorealTaiga,
        Biome::TropicalGrassland,
        Biome::TemperateGrassland,
        Biome::FloodedGrassland,
        Biome::MontaneGrassland,
        Biome::Tundra,
        Biome::Mediterranean,
        Biome::Desert,
        Biome::Mangrove,
        Biome::Ocean,
        Biome::Lake,
        Biome::IceSheet,
    ];
    for i in 0..all.len() {
        for j in (i + 1)..all.len() {
            assert_ne!(
                biome_color(all[i]),
                biome_color(all[j]),
                "{:?} and {:?} share a color",
                all[i],
                all[j]
            );
        }
    }
}

// --- budget ------------------------------------------------------------

#[test]
#[ignore = "timing; run with --ignored --release"]
fn step_budget_level_6() {
    use std::time::Instant;
    let (mut planet, mesh) = varied_planet(6);
    let mut process = BiomeProcess::new();
    run(&mut planet, &mesh, &mut process, 2); // warm up

    let start = Instant::now();
    let steps = 20;
    run(&mut planet, &mesh, &mut process, steps);
    let per_step = start.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    println!("biomes step at level 6: {per_step:.3} ms");
    assert!(per_step < 5.0, "biomes step took {per_step:.3} ms");
}
