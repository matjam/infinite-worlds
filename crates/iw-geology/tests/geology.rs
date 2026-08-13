//! Acceptance tests for WP5 (iw-geology): isostasy anchors, the hypsometric
//! sea-level solve, the metamorphic transition table, igneous emplacement,
//! determinism and process statelessness.

use iw_core::planet::cell_flags;
use iw_core::{
    rng_for, CrustType, MassLedger, MetamorphicGrade, NullProgress, Planet, PlanetConfig, Process,
    RockType, StepCtx,
};
use iw_geology::{isostasy, metamorphism, sea_level, GeologyProcess, METAMORPHISM_INTERVAL_STEPS};
use iw_mesh::Mesh;

const LEVEL: u8 = 4;

fn config() -> PlanetConfig {
    PlanetConfig {
        subdivision_level: LEVEL,
        ..Default::default()
    }
}

fn uniform_planet(mesh: &Mesh, thickness_m: f32, density: f32, ct: CrustType) -> Planet {
    let mut p = Planet::new(config(), mesh.n_cells());
    p.crust_thickness_m.fill(thickness_m);
    p.crust_density_kg_m3.fill(density);
    p.crust_type.fill(ct);
    p
}

/// One step, standing in for what `iw-sim` does around a process.
fn step(proc: &mut GeologyProcess, planet: &mut Planet, mesh: &Mesh, dt_myr: f64) -> MassLedger {
    let mut ledger = MassLedger::default();
    {
        let mut ctx = StepCtx {
            rng: rng_for(planet.config.seed, "geology", planet.step_index),
            progress: &NullProgress,
            ledger: &mut ledger,
        };
        proc.step(planet, mesh, dt_myr, &mut ctx);
    }
    planet.step_index += 1;
    planet.time_myr += dt_myr;
    ledger
}

/// Anchor band: the stated target range widened by 30% on each side.
#[track_caller]
fn assert_band(what: &str, v: f32, lo: f32, hi: f32) {
    let lo_e = lo - 0.3 * lo.abs();
    let hi_e = hi + 0.3 * hi.abs();
    assert!(
        v >= lo_e && v <= hi_e,
        "{what}: {v:.0} m outside {lo_e:.0}..{hi_e:.0} (target {lo:.0}..{hi:.0})"
    );
}

/// Elevation of a uniform world after a full step (flexure is a no-op on a
/// uniform field, so this isolates the local solve).
fn uniform_elevation(mesh: &Mesh, thickness_m: f32, density: f32, ct: CrustType) -> f32 {
    let mut p = uniform_planet(mesh, thickness_m, density, ct);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, mesh, 1.0);
    p.elevation_m[0]
}

#[test]
fn isostasy_anchors() {
    let mesh = Mesh::build(LEVEL);
    let cont = uniform_elevation(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    assert_band("35 km / 2700 continental", cont, 800.0, 800.0);

    let fresh = uniform_elevation(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    assert_band("7 km / 3000 fresh ocean", fresh, -3000.0, -2400.0);

    // Calibration: the abyssal anchor was moved from -5564 m to -4200 m. This
    // model's ocean floor has no age spread (rigid plates, no advection), so it
    // all sits at this one depth; anchoring it at Earth's *mean* ocean depth
    // rather than Earth's abyssal-plain depth is what puts the solved sea level
    // near the geoid. See `isostasy::ANCHOR_OCEAN_OLD_ELEV_M`.
    let old = uniform_elevation(&mesh, 7_000.0, 3300.0, CrustType::Oceanic);
    assert_band("7 km / 3300 old ocean", old, -4500.0, -3900.0);

    let orogen = uniform_elevation(&mesh, 70_000.0, 2750.0, CrustType::Continental);
    assert_band("70 km / 2750 orogen", orogen, 5000.0, 8000.0);
}

#[test]
fn old_ocean_is_deeper_than_young_ocean() {
    let mesh = Mesh::build(LEVEL);
    let young = uniform_elevation(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    let mid = uniform_elevation(&mesh, 7_000.0, 3150.0, CrustType::Oceanic);
    let old = uniform_elevation(&mesh, 7_000.0, 3300.0, CrustType::Oceanic);
    assert!(old < mid && mid < young, "{old} < {mid} < {young}");
    // Calibration: ridge-to-abyss relief is now 1500 m (ridge -2700, abyss
    // -4200), not the >2000 m the pre-calibration anchors gave. The lost relief
    // went into `OCEAN_LID_RESIDUAL_M`, the age-independent part of the lid,
    // because the ocean floor here never gets young again and the whole basin
    // would otherwise sit at the deep end of the ramp.
    assert!(
        young - old > 1200.0,
        "thermal subsidence only {} m",
        young - old
    );
}

#[test]
fn thickened_crust_stands_higher() {
    let mesh = Mesh::build(LEVEL);
    let normal = uniform_elevation(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    let thick = uniform_elevation(&mesh, 55_000.0, 2700.0, CrustType::Continental);
    assert!(thick > normal + 2000.0, "{thick} vs {normal}");
}

#[test]
fn ice_load_depresses_the_surface() {
    let mesh = Mesh::build(LEVEL);
    let bare = uniform_elevation(&mesh, 35_000.0, 2700.0, CrustType::Continental);

    let mut iced = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    iced.ice_thickness_m.fill(3000.0);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut iced, &mesh, 1.0);

    let drop = bare - iced.elevation_m[0];
    let expected = 3000.0 * isostasy::RHO_ICE_KG_M3 / isostasy::RHO_MANTLE_KG_M3;
    assert!(drop > 0.0, "ice did not depress the surface");
    assert!(
        (drop - expected).abs() < 1.0,
        "depression {drop} m, expected {expected} m"
    );
}

#[test]
fn sediment_load_is_buoyant_but_present() {
    let mesh = Mesh::build(LEVEL);
    let bare = uniform_elevation(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    let mut buried = uniform_planet(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    buried.sediment_m.fill(1000.0);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut buried, &mesh, 1.0);
    let rise = buried.elevation_m[0] - bare;
    // 1 km of 2000 kg/m^3 fill on 3300 kg/m^3 mantle floats 394 m of it.
    assert!((rise - 393.9).abs() < 2.0, "sediment rise {rise} m");
}

// --- sea level ---------------------------------------------------------------

#[test]
fn sea_level_two_level_world_is_analytic() {
    let mesh = Mesh::build(LEVEL);
    // Northern hemisphere at +1000 m, southern at -1000 m, real cell areas.
    let elev: Vec<f32> = mesh
        .latlon
        .iter()
        .map(|ll| if ll[0] > 0.0 { 1000.0 } else { -1000.0 })
        .collect();
    let low_area: f64 = elev
        .iter()
        .zip(&mesh.areas_km2)
        .filter(|(e, _)| **e < 0.0)
        .map(|(_, a)| *a as f64)
        .sum();
    // Volume that fills the low half to exactly -400 m.
    let target = 0.6 * low_area;
    let level = sea_level::solve_sea_level_m(&elev, &mesh.areas_km2, target);
    assert!((level - -400.0).abs() < 1.0, "level {level}");

    let v = sea_level::flooded_volume_km3(&elev, &mesh.areas_km2, level);
    // Residual expressed as a level error must be under 1 m.
    let level_err_m = (v - target).abs() / low_area * 1000.0;
    assert!(level_err_m < 1.0, "residual {level_err_m} m of level");
}

#[test]
fn sea_level_budget_zero_leaves_no_ocean() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    p.config.water_budget = 0.0;
    // Give it some relief so the minimum is not degenerate.
    for (i, t) in p.crust_thickness_m.iter_mut().enumerate() {
        *t += (i % 7) as f32 * 500.0;
    }
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);

    let min = p.elevation_m.iter().copied().fold(f32::INFINITY, f32::min);
    assert!((p.sea_level_m - (min - 1.0)).abs() < 1e-3);
    assert!(
        !(0..p.n_cells() as u32).any(|c| p.is_ocean(c)),
        "cells flooded"
    );
}

#[test]
fn sea_level_increases_with_budget() {
    let mesh = Mesh::build(LEVEL);
    let mut levels = Vec::new();
    for budget in [0.5, 1.0, 2.0] {
        let mut p = half_and_half(&mesh);
        p.config.water_budget = budget;
        let mut proc = GeologyProcess::new();
        step(&mut proc, &mut p, &mesh, 1.0);

        // The solved level must hold exactly the requested volume.
        let want = sea_level::water_volume_km3(&p.config, &mesh);
        let got = sea_level::flooded_volume_km3(&p.elevation_m, &mesh.areas_km2, p.sea_level_m);
        let wet_area: f64 = p
            .elevation_m
            .iter()
            .zip(&mesh.areas_km2)
            .filter(|(e, _)| **e < p.sea_level_m)
            .map(|(_, a)| *a as f64)
            .sum();
        let level_err_m = (got - want).abs() / wet_area * 1000.0;
        assert!(level_err_m < 1.0, "budget {budget}: {level_err_m} m off");
        levels.push(p.sea_level_m);
    }
    assert!(
        levels[0] < levels[1] && levels[1] < levels[2],
        "not monotonic: {levels:?}"
    );
}

/// Continents north of the equator, ocean south.
fn half_and_half(mesh: &Mesh) -> Planet {
    let mut p = uniform_planet(mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    for c in 0..mesh.n_cells() {
        if mesh.latlon[c][0] > 0.0 {
            p.crust_type[c] = CrustType::Continental;
            p.crust_thickness_m[c] = 38_000.0;
            p.crust_density_kg_m3[c] = 2700.0;
        }
    }
    p
}

// --- metamorphism ------------------------------------------------------------

/// Column of `cover_m` of inert granite over a 100 m marker of `marker`.
fn buried_marker(planet: &mut Planet, cell: u32, marker: RockType, cover_m: f32) {
    planet.columns.deposit(cell, marker, 100.0, 0.0);
    planet
        .columns
        .deposit(cell, RockType::Granite, cover_m, 0.0);
}

fn find_rock(
    planet: &Planet,
    cell: u32,
    kinds: &[RockType],
) -> Option<(RockType, MetamorphicGrade)> {
    planet
        .columns
        .col(cell)
        .iter()
        .find(|s| kinds.contains(&s.rock))
        .map(|s| (s.rock, s.grade))
}

#[test]
fn metamorphic_grade_follows_burial_depth() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    let pelitic = [
        RockType::Shale,
        RockType::Slate,
        RockType::Schist,
        RockType::Gneiss,
    ];
    // 25 C/km on the mid-point of the marker.
    buried_marker(&mut p, 0, RockType::Shale, 5_000.0); // ~126 C
    buried_marker(&mut p, 1, RockType::Shale, 10_000.0); // ~251 C
    buried_marker(&mut p, 2, RockType::Shale, 16_000.0); // ~401 C
    buried_marker(&mut p, 3, RockType::Shale, 24_000.0); // ~601 C

    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);

    assert_eq!(
        find_rock(&p, 0, &pelitic),
        Some((RockType::Shale, MetamorphicGrade::None))
    );
    assert_eq!(
        find_rock(&p, 1, &pelitic),
        Some((RockType::Slate, MetamorphicGrade::Low))
    );
    assert_eq!(
        find_rock(&p, 2, &pelitic),
        Some((RockType::Schist, MetamorphicGrade::Medium))
    );
    assert_eq!(
        find_rock(&p, 3, &pelitic),
        Some((RockType::Gneiss, MetamorphicGrade::High))
    );
    // Deposition ages survive transformation.
    assert!(p.columns.col(3).iter().all(|s| s.deposited_myr == 0.0));
}

#[test]
fn limestone_under_collision_becomes_marble() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 40_000.0, 2700.0, CrustType::Continental);
    // 4 km deep: 101 C from burial alone, below any threshold...
    buried_marker(&mut p, 0, RockType::Limestone, 4_000.0);
    buried_marker(&mut p, 1, RockType::Limestone, 4_000.0);
    // ...until the collision bonus is added.
    p.tectonic_flags[1] = cell_flags::COLLISION;

    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);

    let carbonate = [RockType::Limestone, RockType::Marble];
    assert_eq!(
        find_rock(&p, 0, &carbonate),
        Some((RockType::Limestone, MetamorphicGrade::None))
    );
    assert_eq!(
        find_rock(&p, 1, &carbonate),
        Some((RockType::Marble, MetamorphicGrade::Low))
    );
}

#[test]
fn sandstone_becomes_quartzite_when_buried() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    buried_marker(&mut p, 0, RockType::Sandstone, 12_000.0);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);
    assert_eq!(
        find_rock(&p, 0, &[RockType::Sandstone, RockType::Quartzite]),
        Some((RockType::Quartzite, MetamorphicGrade::Low))
    );
}

#[test]
fn basalt_becomes_amphibolite_at_medium_grade() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    buried_marker(&mut p, 0, RockType::Basalt, 8_000.0); // ~201 C, low grade only
    buried_marker(&mut p, 1, RockType::Basalt, 18_000.0); // ~451 C
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);
    assert_eq!(
        find_rock(&p, 0, &[RockType::Basalt, RockType::Amphibolite]),
        Some((RockType::Basalt, MetamorphicGrade::None))
    );
    assert_eq!(
        find_rock(&p, 1, &[RockType::Basalt, RockType::Amphibolite]),
        Some((RockType::Amphibolite, MetamorphicGrade::Medium))
    );
}

#[test]
fn metamorphism_is_idempotent() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    for (cell, cover) in [
        (0u32, 6_000.0f32),
        (1, 12_000.0),
        (2, 20_000.0),
        (3, 30_000.0),
    ] {
        buried_marker(&mut p, cell, RockType::Shale, cover);
        buried_marker(&mut p, cell + 8, RockType::Limestone, cover);
    }
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);
    let after_first: Vec<_> = (0..16u32).map(|c| p.columns.col(c).to_vec()).collect();

    // No flags, so nothing else touches the columns; sweep again on the next
    // scheduled step.
    p.step_index = METAMORPHISM_INTERVAL_STEPS;
    step(&mut proc, &mut p, &mesh, 1.0);
    let after_second: Vec<_> = (0..16u32).map(|c| p.columns.col(c).to_vec()).collect();
    assert_eq!(after_first, after_second);
}

#[test]
fn sweep_cadence_is_derived_from_step_index() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 35_000.0, 2700.0, CrustType::Continental);
    buried_marker(&mut p, 0, RockType::Shale, 12_000.0);
    let mut proc = GeologyProcess::new();

    p.step_index = 1; // not a sweep step
    step(&mut proc, &mut p, &mesh, 1.0);
    assert_eq!(p.columns.col(0)[0].rock, RockType::Shale);

    p.step_index = METAMORPHISM_INTERVAL_STEPS;
    step(&mut proc, &mut p, &mesh, 1.0);
    assert_eq!(p.columns.col(0)[0].rock, RockType::Slate);
}

#[test]
fn geotherm_matches_documented_thresholds() {
    // 8 km -> low, 15 km -> medium, 25 km -> high, with no flag bonus.
    assert_eq!(
        metamorphism::grade_for_temperature(metamorphism::temperature_c(8_000.0, 0)),
        MetamorphicGrade::Low
    );
    assert_eq!(
        metamorphism::grade_for_temperature(metamorphism::temperature_c(15_000.0, 0)),
        MetamorphicGrade::Medium
    );
    assert_eq!(
        metamorphism::grade_for_temperature(metamorphism::temperature_c(25_000.0, 0)),
        MetamorphicGrade::High
    );
    assert!(metamorphism::lithostatic_pressure_mpa(10_000.0) > 200.0);
}

// --- igneous emplacement -----------------------------------------------------

#[test]
fn arc_flags_produce_intrusions_and_ledger_mass() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 30_000.0, 2700.0, CrustType::Continental);
    for c in 0..12 {
        p.tectonic_flags[c] = cell_flags::ARC;
        // Something to intrude into.
        p.columns
            .deposit(c as u32, RockType::Andesite, 8_000.0, 0.0);
    }
    let mut proc = GeologyProcess::new();
    let ledger = step(&mut proc, &mut p, &mesh, 1.0);

    assert!(ledger.created_m3 > 0.0, "no mass recorded");
    let plutonic = [RockType::Diorite, RockType::Granite];
    let with_pluton = (0..12u32)
        .filter(|c| p.columns.col(*c).iter().any(|s| plutonic.contains(&s.rock)))
        .count();
    assert_eq!(with_pluton, 12, "arc cells without plutons");
    assert!(p.crust_thickness_m[0] > 30_000.0, "crust did not thicken");
    // Unflagged cells are untouched.
    assert!(p.columns.col(100).is_empty());
    assert_eq!(p.crust_thickness_m[100], 30_000.0);
}

#[test]
fn arc_erupts_volcanics_over_time() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 30_000.0, 2700.0, CrustType::Continental);
    p.tectonic_flags[0] = cell_flags::ARC;
    let mut proc = GeologyProcess::new();
    for _ in 0..20 {
        step(&mut proc, &mut p, &mesh, 1.0);
    }
    let volcanic = [RockType::Andesite, RockType::Tuff];
    assert!(
        p.columns.col(0).iter().any(|s| volcanic.contains(&s.rock)),
        "20 Myr of arc produced no surface volcanics"
    );
}

#[test]
fn hotspot_and_collision_emplace_their_rocks() {
    let mesh = Mesh::build(LEVEL);
    let mut p = uniform_planet(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
    p.tectonic_flags[0] = cell_flags::HOTSPOT;
    p.tectonic_flags[1] = cell_flags::COLLISION;
    p.columns.deposit(1, RockType::Shale, 20_000.0, 0.0);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);

    assert_eq!(p.columns.top_rock(0), Some(RockType::Basalt));
    assert!(p.columns.col(1).iter().any(|s| s.rock == RockType::Granite));
}

#[test]
fn emplacement_scales_with_vigor_and_dt() {
    let mesh = Mesh::build(LEVEL);
    let mut totals = Vec::new();
    for (vigor, dt) in [(1.0f32, 1.0f64), (2.0, 1.0), (1.0, 0.5)] {
        let mut p = uniform_planet(&mesh, 7_000.0, 3000.0, CrustType::Oceanic);
        p.config.tectonic_vigor = vigor;
        p.tectonic_flags[0] = cell_flags::HOTSPOT;
        let mut proc = GeologyProcess::new();
        step(&mut proc, &mut p, &mesh, dt);
        totals.push(p.columns.total_thickness_m(0));
    }
    assert!((totals[1] - 2.0 * totals[0]).abs() < 1e-2, "{totals:?}");
    assert!((totals[2] - 0.5 * totals[0]).abs() < 1e-2, "{totals:?}");
}

// --- determinism & statelessness ---------------------------------------------

/// A world with enough going on that every code path contributes.
fn busy_planet(mesh: &Mesh) -> Planet {
    let mut p = half_and_half(mesh);
    for c in 0..mesh.n_cells() {
        match c % 11 {
            0 => p.tectonic_flags[c] = cell_flags::ARC,
            1 => p.tectonic_flags[c] = cell_flags::COLLISION,
            2 => p.tectonic_flags[c] = cell_flags::HOTSPOT,
            3 => p.tectonic_flags[c] = cell_flags::ARC | cell_flags::COLLISION,
            4 => p.sediment_m[c] = 400.0,
            5 => p.ice_thickness_m[c] = 1500.0,
            _ => {}
        }
        if c % 3 == 0 {
            p.columns.deposit(c as u32, RockType::Shale, 14_000.0, 0.0);
            p.columns.deposit(c as u32, RockType::Limestone, 900.0, 0.0);
        }
    }
    p
}

fn fingerprint(p: &Planet) -> (Vec<u32>, u32, Vec<(u8, u32, u8)>) {
    let elev = p.elevation_m.iter().map(|e| e.to_bits()).collect();
    let cols = (0..p.n_cells() as u32)
        .flat_map(|c| {
            p.columns
                .col(c)
                .iter()
                .map(|s| (s.rock as u8, s.thickness_m.to_bits(), s.grade as u8))
                .collect::<Vec<_>>()
        })
        .collect();
    (elev, p.sea_level_m.to_bits(), cols)
}

#[test]
fn two_runs_are_bit_identical() {
    let mesh = Mesh::build(LEVEL);
    let mut a = busy_planet(&mesh);
    let mut b = busy_planet(&mesh);
    let mut pa = GeologyProcess::new();
    let mut pb = GeologyProcess::new();
    let mut la = MassLedger::default();
    let mut lb = MassLedger::default();
    for _ in 0..12 {
        let x = step(&mut pa, &mut a, &mesh, 0.5);
        let y = step(&mut pb, &mut b, &mesh, 0.5);
        la.created_m3 += x.created_m3;
        lb.created_m3 += y.created_m3;
    }
    assert_eq!(fingerprint(&a), fingerprint(&b));
    assert_eq!(la.created_m3.to_bits(), lb.created_m3.to_bits());
    assert!(la.created_m3 > 0.0);
}

#[test]
fn a_fresh_process_resumes_identically() {
    let mesh = Mesh::build(LEVEL);
    let mut a = busy_planet(&mesh);
    let mut b = busy_planet(&mesh);
    let mut pa = GeologyProcess::new();
    let mut pb = GeologyProcess::new();
    for i in 0..12 {
        step(&mut pa, &mut a, &mesh, 0.5);
        // Swap in a brand-new process mid-run, as checkpoint resume does.
        if i == 5 || i == 9 {
            pb = GeologyProcess::new();
        }
        step(&mut pb, &mut b, &mesh, 0.5);
    }
    assert_eq!(fingerprint(&a), fingerprint(&b));
}

#[test]
fn elevation_does_not_drift_across_repeated_steps() {
    let mesh = Mesh::build(LEVEL);
    let mut p = half_and_half(&mesh);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 0.5);
    let first = p.elevation_m.clone();
    for _ in 0..10 {
        step(&mut proc, &mut p, &mesh, 0.5);
    }
    // Nothing in this world changes state, so a from-scratch recompute must
    // return exactly the same field.
    assert!(first
        .iter()
        .zip(&p.elevation_m)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn continental_margins_stay_sharp_under_flexure() {
    let mesh = Mesh::build(LEVEL);
    let mut p = half_and_half(&mesh);
    let mut proc = GeologyProcess::new();
    step(&mut proc, &mut p, &mesh, 1.0);

    // Interior cells (all neighbours on the same side) keep their local solve.
    let interior_land = (0..mesh.n_cells() as u32)
        .find(|c| {
            mesh.latlon[*c as usize][0] > 0.3
                && mesh
                    .neighbors_of(*c)
                    .iter()
                    .all(|m| mesh.latlon[*m as usize][0] > 0.0)
        })
        .expect("no interior land cell");
    let land = isostasy::airy_elevation_m(38_000.0, 2700.0, false, 0.0, 0.0);
    assert!(
        (p.elevation_m[interior_land as usize] - land).abs() < 50.0,
        "interior land smoothed away"
    );

    // The step across the shoreline survives, at 60%+ of its raw height.
    let ocean = isostasy::airy_elevation_m(7_000.0, 3000.0, true, 0.0, 0.0);
    let max = p
        .elevation_m
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let min = p.elevation_m.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(max - min > 0.6 * (land - ocean), "relief {} m", max - min);
}
