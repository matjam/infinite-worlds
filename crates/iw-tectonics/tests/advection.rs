//! Drift v2 behavioral guarantees: crust actually moves, rifts genuinely
//! open, sea floor records an age gradient, and hotspots build chains.

use std::sync::Arc;

use glam::{DQuat, DVec3};
use iw_core::{
    rng_for, CrustType, MassLedger, NullProgress, Phase, Planet, PlanetConfig, Plate, Process,
    RockType, StepCtx,
};
use iw_mesh::Mesh;
use iw_tectonics::TectonicsProcess;

fn config(level: u8) -> PlanetConfig {
    let mut c = PlanetConfig {
        seed: 4242,
        subdivision_level: level,
        ..Default::default()
    };
    c.sanitize();
    c
}

fn step(
    p: &mut TectonicsProcess,
    planet: &mut Planet,
    mesh: &Mesh,
    dt: f64,
    total: &mut MassLedger,
) {
    let mut ledger = MassLedger::default();
    let mut ctx = StepCtx {
        rng: rng_for(planet.config.seed, p.name(), planet.step_index),
        progress: &NullProgress,
        ledger: &mut ledger,
    };
    p.step(planet, mesh, dt, &mut ctx);
    planet.step_index += 1;
    planet.time_myr += dt;
    total.subducted_m3 += ledger.subducted_m3;
    total.created_m3 += ledger.created_m3;
}

/// One continental cap on its own plate, rotating steadily about +Z, ocean at
/// rest on a second plate. The cap must actually travel across the grid.
#[test]
fn continents_translate_across_the_grid() {
    let level = 5;
    let mesh = Arc::new(Mesh::build(level));
    let mut planet = Planet::new(config(level), mesh.n_cells());
    planet.phase = Phase::RecentPast; // frozen force balance: imposed kinematics persist, advection still runs

    let cap_center = glam::Vec3::new(1.0, 0.0, 0.0);
    for c in 0..mesh.n_cells() as u32 {
        let inside = mesh.centers[c as usize].dot(cap_center) > 0.94; // ~20 deg cap
        if inside {
            planet.plate_id[c as usize] = 0;
            planet.crust_type[c as usize] = CrustType::Continental;
            planet.crust_thickness_m[c as usize] = 38_000.0;
            planet.crust_density_kg_m3[c as usize] = 2_700.0;
            planet.columns.deposit(c, RockType::Granite, 8_000.0, 0.0);
        } else {
            planet.plate_id[c as usize] = 1;
            planet.columns.deposit(c, RockType::Basalt, 2_000.0, 0.0);
        }
    }
    // ~5.6 cm/yr eastward at the equator.
    let omega = 0.05 / iw_mesh::EARTH_RADIUS_M * 1.0e6;
    planet.plates = vec![
        Plate {
            euler_pole: DVec3::Z,
            omega_rad_myr: omega,
            welded_to: None,
            accum: DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
        Plate {
            euler_pole: DVec3::Z,
            omega_rad_myr: 0.0,
            welded_to: None,
            accum: DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
    ];

    let centroid = |planet: &Planet| -> glam::Vec3 {
        let mut sum = glam::Vec3::ZERO;
        for c in 0..planet.n_cells() {
            if planet.crust_type[c] == CrustType::Continental {
                sum += mesh.centers[c];
            }
        }
        sum.normalize()
    };
    let cont_count_before = planet
        .crust_type
        .iter()
        .filter(|t| **t == CrustType::Continental)
        .count();
    let start = centroid(&planet);
    let mut total = MassLedger::default();
    let mut p = TectonicsProcess::default();
    for _ in 0..100 {
        step(&mut p, &mut planet, &mesh, 0.5, &mut total);
    }
    let end = centroid(&planet);
    let travelled_km = start.angle_between(end) * iw_mesh::EARTH_RADIUS_KM;

    // 50 Myr at ~5 cm/yr is ~2800 km; force feedback may slow it, but the cap
    // must move at least ~4 cells to prove fields advect.
    assert!(
        travelled_km > 800.0,
        "continent centroid moved only {travelled_km:.0} km — crust is not advecting"
    );
    // Longitude direction: eastward (+y from +x under +Z rotation).
    assert!(
        end.y > start.y + 0.05,
        "continent did not move in the imposed direction"
    );
    // The cap survives the trip more-or-less whole.
    let cont_count_after = planet
        .crust_type
        .iter()
        .filter(|t| **t == CrustType::Continental)
        .count();
    assert!(
        cont_count_after as f64 > cont_count_before as f64 * 0.7,
        "continent lost too much area in transit: {cont_count_before} -> {cont_count_after}"
    );
}

/// Two oceanic plates pulling apart must leave a young-in-the-middle age
/// gradient behind, and the gap floor must be basalt-topped ridge crust.
#[test]
fn divergence_leaves_an_age_gradient() {
    let level = 5;
    let mesh = Arc::new(Mesh::build(level));
    let mut planet = Planet::new(config(level), mesh.n_cells());
    planet.phase = Phase::RecentPast; // frozen force balance: imposed kinematics persist, advection still runs
    for c in 0..mesh.n_cells() as u32 {
        planet.plate_id[c as usize] = u16::from(mesh.centers[c as usize].y <= 0.0);
        planet.crust_age_myr[c as usize] = 60.0;
        planet.columns.deposit(c, RockType::Basalt, 2_000.0, 0.0);
    }
    // Both plates rotate about +X in opposite senses: the y=0 great circle
    // near +Z/-Z... choose poles so the boundary at y=0, x-z plane, diverges.
    let omega = 0.04 / iw_mesh::EARTH_RADIUS_M * 1.0e6;
    planet.plates = vec![
        Plate {
            euler_pole: DVec3::X,
            omega_rad_myr: omega,
            welded_to: None,
            accum: DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
        Plate {
            euler_pole: DVec3::X,
            omega_rad_myr: -omega,
            welded_to: None,
            accum: DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
    ];
    let mut total = MassLedger::default();
    let mut p = TectonicsProcess::default();
    for _ in 0..80 {
        step(&mut p, &mut planet, &mesh, 0.5, &mut total);
    }

    // Sample cells by distance from the (initial) boundary plane y=0 on the
    // +z hemisphere where divergence happened (about +X rotation, +y side
    // moves toward +z at... sample both sides symmetrically).
    let mut near: Vec<f32> = Vec::new();
    let mut far: Vec<f32> = Vec::new();
    for c in 0..planet.n_cells() {
        let y = mesh.centers[c].y.abs();
        if planet.crust_type[c] != CrustType::Oceanic {
            continue;
        }
        // Exclude the neighbourhood of the Euler poles (±X): relative velocity
        // vanishes there, so no spreading happens at that part of the boundary.
        if mesh.centers[c].x.abs() > 0.5 {
            continue;
        }
        if y < 0.1 {
            near.push(planet.crust_age_myr[c]);
        } else if y > 0.5 {
            far.push(planet.crust_age_myr[c]);
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    assert!(
        !near.is_empty() && !far.is_empty(),
        "sampling bands are empty"
    );
    assert!(
        mean(&near) < mean(&far) * 0.6,
        "no age gradient: near-boundary mean {:.1} Myr vs far {:.1} Myr",
        mean(&near),
        mean(&far)
    );
    assert!(
        total.created_m3 > 0.0,
        "divergence created no ridge crust at all"
    );
}

/// The whole point of drift v2, measured: over a default 200 Myr drift era the
/// continental map must genuinely rearrange under the emergent force balance —
/// not just wobble in place. Jaccard overlap between the start-of-drift and
/// end-of-drift continental masks is the metric ("1.0" would mean nothing
/// moved at all).
#[test]
fn continents_rearrange_over_a_drift_era() {
    let level = 5;
    let mesh = Arc::new(Mesh::build(level));
    let mut planet = Planet::new(config(level), mesh.n_cells());
    let mut p = TectonicsProcess::default();
    let mut total = MassLedger::default();
    // Full crustal-formation era builds cratons and hands off plates.
    planet.phase = Phase::CrustalFormation;
    for _ in 0..200 {
        step(&mut p, &mut planet, &mesh, 1.0, &mut total);
    }
    planet.phase = Phase::Drift;
    // A couple of steps so the hand-off partition exists before sampling.
    for _ in 0..4 {
        step(&mut p, &mut planet, &mesh, 0.5, &mut total);
    }
    let start: Vec<bool> = planet
        .crust_type
        .iter()
        .map(|t| *t == CrustType::Continental)
        .collect();
    for _ in 0..396 {
        step(&mut p, &mut planet, &mesh, 0.5, &mut total);
    }
    let mut inter = 0usize;
    let mut union = 0usize;
    for (t, was) in planet.crust_type.iter().zip(&start) {
        let now = *t == CrustType::Continental;
        if now && *was {
            inter += 1;
        }
        if now || *was {
            union += 1;
        }
    }
    let jaccard = inter as f64 / union.max(1) as f64;
    println!("continental mask Jaccard over 198 Myr of drift: {jaccard:.3}");
    assert!(
        jaccard < 0.65,
        "continents barely moved: Jaccard {jaccard:.3} (want < 0.65 turnover)"
    );
    assert!(
        jaccard > 0.05,
        "continental map unrecognizable: Jaccard {jaccard:.3} — mass not conserved?"
    );
}

/// Fixed hotspot under a moving oceanic plate deposits a spatially extended
/// basalt trail, not a single immortal cone.
#[test]
fn hotspots_build_chains_on_moving_plates() {
    let level = 5;
    let mesh = Arc::new(Mesh::build(level));
    let mut cfg = config(level);
    cfg.hotspot_count = 1;
    let mut planet = Planet::new(cfg, mesh.n_cells());
    planet.phase = Phase::RecentPast; // frozen force balance: imposed kinematics persist, advection still runs
    for c in 0..mesh.n_cells() as u32 {
        planet.plate_id[c as usize] = 0;
        planet.columns.deposit(c, RockType::Gabbro, 5_000.0, 0.0);
    }
    let omega = 0.06 / iw_mesh::EARTH_RADIUS_M * 1.0e6;
    planet.plates = vec![Plate {
        euler_pole: DVec3::Z,
        omega_rad_myr: omega,
        welded_to: None,
        accum: DQuat::IDENTITY,
        rift_partner: None,
        rift_born_myr: f64::NEG_INFINITY,
    }];
    let mut total = MassLedger::default();
    let mut p = TectonicsProcess::default();
    for _ in 0..120 {
        step(&mut p, &mut planet, &mesh, 0.5, &mut total);
    }
    // Cells carrying a meaningful surface basalt load beyond the background:
    // with advection the deposits ride away from the plume, so the trail is
    // many cells long; without it, exactly one cone exists.
    let trail: usize = (0..planet.n_cells() as u32)
        .filter(|c| {
            planet
                .columns
                .col(*c)
                .iter()
                .filter(|s| s.rock == RockType::Basalt)
                .map(|s| s.thickness_m)
                .sum::<f32>()
                > 300.0
        })
        .count();
    assert!(
        trail >= 5,
        "hotspot trail spans only {trail} cells — deposits are not advecting off the plume"
    );
}
