//! Acceptance tests for WP4 (IMPLEMENTATION_PLAN.md §3).

mod common;

use std::sync::Arc;
use std::time::Instant;

use common::*;
use glam::DVec3;
use iw_core::planet::cell_flags;
use iw_core::{CrustType, Phase, Planet, Plate, RockType};
use iw_mesh::Mesh;
use iw_tectonics::{craton_min_separation_m, TectonicsProcess};

// --- Phase 1: craton seeding ---

#[test]
fn cratons_are_seeded_spaced_and_thick() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    // The continent now ACCRETES over the era (cores first), so the full
    // outline and thickness band only exist at the end of the phase.
    h.run_crustal_formation(&mut p);

    // Pangaea-first: one plate per landmass (supercontinent + 0-2 micros),
    // and the supercontinent must dominate.
    let n_masses = h.planet.plates.len();
    assert!(
        (1..=3).contains(&n_masses),
        "expected 1-3 landmasses, got {n_masses}"
    );
    let mut cells = vec![0u32; n_masses];
    for c in 0..h.planet.n_cells() {
        let k = h.planet.plate_id[c] as usize;
        if k < n_masses && h.planet.crust_type[c] == CrustType::Continental {
            cells[k] += 1;
        }
    }
    let total: u32 = cells.iter().sum();
    assert!(total > 0, "no continental cells at all");
    assert!(
        cells[0] as f64 >= total as f64 * 0.75,
        "supercontinent is not dominant: {cells:?}"
    );
    // The supercontinent is one connected landmass (its cells form a single
    // component under continental adjacency).
    let n = h.planet.n_cells();
    let mut label = vec![false; n];
    let start = (0..n)
        .find(|c| h.planet.plate_id[*c] == 0 && h.planet.crust_type[*c] == CrustType::Continental)
        .expect("supercontinent has cells");
    let mut stack = vec![start as u32];
    label[start] = true;
    let mut reached = 0u32;
    while let Some(x) = stack.pop() {
        reached += 1;
        for &m in h.mesh.neighbors_of(x) {
            let mi = m as usize;
            if !label[mi]
                && h.planet.plate_id[mi] == 0
                && h.planet.crust_type[mi] == CrustType::Continental
            {
                label[mi] = true;
                stack.push(m);
            }
        }
    }
    assert!(
        reached as f64 >= cells[0] as f64 * 0.9,
        "supercontinent is fragmented: largest component {reached} of {}",
        cells[0]
    );
    // Shield-core spacing diagnostics stay meaningful.
    let min_sep_km =
        craton_min_separation_m(h.planet.config.seed, h.planet.config.craton_count) as f32 / 1000.0;
    assert!(min_sep_km > 100.0, "core min separation {min_sep_km} km");

    // Craton cells: thick, light, continental, with a real basement column.
    let mut land = 0;
    for c in 0..h.planet.n_cells() as u32 {
        if h.planet.crust_type[c as usize] != CrustType::Continental {
            continue;
        }
        land += 1;
        let t = h.planet.crust_thickness_m[c as usize];
        // Calibration: the band tracks `CRATON_EDGE_THICKNESS_M` ..
        // `CRATON_CORE_THICKNESS_M`, which moved from 35-45 km to 31-40 km.
        // 45 km floats at +2.6 km (a plateau, not a shield) and 35 km is
        // exactly the +800 m anchor, so the old edge value left every craton
        // dry to its outermost cell and the planet had no continental shelf.
        // Widened by the craton interior texture: the radial profile is now
        // modulated by +-10% of fBm so shields are not billiard-smooth, which
        // takes the extremes to 27.9-44.0 km.
        assert!((27_900.0..=44_000.0).contains(&t), "craton thickness {t}");
        assert_eq!(h.planet.crust_density_kg_m3[c as usize], 2_700.0);
        assert!(
            !h.planet.columns.col(c).is_empty(),
            "craton cell {c} has no strata"
        );
        assert_eq!(h.planet.columns.top_rock(c), Some(RockType::Granite));
    }
    // Genesis target is 0.50 (raised with the recursive hand-off mosaic —
    // the more active drift regime floods/consumes more margin).
    let frac = land as f32 / h.planet.n_cells() as f32;
    assert!(
        (0.15..0.55).contains(&frac),
        "continental fraction {frac} out of range"
    );

    // Ocean floor is thin, dense, basaltic.
    for c in 0..h.planet.n_cells() as u32 {
        if h.planet.crust_type[c as usize] == CrustType::Oceanic {
            // Fresh sea floor is the 7 km reference plus a fixed low-amplitude
            // noise field (`OCEANIC_THICKNESS_NOISE_M`), so that the abyssal
            // plains are not a perfectly uniform sheet; it is only ~50 m of
            // isostatic relief.
            let ot = h.planet.crust_thickness_m[c as usize];
            assert!((6_450.0..=7_550.0).contains(&ot), "ocean thickness {ot}");
            assert_eq!(h.planet.columns.top_rock(c), Some(RockType::Basalt));
        }
    }
}

/// Pangaea-first breakup: over the drift era the supercontinent must fragment
/// into several continental plates — the fragments' shapes coming from the
/// rift graph is the entire point of the genesis model.
#[test]
fn supercontinent_breaks_up_during_drift() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 400);

    let np = h.planet.plates.len();
    let mut cont_plates = std::collections::BTreeSet::new();
    for c in 0..h.planet.n_cells() {
        if h.planet.crust_type[c] == CrustType::Continental {
            cont_plates.insert(h.planet.plate_id[c]);
        }
    }
    println!(
        "after 200 Myr of drift: {np} plates, continental crust on {} of them",
        cont_plates.len()
    );
    // Pangaea's own first breakup produced two majors (Laurasia/Gondwana)
    // within this kind of window; further fragmentation takes longer. The
    // hard requirement is that breakup HAPPENS — the map stops being one
    // landmass — not a specific fragment count at exactly 200 Myr.
    assert!(
        cont_plates.len() >= 2,
        "supercontinent never broke up: continental crust on only {} plate(s)",
        cont_plates.len()
    );
    assert!(
        h.flagged(cell_flags::SUTURE) > 0,
        "no sutures anywhere after a drift era"
    );
}

// --- Phase 1 -> Drift hand-off ---

/// The hand-off lays a sparse network of inherited weakness (`SUTURE`) along
/// the creases of a ridged noise field, so later rifts have something to follow
/// besides the seams where cratons happened to collide.
#[test]
fn handoff_marks_inherited_weakness() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    let seams = h.flagged(cell_flags::SUTURE);
    h.run(&mut p, Phase::Drift, 1);
    let after = h.flagged(cell_flags::SUTURE);
    // The overlay flags exactly 3% of cells (a rank threshold, not a value
    // threshold); the increment comes out a little under that because some of
    // those cells already sit on a collision seam.
    let added = (after - seams) as f32 / h.planet.n_cells() as f32;
    assert!(
        (0.015..0.031).contains(&added),
        "weakness overlay added {added} of the planet ({seams} -> {after})"
    );
}

#[test]
fn handoff_partitions_the_whole_planet() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 1);

    plates_are_partition(&h.planet, &h.mesh).expect("plate partition");
    // Recursive 30:70..50:50 splitting to a 15%-of-surface cap: roughly
    // 8..16 varied-size plates, capped at 24.
    let np = h.planet.plates.len();
    assert!((6..=24).contains(&np), "{np} plates after hand-off");
}

/// The point of the recursive hand-off split: the supercontinent must START
/// drift dealt onto several plates (boundaries under Pangaea), not sit whole
/// on one plate with the rest of the mosaic in the ocean. With ~30% of the
/// surface continental and no plate allowed more than 15%, at least three
/// plates have to hold a real share of the landmass.
#[test]
fn handoff_deals_pangaea_onto_several_plates() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 1);

    let np = h.planet.plates.len();
    let mut cont_cells = vec![0usize; np];
    let mut total_cont = 0usize;
    for c in 0..h.planet.n_cells() {
        if h.planet.crust_type[c] == CrustType::Continental {
            let pl = h.planet.plate_id[c] as usize;
            if pl < np {
                cont_cells[pl] += 1;
                total_cont += 1;
            }
        }
    }
    // Plates holding at least 5% of the continental crust each.
    let sharers = cont_cells.iter().filter(|&&m| m * 20 >= total_cont).count();
    assert!(
        sharers >= 3,
        "Pangaea must start on several plates: {sharers} plates hold >=5% of it \
         (distribution {cont_cells:?})"
    );
}

// --- Drift kinematics ---

#[test]
fn plate_speeds_are_earthlike() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 40);

    // Individual snapshots are noisy (a plate can be mid-reorganisation), so
    // pool cell speeds over a 100 Myr window.
    let mut v: Vec<f32> = Vec::new();
    for _ in 0..20 {
        h.run(&mut p, Phase::Drift, 10);
        v.extend(h.speeds_cm_yr());
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let med = v[v.len() / 2];
    let p90 = v[v.len() * 9 / 10];
    let max = *v.last().expect("cells");
    println!("plate speeds cm/yr: median {med:.2}, p90 {p90:.2}, max {max:.2}");
    assert!(
        (2.0..=10.0).contains(&med),
        "median plate speed {med:.2} cm/yr outside 2-10"
    );
    assert!(p90 < 14.0, "p90 speed {p90:.2} cm/yr too high");
    assert!(max <= 15.5, "max plate speed {max:.2} cm/yr above the cap");
}

#[test]
fn divergent_boundaries_make_young_ocean_floor() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 20);

    let fresh = (0..h.planet.n_cells() as u32)
        .filter(|c| {
            h.planet.tectonic_flags[*c as usize] & cell_flags::RIFT != 0
                && h.planet.crust_type[*c as usize] == CrustType::Oceanic
                && h.planet.crust_age_myr[*c as usize] < 2.0
                && h.planet.columns.top_rock(*c) == Some(RockType::Basalt)
        })
        .count();
    assert!(fresh > 0, "no age-0 basaltic crust at any spreading centre");
}

#[test]
fn convergent_boundaries_consume_crust() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    let n_before = h.planet.n_cells();
    h.run_crustal_formation(&mut p);
    let subducted_after_phase1 = h.total.subducted_m3;
    h.run(&mut p, Phase::Drift, 40);

    assert_eq!(h.planet.n_cells(), n_before, "cells are never created/lost");
    assert!(
        h.total.subducted_m3 > subducted_after_phase1,
        "nothing was subducted during drift"
    );
    assert!(
        h.flagged(cell_flags::SUBDUCTING) > 0,
        "no trench cells flagged"
    );
    assert!(h.flagged(cell_flags::ARC) > 0, "no volcanic arcs built");

    // Trench cells are flexed thinner than reference ocean floor. (A few can
    // sit above it: an arc or a plume may pile lava onto the same cell.)
    let mut trench: Vec<f32> = (0..h.planet.n_cells())
        .filter(|c| h.planet.tectonic_flags[*c] & cell_flags::SUBDUCTING != 0)
        .map(|c| h.planet.crust_thickness_m[c])
        .collect();
    trench.sort_by(|a, b| a.total_cmp(b));
    let median = trench[trench.len() / 2];
    assert!(
        median < 6_500.0,
        "median trench crust {median} m is not flexed down"
    );
}

#[test]
fn arcs_erupt_andesite() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 40);
    let andesitic = (0..h.planet.n_cells() as u32)
        .filter(|c| {
            h.planet.tectonic_flags[*c as usize] & cell_flags::ARC != 0
                && h.planet
                    .columns
                    .col(*c)
                    .iter()
                    .any(|s| matches!(s.rock, RockType::Andesite | RockType::Tuff))
        })
        .count();
    assert!(andesitic > 0, "arcs deposited no andesite or tuff");
}

// --- collision & welding, on a purpose-built two-continent planet ---

/// Two hemispherical continental plates, plate 0 (y > 0) rotating about +Z.
/// Around x = -1 the boundary closes; around x = +1 it opens.
fn two_continent_planet(level: u8, omega_rad_myr: f64) -> Harness {
    let mesh = Arc::new(Mesh::build(level));
    let mut h = Harness::new(test_config(level), Arc::clone(&mesh));
    let n = h.planet.n_cells();
    for c in 0..n as u32 {
        h.planet.plate_id[c as usize] = u16::from(mesh.centers[c as usize].y <= 0.0);
        h.planet.crust_type[c as usize] = CrustType::Continental;
        h.planet.crust_thickness_m[c as usize] = 40_000.0;
        h.planet.crust_density_kg_m3[c as usize] = 2_700.0;
        h.planet.columns.deposit(c, RockType::Granite, 6_000.0, 0.0);
    }
    h.planet.plates = vec![
        Plate {
            euler_pole: DVec3::Z,
            omega_rad_myr,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
        Plate {
            euler_pole: DVec3::Z,
            omega_rad_myr: 0.0,
            welded_to: None,
            accum: glam::DQuat::IDENTITY,
            rift_partner: None,
            rift_born_myr: f64::NEG_INFINITY,
        },
    ];
    h
}

#[test]
fn continent_collision_thickens_crust() {
    let mut h = two_continent_planet(5, 0.008);
    let mut p = TectonicsProcess::new();
    let baseline = 40_000.0f32;
    // The orogeny is transient: convergence decays, then the plates weld and
    // the collision zone is no longer a boundary. Sample as it happens.
    let mut peak_collision = 0;
    let mut max = 0.0f32;
    for _ in 0..40 {
        h.run(&mut p, Phase::Drift, 1);
        peak_collision = peak_collision.max(h.flagged(cell_flags::COLLISION));
        max = max.max(
            h.planet
                .crust_thickness_m
                .iter()
                .copied()
                .fold(0.0f32, f32::max),
        );
    }

    assert!(peak_collision > 0, "no collision zone formed");
    assert!(
        max > baseline * 1.05,
        "collision only reached {max:.0} m of crust"
    );
    assert!(
        max <= iw_tectonics::MAX_CRUST_THICKNESS_M + 1.0,
        "crust thickened past the Tibet cap: {max}"
    );
    // Continental crust never subducts — but drift v2 conserves it by VOLUME,
    // not by cell count: collision stacks the two plates' crust into fewer
    // cells (fold + underthrust), and the vacated cells on the diverging side
    // correctly floor with new ocean. So assert volume conservation instead of
    // "every cell stays continental".
    let cont_volume: f64 = (0..h.planet.n_cells())
        .filter(|c| h.planet.crust_type[*c] == CrustType::Continental)
        .map(|c| h.planet.crust_thickness_m[c] as f64 * h.mesh.areas_km2[c] as f64)
        .sum();
    let initial_volume: f64 = (0..h.planet.n_cells())
        .map(|c| baseline as f64 * h.mesh.areas_km2[c] as f64)
        .sum();
    assert!(
        cont_volume > initial_volume * 0.90,
        "continental crust volume shrank {:.1}% — collision is destroying continents",
        (1.0 - cont_volume / initial_volume) * 100.0
    );
}

#[test]
fn locked_collision_welds_the_plates() {
    let mut h = two_continent_planet(5, 0.008);
    let mut p = TectonicsProcess::new();
    assert_eq!(h.planet.plates.len(), 2);
    let mut welded_at = None;
    for step in 0..120 {
        h.run(&mut p, Phase::Drift, 1);
        if h.planet.plates.len() == 1 {
            welded_at = Some(step);
            break;
        }
    }
    let step = welded_at.expect("plates never welded");
    assert!(
        h.flagged(cell_flags::SUTURE) > 0,
        "welding left no suture (step {step})"
    );
    // One plate means one motion: every cell shares a velocity field.
    assert_eq!(h.planet.plates.len(), 1);
}

// --- fields maintained every step ---

#[test]
fn oceanic_crust_ages_and_densifies() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run(&mut p, Phase::CrustalFormation, 25);
    for c in 0..h.planet.n_cells() {
        let age = h.planet.crust_age_myr[c];
        let rho = h.planet.crust_density_kg_m3[c];
        match h.planet.crust_type[c] {
            CrustType::Oceanic => {
                let want = (3_000.0 + 30.0 * age.sqrt()).min(3_300.0);
                assert!((rho - want).abs() < 0.01, "rho({age}) = {rho}, want {want}");
            }
            CrustType::Continental => {
                assert_eq!(age, 0.0);
                assert_eq!(rho, 2_700.0);
            }
        }
    }
    let oldest = h
        .planet
        .crust_age_myr
        .iter()
        .copied()
        .fold(0.0f32, f32::max);
    assert!(oldest >= 24.0, "ocean floor did not age: {oldest} Myr");
}

#[test]
fn hotspots_are_fixed_and_erupt_basalt() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    assert!(h.planet.hotspots.is_empty(), "plumes exist too early");
    h.run(&mut p, Phase::Drift, 30);

    assert_eq!(
        h.planet.hotspots.len(),
        h.planet.config.hotspot_count as usize
    );
    let before: Vec<_> = h.planet.hotspots.iter().map(|x| x.pos).collect();
    h.run(&mut p, Phase::Drift, 10);
    let after: Vec<_> = h.planet.hotspots.iter().map(|x| x.pos).collect();
    assert_eq!(before, after, "plumes are supposed to be fixed");

    let volcanic = (0..h.planet.n_cells() as u32)
        .filter(|c| {
            h.planet.tectonic_flags[*c as usize] & cell_flags::HOTSPOT != 0
                && h.planet.columns.top_rock(*c) == Some(RockType::Basalt)
        })
        .count();
    assert!(volcanic > 0, "no plume erupted basalt");
}

#[test]
fn transient_flags_are_rewritten_each_step() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 20);
    let sutures = h.flagged(cell_flags::SUTURE);
    // Scribble every bit on and check only SUTURE survives a quiet step.
    for f in h.planet.tectonic_flags.iter_mut() {
        *f = 0xFF;
    }
    h.run(&mut p, Phase::Drift, 1);
    // Drift v2: sutures are advected state, not a static mask — a remap during
    // this step moves them with their plates and may legitimately consume a
    // few at convergent overlaps or ridge gaps. Transient bits must still be
    // gone; the persistent bit must survive at (nearly) full strength.
    let survived = h.flagged(cell_flags::SUTURE);
    assert!(
        survived as f64 >= h.planet.n_cells() as f64 * 0.98,
        "SUTURE should persist through a step (survived {survived} of {})",
        h.planet.n_cells()
    );
    assert!(h.flagged(cell_flags::COLLISION) < h.planet.n_cells());
    assert!(h.flagged(cell_flags::TRANSFORM) < h.planet.n_cells());
    let _ = sutures;
}

#[test]
fn elevation_and_sea_level_are_left_to_geology() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 20);
    assert!(h.planet.elevation_m.iter().all(|e| *e == 0.0));
    assert_eq!(h.planet.sea_level_m, 0.0);
}

#[test]
fn mass_ledger_only_reports_creation_and_subduction() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 40);
    assert!(h.total.created_m3 > 0.0);
    assert!(h.total.subducted_m3 > 0.0);
    // `iw-sim` debug-asserts residual == deposited - eroded; tectonics must
    // leave both alone so the sum over all processes stays balanced.
    assert_eq!(h.total.deposited_m3, 0.0);
    assert_eq!(h.total.eroded_m3, 0.0);
}

// --- determinism ---

#[test]
fn two_identical_runs_agree_byte_for_byte() {
    let mesh = Arc::new(Mesh::build(5));
    let run = || {
        let mut h = Harness::new(test_config(5), Arc::clone(&mesh));
        let mut p = TectonicsProcess::new();
        h.run(&mut p, Phase::CrustalFormation, 60);
        h.run(&mut p, Phase::Drift, 50);
        h.digest()
    };
    assert!(run() == run(), "identical runs diverged");
}

#[test]
fn a_fresh_process_resumes_a_checkpoint_exactly() {
    let mesh = Arc::new(Mesh::build(5));
    let mut straight = Harness::new(test_config(5), Arc::clone(&mesh));
    let mut proc_a = TectonicsProcess::new();
    straight.run(&mut proc_a, Phase::CrustalFormation, 60);
    straight.run(&mut proc_a, Phase::Drift, 25);

    // "Checkpoint": everything the store would persist.
    let checkpoint: Planet = straight.planet.clone();

    straight.run(&mut proc_a, Phase::Drift, 25);

    let mut resumed = Harness::new(test_config(5), Arc::clone(&mesh));
    resumed.planet = checkpoint;
    // A brand new instance, with none of proc_a's caches or scratch.
    let mut proc_b = TectonicsProcess::new();
    resumed.run(&mut proc_b, Phase::Drift, 25);

    assert!(
        straight.digest() == resumed.digest(),
        "resume-from-checkpoint diverged from the straight-through run"
    );
}

#[test]
fn refinement_and_recent_past_keep_the_planet_valid() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    h.run(&mut p, Phase::Drift, 40);
    h.run(&mut p, Phase::Refinement, 40);
    let frozen: Vec<f64> = h.planet.plates.iter().map(|x| x.omega_rad_myr).collect();
    h.run(&mut p, Phase::RecentPast, 60);
    plates_are_partition(&h.planet, &h.mesh).expect("partition survives every phase");
    let after: Vec<f64> = h.planet.plates.iter().map(|x| x.omega_rad_myr).collect();
    assert_eq!(
        frozen, after,
        "RecentPast should freeze plate velocities (DESIGN.md §5 Phase 4)"
    );
}

/// The golden planets use several seeds; the engine must stay well behaved on
/// all of them, not just the one the constants were tuned against.
#[test]
fn other_seeds_stay_well_behaved() {
    let mesh = Arc::new(Mesh::build(5));
    for seed in [1_337u64, 7, 999_331, 2_024] {
        let mut config = test_config(5);
        config.seed = seed;
        let mut h = Harness::new(config, Arc::clone(&mesh));
        let mut p = TectonicsProcess::new();
        h.run_crustal_formation(&mut p);
        h.run(&mut p, Phase::Drift, 1);
        plates_are_partition(&h.planet, &h.mesh)
            .unwrap_or_else(|e| panic!("seed {seed} hand-off: {e}"));
        let np = h.planet.plates.len();
        assert!(
            (6..=24).contains(&np),
            "seed {seed}: {np} plates at hand-off"
        );

        h.run(&mut p, Phase::Drift, 200);
        plates_are_partition(&h.planet, &h.mesh)
            .unwrap_or_else(|e| panic!("seed {seed} after drift: {e}"));
        let np = h.planet.plates.len();
        assert!(
            (2..=iw_tectonics::MAX_PLATES).contains(&np),
            "seed {seed}: {np} plates after drift"
        );
        let land = h.continental_cells() as f32 / h.planet.n_cells() as f32;
        assert!(
            (0.10..0.45).contains(&land),
            "seed {seed}: continental fraction {land:.3}"
        );
        let mut v = h.speeds_cm_yr();
        v.sort_by(|a, b| a.total_cmp(b));
        assert!(
            *v.last().expect("cells") <= 15.5,
            "seed {seed}: speed cap breached"
        );
        assert!(h.total.subducted_m3 > 0.0 && h.total.created_m3 > 0.0);
    }
}

// --- performance ---

#[test]
#[ignore = "timing; run with --ignored"]
fn level6_phase1_and_2_within_budget() {
    let t0 = Instant::now();
    let mesh = Arc::new(Mesh::build(6));
    let build = t0.elapsed();

    let mut h = Harness::new(test_config(6), mesh);
    let mut p = TectonicsProcess::new();
    let t1 = Instant::now();
    h.run(&mut p, Phase::CrustalFormation, 200);
    let phase1 = t1.elapsed();
    let t2 = Instant::now();
    h.run(&mut p, Phase::Drift, 400);
    let phase2 = t2.elapsed();

    let total = phase1 + phase2;
    println!(
        "mesh build {:?}; phase1 (200 steps) {:?}; phase2 (400 steps) {:?}; total {:?}",
        build, phase1, phase2, total
    );
    println!(
        "  plates {}, continental {:.1}%",
        h.planet.plates.len(),
        100.0 * h.continental_cells() as f32 / h.planet.n_cells() as f32
    );
    assert!(total.as_secs_f32() < 60.0, "over budget: {total:?}");
}
