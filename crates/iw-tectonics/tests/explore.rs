//! Diagnostic runs used to calibrate the model and check scaling. These print
//! rather than assert; run them with `cargo test -- --ignored --nocapture`.

mod common;

use common::*;
use iw_core::planet::cell_flags;
use iw_core::{CrustType, Phase};
use iw_tectonics::TectonicsProcess;

fn pct(v: &mut [f32], q: f32) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    if v.is_empty() {
        return 0.0;
    }
    v[((v.len() - 1) as f32 * q) as usize]
}

#[test]
#[ignore = "diagnostic"]
fn explore() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    println!(
        "after phase1: continental {} / {} ({:.1}%), plates {}, sutures {}",
        h.continental_cells(),
        h.planet.n_cells(),
        100.0 * h.continental_cells() as f32 / h.planet.n_cells() as f32,
        h.planet.plates.len(),
        h.flagged(cell_flags::SUTURE),
    );

    for block in 0..16 {
        h.run(&mut p, Phase::Drift, 50);
        let mut sp = h.speeds_cm_yr();
        let flags: Vec<String> = ALL_FLAGS
            .iter()
            .map(|(n, b)| format!("{n}={}", h.flagged(*b)))
            .collect();
        let mut th: Vec<f32> = h
            .planet
            .crust_thickness_m
            .iter()
            .zip(&h.planet.crust_type)
            .filter(|(_, t)| **t == CrustType::Continental)
            .map(|(x, _)| *x)
            .collect();
        println!(
            "t={:6.1} plates={:2} land={:5} speed p10/p50/p90 = {:.1}/{:.1}/{:.1} cm/yr  maxcont={:.1} km  {}",
            h.planet.time_myr,
            h.planet.plates.len(),
            h.continental_cells(),
            pct(&mut sp, 0.1),
            pct(&mut sp, 0.5),
            pct(&mut sp, 0.9),
            pct(&mut th, 1.0) / 1000.0,
            flags.join(" ")
        );
        // Per-plate speed at the plate centroid, and continent-continent contacts.
        let mut cc_edges = 0;
        for c in 0..h.planet.n_cells() as u32 {
            for &m in h.mesh.neighbors_of(c) {
                if m > c
                    && h.planet.plate_id[c as usize] != h.planet.plate_id[m as usize]
                    && h.planet.crust_type[c as usize] == CrustType::Continental
                    && h.planet.crust_type[m as usize] == CrustType::Continental
                {
                    cc_edges += 1;
                }
            }
        }
        let mut per_plate: Vec<String> = Vec::new();
        for (pi, pl) in h.planet.plates.iter().enumerate() {
            let mut sum = glam::Vec3::ZERO;
            let mut nc = 0;
            let mut cont = 0;
            for c in 0..h.planet.n_cells() {
                if h.planet.plate_id[c] as usize == pi {
                    sum += h.mesh.centers[c];
                    nc += 1;
                    if h.planet.crust_type[c] == CrustType::Continental {
                        cont += 1;
                    }
                }
            }
            if nc == 0 {
                continue;
            }
            let v = pl.velocity_m_yr(sum.normalize()).length() * 100.0;
            per_plate.push(format!("{nc}c/{}%/{v:.1}", 100 * cont / nc.max(1)));
        }
        println!(
            "    cc_edges={cc_edges} plates[cells/cont%/cm-yr]: {}",
            per_plate.join(" ")
        );
        let _ = block;
        if let Err(e) = plates_are_partition(&h.planet, &h.mesh) {
            println!("  !! partition broken: {e}");
        }
    }
    println!(
        "ledger: created {:.3e} m3, subducted {:.3e} m3",
        h.total.created_m3, h.total.subducted_m3
    );
}

/// Scaling check at the default subdivision level. Reports per-step cost so
/// regressions in the boundary passes are visible.
#[test]
#[ignore = "diagnostic"]
fn level8_scaling() {
    use std::time::Instant;
    let t = Instant::now();
    let mut h = Harness::level(8);
    println!("mesh build {:?}", t.elapsed());
    let mut p = TectonicsProcess::new();
    let t = Instant::now();
    h.run(&mut p, Phase::CrustalFormation, 200);
    println!("phase1 200 steps: {:?}", t.elapsed());
    let t = Instant::now();
    h.run(&mut p, Phase::Drift, 60);
    println!(
        "drift 60 steps: {:?}  plates={} land={:.1}%",
        t.elapsed(),
        h.planet.plates.len(),
        100.0 * h.continental_cells() as f32 / h.planet.n_cells() as f32
    );
    plates_are_partition(&h.planet, &h.mesh).expect("partition");
}
