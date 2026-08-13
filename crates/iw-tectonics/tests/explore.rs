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

/// Dump an equirectangular PPM of the crust after CrustalFormation, so craton
/// outlines and any rasterization banding can be inspected directly instead of
/// through five more processes' worth of geology and erosion.
///
/// `IW_DUMP_LEVEL`, `IW_DUMP_SEED`, `IW_DUMP_STEPS` and `IW_DUMP_OUT` override
/// the defaults.
/// Diagnostic bisection for the Voronoi confetti: left half of the image is
/// genesis membership sampled DIRECTLY by pixel direction (ground truth);
/// right half is membership as painted onto the Voronoi mesh's cells and read
/// back through `cell_at`. If the left is coherent and the right is speckle,
/// the mesh or its lookup is scrambling spatial data; if both speckle, the
/// membership function itself is at fault at this sampling density.
#[test]
#[ignore = "diagnostic"]
fn dump_voronoi_membership() {
    use std::io::Write;

    let seed = 42u64;
    let budget = 160_000u32;
    let density = iw_tectonics::genesis_density(seed, 14);
    let mesh = iw_mesh::Mesh::build_voronoi(budget, seed, &density);
    // Paint per-cell: density as a proxy for membership (rim-graded).
    let per_cell: Vec<f32> = mesh.centers.iter().map(|c| density(*c)).collect();

    let (w, ht) = (2048usize, 1024usize);
    let mut buf = vec![0u8; w * ht * 3];
    for y in 0..ht {
        let lat = (0.5 - (y as f32 + 0.5) / ht as f32) * std::f32::consts::PI;
        for x in 0..w {
            let lon = ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
            let dir = glam::Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let v = if x < w / 2 {
                density(dir)
            } else {
                per_cell[mesh.cell_at(dir) as usize]
            };
            let g = (v * 255.0) as u8;
            let i = (y * w + x) * 3;
            buf[i] = g;
            buf[i + 1] = g;
            buf[i + 2] = if x == w / 2 { 255 } else { g };
        }
    }
    let mut f = std::fs::File::create("/tmp/vor_membership.ppm").unwrap();
    writeln!(f, "P6 {w} {ht} 255").unwrap();
    f.write_all(&buf).unwrap();
    println!("wrote /tmp/vor_membership.ppm");

    // Third panel, separate file: the ACTUAL phase-1 seeding on this exact
    // mesh, crust type read back per pixel. If this speckles while the halves
    // above are coherent, the corruption is inside the seeding/simulation
    // path, not the mesh or the shapes.
    let mesh = std::sync::Arc::new(mesh);
    let mut config = iw_core::PlanetConfig {
        seed,
        cell_budget: budget,
        ..Default::default()
    };
    config.sanitize();
    let mut h = Harness::new(config, std::sync::Arc::clone(&mesh));
    let mut p = TectonicsProcess::new();
    h.run(&mut p, Phase::CrustalFormation, 3);
    let mut buf = vec![0u8; w * ht * 3];
    for y in 0..ht {
        let lat = (0.5 - (y as f32 + 0.5) / ht as f32) * std::f32::consts::PI;
        for x in 0..w {
            let lon = ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
            let dir = glam::Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let c = h.mesh.cell_at(dir) as usize;
            let g = if h.planet.crust_type[c] == CrustType::Continental {
                200
            } else {
                30
            };
            let i = (y * w + x) * 3;
            buf[i] = g;
            buf[i + 1] = g;
            buf[i + 2] = g;
        }
    }
    let mut f = std::fs::File::create("/tmp/vor_seeded.ppm").unwrap();
    writeln!(f, "P6 {w} {ht} 255").unwrap();
    f.write_all(&buf).unwrap();
    println!("wrote /tmp/vor_seeded.ppm");
}

#[test]
#[ignore = "diagnostic"]
fn dump_phase1_crust() {
    use std::io::Write;
    use std::sync::Arc;

    let env = |k: &str, d: u64| -> u64 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let level = env("IW_DUMP_LEVEL", 6) as u8;
    let seed = env("IW_DUMP_SEED", 42);
    let steps = env("IW_DUMP_STEPS", 200);
    let out = std::env::var("IW_DUMP_OUT").unwrap_or_else(|_| "/tmp/phase1.ppm".into());

    let mut config = test_config(level);
    config.seed = seed;
    let mut h = Harness::new(config, Arc::new(iw_mesh::Mesh::build(level)));
    let mut p = TectonicsProcess::new();
    h.run(&mut p, Phase::CrustalFormation, steps);

    let (w, ht) = (2048usize, 1024usize);
    let mut buf = vec![0u8; w * ht * 3];
    for y in 0..ht {
        let lat = (0.5 - (y as f32 + 0.5) / ht as f32) * std::f32::consts::PI;
        for x in 0..w {
            let lon = ((x as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
            let dir = glam::Vec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let c = h.mesh.cell_at(dir) as usize;
            let t = h.planet.crust_thickness_m[c];
            let rgb = if h.planet.crust_type[c] == CrustType::Continental {
                // 28 km (drowned margin) -> 44 km (shield core).
                let f = ((t - 28_000.0) / 16_000.0).clamp(0.0, 1.0);
                [
                    (60.0 + 195.0 * f) as u8,
                    (110.0 + 100.0 * f) as u8,
                    (50.0 + 90.0 * f) as u8,
                ]
            } else {
                let f = ((t - 6_300.0) / 1_400.0).clamp(0.0, 1.0);
                [0, (20.0 + 40.0 * f) as u8, (70.0 + 110.0 * f) as u8]
            };
            buf[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&rgb);
        }
    }
    let mut f = std::fs::File::create(&out).expect("create dump");
    write!(f, "P6\n{w} {ht}\n255\n").unwrap();
    f.write_all(&buf).unwrap();
    println!(
        "wrote {out}: level {level}, seed {seed}, {steps} steps, land {:.1}%",
        100.0 * h.continental_cells() as f32 / h.planet.n_cells() as f32
    );
}

/// Probe: per-plate area/continental fractions through the drift era, to see
/// why rifting does or does not fire on the continental plate.
#[test]
#[ignore]
fn probe_drift_plate_stats() {
    let mut h = Harness::level(5);
    let mut p = TectonicsProcess::new();
    h.run_crustal_formation(&mut p);
    for block in 0..20 {
        h.run(&mut p, Phase::Drift, 20);
        let np = h.planet.plates.len();
        let n = h.planet.n_cells();
        let mut area = vec![0usize; np];
        let mut cont = vec![0usize; np];
        for c in 0..n {
            let pl = h.planet.plate_id[c] as usize;
            if pl < np {
                area[pl] += 1;
                if h.planet.crust_type[c] == CrustType::Continental {
                    cont[pl] += 1;
                }
            }
        }
        let t = h.planet.time_myr;
        let mut rows: Vec<String> = (0..np)
            .filter(|&i| area[i] > 0)
            .map(|i| {
                format!(
                    "p{i}: {:.2}A {:.2}C",
                    area[i] as f64 / n as f64,
                    if area[i] > 0 {
                        cont[i] as f64 / area[i] as f64
                    } else {
                        0.0
                    }
                )
            })
            .collect();
        rows.sort();
        println!("t={t:.0} block={block} plates={np}: {}", rows.join(" | "));
    }
}
