//! Performance probe and calibration report. Both are `#[ignore]`d; run with
//!
//! ```text
//! cargo test -p iw-geology --release -- --ignored --nocapture
//! ```

use iw_core::planet::cell_flags;
use iw_core::{
    rng_for, CrustType, MassLedger, NullProgress, Planet, PlanetConfig, Process, RockType, StepCtx,
};
use iw_geology::{isostasy, GeologyProcess};
use iw_mesh::Mesh;

const LEVEL: u8 = 6;

fn busy_planet(mesh: &Mesh) -> Planet {
    let c = PlanetConfig {
        subdivision_level: LEVEL,
        ..Default::default()
    };
    let mut p = Planet::new(c, mesh.n_cells());
    for cell in 0..mesh.n_cells() {
        let lat = mesh.latlon[cell][0];
        if lat > 0.0 {
            p.crust_type[cell] = CrustType::Continental;
            p.crust_thickness_m[cell] = 38_000.0;
            p.crust_density_kg_m3[cell] = 2700.0;
        }
        match cell % 9 {
            0 => p.tectonic_flags[cell] = cell_flags::ARC,
            1 => p.tectonic_flags[cell] = cell_flags::COLLISION,
            2 => p.tectonic_flags[cell] = cell_flags::HOTSPOT,
            3 => p.sediment_m[cell] = 300.0,
            4 => p.ice_thickness_m[cell] = 1200.0,
            _ => {}
        }
        // Six strata per cell: a realistic column to walk.
        for (rock, t) in [
            (RockType::Basalt, 6_000.0),
            (RockType::Shale, 4_000.0),
            (RockType::Sandstone, 3_000.0),
            (RockType::Limestone, 2_000.0),
            (RockType::Shale, 1_500.0),
            (RockType::Conglomerate, 500.0),
        ] {
            p.columns.deposit(cell as u32, rock, t, 0.0);
        }
    }
    p
}

#[test]
#[ignore = "performance probe"]
fn full_step_at_level_6_is_under_budget() {
    let mesh = Mesh::build(LEVEL);
    let mut p = busy_planet(&mesh);
    let mut proc = GeologyProcess::new();
    let mut ledger = MassLedger::default();

    let mut times = Vec::new();
    for _ in 0..40 {
        let t0 = std::time::Instant::now();
        {
            let mut ctx = StepCtx {
                rng: rng_for(p.config.seed, "geology", p.step_index),
                progress: &NullProgress,
                ledger: &mut ledger,
            };
            proc.step(&mut p, &mesh, 0.5, &mut ctx);
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
        p.step_index += 1;
        p.time_myr += 0.5;
    }
    // Drop the first step (cold caches, scratch allocation).
    let warm = &times[1..];
    let mean = warm.iter().sum::<f64>() / warm.len() as f64;
    let mut sorted = warm.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let max = sorted[sorted.len() - 1];
    println!(
        "level {LEVEL} ({} cells): mean {mean:.2} ms, median {median:.2} ms, max {max:.2} ms \
         (max includes a metamorphism sweep step)",
        mesh.n_cells()
    );
    assert!(median < 15.0, "median step {median:.2} ms exceeds 15 ms");
}

#[test]
#[ignore = "calibration report"]
fn print_isostasy_anchors() {
    println!("offset C = {:.1} m", isostasy::ISOSTATIC_OFFSET_M);
    println!("thermal lid = {:.1} m", isostasy::THERMAL_LID_M);
    for (name, h, rho, oceanic) in [
        ("35 km / 2700 continental", 35_000.0, 2700.0, false),
        ("7 km / 3000 fresh ocean", 7_000.0, 3000.0, true),
        ("7 km / 3150 mid-age ocean", 7_000.0, 3150.0, true),
        ("7 km / 3300 old ocean", 7_000.0, 3300.0, true),
        ("70 km / 2750 orogen", 70_000.0, 2750.0, false),
        ("20 km / 2700 rifted margin", 20_000.0, 2700.0, false),
    ] {
        let e = isostasy::airy_elevation_m(h, rho, oceanic, 0.0, 0.0);
        println!("{name:28} -> {e:8.0} m");
    }
}
